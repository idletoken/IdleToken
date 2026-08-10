// Prevents an extra console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::Value;
use tauri_plugin_shell::ShellExt;

mod engine;
mod pairing;
mod weights;

/// Run the native engine's hardware probe as a sidecar and return the parsed
/// JSON report. This is the local RPC boundary between the GUI (web frontend)
/// and the decoupled inference engine (design philosophy 17): the client never
/// links the engine, it launches `idletoken-worker --probe-json` as a separate
/// process and reads one line of JSON back.
///
/// `max_vram_mb` / `max_ram_mb` are the client's usage caps (0 = no cap); they
/// are forwarded so the engine clamps its reported usable values.
#[tauri::command]
async fn probe_resources(
    app: tauri::AppHandle,
    max_vram_mb: u64,
    max_ram_mb: u64,
) -> Result<Value, String> {
    let sidecar = app
        .shell()
        .sidecar("idletoken-worker")
        .map_err(|e| format!("sidecar not found (bundle binaries/idletoken-worker): {e}"))?;

    let mut args: Vec<String> = vec!["--probe-json".into()];
    if max_vram_mb > 0 {
        args.push("--max-vram-mb".into());
        args.push(max_vram_mb.to_string());
    }
    if max_ram_mb > 0 {
        args.push("--max-ram-mb".into());
        args.push(max_ram_mb.to_string());
    }

    let output = sidecar
        .args(args)
        .output()
        .await
        .map_err(|e| format!("failed to run worker: {e}"))?;

    // Exit code 2 is not a failure to probe — it is a successful probe whose
    // verdict is "this machine cannot serve layers" (hardware floor, G-HW).
    // The JSON is still on stdout and carries hw_status/hw_reason, and the
    // dashboard renders that as a readable notice. Turning it into a generic
    // Err here would replace the one message the user needs with "worker
    // exited with 2".
    let code = output.status.code().unwrap_or(-1);
    if !output.status.success() && code != 2 {
        return Err(format!(
            "worker exited with {}: {}",
            code,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    // The probe prints diagnostics to stderr and exactly one JSON object on
    // stdout. Take the last line that looks like a JSON object to be robust
    // against any leading warnings.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| format!("no JSON in probe output: {stdout}"))?;

    serde_json::from_str::<Value>(line).map_err(|e| format!("bad probe JSON: {e}"))
}

/// "What can this machine run?" — the capability table (G-ADVISE) straight
/// from the engine, so the UI never re-derives the fit rule in TypeScript.
/// Exit code 2 means "probe fine, hardware below the floor": the JSON is still
/// valid and the caller renders it, same contract as `probe_resources`.
#[tauri::command]
async fn advise_capability(app: tauri::AppHandle) -> Result<Value, String> {
    let sidecar = app
        .shell()
        .sidecar("idletoken-worker")
        .map_err(|e| format!("sidecar not found (bundle binaries/idletoken-worker): {e}"))?;
    let output = sidecar
        .args(["--advise-json"])
        .output()
        .await
        .map_err(|e| format!("failed to run worker: {e}"))?;
    let code = output.status.code().unwrap_or(-1);
    if !output.status.success() && code != 2 {
        return Err(format!(
            "worker exited with {}: {}",
            code,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| format!("no JSON in advise output: {stdout}"))?;
    serde_json::from_str::<Value>(line).map_err(|e| format!("bad advise JSON: {e}"))
}

/// One-shot chat against the cluster's own HTTP API (the dashboard's "try it"
/// box). Lives in Rust because the engine speaks plain LAN HTTP without CORS
/// headers — the webview cannot fetch it directly. std-only HTTP/1.1 client:
/// the target is our own coordinator on the LAN, no TLS/redirect/chunked needs
/// beyond what the engine emits (it answers with Content-Length JSON).
#[tauri::command]
async fn api_chat(
    base_url: String,
    prompt: String,
    token: String,
    model: Option<String>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        use std::io::{Read, Write};
        let host_port = base_url
            .trim()
            .strip_prefix("http://")
            .ok_or_else(|| format!("unsupported API url (need http://): {base_url}"))?
            .trim_end_matches('/')
            .to_string();
        // Model id comes from the caller's settings; the engine echoes back
        // whatever it actually serves (it does not switch models per request).
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "deepseek-v4-flash".into());
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 200,
            "messages": [{ "role": "user", "content": prompt }],
        })
        .to_string();
        let auth = if token.is_empty() {
            String::new()
        } else {
            format!("Authorization: Bearer {token}\r\n")
        };
        let req = format!(
            "POST /v1/messages HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n{auth}Content-Length: {len}\r\nConnection: close\r\n\r\n{body}",
            host = host_port.split(':').next().unwrap_or(&host_port),
            len = body.len(),
        );
        let mut stream = std::net::TcpStream::connect(&host_port)
            .map_err(|e| format!("connect {host_port}: {e}"))?;
        // Long ceiling, not a latency promise: first tokens on a big prompt can be slow.
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(180)))
            .ok();
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| format!("read: {e}"))?;
        let text = String::from_utf8_lossy(&raw);
        let payload = text
            .split_once("\r\n\r\n")
            .map(|(_, b)| b)
            .ok_or("malformed HTTP response")?;
        let v: Value =
            serde_json::from_str(payload.trim()).map_err(|e| format!("bad API JSON: {e}"))?;
        // Anthropic shape: content[0].text; surface engine errors verbatim.
        if let Some(t) = v["content"][0]["text"].as_str() {
            return Ok(t.to_string());
        }
        if let Some(err) = v["error"]["message"].as_str() {
            return Err(err.to_string());
        }
        Err(format!("unexpected API response: {}", payload.trim()))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// GET the cluster's serving counters (engine `GET /v1/stats`) for the
/// dashboard activity row. Rust-side for the usual reason: the engine speaks
/// plain LAN HTTP without CORS headers.
#[tauri::command]
async fn api_stats(base_url: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        use std::io::{Read, Write};
        let host_port = base_url
            .trim()
            .strip_prefix("http://")
            .ok_or_else(|| format!("unsupported API url (need http://): {base_url}"))?
            .trim_end_matches('/')
            .to_string();
        let req = format!(
            "GET /v1/stats HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n",
            host = host_port.split(':').next().unwrap_or(&host_port),
        );
        let mut stream = std::net::TcpStream::connect(&host_port)
            .map_err(|e| format!("connect {host_port}: {e}"))?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .ok();
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| format!("read: {e}"))?;
        let text = String::from_utf8_lossy(&raw);
        let payload = text
            .split_once("\r\n\r\n")
            .map(|(_, b)| b)
            .ok_or("malformed HTTP response")?;
        serde_json::from_str::<Value>(payload.trim()).map_err(|e| format!("bad stats JSON: {e}"))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Streaming variant of `api_chat` for the dashboard's try-it box: sets
/// `"stream": true` and relays the engine's SSE frames (Anthropic event
/// sequence, close-delimited plain HTTP — see engine sse_begin/sse_delta) to
/// the webview as `api-chat` events tagged with the caller's `id`:
///   { id, kind: "delta", text }  per text_delta
///   { id, kind: "done" }         on message_stop / socket close
///   { id, kind: "error", message } on any failure
/// Rust-side for the same reason as `api_chat` (no CORS on the engine).
#[tauri::command]
async fn api_chat_stream(
    app: tauri::AppHandle,
    id: String,
    base_url: String,
    messages: Value,
    token: String,
    model: Option<String>,
) -> Result<(), String> {
    use tauri::Emitter;
    tauri::async_runtime::spawn_blocking(move || {
        let emit = |kind: &str, text: Option<&str>, message: Option<&str>| {
            let _ = app.emit(
                "api-chat",
                serde_json::json!({ "id": id, "kind": kind, "text": text, "message": message }),
            );
        };
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "deepseek-v4-flash".into());
        match stream_chat_inner(&base_url, &messages, &token, &model, &mut |t| {
            emit("delta", Some(t), None)
        }) {
            Ok(()) => emit("done", None, None),
            Err(e) => emit("error", None, Some(&e)),
        }
    })
    .await
    .map_err(|e| e.to_string())
}

/// Blocking SSE consumer: connect → POST with stream:true → parse
/// `event:/data:` frames until the engine closes the connection.
/// `messages` is the FULL conversation ([{role, content}, …]) — the engine is
/// stateless per request, so multi-turn chat means resending the history.
fn stream_chat_inner(
    base_url: &str,
    messages: &Value,
    token: &str,
    model: &str,
    on_delta: &mut dyn FnMut(&str),
) -> Result<(), String> {
    use std::io::{Read, Write};
    let host_port = base_url
        .trim()
        .strip_prefix("http://")
        .ok_or_else(|| format!("unsupported API url (need http://): {base_url}"))?
        .trim_end_matches('/')
        .to_string();
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 512,
        "stream": true,
        "messages": messages,
    })
    .to_string();
    let auth = if token.is_empty() {
        String::new()
    } else {
        format!("Authorization: Bearer {token}\r\n")
    };
    let req = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n{auth}Content-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        host = host_port.split(':').next().unwrap_or(&host_port),
        len = body.len(),
    );
    let mut stream = std::net::TcpStream::connect(&host_port)
        .map_err(|e| format!("connect {host_port}: {e}"))?;
    // Per-read ceiling: generous for slow first tokens on big prompts, but a
    // dead cluster still errors out instead of hanging the box forever.
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(300)))
        .ok();
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut headers_done = false;
    let mut is_sse = false;
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if !headers_done {
            if let Some(pos) = find_seq(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                is_sse = head.contains("text/event-stream");
                buf.drain(..pos + 4);
                headers_done = true;
            } else {
                continue;
            }
        }
        if !is_sse {
            continue; // non-SSE (error JSON / old engine): fall through to the tail parse
        }
        // Drain complete SSE frames (blank-line terminated).
        while let Some(pos) = find_seq(&buf, b"\n\n") {
            let frame = String::from_utf8_lossy(&buf[..pos]).to_string();
            buf.drain(..pos + 2);
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data: ") else { continue };
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                if v["type"] == "content_block_delta" {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        on_delta(t);
                    }
                }
            }
        }
    }
    if !headers_done {
        return Err("empty response from the cluster API".into());
    }
    if !is_sse {
        // The engine answered non-streaming (error body or no SSE support):
        // surface its JSON honestly — text as one delta, errors as errors.
        let payload = String::from_utf8_lossy(&buf);
        let v: Value = serde_json::from_str(payload.trim())
            .map_err(|_| format!("unexpected non-stream response: {}", payload.trim()))?;
        if let Some(t) = v["content"][0]["text"].as_str() {
            on_delta(t);
            return Ok(());
        }
        if let Some(err) = v["error"]["message"].as_str() {
            return Err(err.to_string());
        }
        return Err(format!("unexpected API response: {}", payload.trim()));
    }
    Ok(())
}

fn find_seq(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// UI-test channel (acceptance §8): expose the launcher's IDLETOKEN_UI_TEST
/// directive list to the frontend, which executes them through the same
/// provider paths user actions take. Unset = empty = no effect.
#[tauri::command]
fn ui_test_directives() -> Vec<String> {
    std::env::var("IDLETOKEN_UI_TEST")
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Sink for UI-test assertions: the frontend reports structured results here
/// and they land on the client's stderr, where acceptance.sh can grep them.
/// Print-only, no side effects; unused outside IDLETOKEN_UI_TEST runs.
#[tauri::command]
fn ui_test_report(tag: String, data: String) {
    eprintln!("UI_TEST_REPORT {tag} {data}");
}

/// One-click diagnostics bundle (for support): everything needed to debug an
/// incident, collected into a single JSON the user can simply send over.
///
/// Why it has to exist: this is a GUI product, and a user facing "the cluster
/// will not start" has nothing in hand -- the engine log is in the sidecar's ring
/// buffer, the hardware verdict is in a worker's stderr, and the cluster topology
/// is behind the coordinator's HTTP. Without this button, every support
/// conversation starts with "could you open a terminal", which is precisely what
/// we promised users they would not have to do.
///
/// **Redaction is a hard requirement, not a nice-to-have**: access tokens,
/// platform JWTs and prompts must never go in. With a bundle that leaks a token,
/// the more cooperative the user is the more dangerous it gets. This collects
/// only machine- and engine-side facts; settings are merged in by the front end
/// after selecting fields against an **allowlist** (a missed entry in a denylist
/// is a leak, a missed entry in an allowlist is just one absent field -- the same
/// trade-off as the public mirror).
#[tauri::command]
async fn collect_diagnostics(
    app: tauri::AppHandle,
    base_url: Option<String>,
) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    out.insert("schema".into(), Value::from("idletoken-diagnostics/1"));
    out.insert(
        "generatedAt".into(),
        Value::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        ),
    );
    out.insert(
        "app".into(),
        serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        }),
    );

    // Hardware and capability: ask the engine rather than computing a second
    // version in the client. The verdict must share a source with the planner
    // (the lesson of G-ADVISE: change one, drift in two, and the advisor promises
    // what the cluster then refuses).
    // Failures are recorded faithfully too -- "the probe failed" is often the
    // answer itself (missing driver, missing DLL, insufficient compute).
    match probe_resources(app.clone(), 0, 0).await {
        Ok(v) => out.insert("probe".into(), v),
        Err(e) => out.insert("probe".into(), serde_json::json!({ "error": e })),
    };
    match advise_capability(app.clone()).await {
        Ok(v) => out.insert("advise".into(), v),
        Err(e) => out.insert("advise".into(), serde_json::json!({ "error": e })),
    };

    // Current cluster state (the coordinator's HTTP). With no cluster running
    // this is an error, not an empty value -- the two must stay distinguishable.
    if let Some(base) = base_url.as_ref().filter(|b| !b.trim().is_empty()) {
        let url = format!("{}/v1/cluster/status", base.trim_end_matches('/'));
        out.insert(
            "cluster".into(),
            match http_get_json(&url).await {
                Ok(v) => v,
                Err(e) => serde_json::json!({ "error": e }),
            },
        );
    }
    Ok(Value::Object(out))
}

/// A minimal GET returning JSON. It lives on the Rust side for the same reason as
/// `api_stats`: the engine speaks plaintext LAN HTTP without CORS headers, which
/// the webview cannot fetch directly.
async fn http_get_json(url: &str) -> Result<Value, String> {
    let url = url.to_string();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        use std::io::{Read, Write};
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| format!("unsupported url (need http://): {url}"))?;
        let (host_port, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n",
            host = host_port.split(':').next().unwrap_or(host_port),
        );
        let mut stream = std::net::TcpStream::connect(host_port)
            .map_err(|e| format!("connect {host_port}: {e}"))?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .ok();
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| format!("read: {e}"))?;
        let text = String::from_utf8_lossy(&raw);
        let payload = text
            .split_once("\r\n\r\n")
            .map(|(_, b)| b)
            .ok_or("malformed HTTP response")?;
        serde_json::from_str::<Value>(payload.trim()).map_err(|e| format!("bad JSON: {e}"))
    })
    .await
    .map_err(|e| e.to_string())?
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(engine::Engine::default())
        .manage(pairing::Pairing::default())
        .setup(|app| {
            // Headless pairing trigger (acceptance §8): drive the real LAN
            // pairing path without the webview, for harnesses on machines whose
            // GUI can't run (e.g. a locked Windows session). No-op if unset.
            if let Ok(spec) = std::env::var("IDLETOKEN_HEADLESS_PAIR") {
                if !spec.trim().is_empty() {
                    pairing::headless_pair(&app.handle().clone(), spec.trim());
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            probe_resources,
            advise_capability,
            api_chat,
            api_chat_stream,
            api_stats,
            ui_test_directives,
            ui_test_report,
            collect_diagnostics,
            engine::engine_start,
            engine::engine_stop,
            engine::engine_status,
            engine::engine_logs,
            engine::clear_kv_cache,
            engine::platform_agent_start,
            engine::platform_agent_stop,
            engine::platform_agent_status,
            pairing::pairing_create,
            pairing::pairing_join,
            pairing::pairing_start,
            pairing::pairing_set_coordinator,
            pairing::pairing_leave,
            pairing::pairing_status,
            weights::weights_fetch,
            weights::weights_default_dir,
            weights::weights_state,
            weights::weights_cancel,
        ])
        .build(tauri::generate_context!())
        .expect("error while building IdleToken client")
        .run(|app, event| {
            // Graceful shutdown: take the engine down with the client so a
            // normal quit never leaves an orphaned sidecar. (A SIGKILLed
            // client can still orphan it — parent-death detection in the
            // engine is the follow-up for that.)
            match event {
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
                    eprintln!("idletoken-client: shutdown event — stopping all sidecars");
                    let res = engine::stop_all(app);
                    eprintln!("idletoken-client: stop_all -> {res:?}");
                }
                _ => {}
            }
        });
}
