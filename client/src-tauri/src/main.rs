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

/// GET `path` from an engine base URL and parse the JSON body.
///
/// Rust-side for the usual reason: the engine speaks plain LAN HTTP with **no
/// CORS headers**, so the webview's own `fetch` is rejected before it starts —
/// that is what "读取能力报告失败: TypeError: Failed to fetch" was.
///
/// Reads by Content-Length rather than to EOF. The coordinator answers
/// `Connection: close` and then leaves the socket open (verified 2026-08-11),
/// so `read_to_end` blocks until the read timeout and — crucially — returns Err,
/// throwing away a complete response it had already buffered. Three copies of
/// that same loop existed here; this is the one they now share.
fn engine_get_json_blocking(base_url: &str, path: &str) -> Result<Value, String> {
    use std::io::{Read, Write};
    let host_port = base_url
        .trim()
        .strip_prefix("http://")
        .ok_or_else(|| format!("unsupported API url (need http://): {base_url}"))?
        .trim_end_matches('/')
        .to_string();
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n",
        host = host_port.split(':').next().unwrap_or(&host_port),
    );
    let mut stream = std::net::TcpStream::connect(&host_port)
        .map_err(|e| format!("connect {host_port}: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

    let mut raw: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&chunk[..n]);
                if let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4) {
                    let headers = String::from_utf8_lossy(&raw[..end]);
                    let len = headers.lines().find_map(|l| {
                        l.split_once(':')
                            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                            .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                    });
                    match len {
                        Some(len) if raw.len() - end >= len => break,
                        None => break, // no length: EOF is the only terminator
                        _ => {}
                    }
                }
            }
            Err(_) => break, // timeout: keep what arrived rather than lose it
        }
    }
    let text = String::from_utf8_lossy(&raw);
    let payload = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .ok_or("malformed HTTP response")?;
    serde_json::from_str::<Value>(payload.trim()).map_err(|e| format!("bad JSON from {path}: {e}"))
}

/// The cluster's capability table (engine `GET /v1/capability`).
#[tauri::command]
async fn api_capability(base_url: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || engine_get_json_blocking(&base_url, "/v1/capability"))
        .await
        .map_err(|e| e.to_string())?
}

/// GET the cluster's serving counters (engine `GET /v1/stats`) for the
/// dashboard activity row. Rust-side for the usual reason: the engine speaks
/// plain LAN HTTP without CORS headers.
#[tauri::command]
async fn api_stats(base_url: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || engine_get_json_blocking(&base_url, "/v1/stats"))
        .await
        .map_err(|e| e.to_string())?
}

/// In-flight chat streams: id -> cancel flag. A generation can run for minutes;
/// "stop" is not a nicety, it is the only way out short of quitting the app.
///
/// Dropping the socket is what stops the ENGINE: the coordinator's next SSE
/// write hits a broken pipe and it abandons the request, so a cancelled reply
/// stops costing GPU almost immediately rather than generating into the void.
static CHAT_CANCEL: std::sync::Mutex<Vec<(String, std::sync::Arc<std::sync::atomic::AtomicBool>)>> =
    std::sync::Mutex::new(Vec::new());

fn chat_register(id: &str) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut a = CHAT_CANCEL.lock().unwrap();
    a.retain(|(k, _)| k != id);
    a.push((id.to_string(), flag.clone()));
    flag
}

fn chat_unregister(id: &str) {
    CHAT_CANCEL.lock().unwrap().retain(|(k, _)| k != id);
}

/// Stop a running generation. Returns false if it had already finished.
#[tauri::command]
fn api_chat_cancel(id: String) -> bool {
    let a = CHAT_CANCEL.lock().unwrap();
    match a.iter().find(|(k, _)| *k == id) {
        Some((_, flag)) => {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            true
        }
        None => false,
    }
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
    // 0 / absent = do not send the field, i.e. let the engine generate until
    // EOS or the context runs out. The 512 that used to be hardcoded here made
    // the "Max tokens per reply" setting a decoration: the user changed it and
    // nothing happened.
    max_tokens: Option<u32>,
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
        let cancel = chat_register(&id);
        let out = stream_chat_inner(
            &base_url, &messages, &token, &model, max_tokens, &cancel,
            &mut |t| emit("delta", Some(t), None),
            // Prefill progress: "128/512" tokens of the prompt processed. On a
            // LAN cluster this phase is minutes, so it is the difference
            // between a live UI and one that looks hung.
            &mut |done, total, reused| {
                let _ = app.emit(
                    "api-chat",
                    serde_json::json!({ "id": id, "kind": "progress", "done": done,
                                        "total": total, "reused": reused }),
                );
            },
        );
        // Every id is fresh, so without this the registry grows for the life of
        // the process and holds a flag nobody can ever reach again.
        chat_unregister(&id);
        match out {
            Ok(()) => emit("done", None, None),
            Err(e) => emit("error", None, Some(&e)),
        }
    })
    .await
    .map_err(|e| e.to_string())
}

/// How long a streaming socket may stay completely silent before we call it
/// dead. Named because the error message quotes it — a number the user is told
/// and a number the code enforces must not be able to drift apart.
const READ_TIMEOUT_S: u64 = 300;

/// Blocking SSE consumer: connect → POST with stream:true → parse
/// `event:/data:` frames until the engine closes the connection.
/// `messages` is the FULL conversation ([{role, content}, …]) — the engine is
/// stateless per request, so multi-turn chat means resending the history.
fn stream_chat_inner(
    base_url: &str,
    messages: &Value,
    token: &str,
    model: &str,
    max_tokens: Option<u32>,
    cancel: &std::sync::atomic::AtomicBool,
    on_delta: &mut dyn FnMut(&str),
    on_progress: &mut dyn FnMut(u32, u32, u32),
) -> Result<(), String> {
    use std::io::{Read, Write};
    let host_port = base_url
        .trim()
        .strip_prefix("http://")
        .ok_or_else(|| format!("unsupported API url (need http://): {base_url}"))?
        .trim_end_matches('/')
        .to_string();
    let mut body_obj = serde_json::json!({
        "model": model,
        "stream": true,
        "messages": messages,
    });
    if let Some(n) = max_tokens.filter(|n| *n > 0) {
        body_obj["max_tokens"] = serde_json::json!(n);
    }
    let body = body_obj.to_string();
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
    // Per-read ceiling: a dead cluster must error out rather than hang the box
    // forever. This is now a genuine liveness check rather than a race against
    // prefill — the engine ticks the stream once per prefill chunk, so the
    // socket only goes quiet this long if something is actually wrong. (Before
    // that tick existed, a prompt whose prefill ran past the ceiling died here
    // with a bare "os error 10060".)
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(READ_TIMEOUT_S)))
        .ok();
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut headers_done = false;
    let mut is_sse = false;
    let mut got_bytes = false;
    let mut status: u16 = 0;
    let mut chunk = [0u8; 4096];
    loop {
        // Checked before every read AND after every frame below: deltas arrive
        // every few milliseconds while generating, so a stop lands almost at
        // once. Returning drops `stream`, which is what tells the engine.
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }
        let n = match stream.read(&mut chunk) {
            Ok(n) => n,
            // A read timeout is not the same failure as a broken socket, and
            // it must not be reported as one. The engine keeps the stream warm
            // with a tick per prefill chunk, so silence this long means the
            // cluster really did stall — say that, in words, instead of
            // surfacing "os error 10060", which is what the user actually saw.
            Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => {
                return Err(if got_bytes {
                    format!("the cluster went silent for {READ_TIMEOUT_S}s mid-reply, so this generation was cut off (what arrived is kept). Check the coordinator — it is most likely stuck, not merely slow.")
                } else {
                    format!("no response from the cluster in {READ_TIMEOUT_S}s. Check that the coordinator is running and not stuck loading the model.")
                });
            }
            Err(e) => return Err(format!("read: {e}")),
        };
        if n == 0 {
            break;
        }
        got_bytes = true;
        buf.extend_from_slice(&chunk[..n]);
        if !headers_done {
            if let Some(pos) = find_seq(&buf, b"\r\n\r\n") {
                let raw_head = String::from_utf8_lossy(&buf[..pos]).to_string();
                let head = raw_head.to_lowercase();
                is_sse = head.contains("text/event-stream");
                // Keep the status line. When the body turns out not to be a
                // shape we recognise, the status is often the only thing that
                // says WHICH failure it was, and reporting the body alone threw
                // it away.
                status = raw_head
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|c| c.parse::<u16>().ok())
                    .unwrap_or(0);
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
        let mut stream_ended = false;
        while let Some(pos) = find_seq(&buf, b"\n\n") {
            let frame = String::from_utf8_lossy(&buf[..pos]).to_string();
            buf.drain(..pos + 2);
            for line in frame.lines() {
                // SSE comment carrying prefill progress (`: prefill 128/512`).
                // Comments are ignored by the spec, so this is invisible to
                // Claude Code and every other client; ours reads it.
                if let Some(rest) = line.strip_prefix(": prefill ") {
                    // "<done>/<total> reuse=<n>"; the reuse field is what makes
                    // a cache hit distinguishable from a miss on screen.
                    let rest = rest.trim();
                    let (counts, reused) = match rest.split_once(" reuse=") {
                        Some((c, r)) => (c, r.parse::<u32>().unwrap_or(0)),
                        None => (rest, 0),
                    };
                    if let Some((a, b)) = counts.split_once('/') {
                        if let (Ok(done), Ok(total)) = (a.parse::<u32>(), b.parse::<u32>()) {
                            on_progress(done, total, reused);
                        }
                    }
                    continue;
                }
                let Some(data) = line.strip_prefix("data: ") else { continue };
                // OpenAI's terminator is a sentinel, not JSON, so it has to be
                // matched before the parse.
                if data.trim() == "[DONE]" { stream_ended = true; continue; }
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                // Anthropic's terminator. END ON THE PROTOCOL, NOT ON THE
                // SOCKET: this loop used to exit only on EOF, so if the server
                // held the connection open after the last frame — which is
                // ordinary HTTP keep-alive behaviour, and which our own engine
                // did on Windows because close() does not close a socket there
                // — the reply was complete on screen while the client sat
                // waiting, and 300 seconds later reported the generation as cut
                // off. A finished stream says so; waiting for the peer to hang
                // up is not a termination condition.
                if v["type"] == "message_stop" { stream_ended = true; continue; }
                if v["type"] == "content_block_delta" {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        on_delta(t);
                    }
                }
                // A failure partway through: the HTTP status was committed to
                // 200 before the engine knew, so the stream is the only place
                // the truth can arrive. This used to be dropped on the floor —
                // a failed generation looked like a successful empty reply.
                if let Some(msg) = v["error"]["message"].as_str() {
                    return Err(msg.to_string());
                }
            }
            if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                return Ok(());
            }
        }
        if stream_ended {
            return Ok(());
        }
    }
    if !headers_done {
        return Err("empty response from the cluster API".into());
    }
    if !is_sse {
        // The engine answered non-streaming (error body or no SSE support):
        // surface its JSON honestly — text as one delta, errors as errors.
        let payload = String::from_utf8_lossy(&buf);
        let v: Value = serde_json::from_str(payload.trim()).map_err(|_| {
            format!("cluster replied HTTP {status} with: {}", payload.trim())
        })?;
        if let Some(t) = v["content"][0]["text"].as_str() {
            on_delta(t);
            return Ok(());
        }
        if let Some(err) = v["error"]["message"].as_str() {
            return Err(err.to_string());
        }
        return Err(format!("cluster replied HTTP {status}: {}", payload.trim()));
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
    tauri::async_runtime::spawn_blocking(move || {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| format!("unsupported url (need http://): {url}"))?;
        let (host_port, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        engine_get_json_blocking(&format!("http://{host_port}"), path)
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
            api_capability,
            api_chat_cancel,
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

/// Tests for the SSE consumer. It is worth testing in isolation because the
/// bugs it has had were all "the wire said X and we showed the user Y":
/// a mid-stream error reported as a successful empty reply, and a read timeout
/// reported as a raw Windows error code. Each test speaks canned frames over a
/// real loopback socket, so the framing is exercised, not mocked around.
#[cfg(test)]
mod stream_tests {
    use std::io::{Read, Write};
    use std::sync::atomic::AtomicBool;

    /// Serve one connection with `script` (raw bytes after the request is read)
    /// and return the base URL to point the consumer at.
    fn serve(script: &'static [u8]) -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf); // drain the request; we do not assert on it here
            let _ = s.write_all(script);
        });
        format!("http://127.0.0.1:{port}")
    }

    fn run(script: &'static [u8]) -> (Result<(), String>, String, Vec<(u32, u32, u32)>) {
        let url = serve(script);
        let cancel = AtomicBool::new(false);
        let mut text = String::new();
        let mut prog: Vec<(u32, u32, u32)> = Vec::new();
        let msgs = serde_json::json!([{ "role": "user", "content": "hi" }]);
        let out = super::stream_chat_inner(
            &url, &msgs, "", "m", None, &cancel,
            &mut |t| text.push_str(t),
            &mut |d, n, r| prog.push((d, n, r)),
        );
        (out, text, prog)
    }

    const HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";

    #[test]
    fn deltas_accumulate_and_progress_comments_are_reported() {
        let script: &'static [u8] = Box::leak(
            format!(
                "{HEAD}\
                 : prefill 6/8 reuse=6\n\n\
                 : prefill 8/8 reuse=6\n\n\
                 event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"he\"}}}}\n\n\
                 event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"llo\"}}}}\n\n\
                 event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
            )
            .into_boxed_str(),
        )
        .as_bytes();
        let (out, text, prog) = run(script);
        assert!(out.is_ok(), "{out:?}");
        assert_eq!(text, "hello");
        // The comment must never leak into the reply text, and must be decoded.
        // A cache HIT must be readable as one: 6 of 8 prompt tokens reused.
        assert_eq!(prog, vec![(6, 8, 6), (8, 8, 6)]);
    }

    #[test]
    fn mid_stream_error_event_fails_the_request() {
        // Regression: this used to be ignored, so a failed generation arrived
        // in the UI as a successful reply with no text at all.
        let script: &'static [u8] = Box::leak(
            format!(
                "{HEAD}\
                 event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"partial\"}}}}\n\n\
                 event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"cluster prefill failed\"}}}}\n\n"
            )
            .into_boxed_str(),
        )
        .as_bytes();
        let (out, text, _) = run(script);
        assert_eq!(out.unwrap_err(), "cluster prefill failed");
        assert_eq!(text, "partial", "text received before the error must survive");
    }

    #[test]
    fn openai_shaped_error_frame_also_fails() {
        let script: &'static [u8] = Box::leak(
            format!("{HEAD}data: {{\"error\":{{\"type\":\"api_error\",\"message\":\"boom\"}}}}\n\n")
                .into_boxed_str(),
        )
        .as_bytes();
        assert_eq!(run(script).0.unwrap_err(), "boom");
    }

    #[test]
    fn context_overflow_413_reaches_the_user_as_its_own_sentence() {
        // The coordinator refuses an over-long conversation with a 413 whose
        // body carries diagnostic fields ALONGSIDE `error`. The parser must
        // still find the message: this is the one error a user is expected to
        // act on, and it is useless if it arrives as "unexpected response".
        let body = "{\"error\":{\"type\":\"context_length_exceeded\",\"message\":\"This \
conversation needs 33000 tokens but the cluster is running a 32768-token context. \
Start a new conversation, or raise the context tier in Settings and restart the \
cluster.\"},\"prompt_tokens\":32984,\"context_size\":32768}";
        let script: &'static [u8] = Box::leak(
            format!(
                "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        )
        .as_bytes();
        let err = run(script).0.unwrap_err();
        assert!(err.starts_with("This conversation needs 33000 tokens"), "{err}");
        assert!(err.contains("raise the context tier"), "{err}");
    }

    #[test]
    fn non_sse_error_body_is_surfaced_verbatim() {
        let body = "{\"error\":{\"message\":\"all sequence slots busy\"}}";
        let script: &'static [u8] = Box::leak(
            format!(
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        )
        .as_bytes();
        assert_eq!(run(script).0.unwrap_err(), "all sequence slots busy");
    }
}
