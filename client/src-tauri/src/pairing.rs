// LAN pairing (acceptance P3) — the client-side layer that forms a cluster
// roster before any engine process exists, then materializes it as a real
// engine cluster (philosophy 17: engine keeps its simple contract, the client
// owns the UX-level orchestration).
//
// Flow:
//   create  → mint code (JS side), broadcast a UDP beacon carrying a HASH of
//             the code (never the code itself) + our roster TCP port, and
//             serve a tiny newline-JSON roster protocol over TCP.
//   join    → listen for a beacon whose hash matches the typed code, then
//             register over TCP (the full code is the proof) and poll the
//             roster.
//   start   → the creator freezes the roster: the chosen coordinator machine
//             starts `idletoken-coord --num-workers N --http`, every other
//             machine starts `idletoken-worker --coordinator <addr>`. The engine
//             supervisor's backoff restart makes start order irrelevant —
//             workers that dial too early simply retry.
//   ready   → everyone polls the coordinator's `GET /v1/cluster/status` and
//             merges the real stage/layer plan into the roster.
//
// State is pushed to the webview as `pairing:status` events mirroring the TS
// `PairingSnapshot` (client/src/pairing.ts).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};

/// UDP port the creator broadcasts beacons on (joiner binds it to listen).
/// Default for Tuning::discovery_port (settings.discoveryPort).
const DISCOVERY_PORT: u16 = 14099;
/// TCP port of the creator's roster service.
const ROSTER_PORT: u16 = 14098;
/// Engine defaults (worker-facing coord port / coord HTTP API port).
const COORD_PORT: u16 = 14100;
const API_PORT: u16 = 8000;
/// Default inter-stage (HC) port a worker binds (settings.interStagePort).
/// The coordinator machine's co-located worker binds this + 1 so it never
/// collides with a worker sharing the same host — e.g. two test instances on
/// one box, or any future same-machine layout.
const INTER_STAGE_PORT: u16 = 14101;
/// Port the coordinator serves the layer-shard weight repo on (an isolated
/// `idletoken-worker --serve-weights` sidecar). Remote workers fetch only their
/// assigned layers from here instead of holding the whole 80GB GGUF.
const WEIGHT_PORT: u16 = 8001;

/// Engine tuning derived from the client's settings panel and passed along
/// with `pairing_create` / `pairing_join` (task 1.2: AppSettings → engine CLI
/// args). Every field defaults to the historical hard-coded value, so an old
/// caller that omits `tuning` behaves exactly as before.
///
/// Scope note (honesty rule): only settings the engine really implements are
/// carried here. KV size/TTL/eviction and computeMode have no engine
/// implementation yet — they stay `reserved` in settings.ts instead of being
/// silently dropped or faked as flags.
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Tuning {
    /// Coord HTTP API bind host (settings.apiHost) → `--api-bind host:port`.
    api_host: String,
    /// Coord HTTP API port (settings.apiPort). The creator broadcasts it via
    /// the roster so joiners poll /v1/cluster/status on the right port.
    api_port: u16,
    /// Coord API access token (settings.apiToken) → `--api-token`. Empty =
    /// no auth (LAN default). Never broadcast over the roster.
    api_token: String,
    /// Worker inter-stage bind port (settings.interStagePort) → `--bind`.
    inter_stage_port: u16,
    /// UDP beacon port for THIS pairing layer (settings.discoveryPort).
    /// Creator and joiner must agree — it is each machine's own setting.
    discovery_port: u16,
    /// Model to serve (settings.modelId) → coord `--model-id`. The CREATOR
    /// decides the cluster's model; joiners adopt it from the roster broadcast
    /// (same pattern as api_port) so every client shows the same model.
    model_id: String,
    /// Selected precision (settings.quant) → coord `--quant`. Empty = the
    /// model's default variant / single-precision models. Broadcast with the
    /// model_id so joiners host the same precision.
    quant: String,
    /// Context window (settings.tier → ctx) → coord `--ctx-size`. Feeds the
    /// engine's mode decision + per-node overhead in the layer split.
    ctx_size: u32,
    /// Per-request generation ceiling (settings.maxTokens) → coord
    /// `--max-decode`. Configuration, not a compiled-in constant: 4096 used to
    /// be hardcoded in the engine, so "Max tokens per reply" could not raise it.
    /// 0 = bounded only by the remaining context.
    #[serde(default = "default_max_decode")]
    max_decode: u32,
}

/// 0 = context-bound, matching DEFAULT_SETTINGS.maxTokens. A different number
/// here would mean a caller that omits the field gets a ceiling the UI never
/// shows — the divergence is only reachable from a stale/hand-written payload,
/// which is exactly when a silent cap would be hardest to diagnose.
fn default_max_decode() -> u32 {
    0
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            api_host: "0.0.0.0".into(),
            api_port: API_PORT,
            api_token: String::new(),
            inter_stage_port: INTER_STAGE_PORT,
            discovery_port: DISCOVERY_PORT,
            model_id: "deepseek-v4-flash".into(),
            quant: String::new(),
            ctx_size: 8192,
            max_decode: default_max_decode(),
        }
    }
}

const BEACON_MAGIC: &str = "HOMEAI1";

/// Windows only: allow the pairing traffic *inbound* before we start listening.
///
/// The engine self-provisions its own ports (`idletoken_win_ensure_firewall_rule`),
/// but the client's two ports are opened by this process, so nothing was
/// provisioning them. On a freshly installed Windows machine the joiner then
/// never sees the creator's UDP beacon and reports "no cluster found for that
/// code on this LAN" — a silent dead end that looks like a discovery bug.
///
/// Idempotent: `show rule` exits 0 when the rule already exists. Adding needs
/// elevation; when we are not elevated we print the exact command instead of
/// pretending it worked (the user can run it once, or install elevated).
#[cfg(windows)]
fn ensure_pairing_firewall(discovery_port: u16) {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000; // no console flash on a GUI app

    for (proto, port) in [("UDP", discovery_port), ("TCP", ROSTER_PORT)] {
        let name = format!("IdleToken client {proto} {port}");
        let exists = Command::new("netsh")
            .args(["advfirewall", "firewall", "show", "rule", &format!("name={name}")])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if exists {
            continue;
        }
        let added = Command::new("netsh")
            .args([
                "advfirewall", "firewall", "add", "rule",
                &format!("name={name}"),
                "dir=in", "action=allow",
                &format!("protocol={proto}"),
                &format!("localport={port}"),
                "profile=any",
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if added {
            eprintln!("[pairing] firewall rule added: {name}");
        } else {
            eprintln!(
                "[pairing] could not add firewall rule (not elevated?). Run once as admin:\n  \
                 netsh advfirewall firewall add rule name=\"{name}\" dir=in action=allow \
                 protocol={proto} localport={port} profile=any"
            );
        }
    }
}

#[cfg(not(windows))]
fn ensure_pairing_firewall(_discovery_port: u16) {}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")] // TS PeerNode: layerLo / layerHi
pub struct Peer {
    id: String,
    hostname: String,
    gpu: String,
    role: &'static str, // "coordinator" | "worker"
    #[serde(rename = "self")]
    is_self: bool,
    stage: String, // NodeStage: joined|probing|assigned|loading|ready
    #[serde(skip_serializing_if = "Option::is_none")]
    layer_lo: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer_hi: Option<u32>,
    #[serde(skip)]
    ip: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Off,
    Creator,
    Joiner,
}

struct Inner {
    mode: Mode,
    code: Option<String>,
    self_id: String,
    self_host: String,
    self_gpu: String,
    peers: Vec<Peer>,
    coordinator_id: Option<String>,
    /// idle (roster forming) | starting (engines launching) | ready
    phase: String,
    /// ip of the machine running idletoken-coord (set at start)
    coord_ip: Option<String>,
    /// this machine's local GGUF path (from settings.ggufPath). Empty = mock
    /// load (P3: no model needed). Non-empty = real DSv4 weights → passed to
    /// this machine's coord/worker so they load the model (P4/P6, task #5).
    model_path: String,
    /// URL of the coordinator's layer-shard weight repo (set by the creator
    /// when it has a model, propagated to joiners via the roster). Remote
    /// workers fetch only their layers from here.
    shard_repo: String,
    engine_started: bool,
    /// Account-mode pairing (integration plan 3.3): the "code" is a secret
    /// derived on the JS side from stable account material (platform user id +
    /// platform URL + cluster name), not a human-typed 6-char code. The wire
    /// mechanics are identical (beacon broadcasts only the FNV hash, the full
    /// secret is the TCP join proof) — this flag only changes presentation:
    /// the snapshot hides `code` (nothing to read aloud) and sets
    /// `accountMode` so the UI labels the cluster as account-formed.
    account_mode: bool,
    generation: u64,
    /// Settings-derived engine tuning (defaults = historical hard-coded
    /// ports). On a joiner, `api_port` is overwritten by the roster so it
    /// polls the coordinator on the creator's configured port.
    tuning: Tuning,
}

pub struct Pairing(Mutex<Inner>);

impl Default for Pairing {
    fn default() -> Self {
        Pairing(Mutex::new(Inner {
            mode: Mode::Off,
            code: None,
            self_id: String::new(),
            self_host: String::new(),
            self_gpu: String::new(),
            peers: Vec::new(),
            coordinator_id: None,
            phase: "idle".into(),
            coord_ip: None,
            model_path: String::new(),
            shard_repo: String::new(),
            engine_started: false,
            account_mode: false,
            generation: 0,
            tuning: Tuning::default(),
        }))
    }
}

/// FNV-1a of the code — enough to match a beacon to a typed code without
/// broadcasting the code itself (the TCP join carries the real code as proof).
fn code_hash(code: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in code.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn snapshot_json(inner: &Inner) -> Value {
    json!({
        // Account mode: the derived secret is not a shareable code — hide it.
        "code": if inner.account_mode { &None } else { &inner.code },
        "accountMode": inner.account_mode,
        "peers": inner.peers,
        "coordinatorId": inner.coordinator_id,
        "phase": inner.phase,
        "api": if inner.phase == "ready" {
            let api_port = inner.tuning.api_port;
            inner.coord_ip.as_ref().map(|ip| json!({
                "baseUrl": format!("http://{ip}:{api_port}"),
                "status": "online",
            }))
        } else { None },
        "source": "engine",
        "canStart": inner.mode == Mode::Creator && inner.phase == "idle" && inner.peers.len() >= 2,
    })
}

fn emit_snapshot(app: &AppHandle) {
    let pairing = app.state::<Pairing>();
    let inner = pairing.0.lock().unwrap();
    let snap = snapshot_json(&inner);
    drop(inner);
    let _ = app.emit("pairing:status", &snap);
}

fn roster_reply(inner: &Inner) -> Value {
    json!({
        "ok": true,
        "phase": inner.phase,
        "coordinatorId": inner.coordinator_id,
        "coordIp": inner.coord_ip,
        "shardRepo": inner.shard_repo,
        // The creator's configured API port, so a joiner with different (or
        // default) settings still polls /v1/cluster/status on the right port.
        // The token itself is NOT broadcast — status stays unauthenticated.
        "apiPort": inner.tuning.api_port,
        // The cluster's model is the CREATOR's choice; joiners adopt it so
        // their UI/status reflect what the coordinator actually serves (the
        // engine additionally enforces this via ASSIGN_PLAN's model_id).
        "modelId": inner.tuning.model_id,
        // Precision travels with the model so joiners host the same variant.
        "quant": inner.tuning.quant,
        "members": inner.peers.iter().map(|p| json!({
            "id": p.id, "hostname": p.hostname, "gpu": p.gpu, "ip": p.ip,
            "stage": p.stage,
        })).collect::<Vec<_>>(),
    })
}

/// Map an engine lifecycle state to the roster's per-node stage (P4 live
/// progress). "ready" is set only by merge_engine_status (coordinator truth).
fn stage_for_engine(state: &str) -> &'static str {
    match state {
        "starting" | "restarting" | "running" => "loading",
        "crashed" => "error",
        _ => "joined",
    }
}

/// Apply a roster reply on the joiner side. Returns true when the engine for
/// this machine must be started now (phase flipped to starting).
fn apply_roster(inner: &mut Inner, v: &Value) -> bool {
    let members = v["members"].as_array().cloned().unwrap_or_default();
    let coordinator_id = v["coordinatorId"].as_str().map(String::from);
    let phase = v["phase"].as_str().unwrap_or("idle").to_string();
    let coord_ip = v["coordIp"].as_str().map(String::from);
    if let Some(sr) = v["shardRepo"].as_str() {
        if !sr.is_empty() {
            inner.shard_repo = sr.to_string();
        }
    }
    // Adopt the creator's API port (see roster_reply): the coordinator's
    // engine binds it, so status polling and the exposed baseUrl must match.
    if let Some(p) = v["apiPort"].as_u64() {
        if p > 0 && p <= u16::MAX as u64 {
            inner.tuning.api_port = p as u16;
        }
    }
    // Adopt the creator's model: the cluster serves ONE model and the
    // coordinator picked it. A joiner's own modelId setting only applies when
    // it creates a cluster itself.
    if let Some(m) = v["modelId"].as_str() {
        if !m.is_empty() {
            inner.tuning.model_id = m.to_string();
        }
    }
    // Adopt the creator's precision alongside the model (absent for older
    // creators → keep our default, which the coord maps to the model default).
    if let Some(q) = v["quant"].as_str() {
        inner.tuning.quant = q.to_string();
    }

    inner.peers = members
        .iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            Peer {
                is_self: id == inner.self_id,
                role: if Some(id.as_str()) == coordinator_id.as_deref() { "coordinator" } else { "worker" },
                stage: m["stage"].as_str().unwrap_or("joined").to_string(),
                id,
                hostname: m["hostname"].as_str().unwrap_or("").to_string(),
                gpu: m["gpu"].as_str().unwrap_or("").to_string(),
                layer_lo: None,
                layer_hi: None,
                ip: m["ip"].as_str().unwrap_or("").to_string(),
            }
        })
        .collect();
    inner.coordinator_id = coordinator_id;
    let must_start = phase == "starting" && inner.phase == "idle" && !inner.engine_started;
    if inner.phase != "ready" {
        inner.phase = phase;
    }
    inner.coord_ip = coord_ip;
    must_start
}

/// Model-path args for one of this machine's engines (empty = mock load, P3).
/// The coord and worker binaries spell the flag differently: coord takes
/// `--model-path` (it propagates the path to workers via ASSIGN_PLAN), the
/// worker takes `--model`.
fn model_args(role: &str, model_path: &str) -> Vec<String> {
    if model_path.is_empty() {
        Vec::new()
    } else if role == "coordinator" {
        vec!["--model-path".into(), model_path.to_string()]
    } else {
        vec!["--model".into(), model_path.to_string()]
    }
}

/// Start this machine's engine(s) per its role in the frozen roster. The
/// coordinator machine runs BOTH idletoken-coord AND a co-located idletoken-worker so
/// it also contributes compute (task #4); a plain worker machine runs one
/// worker. `--num-workers` therefore counts every non-coordinator machine plus
/// the coordinator's own worker.
fn materialize_engine(app: &AppHandle) {
    let (is_coord, coord_ip, workers, model_path, shard_repo, tuning) = {
        let pairing = app.state::<Pairing>();
        let mut inner = pairing.0.lock().unwrap();
        if inner.engine_started {
            return;
        }
        inner.engine_started = true;
        let is_coord = inner.coordinator_id.as_deref() == Some(inner.self_id.as_str());
        // one worker per non-coordinator machine + one on the coordinator itself
        let workers = inner.peers.iter().filter(|p| p.role != "coordinator").count() + 1;
        (
            is_coord,
            inner.coord_ip.clone().unwrap_or_default(),
            workers,
            inner.model_path.clone(),
            inner.shard_repo.clone(),
            inner.tuning.clone(),
        )
    };
    if is_coord {
        // 0) the layer-shard weight repo: an isolated idletoken-worker sidecar
        //    serving the master GGUF over HTTP byte-range so remote workers
        //    fetch only their layers. Only when we actually have the model.
        if !model_path.is_empty() {
            let weights_args = vec![
                "--serve-weights".into(), model_path.clone(),
                "--weights-port".into(), WEIGHT_PORT.to_string(),
            ];
            if let Err(e) = crate::engine::start_engine(app, "weights".into(), weights_args, Vec::new()) {
                eprintln!("[pairing] weight server start failed: {e}");
            }
        }
        // 1) the coordinator process (loads the model locally for the tokenizer)
        let mut coord_args = vec![
            "--bind".into(), format!("0.0.0.0:{COORD_PORT}"),
            "--num-workers".into(), workers.to_string(),
            "--http".into(),
            "--api-bind".into(), format!("{}:{}", tuning.api_host, tuning.api_port),
            "--n-predict".into(), "0".into(),
            "--model-id".into(), tuning.model_id.clone(),
            "--ctx-size".into(), tuning.ctx_size.to_string(),
            "--max-decode".into(), tuning.max_decode.to_string(),
        ];
        if !tuning.api_token.is_empty() {
            coord_args.push("--api-token".into());
            coord_args.push(tuning.api_token.clone());
        }
        // Selected precision (small models). Empty = coord uses the model's
        // default variant, so only pass the flag when a quant was chosen.
        if !tuning.quant.is_empty() {
            coord_args.push("--quant".into());
            coord_args.push(tuning.quant.clone());
        }
        coord_args.extend(model_args("coordinator", &model_path));
        if let Err(e) = crate::engine::start_engine(app, "coordinator".into(), coord_args, Vec::new()) {
            eprintln!("[pairing] coord start failed: {e}");
        }
        // 2) a co-located worker so the coordinator machine also serves layers,
        //    dialing the coord over loopback. It has the model locally, so it
        //    loads from disk (no shard fetch). Binds interStagePort+1 (default
        //    14102) so it never collides with a plain worker on the same host.
        let mut worker_args = vec![
            "--coordinator".into(), format!("127.0.0.1:{COORD_PORT}"),
            "--bind".into(), format!("0.0.0.0:{}", tuning.inter_stage_port.saturating_add(1)),
        ];
        worker_args.extend(model_args("worker", &model_path));
        if let Err(e) = crate::engine::start_engine(app, "worker".into(), worker_args, Vec::new()) {
            eprintln!("[pairing] co-located worker start failed: {e}");
        }
    } else {
        // Remote worker: fetch only our layers from the coordinator's repo when
        // one was advertised (no local 80GB GGUF needed); else fall back to a
        // local model path if this machine happens to have one.
        let mut worker_args = vec![
            "--coordinator".into(), format!("{coord_ip}:{COORD_PORT}"),
            "--bind".into(), format!("0.0.0.0:{}", tuning.inter_stage_port),
        ];
        let env = if !shard_repo.is_empty() {
            vec![("IDLETOKEN_SHARD_REPO".into(), shard_repo.clone())]
        } else {
            worker_args.extend(model_args("worker", &model_path));
            Vec::new()
        };
        if let Err(e) = crate::engine::start_engine(app, "worker".into(), worker_args, env) {
            eprintln!("[pairing] worker start failed: {e}");
        }
    }
    emit_snapshot(app);
}

/// Blocking HTTP GET with a short timeout; returns the body. Raw TCP on
/// purpose — one tiny LAN GET does not justify an HTTP client dependency.
fn http_get_body(ip: &str, port: u16, path: &str) -> Option<String> {
    let addr: SocketAddr = format!("{ip}:{port}").parse().ok()?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: {ip}\r\nConnection: close\r\n\r\n").ok()?;

    // Read until Content-Length is satisfied, or until EOF, and KEEP whatever
    // arrived if the read times out.
    //
    // This used to be `read_to_string`, which waits for EOF and — on timeout —
    // returns Err and throws away everything it buffered. The coordinator
    // answers `Connection: close` in the header and then leaves the socket
    // open (verified 2026-08-11: 256 bytes delivered, no FIN, read times out).
    // So the client discarded a perfectly good reply once a second, the
    // cluster never flipped to "ready", and the UI sat at "starting" forever
    // while the very same coordinator was serving inference.
    //
    // Trusting a server's `Connection: close` is optional; reading the length
    // it told us is not.
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,                       // clean EOF
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if let Some(end) = find_header_end(&buf) {
                    match content_length(&buf[..end]) {
                        Some(len) if buf.len() - end >= len => break, // whole body in hand
                        None => break, // no length: EOF is the only terminator, take what came
                        _ => {}
                    }
                }
            }
            Err(_) => break, // timeout or reset: use what we already have
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let body = text.split_once("\r\n\r\n")?.1.to_string();
    Some(body)
}

/// Index just past the blank line that ends the HTTP headers.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

/// `Content-Length` from a header block, case-insensitively.
fn content_length(headers: &[u8]) -> Option<usize> {
    let text = String::from_utf8_lossy(headers);
    text.lines()
        .find_map(|l| l.split_once(':').filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length")))
        .and_then(|(_, v)| v.trim().parse().ok())
}

/// Poll the engine coordinator's status API and merge the real stage/layer
/// plan into the roster (creator and joiners both run this once starting).
fn merge_engine_status(app: &AppHandle) -> bool {
    let (coord_ip, api_port) = {
        let pairing = app.state::<Pairing>();
        let inner = pairing.0.lock().unwrap();
        match (&inner.coord_ip, inner.phase.as_str()) {
            (Some(ip), "starting") | (Some(ip), "ready") => (ip.clone(), inner.tuning.api_port),
            _ => return false,
        }
    };
    let Some(body) = http_get_body(&coord_ip, api_port, "/v1/cluster/status") else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&body) else {
        return false;
    };
    let members = v["members"].as_array().cloned().unwrap_or_default();
    let pairing = app.state::<Pairing>();
    let mut inner = pairing.0.lock().unwrap();
    for p in inner.peers.iter_mut() {
        // The coord only serves this status once the whole cluster is ready, so
        // every roster peer is ready here. Layers are attributed by hostname
        // where the engine reports one (two test instances sharing a machine's
        // hostname can't be told apart — the layer plan itself is asserted at
        // the engine level; see P3).
        if let Some(m) = members.iter().find(|m| m["hostname"].as_str() == Some(p.hostname.as_str())) {
            p.layer_lo = m["layer_lo"].as_u64().map(|x| x as u32);
            p.layer_hi = m["layer_hi"].as_u64().map(|x| x as u32);
        }
        p.stage = "ready".into();
    }
    inner.phase = "ready".into();
    drop(inner);
    emit_snapshot(app);
    true
}

/// One roster-protocol request on the creator side.
fn handle_roster_conn(app: &AppHandle, stream: TcpStream, generation: u64) {
    let peer_ip = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let Ok(req) = serde_json::from_str::<Value>(&line) else { return };
    let pairing = app.state::<Pairing>();
    let mut inner = pairing.0.lock().unwrap();
    if inner.generation != generation || inner.mode != Mode::Creator {
        return;
    }
    let reply = match req["op"].as_str() {
        Some("join") => {
            if req["code"].as_str() != inner.code.as_deref() {
                json!({"ok": false, "err": "bad code"})
            } else {
                let id = req["hostname"].as_str().unwrap_or("?").to_string();
                if !inner.peers.iter().any(|p| p.id == id) {
                    inner.peers.push(Peer {
                        id: id.clone(),
                        hostname: id.clone(),
                        gpu: req["gpu"].as_str().unwrap_or("").to_string(),
                        role: "worker",
                        is_self: false,
                        stage: "joined".into(),
                        layer_lo: None,
                        layer_hi: None,
                        ip: peer_ip.clone(),
                    });
                }
                let mut r = roster_reply(&inner);
                r["id"] = json!(id);
                r
            }
        }
        Some("roster") => {
            // Live per-node progress: a member's poll carries its engine state.
            let id = req["id"].as_str().unwrap_or("");
            let eng = req["engine"].as_str().unwrap_or("");
            if !id.is_empty() && !eng.is_empty() && inner.phase != "idle" {
                if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                    if p.stage != "ready" {
                        p.stage = stage_for_engine(eng).to_string();
                    }
                }
            }
            roster_reply(&inner)
        }
        Some("leave") => {
            let id = req["id"].as_str().unwrap_or("");
            inner.peers.retain(|p| p.id != id);
            json!({"ok": true})
        }
        _ => json!({"ok": false, "err": "bad op"}),
    };
    drop(inner);
    emit_snapshot(app);
    let mut stream = reader.into_inner();
    let _ = writeln!(stream, "{reply}");
}

fn spawn_creator_tasks(app: AppHandle, generation: u64, code: String, discovery_port: u16) {
    // UDP beacon: `HOMEAI1|<fnv1a(code)>|<roster_port>` once a second.
    let beacon_app = app.clone();
    let hash = code_hash(&code);
    std::thread::spawn(move || {
        let Ok(sock) = UdpSocket::bind(("0.0.0.0", 0)) else { return };
        let _ = sock.set_broadcast(true);
        let msg = format!("{BEACON_MAGIC}|{hash}|{ROSTER_PORT}");
        loop {
            {
                let pairing = beacon_app.state::<Pairing>();
                let inner = pairing.0.lock().unwrap();
                if inner.generation != generation {
                    return;
                }
                // Keep announcing while forming; stop once started.
                if inner.phase != "idle" {
                    return;
                }
            }
            let _ = sock.send_to(msg.as_bytes(), ("255.255.255.255", discovery_port));
            let _ = sock.send_to(msg.as_bytes(), ("127.0.0.1", discovery_port)); // same-host joiners
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    // Roster TCP service. Nonblocking accept so the generation check can end it.
    std::thread::spawn(move || {
        // A superseded generation's listener may take a beat to drop; retry
        // briefly instead of failing the whole create.
        let mut listener = None;
        for _ in 0..10 {
            match TcpListener::bind(("0.0.0.0", ROSTER_PORT)) {
                Ok(l) => {
                    listener = Some(l);
                    break;
                }
                Err(_) => std::thread::sleep(Duration::from_millis(500)),
            }
        }
        let Some(listener) = listener else {
            eprintln!("[pairing] roster port {ROSTER_PORT} busy");
            return;
        };
        let _ = listener.set_nonblocking(true);
        loop {
            {
                let pairing = app.state::<Pairing>();
                let inner = pairing.0.lock().unwrap();
                if inner.generation != generation {
                    return;
                }
            }
            match listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    handle_roster_conn(&app, stream, generation);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(200)),
            }
        }
    });
}

/// Joiner: wait for a matching beacon, register, then poll the roster.
/// `discovery_port` is this machine's settings.discoveryPort — creator and
/// joiner must be configured alike for the beacon to be heard.
fn spawn_joiner_tasks(app: AppHandle, generation: u64, code: String, discovery_port: u16) {
    std::thread::spawn(move || {
        // 1) discover the creator via beacon
        let hash = code_hash(&code);
        let creator_ip: Option<String> = (|| {
            let sock = UdpSocket::bind(("0.0.0.0", discovery_port)).ok()?;
            sock.set_read_timeout(Some(Duration::from_secs(1))).ok()?;
            let mut buf = [0u8; 256];
            for _ in 0..8 {
                {
                    let pairing = app.state::<Pairing>();
                    if pairing.0.lock().unwrap().generation != generation {
                        return None;
                    }
                }
                if let Ok((n, src)) = sock.recv_from(&mut buf) {
                    let msg = String::from_utf8_lossy(&buf[..n]);
                    let parts: Vec<&str> = msg.trim().split('|').collect();
                    if parts.len() == 3 && parts[0] == BEACON_MAGIC && parts[1] == hash {
                        return Some(src.ip().to_string());
                    }
                }
            }
            None
        })();

        let Some(creator_ip) = creator_ip else {
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            if inner.generation == generation {
                inner.mode = Mode::Off;
                inner.phase = "idle".into();
                inner.code = None;
            }
            drop(inner);
            emit_snapshot(&app);
            eprintln!("[pairing] no cluster found for that code on this LAN");
            return;
        };

        // 2) join + poll loop over the roster protocol
        let (self_host, self_gpu) = {
            let pairing = app.state::<Pairing>();
            let inner = pairing.0.lock().unwrap();
            (inner.self_host.clone(), inner.self_gpu.clone())
        };
        let mut joined = false;
        loop {
            {
                let pairing = app.state::<Pairing>();
                if pairing.0.lock().unwrap().generation != generation {
                    return;
                }
            }
            let req = if joined {
                // report this machine's live engine state so the whole roster
                // sees per-node progress (P4)
                json!({"op": "roster", "id": self_host,
                       "engine": crate::engine::current_state_str(&app)})
            } else {
                json!({"op": "join", "code": code, "hostname": self_host, "gpu": self_gpu})
            };
            let reply: Option<Value> = (|| {
                let addr: SocketAddr = format!("{creator_ip}:{ROSTER_PORT}").parse().ok()?;
                let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
                s.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
                writeln!(s, "{req}").ok()?;
                let mut line = String::new();
                BufReader::new(s).read_line(&mut line).ok()?;
                serde_json::from_str(&line).ok()
            })();
            if let Some(v) = reply {
                if v["ok"].as_bool() == Some(false) {
                    eprintln!("[pairing] join rejected: {}", v["err"]);
                    return;
                }
                joined = true;
                let must_start = {
                    let pairing = app.state::<Pairing>();
                    let mut inner = pairing.0.lock().unwrap();
                    if inner.generation != generation {
                        return;
                    }
                    apply_roster(&mut inner, &v)
                };
                emit_snapshot(&app);
                if must_start {
                    materialize_engine(&app);
                }
                // after starting, also watch the engine coordinator for readiness
                merge_engine_status(&app);
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    });
}

/// Creator-side readiness watcher (joiners poll inside their roster loop).
fn spawn_ready_watcher(app: AppHandle, generation: u64) {
    std::thread::spawn(move || loop {
        {
            let st = crate::engine::current_state_str(&app);
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            if inner.generation != generation {
                return;
            }
            if inner.phase == "ready" {
                return;
            }
            // creator's own live progress into the roster (P4)
            let self_id = inner.self_id.clone();
            if let Some(p) = inner.peers.iter_mut().find(|p| p.id == self_id) {
                if p.stage != "ready" {
                    p.stage = stage_for_engine(st).to_string();
                }
            }
        }
        emit_snapshot(&app);
        if merge_engine_status(&app) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    });
}

// ---- commands ---------------------------------------------------------------

#[tauri::command]
pub fn pairing_create(
    app: AppHandle,
    state: State<'_, Pairing>,
    code: String,
    hostname: String,
    gpu: String,
    model_path: Option<String>,
    tuning: Option<Tuning>,
    account: Option<bool>,
) -> Result<(), String> {
    let generation;
    let discovery_port;
    {
        let mut inner = state.0.lock().unwrap();
        inner.generation += 1;
        generation = inner.generation;
        inner.mode = Mode::Creator;
        inner.account_mode = account.unwrap_or(false);
        inner.code = Some(code.clone());
        inner.self_id = hostname.clone();
        inner.self_host = hostname.clone();
        inner.self_gpu = gpu.clone();
        inner.coordinator_id = Some(hostname.clone());
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.model_path = model_path.unwrap_or_default();
        inner.shard_repo = String::new();
        inner.engine_started = false;
        inner.tuning = tuning.unwrap_or_default();
        discovery_port = inner.tuning.discovery_port;
        inner.peers = vec![Peer {
            id: hostname.clone(),
            hostname,
            gpu,
            role: "coordinator",
            is_self: true,
            stage: "joined".into(),
            layer_lo: None,
            layer_hi: None,
            ip: String::new(),
        }];
    }
    emit_snapshot(&app);
    ensure_pairing_firewall(discovery_port);
    spawn_creator_tasks(app, generation, code, discovery_port);
    Ok(())
}

#[tauri::command]
pub fn pairing_join(
    app: AppHandle,
    state: State<'_, Pairing>,
    code: String,
    hostname: String,
    gpu: String,
    model_path: Option<String>,
    tuning: Option<Tuning>,
    account: Option<bool>,
) -> Result<(), String> {
    let generation;
    let discovery_port;
    {
        let mut inner = state.0.lock().unwrap();
        inner.generation += 1;
        generation = inner.generation;
        inner.mode = Mode::Joiner;
        inner.account_mode = account.unwrap_or(false);
        inner.code = None; // only the creator shows the code
        inner.self_id = hostname.clone();
        inner.self_host = hostname;
        inner.self_gpu = gpu;
        inner.coordinator_id = None;
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.model_path = model_path.unwrap_or_default();
        inner.shard_repo = String::new();
        inner.engine_started = false;
        inner.peers = Vec::new();
        inner.tuning = tuning.unwrap_or_default();
        discovery_port = inner.tuning.discovery_port;
    }
    emit_snapshot(&app);
    ensure_pairing_firewall(discovery_port);
    spawn_joiner_tasks(app, generation, code, discovery_port);
    Ok(())
}

/// Creator freezes the roster and everyone launches engines. The chosen
/// coordinator machine's ip comes from the roster (creator = local).
#[tauri::command]
pub fn pairing_start(
    app: AppHandle,
    state: State<'_, Pairing>,
    allow_solo: Option<bool>,
) -> Result<(), String> {
    let generation;
    {
        let mut inner = state.0.lock().unwrap();
        if inner.mode != Mode::Creator {
            return Err("only the cluster creator can start it".into());
        }
        // Two machines is the right floor for the PAIRING flow — pressing Start
        // before anyone joined is a mistake there. It is the wrong floor for the
        // single-machine flow, whose entire point is one machine, and which the
        // rest of this path already supports: materialize_engine computes
        // `workers = non-coordinator peers + 1`, i.e. 1 co-located worker.
        //
        // Until 2026-08-11 there was no way to say which flow you were in, so
        // "Run it here" downloaded the weights and then died on this line with
        // "need at least 2 machines" — the headline single-node feature could
        // not start at all.
        if inner.peers.len() < 2 && allow_solo != Some(true) {
            return Err("need at least 2 machines".into());
        }
        generation = inner.generation;
        let coord_id = inner.coordinator_id.clone().unwrap_or_else(|| inner.self_id.clone());
        let coord_ip = inner
            .peers
            .iter()
            .find(|p| p.id == coord_id)
            .map(|p| if p.is_self || p.ip.is_empty() { local_lan_ip() } else { p.ip.clone() })
            .unwrap_or_else(local_lan_ip);
        // When the coordinator has the model, advertise its layer-shard repo so
        // remote workers fetch only their layers. URL = the master's basename
        // under the coordinator's weight server.
        if !inner.model_path.is_empty() {
            let base = inner
                .model_path
                .rsplit(|c| c == '/' || c == '\\')
                .next()
                .unwrap_or(&inner.model_path)
                .to_string();
            inner.shard_repo = format!("http://{coord_ip}:{WEIGHT_PORT}/{base}");
        }
        inner.coord_ip = Some(coord_ip);
        inner.phase = "starting".into();
        for p in inner.peers.iter_mut() {
            p.stage = "loading".into();
        }
    }
    emit_snapshot(&app);
    materialize_engine(&app);
    spawn_ready_watcher(app, generation);
    Ok(())
}

#[tauri::command]
pub fn pairing_set_coordinator(
    app: AppHandle,
    state: State<'_, Pairing>,
    peer_id: String,
) -> Result<(), String> {
    {
        let mut inner = state.0.lock().unwrap();
        if inner.mode != Mode::Creator {
            return Err("only the cluster creator can pick the coordinator".into());
        }
        if inner.phase != "idle" {
            return Err("cluster already started".into());
        }
        if !inner.peers.iter().any(|p| p.id == peer_id) {
            return Err("no such member".into());
        }
        inner.coordinator_id = Some(peer_id.clone());
        for p in inner.peers.iter_mut() {
            p.role = if p.id == peer_id { "coordinator" } else { "worker" };
        }
    }
    emit_snapshot(&app);
    Ok(())
}

#[tauri::command]
pub fn pairing_leave(app: AppHandle, state: State<'_, Pairing>) -> Result<(), String> {
    {
        let mut inner = state.0.lock().unwrap();
        inner.generation += 1; // ends beacon/roster/poll threads
        inner.mode = Mode::Off;
        inner.code = None;
        inner.peers.clear();
        inner.coordinator_id = None;
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.shard_repo = String::new();
        inner.engine_started = false;
        inner.account_mode = false;
    }
    // Leaving the cluster also stops this machine's engine process.
    let _ = crate::engine::stop_engine(&app);
    emit_snapshot(&app);
    Ok(())
}

#[tauri::command]
pub fn pairing_status(state: State<'_, Pairing>) -> Value {
    snapshot_json(&state.0.lock().unwrap())
}

/// Headless pairing entry (acceptance §8 sanctions a "headless mode" as a valid
/// product-gate mechanism). Lets an automated harness drive the real LAN
/// pairing path without the webview — needed where the GUI can't run (e.g. a
/// locked Windows session, or CI). Spec: `create:<CODE>:<name>` or
/// `join:<CODE>:<name>` (optional `:apiPort=<n>`, `:apiToken=<s>`,
/// `:model=<path>` — same tuning overrides as the UI-test directives, so the
/// P5 settings gate can also drive GUI-less machines). The creator auto-starts
/// once a second machine joins, mirroring the `pairing-auto-start` UI directive.
pub fn headless_pair(app: &AppHandle, spec: &str) {
    // op : code : name [ : apiPort=<n> ] [ : apiToken=<s> ] [ : model=<path> ]
    let mut op = "";
    let mut code = "";
    let mut name = "headless-node";
    let mut model = String::new();
    let mut tuning = Tuning::default();
    let mut tuned = false;
    for (i, tok) in spec.split(':').enumerate() {
        match i {
            0 => op = tok,
            1 => code = tok,
            2 => name = tok,
            _ => {
                if let Some(m) = tok.strip_prefix("model=") {
                    model = m.to_string();
                } else if let Some(p) = tok.strip_prefix("apiPort=") {
                    if let Ok(p) = p.parse::<u16>() {
                        tuning.api_port = p;
                        tuned = true;
                    }
                } else if let Some(t) = tok.strip_prefix("apiToken=") {
                    tuning.api_token = t.to_string();
                    tuned = true;
                }
            }
        }
    }
    if code.is_empty() {
        eprintln!("[pairing] headless: bad spec '{spec}'");
        return;
    }
    let code = code.to_string();
    let name = name.to_string();
    let model_opt = if model.is_empty() { None } else { Some(model) };
    // Headless mode has no settings panel — tuning stays at the defaults unless
    // the spec carries explicit apiPort=/apiToken= overrides (P5 gate).
    let tuning_opt = if tuned { Some(tuning) } else { None };
    eprintln!("[pairing] headless {op} code={code} as={name}");
    match op {
        "create" => {
            let _ = pairing_create(app.clone(), app.state(), code, name, "headless".into(), model_opt, tuning_opt, None);
            // Auto-start once a second machine joins (mirrors pairing-auto-start).
            let app2 = app.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_secs(1));
                let ready_to_start = {
                    let p = app2.state::<Pairing>();
                    let inner = p.0.lock().unwrap();
                    inner.mode == Mode::Creator && inner.phase == "idle" && inner.peers.len() >= 2
                };
                if ready_to_start {
                    let _ = pairing_start(app2.clone(), app2.state(), None);
                    return;
                }
            });
        }
        "join" => {
            let _ = pairing_join(app.clone(), app.state(), code, name, "headless".into(), model_opt, tuning_opt, None);
        }
        _ => eprintln!("[pairing] headless: unknown op '{op}'"),
    }
}

/// Best-effort LAN ip of this machine (the address peers should dial): open a
/// UDP socket "towards" a public address (no packet is sent) and read the
/// local address the OS picked.
fn local_lan_ip() -> String {
    UdpSocket::bind(("0.0.0.0", 0))
        .and_then(|s| {
            s.connect(("8.8.8.8", 80))?;
            s.local_addr()
        })
        .map(|a: SocketAddr| a.ip())
        .map(|ip: IpAddr| ip.to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}
