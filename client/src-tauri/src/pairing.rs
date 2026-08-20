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
//   start   → the creator freezes the roster: the chosen coordinator starts
//             idletoken-coord in llama.cpp mode; every other machine starts
//             idletoken-worker --rpc-supervisor. Pairing transports the RPC
//             TLS credential and the coordinator drives one llama-server.
//   ready   → everyone polls the coordinator's `GET /idletoken/v1/cluster/status` and
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
/// Default ggml-RPC port a worker binds (the setting retains its historical
/// interStagePort name for storage compatibility).
const INTER_STAGE_PORT: u16 = 14101;
const LLAMA_PORT: u16 = 18081;

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
    /// the roster so joiners poll /idletoken/v1/cluster/status on the right port.
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
    /// How much of THIS machine IdleToken may use (settings "This machine's
    /// usage") → worker `--max-vram-mb` / `--max-ram-mb`, MiB, 0 = no cap.
    ///
    /// Per-machine, so it is never adopted from the roster the way model_id and
    /// quant are: the creator's "conservative" says nothing about how much of
    /// YOUR computer you are lending. Until 2026-08-13 these never left the
    /// client — the probe was told, the serving worker was not, so the setting
    /// changed the dashboard's numbers and nothing else.
    #[serde(default)]
    max_vram_mb: u64,
    #[serde(default)]
    max_ram_mb: u64,
    // ---- pairing behaviour (settings "Pairing & discovery") ------------------
    // Wired on 2026-08-13. Before that these six were rendered as live controls
    // and read by nobody; settings.ts resets stored values once (schema v3)
    // because a value nothing obeyed is not a choice.
    /// Announce (creator) / listen for (joiner) the UDP discovery beacon.
    /// False = this machine pairs only through `manual_peers`.
    #[serde(default = "default_true")]
    lan_discovery: bool,
    /// Comma-separated IPs to dial directly when the beacon finds nothing.
    #[serde(default)]
    manual_peers: String,
    /// Roster poll interval in seconds (clamped 1..=60 at use).
    #[serde(default = "default_heartbeat")]
    heartbeat_sec: u32,
    /// Joiner asks the creator to hand it the coordinator role.
    #[serde(default)]
    prefer_coordinator: bool,
    /// Refuse peers outside this machine's /24.
    #[serde(default)]
    same_subnet_only: bool,
    /// "auto"/empty, or an IPv4 this machine binds to and advertises.
    #[serde(default)]
    bind_nic: String,
    // ---- overflow: borrow another machine when this one is full -------------
    //
    // Launch parameters. The coordinator reads them once, at start, and refuses
    // to start at all if overflow is asked for without an api_token -- which is
    // why these travel together with it rather than through a separate channel.
    //
    // Empty url = do not enable. There is deliberately no boolean: a flag and a
    // credential that can disagree is a flag that will.
    #[serde(default)]
    overflow_url: String,
    #[serde(default)]
    overflow_key: String,
    #[serde(default)]
    overflow_wait_s: u32,
    #[serde(default)]
    overflow_daily_cap_milli: u64,
}

fn default_true() -> bool {
    true
}

/// 1s — what the roster loop has always slept, so a payload that omits the
/// field behaves exactly as before it existed.
fn default_heartbeat() -> u32 {
    1
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
            api_host: "127.0.0.1".into(),
            api_port: API_PORT,
            api_token: String::new(),
            inter_stage_port: INTER_STAGE_PORT,
            discovery_port: DISCOVERY_PORT,
            model_id: "deepseek-v4-flash".into(),
            quant: String::new(),
            ctx_size: 8192,
            max_decode: default_max_decode(),
            max_vram_mb: 0,
            max_ram_mb: 0,
            lan_discovery: true,
            manual_peers: String::new(),
            heartbeat_sec: default_heartbeat(),
            prefer_coordinator: false,
            same_subnet_only: false,
            bind_nic: String::new(),
            overflow_url: String::new(),
            overflow_key: String::new(),
            overflow_wait_s: 0,
            overflow_daily_cap_milli: 0,
        }
    }
}

const BEACON_MAGIC: &str = "IDLETOKEN1";

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

/// A member that has not polled the creator's roster for this long is marked
/// offline (audit 2.8: a dead/unplugged machine used to stay green in the
/// member list forever — the protocol only knew a voluntary "leave"). The
/// effective timeout per member is `max(OFFLINE_AFTER_S, 3 × its poll
/// interval)`, so a deliberately slow heartbeat setting does not flap.
const OFFLINE_AFTER_S: u64 = 30;

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
    /// false = no roster poll from this member within its timeout (creator
    /// side; joiners adopt the flag from the roster broadcast). Not removal:
    /// the machine may come back, and a re-register under the same id — the
    /// joiner loop re-joins on its own — flips it online again.
    online: bool,
    /// When the creator last heard this member (join or roster poll). None on
    /// the creator's own entry and on joiner-side mirrors — never swept.
    #[serde(skip)]
    last_seen: Option<std::time::Instant>,
    /// The member's own roster poll interval (from its "hb" field), seconds.
    /// 0 = not reported → the default timeout applies.
    #[serde(skip)]
    hb_secs: u32,
    #[serde(skip)]
    ip: String,
    /// What this member brings to the pool: free VRAM and free RAM in bytes,
    /// as ITS OWN probe measured them (2026-08-15). Every machine already
    /// knows its own memory; sending it with the join is what lets the whole
    /// cluster answer "is this enough for the model we picked" BEFORE anyone
    /// presses Start — until now that question was only answered by the
    /// coordinator refusing after the fact. 0 = an older client that does not
    /// report it; the UI then says it cannot tell rather than guessing.
    #[serde(rename = "vramFree")]
    vram_free: u64,
    #[serde(rename = "ramFree")]
    ram_free: u64,
    /// Unified memory (Apple Silicon): VRAM and RAM are one physical pool and
    /// must be counted once, not summed (the engine's plan.c rule).
    #[serde(rename = "unifiedMemory")]
    unified_memory: bool,
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
    /// Six-character engine pairing identity. Kept separately because joiners
    /// hide the presentation code and account-mode secrets are not themselves
    /// valid `--pair-code` values.
    engine_code: String,
    self_id: String,
    self_host: String,
    self_gpu: String,
    /// This machine's own free VRAM/RAM (bytes) and whether it is unified
    /// memory, as reported by the UI's probe through `pairing_report_memory`.
    /// Sent with every join/poll so the roster can total the pool.
    self_vram_free: u64,
    self_ram_free: u64,
    self_unified: bool,
    peers: Vec<Peer>,
    coordinator_id: Option<String>,
    /// idle (roster forming) | starting (engines launching) | ready
    phase: String,
    /// ip of the machine running idletoken-coord (set at start)
    coord_ip: Option<String>,
    /// The GGUF is required on the coordinator. rpc workers do not open it;
    /// llama-server streams their assigned tensors over authenticated RPC.
    model_path: String,
    engine_started: bool,
    /// Account-mode pairing (integration plan 3.3): the "code" is a secret
    /// derived on the JS side from stable account material (platform user id +
    /// platform URL + cluster name), not a human-typed 6-char code. The wire
    /// mechanics are identical (beacon broadcasts only the FNV hash, the full
    /// secret is the TCP join proof) — this flag only changes presentation:
    /// the snapshot hides `code` (nothing to read aloud) and sets
    /// `accountMode` so the UI labels the cluster as account-formed.
    account_mode: bool,
    /// Last pairing failure, as (code, detail), surfaced in the snapshot so the
    /// UI can say WHY a join died instead of silently resetting to idle (which
    /// looked exactly like the button doing nothing). `code` is a stable
    /// identifier the front end maps to a localized sentence; `detail` carries
    /// the variable part (the discovery port, or the creator's verbatim
    /// rejection). Cleared on every new create/join/leave.
    last_error: Option<(String, String)>,
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
            engine_code: String::new(),
            self_id: String::new(),
            self_host: String::new(),
            self_gpu: String::new(),
            self_vram_free: 0,
            self_ram_free: 0,
            self_unified: false,
            peers: Vec::new(),
            coordinator_id: None,
            phase: "idle".into(),
            coord_ip: None,
            model_path: String::new(),
            engine_started: false,
            account_mode: false,
            last_error: None,
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

/// Convert the roster proof into the native engine's six-character pairing
/// alphabet. Human code mode remains byte-for-byte identical. Account mode
/// starts from a SHA-256-derived secret and maps its FNV digest to 30 bits —
/// the same online guessing bar as a normal join code, without exposing the
/// account secret to the engine command line.
fn engine_pair_code(proof: &str, account_mode: bool) -> String {
    if !account_mode {
        return proof.trim().to_uppercase();
    }
    const ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut h: u64 = 0xcbf29ce484222325;
    for b in proof.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (0..6)
        .map(|i| ALPHABET[((h >> (i * 5)) & 31) as usize] as char)
        .collect()
}

fn snapshot_json(inner: &Inner) -> Value {
    json!({
        // Account mode: the derived secret is not a shareable code — hide it.
        "code": if inner.account_mode { &None } else { &inner.code },
        "accountMode": inner.account_mode,
        "peers": inner.peers,
        "coordinatorId": inner.coordinator_id,
        "phase": inner.phase,
        // The inference API is loopback-only on the coordinator (2026-08-15,
        // coord enforces it) — so only the coordinator's own UI gets a base
        // URL. Joiner machines contribute compute; chatting happens on the
        // machine that runs the coordinator.
        "api": if inner.phase == "ready" && inner.mode == Mode::Creator {
            let api_port = inner.tuning.api_port;
            Some(json!({
                "baseUrl": format!("http://127.0.0.1:{api_port}"),
                "status": "online",
            }))
        } else { None },
        "source": "engine",
        "canStart": inner.mode == Mode::Creator && inner.phase == "idle" && inner.peers.len() >= 2,
        // Why the last join attempt failed (see Inner::last_error). null while
        // nothing has failed, or after a new attempt started.
        "lastError": inner
            .last_error
            .as_ref()
            .map(|(code, detail)| json!({ "code": code, "detail": detail })),
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
        // The creator's configured API port, so a joiner with different (or
        // default) settings still polls /idletoken/v1/cluster/status on the right port.
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
            // Liveness travels with the roster so every member's UI shows the
            // same offline states the creator sees.
            "online": p.online,
            // The layer plan travels too (2026-08-15): joiners can no longer
            // read /idletoken/v1/cluster/status themselves — the API answers
            // only the coordinator's machine — so the roster is the one place
            // their UI learns which layers each node holds.
            "layerLo": p.layer_lo, "layerHi": p.layer_hi,
            // Each member's own measurement, echoed to everyone so any machine
            // can total the pool (not just the creator that collected them).
            "vramFree": p.vram_free, "ramFree": p.ram_free,
            "unifiedMemory": p.unified_memory,
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

/// What the joiner must do after a roster reply.
#[derive(Default)]
struct RosterEffect {
    /// The cluster flipped to "starting": launch this machine's engine.
    start_engine: bool,
    /// The cluster was torn down and is forming again (the coordinator switched
    /// model, or restarted for any other reason). Our worker is loaded with the
    /// OLD weights, so it has to go before the next start can bring up the new
    /// ones.
    stop_engine: bool,
    /// We are not in the roster any more — re-send "join". Without this a
    /// coordinator restart silently strands every other machine: the joiner
    /// keeps polling with op="roster", which the new creator answers politely
    /// and ignores, so the machine never reappears in anyone's list.
    rejoin: bool,
}

/// Apply a roster reply on the joiner side.
fn apply_roster(inner: &mut Inner, v: &Value) -> RosterEffect {
    let members = v["members"].as_array().cloned().unwrap_or_default();
    let coordinator_id = v["coordinatorId"].as_str().map(String::from);
    let phase = v["phase"].as_str().unwrap_or("idle").to_string();
    let coord_ip = v["coordIp"].as_str().map(String::from);
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
                // From the creator's engine-status merge, via the roster
                // (absent on an older creator → None, same as before).
                layer_lo: m["layerLo"].as_u64().map(|x| x as u32),
                layer_hi: m["layerHi"].as_u64().map(|x| x as u32),
                // Adopted from the creator's sweep; absent (older creator)
                // reads as online, which is what the field's absence meant.
                online: m["online"].as_bool().unwrap_or(true),
                last_seen: None,
                hb_secs: 0,
                ip: m["ip"].as_str().unwrap_or("").to_string(),
                // Memory travels with the roster so every machine can total
                // the pool, not just the creator (0 = an older peer).
                vram_free: m["vramFree"].as_u64().unwrap_or(0),
                ram_free: m["ramFree"].as_u64().unwrap_or(0),
                unified_memory: m["unifiedMemory"].as_bool().unwrap_or(false),
            }
        })
        .collect();
    inner.coordinator_id = coordinator_id;
    let mut eff = RosterEffect::default();
    eff.rejoin = !inner.peers.iter().any(|p| p.is_self);
    eff.start_engine = phase == "starting" && inner.phase == "idle" && !inner.engine_started;
    // Back to "idle" after we had already started means the cluster we belong
    // to no longer exists in the form we joined. Drop our engine and re-arm, so
    // the next "starting" launches a worker for whatever model the coordinator
    // has now (its weight server is the source, so nothing local to update).
    eff.stop_engine = phase == "idle" && inner.engine_started;
    if eff.stop_engine {
        inner.engine_started = false;
    }
    // `!= "ready"` keeps a locally-ready worker from regressing to the
    // coordinator's "starting"; "idle" is the exception, because that is the
    // teardown above and pinning ready through it would freeze the UI on a
    // cluster that is gone.
    if inner.phase != "ready" || phase == "idle" {
        inner.phase = phase;
    }
    inner.coord_ip = coord_ip;
    eff
}

/// The address this machine binds cluster traffic to and advertises to the
/// others: the "Bind interface / IP" setting when it is a usable IPv4,
/// otherwise whatever the OS routes from (`local_lan_ip`).
///
/// One function for both jobs on purpose. Binding to a specific NIC while
/// still telling peers the auto-detected address is the multi-homed failure
/// mode that produces "connection refused" against a machine that is plainly
/// up — the two answers have to come from the same place.
fn self_ip(tuning: &Tuning) -> String {
    let nic = tuning.bind_nic.trim();
    if !nic.is_empty() && nic != "auto" && nic.parse::<std::net::Ipv4Addr>().is_ok() {
        return nic.to_string();
    }
    local_lan_ip()
}

/// Bind host for listeners: a chosen NIC, else every interface.
fn bind_host(tuning: &Tuning) -> String {
    let nic = tuning.bind_nic.trim();
    if !nic.is_empty() && nic != "auto" && nic.parse::<std::net::Ipv4Addr>().is_ok() {
        return nic.to_string();
    }
    "0.0.0.0".into()
}

/// Same /24? Used by "Only same subnet" on both sides of the handshake.
/// IPv6 and unparseable addresses are treated as "not the same subnet": the
/// setting is a restriction, and a restriction that silently passes whatever it
/// cannot classify is not one.
fn same_subnet(a: &str, b: &str) -> bool {
    match (a.parse::<std::net::Ipv4Addr>(), b.parse::<std::net::Ipv4Addr>()) {
        (Ok(x), Ok(y)) => x.octets()[..3] == y.octets()[..3],
        _ => false,
    }
}

/// The manual peer list, cleaned up. Accepts commas, spaces or newlines so a
/// pasted list works whatever it was copied from.
fn manual_peer_list(tuning: &Tuning) -> Vec<String> {
    tuning
        .manual_peers
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Roster poll interval. Clamped: 0 would spin, and anything past a minute
/// makes the member list look frozen.
fn heartbeat(tuning: &Tuning) -> Duration {
    Duration::from_secs(tuning.heartbeat_sec.clamp(1, 60) as u64)
}

/// "This machine's usage" as rpc-supervisor flags. The worker probes and sends
/// these capped resources in HELLO; the llama.cpp planner consumes them.
///
/// 0 = no cap, and then no flag at all: an explicit `--max-vram-mb 0` and a
/// missing flag mean the same thing to the worker, but the shorter command line
/// is the one that reads correctly in a log.
fn usage_cap_args(tuning: &Tuning) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    if tuning.max_vram_mb > 0 {
        v.push("--max-vram-mb".into());
        v.push(tuning.max_vram_mb.to_string());
    }
    if tuning.max_ram_mb > 0 {
        v.push("--max-ram-mb".into());
        v.push(tuning.max_ram_mb.to_string());
    }
    v
}

/// Overflow flags for the coordinator ("borrow another machine when this one is
/// full"). Both the URL and the key or nothing: the coordinator treats their
/// presence as the switch, so passing one without the other would ask it to
/// enable a feature it cannot use and it would refuse to start.
///
/// Deliberately NOT guarded on api_token here. The coordinator's own refusal is
/// the guard, and duplicating it would mean a client that silently drops the
/// flags instead of surfacing why -- the user would see sharing "on" in the
/// panel and a machine that never borrows.
fn overflow_args(tuning: &Tuning) -> Vec<String> {
    if tuning.overflow_url.is_empty() || tuning.overflow_key.is_empty() {
        return Vec::new();
    }
    let mut v = vec![
        "--overflow-url".into(), tuning.overflow_url.clone(),
        "--overflow-key".into(), tuning.overflow_key.clone(),
        "--overflow-wait-s".into(), tuning.overflow_wait_s.to_string(),
    ];
    // 0 means "use the coordinator's own default", which is a real ceiling --
    // never "no ceiling". Omitting the flag says the same thing more plainly.
    if tuning.overflow_daily_cap_milli > 0 {
        v.push("--overflow-daily-cap".into());
        v.push(tuning.overflow_daily_cap_milli.to_string());
    }
    v
}

/// Start this machine's engine(s) per its role in the frozen roster. The
/// coordinator's llama-server uses local compute directly; only other machines
/// run rpc-supervisors. There is deliberately no co-located RPC worker and no
/// legacy layer-shard server in this topology.
fn materialize_engine(app: &AppHandle) {
    let (is_coord, coord_ip, remote_workers, model_path, engine_code, tuning) = {
        let pairing = app.state::<Pairing>();
        let mut inner = pairing.0.lock().unwrap();
        if inner.engine_started {
            return;
        }
        inner.engine_started = true;
        let is_coord = inner.coordinator_id.as_deref() == Some(inner.self_id.as_str());
        let remote_workers = inner.peers.iter().filter(|p| p.role != "coordinator").count();
        (
            is_coord,
            inner.coord_ip.clone().unwrap_or_default(),
            remote_workers,
            inner.model_path.clone(),
            inner.engine_code.clone(),
            inner.tuning.clone(),
        )
    };
    if is_coord {
        if model_path.is_empty() {
            eprintln!("[pairing] coordinator start refused: no GGUF file selected");
            return;
        }
        let engine_bin = match crate::engine::llama_server_bin() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[pairing] coordinator start refused: {e}");
                return;
            }
        };
        let host = bind_host(&tuning);   // "Bind interface / IP", else 0.0.0.0
        let mut coord_args = vec![
            // Hardened engine, always — not only once someone presses "share
            // compute". The flags that keep a buyer's prompt unreadable on this
            // machine (engine args locked, unix-socket link, binary digest
            // checked) are all fixed when the engine process starts, and
            // sharing is turned on long after the cluster is up. Deciding it
            // here means starting to share costs no restart and needs no
            // sentence explaining why the model has to reload.
            //
            // It takes nothing away from local use: IDLETOKEN_LLAMA_ARGS is a
            // development variable that no client user sets, and anyone who
            // wants it runs the coordinator from a shell, where --shared stays
            // opt-in. See docs/shared-mode-plan-2026-08.md P0-1.
            "--shared".into(),
            "--bind".into(), format!("{host}:{COORD_PORT}"),
            "--llama-server-bin".into(), engine_bin.to_string_lossy().into_owned(),
            "--llama-gguf".into(), model_path,
            "--llama-port".into(), LLAMA_PORT.to_string(),
            "--http".into(),
            // Always loopback (2026-08-15): the coordinator rewrites anything
            // else to 127.0.0.1 anyway; upgraded installs may still store
            // "0.0.0.0" in settings, so it is normalized here too.
            "--api-bind".into(), format!("127.0.0.1:{}", tuning.api_port),
            "--ctx-size".into(), tuning.ctx_size.to_string(),
            "--max-decode".into(), tuning.max_decode.to_string(),
        ];
        if remote_workers > 0 {
            coord_args.push("--num-workers".into());
            coord_args.push(remote_workers.to_string());
            coord_args.push("--pair-code".into());
            coord_args.push(engine_code);
        }
        if !tuning.api_token.is_empty() {
            coord_args.push("--api-token".into());
            coord_args.push(tuning.api_token.clone());
        }
        coord_args.extend(overflow_args(&tuning));
        if let Err(e) = crate::engine::start_engine(app, "coordinator".into(), coord_args, Vec::new()) {
            eprintln!("[pairing] coord start failed: {e}");
        }
    } else {
        let engine_dir = match crate::engine::llama_engine_dir() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[pairing] rpc worker start refused: {e}");
                return;
            }
        };
        let mut worker_args = vec![
            "--rpc-supervisor".into(),
            "--engine-dir".into(), engine_dir.to_string_lossy().into_owned(),
            "--pair-code".into(), engine_code,
            "--coordinator".into(), format!("{coord_ip}:{COORD_PORT}"),
            "--discovery-port".into(), tuning.discovery_port.to_string(),
            "--rpc-host".into(), self_ip(&tuning),
            "--rpc-port".into(), tuning.inter_stage_port.to_string(),
        ];
        worker_args.extend(usage_cap_args(&tuning));
        if let Err(e) = crate::engine::start_engine(app, "worker".into(), worker_args, Vec::new()) {
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
/// plan into the roster. Creator only, over loopback: the API answers its own
/// machine exclusively (coord enforces it), so a joiner cannot poll it — a
/// joiner's phase/stage/layers all arrive through the roster protocol instead
/// (roster_reply carries them from the creator's merge).
fn merge_engine_status(app: &AppHandle) -> bool {
    let api_port = {
        let pairing = app.state::<Pairing>();
        let inner = pairing.0.lock().unwrap();
        match (inner.mode, inner.phase.as_str()) {
            (Mode::Creator, "starting") | (Mode::Creator, "ready") => inner.tuning.api_port,
            _ => return false,
        }
    };
    let Some(body) = http_get_body("127.0.0.1", api_port, "/idletoken/v1/cluster/status") else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&body) else {
        return false;
    };
    if v["phase"].as_str() != Some("ready") {
        return false;
    }
    if let Some(state) = v["engine_state"].as_str() {
        if state != "ready" {
            return false;
        }
    }
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
            // "Only same subnet": refuse before the code is even considered, and
            // say why. A silent drop here is indistinguishable from a firewall
            // and would send someone hunting the wrong problem.
            if inner.tuning.same_subnet_only && !same_subnet(&peer_ip, &self_ip(&inner.tuning)) {
                json!({"ok": false, "err": "different subnet (this cluster is restricted to one subnet)"})
            } else if req["code"].as_str() != inner.code.as_deref() {
                json!({"ok": false, "err": "bad code"})
            } else {
                let id = req["hostname"].as_str().unwrap_or("?").to_string();
                let hb = req["hb"].as_u64().unwrap_or(0) as u32;
                if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                    // Re-register under a known id: the machine is back (or
                    // re-joined after a creator restart). Refresh liveness and
                    // its address — a reboot may have changed the IP.
                    p.online = true;
                    p.last_seen = Some(std::time::Instant::now());
                    p.hb_secs = hb;
                    p.ip = peer_ip.clone();
                } else {
                    inner.peers.push(Peer {
                        id: id.clone(),
                        hostname: id.clone(),
                        gpu: req["gpu"].as_str().unwrap_or("").to_string(),
                        role: "worker",
                        is_self: false,
                        stage: "joined".into(),
                        layer_lo: None,
                        layer_hi: None,
                        online: true,
                        last_seen: Some(std::time::Instant::now()),
                        hb_secs: hb,
                        ip: peer_ip.clone(),
                        vram_free: req["vramFree"].as_u64().unwrap_or(0),
                        ram_free: req["ramFree"].as_u64().unwrap_or(0),
                        unified_memory: req["unifiedMemory"].as_bool().unwrap_or(false),
                    });
                }
                // The joiner asked to be the coordinator ("Prefer this machine
                // as coordinator" on ITS settings page). Same effect as the
                // creator picking it by hand in Manage cluster, so the roles
                // stay in one place — and only while the roster is still open:
                // moving the coordinator after the engines are up would point
                // half the cluster at a coord that is not running.
                if req["prefer"].as_bool() == Some(true) && inner.phase == "idle" {
                    inner.coordinator_id = Some(id.clone());
                    for p in inner.peers.iter_mut() {
                        p.role = if p.id == id { "coordinator" } else { "worker" };
                    }
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
            if !id.is_empty() {
                let forming = inner.phase == "idle"; // read before the &mut borrow below
                if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                    // The poll IS the liveness signal: hearing it revives a
                    // member the sweep had marked offline.
                    p.online = true;
                    p.last_seen = Some(std::time::Instant::now());
                    p.hb_secs = req["hb"].as_u64().unwrap_or(0) as u32;
                    if !eng.is_empty() && !forming && p.stage != "ready" {
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
    // "LAN auto-discovery" off: stay silent and let the roster service below do
    // the work — machines that were given our IP by hand still get in. Read
    // once here, not per tick: the cluster's discovery mode is decided when it
    // is created, and a beacon that stops mid-forming would be a worse setting
    // than one that never started.
    let announce = {
        let pairing = app.state::<Pairing>();
        let inner = pairing.0.lock().unwrap();
        inner.tuning.lan_discovery
    };
    // UDP beacon: `IDLETOKEN1|<fnv1a(code)>|<roster_port>` once a second.
    let beacon_app = app.clone();
    let hash = code_hash(&code);
    std::thread::spawn(move || {
        if !announce {
            return;
        }
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
            // Without the roster service the cluster can never form — tell the
            // UI instead of leaving a beacon inviting joiners to a port nobody
            // answers. Bumping the generation ends that beacon thread too.
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            if inner.generation == generation {
                inner.generation += 1;
                inner.mode = Mode::Off;
                inner.phase = "idle".into();
                inner.code = None;
                inner.peers.clear();
                inner.coordinator_id = None;
                inner.last_error = Some(("portBusy".into(), ROSTER_PORT.to_string()));
            }
            drop(inner);
            emit_snapshot(&app);
            return;
        };
        let _ = listener.set_nonblocking(true);
        loop {
            {
                let pairing = app.state::<Pairing>();
                let mut inner = pairing.0.lock().unwrap();
                if inner.generation != generation {
                    return;
                }
                // Liveness sweep: a member that stopped polling is marked
                // offline, not removed — the machine may come back, and its
                // joiner loop re-registers under the same id when it does
                // (which flips it online again above). The creator's own
                // entry has no last_seen and is never swept.
                let now = std::time::Instant::now();
                let mut changed = false;
                for p in inner.peers.iter_mut() {
                    let Some(seen) = p.last_seen else { continue };
                    let timeout = Duration::from_secs(
                        (p.hb_secs.clamp(1, 60) as u64 * 3).max(OFFLINE_AFTER_S),
                    );
                    if p.online && now.duration_since(seen) > timeout {
                        p.online = false;
                        changed = true;
                        eprintln!("[pairing] member {} went silent — marked offline", p.id);
                    }
                }
                drop(inner);
                if changed {
                    emit_snapshot(&app);
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
        // This machine's own pairing settings (the creator's are irrelevant
        // here — every one of these answers a question about THIS computer).
        let (listen, manual, prefer, subnet_only, my_ip, poll) = {
            let pairing = app.state::<Pairing>();
            let inner = pairing.0.lock().unwrap();
            (
                inner.tuning.lan_discovery,
                manual_peer_list(&inner.tuning),
                inner.tuning.prefer_coordinator,
                inner.tuning.same_subnet_only,
                self_ip(&inner.tuning),
                heartbeat(&inner.tuning),
            )
        };

        // 1) discover the creator via beacon — unless LAN auto-discovery is off,
        //    in which case we go straight to the addresses we were given.
        let hash = code_hash(&code);
        let beacon_ip: Option<String> = if !listen {
            None
        } else {
            (|| {
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
                            let ip = src.ip().to_string();
                            // "Only same subnet" also filters what we LISTEN to,
                            // not just what we accept: on a bridged VM or a VPN
                            // the beacon can arrive from a network the user
                            // deliberately excluded.
                            if subnet_only && !same_subnet(&ip, &my_ip) {
                                eprintln!("[pairing] ignoring beacon from {ip}: different subnet");
                                continue;
                            }
                            return Some(ip);
                        }
                    }
                }
                None
            })()
        };

        // 2) manual peers are the fallback the beacon cannot be: broadcast does
        //    not cross subnets and plenty of networks drop it entirely. Each
        //    candidate is tried by simply attempting the join below.
        let creator_ip: Option<String> = beacon_ip.or_else(|| {
            manual
                .iter()
                .find(|ip| {
                    if subnet_only && !same_subnet(ip, &my_ip) {
                        eprintln!("[pairing] skipping manual peer {ip}: different subnet");
                        return false;
                    }
                    let ok = format!("{ip}:{ROSTER_PORT}")
                        .parse::<SocketAddr>()
                        .ok()
                        .and_then(|a| TcpStream::connect_timeout(&a, Duration::from_secs(2)).ok())
                        .is_some();
                    if !ok {
                        eprintln!("[pairing] manual peer {ip} did not answer on {ROSTER_PORT}");
                    }
                    ok
                })
                .cloned()
        });

        let Some(creator_ip) = creator_ip else {
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            if inner.generation == generation {
                inner.mode = Mode::Off;
                inner.phase = "idle".into();
                inner.code = None;
                // Tell the UI, not just stderr: a silent reset to idle is
                // indistinguishable from the Join button doing nothing.
                inner.last_error = Some(if listen {
                    ("notFound".into(), discovery_port.to_string())
                } else {
                    ("notFoundManual".into(), String::new())
                });
            }
            drop(inner);
            emit_snapshot(&app);
            eprintln!(
                "[pairing] no cluster found for that code{}",
                if listen { " on this LAN" } else { " (LAN auto-discovery is off; add the host's IP under Manual peer IPs)" }
            );
            return;
        };

        // 3) join + poll loop over the roster protocol
        let (self_host, self_gpu) = {
            let pairing = app.state::<Pairing>();
            let inner = pairing.0.lock().unwrap();
            (inner.self_host.clone(), inner.self_gpu.clone())
        };
        let mut joined = false;
        // Creator-loss detection (the inverse of the member sweep): the polls
        // this loop already sends are the liveness probe. Silence longer than
        // max(30s, 3×interval) flips the snapshot to "creator lost" — members
        // kept but grayed, last_error = creatorLost — and the loop keeps
        // polling; the first successful reply clears it and apply_roster
        // restores the live member states from the wire.
        let mut last_ok = std::time::Instant::now();
        let mut lost = false;
        let lost_after =
            Duration::from_secs(poll.as_secs().saturating_mul(3).max(OFFLINE_AFTER_S));
        loop {
            {
                let pairing = app.state::<Pairing>();
                if pairing.0.lock().unwrap().generation != generation {
                    return;
                }
            }
            // "hb": this machine's own poll interval, so the creator can size
            // the liveness timeout instead of guessing (a slow deliberate
            // heartbeat must not read as a dead machine).
            let req = if joined {
                // report this machine's live engine state so the whole roster
                // sees per-node progress (P4)
                json!({"op": "roster", "id": self_host,
                       "engine": crate::engine::current_state_str(&app),
                       "hb": poll.as_secs()})
            } else {
                // `prefer`: this machine's own "Prefer this machine as
                // coordinator". Sent on every (re)join so it survives the
                // creator restarting the roster.
                // Memory goes with the join: the machine that owns the
                // numbers is the one that measured them, and the roster is
                // where the cluster totals them up (pre-flight "will this
                // model fit on all of us together").
                let (vram_free, ram_free, unified) = {
                    let pairing = app.state::<Pairing>();
                    let inner = pairing.0.lock().unwrap();
                    (inner.self_vram_free, inner.self_ram_free, inner.self_unified)
                };
                json!({"op": "join", "code": code, "hostname": self_host, "gpu": self_gpu,
                       "prefer": prefer, "hb": poll.as_secs(),
                       "vramFree": vram_free, "ramFree": ram_free, "unifiedMemory": unified})
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
                    let err = v["err"].as_str().unwrap_or("").to_string();
                    eprintln!("[pairing] join rejected: {err}");
                    // Surface the rejection and reset to idle, mirroring the
                    // not-found path: the roster said no, and retrying with the
                    // same request would only repeat the answer.
                    let pairing = app.state::<Pairing>();
                    let mut inner = pairing.0.lock().unwrap();
                    if inner.generation == generation {
                        inner.mode = Mode::Off;
                        inner.phase = "idle".into();
                        inner.code = None;
                        inner.last_error = Some(if err == "bad code" {
                            ("badCode".into(), String::new())
                        } else if err.starts_with("different subnet") {
                            ("subnet".into(), String::new())
                        } else {
                            ("rejected".into(), err)
                        });
                    }
                    drop(inner);
                    emit_snapshot(&app);
                    return;
                }
                joined = true;
                let eff = {
                    let pairing = app.state::<Pairing>();
                    let mut inner = pairing.0.lock().unwrap();
                    if inner.generation != generation {
                        return;
                    }
                    // Recovery: the creator answered again. Clear the lost
                    // flag (only our own error — a fresher one is not ours to
                    // erase); apply_roster below restores every member's live
                    // online state from the wire.
                    if lost && inner.last_error.as_ref().is_some_and(|(c, _)| c == "creatorLost") {
                        inner.last_error = None;
                    }
                    apply_roster(&mut inner, &v)
                };
                if lost {
                    eprintln!("[pairing] creator at {creator_ip} is back — resyncing");
                }
                lost = false;
                last_ok = std::time::Instant::now();
                emit_snapshot(&app);
                // Order matters: stop before rejoining. The engine we are
                // holding belongs to the cluster that just went away, and the
                // "join" below can be answered by a coordinator already
                // forming the next one.
                if eff.stop_engine {
                    let _ = crate::engine::stop_engine(&app);
                }
                if eff.rejoin {
                    joined = false;
                }
                if eff.start_engine {
                    materialize_engine(&app);
                }
                // after starting, also watch the engine coordinator for readiness
                merge_engine_status(&app);
            } else if !lost && last_ok.elapsed() > lost_after {
                // The creator has not answered for the whole window: say so
                // instead of polling silently forever. Members are kept as the
                // last known state, grayed — not cleared: the cluster may well
                // still exist, we just cannot see it from here.
                lost = true;
                let pairing = app.state::<Pairing>();
                let mut inner = pairing.0.lock().unwrap();
                if inner.generation != generation {
                    return;
                }
                for p in inner.peers.iter_mut() {
                    p.online = false;
                }
                inner.last_error = Some(("creatorLost".into(), String::new()));
                drop(inner);
                emit_snapshot(&app);
                eprintln!("[pairing] lost contact with the cluster creator at {creator_ip} — still retrying");
            }
            std::thread::sleep(poll);
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
            // The coordinator process is GONE and the cluster never came up
            // (2026-08-15). Without this the formation had no failure exit:
            // the roster kept answering "starting" to every joiner, so their
            // machines sat at "loading" forever waiting for a coordinator that
            // had already exited — and the creator's own card said "starting"
            // just as long. Dropping back to idle is what ends it on BOTH
            // sides: joiners read phase=idle as teardown and stop their
            // workers (apply_roster's stop_engine), and the creator gets its
            // Start button back. The refusal sentence itself is already on
            // screen — the engine card carries `refusedReason`.
            // Whichever role THIS machine started (the creator is usually the
            // coordinator, but `preferCoordinator` can put that job on another
            // machine, and then the creator runs a worker).
            let my_role = if inner.coordinator_id.as_deref() == Some(inner.self_id.as_str()) {
                "coordinator"
            } else {
                "worker"
            };
            let my_st = crate::engine::role_state_str(&app, my_role);
            if my_st == "crashed" || my_st == "stopped" {
                eprintln!(
                    "[pairing] {my_role} engine is {my_st} while forming — cluster back to idle"
                );
                inner.phase = "idle".into();
                inner.engine_started = false;
                for p in inner.peers.iter_mut() {
                    p.stage = "joined".into();
                }
                drop(inner);
                emit_snapshot(&app);
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
        inner.engine_code = engine_pair_code(&code, inner.account_mode);
        inner.self_id = hostname.clone();
        inner.self_host = hostname.clone();
        inner.self_gpu = gpu.clone();
        inner.coordinator_id = Some(hostname.clone());
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.model_path = model_path.unwrap_or_default();
        inner.engine_started = false;
        inner.last_error = None;
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
            // The creator is this process: always online, never swept.
            online: true,
            last_seen: None,
            hb_secs: 0,
            ip: String::new(),
            // Filled by pairing_report_memory once the probe has run — the
            // creator's own numbers come from the same command the joiners
            // send, so one path fills every entry.
            vram_free: 0,
            ram_free: 0,
            unified_memory: false,
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
        inner.engine_code = engine_pair_code(&code, inner.account_mode);
        inner.self_id = hostname.clone();
        inner.self_host = hostname;
        inner.self_gpu = gpu;
        inner.coordinator_id = None;
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.model_path = model_path.unwrap_or_default();
        inner.engine_started = false;
        inner.last_error = None;
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
            // The "[CODE] detail" prefix is the client-error convention (see
            // ERROR_KEYS in client/src/i18n.ts): the UI translates the code,
            // logs keep the English sentence.
            return Err("[PAIR_NOT_CREATOR] only the cluster creator can start it".into());
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
            return Err("[PAIR_NEED_TWO] need at least 2 machines".into());
        }
        generation = inner.generation;
        let coord_id = inner.coordinator_id.clone().unwrap_or_else(|| inner.self_id.clone());
        // self_ip, not local_lan_ip: on a machine with a chosen NIC the address
        // we hand out has to be the one we are listening on.
        let mine = self_ip(&inner.tuning);
        let coord_ip = inner
            .peers
            .iter()
            .find(|p| p.id == coord_id)
            .map(|p| if p.is_self || p.ip.is_empty() { mine.clone() } else { p.ip.clone() })
            .unwrap_or_else(|| mine.clone());
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
            return Err("[PAIR_NOT_CREATOR] only the cluster creator can pick the coordinator".into());
        }
        if inner.phase != "idle" {
            return Err("[PAIR_ALREADY_STARTED] cluster already started".into());
        }
        if !inner.peers.iter().any(|p| p.id == peer_id) {
            return Err("[PAIR_NO_MEMBER] no such member".into());
        }
        inner.coordinator_id = Some(peer_id.clone());
        for p in inner.peers.iter_mut() {
            p.role = if p.id == peer_id { "coordinator" } else { "worker" };
        }
    }
    emit_snapshot(&app);
    Ok(())
}

/// This machine's free memory, from the UI's own probe snapshot.
///
/// Called whenever the probe refreshes, in every mode — the numbers must be in
/// place BEFORE a join is sent, and the creator's own roster entry is filled
/// from here too, so one path feeds every member. Cheap and idempotent: it
/// only writes three integers and refreshes this machine's roster row.
#[tauri::command]
pub fn pairing_report_memory(
    app: AppHandle,
    state: State<'_, Pairing>,
    vram_free: u64,
    ram_free: u64,
    unified_memory: bool,
) -> Result<(), String> {
    let changed = {
        let mut inner = state.0.lock().unwrap();
        inner.self_vram_free = vram_free;
        inner.self_ram_free = ram_free;
        inner.self_unified = unified_memory;
        let self_id = inner.self_id.clone();
        match inner.peers.iter_mut().find(|p| p.id == self_id) {
            Some(p) if p.vram_free != vram_free || p.ram_free != ram_free => {
                p.vram_free = vram_free;
                p.ram_free = ram_free;
                p.unified_memory = unified_memory;
                true
            }
            _ => false,
        }
    };
    if changed {
        emit_snapshot(&app);
    }
    Ok(())
}

#[tauri::command]
pub fn pairing_leave(app: AppHandle, state: State<'_, Pairing>) -> Result<(), String> {
    {
        let mut inner = state.0.lock().unwrap();
        inner.generation += 1; // ends beacon/roster/poll threads
        inner.mode = Mode::Off;
        inner.code = None;
        inner.engine_code.clear();
        inner.peers.clear();
        inner.coordinator_id = None;
        inner.phase = "idle".into();
        inner.coord_ip = None;
        inner.engine_started = false;
        inner.account_mode = false;
        inner.last_error = None;
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

/// The machine's LAN address, for the front end ("connect a client" needs a
/// dialable address to show, and the bind address 0.0.0.0 is not one).
#[tauri::command]
pub fn net_lan_ip() -> String {
    local_lan_ip()
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

/// Tests for the pairing settings helpers.
///
/// These six settings were rendered for months as controls nothing read, so the
/// first thing their wiring owes anyone is evidence that the reading part is
/// right. Every helper here is pure, which is the reason the parsing lives in
/// one: the parts that need a LAN (the beacon, the roster handshake) cannot be
/// tested from here, so as much decision-making as possible was pushed into
/// functions that can be.
#[cfg(test)]
mod pairing_settings_tests {
    use super::*;

    fn tuning(f: impl FnOnce(&mut Tuning)) -> Tuning {
        let mut t = Tuning::default();
        f(&mut t);
        t
    }

    #[test]
    fn same_subnet_compares_the_first_three_octets() {
        assert!(same_subnet("192.168.1.10", "192.168.1.250"));
        assert!(!same_subnet("192.168.1.10", "192.168.2.10"));
        // Tailscale-style CGNAT peers are a different subnet, which is exactly
        // why "Only same subnet" defaults to off.
        assert!(!same_subnet("100.101.1.2", "192.168.1.10"));
    }

    #[test]
    fn same_subnet_refuses_what_it_cannot_classify() {
        // A restriction that passes everything it fails to parse is not one.
        assert!(!same_subnet("fe80::1", "fe80::2"));
        assert!(!same_subnet("", "192.168.1.10"));
        assert!(!same_subnet("not-an-ip", "192.168.1.10"));
    }

    #[test]
    fn manual_peers_accept_commas_spaces_and_newlines() {
        let t = tuning(|t| t.manual_peers = " 192.168.1.50, 192.168.1.51\n192.168.1.52 ".into());
        assert_eq!(
            manual_peer_list(&t),
            vec!["192.168.1.50", "192.168.1.51", "192.168.1.52"]
        );
        assert!(manual_peer_list(&tuning(|t| t.manual_peers = " , ,\n".into())).is_empty());
        assert!(manual_peer_list(&Tuning::default()).is_empty());
    }

    #[test]
    fn heartbeat_is_clamped_to_a_sane_range() {
        // 0 would spin the roster loop; an hour would look like a frozen UI.
        assert_eq!(heartbeat(&tuning(|t| t.heartbeat_sec = 0)), Duration::from_secs(1));
        assert_eq!(heartbeat(&tuning(|t| t.heartbeat_sec = 5)), Duration::from_secs(5));
        assert_eq!(heartbeat(&tuning(|t| t.heartbeat_sec = 9999)), Duration::from_secs(60));
        // The default must equal what the loop slept before the setting existed.
        assert_eq!(heartbeat(&Tuning::default()), Duration::from_secs(1));
    }

    #[test]
    fn bind_nic_governs_both_the_bind_and_the_advertised_address() {
        let t = tuning(|t| t.bind_nic = "10.0.0.7".into());
        assert_eq!(bind_host(&t), "10.0.0.7");
        assert_eq!(self_ip(&t), "10.0.0.7", "peers must be told the address we listen on");
    }

    #[test]
    fn bind_nic_falls_back_when_it_is_not_a_usable_address() {
        for v in ["", "auto", " AUTO-ish ", "eth0", "999.1.1.1"] {
            let t = tuning(|t| t.bind_nic = v.into());
            assert_eq!(bind_host(&t), "0.0.0.0", "{v:?} must not become a bind host");
            assert_ne!(self_ip(&t), v.trim(), "{v:?} must not be advertised verbatim");
        }
    }

    #[test]
    fn usage_caps_become_flags_only_when_set() {
        assert!(usage_cap_args(&Tuning::default()).is_empty());
        let t = tuning(|t| {
            t.max_vram_mb = 8192;
            t.max_ram_mb = 16384;
        });
        assert_eq!(
            usage_cap_args(&t),
            vec!["--max-vram-mb", "8192", "--max-ram-mb", "16384"]
        );
        let only_vram = tuning(|t| t.max_vram_mb = 4096);
        assert_eq!(usage_cap_args(&only_vram), vec!["--max-vram-mb", "4096"]);
    }

    /// Overflow travels as a pair. One half without the other would ask the
    /// coordinator to enable a feature it cannot use, and it answers that by
    /// refusing to start -- so the client must not send half.
    #[test]
    fn overflow_flags_need_both_url_and_key() {
        assert!(overflow_args(&Tuning::default()).is_empty());
        assert!(overflow_args(&tuning(|t| t.overflow_url = "http://p".into())).is_empty());
        assert!(overflow_args(&tuning(|t| t.overflow_key = "sk".into())).is_empty());
        let t = tuning(|t| {
            t.overflow_url = "http://p".into();
            t.overflow_key = "sk".into();
            t.overflow_wait_s = 5;
            t.overflow_daily_cap_milli = 2500;
        });
        assert_eq!(
            overflow_args(&t),
            vec!["--overflow-url", "http://p", "--overflow-key", "sk",
                 "--overflow-wait-s", "5", "--overflow-daily-cap", "2500"]
        );
        // A cap of 0 means "the coordinator's own default", which is a real
        // ceiling. It must never be sent as an explicit 0, which would read as
        // a cap of zero -- and it must never be read as "no ceiling".
        let no_cap = tuning(|t| {
            t.overflow_url = "http://p".into();
            t.overflow_key = "sk".into();
        });
        assert!(!overflow_args(&no_cap).contains(&"--overflow-daily-cap".to_string()));
    }

    /// The client sends camelCase; a rename on either side must fail loudly
    /// here rather than silently deserialize to a default (which is how a
    /// setting goes back to doing nothing).
    #[test]
    fn tuning_deserializes_the_clients_field_names() {
        let t: Tuning = serde_json::from_value(serde_json::json!({
            "apiHost": "0.0.0.0", "apiPort": 8000, "apiToken": "",
            "interStagePort": 14101, "discoveryPort": 14099,
            "modelId": "qwen3-8b", "quant": "Q4_K_M", "ctxSize": 32768, "maxDecode": 0,
            "maxVramMb": 8192, "maxRamMb": 16384,
            "lanDiscovery": false, "manualPeers": "192.168.1.50",
            "heartbeatSec": 3, "preferCoordinator": true,
            "sameSubnetOnly": true, "bindNic": "192.168.1.9",
            "overflowUrl": "http://platform", "overflowKey": "sk-x",
            "overflowWaitS": 7, "overflowDailyCapMilli": 4200,
        }))
        .expect("client payload must deserialize");
        assert_eq!(t.overflow_url, "http://platform");
        assert_eq!(t.overflow_wait_s, 7);
        assert_eq!(t.overflow_daily_cap_milli, 4200);
        assert_eq!(t.max_vram_mb, 8192);
        assert!(!t.lan_discovery);
        assert_eq!(manual_peer_list(&t), vec!["192.168.1.50"]);
        assert_eq!(heartbeat(&t), Duration::from_secs(3));
        assert!(t.prefer_coordinator);
        assert!(t.same_subnet_only);
        assert_eq!(bind_host(&t), "192.168.1.9");
    }

    /// An older/hand-written payload without the new keys must behave exactly
    /// as the client did before they existed.
    #[test]
    fn missing_new_fields_keep_the_old_behaviour() {
        let t: Tuning = serde_json::from_value(serde_json::json!({ "apiPort": 8000 })).unwrap();
        assert!(t.lan_discovery, "the beacon must still run");
        assert!(!t.same_subnet_only, "no restriction anyone did not ask for");
        assert_eq!(heartbeat(&t), Duration::from_secs(1));
        assert!(usage_cap_args(&t).is_empty());
    }
}
