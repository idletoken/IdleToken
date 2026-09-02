// LAN pairing (acceptance P3) — the client-side layer that forms a cluster
// roster before any engine process exists, then materializes it as a real
// engine cluster (philosophy 17: engine keeps its simple contract, the client
// owns the UX-level orchestration).
//
// Flow:
//   create  → mint code (JS side), broadcast a UDP beacon carrying a random
//             session id + a per-packet nonce + our roster TCP port — nothing
//             derived from the code (see BEACON_MAGIC) — and serve a tiny
//             newline-JSON roster protocol over TCP.
//   join    → collect the machines whose beacons we hear (or the manual peer
//             list), then offer the typed code to each over TCP until one
//             accepts. The reply carries a per-member token that every later
//             roster/leave request must present.
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
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

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
    /// Coord API access token (settings.apiToken) → the coordinator's
    /// `IDLETOKEN_API_TOKEN` environment variable (A-P0-4: not `--api-token`,
    /// because argv is world-readable). Empty = no auth. Never broadcast over
    /// the roster.
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
    /// Exact product context window (256K or explicit 1M) passed to coord
    /// `--ctx-size` for the runtime VRAM admission check.
    ctx_size: u32,
    /// KV cache dtypes (settings "KV cache precision") → the coordinator's
    /// IDLETOKEN_KV_CACHE_TYPE / _V environment variables. Empty = engine
    /// default (f16). Env, not argv, to match the coordinator's existing knob;
    /// the value set is validated there against the engine's own allow-list.
    #[serde(default)]
    kv_cache_k: String,
    #[serde(default)]
    kv_cache_v: String,
    /// Per-request generation ceiling (settings.maxTokens) → coord
    /// `--max-decode`. Configuration, not a compiled-in constant: 4096 used to
    /// be hardcoded in the engine, so "Max tokens per reply" could not raise it.
    /// 0 = bounded only by the remaining context.
    #[serde(default = "default_max_decode")]
    max_decode: u32,
    /// How much of THIS machine IdleToken may use (settings "This machine's
    /// usage") → worker `--max-vram-mb`, MiB, 0 = no cap.
    ///
    /// Per-machine, so it is never adopted from the roster the way model_id and
    /// quant are: the creator's "conservative" says nothing about how much of
    /// YOUR computer you are lending. Until 2026-08-13 these never left the
    /// client — the probe was told, the serving worker was not, so the setting
    /// changed the dashboard's numbers and nothing else.
    #[serde(default)]
    max_vram_mb: u64,
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

/// The only launch settings that may change after a roster has formed.
///
/// Model, network and capacity fields are frozen when pairing starts because
/// peers have already agreed on them. Overflow is different: the creator is
/// offered its switch beside Start, so the coordinator must refresh these
/// fields immediately before materializing the engine. Keeping this as a
/// narrow type prevents a late UI action from silently changing the roster's
/// model, ports or resource contract.
#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OverflowTuning {
    enabled: bool,
    overflow_url: String,
    overflow_key: String,
    overflow_wait_s: u32,
    overflow_daily_cap_milli: u64,
}

fn apply_overflow_tuning(tuning: &mut Tuning, latest: OverflowTuning) {
    if latest.enabled {
        tuning.overflow_url = latest.overflow_url;
        tuning.overflow_key = latest.overflow_key;
        tuning.overflow_wait_s = latest.overflow_wait_s;
        tuning.overflow_daily_cap_milli = latest.overflow_daily_cap_milli;
    } else {
        // The boolean is authoritative. Never let a stale key or URL turn an
        // off switch into a partially configured forwarding path.
        tuning.overflow_url.clear();
        tuning.overflow_key.clear();
        tuning.overflow_wait_s = 0;
        tuning.overflow_daily_cap_milli = 0;
    }
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
            ctx_size: 262144,
            kv_cache_k: String::new(),
            kv_cache_v: String::new(),
            max_decode: default_max_decode(),
            max_vram_mb: 0,
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

/// Discovery beacon wire version.
///
/// `IDLETOKEN1` broadcast `fnv1a(code)` once a second. A join code is six
/// characters of a 32-symbol alphabet — 30 bits — and FNV-1a is not a password
/// hash, so anyone who could hear the beacon could recover the code offline in
/// seconds and then join as a worker, which hands them model layers, hidden
/// states and the cluster TLS PSK (2026-08-20 audit A-P0-3). The old comment
/// here called that hash "protection"; it was not.
///
/// `IDLETOKEN2` carries **nothing derived from the code**: a random session id
/// (so a machine's own beacons can be told apart from another cluster's) and a
/// fresh random nonce per packet, which is the whole point — there is no
/// function of the code on the wire to invert. Discovery and proof are
/// separate now: the beacon only says "a cluster answers at this address", and
/// the code is proved to that address over the unicast TCP join.
const BEACON_MAGIC: &str = "IDLETOKEN2";
/// Still accepted when *listening*: an older creator's beacon is a perfectly
/// good "someone is forming a cluster here" signal, and the join below is what
/// decides whether it is OUR cluster. Its hash field is ignored.
const BEACON_MAGIC_LEGACY: &str = "IDLETOKEN1";

/// Cryptographically random hex, from the OS. Used for the roster's per-member
/// tokens and the beacon's session id/nonce.
///
/// Panics rather than falling back to anything cheaper: a predictable token is
/// not a token, and "no silent fallback" (CLAUDE.md hard constraint 11) applies
/// hardest where the value's only job is to be unguessable.
fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).expect("the operating system's CSPRNG is unavailable");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time-ish comparison for the roster tokens. Both values are hex of
/// the same length, so this only has to avoid the early return of `==`.
fn token_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Proof that the sender knows the join code, bound to one nonce and one
/// direction ("creator" or "joiner").
///
/// Why this exists at all: the beacon no longer identifies which cluster it
/// belongs to (A-P0-3), so a joiner has to try every machine it heard from.
/// Sending the raw code to each of them would hand it to anything on the LAN
/// willing to broadcast a beacon — trading an offline break for an online one.
/// So the CREATOR proves the code first, against a nonce the joiner chose, and
/// only a creator that passes ever sees the joiner's own proof.
///
/// SHA-256 of a domain-separated concatenation rather than HMAC: what HMAC buys
/// over `H(key || msg)` is length-extension resistance, and there is nothing to
/// extend here — both fields are fixed-format, the digest is compared whole and
/// never used as a prefix. Using it avoids adding a crate to a bundle whose
/// size is a hard constraint. The `v1` tag is there so a future change of
/// scheme cannot be replayed as this one.
/// Iterations folded into the proof (see `pair_proof`).
///
/// The number is chosen from the ATTACKER's side of the trade, not ours. One
/// `hello` answer is a verifiable sample of the code, and the code is ~30 bits,
/// so a single plain SHA-256 lets the whole space be walked in seconds on a
/// GPU. Stretching multiplies that by the iteration count: at 600k a full sweep
/// costs ~2^49 compressions instead of ~2^30. It is not "safe" — 30 bits never
/// is against an offline attack — but it moves the cost from "seconds" to
/// "days per cluster", which is the difference between a drive-by and a
/// deliberate campaign.
///
/// The price we pay is one ~70 ms computation per handshake, once per join
/// (measured, release build, Mac control machine 2026-08-30: 70 ms median of 3;
/// the same loop in a debug build is ~3.6 s, which is why the unit test states
/// its ceiling per profile). The price an attacker pays is the same 70 ms per
/// GUESS. That asymmetry is the
/// entire mechanism, and it is also why the roster port has a request-rate gate
/// (`GATE_REQ_MAX`): without one, this constant would be a CPU-exhaustion lever
/// pointed at the creator.
const PAIR_PROOF_ROUNDS: u32 = 600_000;

/// Proof that the sender knows the join code, bound to one nonce and one
/// direction ("creator" or "joiner").
///
/// v2 (2026-08-30) — why the iteration exists. The v1 proof was a single
/// SHA-256 over a fixed format, and the creator answers `hello` from ANY
/// unauthenticated caller on the LAN. One request therefore yielded
/// `H("idletoken-pair-v1|creator|" + code + "|" + attacker_chosen_nonce)`: a
/// perfect offline oracle for a 30-bit secret. The online-guessing throttle
/// cannot touch that, because after the first reply the attacker never talks to
/// us again. This was strictly worse than the beacon hash that was removed in
/// A-P0-3 — the leak had moved rather than closed.
///
/// The domain tag changes with the scheme so a v1 transcript can never be
/// replayed as a v2 one, and the version travels in the handshake so a peer
/// that cannot do v2 is told to update instead of silently falling back to the
/// breakable proof.
fn pair_proof(role: &str, code: &str, nonce: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut acc = {
        let mut h = Sha256::new();
        h.update(b"idletoken-pair-v2|");
        h.update(role.as_bytes());
        h.update(b"|");
        h.update(code.as_bytes());
        h.update(b"|");
        h.update(nonce.as_bytes());
        h.finalize()
    };
    // Chained, so the work cannot be parallelised within one guess; the
    // counter keeps a fixed point from collapsing the chain into a short cycle.
    for i in 0..PAIR_PROOF_ROUNDS {
        let mut h = Sha256::new();
        h.update(acc);
        h.update(i.to_le_bytes());
        acc = h.finalize();
    }
    acc.iter().map(|b| format!("{b:02x}")).collect()
}

/// Handshake version this build speaks. Sent in `hello` and echoed in the
/// reply; a missing value on either side means the peer predates the stretched
/// proof and is refused with an explicit "update that machine" rather than
/// being handed the v1 oracle.
const PAIR_PROTO_V: u64 = 2;

/// This machine's stable device identity (CLUS-06, CLUS-20).
///
/// The roster used to key members by hostname, which is neither unique nor
/// secret: anyone holding the join code could register as "DESKTOP-PC" and take
/// over the real DESKTOP-PC's row — its address, its token, its resource
/// numbers. Hostname is a LABEL; this is the IDENTITY.
///
/// Persisted, because a device id that changed on restart would defeat its own
/// purpose: every reboot would look like a new machine and the roster would
/// fill with ghosts of the same computer (which is the CLUS-20 pollution this
/// is also meant to prevent). Not a secret and not a credential — it only has
/// to be stable and collision-free, so a read by another local program buys
/// nothing that the member token does not already gate.
fn device_id() -> String {
    let dir = dirs_home().join(".idletoken");
    let path = dir.join("device-id");
    if let Ok(s) = std::fs::read_to_string(&path) {
        let s = s.trim().to_string();
        if s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()) {
            return s;
        }
    }
    let fresh = random_hex(16);
    let _ = std::fs::create_dir_all(&dir);
    if std::fs::write(&path, &fresh).is_err() {
        // Non-fatal and deliberately loud: an unwritable config dir means the
        // id is fresh every launch, so rejoins look like new machines. That is
        // a degraded roster, not an unsafe one — impersonation still fails,
        // because the incumbent's id will not match either.
        eprintln!(
            "[pairing] could not persist the device id at {} — this machine will \
             look like a new one to the cluster after every restart",
            path.display()
        );
    }
    fresh
}

/// A short, non-secret label for the creator to compare across role requests.
///
/// The raw device id stays creator-side bookkeeping.  Only this digest prefix
/// enters the creator's own webview snapshot; it is never included in
/// `roster_reply`, so ordinary members do not receive a stable identifier for
/// every other machine (HOST-15).  This is continuity evidence inside one
/// roster, not remote attestation: a modified client can choose its own device
/// id and this label says nothing about its binary or hardware.
fn device_identity_label(device: &str) -> Option<String> {
    use sha2::{Digest, Sha256};
    if device.is_empty() {
        return None;
    }
    let digest = Sha256::digest(device.as_bytes());
    let groups = digest[..16]
        .chunks(4)
        .map(|chunk| chunk.iter().map(|b| format!("{b:02x}")).collect::<String>())
        .collect::<Vec<_>>();
    Some(groups.join("-"))
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Accept a peer-chosen string only if it is short, printable ASCII and free of
/// the bytes that break the JSON and log lines it gets spliced into (CLUS-14).
///
/// The Rust mirror of `idletoken_peer_label_ok` in the engine. Same reasoning,
/// same alphabet: this side stores these values in the roster and echoes them
/// to every other member, so a name that can close a JSON string here rewrites
/// the document every member's UI parses.
fn peer_field_ok(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_PEER_FIELD
        && s.bytes()
            .all(|b| (0x20..0x7f).contains(&b) && b != b'"' && b != b'\\')
}

/// The same gate, applied to an optional field: absent is fine (older client),
/// present-but-hostile is not.
fn peer_field_opt_ok(v: Option<&str>) -> bool {
    v.map_or(true, peer_field_ok)
}

/// How long a nonce the creator handed out stays usable, and how many are kept.
/// Both are small on purpose: the window only has to cover one round trip on a
/// LAN, and an unbounded list is a memory leak anyone on the network can drive.
const CHALLENGE_TTL: Duration = Duration::from_secs(60);
const MAX_CHALLENGES: usize = 128;

/// How many outstanding nonces ONE source may hold (CLUS-05).
///
/// The global cap alone is not a fair-share rule: with a single list and
/// oldest-out eviction, a machine spraying `hello` pushes every honest joiner's
/// nonce out before it can be used, and the join that follows fails with "bad
/// code" for a reason that has nothing to do with the code. A per-source quota
/// means a sprayer can only ever evict *its own* nonces.
const MAX_CHALLENGES_PER_SOURCE: usize = 4;

/// Hard ceiling on one roster request (CLUS-05, CLUS-14).
///
/// `BufRead::read_line` grows a String until it meets a newline. Nothing on
/// this socket is authenticated before it is parsed, so without a ceiling any
/// device on the LAN can open TCP 14098 and stream bytes that never contain
/// `\n` until the client is out of memory — no join code required. A roster
/// request is a small flat JSON object; 8 KiB is far more than the largest one
/// this protocol can legitimately produce.
const MAX_ROSTER_REQUEST_BYTES: u64 = 8 * 1024;

/// Longest peer-chosen string this side will store or echo (CLUS-14).
/// Mirrors the engine's 64-byte identity fields so the two faces of the same
/// cluster agree on what a name may be.
const MAX_PEER_FIELD: usize = 64;

// ---- LAN admission budgets (CLUS-02, CLUS-05) -----------------------------
//
// The engine's join port already prices guessing per source and cluster-wide
// (discovery.c). The CLIENT's roster port had none of that: `hello`/`join` were
// answered as fast as a machine could ask. The numbers below are deliberately
// the same as the engine's, because they are the same promise to the same user
// about the same 30-bit code, and two faces of one cluster disagreeing about
// how many typos are free is a bug report waiting to happen.
const GATE_FREE_TRIES: u32 = 5;
const GATE_BASE: Duration = Duration::from_secs(1);
const GATE_MAX: Duration = Duration::from_secs(30);
const GATE_FORGET: Duration = Duration::from_secs(600);
const GATE_GLOBAL_WINDOW: Duration = Duration::from_secs(60);
const GATE_GLOBAL_MAX_FAILS: u32 = 120;
const GATE_GLOBAL_BLOCK: Duration = Duration::from_secs(5);
/// Request-rate ceiling per source, independent of whether requests succeed.
/// Failure backoff prices *wrong* answers; this prices *volume*, which is what
/// a `hello` flood is — every one of those is a correct, answerable request.
const GATE_REQ_WINDOW: Duration = Duration::from_secs(10);
const GATE_REQ_MAX: u32 = 40;
const GATE_REQ_BLOCK: Duration = Duration::from_secs(5);
/// Bounded source table. Sized like the engine's for the same reason: an
/// attacker with more addresses than slots must not be able to evict penalties
/// simply by rotating through them.
const GATE_SOURCES: usize = 256;

/// One source address's admission record.
struct SourceRecord {
    ip: String,
    fails: u32,
    last: Instant,
    blocked_until: Option<Instant>,
    req_window: Instant,
    reqs: u32,
}

/// Per-source and cluster-wide admission state for the roster port.
///
/// Every method takes `now` rather than reading the clock itself: the backoff
/// curve IS the security property, and a property that can only be tested by
/// sleeping is a property that ends up untested.
#[derive(Default)]
struct LanGate {
    sources: Vec<SourceRecord>,
    global_window: Option<Instant>,
    global_fails: u32,
    global_blocked_until: Option<Instant>,
}

/// How long a source waits after `fails` consecutive failures. Free allowance
/// first (typos cost nothing), then doubling to a ceiling — never a permanent
/// lockout, which would turn a mistyped code into a support call.
fn gate_backoff(fails: u32) -> Option<Duration> {
    if fails <= GATE_FREE_TRIES {
        return None;
    }
    let steps = fails - GATE_FREE_TRIES - 1;
    let mut ms = GATE_BASE;
    for _ in 0..steps.min(16) {
        if ms >= GATE_MAX {
            break;
        }
        ms *= 2;
    }
    Some(ms.min(GATE_MAX))
}

impl LanGate {
    /// Find or claim this source's slot. Eviction prefers a free slot, then the
    /// least recently seen slot that is NOT currently serving a penalty. Plain
    /// LRU would let an attacker's next address erase the penalty the previous
    /// one had just earned — the table would be busiest exactly when it stopped
    /// working.
    fn slot(&mut self, ip: &str, now: Instant) -> &mut SourceRecord {
        if let Some(i) = self.sources.iter().position(|s| s.ip == ip) {
            if now.duration_since(self.sources[i].last) > GATE_FORGET {
                self.sources[i].fails = 0;
                self.sources[i].blocked_until = None;
            }
            return &mut self.sources[i];
        }
        if self.sources.len() < GATE_SOURCES {
            self.sources.push(SourceRecord {
                ip: ip.to_string(),
                fails: 0,
                last: now,
                blocked_until: None,
                req_window: now,
                reqs: 0,
            });
            let i = self.sources.len() - 1;
            return &mut self.sources[i];
        }
        let victim = self
            .sources
            .iter()
            .enumerate()
            .filter(|(_, s)| s.blocked_until.map_or(true, |b| b <= now))
            .min_by_key(|(_, s)| s.last)
            .map(|(i, _)| i)
            .or_else(|| {
                self.sources
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.last)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        self.sources[victim] = SourceRecord {
            ip: ip.to_string(),
            fails: 0,
            last: now,
            blocked_until: None,
            req_window: now,
            reqs: 0,
        };
        &mut self.sources[victim]
    }

    /// May this source be served right now? `Err` carries how long it must
    /// wait. Touching `last` on every check (not only on failures) is what
    /// stops a source from aging out its own penalty by hammering.
    fn admit(&mut self, ip: &str, now: Instant) -> Result<(), Duration> {
        if let Some(until) = self.global_blocked_until {
            if until > now {
                return Err(until - now);
            }
        }
        if self
            .global_window
            .map_or(true, |w| now.duration_since(w) > GATE_GLOBAL_WINDOW)
        {
            self.global_window = Some(now);
            self.global_fails = 0;
        }
        let s = self.slot(ip, now);
        s.last = now;
        if let Some(until) = s.blocked_until {
            if until > now {
                return Err(until - now);
            }
            // The block has expired, so the count that tripped it has to go
            // with it. Without this the counter is still above the ceiling when
            // the source comes back, the next request re-blocks it, and it is
            // told "5 s" again — forever, as long as it keeps asking on time.
            // A rate cap that a well-behaved client can never come back from is
            // a ban, and an honest member polling its own roster is exactly the
            // client that would hit it.
            //
            // `fails` is deliberately NOT cleared here. `blocked_until` carries
            // both the request-rate block and the online-guessing backoff, and
            // only the rate half is a fresh-start-after-serving-it rule. The
            // guessing curve is a property of how many wrong codes this source
            // has offered, so it keeps counting across expiries and the next
            // gate_backoff() still returns the escalated wait.
            s.blocked_until = None;
            s.req_window = now;
            s.reqs = 0;
        }
        if now.duration_since(s.req_window) > GATE_REQ_WINDOW {
            s.req_window = now;
            s.reqs = 0;
        }
        s.reqs += 1;
        if s.reqs > GATE_REQ_MAX {
            s.blocked_until = Some(now + GATE_REQ_BLOCK);
            let ip = s.ip.clone();
            eprintln!(
                "[pairing] {ip} is sending roster requests faster than any real \
                 client does — refusing it for {}s",
                GATE_REQ_BLOCK.as_secs()
            );
            return Err(GATE_REQ_BLOCK);
        }
        Ok(())
    }

    /// A failed proof. Counts against this source AND against the cluster-wide
    /// meter, which is what prices the same attacker spread over many addresses
    /// (each of those otherwise buys its own free allowance).
    fn note_failure(&mut self, ip: &str, now: Instant) {
        self.global_fails += 1;
        if self.global_fails >= GATE_GLOBAL_MAX_FAILS {
            self.global_blocked_until = Some(now + GATE_GLOBAL_BLOCK);
            eprintln!(
                "[pairing] {} failed joins cluster-wide inside {}s — that is a machine \
                 walking the join-code space, not typing mistakes. Refusing new joins \
                 for {}s at a time. If you did not expect this, form the cluster again \
                 with a fresh code.",
                self.global_fails,
                GATE_GLOBAL_WINDOW.as_secs(),
                GATE_GLOBAL_BLOCK.as_secs()
            );
        }
        let s = self.slot(ip, now);
        s.last = now;
        s.fails = s.fails.saturating_add(1);
        if let Some(back) = gate_backoff(s.fails) {
            s.blocked_until = Some(now + back);
            let (ip, fails) = (s.ip.clone(), s.fails);
            eprintln!(
                "[pairing] refused join #{fails} from {ip}; ignoring it for {}s \
                 (online-guessing throttle)",
                back.as_secs()
            );
        }
    }

    /// A proof that checked out. Clears this source's record only — a success
    /// says nothing about the hundred failures that came from elsewhere, so the
    /// cluster-wide meter deliberately survives it.
    fn note_success(&mut self, ip: &str, now: Instant) {
        let s = self.slot(ip, now);
        s.last = now;
        s.fails = 0;
        s.blocked_until = None;
    }
}

/// A nonce this creator handed out, and who asked for it.
struct Challenge {
    nonce: String,
    at: Instant,
    src: String,
}

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
    /// Proof of membership, minted by the creator when this machine's `join`
    /// was accepted and handed back in that reply (2026-08-20 audit A-P0-2).
    ///
    /// Every later `roster`/`leave` request must present it, or the creator
    /// drops the request: until this existed, only `join` checked the code, so
    /// anything on the LAN that could open TCP 14098 could read the whole
    /// roster — hostnames, GPUs, LAN addresses, free VRAM, the model, the API
    /// port — and could evict members by sending `leave`. The join code is
    /// deliberately NOT reused for this: it is 30 bits and it is a *shared*
    /// secret, whereas this is per member, 128 bits, and dies with the
    /// cluster (the peer list is cleared on every generation bump).
    ///
    /// Never serialized: it must not travel in the roster broadcast or in the
    /// snapshot the webview sees.
    #[serde(skip)]
    token: String,
    /// Stable per-install identity behind `id` (CLUS-06). `id`/`hostname` are a
    /// LABEL the user chose and two machines can share; this is what actually
    /// decides "is the machine sending this the same one that joined". Empty
    /// for a member running a client from before device ids existed, which is
    /// its own identity value — see `join_identity_ok`.
    ///
    /// Never serialized: it is roster bookkeeping, and broadcasting a stable
    /// per-machine identifier to every member is exactly the profiling material
    /// HOST-15 says to keep to a minimum.
    #[serde(skip)]
    device_id: String,
    /// This member asked to be the coordinator (its own "Prefer this machine as
    /// coordinator" setting). A REQUEST, never an instruction (CLUS-08): the
    /// coordinator is the cluster's plaintext window, so moving it is the
    /// creator's decision and nobody else's.
    #[serde(rename = "wantsCoordinator")]
    wants_coordinator: bool,
    /// What this member brings to the pool: scheduler-usable VRAM and RAM after
    /// OS/engine reserves, in bytes, as ITS OWN probe measured them. Every machine already
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
    /// True only when this machine has the creator-selected model + precision
    /// as a complete, integrity-checked local GGUF. The path never leaves the
    /// machine; the boolean is roster state and gates Start.
    #[serde(rename = "modelReady")]
    model_ready: bool,
    /// The identity behind model_ready. Kept creator-side so a machine cannot
    /// report "ready" for Q4 while the cluster is about to serve Q2.
    #[serde(skip)]
    model_id: String,
    #[serde(skip)]
    quant: String,
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
    /// This machine's own scheduler-usable VRAM/RAM (bytes) and whether it is unified
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
    /// Every compute node keeps the complete selected GGUF on its own disk.
    /// Each RPC worker copies only its assigned tensors into its persistent
    /// cache and loads only that slice; inference RPC then carries graphs and
    /// activations, not model payloads.
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
    /// The (model id, quant) a cluster demanded when it refused this machine
    /// for not having the weights (2026-09-01). A joiner is refused BEFORE it
    /// is in the roster, so the roster — which used to be how it learned what
    /// to download — is no longer available to tell it. The refusal carries the
    /// identity instead, and it is kept here so the UI can name the exact model
    /// and offer to fetch it. Cleared on every new create/join/leave.
    required_model: Option<(String, String)>,
    /// Nonces this creator issued in `hello` replies and has not seen used yet
    /// (see pair_proof). Consumed by the matching `join`, swept by age, and
    /// quota'd per source so one sprayer cannot evict everyone else's.
    challenges: Vec<Challenge>,
    /// This machine's stable device identity, sent with every join/poll.
    self_device_id: String,
    /// Per-source and cluster-wide admission budgets for the roster port.
    gate: LanGate,
    /// Membership changes, newest last, bounded. Not decoration: a roster that
    /// silently absorbs a rejoin under a different device, or flaps a member in
    /// and out, is one where CLUS-06 and CLUS-20 leave no trace at all. Kept in
    /// memory only — it describes the current cluster, and writing home LAN
    /// topology to disk is the profiling material HOST-15 warns about.
    audit: Vec<String>,
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
            required_model: None,
            challenges: Vec::new(),
            self_device_id: String::new(),
            gate: LanGate::default(),
            audit: Vec::new(),
            generation: 0,
            tuning: Tuning::default(),
        }))
    }
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
    // Peer::device_id is deliberately #[serde(skip)].  The creator needs a
    // stable, reviewable identity when a peer asks for the plaintext-bearing
    // coordinator role, but joiners must not receive a durable home-network
    // fingerprint.  Enrich only the creator's local snapshot, and only with a
    // short digest label rather than the raw id.
    let peers: Vec<Value> = inner
        .peers
        .iter()
        .map(|peer| {
            let mut value = serde_json::to_value(peer).unwrap_or_else(|_| json!({}));
            if inner.mode == Mode::Creator {
                if let Some(label) = device_identity_label(&peer.device_id) {
                    value["deviceIdentity"] = json!(label);
                }
            } else if let Some(object) = value.as_object_mut() {
                // A role request is addressed to the creator, not roster
                // gossip for every member.  Hide it at the native snapshot
                // boundary as well as in the UI so a joiner's renderer never
                // receives another machine's request state.
                object.remove("wantsCoordinator");
            }
            value
        })
        .collect();
    json!({
        // Account mode: the derived secret is not a shareable code — hide it.
        "code": if inner.account_mode { &None } else { &inner.code },
        "accountMode": inner.account_mode,
        "peers": peers,
        // Fail closed on old snapshot shapes: the TypeScript side treats an
        // absent value as false and never shows a role-approval action.
        "isCreator": inner.mode == Mode::Creator,
        "coordinatorId": inner.coordinator_id,
        "phase": inner.phase,
        "modelId": inner.tuning.model_id,
        "quant": inner.tuning.quant,
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
        "canStart": inner.mode == Mode::Creator
            && inner.phase == "idle"
            && inner.peers.len() >= 2
            && inner.peers.iter().all(|p| p.online && p.model_ready),
        // Why the last join attempt failed (see Inner::last_error). null while
        // nothing has failed, or after a new attempt started.
        "lastError": inner
            .last_error
            .as_ref()
            .map(|(code, detail)| json!({ "code": code, "detail": detail })),
        // The model this cluster demanded when it refused us for not having the
        // weights (paired with lastError.code == "modelNotReady"). The UI needs
        // the two fields apart, not as prose: it turns them into the exact
        // download the user is asked to confirm. null when nothing was refused.
        "requiredModel": inner
            .required_model
            .as_ref()
            .map(|(id, quant)| json!({ "modelId": id, "quant": quant })),
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
        // Per-member LAN addresses are deliberately NOT here (CLUS-19,
        // HOST-15). Every member needs the COORDINATOR's address, which is
        // `coordIp` above; none of them needs each other's, because workers
        // never dial each other — the coordinator drives every RPC link. The
        // field used to be broadcast to everyone anyway, which handed any
        // member (or anything that got one member's token) a map of the home
        // network for free. The creator still keeps each member's address
        // privately, in Peer::ip, which is what member_authorized checks
        // against and what the layer plan is built from.
        "members": inner.peers.iter().map(|p| json!({
            "id": p.id, "hostname": p.hostname, "gpu": p.gpu,
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
            "modelReady": p.model_ready,
        })).collect::<Vec<_>>(),
    })
}

fn model_path_ready(path: &str) -> bool {
    if path.trim().is_empty() {
        return false;
    }
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Merge a member's local model report and bind readiness to this cluster's
/// exact model identity. Missing fields are an old client and fail closed.
fn merge_peer_model(peer: &mut Peer, req: &Value, model_id: &str, quant: &str) {
    if let Some(v) = req["modelId"].as_str() {
        peer.model_id = v.to_string();
    }
    if let Some(v) = req["quant"].as_str() {
        peer.quant = v.to_string();
    }
    peer.model_ready = req["modelReady"].as_bool() == Some(true)
        && peer.model_id == model_id
        && peer.quant == quant;
}

/// The largest memory figure a member may claim (CLUS-09).
///
/// These numbers are self-reported and the cluster adds them up to decide
/// whether the selected model fits before anyone presses Start. A member that
/// claims 2^63 bytes of free VRAM does not just lie about itself — it makes the
/// pool total meaningless, so every machine's UI says "this model fits" and the
/// engine fails at load time instead, which is the least debuggable place for
/// it to fail. 4 TiB is far above any home machine and far below the range
/// where the totals stop making sense.
const MAX_CLAIMED_MEMORY: u64 = 4 << 40;

/// Refresh a roster member's resource report from a join or heartbeat.
///
/// Missing keys mean an older client and leave the last observation intact;
/// an explicit zero remains meaningful (probe unavailable / no such pool).
/// Keeping this in one helper prevents the accepted-join path and the steady
/// roster path from drifting again.
///
/// An out-of-range claim is dropped loudly rather than clamped: clamping would
/// silently turn a lie into a plausible number and the pool total would still
/// be wrong, just harder to notice. Nothing here can verify a claim that IS
/// plausible — that residual is CLUS-09 and it stays open.
fn merge_peer_memory(peer: &mut Peer, req: &Value) {
    let sane = |v: u64, what: &str| -> Option<u64> {
        if v > MAX_CLAIMED_MEMORY {
            eprintln!(
                "[pairing] {} reported {v} bytes of free {what} — refusing a figure no \
                 machine has; its contribution is counted as unknown",
                peer.id
            );
            None
        } else {
            Some(v)
        }
    };
    if let Some(v) = req["vramFree"].as_u64().and_then(|v| sane(v, "VRAM")) {
        peer.vram_free = v;
    }
    if let Some(v) = req["ramFree"].as_u64().and_then(|v| sane(v, "RAM")) {
        peer.ram_free = v;
    }
    if let Some(v) = req["unifiedMemory"].as_bool() {
        peer.unified_memory = v;
    }
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
    // Adopt the creator's exact model identity: the cluster serves ONE model.
    // A local path resolved for another selection must be forgotten here. It
    // is especially dangerous on a joiner that selected Q4 before joining a
    // creator serving Q2: existence alone would otherwise make the wrong file
    // look ready and the worker would seed its cache with unrelated tensors.
    let old_model = inner.tuning.model_id.clone();
    let old_quant = inner.tuning.quant.clone();
    if let Some(m) = v["modelId"].as_str().filter(|m| !m.is_empty()) {
        inner.tuning.model_id = m.to_string();
    }
    // Adopt the creator's precision alongside the model (absent for older
    // creators → keep our default, which the coord maps to the model default).
    if let Some(q) = v["quant"].as_str().filter(|q| !q.is_empty()) {
        inner.tuning.quant = q.to_string();
    }
    if inner.tuning.model_id != old_model || inner.tuning.quant != old_quant {
        inner.model_path.clear();
    }
    let cluster_model = inner.tuning.model_id.clone();
    let cluster_quant = inner.tuning.quant.clone();

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
                // Joiner-side mirror: tokens are the creator's bookkeeping and
                // are never broadcast, so every entry here is blank. This
                // machine's own token lives in its roster loop.
                token: String::new(),
                // Device ids and peer addresses are creator-side bookkeeping
                // too, and are not broadcast (CLUS-19/HOST-15).
                device_id: String::new(),
                wants_coordinator: m["wantsCoordinator"].as_bool().unwrap_or(false),
                // Memory travels with the roster so every machine can total
                // the pool, not just the creator (0 = an older peer).
                vram_free: m["vramFree"].as_u64().unwrap_or(0),
                ram_free: m["ramFree"].as_u64().unwrap_or(0),
                unified_memory: m["unifiedMemory"].as_bool().unwrap_or(false),
                model_ready: m["modelReady"].as_bool().unwrap_or(false),
                model_id: cluster_model.clone(),
                quant: cluster_quant.clone(),
            }
        })
        .collect();
    inner.coordinator_id = coordinator_id;
    let mut eff = RosterEffect::default();
    eff.rejoin = !inner.peers.iter().any(|p| p.is_self);
    eff.start_engine = phase == "starting" && inner.phase == "idle" && !inner.engine_started;
    // Back to "idle" after we had already started means the cluster we belong
    // to no longer exists in the form we joined. Drop our engine and re-arm, so
    // the next "starting" launches a worker for the newly selected local GGUF.
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
    v
}

/// Overflow flags for the coordinator ("borrow another machine when this one is
/// full"). Both the URL and the key or nothing: the coordinator treats their
/// presence as the switch, so passing one without the other would ask it to
/// enable a feature it cannot use and it would refuse to start.
///
/// The KEY does not appear here — it goes through `secret_env` below. The URL
/// does, because it is not a secret and a command line that says which platform
/// this machine borrows from is worth having in a log.
///
/// Deliberately NOT guarded on api_token here. The coordinator's own refusal is
/// the guard, and duplicating it would mean a client that silently drops the
/// flags instead of surfacing why -- the user would see sharing "on" in the
/// panel and a machine that never borrows.
fn overflow_args(tuning: &Tuning) -> Vec<String> {
    if tuning.overflow_url.is_empty() || tuning.overflow_key.is_empty() {
        return Vec::new();
    }
    // The coordinator has no TLS client and refuses https:// at startup —
    // which, on this launch path, is a crash loop the user reads as "the
    // engine keeps dying" (2026-08-21). Hand it the plaintext spelling.
    let mut v = vec![
        "--overflow-url".into(), crate::engine::engine_platform_url(&tuning.overflow_url),
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

/// Credentials for the coordinator, as environment variables rather than
/// command-line arguments (2026-08-20 audit A-P0-4).
///
/// `/proc/<pid>/cmdline` is world-readable on Linux and `ps` shows the same
/// thing everywhere, so anything in argv is readable by every other account on
/// the machine — including the one that gates this machine's inference API and
/// the one that spends this account's Sparks. A child's environment is not:
/// `/proc/<pid>/environ` is owner-only, and the client passes these per child
/// (`Command::env`) rather than setting them on itself.
///
/// The names are the ones `src/coord/coord_main.c` already reads as fallbacks
/// (`IDLETOKEN_API_TOKEN`, `IDLETOKEN_OVERFLOW_URL/KEY`), so this is a change of
/// channel, not of contract — verified against the coordinator's own argument
/// parser before switching, because "passed it but the engine never read it"
/// would silently turn the API token off.
fn secret_env(tuning: &Tuning) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();
    if !tuning.api_token.is_empty() {
        env.push(("IDLETOKEN_API_TOKEN".into(), tuning.api_token.clone()));
    }
    if !tuning.overflow_url.is_empty() && !tuning.overflow_key.is_empty() {
        env.push(("IDLETOKEN_OVERFLOW_KEY".into(), tuning.overflow_key.clone()));
    }
    // Not secrets — they ride here because this is the one env channel every
    // coordinator spawn already passes through, and a second channel would be
    // the kind that one launch path forgets.
    if !tuning.kv_cache_k.is_empty() {
        env.push(("IDLETOKEN_KV_CACHE_TYPE".into(), tuning.kv_cache_k.clone()));
    }
    if !tuning.kv_cache_v.is_empty() {
        env.push(("IDLETOKEN_KV_CACHE_TYPE_V".into(), tuning.kv_cache_v.clone()));
    }
    env
}

/// Start this machine's engine(s) per its role in the frozen roster. The
/// coordinator's llama-server uses local compute directly; only other machines
/// run rpc-supervisors. The coordinator's repository remains the authoritative
/// tensor index (and an old-client compatibility source), while current workers
/// seed only their assigned tensors from their own complete local GGUF.
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
        let engine_bin_arg = match crate::engine::native_path_arg(
            &engine_bin, "llama-server path") {
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
            "--llama-server-bin".into(), engine_bin_arg,
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
        // Shared mode's second door — the SAME one llamacpp_serve opens, and
        // missing here until 2026-08-21. The comment above says this engine is
        // hardened "always, not only once someone presses share compute", and
        // lists "unix-socket link" among the flags that do it; that flag was
        // simply never passed. The result was a cluster-path engine that looked
        // hardened and had no socket, so switching sharing on produced
        //   platform-agent: refuse: --coord-unix ...\coord-api.sock is not
        //   accepting connections
        // and the agent stopped — correctly, since it will not put a buyer's
        // prompt on loopback TCP where this machine's owner can read it. The
        // agent was right; the door was missing.
        //
        // Taken from engine.rs rather than rebuilt here: the coordinator and
        // the agent must agree on this path, and a second derivation is the
        // kind of copy that drifts silently.
        let coord_socket = match crate::engine::coord_api_socket() {
            Ok(sock) => sock,
            Err(e) => {
                eprintln!("[pairing] coordinator start refused: {e}");
                return;
            }
        };
        coord_args.push("--api-unix".into());
        coord_args.push(coord_socket);
        // The user's "Resource usage" caps, which reached the WORKER (line
        // ~925) and not the coordinator until 2026-08-21. On a single machine
        // the coordinator budgets from its own probe rather than from a
        // worker's HELLO — that is the "SINGLE: N GiB needed ... fits the
        // coordinator's M GiB usable" decision in its log — so an uncapped
        // coordinator plans against the whole machine no matter where the
        // sliders are set. Same flags, same contract, same helper as the
        // worker: one source, so the two cannot drift apart again.
        coord_args.extend(usage_cap_args(&tuning));
        if remote_workers > 0 {
            coord_args.push("--num-workers".into());
            coord_args.push(remote_workers.to_string());
            // Reaching this branch means the user explicitly chose the
            // multi-machine flow. Capacity still decides which choice the UI
            // highlights by default; it must not silently undo this choice
            // just because the selected quant also fits the coordinator.
            coord_args.push("--force-cluster".into());
            // ⚠ Still argv, unlike the API token below: the coordinator has no
            // IDLETOKEN_PAIR_CODE fallback (the worker does — see the worker
            // branch). Moving it needs a one-line env read in
            // src/coord/coord_main.c, which is another session's tree; passing
            // it through a variable the engine does not read would leave the
            // cluster with no pairing code at all. Tracked as a cross-tree
            // follow-up in the A-P0-4 report.
            coord_args.push("--pair-code".into());
            coord_args.push(engine_code);
        }
        coord_args.extend(overflow_args(&tuning));
        // Credentials go through the environment, never argv (A-P0-4).
        if let Err(e) = crate::engine::start_engine(app, "coordinator".into(), coord_args, secret_env(&tuning)) {
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
        let engine_dir_arg = match crate::engine::native_path_arg(
            &engine_dir, "llama.cpp engine directory") {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[pairing] rpc worker start refused: {e}");
                return;
            }
        };
        let mut worker_args = vec![
            "--rpc-supervisor".into(),
            "--engine-dir".into(), engine_dir_arg,
            "--coordinator".into(), format!("{coord_ip}:{COORD_PORT}"),
            "--discovery-port".into(), tuning.discovery_port.to_string(),
            "--rpc-host".into(), self_ip(&tuning),
            "--rpc-port".into(), tuning.inter_stage_port.to_string(),
        ];
        // Every node keeps the complete curated GGUF on its own disk. The RPC
        // supervisor uses it only as a source for the tensor range assigned to
        // this machine; llama.cpp still imports just that range into the
        // worker's cache/GPU/RAM. Passing both the primary file and its folder
        // also handles split GGUF siblings without teaching the client their
        // naming scheme.
        if model_path_ready(&model_path) {
            worker_args.push("--model".into());
            worker_args.push(model_path.clone());
            if let Some(parent) = Path::new(&model_path).parent() {
                match crate::engine::native_path_arg(parent, "GGUF directory") {
                    Ok(path) => {
                        worker_args.push("--gguf-dir".into());
                        worker_args.push(path);
                    }
                    Err(e) => {
                        eprintln!("[pairing] rpc worker start refused: {e}");
                        return;
                    }
                }
            }
        }
        worker_args.extend(usage_cap_args(&tuning));
        // The pairing code is the secret the cluster TLS PSK is derived from,
        // so it does not belong in a world-readable command line (A-P0-4).
        // `src/worker/worker_main.c` reads IDLETOKEN_PAIR_CODE as a fallback,
        // which is what makes this safe to switch.
        let worker_env = vec![("IDLETOKEN_PAIR_CODE".to_string(), engine_code)];
        if let Err(e) = crate::engine::start_engine(app, "worker".into(), worker_args, worker_env) {
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

/// The refusal every unauthenticated `roster`/`leave` gets.
///
/// One string, and an explicit one: a request that arrives without a valid
/// token is REFUSED, never quietly served a trimmed reply. An older client
/// that does not send the field sees this and can say why (the joiner maps it
/// to a re-join); a stranger on the LAN sees it and gets nothing.
const ERR_UNAUTHORIZED: &str = "unauthorized: this request needs the member token issued at join";

/// Does this request carry the token the creator minted for `id`, from the
/// address that member joined from?
///
/// Both halves matter. The token alone would still let someone who saw one go
/// on using it from anywhere; the address alone is trivially spoofable on a
/// LAN. A member whose address changed (DHCP lease, reconnect) fails this,
/// gets `ERR_UNAUTHORIZED`, and its loop re-joins — which re-records the new
/// address. That is the intended path, not an error state.
fn member_authorized(inner: &Inner, req: &Value, peer_ip: &str) -> Option<String> {
    let id = req["id"].as_str().unwrap_or("");
    let token = req["token"].as_str().unwrap_or("");
    if id.is_empty() || token.is_empty() {
        return None;
    }
    let p = inner.peers.iter().find(|p| p.id == id)?;
    if !token_eq(&p.token, token) || p.ip != peer_ip {
        return None;
    }
    Some(id.to_string())
}

/// Is this join allowed to bind to `id`, given who already holds it? (CLUS-06)
///
/// The rule is first-wins on the LABEL, with the DEVICE deciding sameness:
///
/// * nobody holds the name           → bind it;
/// * the holder's device matches     → the same machine is back (reboot, new
///                                     DHCP lease, creator restart) → rebind;
/// * the holder is a pre-device-id   → adopt it only if it has gone offline,
///   client and this one has an id     so an upgrade is not a takeover of a
///                                     machine that is sitting right there;
/// * anything else                   → a DIFFERENT machine wants a name that is
///                                     taken. Refused, not merged.
///
/// The refusal matters more than it looks. Before this, a second machine
/// claiming an existing hostname simply overwrote that member's address, token
/// and resource numbers — so anyone holding the join code could silently
/// displace a real node, and the displaced machine's next poll would be told
/// "unauthorized" and re-join, producing a flap rather than an error anyone
/// could see (CLUS-20).
enum JoinIdentity {
    /// Free to take (new member, or the same device coming back).
    Bind,
    /// Taken by a different machine that is still present.
    Conflict,
}

fn join_identity(inner: &Inner, id: &str, device: &str, now: Instant) -> JoinIdentity {
    let Some(p) = inner.peers.iter().find(|p| p.id == id) else {
        return JoinIdentity::Bind;
    };
    if !p.device_id.is_empty() && p.device_id == device {
        return JoinIdentity::Bind;
    }
    if p.device_id.is_empty() && !device.is_empty() {
        // An older client is holding the name. Let a device-id client take it
        // over only once that machine has stopped answering — otherwise "I
        // upgraded" and "I am impersonating the machine next to me" are the
        // same request.
        let stale = p.last_seen.map_or(true, |seen| {
            now.duration_since(seen)
                > Duration::from_secs((p.hb_secs.clamp(1, 60) as u64 * 3).max(OFFLINE_AFTER_S))
        });
        return if stale || !p.online { JoinIdentity::Bind } else { JoinIdentity::Conflict };
    }
    if p.device_id.is_empty() && device.is_empty() {
        // Both sides predate device ids. Nothing here can tell impersonation
        // from a legitimate rejoin, so keep the historical behaviour rather
        // than lock out working clusters — and say so in the audit trail.
        return JoinIdentity::Bind;
    }
    JoinIdentity::Conflict
}

/// Append one bounded line to the membership audit trail.
fn audit(inner: &mut Inner, line: String) {
    eprintln!("[pairing] {line}");
    inner.audit.push(line);
    // Bounded: an attacker who can drive membership changes must not be able to
    // drive memory growth with them.
    while inner.audit.len() > 64 {
        inner.audit.remove(0);
    }
}

/// The roster protocol, as a pure-ish function of (state, request, source).
///
/// Split out of the socket handling on purpose: everything below is reachable
/// by any device on the LAN, so it is the part that has to be testable without
/// a network. `now` is a parameter for the same reason the engine's backoff
/// curve is a pure function — a timing property asserted by sleeping is a
/// property that is not really asserted.
fn roster_request(inner: &mut Inner, req: &Value, peer_ip: &str, now: Instant) -> Value {
    // "Only same subnet": refuse before anything else is considered, and say
    // why. A silent drop here is indistinguishable from a firewall and would
    // send someone hunting the wrong problem. Applied to `hello` too, because
    // that step is where THIS machine proves it knows the code — a restriction
    // that leaks the proof to the excluded network is not one.
    let wrong_subnet =
        inner.tuning.same_subnet_only && !same_subnet(peer_ip, &self_ip(&inner.tuning));
    let subnet_err =
        json!({"ok": false, "err": "different subnet (this cluster is restricted to one subnet)"});

    // Every peer-chosen string that will be stored or echoed is gated here,
    // once, before any of it reaches the roster (CLUS-14). A hostname that can
    // close a JSON string rewrites the document every member's UI parses, and
    // it does not need a parser bug to do it.
    let fields_ok = peer_field_opt_ok(req["id"].as_str())
        && peer_field_opt_ok(req["hostname"].as_str())
        && peer_field_opt_ok(req["gpu"].as_str())
        && peer_field_opt_ok(req["modelId"].as_str())
        && peer_field_opt_ok(req["quant"].as_str())
        && peer_field_opt_ok(req["token"].as_str())
        && peer_field_opt_ok(req["deviceId"].as_str())
        && peer_field_opt_ok(req["proof"].as_str())
        && peer_field_opt_ok(req["engine"].as_str());
    if !fields_ok {
        eprintln!("[pairing] refused a request from {peer_ip} carrying an unusable field value");
        return json!({"ok": false, "err": "bad field"});
    }

    match req["op"].as_str() {
        // Step one of the join handshake (A-P0-3): answer the caller's nonce
        // with proof that we hold the same join code, and hand out a nonce of
        // our own for its answering proof. A caller that cannot verify this
        // never sends us anything.
        Some("hello") if wrong_subnet => subnet_err,
        Some("hello") => {
            let their_nonce = req["nonce"].as_str().unwrap_or("");
            let code = inner.code.clone().unwrap_or_default();
            // A peer that does not speak v2 is told to update rather than
            // handed a v1 proof. v1 was a single SHA-256 over a fixed format,
            // i.e. an offline oracle for the whole 30-bit code space; answering
            // it "for compatibility" would keep the hole open for anyone who
            // simply omits the field.
            if req["v"].as_u64().unwrap_or(1) < PAIR_PROTO_V {
                return json!({"ok": false, "err":
                    "this machine is running an older IdleToken whose pairing handshake \
                     is no longer accepted; update it and try again"});
            }
            if their_nonce.is_empty() || !peer_field_ok(their_nonce) || code.is_empty() {
                return json!({"ok": false, "err": "bad hello"});
            }
            let ours = random_hex(16);
            inner.challenges.retain(|c| now.duration_since(c.at) < CHALLENGE_TTL);
            // Fair share: a source spraying `hello` may only ever evict its
            // OWN oldest nonce. Before this, one sprayer emptied the shared
            // 64-entry list and every honest joiner's follow-up `join` failed
            // with "bad code" — a denial of service that reported itself as a
            // wrong code.
            while inner.challenges.iter().filter(|c| c.src == peer_ip).count()
                >= MAX_CHALLENGES_PER_SOURCE
            {
                if let Some(i) = inner.challenges.iter().position(|c| c.src == peer_ip) {
                    inner.challenges.remove(i);
                } else {
                    break;
                }
            }
            while inner.challenges.len() >= MAX_CHALLENGES {
                inner.challenges.remove(0);
            }
            inner.challenges.push(Challenge {
                nonce: ours.clone(),
                at: now,
                src: peer_ip.to_string(),
            });
            json!({
                "ok": true,
                "v": PAIR_PROTO_V,
                "proof": pair_proof("creator", &code, their_nonce),
                "nonce": ours,
            })
        }
        Some("join") if wrong_subnet => subnet_err,
        Some("join") => {
            // The code itself no longer travels: the joiner answers the nonce
            // we issued above. An unknown or expired nonce is refused rather
            // than treated as a fresh handshake — otherwise the challenge is
            // decoration.
            let proof = req["proof"].as_str().unwrap_or("");
            let code = inner.code.clone().unwrap_or_default();
            inner.challenges.retain(|c| now.duration_since(c.at) < CHALLENGE_TTL);
            // Only nonces WE issued to THIS source count. A nonce is handed to
            // one address; letting another address spend it would turn the
            // challenge list into a shared pool.
            let matched = inner.challenges.iter().position(|c| {
                c.src == peer_ip && token_eq(&pair_proof("joiner", &code, &c.nonce), proof)
            });
            let Some(at) = matched else {
                inner.gate.note_failure(peer_ip, now);
                return json!({"ok": false, "err": "bad code"});
            };
            inner.challenges.remove(at);
            inner.gate.note_success(peer_ip, now);

            let id = req["hostname"].as_str().unwrap_or("?").to_string();
            let device = req["deviceId"].as_str().unwrap_or("").to_string();
            if let JoinIdentity::Conflict = join_identity(inner, &id, &device, now) {
                audit(
                    inner,
                    format!(
                        "refused a join from {peer_ip}: the name {id:?} already belongs to \
                         another machine in this cluster. Rename one of them and try again."
                    ),
                );
                return json!({"ok": false, "err":
                    "that machine name is already used by another machine in this cluster — \
                     rename one of them and try again"});
            }

            let cluster_model = inner.tuning.model_id.clone();
            let cluster_quant = inner.tuning.quant.clone();

            // Admission requires the weights to ALREADY be on the joiner's disk
            // (2026-09-01). Until now a member was admitted first and
            // downloaded afterwards, which had two costs: a roster could sit
            // for an hour in a state that could not start, and joining silently
            // began a transfer the size of the model — 90 GB, on a link the
            // user never agreed to spend.
            //
            // The refusal carries this cluster's model identity because being
            // refused is now the ONLY way a joiner can learn it: it never
            // reaches the roster that used to carry it. That is also why this
            // check sits AFTER the code proof — the identity goes only to a
            // machine that proved it holds the join code, not to anything on
            // the LAN that opens this port.
            let their_model = req["modelId"].as_str().unwrap_or("");
            let their_quant = req["quant"].as_str().unwrap_or("");
            let claims_ready = req["modelReady"].as_bool() == Some(true);
            if !claims_ready || their_model != cluster_model || their_quant != cluster_quant {
                audit(
                    inner,
                    format!(
                        "refused a join from {peer_ip}: this cluster runs {cluster_model} \
                         {cluster_quant}, and that machine reported {their_model} {their_quant} \
                         (weights ready: {claims_ready})"
                    ),
                );
                return json!({
                    "ok": false,
                    "err": "model not ready",
                    "modelId": cluster_model,
                    "quant": cluster_quant,
                });
            }

            let hb = (req["hb"].as_u64().unwrap_or(0) as u32).clamp(0, 60);
            // A fresh token on every accepted join, including a re-join.
            // Rotating it is what keeps a token that leaked from outliving
            // the session it was seen in; the member always gets the new
            // one in this very reply, so nothing has to be reconciled.
            let token = random_hex(16);
            let known = inner.peers.iter().any(|p| p.id == id);
            if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                // Re-register under a known id: the machine is back (or
                // re-joined after a creator restart). Refresh liveness and
                // its address — a reboot may have changed the IP.
                p.online = true;
                p.last_seen = Some(now);
                p.hb_secs = hb;
                p.ip = peer_ip.to_string();
                p.token = token.clone();
                p.device_id = device.clone();
                // The probe may have completed after the first join, or a
                // restarted client may now know more than its old roster
                // entry. Re-registration must refresh resources too.
                merge_peer_memory(p, req);
                merge_peer_model(p, req, &cluster_model, &cluster_quant);
            } else {
                let mut peer = Peer {
                    id: id.clone(),
                    hostname: id.clone(),
                    gpu: req["gpu"].as_str().unwrap_or("").to_string(),
                    role: "worker",
                    is_self: false,
                    stage: "joined".into(),
                    layer_lo: None,
                    layer_hi: None,
                    online: true,
                    last_seen: Some(now),
                    hb_secs: hb,
                    ip: peer_ip.to_string(),
                    token: token.clone(),
                    device_id: device.clone(),
                    wants_coordinator: false,
                    vram_free: 0,
                    ram_free: 0,
                    unified_memory: req["unifiedMemory"].as_bool().unwrap_or(false),
                    model_ready: false,
                    model_id: String::new(),
                    quant: String::new(),
                };
                merge_peer_memory(&mut peer, req);
                merge_peer_model(&mut peer, req, &cluster_model, &cluster_quant);
                inner.peers.push(peer);
            }

            // The joiner's "Prefer this machine as coordinator" is recorded as
            // a REQUEST and nothing more (CLUS-08). It used to move
            // coordinator_id on the spot, which handed any machine holding the
            // join code the power to relocate the cluster's plaintext window —
            // the one node that sees every prompt in the clear — to itself,
            // with no confirmation anywhere. Applying it is now
            // `pairing_set_coordinator`, which only the creator's own UI calls.
            let wants = req["prefer"].as_bool() == Some(true);
            if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                let changed = p.wants_coordinator != wants;
                p.wants_coordinator = wants;
                if wants && changed {
                    let line = format!(
                        "{id} asked to become the coordinator. It stays a worker until you \
                         approve it in Manage cluster — the coordinator is the machine that \
                         sees prompts in the clear."
                    );
                    audit(inner, line);
                }
            }
            if !known {
                audit(inner, format!("{id} joined from {peer_ip}"));
            }

            let mut r = roster_reply(inner);
            r["id"] = json!(id);
            // The member's proof for every later request. It travels only
            // in the reply to the machine that just proved the code.
            r["token"] = json!(token);
            r
        }
        Some("roster") => match member_authorized(inner, req, peer_ip) {
            None => {
                eprintln!(
                    "[pairing] refused an unauthenticated roster request from {peer_ip} \
                     (id={:?}) — the roster is only served to machines that joined",
                    req["id"].as_str().unwrap_or("")
                );
                json!({"ok": false, "err": ERR_UNAUTHORIZED})
            }
            Some(id) => {
                // Live per-node progress: a member's poll carries its engine state.
                let eng = req["engine"].as_str().unwrap_or("");
                let forming = inner.phase == "idle"; // read before the &mut borrow below
                let cluster_model = inner.tuning.model_id.clone();
                let cluster_quant = inner.tuning.quant.clone();
                if let Some(p) = inner.peers.iter_mut().find(|p| p.id == id) {
                    // The poll IS the liveness signal: hearing it revives a
                    // member the sweep had marked offline.
                    p.online = true;
                    p.last_seen = Some(now);
                    p.hb_secs = (req["hb"].as_u64().unwrap_or(0) as u32).clamp(0, 60);
                    // Memory is live roster data, not a one-shot join fact.
                    // This is what makes a probe that finishes after pairing
                    // repair the pool verdict without leaving/rejoining.
                    merge_peer_memory(p, req);
                    merge_peer_model(p, req, &cluster_model, &cluster_quant);
                    if !eng.is_empty() && !forming && p.stage != "ready" {
                        p.stage = stage_for_engine(eng).to_string();
                    }
                }
                roster_reply(inner)
            }
        },
        Some("leave") => match member_authorized(inner, req, peer_ip) {
            None => {
                eprintln!(
                    "[pairing] refused an unauthenticated leave from {peer_ip} (id={:?}) \
                     — evicting a member needs that member's own token",
                    req["id"].as_str().unwrap_or("")
                );
                json!({"ok": false, "err": ERR_UNAUTHORIZED})
            }
            // Only the entry the token belongs to. `member_authorized` returns
            // the id the TOKEN maps to, not the one the request asked for, so
            // "leave" cannot be aimed at anybody else's machine.
            Some(id) => {
                inner.peers.retain(|p| p.id != id);
                audit(inner, format!("{id} left"));
                json!({"ok": true})
            }
        },
        _ => json!({"ok": false, "err": "bad op"}),
    }
}

/// One roster-protocol request on the creator side: bounded I/O around
/// `roster_request`.
fn handle_roster_conn(app: &AppHandle, stream: TcpStream, generation: u64) {
    let peer_ip = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));

    // Admission BEFORE the read (CLUS-05). The accept loop is serial, so a
    // source that is already serving a penalty must cost us a close() and not a
    // read window — otherwise being refused is itself the way to hold the
    // thread. Nothing is parsed, nothing is allocated.
    {
        let pairing = app.state::<Pairing>();
        let mut inner = pairing.0.lock().unwrap();
        if inner.generation != generation || inner.mode != Mode::Creator {
            return;
        }
        let now = Instant::now();
        if let Err(wait) = inner.gate.admit(&peer_ip, now) {
            drop(inner);
            let mut s = stream;
            let _ = writeln!(
                s,
                "{}",
                json!({"ok": false, "err": format!(
                    "too many attempts from this machine — try again in {}s", wait.as_secs().max(1))})
            );
            return;
        }
    }

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    // `read_line` on a raw socket grows until it meets a newline. `take` is what
    // makes that bounded: an unauthenticated LAN peer streaming bytes with no
    // `\n` used to be an out-of-memory condition, reachable without the join
    // code (CLUS-05, CHAIN-08).
    let read = (&mut reader)
        .take(MAX_ROSTER_REQUEST_BYTES + 1)
        .read_line(&mut line);
    match read {
        Ok(n) if n as u64 > MAX_ROSTER_REQUEST_BYTES => {
            eprintln!(
                "[pairing] refused an oversized roster request from {peer_ip} \
                 ({n} bytes with no end of line)"
            );
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            inner.gate.note_failure(&peer_ip, Instant::now());
            return;
        }
        Ok(0) | Err(_) => return,
        Ok(_) => {}
    }
    let Ok(req) = serde_json::from_str::<Value>(&line) else {
        // Unparseable input from an unauthenticated source is not a neutral
        // event on a serial loop; it costs the sender the same as a wrong code.
        let pairing = app.state::<Pairing>();
        let mut inner = pairing.0.lock().unwrap();
        inner.gate.note_failure(&peer_ip, Instant::now());
        return;
    };
    // A JSON scalar or array indexes as null everywhere below, which would read
    // as "every field absent" rather than "this is not a request".
    if !req.is_object() {
        return;
    }

    let pairing = app.state::<Pairing>();
    let mut inner = pairing.0.lock().unwrap();
    if inner.generation != generation || inner.mode != Mode::Creator {
        return;
    }
    let reply = roster_request(&mut inner, &req, &peer_ip, Instant::now());
    drop(inner);
    emit_snapshot(app);
    let mut stream = reader.into_inner();
    let _ = writeln!(stream, "{reply}");
}

/// One discovery beacon packet: `IDLETOKEN2|<session>|<nonce>|<roster port>`.
///
/// Split out so a test can assert the property the whole change rests on — the
/// bytes on the wire do not depend on the join code (A-P0-3).
fn beacon_packet(session: &str, nonce: &str, roster_port: u16) -> String {
    format!("{BEACON_MAGIC}|{session}|{nonce}|{roster_port}")
}

fn spawn_creator_tasks(app: AppHandle, generation: u64, _code: String, discovery_port: u16) {
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
    // UDP beacon: `IDLETOKEN2|<session>|<nonce>|<roster_port>` once a second.
    // Nothing in it is derived from the join code (see BEACON_MAGIC) — the
    // session id is random and per cluster, the nonce is random and per packet.
    let beacon_app = app.clone();
    let session = random_hex(8);
    std::thread::spawn(move || {
        if !announce {
            return;
        }
        let Ok(sock) = UdpSocket::bind(("0.0.0.0", 0)) else { return };
        let _ = sock.set_broadcast(true);
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
            let msg = beacon_packet(&session, &random_hex(8), ROSTER_PORT);
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

/// One newline-JSON round trip to a roster service. `None` = did not answer.
fn roster_call(ip: &str, req: &Value) -> Option<Value> {
    let addr: SocketAddr = format!("{ip}:{ROSTER_PORT}").parse().ok()?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    writeln!(s, "{req}").ok()?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line).ok()?;
    serde_json::from_str(&line).ok()
}

/// Joiner: collect the machines announcing themselves, offer the code to each,
/// then poll the one that accepted it.
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

        // 1) collect the machines announcing themselves — every one of them, not
        //    the one whose beacon "matches". Since A-P0-3 the beacon carries
        //    nothing derived from the code (that is what stopped it from being
        //    brute-forceable), so which cluster is ours is decided by offering
        //    the code to each candidate in step 3 and seeing who accepts.
        //    Skipped entirely when LAN auto-discovery is off.
        let mut candidates: Vec<String> = Vec::new();
        if listen {
            if let Ok(sock) = UdpSocket::bind(("0.0.0.0", discovery_port)) {
                let _ = sock.set_read_timeout(Some(Duration::from_secs(1)));
                let mut buf = [0u8; 256];
                for _ in 0..8 {
                    {
                        let pairing = app.state::<Pairing>();
                        if pairing.0.lock().unwrap().generation != generation {
                            return;
                        }
                    }
                    let Ok((n, src)) = sock.recv_from(&mut buf) else { continue };
                    let msg = String::from_utf8_lossy(&buf[..n]);
                    let parts: Vec<&str> = msg.trim().split('|').collect();
                    // v2 = magic|session|nonce|port, v1 = magic|hash|port. The
                    // payload is not read either way: only the source address
                    // matters, and an older creator is still a creator.
                    let known = matches!(parts.first(), Some(&m) if m == BEACON_MAGIC || m == BEACON_MAGIC_LEGACY);
                    if !known {
                        continue;
                    }
                    let ip = src.ip().to_string();
                    // "Only same subnet" also filters what we LISTEN to, not
                    // just what we accept: on a bridged VM or a VPN the beacon
                    // can arrive from a network the user deliberately excluded.
                    if subnet_only && !same_subnet(&ip, &my_ip) {
                        eprintln!("[pairing] ignoring beacon from {ip}: different subnet");
                        continue;
                    }
                    if !candidates.contains(&ip) {
                        candidates.push(ip);
                    }
                }
            }
        }

        // 2) manual peers are the fallback the beacon cannot be: broadcast does
        //    not cross subnets and plenty of networks drop it entirely.
        for ip in &manual {
            if subnet_only && !same_subnet(ip, &my_ip) {
                eprintln!("[pairing] skipping manual peer {ip}: different subnet");
                continue;
            }
            if !candidates.contains(ip) {
                candidates.push(ip.clone());
            }
        }

        let self_device = {
            let pairing = app.state::<Pairing>();
            let inner = pairing.0.lock().unwrap();
            inner.self_device_id.clone()
        };
        let (self_host, self_gpu) = {
            let pairing = app.state::<Pairing>();
            let inner = pairing.0.lock().unwrap();
            (inner.self_host.clone(), inner.self_gpu.clone())
        };

        /// Why a join can fail before it is even sent.
        enum Handshake {
            /// The machine answered and holds the same join code; here is the
            /// nonce to answer with.
            Ok(String),
            /// Answered, but could not prove the code: a different cluster, or
            /// something pretending to be one. Never send it our proof.
            NotOurs,
            /// Answered, but does not speak the handshake at all — an older
            /// IdleToken. Refused loudly rather than falling back to sending
            /// the code in the clear, which is the hole the handshake closes.
            TooOld,
            /// Did not answer.
            Silent,
        }

        // Step one of the join (A-P0-3): make the other machine prove it knows
        // the code, against a nonce we pick, BEFORE we prove anything to it.
        let hello = |ip: &str| -> Handshake {
            let mine = random_hex(16);
            let Some(v) = roster_call(ip, &json!({"op": "hello", "nonce": mine, "v": PAIR_PROTO_V}))
            else {
                return Handshake::Silent;
            };
            if v["ok"].as_bool() == Some(false) {
                return match v["err"].as_str() {
                    Some("bad op") => Handshake::TooOld,
                    // A v2 creator refusing a v1 handshake says so explicitly;
                    // surface it as "that machine is too old" rather than as a
                    // wrong code, which is what it would otherwise look like.
                    Some(e) if e.contains("older IdleToken") => Handshake::TooOld,
                    _ => Handshake::NotOurs,
                };
            }
            // No version in the reply means a creator that predates the
            // stretched proof (pair_proof v2). Refuse rather than recompute the
            // v1 proof: v1 is the offline oracle this replaced, and a client
            // that silently falls back keeps the hole open for both machines.
            if v["v"].as_u64().unwrap_or(1) < PAIR_PROTO_V {
                return Handshake::TooOld;
            }
            let want = pair_proof("creator", &code, &mine);
            if !token_eq(v["proof"].as_str().unwrap_or(""), &want) {
                return Handshake::NotOurs;
            }
            match v["nonce"].as_str().filter(|n| !n.is_empty()) {
                Some(n) => Handshake::Ok(n.to_string()),
                None => Handshake::TooOld,
            }
        };

        // The join request, answering the creator's nonce. Rebuilt on every
        // attempt because the memory probe may have landed in between.
        //   `prefer`: this machine's own "Prefer this machine as coordinator".
        //   Sent on every (re)join so it survives the creator restarting the
        //   roster.
        //   Memory goes with the join: the machine that owns the numbers is the
        //   one that measured them, and the roster is where the cluster totals
        //   them up (pre-flight "will this model fit on all of us together").
        let join_req = |nonce: &str| {
            let (vram_free, ram_free, unified, model_id, quant, model_ready) = {
                let pairing = app.state::<Pairing>();
                let inner = pairing.0.lock().unwrap();
                (
                    inner.self_vram_free,
                    inner.self_ram_free,
                    inner.self_unified,
                    inner.tuning.model_id.clone(),
                    inner.tuning.quant.clone(),
                    model_path_ready(&inner.model_path),
                )
            };
            json!({"op": "join", "proof": pair_proof("joiner", &code, nonce),
                   "hostname": self_host, "gpu": self_gpu,
                   // Stable identity for this install (CLUS-06): what tells a
                   // rejoin of THIS machine apart from another machine that
                   // happens to share its name.
                   "deviceId": self_device,
                   "prefer": prefer, "hb": poll.as_secs(),
                   "vramFree": vram_free, "ramFree": ram_free, "unifiedMemory": unified,
                   "modelId": model_id, "quant": quant, "modelReady": model_ready})
        };

        // 3) offer the code to each candidate until one accepts. A refusal is
        //    not fatal on its own — with several clusters on the LAN, "bad
        //    code" simply means "not this one" — but the LAST refusal is kept,
        //    because if nobody accepts it is the honest reason to show.
        let mut creator_ip: Option<String> = None;
        let mut member_token = String::new();
        let mut first_reply: Option<Value> = None;
        let mut last_refusal: Option<String> = None;
        // What the cluster demanded when it refused us for missing weights.
        // Carried out of the loop so the UI can name the model to download —
        // a refused joiner never enters the roster that used to tell it.
        let mut required_model: Option<(String, String)> = None;
        for cand in &candidates {
            {
                let pairing = app.state::<Pairing>();
                if pairing.0.lock().unwrap().generation != generation {
                    return;
                }
            }
            let nonce = match hello(cand) {
                Handshake::Ok(n) => n,
                Handshake::Silent => {
                    eprintln!("[pairing] {cand} did not answer on {ROSTER_PORT}");
                    continue;
                }
                Handshake::NotOurs => {
                    // Not an error worth showing on its own: with more than one
                    // cluster on the network this is simply "not that one". It
                    // becomes the reported reason only if nobody accepts.
                    eprintln!("[pairing] {cand} could not prove the join code — not our cluster");
                    last_refusal = Some("bad code".into());
                    continue;
                }
                Handshake::TooOld => {
                    eprintln!("[pairing] {cand} does not support the join handshake — older IdleToken");
                    last_refusal = Some("old creator".into());
                    continue;
                }
            };
            let Some(v) = roster_call(cand, &join_req(&nonce)) else {
                eprintln!("[pairing] {cand} did not answer on {ROSTER_PORT}");
                continue;
            };
            if v["ok"].as_bool() == Some(false) {
                let err = v["err"].as_str().unwrap_or("").to_string();
                eprintln!("[pairing] {cand} refused this machine: {err}");
                if err == "model not ready" {
                    // The one refusal that carries a payload: without it the
                    // user would be told to download something the UI cannot
                    // name. Both fields must be present — a creator that sends
                    // neither is one this build cannot guide the user through,
                    // so leave it unset and fall back to the plain refusal.
                    if let (Some(m), Some(q)) = (v["modelId"].as_str(), v["quant"].as_str()) {
                        required_model = Some((m.to_string(), q.to_string()));
                    }
                }
                last_refusal = Some(err);
                continue;
            }
            // The creator mints this when it accepts the code; every later
            // roster/leave request carries it (A-P0-2). A creator that does not
            // issue one is one this build cannot talk to past the join — say so
            // rather than poll forever getting "unauthorized".
            let Some(token) = v["token"].as_str().filter(|t| !t.is_empty()) else {
                eprintln!(
                    "[pairing] {cand} accepted the code but issued no member token — \
                     that machine is running an older IdleToken; update it"
                );
                last_refusal = Some("no member token issued (the hosting machine needs an update)".into());
                continue;
            };
            member_token = token.to_string();
            creator_ip = Some(cand.clone());
            first_reply = Some(v);
            break;
        }

        let Some(creator_ip) = creator_ip else {
            let pairing = app.state::<Pairing>();
            let mut inner = pairing.0.lock().unwrap();
            if inner.generation == generation {
                inner.mode = Mode::Off;
                inner.phase = "idle".into();
                inner.code = None;
                // Tell the UI, not just stderr: a silent reset to idle is
                // indistinguishable from the Join button doing nothing. A
                // refusal is a different answer from silence — somebody was
                // there and said no.
                inner.last_error = Some(match last_refusal.as_deref() {
                    Some("bad code") => ("badCode".into(), String::new()),
                    Some("old creator") => ("oldCreator".into(), String::new()),
                    Some(e) if e.starts_with("different subnet") => ("subnet".into(), String::new()),
                    // Only claim "download this" when we actually know WHAT to
                    // download. A creator that refused without naming the model
                    // falls through to the verbatim refusal below rather than
                    // to a sentence with a hole in it.
                    Some("model not ready") if required_model.is_some() => {
                        ("modelNotReady".into(), String::new())
                    }
                    Some(e) => ("rejected".into(), e.to_string()),
                    None if listen => ("notFound".into(), discovery_port.to_string()),
                    None => ("notFoundManual".into(), String::new()),
                });
                inner.required_model = required_model;
            }
            drop(inner);
            emit_snapshot(&app);
            eprintln!(
                "[pairing] no cluster accepted that code{}",
                if listen { " on this LAN" } else { " (LAN auto-discovery is off; add the host's IP under Manual peer IPs)" }
            );
            return;
        };

        // 4) apply the accepted join, then poll the roster
        let mut joined = true;
        if let Some(v) = first_reply {
            let eff = {
                let pairing = app.state::<Pairing>();
                let mut inner = pairing.0.lock().unwrap();
                if inner.generation != generation {
                    return;
                }
                apply_roster(&mut inner, &v)
            };
            emit_snapshot(&app);
            if eff.rejoin {
                joined = false;
            }
        }
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
                // sees per-node progress (P4). The token is what makes this a
                // MEMBER's poll rather than anyone's TCP connection (A-P0-2).
                let (vram_free, ram_free, unified, model_id, quant, model_ready) = {
                    let pairing = app.state::<Pairing>();
                    let inner = pairing.0.lock().unwrap();
                    (
                        inner.self_vram_free,
                        inner.self_ram_free,
                        inner.self_unified,
                        inner.tuning.model_id.clone(),
                        inner.tuning.quant.clone(),
                        model_path_ready(&inner.model_path),
                    )
                };
                json!({"op": "roster", "id": self_host, "token": member_token,
                       "deviceId": self_device,
                       "engine": crate::engine::current_state_str(&app),
                       "hb": poll.as_secs(),
                       "vramFree": vram_free, "ramFree": ram_free,
                       "unifiedMemory": unified,
                       "modelId": model_id, "quant": quant, "modelReady": model_ready})
            } else {
                // Re-joining (the creator restarted, or our token aged out).
                // A fresh handshake every time: the creator consumed the last
                // nonce, and reusing one would be a replay by definition.
                match hello(&creator_ip) {
                    Handshake::Ok(n) => join_req(&n),
                    _ => {
                        // Keep retrying. The creator is very likely mid-restart,
                        // and the loop's own "creator lost" timer is what turns
                        // a lasting silence into a message.
                        std::thread::sleep(poll);
                        continue;
                    }
                }
            };
            let reply: Option<Value> = roster_call(&creator_ip, &req);
            if let Some(v) = reply {
                if v["ok"].as_bool() == Some(false) {
                    let err = v["err"].as_str().unwrap_or("").to_string();
                    // Our token stopped being accepted: the creator rotated the
                    // roster (restart), or this machine's address changed under
                    // it. Re-join rather than die — the join is the one request
                    // that does not need a token, and it hands us a fresh one.
                    if err.starts_with("unauthorized") {
                        eprintln!("[pairing] member token no longer accepted — re-joining");
                        joined = false;
                        std::thread::sleep(poll);
                        continue;
                    }
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
                // A join reply carries a fresh token; a roster reply does not
                // (and must not clear the one we hold).
                if let Some(tok) = v["token"].as_str().filter(|t| !t.is_empty()) {
                    member_token = tok.to_string();
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
    let model_path = model_path.unwrap_or_default();
    // Formation itself is gated, not only Start. This prevents a creator from
    // advertising a cluster that can never become runnable and fixes the
    // async UI race where an empty path was captured while the verified GGUF
    // resolver was still finishing.
    if !model_path_ready(&model_path) {
        return Err("[WEIGHTS_NOT_DOWNLOADED] selected GGUF is not ready".into());
    }
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
        inner.model_path = model_path.clone();
        inner.engine_started = false;
        inner.last_error = None;
        inner.required_model = None;
        inner.challenges.clear();
        // A new cluster starts with a clean admission table and audit trail:
        // both describe the cluster that is forming, not the one before it.
        inner.gate = LanGate::default();
        inner.audit.clear();
        if inner.self_device_id.is_empty() {
            inner.self_device_id = device_id();
        }
        let self_device = inner.self_device_id.clone();
        inner.tuning = tuning.unwrap_or_default();
        discovery_port = inner.tuning.discovery_port;
        // The hardware probe normally runs before the user creates a cluster.
        // Preserve that already-known observation instead of resetting the
        // roster row to zero and waiting for a probe that may not run again.
        let self_vram_free = inner.self_vram_free;
        let self_ram_free = inner.self_ram_free;
        let self_unified = inner.self_unified;
        let model_id = inner.tuning.model_id.clone();
        let quant = inner.tuning.quant.clone();
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
            // The creator never authenticates to itself: its own entry is
            // written in-process, never over the roster socket.
            token: String::new(),
            device_id: self_device.clone(),
            // The creator is already the coordinator; it never has to ask.
            wants_coordinator: false,
            // Usually filled already by pairing_report_memory; later probe
            // refreshes still update this same row.
            vram_free: self_vram_free,
            ram_free: self_ram_free,
            unified_memory: self_unified,
            model_ready: true,
            model_id,
            quant,
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
        inner.required_model = None;
        inner.challenges.clear();
        inner.gate = LanGate::default();
        inner.audit.clear();
        if inner.self_device_id.is_empty() {
            inner.self_device_id = device_id();
        }
        inner.peers = Vec::new();
        inner.tuning = tuning.unwrap_or_default();
        discovery_port = inner.tuning.discovery_port;
    }
    emit_snapshot(&app);
    ensure_pairing_firewall(discovery_port);
    spawn_joiner_tasks(app, generation, code, discovery_port);
    Ok(())
}

/// Tell the native roster that this machine has finished resolving,
/// downloading and integrity-checking the creator-selected GGUF. The file path
/// remains local; only the exact identity-bound readiness bit is broadcast.
#[tauri::command]
pub fn pairing_update_model(
    app: AppHandle,
    state: State<'_, Pairing>,
    model_id: String,
    quant: String,
    model_path: String,
) -> Result<(), String> {
    {
        let mut inner = state.0.lock().unwrap();
        if inner.mode == Mode::Off {
            return Err("[PAIR_NO_MEMBER] this machine is not in a cluster".into());
        }
        if model_id != inner.tuning.model_id || quant != inner.tuning.quant {
            return Err(format!(
                "[PAIR_MODEL_MISMATCH] cluster needs {} {}, not {} {}",
                inner.tuning.model_id, inner.tuning.quant, model_id, quant
            ));
        }
        if !model_path_ready(&model_path) {
            return Err("[WEIGHTS_NOT_DOWNLOADED] selected GGUF is not ready".into());
        }
        inner.model_path = model_path;
        let self_id = inner.self_id.clone();
        if let Some(p) = inner.peers.iter_mut().find(|p| p.id == self_id) {
            p.model_ready = true;
            p.model_id = model_id;
            p.quant = quant;
        }
    }
    emit_snapshot(&app);
    Ok(())
}

/// Creator freezes the roster and everyone launches engines. The chosen
/// coordinator machine's ip comes from the roster (creator = local).
#[tauri::command]
pub fn pairing_start(
    app: AppHandle,
    state: State<'_, Pairing>,
    allow_solo: Option<bool>,
    model_path: Option<String>,
    overflow_tuning: Option<OverflowTuning>,
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
        // The user can change this beside the Start button, after pairing_create
        // captured the rest of the tuning. Apply both the on and off shapes;
        // the explicit backend gate clears stale credentials in the off state.
        if let Some(latest) = overflow_tuning {
            apply_overflow_tuning(&mut inner.tuning, latest);
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
        // Refresh the path at the point of use. The JS resolver is async and
        // old clients captured its initial empty value at create time.
        if let Some(path) = model_path.filter(|p| model_path_ready(p)) {
            inner.model_path = path;
            let self_id = inner.self_id.clone();
            let model_id = inner.tuning.model_id.clone();
            let quant = inner.tuning.quant.clone();
            if let Some(p) = inner.peers.iter_mut().find(|p| p.id == self_id) {
                p.model_ready = true;
                p.model_id = model_id;
                p.quant = quant;
            }
        }
        if let Some(p) = inner.peers.iter().find(|p| !p.online || !p.model_ready) {
            return Err(format!(
                "[PAIR_MODEL_NOT_READY] {} has not prepared {} {}",
                p.hostname, inner.tuning.model_id, inner.tuning.quant
            ));
        }
        let coord_id = inner.coordinator_id.clone().unwrap_or_else(|| inner.self_id.clone());
        // The creator used to transition the whole roster to "starting" and
        // return Ok before materialize_engine noticed an empty path. That
        // function could only write to stderr, so the installed UI looked as
        // if Start did nothing and no engine process appeared. Reject while
        // the roster is still idle: the invoke reaches PairingPanel's visible
        // error strip and the Start button remains available after the user
        // puts the selected weights in the model folder.
        if coord_id == inner.self_id && !model_path_ready(&inner.model_path) {
            return Err("[WEIGHTS_NOT_DOWNLOADED] no GGUF file selected".into());
        }
        generation = inner.generation;
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
        approve_coordinator_request(&mut inner, &peer_id)?;
    }
    emit_snapshot(&app);
    Ok(())
}

/// Apply one creator-approved coordinator request.
///
/// This is the security boundary behind the UI.  A fork can remove the
/// confirmation dialog, but it still cannot promote an arbitrary row: the
/// native roster must currently contain an online member with a stable device
/// identity and an outstanding request.  The request is consumed on success,
/// so replaying the command is refused rather than silently reapplying it.
fn approve_coordinator_request(inner: &mut Inner, peer_id: &str) -> Result<(), String> {
    if inner.mode != Mode::Creator {
        return Err("[PAIR_NOT_CREATOR] only the cluster creator can approve a coordinator request".into());
    }
    if inner.phase != "idle" {
        return Err("[PAIR_ALREADY_STARTED] cluster already started".into());
    }
    let Some(peer) = inner.peers.iter().find(|p| p.id == peer_id) else {
        return Err("[PAIR_NO_MEMBER] no such member".into());
    };
    if !peer.online {
        return Err("[PAIR_ROLE_REQUEST_STALE] that machine is no longer online".into());
    }
    if !peer.wants_coordinator {
        return Err("[PAIR_ROLE_NOT_REQUESTED] that machine is not requesting the coordinator role".into());
    }
    if peer.device_id.is_empty() {
        return Err(
            "[PAIR_DEVICE_ID_REQUIRED] update that machine before approving a coordinator request"
                .into(),
        );
    }

    let hostname = peer.hostname.clone();
    let device = device_identity_label(&peer.device_id).unwrap_or_else(|| "unknown".into());
    inner.coordinator_id = Some(peer_id.to_string());
    for p in inner.peers.iter_mut() {
        p.role = if p.id == peer_id { "coordinator" } else { "worker" };
        if p.id == peer_id {
            // One approval consumes exactly the request that authorized it.
            p.wants_coordinator = false;
        }
    }
    audit(
        inner,
        format!(
            "approved coordinator request from {hostname} (device {device}); this machine may now see local prompts in plaintext and control the cluster"
        ),
    );
    Ok(())
}

/// This machine's scheduler-usable memory, from the UI's own probe snapshot.
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
            Some(p) if p.vram_free != vram_free
                || p.ram_free != ram_free
                || p.unified_memory != unified_memory => {
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
        inner.required_model = None;
        inner.challenges.clear();
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
                    let _ = pairing_start(app2.clone(), app2.state(), None, None, None);
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
        });
        assert_eq!(
            usage_cap_args(&t),
            vec!["--max-vram-mb", "8192"]
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
            vec!["--overflow-url", "http://p",
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

    #[test]
    fn launch_time_overflow_refresh_changes_only_the_routing_fields() {
        let mut t = tuning(|t| {
            t.model_id = "qwen3-8b".into();
            t.api_port = 8123;
            t.max_vram_mb = 8192;
            t.overflow_url = "http://old-platform".into();
            t.overflow_key = "old-key".into();
        });
        let latest: OverflowTuning = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "overflowUrl": "http://new-platform",
            "overflowKey": "new-key",
            "overflowWaitS": 7,
            "overflowDailyCapMilli": 4200,
        }))
        .expect("the launch button's camelCase routing payload must deserialize");
        apply_overflow_tuning(&mut t, latest);
        assert_eq!(t.overflow_url, "http://new-platform");
        assert_eq!(t.overflow_key, "new-key");
        assert_eq!(t.overflow_wait_s, 7);
        assert_eq!(t.overflow_daily_cap_milli, 4200);
        assert_eq!(t.model_id, "qwen3-8b", "launch routing must not change the roster model");
        assert_eq!(t.api_port, 8123, "launch routing must not move the local API");
        assert_eq!(t.max_vram_mb, 8192, "launch routing must not alter resource caps");

        // Off is an explicit backend decision. Even stale credentials in the
        // payload must not revive a choice captured when the roster was made.
        let off_with_stale_credentials: OverflowTuning = serde_json::from_value(serde_json::json!({
            "enabled": false,
            "overflowUrl": "http://stale-platform",
            "overflowKey": "stale-key",
            "overflowWaitS": 30,
            "overflowDailyCapMilli": 9999,
        }))
        .expect("the explicit off payload must deserialize");
        apply_overflow_tuning(&mut t, off_with_stale_credentials);
        assert!(t.overflow_url.is_empty());
        assert!(t.overflow_key.is_empty());
        assert_eq!(t.overflow_wait_s, 0);
        assert_eq!(t.overflow_daily_cap_milli, 0);
    }

    /// The bug that shipped as 0.1.4 (2026-08-21): the client handed the
    /// coordinator its own https:// platform URL, and the coordinator — which
    /// has no TLS client and rightly refuses to downgrade — exited with
    /// "refuse: overflow cannot be enabled" on every start. To the user that
    /// was a crash-looping engine and a machine that never joined the cluster.
    /// The argv the coordinator receives must never spell https.
    #[test]
    fn overflow_url_reaches_the_coordinator_as_plaintext() {
        let t = tuning(|t| {
            t.overflow_url = "https://api.idletoken.ai".into();
            t.overflow_key = "sk".into();
        });
        let argv = overflow_args(&t).join(" ");
        assert!(argv.contains("--overflow-url http://api.idletoken.ai:8080"), "{argv}");
        assert!(!argv.contains("https://"), "{argv}");
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

    /// A-P0-4: nothing secret may reach a command line. `ps` and
    /// `/proc/<pid>/cmdline` are readable by every account on the machine;
    /// `/proc/<pid>/environ` is not.
    #[test]
    fn credentials_travel_in_the_environment_not_in_argv() {
        let t = tuning(|t| {
            t.api_token = "tok-should-not-be-in-argv".into();
            t.overflow_url = "http://platform".into();
            t.overflow_key = "sk-should-not-be-in-argv".into();
            t.overflow_wait_s = 5;
        });
        let argv = overflow_args(&t).join(" ");
        assert!(!argv.contains("sk-should-not-be-in-argv"), "{argv}");
        assert!(!argv.contains("tok-should-not-be-in-argv"), "{argv}");
        // The URL is not a secret and stays visible — a log that says which
        // platform this machine borrows from is worth having.
        assert!(argv.contains("--overflow-url http://platform"), "{argv}");

        // ...and they are handed over as the variables the engine actually
        // reads. The names are checked literally: a typo here would not fail
        // to compile, it would silently start an engine with no API token.
        let env = secret_env(&t);
        assert!(env.contains(&("IDLETOKEN_API_TOKEN".into(), "tok-should-not-be-in-argv".into())));
        assert!(env.contains(&("IDLETOKEN_OVERFLOW_KEY".into(), "sk-should-not-be-in-argv".into())));

        // No token configured = no variable at all, so an empty string can
        // never be mistaken for "auth is on with a blank token".
        assert!(secret_env(&Tuning::default()).is_empty());
        // Overflow needs both halves; half a credential is not one.
        assert!(secret_env(&tuning(|t| t.overflow_key = "sk".into())).is_empty());
    }

    // ---- A-P0-3: the beacon must not leak the join code -------------------
    //
    // The beacon used to broadcast `fnv1a(code)` once a second. The reverse
    // verification the audit asks for is below, and it is the whole argument:
    // `legacy_code_hash` is the function that shipped, and a plain search
    // recovers the code from its output. The new packet is asserted to be a
    // function of the session id, the nonce and the port ONLY — so there is
    // nothing on the wire to search against.

    /// The function `IDLETOKEN1` beacons used. Kept here, and only here, so the
    /// claim "the old one was breakable" is demonstrated rather than asserted.
    fn legacy_code_hash(code: &str) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in code.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    const CODE_ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

    #[test]
    fn the_old_beacon_gave_the_code_away() {
        // A real code is 6 characters of this alphabet — 32^6 ≈ 1.07e9, which
        // a laptop grinds through in seconds because FNV-1a is a table hash,
        // not a password hash. Searching the full space would make this test
        // take minutes for no extra information, so it searches the same way
        // over the last three characters with the first three known. What the
        // assertion establishes is the METHOD: given the broadcast digest, the
        // code falls out of an offline search of the code space.
        let secret = "QK7MZ3";
        let seen_on_the_wire = legacy_code_hash(secret);
        let mut recovered = None;
        'outer: for a in CODE_ALPHABET {
            for b in CODE_ALPHABET {
                for c in CODE_ALPHABET {
                    let guess = format!("QK7{}{}{}", *a as char, *b as char, *c as char);
                    if legacy_code_hash(&guess) == seen_on_the_wire {
                        recovered = Some(guess);
                        break 'outer;
                    }
                }
            }
        }
        assert_eq!(
            recovered.as_deref(),
            Some(secret),
            "this is the failure the old beacon had: the digest it broadcast is invertible"
        );
    }

    #[test]
    fn the_new_beacon_carries_nothing_derived_from_the_code() {
        // The property, stated directly: two clusters with different codes and
        // the same session/nonce put identical bytes on the wire. There is
        // therefore no function of the code to invert, whatever the attacker's
        // budget — which is what the test above cannot say about the old one.
        let a = beacon_packet("0123456789abcdef", "fedcba9876543210", ROSTER_PORT);
        let b = beacon_packet("0123456789abcdef", "fedcba9876543210", ROSTER_PORT);
        assert_eq!(a, b);
        assert!(a.starts_with("IDLETOKEN2|"), "{a}");
        assert!(a.ends_with(&format!("|{ROSTER_PORT}")), "{a}");
        // And no packet built by the running code repeats: the nonce moves.
        let live = |session: &str| beacon_packet(session, &random_hex(8), ROSTER_PORT);
        let s = random_hex(8);
        assert_ne!(live(&s), live(&s), "the nonce must change per packet");

        // Belt and braces against a future edit reintroducing a code-derived
        // field: for every code we can think of, the digest the old beacon
        // would have carried appears nowhere in the new packet.
        //
        // Positive control first (CLAUDE.md: a check that cannot fail proves
        // nothing) — the same assertion against the packet the OLD code built
        // has to red, or the loop below is decoration.
        let old_style = format!("IDLETOKEN1|{}|{ROSTER_PORT}", legacy_code_hash("QK7MZ3"));
        assert!(
            old_style.contains(&legacy_code_hash("QK7MZ3")),
            "positive control: this check must be able to catch a leaking beacon"
        );
        for code in ["QK7MZ3", "AAAAAA", "999999", "ACCT-deadbeef"] {
            assert!(
                !a.contains(&legacy_code_hash(code)),
                "the beacon must not carry anything derived from {code}"
            );
        }
    }

    /// A-P0-3, second half. Removing the code's fingerprint from the beacon
    /// means a joiner can no longer tell which broadcaster is its cluster, so it
    /// has to ask each one. It must not ask by handing over the code — that
    /// would trade an offline break for an online one, where anything on the
    /// LAN broadcasts a beacon and collects codes from whoever is joining.
    #[test]
    fn the_creator_proves_the_code_before_the_joiner_does() {
        let code = "QK7MZ3";
        let joiner_nonce = "0f0e0d0c0b0a09080706050403020100";

        // What an impostor can produce with no code: nothing that verifies.
        let expected = pair_proof("creator", code, joiner_nonce);
        assert!(!token_eq(&pair_proof("creator", "AAAAAA", joiner_nonce), &expected));
        assert!(!token_eq("", &expected));
        assert!(!token_eq(&"0".repeat(64), &expected));
        assert!(token_eq(&pair_proof("creator", code, joiner_nonce), &expected));

        // Direction matters: a creator's proof must not be replayable as the
        // joiner's answer, or an eavesdropper could join with what it heard.
        assert_ne!(
            pair_proof("creator", code, joiner_nonce),
            pair_proof("joiner", code, joiner_nonce)
        );
        // And so does the nonce: one exchange proves nothing about the next.
        assert_ne!(pair_proof("joiner", code, "aaaa"), pair_proof("joiner", code, "bbbb"));

        // The proof is not the code, and does not contain it — this is the
        // whole reason the code stops travelling on the wire.
        let p = pair_proof("joiner", code, joiner_nonce);
        assert_eq!(p.len(), 64);
        assert!(!p.contains(code));
        assert!(!p.to_uppercase().contains(code));
    }

    // ---- A-P0-2: roster/leave need the member token ------------------------

    fn joined_member(id: &str, ip: &str, token: &str) -> Peer {
        Peer {
            id: id.into(),
            hostname: id.into(),
            gpu: String::new(),
            role: "worker",
            is_self: false,
            stage: "joined".into(),
            layer_lo: None,
            layer_hi: None,
            online: true,
            last_seen: None,
            hb_secs: 0,
            ip: ip.into(),
            token: token.into(),
            device_id: String::new(),
            wants_coordinator: false,
            vram_free: 0,
            ram_free: 0,
            unified_memory: false,
            model_ready: false,
            model_id: String::new(),
            quant: String::new(),
        }
    }

    fn inner_with(peers: Vec<Peer>) -> Inner {
        let Pairing(m) = Pairing::default();
        Inner { mode: Mode::Creator, peers, ..m.into_inner().unwrap() }
    }

    #[test]
    fn roster_and_leave_require_the_member_token() {
        let inner = inner_with(vec![joined_member("machine-a", "192.168.1.50", "a1b2c3")]);
        let ip = "192.168.1.50";

        // What anybody on the LAN could send before A-P0-2 — and what used to
        // be answered with the whole roster.
        assert_eq!(member_authorized(&inner, &json!({"op": "roster", "id": "machine-a"}), ip), None);
        assert_eq!(member_authorized(&inner, &json!({"op": "roster"}), ip), None);
        // An empty token is not a token; a missing field must not read as one
        // either (no silent downgrade — the request is refused, not trimmed).
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-a", "token": ""}), ip),
            None
        );
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-a", "token": "guess"}), ip),
            None
        );
        // Right token, wrong source: a stolen token does not travel.
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-a", "token": "a1b2c3"}), "192.168.1.99"),
            None
        );
        // Unknown member.
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-z", "token": "a1b2c3"}), ip),
            None
        );
        // The real member's own poll still works.
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-a", "token": "a1b2c3"}), ip),
            Some("machine-a".to_string())
        );
    }

    #[test]
    fn leave_can_only_evict_the_token_holder() {
        // The id the request ASKS for is irrelevant: member_authorized answers
        // with the id the TOKEN belongs to, which is what "leave" then removes.
        // Aiming someone else's machine at the door was the second half of
        // A-P0-2.
        let inner = inner_with(vec![
            joined_member("machine-a", "192.168.1.50", "aaaa"),
            joined_member("machine-b", "192.168.1.51", "bbbb"),
        ]);
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-b", "token": "aaaa"}), "192.168.1.50"),
            None,
            "machine-a's token must not authenticate as machine-b"
        );
        assert_eq!(
            member_authorized(&inner, &json!({"id": "machine-a", "token": "aaaa"}), "192.168.1.50"),
            Some("machine-a".to_string())
        );
    }

    #[test]
    fn member_tokens_are_unpredictable_and_never_serialized() {
        let a = random_hex(16);
        let b = random_hex(16);
        assert_eq!(a.len(), 32, "128 bits, not the 30 of a join code");
        assert_ne!(a, b);
        assert!(!token_eq(&a, &b));
        assert!(token_eq(&a, &a));
        assert!(!token_eq("", ""), "an empty token can never match");

        // The token must not reach the webview snapshot or the roster
        // broadcast — both go through serde, so one #[serde(skip)] covers it,
        // and this is the test that notices if it is ever removed.
        let peer = joined_member("machine-a", "192.168.1.50", "s3cr3t");
        let as_json = serde_json::to_string(&peer).unwrap();
        assert!(!as_json.contains("s3cr3t"), "{as_json}");
        let inner = inner_with(vec![peer]);
        let roster = roster_reply(&inner).to_string();
        assert!(!roster.contains("s3cr3t"), "{roster}");
        let snap = snapshot_json(&inner).to_string();
        assert!(!snap.contains("s3cr3t"), "{snap}");
    }

    #[test]
    fn roster_heartbeat_refreshes_memory_without_erasing_old_clients() {
        let mut peer = joined_member("machine-a", "192.168.1.50", "token");
        peer.vram_free = 1;
        peer.ram_free = 2;

        merge_peer_memory(
            &mut peer,
            &json!({
                "vramFree": 13_u64 << 30,
                "ramFree": 42_u64 << 30,
                "unifiedMemory": true,
            }),
        );
        assert_eq!(peer.vram_free, 13_u64 << 30);
        assert_eq!(peer.ram_free, 42_u64 << 30);
        assert!(peer.unified_memory);

        // Backward compatibility: an older heartbeat has no resource keys.
        // It must not turn a complete pool back into "cannot tell".
        merge_peer_memory(&mut peer, &json!({"op": "roster"}));
        assert_eq!(peer.vram_free, 13_u64 << 30);
        assert_eq!(peer.ram_free, 42_u64 << 30);
        assert!(peer.unified_memory);
    }

    #[test]
    fn model_readiness_is_bound_to_exact_model_and_precision() {
        let mut peer = joined_member("machine-b", "192.168.1.51", "token");
        merge_peer_model(
            &mut peer,
            &json!({"modelId": "qwen3.8-27b", "quant": "Q4_K_XL", "modelReady": true}),
            "qwen3.8-27b",
            "Q2_K_XL",
        );
        assert!(!peer.model_ready, "Q4 must not satisfy a Q2 cluster");
        merge_peer_model(
            &mut peer,
            &json!({"modelId": "qwen3.8-27b", "quant": "Q2_K_XL", "modelReady": true}),
            "qwen3.8-27b",
            "Q2_K_XL",
        );
        assert!(peer.model_ready);
    }

    #[test]
    fn adopting_creator_selection_forgets_a_stale_local_path() {
        let mut inner = inner_with(Vec::new());
        inner.mode = Mode::Joiner;
        inner.self_id = "machine-b".into();
        inner.model_path = "F:/gguf/Qwen-Q4.gguf".into();
        inner.tuning.model_id = "qwen3.8-27b".into();
        inner.tuning.quant = "Q4_K_XL".into();
        apply_roster(
            &mut inner,
            &json!({
                "phase": "idle",
                "coordinatorId": "machine-a",
                "modelId": "qwen3.8-27b",
                "quant": "Q2_K_XL",
                "members": [{
                    "id": "machine-b", "hostname": "machine-b", "gpu": "GPU",
                    "modelReady": false
                }]
            }),
        );
        assert_eq!(inner.tuning.quant, "Q2_K_XL");
        assert!(inner.model_path.is_empty());
    }

    #[test]
    fn cluster_start_waits_for_the_model_not_the_capacity_estimate() {
        let mut creator = joined_member("machine-a", "", "");
        creator.role = "coordinator";
        creator.is_self = true;
        creator.model_ready = true;
        creator.model_id = "qwen3.8-27b".into();
        creator.quant = "Q2_K_XL".into();
        let worker = joined_member("machine-b", "192.168.1.51", "token");
        let mut inner = inner_with(vec![creator, worker]);
        inner.coordinator_id = Some("machine-a".into());
        assert_eq!(snapshot_json(&inner)["canStart"], false);
        inner.peers[1].model_ready = true;
        // Memory reports feed the guidance card and runtime planner. They do
        // not suppress the UI Start control; authoritative admission happens
        // after the click for the exact selected context.
        for peer in &mut inner.peers {
            peer.vram_free = 0;
            peer.ram_free = 0;
        }
        assert_eq!(snapshot_json(&inner)["canStart"], true);
    }

    // ---- CLUS-02/05/06/08/14/19/20: the roster port's admission rules -------

    /// Drive a request through the creator-side protocol core with a chosen
    /// clock, the way a machine on the LAN would.
    fn ask(inner: &mut Inner, req: Value, ip: &str, now: Instant) -> Value {
        roster_request(inner, &req, ip, now)
    }

    /// A creator holding join code ABC234, with one member already joined.
    fn creator_with_code() -> Inner {
        let mut inner = inner_with(Vec::new());
        inner.code = Some("ABC234".into());
        inner.self_id = "creator".into();
        // A cluster is a model AND a precision (the 2026-09-01 admission gate),
        // and an empty precision is not a realistic cluster — the picker always
        // produces one, and an empty peer field is refused as malformed long
        // before the gate. Set both so these tests join the way a machine does.
        inner.tuning.model_id = "deepseek-v4-flash".into();
        inner.tuning.quant = "IQ2_XXS".into();
        inner
    }

    /// Complete a real `hello` + `join` the way a joiner does.
    fn do_join_request(
        inner: &mut Inner,
        host: &str,
        device: &str,
        ip: &str,
        prefer: bool,
        now: Instant,
    ) -> Value {
        // Admission requires this cluster's exact weights (2026-09-01). Every
        // test below is about identity, tokens or admission — not about
        // downloading — so the joiner presents what the cluster asked for. The
        // weight gate has its own tests, which vary these three fields.
        let model = inner.tuning.model_id.clone();
        let quant = inner.tuning.quant.clone();
        do_join_claiming(inner, host, device, ip, prefer, now, &model, &quant, true)
    }

    /// A join that says whatever it likes about its local weights. Only the
    /// weight-gate tests need this; everything else goes through `do_join`.
    #[allow(clippy::too_many_arguments)]
    fn do_join_claiming(
        inner: &mut Inner,
        host: &str,
        device: &str,
        ip: &str,
        prefer: bool,
        now: Instant,
        model: &str,
        quant: &str,
        ready: bool,
    ) -> Value {
        let mine = random_hex(16);
        let hello = ask(inner, json!({"op": "hello", "nonce": mine, "v": PAIR_PROTO_V}), ip, now);
        let nonce = hello["nonce"].as_str().unwrap_or("").to_string();
        let mut req = json!({"op": "join", "proof": pair_proof("joiner", "ABC234", &nonce),
                             "hostname": host, "gpu": "GPU", "prefer": prefer,
                             "modelId": model, "quant": quant, "modelReady": ready});
        if !device.is_empty() {
            req["deviceId"] = json!(device);
        }
        ask(
            inner,
            req,
            ip,
            now,
        )
    }

    fn do_join(inner: &mut Inner, host: &str, device: &str, ip: &str, now: Instant) -> Value {
        do_join_request(inner, host, device, ip, false, now)
    }

    #[test]
    fn a_join_still_works_end_to_end() {
        // The baseline every assertion below leans on. Without it, "refused"
        // could mean the handshake is simply broken.
        let mut inner = creator_with_code();
        let now = Instant::now();
        let r = do_join(&mut inner, "box-a", "dev-a", "192.168.1.50", now);
        assert_eq!(r["ok"], true, "{r}");
        assert!(!r["token"].as_str().unwrap_or("").is_empty(), "a member token is issued");
        assert_eq!(inner.peers.len(), 1);
    }

    /// 2026-09-01: a member must ALREADY hold this cluster's weights.
    ///
    /// Before this, a machine was admitted first and downloaded afterwards, so
    /// a roster could sit for an hour in a state that could not start, and the
    /// act of joining silently began a transfer the size of the model. The
    /// refusal has to carry the model identity: a refused joiner never reaches
    /// the roster, which used to be how it learned what to fetch.
    #[test]
    fn a_machine_without_the_weights_is_refused_and_told_what_to_fetch() {
        let mut inner = creator_with_code();
        inner.tuning.model_id = "deepseek-v4-flash".into();
        inner.tuning.quant = "IQ2_XXS".into();
        let now = Instant::now();

        let r = do_join_claiming(
            &mut inner, "box-a", "dev-a", "192.168.1.50", false, now,
            "deepseek-v4-flash", "IQ2_XXS", false,
        );
        assert_eq!(r["ok"], false, "{r}");
        assert_eq!(r["err"], "model not ready");
        assert_eq!(r["modelId"], "deepseek-v4-flash", "the refusal must name the model");
        assert_eq!(r["quant"], "IQ2_XXS", "and the precision — the UI fetches one exact file");
        assert!(inner.peers.is_empty(), "a refused machine must not appear in the roster");
    }

    /// A cluster is a model AND a precision. Q2 weights cannot serve an
    /// IQ2_XXS cluster, and admitting them only moves the failure to load time.
    #[test]
    fn a_machine_holding_a_different_precision_is_refused_until_it_matches() {
        let mut inner = creator_with_code();
        inner.tuning.model_id = "deepseek-v4-flash".into();
        inner.tuning.quant = "IQ2_XXS".into();
        let now = Instant::now();

        let r = do_join_claiming(
            &mut inner, "box-a", "dev-a", "192.168.1.50", false, now,
            "deepseek-v4-flash", "Q2_K_XL", true,
        );
        assert_eq!(r["ok"], false, "having SOME copy is not having THIS one: {r}");
        assert_eq!(r["quant"], "IQ2_XXS", "the refusal names what the cluster wants, not what we hold");
        assert!(inner.peers.is_empty());

        // The same machine once it holds the right precision. Being refused for
        // weights must not have burned its admission: the code was correct, so
        // this is a member coming back prepared, not an intruder retrying.
        let r = do_join_claiming(
            &mut inner, "box-a", "dev-a", "192.168.1.50", false, now,
            "deepseek-v4-flash", "IQ2_XXS", true,
        );
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(inner.peers.len(), 1);
        assert!(inner.peers[0].model_ready, "and it lands in the roster already ready");
    }

    /// CLUS-06: hostname is a label, not an identity. A second machine that
    /// claims a name already held must not inherit that member's row.
    #[test]
    fn a_different_machine_cannot_take_over_an_existing_members_name() {
        let mut inner = creator_with_code();
        let now = Instant::now();
        do_join(&mut inner, "box-a", "dev-a", "192.168.1.50", now);
        let victim_token = inner.peers[0].token.clone();

        // Same name, same valid join code, different machine.
        let r = do_join(&mut inner, "box-a", "dev-EVIL", "192.168.1.99", now);
        assert_eq!(r["ok"], false, "the impostor must be refused: {r}");
        assert_eq!(inner.peers.len(), 1, "no second row, and no replaced row");
        assert_eq!(inner.peers[0].ip, "192.168.1.50", "the real machine keeps its address");
        assert_eq!(inner.peers[0].token, victim_token, "and its token is not rotated out from under it");
        assert_eq!(inner.peers[0].device_id, "dev-a");

        // The real machine coming back on a new DHCP lease is NOT a conflict.
        let r = do_join(&mut inner, "box-a", "dev-a", "192.168.1.77", now);
        assert_eq!(r["ok"], true, "the same device rejoining must still work: {r}");
        assert_eq!(inner.peers.len(), 1);
        assert_eq!(inner.peers[0].ip, "192.168.1.77");
        assert_ne!(inner.peers[0].token, victim_token, "a rejoin still rotates the token");
    }

    /// CLUS-08 / CHAIN-04: the coordinator is the plaintext window. A joiner may
    /// ASK for the role; only the creator's own action grants it.
    #[test]
    fn a_joiner_cannot_make_itself_the_coordinator() {
        let mut inner = creator_with_code();
        inner.coordinator_id = Some("creator".into());
        let now = Instant::now();
        let mine = random_hex(16);
        let hello = ask(&mut inner, json!({"op": "hello", "nonce": mine, "v": PAIR_PROTO_V}), "192.168.1.50", now);
        let nonce = hello["nonce"].as_str().unwrap().to_string();
        let r = ask(
            &mut inner,
            json!({"op": "join", "proof": pair_proof("joiner", "ABC234", &nonce),
                   "hostname": "box-a", "deviceId": "dev-a", "prefer": true,
                   // This test is about the coordinator role, so the machine
                   // clears the weight gate the ordinary way.
                   "modelId": "deepseek-v4-flash", "quant": "IQ2_XXS", "modelReady": true}),
            "192.168.1.50",
            now,
        );
        assert_eq!(r["ok"], true, "the join itself is legitimate: {r}");
        assert_eq!(
            inner.coordinator_id.as_deref(),
            Some("creator"),
            "asking must not move the plaintext window"
        );
        assert_eq!(inner.peers.iter().find(|p| p.id == "box-a").unwrap().role, "worker");
        let asked = inner.peers.iter().find(|p| p.id == "box-a").unwrap().wants_coordinator;
        assert!(asked, "the request is recorded so the creator can approve it");
        assert!(
            inner.audit.iter().any(|l| l.contains("asked to become the coordinator")),
            "and it leaves a trace: {:?}",
            inner.audit
        );
    }

    #[test]
    fn creator_snapshot_exposes_only_a_local_device_fingerprint_for_review() {
        let mut inner = creator_with_code();
        inner.mode = Mode::Creator;
        let now = Instant::now();
        let r = do_join_request(
            &mut inner,
            "box-a",
            "0123456789abcdef0123456789abcdef",
            "192.168.1.50",
            true,
            now,
        );
        assert_eq!(r["ok"], true, "{r}");

        let snap = snapshot_json(&inner);
        assert_eq!(snap["isCreator"], true);
        assert_eq!(snap["peers"][0]["wantsCoordinator"], true);
        let label = snap["peers"][0]["deviceIdentity"].as_str().unwrap_or("");
        assert!(
            label.len() == 35 && label.chars().all(|c| c.is_ascii_hexdigit() || c == '-'),
            "the review label is a short stable fingerprint: {label:?}"
        );
        assert_ne!(
            label,
            "0123456789abcdef0123456789abcdef",
            "the raw stable id must not enter the webview"
        );
        assert!(
            !snap.to_string().contains("0123456789abcdef0123456789abcdef"),
            "the creator snapshot carries only the digest label: {snap}"
        );

        let wire = roster_reply(&inner);
        assert!(
            wire["members"][0].get("deviceIdentity").is_none()
                && wire["members"][0].get("deviceId").is_none(),
            "stable device identities must not be broadcast to members: {wire}"
        );

        // The identical snapshot function on a joiner fails closed: no creator
        // bit and no stable labels, even if a hostile local mirror somehow
        // contained one.
        inner.mode = Mode::Joiner;
        let joiner_snap = snapshot_json(&inner);
        assert_eq!(joiner_snap["isCreator"], false);
        assert!(joiner_snap["peers"][0].get("deviceIdentity").is_none());
        assert!(joiner_snap["peers"][0].get("wantsCoordinator").is_none());
    }

    #[test]
    fn coordinator_approval_requires_a_live_current_stable_request() {
        let now = Instant::now();

        let mut non_creator = creator_with_code();
        non_creator.mode = Mode::Joiner;
        assert!(
            approve_coordinator_request(&mut non_creator, "box-a")
                .unwrap_err()
                .contains("PAIR_NOT_CREATOR")
        );

        let mut absent = creator_with_code();
        absent.mode = Mode::Creator;
        assert!(
            approve_coordinator_request(&mut absent, "box-a")
                .unwrap_err()
                .contains("PAIR_NO_MEMBER")
        );

        let mut not_requested = creator_with_code();
        not_requested.mode = Mode::Creator;
        do_join(&mut not_requested, "box-a", "dev-a", "192.168.1.50", now);
        assert!(
            approve_coordinator_request(&mut not_requested, "box-a")
                .unwrap_err()
                .contains("PAIR_ROLE_NOT_REQUESTED")
        );

        let mut offline = creator_with_code();
        offline.mode = Mode::Creator;
        do_join_request(&mut offline, "box-a", "dev-a", "192.168.1.50", true, now);
        offline.peers[0].online = false;
        assert!(
            approve_coordinator_request(&mut offline, "box-a")
                .unwrap_err()
                .contains("PAIR_ROLE_REQUEST_STALE")
        );

        let mut legacy = creator_with_code();
        legacy.mode = Mode::Creator;
        do_join_request(&mut legacy, "box-a", "", "192.168.1.50", true, now);
        assert!(
            approve_coordinator_request(&mut legacy, "box-a")
                .unwrap_err()
                .contains("PAIR_DEVICE_ID_REQUIRED")
        );
    }

    #[test]
    fn request_withdrawal_leave_and_approval_are_fail_closed() {
        let now = Instant::now();

        let mut withdrawn = creator_with_code();
        withdrawn.mode = Mode::Creator;
        do_join_request(&mut withdrawn, "box-a", "dev-a", "192.168.1.50", true, now);
        // A same-device rejoin carrying prefer=false is the protocol's request
        // withdrawal.  It must remove the approval authority immediately.
        do_join_request(
            &mut withdrawn,
            "box-a",
            "dev-a",
            "192.168.1.50",
            false,
            now + Duration::from_millis(1),
        );
        assert_eq!(withdrawn.peers[0].wants_coordinator, false);
        assert!(
            approve_coordinator_request(&mut withdrawn, "box-a")
                .unwrap_err()
                .contains("PAIR_ROLE_NOT_REQUESTED")
        );

        let mut left = creator_with_code();
        left.mode = Mode::Creator;
        let joined = do_join_request(&mut left, "box-a", "dev-a", "192.168.1.50", true, now);
        let token = joined["token"].as_str().unwrap_or("").to_string();
        let reply = ask(
            &mut left,
            json!({"op": "leave", "id": "box-a", "token": token}),
            "192.168.1.50",
            now + Duration::from_millis(1),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        assert!(
            approve_coordinator_request(&mut left, "box-a")
                .unwrap_err()
                .contains("PAIR_NO_MEMBER")
        );

        let mut approved = creator_with_code();
        approved.mode = Mode::Creator;
        approved.coordinator_id = Some("creator".into());
        do_join_request(&mut approved, "box-a", "dev-a", "192.168.1.50", true, now);
        approve_coordinator_request(&mut approved, "box-a").unwrap();
        assert_eq!(approved.coordinator_id.as_deref(), Some("box-a"));
        assert_eq!(approved.peers[0].role, "coordinator");
        assert!(!approved.peers[0].wants_coordinator, "approval consumes the request");
        assert!(
            approved
                .audit
                .iter()
                .any(|line| line.contains("approved coordinator request")
                    && line.contains("device ")),
            "the creator's decision leaves a bounded audit trace: {:?}",
            approved.audit
        );
        assert!(
            approve_coordinator_request(&mut approved, "box-a")
                .unwrap_err()
                .contains("PAIR_ROLE_NOT_REQUESTED"),
            "an already-consumed request is not replayable"
        );
    }

    /// CLUS-05: one source spraying `hello` must not evict the nonces other
    /// machines are about to use.
    #[test]
    fn a_hello_flood_cannot_starve_an_honest_joiner() {
        let mut inner = creator_with_code();
        let now = Instant::now();

        // An honest joiner takes a nonce and is briefly interrupted.
        let honest = random_hex(16);
        let hello = ask(&mut inner, json!({"op": "hello", "nonce": honest, "v": PAIR_PROTO_V}), "192.168.1.50", now);
        let honest_nonce = hello["nonce"].as_str().unwrap().to_string();

        // The attacker sprays several times its own quota. Three times is
        // enough to prove the property: the per-source cap is applied on every
        // request, so a source is already held to MAX_CHALLENGES_PER_SOURCE
        // long before the shared list could fill — spraying MAX_CHALLENGES * 3
        // exercised no additional code path and cost this one test hundreds of
        // production-strength (600k-round) proofs, minutes of it in a debug
        // build.
        for i in 0..(MAX_CHALLENGES_PER_SOURCE * 3) {
            let n = format!("{i:032x}");
            ask(&mut inner, json!({"op": "hello", "nonce": n, "v": PAIR_PROTO_V}), "192.168.1.99", now);
        }
        assert!(
            inner.challenges.iter().filter(|c| c.src == "192.168.1.99").count()
                <= MAX_CHALLENGES_PER_SOURCE,
            "a source may only ever hold its own small quota"
        );

        // The honest joiner's nonce is still spendable.
        let r = ask(
            &mut inner,
            json!({"op": "join", "proof": pair_proof("joiner", "ABC234", &honest_nonce),
                   "hostname": "box-a", "deviceId": "dev-a",
                   // This test is about nonce fairness, so the machine clears
                   // the weight gate the ordinary way.
                   "modelId": "deepseek-v4-flash", "quant": "IQ2_XXS", "modelReady": true}),
            "192.168.1.50",
            now,
        );
        assert_eq!(r["ok"], true, "the flood must not have cost the honest joiner its nonce: {r}");
    }

    /// A nonce is handed to one address; another address must not spend it.
    #[test]
    fn a_nonce_issued_to_one_machine_cannot_be_spent_by_another() {
        let mut inner = creator_with_code();
        let now = Instant::now();
        let mine = random_hex(16);
        let hello = ask(&mut inner, json!({"op": "hello", "nonce": mine, "v": PAIR_PROTO_V}), "192.168.1.50", now);
        let nonce = hello["nonce"].as_str().unwrap().to_string();
        let r = ask(
            &mut inner,
            json!({"op": "join", "proof": pair_proof("joiner", "ABC234", &nonce),
                   "hostname": "box-b", "deviceId": "dev-b"}),
            "192.168.1.99",
            now,
        );
        assert_eq!(r["ok"], false, "a stolen nonce is not a join: {r}");
    }

    /// CLUS-02: the client roster face throttles guessing the same way the
    /// engine's join port does, and the curve is the promise to the user.
    #[test]
    fn wrong_codes_get_slower_but_typos_stay_free() {
        assert_eq!(gate_backoff(1), None, "a typo costs nothing");
        assert_eq!(gate_backoff(GATE_FREE_TRIES), None, "the whole allowance is free");
        assert_eq!(gate_backoff(GATE_FREE_TRIES + 1), Some(Duration::from_secs(1)));
        assert_eq!(gate_backoff(GATE_FREE_TRIES + 2), Some(Duration::from_secs(2)));
        assert_eq!(gate_backoff(99), Some(GATE_MAX), "capped: never a permanent lockout");

        let mut gate = LanGate::default();
        let now = Instant::now();
        assert!(gate.admit("10.0.0.5", now).is_ok(), "a fresh source is served");
        for _ in 0..(GATE_FREE_TRIES + 1) {
            gate.note_failure("10.0.0.5", now);
        }
        assert!(gate.admit("10.0.0.5", now).is_err(), "past the allowance it must wait");
        // A different machine is not punished for its neighbour's mistakes.
        assert!(gate.admit("10.0.0.6", now).is_ok(), "the penalty is per source");
        // And it expires rather than latching.
        assert!(
            gate.admit("10.0.0.5", now + GATE_MAX + Duration::from_secs(1)).is_ok(),
            "the block lifts on its own"
        );
    }

    /// CLUS-05: volume alone is refused, even when every request is well formed.
    #[test]
    fn a_request_flood_is_refused_by_rate_not_only_by_failure() {
        let mut gate = LanGate::default();
        let now = Instant::now();
        for _ in 0..GATE_REQ_MAX {
            assert!(gate.admit("10.0.0.7", now).is_ok(), "normal traffic is served");
        }
        assert!(
            gate.admit("10.0.0.7", now).is_err(),
            "past the ceiling the source is refused without parsing anything"
        );
        assert!(
            gate.admit("10.0.0.7", now + GATE_REQ_BLOCK + Duration::from_secs(1)).is_ok(),
            "and it is a rate cap, not a ban"
        );
    }

    /// CLUS-02 distributed-source residual: many addresses, one attacker.
    #[test]
    fn many_addresses_still_hit_a_cluster_wide_ceiling() {
        let mut gate = LanGate::default();
        let now = Instant::now();
        // Each address stays inside its own free allowance, so only the
        // cluster-wide meter can possibly refuse anything here.
        for i in 0..GATE_GLOBAL_MAX_FAILS {
            gate.note_failure(&format!("10.0.{}.{}", i / 250, i % 250), now);
        }
        assert!(
            gate.admit("10.0.9.9", now).is_err(),
            "a hundred fresh addresses do not buy a hundred free allowances"
        );
        assert!(
            gate.admit("10.0.9.9", now + GATE_GLOBAL_BLOCK + Duration::from_secs(1)).is_ok(),
            "the cluster-wide block is a short rate cap, not a lockout"
        );
    }

    /// CLUS-14 / CHAIN-08: peer-chosen strings are gated before they are stored
    /// or echoed. The first case is the payload that rewrites the document
    /// every member's UI parses.
    #[test]
    fn hostile_field_values_are_refused_before_they_reach_the_roster() {
        assert!(!peer_field_ok("a\",\"role\":\"coordinator\",\"x\":\""), "JSON injection");
        assert!(!peer_field_ok("box-a\nJan 01 coord: all clear"), "log forging");
        assert!(!peer_field_ok("back\\slash"));
        assert!(!peer_field_ok(""), "empty is not a name");
        assert!(!peer_field_ok(&"a".repeat(MAX_PEER_FIELD + 1)), "over-long is refused, not clamped");
        assert!(!peer_field_ok("caf\u{e9}"), "non-ASCII is refused");
        assert!(peer_field_ok("box-a"), "and real names still pass");
        assert!(peer_field_ok(&"a".repeat(MAX_PEER_FIELD)), "exactly the limit passes");

        let mut inner = creator_with_code();
        let now = Instant::now();
        let r = do_join(&mut inner, "a\",\"role\":\"coordinator\",\"x\":\"", "dev-x", "192.168.1.50", now);
        assert_eq!(r["ok"], false, "{r}");
        assert!(inner.peers.is_empty(), "nothing hostile reached the roster");
    }

    /// The whole request surface is reachable by anything on the LAN, so it has
    /// to survive rubbish without panicking or half-applying state.
    #[test]
    fn malformed_requests_never_panic_and_never_authorize() {
        let mut inner = creator_with_code();
        let now = Instant::now();
        do_join(&mut inner, "box-a", "dev-a", "192.168.1.50", now);
        let before = inner.peers.len();

        let corpus = vec![
            json!({}),
            json!({"op": null}),
            json!({"op": 12}),
            json!({"op": "roster"}),
            json!({"op": "roster", "id": "box-a"}),
            json!({"op": "roster", "id": "box-a", "token": ""}),
            json!({"op": "leave", "id": "box-a", "token": "guess"}),
            json!({"op": "join"}),
            json!({"op": "join", "proof": ""}),
            json!({"op": "hello"}),
            json!({"op": "hello", "nonce": ""}),
            json!({"op": "hello", "nonce": "x", "v": 1}),
            json!({"op": "hello", "nonce": {"nested": true}, "v": 2}),
            json!({"op": "join", "hostname": 5, "proof": "x"}),
            json!({"op": "join", "proof": "x", "hostname": "box-b", "vramFree": u64::MAX}),
            json!({"op": "\u{0}\u{1}"}),
            json!({"op": "join", "proof": "x", "hostname": "box-b", "hb": u64::MAX}),
        ];
        for req in corpus {
            let r = ask(&mut inner, req.clone(), "192.168.1.99", now);
            assert!(r.is_object(), "every reply is a JSON object: {req}");
            if r["ok"].as_bool() == Some(true) {
                panic!("malformed request was accepted: {req} -> {r}");
            }
        }
        assert_eq!(inner.peers.len(), before, "and none of it changed the roster");
        assert_eq!(inner.peers[0].ip, "192.168.1.50", "the real member is untouched");
    }

    /// CLUS-09: a member's self-reported memory decides whether the cluster
    /// believes the model fits. An impossible claim must not enter that total.
    #[test]
    fn an_impossible_memory_claim_is_refused_not_believed() {
        let mut peer = joined_member("box-a", "192.168.1.50", "t");
        merge_peer_memory(&mut peer, &json!({"vramFree": 24_u64 << 30}));
        assert_eq!(peer.vram_free, 24 << 30, "a real figure is taken");
        merge_peer_memory(&mut peer, &json!({"vramFree": u64::MAX}));
        assert_eq!(peer.vram_free, 24 << 30, "an impossible one is dropped, not clamped in");
    }

    /// CLUS-19 / HOST-15: the roster broadcast carries what collaboration needs
    /// and no more. This is a whitelist, so ADDING a field breaks it on purpose.
    #[test]
    fn the_roster_broadcast_does_not_hand_out_the_home_network_map() {
        let mut inner = creator_with_code();
        let now = Instant::now();
        do_join(&mut inner, "box-a", "dev-a", "192.168.1.50", now);
        let reply = roster_reply(&inner);
        let member = &reply["members"][0];

        let allowed = [
            "id", "hostname", "gpu", "stage", "online", "layerLo", "layerHi",
            "vramFree", "ramFree", "unifiedMemory", "modelReady",
        ];
        for (k, _) in member.as_object().unwrap() {
            assert!(allowed.contains(&k.as_str()), "new roster field {k:?} needs a disclosure decision");
        }
        let body = reply.to_string();
        assert!(!body.contains("192.168.1.50"), "a member's LAN address is not broadcast: {body}");
        assert!(!body.contains("dev-a"), "nor its stable device id: {body}");
        assert!(!body.contains(&inner.peers[0].token), "nor anyone's member token");
    }

    /// The stretched proof is the only thing standing between one unauthenticated
    /// `hello` and the whole 30-bit code space, so its cost is a property.
    #[test]
    fn the_pairing_proof_is_stretched_and_version_tagged() {
        let a = pair_proof("creator", "ABC234", "nonce-1");
        let b = pair_proof("creator", "ABC234", "nonce-2");
        let c = pair_proof("joiner", "ABC234", "nonce-1");
        assert_ne!(a, b, "the nonce binds the proof");
        assert_ne!(a, c, "so does the direction");
        assert_eq!(a, pair_proof("creator", "ABC234", "nonce-1"), "and it is deterministic");

        // v1 was a single SHA-256 of a fixed format. If this ever matches, the
        // oracle is back.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"idletoken-pair-v1|creator|ABC234|nonce-1");
        let v1: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_ne!(a, v1, "the v1 single-hash proof must not be reachable");

        // Cheap enough to join with, expensive enough to be worth doing.
        //
        // Both halves of that are asserted on the CONSTANT, not on a stopwatch.
        // A wall-clock threshold cannot express this reliably: `cargo test`
        // builds without optimisation and runs these in parallel, so the same
        // 600_000 rounds measure ~70 ms in the release build a user runs and
        // ~3.6 s here (measured on the Mac control machine, 2026-08-30). A bound
        // tight enough to catch a regression in one profile is a flaky failure
        // in the other, and a bound loose enough never to flake asserts nothing.
        //
        // The floor is the security property: below this the stretch stops
        // being the thing that makes the ~2^30 code space expensive to sweep
        // offline (see PAIR_PROOF_ROUNDS). The ceiling is the usability
        // property the old comment was reaching for with "a ten-second join
        // should fail" — at the measured 70 ms per 600_000 rounds, 5_000_000
        // rounds is ~580 ms of release CPU, and anything past that starts being
        // felt on every join and every rejoin after a coordinator restart.
        assert!(
            PAIR_PROOF_ROUNDS >= 100_000,
            "the proof must stay stretched: {PAIR_PROOF_ROUNDS} rounds is close enough to \
             the unstretched v1 hash to put the join code back within an easy offline sweep"
        );
        assert!(
            PAIR_PROOF_ROUNDS <= 5_000_000,
            "{PAIR_PROOF_ROUNDS} rounds is roughly {:.1}s of release CPU per join at the \
             measured 70ms/600k; joining must stay interactive",
            PAIR_PROOF_ROUNDS as f64 * 70e-3 / 600_000.0
        );

        // A smoke check only, and deliberately loose enough that it cannot flake
        // under a loaded parallel run: it catches a proof that never terminates
        // or that has become pathologically slow for a reason the round count
        // does not explain. The two assertions above are the real gate.
        let t0 = Instant::now();
        let _ = pair_proof("creator", "ABC234", "timing");
        let smoke_ceiling = if cfg!(debug_assertions) {
            Duration::from_secs(30)
        } else {
            Duration::from_secs(2)
        };
        assert!(
            t0.elapsed() < smoke_ceiling,
            "one proof took {:?} (smoke ceiling {smoke_ceiling:?}) — that is far beyond what \
             {PAIR_PROOF_ROUNDS} rounds should cost, so something other than the round count \
             changed",
            t0.elapsed()
        );
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
