// Engine sidecar supervisor (acceptance P1 / P4).
//
// The native engine (idletoken-worker / idletoken-coord) is a separate process the
// client launches and babysits (design philosophy 17). This module owns that
// lifecycle: spawn as a Tauri sidecar, capture stdout/stderr into a ring
// buffer, detect exits, restart with exponential backoff, and give up after
// repeated quick crashes. State changes and log lines are pushed to the
// webview as events (`engine:status` / `engine:log`) so the UI stays live
// without polling — the JS side of this contract is
// client/src/provider/engineTauri.ts.
//
// Multi-child: the supervisor keeps one independent slot per role, keyed by
// "coordinator" / "worker". A machine chosen as coordinator runs BOTH (coord +
// a co-located worker) so it also contributes compute (P4); a plain worker
// machine runs one. Each slot has its own backoff/generation; the status the
// UI sees is an aggregate across the live slots.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

const LOG_CAPACITY: usize = 500;
/// Role key for the marketplace platform agent (`idletoken-platform-agent`). It
/// shares the supervisor machinery (backoff, ring buffer, generations) but is
/// EXCLUDED from the aggregate engine status: sharing compute and running the
/// inference engine are independent lifecycles in the UI, and stopping one
/// must not silently stop the other. Its status is pushed separately on the
/// `platform-agent:status` event.
pub const ROLE_PLATFORM_AGENT: &str = "platform-agent";
/// Give up (state = crashed) after this many consecutive quick crashes.
const MAX_QUICK_RESTARTS: u32 = 5;
/// A process that stayed up this long counts as stable: the restart counter
/// resets, so a rare crash after hours of work doesn't inch toward giving up.
const STABLE_UPTIME_MS: u64 = 60_000;

#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineState {
    Stopped,
    Starting,
    Running,
    Restarting,
    Crashed,
}

impl EngineState {
    fn as_str(self) -> &'static str {
        match self {
            EngineState::Stopped => "stopped",
            EngineState::Starting => "starting",
            EngineState::Running => "running",
            EngineState::Restarting => "restarting",
            EngineState::Crashed => "crashed",
        }
    }
    /// Aggregate priority: a more-alarming/active state wins when several slots
    /// disagree, so the single status pill never hides a crashed child.
    fn rank(self) -> u8 {
        match self {
            EngineState::Crashed => 4,
            EngineState::Starting => 3,
            EngineState::Restarting => 2,
            EngineState::Running => 1,
            EngineState::Stopped => 0,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct LogLine {
    ts: u64,
    stream: &'static str,
    line: String,
}

/// Mirror of the TS `EngineStatus` (minus `source`, added JS-side). This is the
/// aggregate over all role slots; `role` becomes e.g. "coordinator+worker".
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    state: EngineState,
    role: Option<String>,
    pid: Option<u32>,
    started_at: Option<u64>,
    restarts: u32,
    last_exit_code: Option<i32>,
}

/// One supervised sidecar process (a role's lifecycle).
struct Slot {
    state: EngineState,
    args: Vec<String>,
    /// Extra env vars for the sidecar (e.g. IDLETOKEN_SHARD_REPO on a remote
    /// worker so it fetches only its layers from the coordinator's repo).
    env: Vec<(String, String)>,
    pid: Option<u32>,
    started_at: Option<u64>,
    restarts: u32,
    last_exit_code: Option<i32>,
    child: Option<CommandChild>,
    /// Bumped on every start/stop touching this slot. A supervisor loop that
    /// observes a generation different from the one it was spawned with is
    /// stale and must exit — this is how "stop" and "user restart" cancel the
    /// old loop without signals.
    generation: u64,
}

impl Slot {
    fn starting(args: Vec<String>, env: Vec<(String, String)>, generation: u64) -> Self {
        Slot {
            state: EngineState::Starting,
            args,
            env,
            pid: None,
            started_at: None,
            restarts: 0,
            last_exit_code: None,
            child: None,
            generation,
        }
    }
}

struct Inner {
    /// role ("coordinator" | "worker") -> its supervised process.
    slots: HashMap<String, Slot>,
    /// Monotonic source for slot generations (shared so stop can invalidate
    /// every live loop with fresh, never-reused values).
    gen_counter: u64,
    logs: VecDeque<LogLine>,
}

pub struct Engine(Mutex<Inner>);

impl Default for Engine {
    fn default() -> Self {
        Engine(Mutex::new(Inner {
            slots: HashMap::new(),
            gen_counter: 0,
            logs: VecDeque::with_capacity(LOG_CAPACITY),
        }))
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Collapse every engine role slot into the single status the UI renders.
/// The platform agent slot is deliberately skipped (see ROLE_PLATFORM_AGENT).
fn aggregate_status(inner: &Inner) -> EngineStatus {
    let mut state = EngineState::Stopped;
    let mut pid: Option<u32> = None;
    let mut restarts = 0u32;
    let mut started_at: Option<u64> = None;
    let mut last_exit_code: Option<i32> = None;
    let mut roles: Vec<&str> = Vec::new();

    for (role, slot) in inner.slots.iter() {
        if role == ROLE_PLATFORM_AGENT {
            continue;
        }
        if slot.state != EngineState::Stopped {
            roles.push(role.as_str());
        }
        if slot.state.rank() > state.rank() {
            state = slot.state;
        }
        restarts = restarts.max(slot.restarts);
        // Prefer the coordinator's pid as the representative one.
        if slot.pid.is_some() && (pid.is_none() || role == "coordinator") {
            pid = slot.pid;
        }
        if let Some(t) = slot.started_at {
            started_at = Some(started_at.map_or(t, |s| s.min(t)));
        }
        if slot.last_exit_code.is_some() {
            last_exit_code = slot.last_exit_code;
        }
    }
    roles.sort_unstable();
    EngineStatus {
        state,
        role: if roles.is_empty() { None } else { Some(roles.join("+")) },
        pid,
        started_at,
        restarts,
        last_exit_code,
    }
}

/// Status of one specific role slot (the platform agent's own pill).
fn slot_status(inner: &Inner, role: &str) -> EngineStatus {
    match inner.slots.get(role) {
        Some(s) => EngineStatus {
            state: s.state,
            role: if s.state != EngineState::Stopped { Some(role.to_string()) } else { None },
            pid: s.pid,
            started_at: s.started_at,
            restarts: s.restarts,
            last_exit_code: s.last_exit_code,
        },
        None => EngineStatus {
            state: EngineState::Stopped,
            role: None,
            pid: None,
            started_at: None,
            restarts: 0,
            last_exit_code: None,
        },
    }
}

fn emit_status(app: &AppHandle) {
    let engine = app.state::<Engine>();
    let (agg, agent) = {
        let inner = engine.0.lock().unwrap();
        (aggregate_status(&inner), slot_status(&inner, ROLE_PLATFORM_AGENT))
    };
    let _ = app.emit("engine:status", &agg);
    let _ = app.emit("platform-agent:status", &agent);
}

fn push_log(app: &AppHandle, role: &str, stream: &'static str, line: String) {
    // Tag the role so a coordinator machine's two children are distinguishable.
    let line = format!("[{role}] {line}");
    // Mirror to the client's own stderr so `idletoken-client` run from a terminal
    // (or a CI harness) shows the engine lifecycle without the UI.
    eprintln!("[engine {stream}] {line}");
    let entry = LogLine { ts: now_ms(), stream, line };
    {
        let engine = app.state::<Engine>();
        let mut inner = engine.0.lock().unwrap();
        if inner.logs.len() >= LOG_CAPACITY {
            inner.logs.pop_front();
        }
        inner.logs.push_back(entry.clone());
    }
    let _ = app.emit("engine:log", &entry);
}

fn sidecar_name(role: &str) -> &'static str {
    // "worker" and "weights" (the repo server) are both the worker binary.
    if role == "coordinator" {
        "idletoken-coord"
    } else if role == ROLE_PLATFORM_AGENT {
        "idletoken-platform-agent"
    } else {
        "idletoken-worker"
    }
}

/// One supervisor loop per (role, user-initiated start). Spawns the sidecar,
/// drains its output, and on exit decides between backoff-restart and giving
/// up. `role` is "coordinator" or "worker".
fn supervise(app: AppHandle, role: String, generation: u64) {
    tauri::async_runtime::spawn(async move {
        loop {
            let (args, extra_env) = {
                let engine = app.state::<Engine>();
                let inner = engine.0.lock().unwrap();
                match inner.slots.get(&role) {
                    Some(s) if s.generation == generation => (s.args.clone(), s.env.clone()),
                    _ => return, // superseded by a stop or a newer start
                }
            };

            let spawned = app
                .shell()
                .sidecar(sidecar_name(&role))
                .map_err(|e| e.to_string())
                .and_then(|mut c| {
                    // Engine self-destructs if this client dies (even SIGKILL)
                    // — see the IDLETOKEN_DIE_WITH_PARENT handling in the engine.
                    // PARENT_PID lets the Windows side watch our process handle
                    // (Linux uses PR_SET_PDEATHSIG and ignores it).
                    c = c
                        .env("IDLETOKEN_DIE_WITH_PARENT", "1")
                        .env("IDLETOKEN_PARENT_PID", std::process::id().to_string());
                    for (k, v) in &extra_env {
                        c = c.env(k, v);
                    }
                    c.args(args).spawn().map_err(|e| e.to_string())
                });

            let mut rx = match spawned {
                Ok((rx, child)) => {
                    let engine = app.state::<Engine>();
                    let mut inner = engine.0.lock().unwrap();
                    match inner.slots.get_mut(&role) {
                        Some(s) if s.generation == generation => {
                            s.pid = Some(child.pid());
                            s.child = Some(child);
                            s.state = EngineState::Running;
                            s.started_at = Some(now_ms());
                        }
                        _ => {
                            let _ = child.kill(); // stopped while we were spawning
                            return;
                        }
                    }
                    drop(inner);
                    emit_status(&app);
                    rx
                }
                Err(e) => {
                    push_log(&app, &role, "stderr", format!("spawn failed: {e}"));
                    if should_retry_after_exit(&app, &role, generation, None).await {
                        continue;
                    }
                    return;
                }
            };

            let mut exit_code: Option<i32> = None;
            while let Some(ev) = rx.recv().await {
                match ev {
                    CommandEvent::Stdout(bytes) => {
                        push_log(&app, &role, "stdout", String::from_utf8_lossy(&bytes).trim_end().to_string());
                    }
                    CommandEvent::Stderr(bytes) => {
                        push_log(&app, &role, "stderr", String::from_utf8_lossy(&bytes).trim_end().to_string());
                    }
                    CommandEvent::Error(e) => {
                        push_log(&app, &role, "stderr", format!("process error: {e}"));
                    }
                    CommandEvent::Terminated(t) => {
                        exit_code = t.code;
                        break;
                    }
                    _ => {}
                }
            }

            if !should_retry_after_exit(&app, &role, generation, exit_code).await {
                return;
            }
        }
    });
}

/// Book-keep a slot's exit and sleep the backoff. Returns false when the loop
/// must end: superseded generation, or too many consecutive quick crashes.
async fn should_retry_after_exit(
    app: &AppHandle,
    role: &str,
    generation: u64,
    code: Option<i32>,
) -> bool {
    let delay_s;
    {
        let engine = app.state::<Engine>();
        let mut inner = engine.0.lock().unwrap();
        let s = match inner.slots.get_mut(role) {
            Some(s) if s.generation == generation => s,
            _ => return false, // user stopped it; the exit is expected
        };
        s.child = None;
        s.pid = None;
        s.last_exit_code = code;
        let uptime = s.started_at.map(|t| now_ms().saturating_sub(t)).unwrap_or(0);
        s.started_at = None;
        if uptime >= STABLE_UPTIME_MS {
            s.restarts = 0;
        }
        if s.restarts >= MAX_QUICK_RESTARTS {
            s.state = EngineState::Crashed;
            drop(inner);
            emit_status(app);
            push_log(
                app,
                role,
                "stderr",
                format!("engine kept crashing (last exit {code:?}); giving up until started again"),
            );
            return false;
        }
        s.restarts += 1;
        s.state = EngineState::Restarting;
        // 2s, 4s, 8s, 16s, 30s
        delay_s = (1u64 << s.restarts.min(5)).min(30);
        drop(inner);
        emit_status(app);
        push_log(app, role, "stderr", format!("engine exited (code {code:?}); restarting in {delay_s}s"));
    }
    tokio::time::sleep(Duration::from_secs(delay_s)).await;
    let engine = app.state::<Engine>();
    let inner = engine.0.lock().unwrap();
    matches!(inner.slots.get(role), Some(s) if s.generation == generation)
}

/// Aggregate lifecycle state as the wire string ("stopped" | "starting" | ...),
/// for the pairing layer to report this machine's progress to the roster.
pub fn current_state_str(app: &AppHandle) -> &'static str {
    let engine = app.state::<Engine>();
    let inner = engine.0.lock().unwrap();
    aggregate_status(&inner).state.as_str()
}

/// Start (or restart) one role's sidecar. Public (not just a command) so the
/// pairing layer can materialize the cluster through the exact same path the UI
/// uses. Idempotent per role: starting a role that is already active is an
/// error, but the coordinator machine can start "coordinator" and "worker"
/// independently.
pub fn start_engine(
    app: &AppHandle,
    role: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
) -> Result<(), String> {
    if role != "worker" && role != "coordinator" && role != "weights" && role != ROLE_PLATFORM_AGENT {
        return Err(format!("unknown role: {role}"));
    }
    let generation;
    {
        let engine = app.state::<Engine>();
        let mut inner = engine.0.lock().unwrap();
        if let Some(s) = inner.slots.get(&role) {
            if matches!(
                s.state,
                EngineState::Starting | EngineState::Running | EngineState::Restarting
            ) {
                return Err(format!("{role} already running"));
            }
        }
        inner.gen_counter += 1;
        generation = inner.gen_counter;
        inner.slots.insert(role.clone(), Slot::starting(args, env, generation));
        drop(inner);
        emit_status(app);
    }
    push_log(app, &role, "stdout", format!("starting {role} sidecar"));
    supervise(app.clone(), role, generation);
    Ok(())
}

/// Stop every slot whose role matches `want` (invalidate its loop with a fresh
/// generation, mark stopped, kill the child).
fn stop_matching(app: &AppHandle, want: impl Fn(&str) -> bool) {
    let mut children: Vec<CommandChild> = Vec::new();
    {
        let engine = app.state::<Engine>();
        let mut inner = engine.0.lock().unwrap();
        // Invalidate each live loop with a fresh generation, then mark stopped.
        let roles: Vec<String> = inner.slots.keys().filter(|r| want(r)).cloned().collect();
        for role in roles {
            inner.gen_counter += 1;
            let g = inner.gen_counter;
            if let Some(s) = inner.slots.get_mut(&role) {
                s.generation = g;
                if let Some(c) = s.child.take() {
                    children.push(c);
                }
                s.state = EngineState::Stopped;
                s.pid = None;
                s.started_at = None;
                s.restarts = 0;
            }
        }
        drop(inner);
        emit_status(app);
    }
    for c in children {
        let _ = c.kill();
    }
}

/// Stop the inference-engine sidecars (worker/coordinator/weights). The
/// platform agent is deliberately NOT touched: leaving the cluster or pressing
/// the engine card's Stop must not silently end compute sharing — in relay
/// mode the agent just waits for a coordinator to come back.
pub fn stop_engine(app: &AppHandle) -> Result<(), String> {
    stop_matching(app, |r| r != ROLE_PLATFORM_AGENT);
    push_log(app, "engine", "stdout", "engine stopped by user".into());
    Ok(())
}

/// Stop everything including the platform agent (app shutdown path).
pub fn stop_all(app: &AppHandle) -> Result<(), String> {
    stop_matching(app, |_| true);
    push_log(app, "engine", "stdout", "all sidecars stopped".into());
    Ok(())
}

#[tauri::command]
pub fn engine_start(
    app: AppHandle,
    role: String,
    args: Option<Vec<String>>,
) -> Result<(), String> {
    start_engine(&app, role, args.unwrap_or_default(), Vec::new())
}

#[tauri::command]
pub fn engine_stop(app: AppHandle) -> Result<(), String> {
    stop_engine(&app)
}

#[tauri::command]
pub fn engine_status(state: State<'_, Engine>) -> EngineStatus {
    aggregate_status(&state.0.lock().unwrap())
}

#[tauri::command]
pub fn engine_logs(state: State<'_, Engine>, max_lines: Option<usize>) -> Vec<LogLine> {
    let inner = state.0.lock().unwrap();
    let n = max_lines.unwrap_or(200).min(LOG_CAPACITY);
    let skip = inner.logs.len().saturating_sub(n);
    inner.logs.iter().skip(skip).cloned().collect()
}

/// Start the marketplace platform agent under the same supervisor as the
/// engine sidecars (crash backoff, log ring, status pushed on the
/// `platform-agent:status` event). Relay mode (integration plan 4.2): the
/// agent dials OUT to the platform (long-poll), so no inbound port is opened
/// on the home side; it self-registers the provider, reconnects on drops, and
/// waits for the local coordinator — starting before the cluster is ready is
/// allowed and honest ("waiting for coord" shows in the log tail).
///
/// A missing `idletoken-platform-agent` sidecar binary fails exactly like a
/// missing engine binary: "spawn failed: …" in the log, backoff, then crashed.
///
/// SECURITY: `jwt` arrives over the local Tauri IPC only and ends up in the
/// child's argv. Do NOT log it — push_log never prints slot args, and nothing
/// here persists it; the frontend passes the live session token on each start.
#[tauri::command]
pub fn platform_agent_start(
    app: AppHandle,
    platform_url: String,
    jwt: String,
    name: String,
    coord_api_port: u16,
) -> Result<(), String> {
    let platform_url = platform_url.trim().to_string();
    if platform_url.is_empty() {
        return Err("platform URL is empty — set it in Settings first".into());
    }
    if jwt.trim().is_empty() {
        return Err("not signed in to the platform (no session token)".into());
    }
    let name = if name.trim().is_empty() { "home".to_string() } else { name.trim().to_string() };
    let args: Vec<String> = vec![
        "--relay".into(),
        "--platform".into(),
        platform_url,
        "--jwt".into(),
        jwt,
        "--name".into(),
        name,
        "--coord".into(),
        format!("http://127.0.0.1:{coord_api_port}"),
    ];
    start_engine(&app, ROLE_PLATFORM_AGENT.into(), args, Vec::new())
}

/// Stop only the platform agent (the engine sidecars keep running).
#[tauri::command]
pub fn platform_agent_stop(app: AppHandle) -> Result<(), String> {
    stop_matching(&app, |r| r == ROLE_PLATFORM_AGENT);
    push_log(&app, ROLE_PLATFORM_AGENT, "stdout", "platform agent stopped by user".into());
    Ok(())
}

/// The platform agent slot's own status (same shape as engine_status).
#[tauri::command]
pub fn platform_agent_status(state: State<'_, Engine>) -> EngineStatus {
    slot_status(&state.0.lock().unwrap(), ROLE_PLATFORM_AGENT)
}

/// "Clear my cache now" (acceptance P5): one-shot `idletoken-worker --kv-clear`.
/// Fails honestly (surfaces in the UI) until the engine implements the flag —
/// the contract is defined here so the engine side has a fixed target.
/// `kv_dir` is settings.kvDir; empty/None = the engine's platform default dir.
#[tauri::command]
pub async fn clear_kv_cache(app: AppHandle, kv_dir: Option<String>) -> Result<(), String> {
    let sidecar = app
        .shell()
        .sidecar("idletoken-worker")
        .map_err(|e| format!("sidecar not found: {e}"))?;
    let mut args: Vec<String> = vec!["--kv-clear".into()];
    if let Some(d) = kv_dir {
        if !d.is_empty() {
            args.push("--kv-dir".into());
            args.push(d);
        }
    }
    let output = sidecar
        .args(args)
        .output()
        .await
        .map_err(|e| format!("failed to run worker: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "kv clear failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}
