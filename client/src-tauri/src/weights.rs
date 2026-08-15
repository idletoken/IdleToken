//! Weight fetching -- the client-side counterpart of `scripts/model_fetch.sh`.
//!
//! # Why this code lives here rather than in the engine or in that script
//!
//! - **Not the script**: `model_fetch.sh` is bash plus curl. The great majority
//!   of compute nodes run Windows, which has neither bash nor a guaranteed curl.
//!   The product promises "you never touch a command line", yet the only path to
//!   the weights would sit in a script Windows cannot run.
//! - **Not the engine**: the engine's HTTP client is a raw socket, and both
//!   `idletoken_http.h` and `weights.c` state **no TLS** -- by design it speaks
//!   only within the LAN (nodes pull shards from each other over HTTP Range).
//!   huggingface.co and its mirrors are HTTPS.
//!
//! So it lands in the client, where TLS is already available on the Rust side.
//!
//! # Relationship to model_fetch.sh
//!
//! Behaviour is deliberately identical (endpoint preference order, Range resumption,
//! size validation, an actionable reason for each of the three failure classes),
//! so a file the script half-downloaded can be finished by the client and vice
//! versa. **The one intentional divergence**: this downloads to `<file>.part` and
//! renames only on completion. A partial file never occupies the final name --
//! otherwise the next startup would try to load it as complete weights, and that
//! failure lands far from its cause. (The same discipline appears in
//! `ops/db-backup.sh`: rename only after verification.)
//!
//! # Who parses the manifest
//!
//! **The front end.** It already `import`s `models/*.json`, so it resolves `repo`,
//! `file` and the expected byte count and passes them in. The Rust side
//! **deliberately does not maintain a second model registry** -- this repo has
//! already been bitten by hand-maintained copies drifting apart
//! (`model_manifest_check.py` exists for exactly that reason).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::{AppHandle, Emitter};

/// Default endpoint order: the origin first, then the mirror. **Probe, do not
/// guess** -- the same machine may reach HF through a VPN today and not tomorrow,
/// and silently hanging for 30 minutes is the worst possible answer.
const DEFAULT_ENDPOINTS: &[&str] = &["https://huggingface.co", "https://hf-mirror.com"];

/// Downloads in progress: id -> (the `.part` being written, cancellation flag).
static ACTIVE: Mutex<Vec<(String, PathBuf, Arc<AtomicBool>)>> = Mutex::new(Vec::new());

fn register(id: &str, part: PathBuf) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let mut a = ACTIVE.lock().unwrap();
    a.retain(|(k, _, _)| k != id);
    a.push((id.to_string(), part, flag.clone()));
    flag
}

/// Is a download with this id still running? An entry lives until the task
/// actually exits, so "still winding down after a cancel" counts as active —
/// which is exactly what the caller must not race against.
fn is_active(id: &str) -> bool {
    ACTIVE.lock().unwrap().iter().any(|(k, _, _)| k == id)
}

/// Is any download — under ANY id — already writing this `.part`? The per-id
/// guard cannot see a second writer that arrives under a different id (the
/// UI-test oracles run real downloads under their own ids), and two writers on
/// one file is the 2026-08-10 corruption all over again. Best-effort: two
/// spellings of the same directory can slip past a path comparison, but every
/// caller in this codebase builds the path the same way.
fn is_active_part(part: &Path) -> bool {
    ACTIVE.lock().unwrap().iter().any(|(_, p, _)| p == part)
}

fn unregister(id: &str) {
    ACTIVE.lock().unwrap().retain(|(k, _, _)| k != id);
}

/// Complete, partial or absent. The front end uses this to decide what the
/// download button says.
#[derive(serde::Serialize)]
pub struct WeightsState {
    /// Absolute path of the final file (always provided, present or not: the
    /// front end uses it to fill in ggufPath).
    path: String,
    /// The final file exists and is no smaller than the **lower bound** (see the
    /// comment on `expect_bytes`).
    complete: bool,
    /// Bytes already on disk (from the complete file or the .part, whichever exists).
    have_bytes: u64,
}

/// Where to download when the user has not set a directory in settings.
///
/// No `dirs`-style dependency: on all three platforms the home directory is one
/// of these two environment variables, and adding a crate for one line of logic
/// is not worth it (bundle size is a hard constraint). If neither is available we
/// fall back to `models/` under the current directory -- ugly, but better than
/// returning an empty string and writing the download to the filesystem root.
#[tauri::command]
pub fn weights_default_dir() -> String {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    if home.is_empty() {
        return "models".into();
    }
    PathBuf::from(home)
        .join(".idletoken")
        .join("models")
        .to_string_lossy()
        .into_owned()
}

// WARNING: `expect_bytes` is a **lower bound, not the file size**.
//
// It comes from the manifest's `layer_weight_bytes + shared_weight_bytes`, which
// counts **tensor data only** and excludes the GGUF header and metadata (tensor
// directory, kv metadata, tokenizer vocabulary, alignment padding).
// Measured one by one against the real server on 2026-08-08, the declared value
// is invariably **5-11 MB smaller than the actual file** (5.33 MB for DSv4, about
// 11 MB for the Qwen family, whose vocabulary metadata is larger):
//
// ```text
//   qwen3.5-4b   manifest 2,729,969,664   server 2,740,937,888
//   dsv4-flash   manifest 86,714,777,664  server 86,720,111,488
// ```
//
// Hence `>=` here and **not** `==`, and this is not a casually loose comparison:
// **changing it to `==` would make the completeness test permanently false**,
// while looking very much like a bug fix.
//
// Real completeness is guaranteed by the download path, which renames `.part` to
// the final name only after validating against the total length from the server's
// Content-Range. This lower bound only rejects obviously truncated files -- for
// instance one copied in half-finished from elsewhere.
/// Downloads nothing; only answers "how far along is this file on this machine".
#[tauri::command]
pub fn weights_state(dest_dir: String, file: String, expect_bytes: u64) -> WeightsState {
    let final_path = PathBuf::from(&dest_dir).join(&file);
    let part_path = part_of(&final_path);
    let final_len = fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
    let part_len = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    // A `.part` present means last time did not finish (the download path renames
    // only after validation), so it always counts as incomplete.
    // Otherwise: the final file exists and is no smaller than the lower bound.
    // expect_bytes==0 means the manifest does not state it, and then "it exists,
    // so it is complete" is the best judgment we can offer.
    let complete = part_len == 0 && final_len > 0 && (expect_bytes == 0 || final_len >= expect_bytes);
    WeightsState {
        path: final_path.to_string_lossy().into_owned(),
        complete,
        have_bytes: if final_len > 0 { final_len } else { part_len },
    }
}

/// Request cancellation of a download. The `.part` already on disk is kept, and
/// the next attempt resumes from it.
#[tauri::command]
pub fn weights_cancel(id: String) -> bool {
    let a = ACTIVE.lock().unwrap();
    match a.iter().find(|(k, _, _)| *k == id) {
        Some((_, _, flag)) => {
            flag.store(true, Ordering::SeqCst);
            true
        }
        None => false,
    }
}

fn part_of(final_path: &Path) -> PathBuf {
    let mut s = final_path.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// One weight file sitting in the model folder.
#[derive(serde::Serialize)]
pub struct StoredWeights {
    /// File name without the `.part` suffix — i.e. the name the manifest knows,
    /// so the front end can put a model label on it.
    pub file: String,
    pub bytes: u64,
    /// An unfinished download (`.part`). Continuing resumes from these bytes.
    pub partial: bool,
}

/// What is actually on disk in the model folder.
///
/// A **scan**, not a lookup of the models we know about: the folder fills up
/// with precisions you tried once and moved away from, and with the leftovers
/// of a switch. Asking the manifests "is qwen3-8b/Q4_K_M here?" can only ever
/// find what we thought to ask for, which is exactly not the file you are
/// hunting when the disk is full.
#[tauri::command]
pub fn weights_list(dest_dir: String) -> Vec<StoredWeights> {
    let mut out: Vec<StoredWeights> = Vec::new();
    let Ok(rd) = fs::read_dir(&dest_dir) else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let (base, partial) = match name.strip_suffix(".part") {
            Some(b) => (b.to_string(), true),
            None => (name.clone(), false),
        };
        if !base.ends_with(".gguf") {
            continue;
        }
        let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(StoredWeights { file: base, bytes, partial });
    }
    out.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    out
}

/// Delete one model's weights from the folder — the finished file and any
/// leftover `.part` of the same name.
///
/// `file` must be a bare file name. Anything with a separator (or `..`) is
/// refused: this command exists to free disk space in ONE directory, and a
/// caller that can steer it elsewhere turns "delete a model" into "delete any
/// file the app can reach". Non-`.gguf` names are refused for the same reason.
///
/// Returns the bytes freed. A file the OS will not let go of (Windows keeps a
/// loaded model open) comes back as an error the UI can show, rather than a
/// silent no-op that leaves the row on screen.
#[tauri::command]
pub fn weights_delete(dest_dir: String, file: String) -> Result<u64, String> {
    if file.is_empty()
        || file.contains('/')
        || file.contains('\\')
        || file.contains("..")
        || !file.ends_with(".gguf")
    {
        return Err(format!("refusing to delete {file:?}: not a weights file name"));
    }
    let final_path = PathBuf::from(&dest_dir).join(&file);
    let part_path = part_of(&final_path);
    let mut freed = 0u64;
    let mut errs: Vec<String> = Vec::new();
    for p in [final_path, part_path] {
        let Ok(md) = fs::metadata(&p) else { continue };
        let len = md.len();
        match fs::remove_file(&p) {
            Ok(()) => freed += len,
            Err(e) => errs.push(format!("{}: {e}", p.display())),
        }
    }
    if errs.is_empty() {
        Ok(freed)
    } else {
        Err(errs.join("; "))
    }
}

/// Download a set of weights. Progress is pushed to the front end through
/// `weights-fetch` events:
/// `{ id, kind: "probe"|"progress"|"done"|"error", ... }`
///
/// An empty `endpoints` uses the default order; a non-empty one uses **only what
/// was given** (mirroring the script's `HF_ENDPOINT`: when the user names an
/// endpoint, do not go trying others behind their back).
#[tauri::command]
pub async fn weights_fetch(
    app: AppHandle,
    id: String,
    repo: String,
    file: String,
    dest_dir: String,
    expect_bytes: u64,
    endpoints: Vec<String>,
) -> Result<(), String> {
    // One download per id, enforced here. `register` used to just drop the old
    // entry and keep going, so a second call left the first task running —
    // uncancellable (its flag was no longer reachable) and still appending to
    // the SAME .part. Two writers on one file produced a partial longer than the
    // source, which the resume path then blessed as a finished download.
    //
    // This is what "the button gave no feedback" cost: the user pressed it
    // again, and again, and each press added a writer.
    if is_active(&id) {
        return Err(
            "a download is already running (if you just cancelled it, it is still stopping — try again in a moment)"
                .into(),
        );
    }
    // Same file under a different id: still two writers on one .part.
    let part = part_of(&PathBuf::from(&dest_dir).join(&file));
    if is_active_part(&part) {
        return Err(
            "these weights are already being downloaded by another task — wait for it to finish or cancel it first"
                .into(),
        );
    }
    let cancel = register(&id, part);
    let id2 = id.clone();
    let cancel_probe = cancel.clone();
    let r = tauri::async_runtime::spawn_blocking(move || {
        let emit = |v: serde_json::Value| {
            let _ = app.emit("weights-fetch", v);
        };
        let out = fetch_inner(
            &id2, &repo, &file, &dest_dir, expect_bytes, &endpoints, &cancel, &emit,
        );
        match &out {
            Ok(path) => emit(serde_json::json!({
                "id": id2, "kind": "done", "path": path
            })),
            // A user pressing Cancel is not a failure. It used to arrive as
            // `kind: "error"`, so the UI announced "download failed" seconds
            // after the user asked for the stop they got — blaming the product
            // for doing what it was told. The flag is the authority on which
            // one this was; the message text is not (it is also produced by a
            // read that was interrupted for other reasons).
            Err(e) if cancel_probe.load(Ordering::SeqCst) => emit(serde_json::json!({
                "id": id2, "kind": "cancelled", "message": e
            })),
            Err(e) => emit(serde_json::json!({
                "id": id2, "kind": "error", "message": e
            })),
        }
        out.map(|_| ())
    })
    .await;
    unregister(&id);
    match r {
        Ok(inner) => inner,
        Err(e) => Err(format!("download task exited abnormally: {e}")),
    }
}

/// The result of probing one endpoint.
struct Probe {
    endpoint: String,
    total: u64,
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        // Short connect timeout, no overall limit: 80 GiB over gigabit takes tens
        // of minutes, and an overall timeout would be a death sentence for large
        // models. Hangs are handled by the cancel button and the zero-byte stall
        // detection below.
        .connect_timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("failed to initialize the HTTP client: {e}"))
}

/// One endpoint's failure to serve the file.
enum ProbeFail {
    /// Trying other endpoints cannot help: the repo/file name itself is wrong.
    Fatal(String),
    /// This endpoint is out; another may still work.
    Soft(String),
}

/// Ask one endpoint whether it can serve the file, and for the total length.
///
/// It uses `Range: bytes=0-0` rather than HEAD: `resolve/main` is a redirect to a
/// CDN, and some CDNs answer HEAD with 405. Asking for the first byte works on
/// both, and the total length can be read from `Content-Range`. (This matches
/// model_fetch.sh, whose comments give the same reason.)
fn probe_one(c: &reqwest::blocking::Client, ep: &str, repo: &str, file: &str) -> Result<u64, ProbeFail> {
    let url = format!("{}/{}/resolve/main/{}", ep.trim_end_matches('/'), repo, file);
    let resp = match c.get(&url).header("Range", "bytes=0-0").send() {
        Ok(r) => r,
        Err(_) => return Err(ProbeFail::Soft(format!("{ep}: unreachable (blocked or offline)"))),
    };
    let code = resp.status().as_u16();
    match code {
        200 | 206 => {
            // A 206 carries Content-Range: bytes 0-0/<total>; a 200 means the
            // server ignored the Range, and then Content-Length is the full
            // length.
            let total = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.rsplit('/').next().map(|x| x.to_string()))
                .and_then(|s| s.trim().parse::<u64>().ok())
                .or_else(|| resp.content_length())
                .unwrap_or(0);
            Ok(total)
        }
        // A 404 is a configuration error and trying another endpoint will not
        // help -- the repo or file name in the manifest is wrong.
        404 => Err(ProbeFail::Fatal(format!(
            "{ep} does not have this file ({repo}/{file}). The repo or file name in the model manifest is wrong; another mirror will not help."
        ))),
        401 | 403 => Err(ProbeFail::Soft(format!("{ep}: requires sign-in (gated repository)"))),
        _ => Err(ProbeFail::Soft(format!("{ep}: HTTP {code}"))),
    }
}

/// How long an earlier-listed endpoint may stay silent once a later one has
/// already answered. Long enough for an origin that is merely slow to respond;
/// short enough that an origin whose packets are silently dropped (that burns
/// the full 20 s connect timeout) does not hold a working mirror hostage.
const PROBE_GRACE: Duration = Duration::from_secs(3);

/// Probe every endpoint at once and return the earliest-listed one that can
/// serve the file.
///
/// Concurrently, not in order: probing in order meant everyone behind a network
/// that silently drops the first endpoint sat through its whole connect timeout
/// before the next was even tried — on every single download. List order still
/// decides preference (an earlier endpoint that answers wins over a later one),
/// but only within PROBE_GRACE of the first success.
///
/// Which endpoint won is not surfaced in the UI — where the bytes come from is
/// an implementation detail, not something the user is asked to think about.
/// The progress events keep it for diagnostics only.
fn probe(
    c: &reqwest::blocking::Client,
    endpoints: &[String],
    repo: &str,
    file: &str,
    emit: &dyn Fn(serde_json::Value),
    id: &str,
    cancel: &AtomicBool,
) -> Result<Probe, String> {
    emit(serde_json::json!({ "id": id, "kind": "probe" }));
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, ep) in endpoints.iter().enumerate() {
        let tx = tx.clone();
        let c = c.clone();
        let (ep, repo, file) = (ep.clone(), repo.to_string(), file.to_string());
        std::thread::spawn(move || {
            // The receiver may be gone already (probe returned early); the
            // straggler's verdict is then simply dropped.
            let _ = tx.send((i, probe_one(&c, &ep, &repo, &file)));
        });
    }
    drop(tx);

    // The earliest-listed success whose predecessors have all failed. With
    // `forced`, predecessors still pending count as failed — used once their
    // grace has run out.
    let pick = |outcome: &[Option<Result<u64, String>>], forced: bool| -> Option<(usize, u64)> {
        for (i, o) in outcome.iter().enumerate() {
            match o {
                Some(Ok(total)) => return Some((i, *total)),
                Some(Err(_)) => continue,
                None if forced => continue,
                None => return None,
            }
        }
        None
    };

    let mut outcome: Vec<Option<Result<u64, String>>> = vec![None; endpoints.len()];
    let mut first_success: Option<Instant> = None;
    loop {
        // Waits are sliced so a cancel is noticed within ~300 ms even while
        // every endpoint is still sitting in its connect timeout. The cancel
        // flag used to go unchecked until the transfer loop, so cancelling (or
        // switching models, which cancels) during a slow probe held the one
        // download slot hostage for the whole connect timeout.
        const SLICE: Duration = Duration::from_millis(300);
        let msg = loop {
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled (what has been downloaded is kept; the next attempt resumes)".into());
            }
            let wait = match first_success {
                // Nothing usable yet: keep waiting for the next verdict,
                // however long its connect timeout takes.
                None => SLICE,
                // Something usable is in hand: the better-preferred stragglers
                // get what is left of the grace, then we stop waiting for them.
                Some(t0) => SLICE.min(PROBE_GRACE.saturating_sub(t0.elapsed())),
            };
            match rx.recv_timeout(wait) {
                Ok(m) => break Some(m),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break None,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(t0) = first_success {
                        if t0.elapsed() >= PROBE_GRACE {
                            break None;
                        }
                    }
                    // No success yet (or grace remains): keep slicing.
                }
            }
        };
        match msg {
            Some((i, Ok(total))) => {
                outcome[i] = Some(Ok(total));
                first_success.get_or_insert_with(Instant::now);
            }
            Some((_, Err(ProbeFail::Fatal(msg)))) => return Err(msg),
            Some((i, Err(ProbeFail::Soft(msg)))) => outcome[i] = Some(Err(msg)),
            // Grace expired on the stragglers, or every thread has reported.
            None => {
                if let Some((i, total)) = pick(&outcome, true) {
                    return Ok(Probe { endpoint: endpoints[i].clone(), total });
                }
                break;
            }
        }
        if let Some((i, total)) = pick(&outcome, false) {
            return Ok(Probe { endpoint: endpoints[i].clone(), total });
        }
        if outcome.iter().all(|o| o.is_some()) {
            break;
        }
    }
    // Every endpoint failed.
    // "[CODE] detail" (client-error convention, ERROR_KEYS in i18n.ts): the UI
    // renders a localized, actionable sentence — check the network / retry /
    // put a hand-downloaded GGUF in the model folder — with this endpoint list
    // as the detail. (The old text pointed at a "mirror" setting that does not
    // exist.)
    let tried: Vec<String> = outcome.into_iter().flatten().filter_map(|r| r.err()).collect();
    Err(format!("[WEIGHTS_NO_SOURCE] tried: {}", tried.join("; ")))
}

#[allow(clippy::too_many_arguments)]
fn fetch_inner(
    id: &str,
    repo: &str,
    file: &str,
    dest_dir: &str,
    expect_bytes: u64,
    endpoints: &[String],
    cancel: &AtomicBool,
    emit: &dyn Fn(serde_json::Value),
) -> Result<String, String> {
    let dir = PathBuf::from(dest_dir);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create the download directory {dest_dir}: {e}"))?;
    let final_path = dir.join(file);
    let part_path = part_of(&final_path);

    // A complete copy already present is used as is -- downloaded by the script,
    // or finished on a previous run.
    if let Ok(m) = fs::metadata(&final_path) {
        if m.len() > 0 && (expect_bytes == 0 || m.len() >= expect_bytes) {
            return Ok(final_path.to_string_lossy().into_owned());
        }
    }

    let eps: Vec<String> = if endpoints.is_empty() {
        DEFAULT_ENDPOINTS.iter().map(|s| s.to_string()).collect()
    } else {
        endpoints.to_vec()
    };

    let c = client()?;
    let p = probe(&c, &eps, repo, file, emit, id, cancel)?;
    let total = if p.total > 0 { p.total } else { expect_bytes };

    let mut have = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    // LONGER than the source cannot be a prefix of the source: the bytes on disk
    // are not a partial download, they are garbage. This used to fall into the
    // branch below and get **renamed and declared complete** — the client would
    // certify a corrupt file as the model's weights, and the failure surfaced
    // much later as an unexplained load error.
    //
    // Seen for real on 2026-08-10: two concurrent downloads appended to the same
    // .part (see weights_fetch) and left it 517 MB over size. The concurrency is
    // fixed there; this is the second lock on the door, because anything that
    // ends with an over-long .part has the same right answer — throw it away.
    if total > 0 && have > total {
        let _ = fs::remove_file(&part_path);
        emit(serde_json::json!({
            "id": id, "kind": "progress",
            // "[CODE] detail" — localized by the UI (ERROR_KEYS in i18n.ts).
            "note": "[WEIGHTS_PART_OVERRUN] the partial file was longer than the source and could not be a resume point; starting over",
            "have": 0u64, "total": total
        }));
        have = 0;
    }
    if total > 0 && have == total {
        // Exactly the right length: finished last time but never renamed.
        fs::rename(&part_path, &final_path)
            .map_err(|e| format!("rename failed for {}: {e}", final_path.display()))?;
        return Ok(final_path.to_string_lossy().into_owned());
    }

    emit(serde_json::json!({
        "id": id, "kind": "progress", "endpoint": p.endpoint,
        "have": have, "total": total
    }));

    let url = format!(
        "{}/{}/resolve/main/{}",
        p.endpoint.trim_end_matches('/'),
        repo,
        file
    );
    let mut req = c.get(&url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let mut resp = req.send().map_err(|e| format!("download request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download refused: HTTP {}", resp.status().as_u16()));
    }
    // A 200 in response to a Range request means the server does not support
    // resumption and is sending the whole file. We must then write from the
    // start, or the full contents would be appended after the existing partial
    // one, producing a file whose length looks sufficient but whose contents are
    // wrong -- far worse than a failed download.
    if have > 0 && resp.status().as_u16() == 200 {
        emit(serde_json::json!({
            "id": id, "kind": "progress",
            // "[CODE] detail" — localized by the UI (ERROR_KEYS in i18n.ts).
            "note": "[WEIGHTS_NO_RESUME] the server does not support resumption; restarting from the beginning",
            "have": 0u64, "total": total
        }));
        have = 0;
    }

    let mut f = if have > 0 {
        fs::OpenOptions::new()
            .append(true)
            .open(&part_path)
            .map_err(|e| format!("cannot open the partial file {}: {e}", part_path.display()))?
    } else {
        fs::File::create(&part_path)
            .map_err(|e| format!("cannot create the download file {}: {e}", part_path.display()))?
    };

    let mut buf = vec![0u8; 1 << 20]; // 1 MiB
    let mut done = have;
    let mut last_emit = Instant::now();
    let mut last_progress = Instant::now();
    let mut last_done = done;
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = f.flush();
            return Err("cancelled (what has been downloaded is kept; the next attempt resumes)".into());
        }
        let n = resp
            .read(&mut buf)
            .map_err(|e| format!("read interrupted: {e}. What has been downloaded is kept; trying again resumes."))?;
        if n == 0 {
            break;
        }
        f.write_all(&buf[..n])
            .map_err(|e| format!("write to disk failed: {e} (disk full?)"))?;
        done += n as u64;

        // Progress is emitted at most every 400 ms: one event per MiB would push
        // eighty thousand events to the front end over an 80 GiB download, and the
        // UI would be dragged down by its own progress bar.
        if last_emit.elapsed() >= Duration::from_millis(400) {
            emit(serde_json::json!({
                "id": id, "kind": "progress", "have": done, "total": total,
                "endpoint": p.endpoint
            }));
            last_emit = Instant::now();
        }
        // Zero-byte stall detection: the connection is still open but not one byte
        // arrives; past 120 s we treat it as dead. Without this, the symptom is a
        // progress bar frozen at 37% that never reports an error.
        if done != last_done {
            last_done = done;
            last_progress = Instant::now();
        } else if last_progress.elapsed() >= Duration::from_secs(120) {
            return Err("the connection stalled: no data received for two minutes. What has been downloaded is kept; trying again resumes.".into());
        }
    }
    f.flush().map_err(|e| format!("write to disk failed: {e}"))?;
    drop(f);

    // Validation: rename only when the size checks out. Renaming a short file
    // means the next startup loads a partial file as complete weights.
    let got = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    if total > 0 && got < total {
        return Err(format!(
            "file is short: got {got} bytes, the server says it should be {total}. Trying again resumes from where it stopped."
        ));
    }
    fs::rename(&part_path, &final_path)
        .map_err(|e| format!("rename failed for {}: {e}", final_path.display()))?;
    Ok(final_path.to_string_lossy().into_owned())
}

/// Tests for the disk-facing half of "delete a model".
///
/// `weights_delete` takes a file name from the front end and removes it, so the
/// guard on that name is the only thing between a housekeeping feature and an
/// arbitrary-file-delete. It is pure string work plus the filesystem, which
/// means it can be tested here rather than argued about.
#[cfg(test)]
mod delete_tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as O};

    static N: AtomicU32 = AtomicU32::new(0);

    /// A private directory for one test (no rand/time available here).
    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "idletoken-wtest-{}-{}",
            std::process::id(),
            N.fetch_add(1, O::SeqCst)
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, name: &str, len: usize) {
        fs::write(dir.join(name), vec![0u8; len]).unwrap();
    }

    #[test]
    fn deletes_the_file_and_its_leftover_part() {
        let d = tmpdir();
        write(&d, "m.gguf", 10);
        write(&d, "m.gguf.part", 5);
        let freed = weights_delete(d.to_string_lossy().into(), "m.gguf".into()).unwrap();
        assert_eq!(freed, 15, "both files count towards the space freed");
        assert!(!d.join("m.gguf").exists());
        assert!(!d.join("m.gguf.part").exists(), "a stale .part would keep the disk full");
    }

    #[test]
    fn deleting_a_partial_only_download_works() {
        let d = tmpdir();
        write(&d, "m.gguf.part", 7);
        let freed = weights_delete(d.to_string_lossy().into(), "m.gguf".into()).unwrap();
        assert_eq!(freed, 7);
        assert!(!d.join("m.gguf.part").exists());
    }

    #[test]
    fn refuses_to_leave_the_folder() {
        let outside = tmpdir();
        write(&outside, "precious.gguf", 3);
        let inside = tmpdir();
        for name in ["../precious.gguf", "..\\precious.gguf", "sub/precious.gguf", "/tmp/precious.gguf"] {
            let r = weights_delete(inside.to_string_lossy().into(), name.into());
            assert!(r.is_err(), "{name:?} must be refused");
        }
        assert!(outside.join("precious.gguf").exists(), "nothing outside the folder may be touched");
    }

    #[test]
    fn refuses_anything_that_is_not_weights() {
        let d = tmpdir();
        write(&d, "notes.txt", 3);
        assert!(weights_delete(d.to_string_lossy().into(), "notes.txt".into()).is_err());
        assert!(weights_delete(d.to_string_lossy().into(), "".into()).is_err());
        assert!(d.join("notes.txt").exists());
    }

    #[test]
    fn missing_files_are_not_an_error() {
        // The row may already be gone (deleted in another window, or by hand).
        let d = tmpdir();
        assert_eq!(weights_delete(d.to_string_lossy().into(), "nope.gguf".into()).unwrap(), 0);
    }

    #[test]
    fn list_reports_partials_under_their_final_name() {
        let d = tmpdir();
        write(&d, "done.gguf", 100);
        write(&d, "half.gguf.part", 40);
        write(&d, "notes.txt", 5);
        let mut got = weights_list(d.to_string_lossy().into());
        got.sort_by(|a, b| a.file.cmp(&b.file));
        assert_eq!(got.len(), 2, "only weights files are listed");
        assert_eq!(got[0].file, "done.gguf");
        assert!(!got[0].partial);
        assert_eq!(got[0].bytes, 100);
        // The .part suffix is stripped so the front end can look the name up in
        // a manifest — otherwise every unfinished download shows as unknown.
        assert_eq!(got[1].file, "half.gguf");
        assert!(got[1].partial);
        assert_eq!(got[1].bytes, 40);
    }

    #[test]
    fn list_of_a_missing_folder_is_empty_not_a_crash() {
        assert!(weights_list("/nonexistent/idletoken/models".into()).is_empty());
    }
}
