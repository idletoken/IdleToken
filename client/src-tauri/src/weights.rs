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
//! Behaviour is deliberately identical (endpoint probing order, Range resumption,
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

/// Downloads in progress: id -> cancellation flag.
static ACTIVE: Mutex<Vec<(String, Arc<AtomicBool>)>> = Mutex::new(Vec::new());

fn register(id: &str) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let mut a = ACTIVE.lock().unwrap();
    a.retain(|(k, _)| k != id);
    a.push((id.to_string(), flag.clone()));
    flag
}

fn unregister(id: &str) {
    ACTIVE.lock().unwrap().retain(|(k, _)| k != id);
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
    match a.iter().find(|(k, _)| *k == id) {
        Some((_, flag)) => {
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
    let cancel = register(&id);
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

/// Probe the endpoints in order and return the first that can serve the file.
///
/// It uses `Range: bytes=0-0` rather than HEAD: `resolve/main` is a redirect to a
/// CDN, and some CDNs answer HEAD with 405. Asking for the first byte works on
/// both, and the total length can be read from `Content-Range`. (This matches
/// model_fetch.sh, whose comments give the same reason.)
fn probe(
    c: &reqwest::blocking::Client,
    endpoints: &[String],
    repo: &str,
    file: &str,
    emit: &dyn Fn(serde_json::Value),
    id: &str,
) -> Result<Probe, String> {
    let mut tried: Vec<String> = Vec::new();
    for ep in endpoints {
        let url = format!("{}/{}/resolve/main/{}", ep.trim_end_matches('/'), repo, file);
        emit(serde_json::json!({ "id": id, "kind": "probe", "endpoint": ep }));
        let resp = match c.get(&url).header("Range", "bytes=0-0").send() {
            Ok(r) => r,
            Err(_) => {
                tried.push(format!("{ep}: unreachable (blocked or offline)"));
                continue;
            }
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
                return Ok(Probe { endpoint: ep.clone(), total });
            }
            // A 404 is a configuration error and trying another endpoint will not
            // help -- the repo or file name in the manifest is wrong.
            404 => {
                return Err(format!(
                    "{ep} does not have this file ({repo}/{file}). The repo or file name in the model manifest is wrong; another mirror will not help."
                ))
            }
            401 | 403 => tried.push(format!("{ep}: requires sign-in (gated repository)")),
            _ => tried.push(format!("{ep}: HTTP {code}")),
        }
    }
    Err(format!(
        "no usable download source. Tried: {}. Check the network, or set a mirror you can reach in settings.",
        tried.join("；")
    ))
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
    let p = probe(&c, &eps, repo, file, emit, id)?;
    let total = if p.total > 0 { p.total } else { expect_bytes };

    let mut have = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    if total > 0 && have >= total {
        // The .part is already long enough (finished last time but never renamed)
        // -- just complete it.
        fs::rename(&part_path, &final_path)
            .map_err(|e| format!("rename failed for {}: {e}", final_path.display()))?;
        return Ok(final_path.to_string_lossy().into_owned());
    }

    emit(serde_json::json!({
        "id": id, "kind": "progress", "endpoint": p.endpoint,
        "have": have, "total": total, "mirror": p.endpoint != DEFAULT_ENDPOINTS[0]
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
            "id": id, "kind": "progress", "note": "the server does not support resumption; restarting from the beginning",
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
