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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
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
    /// The file has passed the SHA-256 gate against the hash the caller expects
    /// (or no hash is expected — an unpinned manifest has nothing to verify).
    /// `complete && !verified` means: run `weights_verify` before serving.
    verified: bool,
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
pub fn weights_state(
    dest_dir: String,
    file: String,
    expect_bytes: u64,
    expect_sha256: String,
) -> WeightsState {
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
    // Cheap check only — this command must stay instant, so it consults the
    // marker written by a past verification, never the file contents. A
    // complete-but-unverified file is the caller's cue to run weights_verify.
    let verified = complete
        && (expect_sha256.is_empty()
            || fs::read_to_string(marker_of(&final_path))
                .map(|m| m.trim().eq_ignore_ascii_case(&expect_sha256))
                .unwrap_or(false));
    WeightsState {
        path: final_path.to_string_lossy().into_owned(),
        complete,
        have_bytes: if final_len > 0 { final_len } else { part_len },
        verified,
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

/// Sidecar recording the SHA-256 this exact file was verified against, so the
/// serve path does not have to re-hash tens of gigabytes on every startup.
/// The marker is written only after a full hash matched the manifest; deleting
/// it costs nothing but a one-time re-verification.
fn marker_of(final_path: &Path) -> PathBuf {
    let mut s = final_path.as_os_str().to_os_string();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Stream a file through SHA-256 with progress events and cancellation.
/// Hashing an 80 GiB model takes minutes; doing it silently looks like a hang.
fn file_sha256(
    path: &Path,
    id: &str,
    cancel: &AtomicBool,
    emit: &dyn Fn(serde_json::Value),
) -> Result<String, String> {
    let total = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut f = fs::File::open(path)
        .map_err(|e| format!("cannot open {} for verification: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20]; // 1 MiB, same unit as the download loop
    let mut done = 0u64;
    let mut last_emit = Instant::now();
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err("verification cancelled".into());
        }
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read failed while verifying {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        done += n as u64;
        if last_emit.elapsed() >= Duration::from_millis(400) {
            emit(serde_json::json!({
                "id": id, "kind": "progress",
                "phase": "verifying",
                // "[CODE] detail" — localized by the UI (ERROR_KEYS in i18n.ts).
                "note": "[WEIGHTS_VERIFYING] checking the file hash against the manifest",
                // `have` / `total` preserve the standalone-verify contract.
                // A split-model download additionally rewrites those two to
                // the monotonic whole-model download position; these phase
                // fields keep the actual hash pass visible without making the
                // download bar walk the same bytes a second time.
                "have": done, "total": total,
                "phaseHave": done, "phaseTotal": total
            }));
            last_emit = Instant::now();
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The single gate between "bytes on disk" and "a file this client will serve":
/// when the manifest pins a hash, nothing is promoted to the final name without
/// matching it. A mismatch deletes the bytes — a file that is provably not the
/// curated model has no legitimate next use, and keeping it around is how a
/// corrupt download gets served "just to check something" (decision 2026-08-15:
/// fail red, no automatic retry; recovery is a fresh download).
fn promote_verified(
    part_path: &Path,
    final_path: &Path,
    expect_sha256: &str,
    id: &str,
    cancel: &AtomicBool,
    emit: &dyn Fn(serde_json::Value),
) -> Result<(), String> {
    if !expect_sha256.is_empty() {
        let got = file_sha256(part_path, id, cancel, emit)?;
        if !got.eq_ignore_ascii_case(expect_sha256) {
            let _ = fs::remove_file(part_path);
            return Err(format!(
                "[WEIGHTS_SHA256_MISMATCH] the downloaded file does not match the curated model (got {got}, manifest says {expect_sha256}). The file was deleted; download it again."
            ));
        }
    }
    fs::rename(part_path, final_path)
        .map_err(|e| format!("rename failed for {}: {e}", final_path.display()))?;
    if !expect_sha256.is_empty() {
        // Best effort: a missing marker only costs a one-time re-verification.
        let _ = fs::write(marker_of(final_path), format!("{expect_sha256}\n"));
    }
    Ok(())
}

/// Verify an already-promoted file (downloaded by an older client or by
/// `model_fetch.sh`, which never passes through the promotion gate). Fast path
/// is the marker; the full hash runs once and leaves one behind.
fn ensure_final_verified(
    final_path: &Path,
    expect_sha256: &str,
    id: &str,
    cancel: &AtomicBool,
    emit: &dyn Fn(serde_json::Value),
) -> Result<(), String> {
    if expect_sha256.is_empty() {
        return Ok(());
    }
    let marker = marker_of(final_path);
    if let Ok(m) = fs::read_to_string(&marker) {
        if m.trim().eq_ignore_ascii_case(expect_sha256) {
            return Ok(());
        }
        // Marker from another manifest revision: stale, re-verify in full.
    }
    let got = file_sha256(final_path, id, cancel, emit)?;
    if !got.eq_ignore_ascii_case(expect_sha256) {
        let _ = fs::remove_file(final_path);
        let _ = fs::remove_file(&marker);
        return Err(format!(
            "[WEIGHTS_SHA256_MISMATCH] the file on disk does not match the curated model (got {got}, manifest says {expect_sha256}). The file was deleted; download it again."
        ));
    }
    let _ = fs::write(&marker, format!("{expect_sha256}\n"));
    Ok(())
}

/// One logical model sitting in the model folder. A split GGUF is one row,
/// even though it occupies several files in a repository subdirectory.
#[derive(serde::Serialize)]
pub struct StoredWeights {
    /// Primary GGUF path, relative to the configured model folder and using
    /// forward slashes. For a split set this is its `00001-of-N` member.
    pub file: String,
    /// Every GGUF member actually present (finished or `.part`), also relative
    /// to the model folder. The delete command receives this exact allow-list.
    pub files: Vec<String>,
    pub bytes: u64,
    /// An unfinished `.part`, or a split set with one or more members missing.
    pub partial: bool,
}

#[derive(Default)]
struct StoredWeightGroup {
    file: String,
    files: BTreeSet<String>,
    bytes: u64,
    has_partial: bool,
    split_total: u32,
    split_parts: BTreeSet<u32>,
}

#[cfg(windows)]
fn metadata_is_link(md: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
    md.file_type().is_symlink()
        || md.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_link(md: &fs::Metadata) -> bool {
    md.file_type().is_symlink()
}

fn relative_for_ui(path: &Path) -> String {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Parse `name-00002-of-00003.gguf` without assuming a particular model,
/// quantization, number of digits, or repository directory. The returned key
/// names part one so the front end can resolve the set against its manifest.
fn split_gguf_key(rel: &str) -> Option<(String, u32, u32)> {
    let (parent, leaf) = rel.rsplit_once('/').unwrap_or(("", rel));
    let body = leaf.strip_suffix(".gguf")?;
    let (left, total_s) = body.rsplit_once("-of-")?;
    let (prefix, part_s) = left.rsplit_once('-')?;
    if part_s.is_empty()
        || total_s.is_empty()
        || !part_s.bytes().all(|b| b.is_ascii_digit())
        || !total_s.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let part = part_s.parse::<u32>().ok()?;
    let total = total_s.parse::<u32>().ok()?;
    if part == 0 || total == 0 || part > total {
        return None;
    }
    let first = format!("{prefix}-{:0width$}-of-{total_s}.gguf", 1, width = part_s.len());
    let key = if parent.is_empty() {
        first
    } else {
        format!("{parent}/{first}")
    };
    Some((key, part, total))
}

fn collect_weight_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, u64, bool)>) {
    // Model repositories use one quantization directory. The bound is only a
    // defence against a pathological directory tree; it is not a product
    // assumption about a particular model layout.
    if depth > 32 {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let Ok(md) = fs::symlink_metadata(&path) else { continue };
        // Do not follow a symlink/junction placed inside the model folder. The
        // configured folder itself may be a junction (a normal Windows disk
        // migration), but a nested reparse point could escape the delete root
        // or create a recursive scan.
        if metadata_is_link(&md) {
            continue;
        }
        if md.is_dir() {
            collect_weight_files(root, &path, depth + 1, out);
            continue;
        }
        if !md.is_file() {
            continue;
        }
        let Ok(rel_path) = path.strip_prefix(root) else { continue };
        let rel = relative_for_ui(rel_path);
        let (base, partial) = match rel.strip_suffix(".part") {
            Some(b) => (b.to_string(), true),
            None => (rel, false),
        };
        if !base.ends_with(".gguf") {
            continue;
        }
        out.push((base, md.len(), partial));
    }
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
    let root = PathBuf::from(&dest_dir);
    let mut found = Vec::new();
    collect_weight_files(&root, &root, 0, &mut found);

    let mut groups: BTreeMap<String, StoredWeightGroup> = BTreeMap::new();
    for (base, bytes, partial) in found {
        let split = split_gguf_key(&base);
        let key = split.as_ref().map(|v| v.0.clone()).unwrap_or_else(|| base.clone());
        let g = groups.entry(key.clone()).or_insert_with(|| StoredWeightGroup {
            file: key,
            ..StoredWeightGroup::default()
        });
        g.files.insert(base);
        g.bytes = g.bytes.saturating_add(bytes);
        g.has_partial |= partial;
        if let Some((_, part, total)) = split {
            g.split_total = total;
            g.split_parts.insert(part);
        }
    }

    let mut out: Vec<StoredWeights> = groups
        .into_values()
        .map(|g| {
            let partial = g.has_partial
                || (g.split_total > 0 && g.split_parts.len() != g.split_total as usize);
            StoredWeights {
                file: g.file,
                files: g.files.into_iter().collect(),
                bytes: g.bytes,
                partial,
            }
        })
        .collect();
    out.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    out
}

fn validate_weight_relative(file: &str) -> Result<PathBuf, String> {
    // The scanner emits this portable form on every OS. Keeping one accepted
    // separator makes the traversal rules identical in Windows and Unix tests.
    if file.is_empty() || file.contains('\\') || file.starts_with('/') {
        return Err(format!("refusing to delete {file:?}: not a relative weights path"));
    }
    let pieces: Vec<&str> = file.split('/').collect();
    if pieces.iter().any(|p| p.is_empty() || *p == "." || *p == ".." || p.contains(':'))
        || !pieces.last().is_some_and(|p| p.ends_with(".gguf"))
    {
        return Err(format!("refusing to delete {file:?}: not a weights path"));
    }
    let path = PathBuf::from(file);
    if path.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(format!("refusing to delete {file:?}: path leaves the model folder"));
    }
    Ok(path)
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// Delete one logical model's weights from the folder. `files` is the exact
/// relative allow-list returned by `weights_list`; split sets are therefore
/// removed as one UI operation without teaching this disk-facing layer about
/// the curated manifest registry.
///
/// Returns the bytes freed. A file the OS will not let go of (Windows keeps a
/// loaded model open) comes back as an error the UI can show, rather than a
/// silent no-op that leaves the row on screen.
#[tauri::command]
pub fn weights_delete(dest_dir: String, files: Vec<String>) -> Result<u64, String> {
    if files.is_empty() {
        return Err("refusing to delete an empty weights set".into());
    }
    let root = fs::canonicalize(&dest_dir)
        .map_err(|e| format!("cannot open model folder {dest_dir}: {e}"))?;
    let mut targets = BTreeSet::new();
    for file in &files {
        let rel = validate_weight_relative(file)?;
        let mut current = root.clone();
        for component in rel.components() {
            let Component::Normal(piece) = component else { unreachable!() };
            current.push(piece);
            if let Ok(md) = fs::symlink_metadata(&current) {
                if metadata_is_link(&md) {
                    return Err(format!(
                        "refusing to delete {file:?}: a symlink or junction leaves the model folder"
                    ));
                }
            }
        }
        targets.insert(current);
    }

    let mut freed = 0u64;
    let mut errs: Vec<String> = Vec::new();
    let mut parents = BTreeSet::new();
    for final_path in targets {
        if let Some(parent) = final_path.parent() {
            parents.insert(parent.to_path_buf());
        }
        // Verification/index sidecars belong to the same exact GGUF. Leaving
        // them behind both wastes space and can bless or describe a later file
        // that the sidecar never saw.
        for p in [
            final_path.clone(),
            part_of(&final_path),
            marker_of(&final_path),
            append_suffix(&final_path, ".idx"),
            append_suffix(&final_path, ".idx.gen"),
        ] {
            let Ok(md) = fs::symlink_metadata(&p) else { continue };
            if metadata_is_link(&md) || !md.is_file() {
                errs.push(format!("{}: refusing a non-regular file", p.display()));
                continue;
            }
            let len = md.len();
            match fs::remove_file(&p) {
                Ok(()) => freed = freed.saturating_add(len),
                Err(e) => errs.push(format!("{}: {e}", p.display())),
            }
        }
    }
    // Remove only newly-empty repository subdirectories. The root is a user
    // setting and must remain even after its last model is deleted.
    let mut parents: Vec<PathBuf> = parents.into_iter().collect();
    parents.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for mut p in parents {
        while p != root {
            if fs::remove_dir(&p).is_err() {
                break;
            }
            let Some(parent) = p.parent() else { break };
            p = parent.to_path_buf();
        }
    }
    if errs.is_empty() {
        Ok(freed)
    } else {
        Err(errs.join("; "))
    }
}

/// Convert one part's progress into the whole split model's progress story.
///
/// Download bytes are offset by the parts already complete. Verification is
/// different: the current part is already fully downloaded, and hashing merely
/// reads it back from local disk. Keep the whole-model download bar parked at
/// the end of that part while `phaseHave` / `phaseTotal` report the hash pass.
fn whole_model_progress(
    mut v: serde_json::Value,
    base: u64,
    declared_total: u64,
    part: u64,
    parts: u64,
) -> serde_json::Value {
    if let Some(o) = v.as_object_mut() {
        let verifying = o.get("phase").and_then(|x| x.as_str()) == Some("verifying");
        if verifying {
            if let Some(part_total) = o.get("phaseTotal").and_then(|x| x.as_u64()) {
                o.insert("have".into(), base.saturating_add(part_total).into());
            }
        } else if let Some(h) = o.get("have").and_then(|x| x.as_u64()) {
            o.insert("have".into(), base.saturating_add(h).into());
        }
        if declared_total > 0 {
            o.insert("total".into(), declared_total.into());
        }
        if parts > 1 {
            o.insert("part".into(), part.into());
            o.insert("parts".into(), parts.into());
        }
    }
    v
}

/// Download a set of weights. Progress is pushed to the front end through
/// `weights-fetch` events:
/// `{ id, kind: "probe"|"progress"|"done"|"error", ... }`
///
/// An empty `endpoints` uses the default order; a non-empty one uses **only what
/// was given** (mirroring the script's `HF_ENDPOINT`: when the user names an
/// endpoint, do not go trying others behind their back).
/// `parts` carries a split GGUF's extra files (2026-08-15), in order, each with
/// its own pinned hash; empty means a single-file model — the one-part case of
/// the same loop. The parts are fetched one after another rather than in
/// parallel: the bottleneck is the link, not the server, and N concurrent
/// streams would turn one resumable transfer into N racing ones, with N times
/// the ways to leave half-written files behind. llama.cpp opens part 1 and
/// finds the rest by name in the same directory, so all that matters is that
/// every part lands complete and verified.
#[tauri::command]
pub async fn weights_fetch(
    app: AppHandle,
    id: String,
    repo: String,
    file: String,
    dest_dir: String,
    expect_bytes: u64,
    expect_sha256: String,
    revision: String,
    endpoints: Vec<String>,
    parts: Vec<SplitPart>,
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
        let raw_emit = move |v: serde_json::Value| {
            let _ = app.emit("weights-fetch", v);
        };
        // One transfer per part, but ONE progress story for the whole model:
        // the row must show "412 GB of 434 GB", not part 7 restarting at zero
        // eleven times. Each part's `have`/`total` is offset by the bytes the
        // finished parts already contributed; everything else passes through.
        let all: Vec<(String, u64, String)> = std::iter::once((
            file.clone(),
            expect_bytes,
            expect_sha256.clone(),
        ))
        .chain(parts.iter().map(|p| (p.file.clone(), p.bytes, p.sha256.clone())))
        .collect();
        // The denominator is the sum of what the manifest declares. A part with
        // an unknown size contributes its server-reported length once it starts,
        // so the total can only get more accurate, never wrong-and-stuck.
        let declared_total: u64 = all.iter().map(|(_, b, _)| *b).sum();
        let done_before = std::sync::Arc::new(Mutex::new(0u64));
        let mut out: Result<String, String> = Err("no parts to download".into());
        for (idx, (pfile, pbytes, psha)) in all.iter().enumerate() {
            let base = *done_before.lock().unwrap();
            let emit = |v: serde_json::Value| {
                raw_emit(whole_model_progress(
                    v,
                    base,
                    declared_total,
                    idx as u64 + 1,
                    all.len() as u64,
                ));
            };
            out = fetch_inner(
                &id2, &repo, pfile, &dest_dir, *pbytes, psha, &revision, &endpoints,
                &cancel, &emit,
            );
            match &out {
                // Only the FIRST part's path is reported as the result: that is
                // the file the engine is handed, and llama.cpp finds the rest by
                // name in the same directory.
                Ok(_) => {
                    let landed = fs::metadata(PathBuf::from(&dest_dir).join(pfile))
                        .map(|m| m.len())
                        .unwrap_or(*pbytes);
                    *done_before.lock().unwrap() = base + landed;
                }
                // A failed part stops the whole model: the parts already on disk
                // are kept, so pressing Download again resumes at this one.
                Err(_) => break,
            }
        }
        let out = out.map(|_| {
            PathBuf::from(&dest_dir)
                .join(&file)
                .to_string_lossy()
                .into_owned()
        });
        let emit = raw_emit;
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

/// One extra file of a split GGUF (`*-00002-of-00006.gguf` and friends).
#[derive(Clone, serde::Deserialize)]
pub struct SplitPart {
    pub file: String,
    /// Tensor bytes this part is expected to carry; 0 = unknown, the server's
    /// length then decides (same rule as `expect_bytes`).
    #[serde(default)]
    pub bytes: u64,
    /// SHA-256 for this part. "" = unpinned; the gate then has nothing to check.
    #[serde(default)]
    pub sha256: String,
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
fn probe_one(
    c: &reqwest::blocking::Client,
    ep: &str,
    repo: &str,
    file: &str,
    revision: &str,
) -> Result<u64, ProbeFail> {
    let url = format!("{}/{}/resolve/{}/{}", ep.trim_end_matches('/'), repo, revision, file);
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
    revision: &str,
    emit: &dyn Fn(serde_json::Value),
    id: &str,
    cancel: &AtomicBool,
) -> Result<Probe, String> {
    emit(serde_json::json!({ "id": id, "kind": "probe", "phase": "probing" }));
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, ep) in endpoints.iter().enumerate() {
        let tx = tx.clone();
        let c = c.clone();
        let (ep, repo, file, revision) =
            (ep.clone(), repo.to_string(), file.to_string(), revision.to_string());
        std::thread::spawn(move || {
            // The receiver may be gone already (probe returned early); the
            // straggler's verdict is then simply dropped.
            let _ = tx.send((i, probe_one(&c, &ep, &repo, &file, &revision)));
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
    expect_sha256: &str,
    revision: &str,
    endpoints: &[String],
    cancel: &AtomicBool,
    emit: &dyn Fn(serde_json::Value),
) -> Result<String, String> {
    // Empty = the manifest pins no revision (curation gap): fall back to the
    // moving branch rather than refusing to download. The hash gate is the
    // authority on content either way; the revision only removes the window
    // where a force-push serves different bytes under the same name.
    let revision = if revision.is_empty() { "main" } else { revision };
    let dir = PathBuf::from(dest_dir);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create the download directory {dest_dir}: {e}"))?;
    let final_path = dir.join(file);
    let part_path = part_of(&final_path);

    // A complete copy already present is used -- downloaded by the script, or
    // finished on a previous run -- but only once it passes the hash gate
    // (script downloads never went through the promotion gate below).
    if let Ok(m) = fs::metadata(&final_path) {
        if m.len() > 0 && (expect_bytes == 0 || m.len() >= expect_bytes) {
            ensure_final_verified(&final_path, expect_sha256, id, cancel, emit)?;
            return Ok(final_path.to_string_lossy().into_owned());
        }
    }

    let eps: Vec<String> = if endpoints.is_empty() {
        DEFAULT_ENDPOINTS.iter().map(|s| s.to_string()).collect()
    } else {
        endpoints.to_vec()
    };

    let c = client()?;
    let p = probe(&c, &eps, repo, file, revision, emit, id, cancel)?;
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
            "id": id, "kind": "progress", "phase": "downloading",
            // "[CODE] detail" — localized by the UI (ERROR_KEYS in i18n.ts).
            "note": "[WEIGHTS_PART_OVERRUN] the partial file was longer than the source and could not be a resume point; starting over",
            "have": 0u64, "total": total
        }));
        have = 0;
    }
    if total > 0 && have == total {
        // Exactly the right length: finished last time but never renamed.
        // Length is necessary, not sufficient — the hash gate still applies.
        promote_verified(&part_path, &final_path, expect_sha256, id, cancel, emit)?;
        return Ok(final_path.to_string_lossy().into_owned());
    }

    emit(serde_json::json!({
        "id": id, "kind": "progress", "phase": "downloading", "endpoint": p.endpoint,
        "have": have, "total": total
    }));

    let url = format!(
        "{}/{}/resolve/{}/{}",
        p.endpoint.trim_end_matches('/'),
        repo,
        revision,
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
            "id": id, "kind": "progress", "phase": "downloading",
            // "[CODE] detail" — localized by the UI (ERROR_KEYS in i18n.ts).
            "note": "[WEIGHTS_NO_RESUME] the server does not support resumption; restarting from the beginning",
            "have": 0u64, "total": total
        }));
        have = 0;
    }

    // Split-GGUF variants live in a SUBDIRECTORY of the model folder (the
    // manifest's file is e.g. "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"),
    // and File::create does not create parents — so the one variant stored
    // under a subdir failed on every machine with "cannot create the download
    // file (os error 2)" while every flat-named quant downloaded fine
    // (measured 2026-08-24, BF16 on the DGX).
    if let Some(parent) = part_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create the download directory {}: {e}", parent.display()))?;
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
                "id": id, "kind": "progress", "phase": "downloading",
                "have": done, "total": total,
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
    promote_verified(&part_path, &final_path, expect_sha256, id, cancel, emit)?;
    Ok(final_path.to_string_lossy().into_owned())
}

/// Verify an already-downloaded file against the manifest hash without
/// downloading anything. This is the serve-path gate for files that predate
/// the promotion gate or came from `model_fetch.sh`. Progress arrives on the
/// same `weights-fetch` channel (kind "progress", note `[WEIGHTS_VERIFYING]`).
/// A mismatch deletes the file and rejects — same contract as the download
/// path: red, no automatic retry, recovery is a fresh download.
#[tauri::command]
pub async fn weights_verify(
    app: AppHandle,
    id: String,
    dest_dir: String,
    file: String,
    expect_sha256: String,
) -> Result<(), String> {
    let final_path = PathBuf::from(&dest_dir).join(&file);
    // Same collision rules as a download: one task per id, one task per file
    // (a verify racing a download of the same file would hash a moving target).
    if is_active(&id) {
        return Err("a task with this id is already running".into());
    }
    let part = part_of(&final_path);
    if is_active_part(&part) {
        return Err("this file is being downloaded — wait for the download to finish".into());
    }
    let cancel = register(&id, part);
    let id2 = id.clone();
    let r = tauri::async_runtime::spawn_blocking(move || {
        let emit = |v: serde_json::Value| {
            let _ = app.emit("weights-fetch", v);
        };
        let out = ensure_final_verified(&final_path, &expect_sha256, &id2, &cancel, &emit);
        match &out {
            Ok(()) => emit(serde_json::json!({
                "id": id2, "kind": "done",
                "path": final_path.to_string_lossy().into_owned()
            })),
            Err(e) if cancel.load(Ordering::SeqCst) => emit(serde_json::json!({
                "id": id2, "kind": "cancelled", "message": e
            })),
            Err(e) => emit(serde_json::json!({
                "id": id2, "kind": "error", "message": e
            })),
        }
        out
    })
    .await;
    unregister(&id);
    match r {
        Ok(inner) => inner,
        Err(e) => Err(format!("verification task exited abnormally: {e}")),
    }
}

/// Regression tests for split-model progress. A hash pass rereads bytes that
/// have already downloaded, so it gets its own counters and never rewinds the
/// whole-model transfer position.
#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn download_progress_is_offset_by_completed_parts() {
        let got = whole_model_progress(
            serde_json::json!({
                "kind": "progress", "phase": "downloading",
                "have": 20u64, "total": 50u64
            }),
            100,
            300,
            2,
            3,
        );
        assert_eq!(got["have"].as_u64(), Some(120));
        assert_eq!(got["total"].as_u64(), Some(300));
        assert_eq!(got["part"].as_u64(), Some(2));
        assert_eq!(got["parts"].as_u64(), Some(3));
    }

    #[test]
    fn verification_keeps_download_position_at_end_of_current_part() {
        let got = whole_model_progress(
            serde_json::json!({
                "kind": "progress", "phase": "verifying",
                "have": 7u64, "total": 50u64,
                "phaseHave": 7u64, "phaseTotal": 50u64
            }),
            100,
            300,
            2,
            3,
        );
        assert_eq!(got["have"].as_u64(), Some(150));
        assert_eq!(got["total"].as_u64(), Some(300));
        assert_eq!(got["phaseHave"].as_u64(), Some(7));
        assert_eq!(got["phaseTotal"].as_u64(), Some(50));
    }
}

/// Tests for the disk-facing half of "delete a model".
///
/// `weights_delete` takes relative paths from the front end and removes them,
/// so the guard on those paths is the only thing between a housekeeping feature
/// and an arbitrary-file-delete. It is pure path work plus the filesystem,
/// which means it can be tested here rather than argued about.
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
        let freed = weights_delete(d.to_string_lossy().into(), vec!["m.gguf".into()]).unwrap();
        assert_eq!(freed, 15, "both files count towards the space freed");
        assert!(!d.join("m.gguf").exists());
        assert!(!d.join("m.gguf.part").exists(), "a stale .part would keep the disk full");
    }

    #[test]
    fn deleting_a_partial_only_download_works() {
        let d = tmpdir();
        write(&d, "m.gguf.part", 7);
        let freed = weights_delete(d.to_string_lossy().into(), vec!["m.gguf".into()]).unwrap();
        assert_eq!(freed, 7);
        assert!(!d.join("m.gguf.part").exists());
    }

    #[test]
    fn refuses_to_leave_the_folder() {
        let outside = tmpdir();
        write(&outside, "precious.gguf", 3);
        let inside = tmpdir();
        for name in [
            "../precious.gguf",
            "..\\precious.gguf",
            "/tmp/precious.gguf",
            "sub//precious.gguf",
            "./precious.gguf",
            "C:/precious.gguf",
        ] {
            let r = weights_delete(inside.to_string_lossy().into(), vec![name.into()]);
            assert!(r.is_err(), "{name:?} must be refused");
        }
        assert!(outside.join("precious.gguf").exists(), "nothing outside the folder may be touched");
    }

    #[test]
    fn refuses_anything_that_is_not_weights() {
        let d = tmpdir();
        write(&d, "notes.txt", 3);
        assert!(weights_delete(d.to_string_lossy().into(), vec!["notes.txt".into()]).is_err());
        assert!(weights_delete(d.to_string_lossy().into(), vec!["".into()]).is_err());
        assert!(weights_delete(d.to_string_lossy().into(), Vec::new()).is_err());
        assert!(d.join("notes.txt").exists());
    }

    #[test]
    fn missing_files_are_not_an_error() {
        // The row may already be gone (deleted in another window, or by hand).
        let d = tmpdir();
        assert_eq!(weights_delete(d.to_string_lossy().into(), vec!["nope.gguf".into()]).unwrap(), 0);
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
        assert_eq!(got[0].files, vec!["done.gguf"]);
        assert!(!got[0].partial);
        assert_eq!(got[0].bytes, 100);
        // The .part suffix is stripped so the front end can look the name up in
        // a manifest — otherwise every unfinished download shows as unknown.
        assert_eq!(got[1].file, "half.gguf");
        assert_eq!(got[1].files, vec!["half.gguf"]);
        assert!(got[1].partial);
        assert_eq!(got[1].bytes, 40);
    }

    #[test]
    fn recursively_groups_a_complete_split_model() {
        let d = tmpdir();
        let q = d.join("UD-IQ2_XXS");
        fs::create_dir_all(&q).unwrap();
        for (part, len) in [(1, 5), (2, 50), (3, 41)] {
            write(
                &q,
                &format!("DeepSeek-V4-Flash-UD-IQ2_XXS-{part:05}-of-00003.gguf"),
                len,
            );
        }
        let got = weights_list(d.to_string_lossy().into());
        assert_eq!(got.len(), 1, "one split model must be one UI row");
        assert_eq!(
            got[0].file,
            "UD-IQ2_XXS/DeepSeek-V4-Flash-UD-IQ2_XXS-00001-of-00003.gguf"
        );
        assert_eq!(got[0].files.len(), 3);
        assert_eq!(got[0].bytes, 96);
        assert!(!got[0].partial);
    }

    #[test]
    fn a_missing_or_partial_split_member_marks_the_set_unfinished() {
        let d = tmpdir();
        let q = d.join("quant");
        fs::create_dir_all(&q).unwrap();
        write(&q, "model-00001-of-00003.gguf", 5);
        write(&q, "model-00002-of-00003.gguf.part", 17);
        let got = weights_list(d.to_string_lossy().into());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].bytes, 22);
        assert!(got[0].partial);
        assert_eq!(got[0].files, vec![
            "quant/model-00001-of-00003.gguf",
            "quant/model-00002-of-00003.gguf",
        ]);
    }

    #[test]
    fn deletes_a_nested_split_set_and_its_sidecars() {
        let d = tmpdir();
        let q = d.join("quant");
        fs::create_dir_all(&q).unwrap();
        for part in 1..=3 {
            let name = format!("model-{part:05}-of-00003.gguf");
            write(&q, &name, part as usize);
            write(&q, &format!("{name}.sha256"), 2);
        }
        write(&q, "model-00001-of-00003.gguf.idx", 4);
        write(&q, "model-00001-of-00003.gguf.idx.gen", 3);
        let row = weights_list(d.to_string_lossy().into()).pop().unwrap();
        let freed = weights_delete(d.to_string_lossy().into(), row.files).unwrap();
        assert_eq!(freed, 19, "weights plus verification/index sidecars are removed");
        assert!(!q.exists(), "an empty quantization directory should not linger");
        assert!(d.exists(), "the configured model folder itself must remain");
    }

    #[cfg(unix)]
    #[test]
    fn scan_and_delete_do_not_follow_a_nested_symlink() {
        use std::os::unix::fs::symlink;
        let outside = tmpdir();
        write(&outside, "precious.gguf", 9);
        let inside = tmpdir();
        symlink(&outside, inside.join("escape")).unwrap();
        assert!(weights_list(inside.to_string_lossy().into()).is_empty());
        assert!(weights_delete(
            inside.to_string_lossy().into(),
            vec!["escape/precious.gguf".into()]
        ).is_err());
        assert!(outside.join("precious.gguf").exists());
    }

    #[test]
    fn list_of_a_missing_folder_is_empty_not_a_crash() {
        assert!(weights_list("/nonexistent/idletoken/models".into()).is_empty());
    }

    #[test]
    fn chinese_model_and_cache_paths_round_trip_without_replacement() {
        let d = tmpdir().join("模型缓存");
        fs::create_dir_all(&d).unwrap();
        write(&d, "中文模型.gguf", 13);
        let dir = d.to_str().expect("the CJK fixture must be Unicode").to_owned();

        let listed = weights_list(dir.clone());
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].file, "中文模型.gguf");
        assert_eq!(listed[0].bytes, 13);

        let state = weights_state(
            dir.clone(), "中文模型.gguf".into(), 13, String::new());
        assert!(state.complete && state.verified);
        assert!(state.path.contains("模型缓存"));
        assert!(state.path.ends_with("中文模型.gguf"));

        assert_eq!(weights_delete(
            dir, vec!["中文模型.gguf".into()]).unwrap(), 13);
        assert!(!d.join("中文模型.gguf").exists());
    }
}

/// Tests for the SHA-256 integrity gate.
///
/// The expected hashes are **external truth** (the published SHA-256 test
/// vector for "abc" and a value computed with `shasum -a 256` outside this
/// codebase), not values produced by the code under test — an oracle that
/// shares the implementation would pass no matter what the implementation
/// does. Every gate also has its positive control: the corrupted-content
/// cases MUST go red, or the passing cases prove nothing.
#[cfg(test)]
mod integrity_tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as O};

    static N: AtomicU32 = AtomicU32::new(0);

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "idletoken-itest-{}-{}",
            std::process::id(),
            N.fetch_add(1, O::SeqCst)
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// FIPS 180-2 test vector: SHA-256("abc").
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn no_emit(_: serde_json::Value) {}

    #[test]
    fn promotion_accepts_a_matching_file_and_leaves_a_marker() {
        let d = tmpdir();
        let part = d.join("m.gguf.part");
        let fin = d.join("m.gguf");
        fs::write(&part, b"abc").unwrap();
        let cancel = AtomicBool::new(false);
        promote_verified(&part, &fin, ABC, "t", &cancel, &no_emit).unwrap();
        assert!(fin.exists() && !part.exists());
        assert_eq!(fs::read_to_string(marker_of(&fin)).unwrap().trim(), ABC);
    }

    #[test]
    fn promotion_rejects_and_deletes_a_corrupted_file() {
        // Positive control for the whole gate: same length, one byte off.
        let d = tmpdir();
        let part = d.join("m.gguf.part");
        let fin = d.join("m.gguf");
        fs::write(&part, b"abd").unwrap();
        let cancel = AtomicBool::new(false);
        let e = promote_verified(&part, &fin, ABC, "t", &cancel, &no_emit).unwrap_err();
        assert!(e.contains("WEIGHTS_SHA256_MISMATCH"), "got: {e}");
        assert!(!part.exists(), "a provably wrong file must not be kept");
        assert!(!fin.exists(), "and must never reach the final name");
    }

    #[test]
    fn promotion_without_a_pinned_hash_just_renames() {
        // Unpinned manifest (transition state): no verification, no marker —
        // a marker would claim a check that never happened.
        let d = tmpdir();
        let part = d.join("m.gguf.part");
        let fin = d.join("m.gguf");
        fs::write(&part, b"anything").unwrap();
        let cancel = AtomicBool::new(false);
        promote_verified(&part, &fin, "", "t", &cancel, &no_emit).unwrap();
        assert!(fin.exists());
        assert!(!marker_of(&fin).exists());
    }

    #[test]
    fn legacy_file_verifies_once_then_hits_the_marker() {
        let d = tmpdir();
        let fin = d.join("m.gguf");
        fs::write(&fin, b"abc").unwrap();
        let cancel = AtomicBool::new(false);
        // No marker (script download): full hash, then a marker appears.
        ensure_final_verified(&fin, ABC, "t", &cancel, &no_emit).unwrap();
        assert!(marker_of(&fin).exists());
        // Second call is the fast path; corrupt the CONTENT but keep the
        // marker to prove the marker is what answers now.
        fs::write(&fin, b"abd").unwrap();
        ensure_final_verified(&fin, ABC, "t", &cancel, &no_emit).unwrap();
    }

    #[test]
    fn legacy_corrupted_file_is_deleted_and_reported() {
        let d = tmpdir();
        let fin = d.join("m.gguf");
        fs::write(&fin, b"abd").unwrap();
        let cancel = AtomicBool::new(false);
        let e = ensure_final_verified(&fin, ABC, "t", &cancel, &no_emit).unwrap_err();
        assert!(e.contains("WEIGHTS_SHA256_MISMATCH"), "got: {e}");
        assert!(!fin.exists());
    }

    #[test]
    fn stale_marker_forces_a_real_rehash() {
        // The manifest hash changed (curation update): the old marker must not
        // bless the file against the NEW expectation.
        let d = tmpdir();
        let fin = d.join("m.gguf");
        fs::write(&fin, b"abc").unwrap();
        fs::write(marker_of(&fin), "0000000000000000000000000000000000000000000000000000000000000000\n").unwrap();
        let cancel = AtomicBool::new(false);
        ensure_final_verified(&fin, ABC, "t", &cancel, &no_emit).unwrap();
        assert_eq!(
            fs::read_to_string(marker_of(&fin)).unwrap().trim(),
            ABC,
            "the marker must be rewritten to the hash that was actually verified"
        );
    }

    #[test]
    fn state_reports_verified_only_with_a_matching_marker() {
        let d = tmpdir();
        let dir = d.to_string_lossy().into_owned();
        fs::write(d.join("m.gguf"), b"abc").unwrap();
        // Complete but never verified.
        let s = weights_state(dir.clone(), "m.gguf".into(), 0, ABC.into());
        assert!(s.complete && !s.verified);
        // No hash pinned: nothing to verify, not blocked.
        let s = weights_state(dir.clone(), "m.gguf".into(), 0, String::new());
        assert!(s.complete && s.verified);
        // Marker present and matching.
        fs::write(marker_of(&d.join("m.gguf")), format!("{ABC}\n")).unwrap();
        let s = weights_state(dir.clone(), "m.gguf".into(), 0, ABC.into());
        assert!(s.complete && s.verified);
        // Marker for a different hash (uppercase compare also covered).
        let s = weights_state(dir, "m.gguf".into(), 0, "AB".repeat(32));
        assert!(s.complete && !s.verified);
    }

    #[test]
    fn delete_removes_the_marker_with_the_file() {
        let d = tmpdir();
        fs::write(d.join("m.gguf"), b"abc").unwrap();
        fs::write(d.join("m.gguf.sha256"), ABC).unwrap();
        weights_delete(d.to_string_lossy().into(), vec!["m.gguf".into()]).unwrap();
        assert!(!d.join("m.gguf.sha256").exists(), "an orphaned marker would bless a future download it never saw");
    }
}
