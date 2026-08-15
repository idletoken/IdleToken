//! In-app update: check a signed release manifest, ask, install, relaunch.
//!
//! Three properties this has to keep, in order of importance:
//!
//! 1. **Nothing installs without a valid signature.** The bytes come from a
//!    release host we do not control the transport of end-to-end, and they are
//!    then executed on the user's machine — that is the definition of a
//!    supply-chain step. `Update::download` verifies the minisign signature
//!    against the pubkey compiled into this binary (tauri.conf.json
//!    `plugins.updater.pubkey`) and returns an error rather than bytes when it
//!    does not match. Nothing here catches that error into a "close enough".
//! 2. **Nothing installs without the user saying so.** `update_check` is the
//!    only thing that runs on its own; downloading and installing are separate
//!    commands the front end calls after the prompt is answered.
//! 3. **"Could not check" and "you are up to date" are different answers.** A
//!    checker that reports an unreachable feed as "no update" is the reason
//!    people run months-old builds believing they are current.
//!
//! The feed is a static JSON on the public repo's releases (the same place the
//! user guide sends people to download from), one per channel. Both the URL
//! and the pubkey can be overridden by environment variables so the acceptance
//! gate can point the production code path at a local feed it signed itself —
//! the gate exercises this file, not a copy of it.

use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

/// Where the release manifests live. `latest.json` is attached to the GitHub
/// release; `/releases/latest/` always resolves to the newest non-prerelease,
/// which is precisely what the stable channel means. Beta builds are published
/// as a pre-release under a fixed `beta` tag, because a pre-release is by
/// definition never what `/latest/` points at.
const STABLE_FEED: &str = "https://github.com/idletoken/IdleToken/releases/latest/download/latest.json";
const BETA_FEED: &str = "https://github.com/idletoken/IdleToken/releases/download/beta/latest.json";

/// Test/self-host overrides. `IDLETOKEN_UPDATE_URL` also lets someone running
/// their own build point at their own feed without patching the binary.
const ENV_URL: &str = "IDLETOKEN_UPDATE_URL";
const ENV_PUBKEY: &str = "IDLETOKEN_UPDATE_PUBKEY";

fn feed_url(channel: &str) -> String {
    if let Ok(url) = std::env::var(ENV_URL) {
        if !url.trim().is_empty() {
            return url.trim().to_string();
        }
    }
    match channel {
        "beta" => BETA_FEED.to_string(),
        _ => STABLE_FEED.to_string(),
    }
}

/// What the front end shows in the prompt.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub version: String,
    pub current_version: String,
    /// Release notes from the manifest's `notes` field (may be absent).
    pub notes: Option<String>,
    /// RFC3339 publish date, when the manifest carries one.
    pub date: Option<String>,
    /// Which channel this answer came from — the prompt says so, because
    /// "0.2.0-beta.3 is available" is a different offer from "0.2.0 is
    /// available".
    pub channel: String,
}

/// The update found by the last successful check, and its downloaded bytes.
///
/// Two steps rather than one `download_and_install` so that the download (and
/// with it the signature check) can be exercised without replacing the running
/// application — see `update_download` and the G-UPDATE gate.
#[derive(Default)]
pub struct Updates {
    pending: Mutex<Option<Update>>,
    bytes: Mutex<Option<Vec<u8>>>,
}

fn build_updater(app: &AppHandle, channel: &str) -> Result<tauri_plugin_updater::Updater, String> {
    let url = feed_url(channel);
    let parsed = url::Url::parse(&url).map_err(|e| format!("bad update feed url {url}: {e}"))?;
    let mut builder = app
        .updater_builder()
        .endpoints(vec![parsed])
        .map_err(|e| format!("update endpoint rejected: {e}"))?;
    if let Ok(pubkey) = std::env::var(ENV_PUBKEY) {
        if !pubkey.trim().is_empty() {
            builder = builder.pubkey(pubkey.trim());
        }
    }
    builder.build().map_err(|e| e.to_string())
}

/// Ask the feed whether something newer exists.
///
/// `Ok(None)` means "checked, and this is the newest build" — an unreachable
/// feed, a malformed manifest or a rejected endpoint all come back as `Err`
/// with the reason, and the UI says so instead of claiming you are current.
#[tauri::command]
pub async fn update_check(app: AppHandle, channel: String) -> Result<Option<UpdateInfo>, String> {
    let updater = build_updater(&app, &channel)?;
    let found = updater
        .check()
        .await
        .map_err(|e| format!("could not check for updates: {e}"))?;

    let Some(update) = found else {
        *app.state::<Updates>().pending.lock().unwrap() = None;
        return Ok(None);
    };

    let info = UpdateInfo {
        version: update.version.clone(),
        current_version: update.current_version.clone(),
        notes: update.body.clone(),
        // Straight from the manifest rather than through the parsed
        // `OffsetDateTime`: it saves pulling in a date-formatting crate to
        // reproduce a string the feed already sent us.
        date: update
            .raw_json
            .get("pub_date")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        channel: channel.clone(),
    };
    let state = app.state::<Updates>();
    *state.pending.lock().unwrap() = Some(update);
    *state.bytes.lock().unwrap() = None;
    Ok(Some(info))
}

/// Download the update found by the last check and verify its signature.
///
/// Returns the number of bytes held in memory, ready for `update_install`.
/// Progress is emitted as `update-progress { downloaded, total }` — on a home
/// connection this is a ~100 MB installer, so a silent five minutes would read
/// as a hang.
#[tauri::command]
pub async fn update_download(app: AppHandle) -> Result<u64, String> {
    // Taken out of the mutex rather than borrowed: the guard cannot be held
    // across the await, and this is also what makes a second concurrent
    // download impossible.
    let update = app
        .state::<Updates>()
        .pending
        .lock()
        .unwrap()
        .take()
        .ok_or("no update has been found — check for updates first")?;

    let mut downloaded: u64 = 0;
    let progress_app = app.clone();
    let result = update
        .download(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = progress_app.emit(
                    "update-progress",
                    serde_json::json!({ "downloaded": downloaded, "total": total }),
                );
            },
            || {},
        )
        .await;

    let state = app.state::<Updates>();
    // Keep the pending update either way: a failed download is worth retrying
    // without a second round trip to the feed.
    *state.pending.lock().unwrap() = Some(update);

    // A signature mismatch arrives here as an ordinary Err. It is passed on
    // verbatim and the UI shows it as a failed update — never as a silent
    // fallback to "install anyway".
    let bytes = result.map_err(|e| format!("update download failed: {e}"))?;
    let len = bytes.len() as u64;
    *state.bytes.lock().unwrap() = Some(bytes);
    Ok(len)
}

/// Install the downloaded bytes and restart into the new version.
///
/// On Windows this hands over to the NSIS installer and the current process
/// exits; on macOS/Linux the bundle is replaced in place and `restart()` execs
/// the new binary. Either way the sidecars must go down first — leaving a
/// worker holding VRAM while its supervisor is replaced is how you get a
/// machine that cannot start the engine again until it is rebooted.
#[tauri::command]
pub fn update_install(app: AppHandle) -> Result<(), String> {
    let bytes = app
        .state::<Updates>()
        .bytes
        .lock()
        .unwrap()
        .take()
        .ok_or("nothing downloaded to install")?;
    let update = app
        .state::<Updates>()
        .pending
        .lock()
        .unwrap()
        .take()
        .ok_or("no update has been found — check for updates first")?;

    let _ = crate::engine::stop_all(&app);
    update
        .install(bytes)
        .map_err(|e| format!("update install failed: {e}"))?;
    // Reached on macOS/Linux (on Windows the installer has already taken the
    // process down).
    app.restart();
}

/// Everything the front end needs to describe the update state without
/// guessing: the running version, the resolved feed, and whether an update is
/// already staged in memory.
#[tauri::command]
pub fn update_state(app: AppHandle, channel: String) -> serde_json::Value {
    let state = app.state::<Updates>();
    serde_json::json!({
        "currentVersion": app.package_info().version.to_string(),
        "feed": feed_url(&channel),
        "feedOverridden": std::env::var(ENV_URL).map(|v| !v.trim().is_empty()).unwrap_or(false),
        "pending": state.pending.lock().unwrap().is_some(),
        "downloaded": state.bytes.lock().unwrap().as_ref().map(|b| b.len()),
    })
}

#[cfg(test)]
mod tests {
    /// The channel → feed mapping is the one piece of routing here that no
    /// integration test can see (the gate overrides the URL by design), and
    /// getting it wrong means a beta user is quietly served stable builds
    /// forever.
    ///
    /// One test rather than two: the override is a process-wide environment
    /// variable, and cargo runs tests in parallel threads — as two tests they
    /// took turns failing depending on who set the variable first.
    #[test]
    fn feed_url_routing() {
        // Guard against the override leaking in from the developer's shell.
        std::env::remove_var(super::ENV_URL);
        let stable = super::feed_url("stable");
        let beta = super::feed_url("beta");
        assert_ne!(stable, beta);
        assert!(stable.contains("/releases/latest/download/"), "{stable}");
        assert!(beta.contains("/releases/download/beta/"), "{beta}");
        // An unknown channel must fall back to stable, not to nothing.
        assert_eq!(super::feed_url("nonsense"), stable);

        // The override wins on every channel — that is what lets the gate
        // drive this exact code path against a feed it signed itself.
        std::env::set_var(super::ENV_URL, "http://127.0.0.1:9/latest.json");
        assert_eq!(super::feed_url("stable"), "http://127.0.0.1:9/latest.json");
        assert_eq!(super::feed_url("beta"), "http://127.0.0.1:9/latest.json");
        // An empty value is not an override; it is an unset variable that
        // happens to exist (`env FOO= client`), and must not blank the feed.
        std::env::set_var(super::ENV_URL, "");
        assert_eq!(super::feed_url("stable"), stable);
        std::env::remove_var(super::ENV_URL);
    }
}
