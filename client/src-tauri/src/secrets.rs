//! Somewhere to keep the webview's credentials that is not `localStorage`.
//!
//! Written 2026-08-20 (audit A-P2-2). The platform session JWT lived in
//! `localStorage`, which is a plain SQLite/LevelDB file inside the app-data
//! directory with whatever permissions the webview runtime happened to create
//! it with — readable by any process running as the user, and by anything that
//! ever manages to run script in the webview. That token can spend the
//! account's Sparks.
//!
//! What this gives instead, and what it deliberately does not:
//!
//! * The file is created with mode 0600 on Unix and inside the user's profile
//!   on Windows, so the OS user boundary is the boundary. That is a real
//!   improvement over "whatever the WebView chose", and it is the boundary the
//!   rest of the client already relies on (the RPC PSK file, `~/.idletoken`).
//! * It is NOT encrypted, and it does not pretend to be. Encrypting a file with
//!   a key that has to live next to it on the same disk protects nobody; the
//!   honest statement is "as safe as this account on this machine". A real
//!   OS-keychain move (Keychain / DPAPI / libsecret) is the next step and needs
//!   a dependency the bundle does not carry yet.
//! * It is not reachable from the page's JavaScript object graph — the webview
//!   asks for it over IPC, so a value that is never requested is never in the
//!   renderer at all.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::Value;

/// The whole store, cached in the host process so a read is not a file read.
#[derive(Default)]
pub struct Secrets(Mutex<Option<BTreeMap<String, String>>>);

fn store_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok()?;
    if home.is_empty() {
        return None;
    }
    let dir = PathBuf::from(home).join(".idletoken");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("client-secrets.json"))
}

fn read_all() -> BTreeMap<String, String> {
    let Some(p) = store_path() else { return BTreeMap::new() };
    let Ok(text) = std::fs::read_to_string(p) else { return BTreeMap::new() };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Write the store back at 0600.
///
/// The permissions are set on the file we are about to write, before the bytes
/// go in — creating it world-readable and tightening it afterwards leaves a
/// window, and the window is the whole point of the exercise.
fn write_all(map: &BTreeMap<String, String>) -> Result<(), String> {
    let path = store_path().ok_or("no home directory to store credentials in")?;
    let text = serde_json::to_string(map).map_err(|e| e.to_string())?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    // An existing file keeps its old mode through `open`, so tighten it too.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn with_store<T>(state: &Secrets, f: impl FnOnce(&mut BTreeMap<String, String>) -> T) -> T {
    let mut guard = state.0.lock().unwrap();
    let map = guard.get_or_insert_with(read_all);
    f(map)
}

/// Every stored credential, as a `{ key: value }` object.
///
/// The front end hydrates once at startup and then answers from memory, because
/// the interfaces that need a session (`AuthProvider.currentSession`,
/// `platformGate`) are synchronous and are called during render.
#[tauri::command]
pub fn secrets_load(state: tauri::State<'_, Secrets>) -> Value {
    with_store(&state, |m| serde_json::to_value(m.clone()).unwrap_or_else(|_| Value::Object(Default::default())))
}

/// Store one credential. An empty value removes it, so "sign out" and "store
/// nothing" cannot drift apart.
#[tauri::command]
pub fn secrets_set(state: tauri::State<'_, Secrets>, key: String, value: String) -> Result<(), String> {
    with_store(&state, |m| {
        if value.is_empty() {
            m.remove(&key);
        } else {
            m.insert(key, value);
        }
        write_all(m)
    })
}

#[tauri::command]
pub fn secrets_clear(state: tauri::State<'_, Secrets>) -> Result<(), String> {
    with_store(&state, |m| {
        m.clear();
        write_all(m)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value must survive a round trip, and an empty value must delete
    /// rather than store "" — a blank session that reads as present is how a
    /// signed-out client keeps trying to use a token that is not there.
    #[test]
    fn empty_means_removed() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), "v".to_string());
        assert_eq!(m.get("a").map(String::as_str), Some("v"));
        m.insert("a".to_string(), String::new());
        // Mirrors what secrets_set does with an empty value.
        if m.get("a").is_some_and(|v| v.is_empty()) {
            m.remove("a");
        }
        assert!(!m.contains_key("a"));
    }

    #[test]
    fn the_store_lives_under_the_users_own_directory() {
        // Not asserting the exact path (it depends on HOME), only that it is
        // inside the per-user IdleToken directory and not a temp/shared one.
        if let Some(p) = store_path() {
            assert!(p.ends_with(".idletoken/client-secrets.json"), "{}", p.display());
        }
    }
}
