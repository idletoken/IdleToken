//! Window/tray preferences and window geometry, owned by the Rust side.
//!
//! Every other setting in this product lives in the web front end's
//! localStorage, and that is the right place for them — the front end is the
//! only reader. These four are different: the shell has to act on them
//! **before the webview exists** (should the window be shown at all?) and
//! **when the webview may be gone** (the user pressed the window's X). So they
//! are mirrored into `<app_config_dir>/window.json`, written by the front end
//! through `window_prefs_set` whenever settings change, and read here at
//! startup.
//!
//! Fail-safe rule, enforced in one place (`hide_allowed`): **the window is
//! only ever hidden when a working tray icon exists**. Hiding the only window
//! of an app that has no tray leaves the user with a process they can see in
//! Task Manager and cannot reach — on Linux that is not hypothetical, tray
//! support depends on an AppIndicator implementation being installed.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize, WebviewWindow};

/// The four toggles the shell needs, plus the remembered geometry.
///
/// Defaults are deliberately the same as `DEFAULT_SETTINGS` in
/// `client/src/settings.ts`; the front end overwrites them on first render, so
/// the only run that uses these values is the very first one after install.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub tray_icon: bool,
    pub close_to_tray: bool,
    pub start_minimized: bool,
    pub remember_window: bool,
    /// Last known geometry, in physical pixels. `None` until a window has been
    /// closed at least once.
    pub geometry: Option<Geometry>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Geometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            tray_icon: true,
            close_to_tray: true,
            start_minimized: false,
            remember_window: true,
            geometry: None,
        }
    }
}

/// Managed state: the live prefs plus whether a tray icon actually exists.
#[derive(Default)]
pub struct SysPrefs(Mutex<Prefs>, Mutex<bool>);

impl SysPrefs {
    pub fn get(&self) -> Prefs {
        self.0.lock().unwrap().clone()
    }
    fn set(&self, p: Prefs) {
        *self.0.lock().unwrap() = p;
    }
    /// Recorded by `tray::install`: false means tray creation failed (or the
    /// user turned it off), and therefore that hiding the window is forbidden.
    pub fn set_tray_alive(&self, alive: bool) {
        *self.1.lock().unwrap() = alive;
    }
    pub fn tray_alive(&self) -> bool {
        *self.1.lock().unwrap()
    }
}

fn prefs_path(app: &AppHandle) -> Option<std::path::PathBuf> {
    let dir = app.path().app_config_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("window.json"))
}

/// Read the mirrored prefs from disk. A missing or unreadable file is not an
/// error — it is a first run, and the defaults above are what a first run
/// should get.
pub fn load(app: &AppHandle) -> Prefs {
    prefs_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<Prefs>(&s).ok())
        .unwrap_or_default()
}

fn save(app: &AppHandle, prefs: &Prefs) {
    if let Some(path) = prefs_path(app) {
        if let Ok(text) = serde_json::to_string_pretty(prefs) {
            let _ = std::fs::write(path, text);
        }
    }
}

/// May the window be hidden right now? See the fail-safe rule in the module
/// comment: only with a tray to bring it back.
pub fn hide_allowed(app: &AppHandle) -> bool {
    let state = app.state::<SysPrefs>();
    let p = state.get();
    p.tray_icon && state.tray_alive()
}

/// The main window, if it exists (it always does outside of teardown).
pub fn main_window(app: &AppHandle) -> Option<WebviewWindow> {
    app.get_webview_window("main")
}

/// Bring the window back from the tray: show, un-minimize, focus. All three
/// are needed — a hidden window that is also minimized comes back invisible if
/// you only call `show`.
pub fn show_main(app: &AppHandle) {
    if let Some(w) = main_window(app) {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// Whether the saved position still lands on a monitor that exists.
///
/// Restoring a position blindly is how an app disappears after someone
/// unplugs the external screen it was last used on: the coordinates are still
/// valid, they just point at empty space. When nothing contains the saved
/// origin we drop the position and keep the size, and the window manager
/// places the window itself.
fn position_on_screen(window: &WebviewWindow, g: &Geometry) -> bool {
    let Ok(monitors) = window.available_monitors() else {
        return false;
    };
    monitors.iter().any(|m| {
        let pos = m.position();
        let size = m.size();
        g.x >= pos.x
            && g.y >= pos.y
            && g.x < pos.x + size.width as i32
            && g.y < pos.y + size.height as i32
    })
}

/// Startup: restore geometry and decide whether the window is shown.
///
/// The window is created with `"visible": false` in tauri.conf.json so that
/// "start minimized" does not flash a window on screen first. That makes THIS
/// function responsible for the window ever appearing, so it errs towards
/// showing: a start-minimized request is honoured only when there is a tray
/// icon to restore it from.
pub fn apply_startup(app: &AppHandle) {
    let prefs = load(app);
    app.state::<SysPrefs>().set(prefs.clone());

    let Some(window) = main_window(app) else { return };

    if prefs.remember_window {
        if let Some(g) = prefs.geometry {
            // Size first, then position: on Windows setting the size of a
            // maximized window un-maximizes it, so the maximize call has to be
            // the last word.
            let _ = window.set_size(PhysicalSize::new(g.width.max(380), g.height.max(560)));
            if position_on_screen(&window, &g) {
                let _ = window.set_position(PhysicalPosition::new(g.x, g.y));
            }
            if g.maximized {
                let _ = window.maximize();
            }
        }
    }

    if prefs.start_minimized && hide_allowed(app) {
        eprintln!("idletoken-client: starting minimized to the tray");
    } else {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Record the current geometry (called when the window is hidden or closed —
/// not on every move/resize event, which would mean a disk write per frame
/// while dragging).
pub fn remember_geometry(app: &AppHandle) {
    let state = app.state::<SysPrefs>();
    let mut prefs = state.get();
    if !prefs.remember_window {
        return;
    }
    let Some(window) = main_window(app) else { return };
    // A minimized window reports a garbage position (-32000 on Windows); the
    // last good geometry is more useful than that.
    if window.is_minimized().unwrap_or(false) {
        return;
    }
    let maximized = window.is_maximized().unwrap_or(false);
    if let (Ok(pos), Ok(size)) = (window.outer_position(), window.inner_size()) {
        // While maximized, keep the restore-size the user last chose rather
        // than the screen-filling one, so un-maximizing lands somewhere sane.
        let geometry = if maximized {
            Geometry { maximized: true, ..prefs.geometry.unwrap_or(Geometry { x: pos.x, y: pos.y, width: size.width, height: size.height, maximized: true }) }
        } else {
            Geometry { x: pos.x, y: pos.y, width: size.width, height: size.height, maximized: false }
        };
        prefs.geometry = Some(geometry);
        state.set(prefs.clone());
        save(app, &prefs);
    }
}

/// Push the front end's current settings into the shell. Called on startup and
/// on every settings change, so the two never drift.
#[tauri::command]
pub fn window_prefs_set(
    app: AppHandle,
    tray_icon: bool,
    close_to_tray: bool,
    start_minimized: bool,
    remember_window: bool,
) -> Result<(), String> {
    let state = app.state::<SysPrefs>();
    let mut prefs = state.get();
    let tray_changed = prefs.tray_icon != tray_icon;
    prefs.tray_icon = tray_icon;
    prefs.close_to_tray = close_to_tray;
    prefs.start_minimized = start_minimized;
    prefs.remember_window = remember_window;
    state.set(prefs.clone());
    save(&app, &prefs);

    if tray_changed {
        if tray_icon {
            crate::tray::install(&app);
        } else {
            // Taking the icon away while the window is hidden would strand the
            // app, so put the window back first.
            crate::tray::remove(&app);
            show_main(&app);
        }
    }
    Ok(())
}

/// What the shell currently believes, for the front end to display (and for
/// the acceptance gate to assert against). `tray_alive` is the honest answer
/// to "is there actually an icon", which on Linux can be false while
/// `tray_icon` is true.
#[tauri::command]
pub fn window_prefs_get(app: AppHandle) -> serde_json::Value {
    let state = app.state::<SysPrefs>();
    let p = state.get();
    serde_json::json!({
        "trayIcon": p.tray_icon,
        "closeToTray": p.close_to_tray,
        "startMinimized": p.start_minimized,
        "rememberWindow": p.remember_window,
        "trayAlive": state.tray_alive(),
        "hideAllowed": hide_allowed(&app),
        "geometry": p.geometry,
        "visible": main_window(&app).and_then(|w| w.is_visible().ok()),
    })
}
