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

#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Managed state: live prefs, tray health, and the macOS full-screen close
/// handshake. AppKit moves a native full-screen window into its own Space;
/// hiding it before the exit animation completes strands that Space as a
/// black screen.
#[derive(Default)]
pub struct SysPrefs {
    prefs: Mutex<Prefs>,
    tray_alive: Mutex<bool>,
    #[cfg(target_os = "macos")]
    fullscreen_hide_pending: AtomicBool,
    #[cfg(target_os = "macos")]
    fullscreen_observer_ready: AtomicBool,
}

impl SysPrefs {
    pub fn get(&self) -> Prefs {
        self.prefs.lock().unwrap().clone()
    }
    fn set(&self, p: Prefs) {
        *self.prefs.lock().unwrap() = p;
    }
    /// Recorded by `tray::install`: false means tray creation failed (or the
    /// user turned it off), and therefore that hiding the window is forbidden.
    pub fn set_tray_alive(&self, alive: bool) {
        *self.tray_alive.lock().unwrap() = alive;
    }
    pub fn tray_alive(&self) -> bool {
        *self.tray_alive.lock().unwrap()
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
    // A Dock/tray activation can race the asynchronous exit from a native
    // full-screen Space. In that case the user's latest request is to show the
    // window, so the exit notification must not hide it afterwards.
    #[cfg(target_os = "macos")]
    app.state::<SysPrefs>()
        .fullscreen_hide_pending
        .store(false, Ordering::SeqCst);

    if let Some(w) = main_window(app) {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// Hide immediately after recording the restored, non-full-screen geometry.
/// On macOS this is called from the native full-screen-exit notification; on
/// other platforms it is the ordinary close-to-tray path.
fn hide_main_now(app: &AppHandle) -> Result<(), String> {
    remember_geometry(app);
    let window = main_window(app).ok_or_else(|| "main window is unavailable".to_string())?;
    window
        .hide()
        .map_err(|e| format!("failed to hide main window: {e}"))?;
    // The webview stays alive while hidden; it shows a one-time "still
    // running in the tray" notice on this signal.
    tauri::Emitter::emit(app, "tray:hidden", ())
        .map_err(|e| format!("failed to emit tray:hidden: {e}"))?;
    Ok(())
}

/// Handle a close-to-tray request without hiding a native full-screen macOS
/// window mid-transition. Returns after scheduling the deferred hide.
pub fn close_to_tray(app: &AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let state = app.state::<SysPrefs>();

        // A second close event can arrive while AppKit is still animating out
        // of full screen. It must not fall through to an immediate hide just
        // because Tauri has already updated its logical full-screen flag.
        if state.fullscreen_hide_pending.load(Ordering::SeqCst) {
            return Ok(());
        }

        let window = main_window(app).ok_or_else(|| "main window is unavailable".to_string())?;
        if window
            .is_fullscreen()
            .map_err(|e| format!("failed to inspect full-screen state: {e}"))?
        {
            if !state.fullscreen_observer_ready.load(Ordering::SeqCst) {
                return Err(
                    "refusing to hide a full-screen window without the AppKit exit observer"
                        .to_string(),
                );
            }

            state.fullscreen_hide_pending.store(true, Ordering::SeqCst);
            if let Err(e) = window.set_fullscreen(false) {
                state.fullscreen_hide_pending.store(false, Ordering::SeqCst);
                return Err(format!("failed to leave full screen before hiding: {e}"));
            }
            eprintln!("idletoken-client: close-to-tray waiting for macOS full-screen exit");
            return Ok(());
        }
    }

    hide_main_now(app)
}

/// Observe the native completion point of AppKit's asynchronous full-screen
/// exit. Tauri's logical `is_fullscreen` flag changes when the exit is
/// requested, before the Space animation is actually done, so it cannot be
/// used as the completion signal.
#[cfg(target_os = "macos")]
pub fn install_fullscreen_exit_observer(app: &AppHandle) -> Result<(), String> {
    use std::ptr::NonNull;

    use block2::RcBlock;
    use objc2::runtime::AnyObject;
    use objc2_app_kit::NSWindowDidExitFullScreenNotification;
    use objc2_foundation::{NSNotification, NSNotificationCenter};

    let window = main_window(app).ok_or_else(|| "main window is unavailable".to_string())?;
    let native_window = window
        .ns_window()
        .map_err(|e| format!("failed to obtain NSWindow: {e}"))?;
    let native_window = unsafe {
        // SAFETY: Tauri documents `ns_window()` as the native NSWindow handle;
        // Objective-C APIs accept every NSWindow as an AnyObject.
        native_window
            .cast::<AnyObject>()
            .as_ref()
            .ok_or_else(|| "Tauri returned a null NSWindow".to_string())?
    };

    let handle = app.clone();
    let callback: RcBlock<dyn Fn(NonNull<NSNotification>)> =
        RcBlock::new(move |_notification: NonNull<NSNotification>| {
            let state = handle.state::<SysPrefs>();
            if state.fullscreen_hide_pending.load(Ordering::SeqCst) {
                // Queue one main-loop turn beyond the notification. Tao's
                // NSWindow delegate restores its normal frame in the same
                // notification cycle; deferring ensures that restoration is
                // complete before geometry is read or the window is hidden.
                // Keeping the pending flag set also lets a racing Dock/tray
                // activation cancel this hide through `show_main`.
                let deferred = handle.clone();
                if let Err(e) = handle.run_on_main_thread(move || {
                    if deferred
                        .state::<SysPrefs>()
                        .fullscreen_hide_pending
                        .swap(false, Ordering::SeqCst)
                    {
                        eprintln!(
                            "idletoken-client: macOS full-screen exit complete — hiding to tray"
                        );
                        if let Err(e) = hide_main_now(&deferred) {
                            eprintln!(
                                "idletoken-client: close-to-tray failed after full-screen exit: {e}"
                            );
                        }
                    }
                }) {
                    state.fullscreen_hide_pending.store(false, Ordering::SeqCst);
                    eprintln!(
                        "idletoken-client: failed to schedule hide after full-screen exit: {e}"
                    );
                }
            }
        });

    let center = NSNotificationCenter::defaultCenter();
    let _observer = unsafe {
        // SAFETY: the notification name and object are the typed AppKit
        // NSWindow values expected by this API. AppKit posts the notification
        // on the main thread, and AppHandle is safe to move into the block.
        center.addObserverForName_object_queue_usingBlock(
            Some(NSWindowDidExitFullScreenNotification),
            Some(native_window),
            None,
            &callback,
        )
    };
    // Foundation retains the returned observer for the registration lifetime.
    // The observed NSWindow and the notification center both live until this
    // process exits, so there is no earlier teardown point to unregister it.

    app.state::<SysPrefs>()
        .fullscreen_observer_ready
        .store(true, Ordering::SeqCst);
    Ok(())
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
