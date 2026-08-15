//! System tray presence: the app keeps serving after its window is closed.
//!
//! A node in this product is a background service — it holds model layers in
//! VRAM, answers API requests and stays in a cluster. Quitting it because
//! someone pressed the window's X is wrong, and so is leaving no way back to
//! the UI. The tray is that way back, plus a place to read the cluster's state
//! without opening anything.
//!
//! The menu is rebuilt from the front end (`tray_sync`) rather than translated
//! here: the language and the engine's state both live over there, and a
//! second copy of either in Rust is a second thing that can go stale. Until
//! the first sync arrives the menu carries the English fallbacks below, which
//! is also what a run with no webview (headless acceptance) sees.

use std::sync::Mutex;

use serde::Deserialize;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager};

const TRAY_ID: &str = "main";

/// Menu text + the status line, supplied by the front end.
#[derive(Debug, Clone, Deserialize)]
pub struct TrayLabels {
    pub open: String,
    /// One line of state, e.g. "Cluster ready · 4 machines". Shown as a
    /// disabled item at the top of the menu and as the icon's tooltip.
    pub status: String,
    pub check_update: String,
    pub quit: String,
}

impl Default for TrayLabels {
    fn default() -> Self {
        Self {
            open: "Open IdleToken".into(),
            status: "IdleToken".into(),
            check_update: "Check for updates…".into(),
            quit: "Quit IdleToken".into(),
        }
    }
}

#[derive(Default)]
pub struct TrayState(Mutex<TrayLabels>);

fn build_menu(app: &AppHandle, labels: &TrayLabels) -> tauri::Result<Menu<tauri::Wry>> {
    // The status line is a menu item rather than tooltip-only because on
    // Windows a tooltip needs a hover of about a second, while the menu is
    // already open when you are looking for it.
    let status = MenuItem::with_id(app, "status", &labels.status, false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", &labels.open, true, None::<&str>)?;
    let update = MenuItem::with_id(app, "update", &labels.check_update, true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", &labels.quit, true, None::<&str>)?;
    Menu::with_items(
        app,
        &[
            &status,
            &PredefinedMenuItem::separator(app)?,
            &open,
            &update,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )
}

/// Create the tray icon. Returns whether it exists afterwards, which is what
/// `window::hide_allowed` keys off — on Linux this can legitimately fail
/// (no AppIndicator implementation installed), and the correct response is to
/// keep the window closable-to-quit rather than to hide it into nowhere.
pub fn install(app: &AppHandle) -> bool {
    if app.tray_by_id(TRAY_ID).is_some() {
        return true;
    }
    let labels = app.state::<TrayState>().0.lock().unwrap().clone();
    let built = (|| -> tauri::Result<()> {
        let menu = build_menu(app, &labels)?;
        let mut builder = TrayIconBuilder::with_id(TRAY_ID)
            .menu(&menu)
            // Left click opens the window (handled below); the menu is the
            // right-click gesture users expect on Windows and Linux.
            .show_menu_on_left_click(false)
            .tooltip(&labels.status)
            .on_menu_event(|app, event| on_menu(app, event.id().as_ref()))
            .on_tray_icon_event(|tray, event| {
                if let TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } = event
                {
                    crate::window::show_main(tray.app_handle());
                }
            });
        if let Some(icon) = app.default_window_icon() {
            builder = builder.icon(icon.clone());
        }
        builder.build(app)?;
        Ok(())
    })();

    let alive = match built {
        Ok(()) => true,
        Err(e) => {
            // Loud, because everything downstream (close-to-tray, start
            // minimized) silently changes behaviour when this fails.
            eprintln!("idletoken-client: tray icon unavailable ({e}); the window will close to quit instead");
            false
        }
    };
    app.state::<crate::window::SysPrefs>().set_tray_alive(alive);
    alive
}

/// Remove the icon (the user turned "Show tray icon" off).
pub fn remove(app: &AppHandle) {
    let _ = app.remove_tray_by_id(TRAY_ID);
    app.state::<crate::window::SysPrefs>().set_tray_alive(false);
}

fn on_menu(app: &AppHandle, id: &str) {
    match id {
        "open" => crate::window::show_main(app),
        "update" => {
            // Show the window first: the update dialog is drawn by the front
            // end, and a prompt nobody can see is not a prompt.
            crate::window::show_main(app);
            let _ = app.emit("tray-check-update", ());
        }
        "quit" => crate::quit(app),
        _ => {}
    }
}

/// Push localized labels + the current cluster state into the tray. Also
/// creates the icon if it should exist and does not yet (first call after the
/// front end has loaded its settings).
#[tauri::command]
pub fn tray_sync(app: AppHandle, labels: TrayLabels) -> Result<(), String> {
    *app.state::<TrayState>().0.lock().unwrap() = labels.clone();
    if !app.state::<crate::window::SysPrefs>().get().tray_icon {
        return Ok(());
    }
    install(&app);
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return Ok(()); // tray unavailable on this desktop; already reported
    };
    let menu = build_menu(&app, &labels).map_err(|e| e.to_string())?;
    tray.set_menu(Some(menu)).map_err(|e| e.to_string())?;
    tray.set_tooltip(Some(&labels.status)).map_err(|e| e.to_string())?;
    Ok(())
}
