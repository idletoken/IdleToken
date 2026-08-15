// Shell-level behaviour that outlives the window: tray icon, close-to-tray,
// start minimized, launch at login, and quitting for real.
//
// Why these four settings are pushed into Rust instead of being read from
// localStorage like every other setting: the shell has to act on them when the
// front end cannot answer. "Should the window be shown at all?" is decided
// before the webview exists, and "the user pressed X" has to be handled
// whether or not the page is alive. src-tauri/src/window.rs mirrors them to a
// small file; this module is what keeps that mirror current.
//
// Outside Tauri (browser dev server) every call here is a no-op — there is no
// tray and no window to remember.
import type { AppSettings } from "./settings";

function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T | null> {
  if (!inTauri()) return null;
  const { invoke } = await import("@tauri-apps/api/core");
  return await invoke<T>(cmd, args);
}

/** Mirror the window/tray settings into the shell. Safe to call on every change. */
export async function syncWindowPrefs(s: AppSettings): Promise<void> {
  await call("window_prefs_set", {
    trayIcon: s.trayIcon,
    closeToTray: s.closeToTray,
    startMinimized: s.startMinimized,
    rememberWindow: s.rememberWindow,
  });
}

export interface ShellWindowState {
  trayIcon: boolean;
  closeToTray: boolean;
  startMinimized: boolean;
  rememberWindow: boolean;
  /** Whether an icon actually exists — on Linux tray support can be missing. */
  trayAlive: boolean;
  /** Whether hiding the window is currently allowed (needs a live tray). */
  hideAllowed: boolean;
  geometry: { x: number; y: number; width: number; height: number; maximized: boolean } | null;
  visible: boolean | null;
}

export async function windowState(): Promise<ShellWindowState | null> {
  return await call<ShellWindowState>("window_prefs_get");
}

export interface TrayLabels {
  open: string;
  status: string;
  check_update: string;
  quit: string;
}

/** Push localized menu text + the current cluster state into the tray. */
export async function syncTray(labels: TrayLabels): Promise<void> {
  await call("tray_sync", { labels });
}

/** "Launch at login", through the OS mechanism. Rejects if the OS refused. */
export async function setAutostart(enabled: boolean): Promise<void> {
  await call("autostart_set", { enabled });
}

/** What the OS reports, which is not always what the setting says. */
export async function getAutostart(): Promise<boolean | null> {
  return await call<boolean>("autostart_get");
}

/**
 * Quit the application for real (as opposed to closing the window, which with
 * close-to-tray on means "keep serving in the background").
 */
export async function quitApp(): Promise<void> {
  await call("app_quit");
}
