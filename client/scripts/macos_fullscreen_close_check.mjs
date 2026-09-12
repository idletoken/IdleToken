import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const mainSource = readFileSync(
  new URL("../src-tauri/src/main.rs", import.meta.url),
  "utf8",
);
const windowSource = readFileSync(
  new URL("../src-tauri/src/window.rs", import.meta.url),
  "utf8",
);

function blockBetween(source, startNeedle, endNeedle) {
  const start = source.indexOf(startNeedle);
  if (start < 0) return "";
  const end = source.indexOf(endNeedle, start + startNeedle.length);
  return end < 0 ? "" : source.slice(start, end);
}

function fullscreenCloseProblems(main, window) {
  const problems = [];
  const closeEvent = blockBetween(
    main,
    ".on_window_event(|window, event| {",
    ".invoke_handler(",
  );
  if (!closeEvent.includes("window::close_to_tray(app)")) {
    problems.push("CloseRequested does not use the guarded close-to-tray path");
  }
  if (/\bwindow\s*\.\s*hide\s*\(/.test(closeEvent)) {
    problems.push("CloseRequested still hides the window directly");
  }

  const setup = blockBetween(main, ".setup(|app| {", ".on_window_event(");
  if (!setup.includes("window::install_fullscreen_exit_observer(&handle)")) {
    problems.push("the AppKit full-screen-exit observer is not installed");
  }

  const closeToTray = blockBetween(
    window,
    "pub fn close_to_tray(",
    "/// Observe the native completion point",
  );
  const pendingGuard = closeToTray.indexOf("fullscreen_hide_pending.load(");
  const markPending = closeToTray.indexOf("fullscreen_hide_pending.store(true");
  const exitFullscreen = closeToTray.indexOf("window.set_fullscreen(false)");
  const deferredReturn = closeToTray.indexOf("return Ok(());", exitFullscreen);
  const immediateHide = closeToTray.lastIndexOf("hide_main_now(app)");
  if (
    pendingGuard < 0 ||
    markPending < 0 ||
    exitFullscreen < 0 ||
    deferredReturn < 0 ||
    immediateHide < 0 ||
    !(pendingGuard < markPending &&
      markPending < exitFullscreen &&
      exitFullscreen < deferredReturn &&
      deferredReturn < immediateHide)
  ) {
    problems.push(
      "full-screen close must guard repeats, request exit, return, then use the normal hide fallback",
    );
  }

  const observer = blockBetween(
    window,
    "pub fn install_fullscreen_exit_observer(",
    "/// Whether the saved position",
  );
  const didExit = observer.indexOf("NSWindowDidExitFullScreenNotification");
  const nextTurn = observer.indexOf("handle.run_on_main_thread(");
  const consumePending = observer.indexOf(
    ".swap(false, Ordering::SeqCst)",
    nextTurn,
  );
  const hideAfterExit = observer.indexOf("hide_main_now(&deferred)");
  if (
    didExit < 0 ||
    nextTurn < 0 ||
    consumePending < 0 ||
    hideAfterExit < 0 ||
    !(didExit < nextTurn && nextTurn < consumePending && consumePending < hideAfterExit)
  ) {
    problems.push(
      "the did-exit observer must defer one main-loop turn, consume the pending request, then hide",
    );
  }

  const showMain = blockBetween(
    window,
    "pub fn show_main(",
    "/// Hide immediately",
  );
  const pendingField = showMain.indexOf("fullscreen_hide_pending");
  const cancelPending = showMain.indexOf(".store(false, Ordering::SeqCst)");
  const show = showMain.indexOf("w.show()");
  if (
    pendingField < 0 ||
    cancelPending < 0 ||
    show < 0 ||
    !(pendingField < cancelPending && cancelPending < show)
  ) {
    problems.push("show_main does not cancel a racing deferred hide");
  }

  return problems;
}

assert.deepEqual(fullscreenCloseProblems(mainSource, windowSource), []);

// Inverse controls prove this check fails for each unsafe regression instead
// of staying green because a source marker silently stopped matching.
const directHide = mainSource.replace(
  "window::close_to_tray(app)",
  "window.hide().map_err(|e| e.to_string())",
);
assert.notEqual(directHide, mainSource, "direct-hide mutation must change source");
assert.match(
  fullscreenCloseProblems(directHide, windowSource).join("\n"),
  /guarded close-to-tray|hides the window directly/,
);

const noExit = windowSource.replace(
  "window.set_fullscreen(false)",
  "Ok(())",
);
assert.notEqual(noExit, windowSource, "full-screen-exit mutation must change source");
assert.match(
  fullscreenCloseProblems(mainSource, noExit).join("\n"),
  /full-screen close must/,
);

const tooEarly = windowSource.replace(
  "NSWindowDidExitFullScreenNotification",
  "NSWindowWillExitFullScreenNotification",
);
assert.notEqual(tooEarly, windowSource, "notification mutation must change source");
assert.match(
  fullscreenCloseProblems(mainSource, tooEarly).join("\n"),
  /did-exit observer/,
);

const sameTurn = windowSource.replace(
  "handle.run_on_main_thread(move || {",
  "Ok::<(), String>({",
);
assert.notEqual(sameTurn, windowSource, "main-loop mutation must change source");
assert.match(
  fullscreenCloseProblems(mainSource, sameTurn).join("\n"),
  /defer one main-loop turn/,
);

const noCancel = windowSource.replace(
  ".fullscreen_hide_pending\n        .store(false, Ordering::SeqCst);",
  ".fullscreen_hide_pending;",
);
assert.notEqual(noCancel, windowSource, "race-cancel mutation must change source");
assert.match(
  fullscreenCloseProblems(mainSource, noCancel).join("\n"),
  /cancel a racing deferred hide/,
);

console.log("MACOS_FULLSCREEN_CLOSE_CHECK_OK (5 inverse controls)");
