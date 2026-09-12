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

function reopenProblems(main, window) {
  const problems = [];
  const helper = blockBetween(
    main,
    "fn handle_macos_reopen(",
    "fn main() {",
  );
  if (!helper.includes("window::show_main(app);")) {
    problems.push("the macOS reopen handler does not restore the main window");
  }

  const eventLoop = blockBetween(main, ".run(|app, event| {", "\n        });");
  if (
    !/#\[cfg\(target_os = "macos"\)\][\s\S]*?tauri::RunEvent::Reopen/.test(
      eventLoop,
    ) ||
    !eventLoop.includes("handle_macos_reopen(app, has_visible_windows)")
  ) {
    problems.push("the Tauri event loop does not route macOS Reopen events");
  }

  const showMain = blockBetween(
    window,
    "pub fn show_main(",
    "/// Whether the saved position",
  );
  const show = showMain.indexOf("w.show()");
  const unminimize = showMain.indexOf("w.unminimize()");
  const focus = showMain.indexOf("w.set_focus()");
  if (
    show < 0 ||
    unminimize < 0 ||
    focus < 0 ||
    !(show < unminimize && unminimize < focus)
  ) {
    problems.push("window restoration must show, un-minimize, then focus");
  }
  return problems;
}

assert.deepEqual(reopenProblems(mainSource, windowSource), []);

// Inverse controls prove that the check fails for the regression it guards,
// rather than staying green merely because it stopped finding the source.
const noRoute = mainSource.replace(
  "handle_macos_reopen(app, has_visible_windows)",
  "()",
);
assert.notEqual(noRoute, mainSource, "route mutation must change the source");
assert.match(reopenProblems(noRoute, windowSource).join("\n"), /does not route/);

const noRestore = mainSource.replace(
  "    window::show_main(app);\n}",
  "    // inverse control: leave the window hidden\n}",
);
assert.notEqual(noRestore, mainSource, "restore mutation must change the source");
assert.match(reopenProblems(noRestore, windowSource).join("\n"), /does not restore/);

const staysMinimized = windowSource.replace(
  "        let _ = w.unminimize();",
  "        // inverse control: leave minimized",
);
assert.notEqual(
  staysMinimized,
  windowSource,
  "un-minimize mutation must change the source",
);
assert.match(
  reopenProblems(mainSource, staysMinimized).join("\n"),
  /show, un-minimize, then focus/,
);

console.log("MACOS_WINDOW_REOPEN_CHECK_OK (3 inverse controls)");
