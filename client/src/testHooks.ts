// Local UI-test channel (acceptance §8: product gates need a scriptable way
// to drive the client — this is it). The launcher sets IDLETOKEN_UI_TEST to a
// comma-separated directive list (e.g. "engine-start:worker"); the Rust side
// exposes it read-only and App executes the directives on mount through the
// SAME provider methods a user click goes through — it scripts the UI, it
// never bypasses it. Unset (the normal case) = empty list = no effect.
import { invoke } from "@tauri-apps/api/core";

export async function uiTestDirectives(): Promise<string[]> {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) return [];
  // Retry: at mount time the IPC bridge is not always up yet, and a single
  // failed invoke was silently swallowed into [] — with the run-once latch
  // already set, the whole directive list was lost for the process. Seen as a
  // startup race on the DGX client (2026-08-15): identical binary, directives
  // ran in one launch and vanished in the next.
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await invoke<string[]>("ui_test_directives");
    } catch {
      await new Promise((r) => setTimeout(r, 400));
    }
  }
  return []; // older shell without the command
}
