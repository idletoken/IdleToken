// Local UI-test channel (acceptance §8: product gates need a scriptable way
// to drive the client — this is it). The launcher sets IDLETOKEN_UI_TEST to a
// comma-separated directive list (e.g. "engine-start:worker"); the Rust side
// exposes it read-only and App executes the directives on mount through the
// SAME provider methods a user click goes through — it scripts the UI, it
// never bypasses it. Unset (the normal case) = empty list = no effect.
import { invoke } from "@tauri-apps/api/core";

export async function uiTestDirectives(): Promise<string[]> {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) return [];
  try {
    return await invoke<string[]>("ui_test_directives");
  } catch {
    return []; // older shell without the command
  }
}
