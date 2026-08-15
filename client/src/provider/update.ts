// In-app update (client side of src-tauri/src/update.rs).
//
// Interface-first, like every other outside-world boundary here (philosophy
// 14): the UI talks to an UpdateProvider. The real one drives the Tauri
// updater — check a signed manifest, download, verify, install, relaunch — and
// the fixture lets the whole prompt be built and reviewed in a plain browser,
// where there is no shell to update.
//
// The distinction the interface exists to preserve: `check()` returns `null`
// for "checked, nothing newer" and **throws** when it could not check. An
// updater that reports an unreachable feed as "up to date" is how people end
// up running months-old builds thinking they are current.
import { updateTauriProvider } from "./updateTauri";
import { updateFixtureProvider } from "./updateFixture";

export interface UpdateInfo {
  version: string;
  currentVersion: string;
  notes: string | null;
  date: string | null;
  channel: string;
}

export interface UpdateProgress {
  downloaded: number;
  /** Absent when the release host sends no Content-Length. */
  total: number | null;
}

export interface UpdateState {
  currentVersion: string;
  feed: string;
  feedOverridden: boolean;
  pending: boolean;
  downloaded: number | null;
  source: "shell" | "dev-fixture";
}

export interface UpdateProvider {
  /** null = this is the newest build. Throws with a reason if the check failed. */
  check(channel: string): Promise<UpdateInfo | null>;
  /** Download + verify the signature. Returns the byte count held for install. */
  download(): Promise<number>;
  /** Install and relaunch. Never returns on success — the process is replaced. */
  install(): Promise<void>;
  state(channel: string): Promise<UpdateState>;
  onProgress(cb: (p: UpdateProgress) => void): () => void;
}

function runningInTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function getUpdateProvider(): UpdateProvider {
  return runningInTauri() ? updateTauriProvider : updateFixtureProvider;
}
