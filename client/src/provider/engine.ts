// Engine lifecycle (acceptance P1): the client supervises the native engine as
// a long-lived sidecar — start/stop, crash detection, restart-with-backoff and
// a log tail. Interface-first (philosophy 14 + 17): the UI talks to an
// EngineProvider; the real one drives the Rust supervisor in src-tauri (which
// owns the OS process), the fixture simulates the same lifecycle in a plain
// browser so the UI can be built and reviewed. Simulated data is always
// labeled `source: "dev-fixture"` (philosophy 9/15).
import { engineTauriProvider } from "./engineTauri";
import { engineFixtureProvider } from "./engineFixture";

export type EngineRole = "worker" | "coordinator";

// stopped -> starting -> running; a crash goes to restarting (supervisor
// respawns with backoff) and, after repeated quick failures, to crashed.
export type EngineState = "stopped" | "starting" | "running" | "restarting" | "crashed";

export interface EngineStatus {
  state: EngineState;
  role: EngineRole | null;
  pid: number | null;
  startedAt: number | null; // epoch ms of the current process, null unless running
  restarts: number; // consecutive automatic restarts
  lastExitCode: number | null;
  // Set when the engine refused to join for a reason retrying cannot fix: the
  // hardware floor, or the coordinator turning this node away (a different OS
  // family from the rest of the cluster, say). The state is "crashed" but the
  // supervisor did NOT retry — show this sentence instead of the crash hint.
  refusedReason: string | null;
  source: "engine" | "dev-fixture";
}

export interface EngineLogLine {
  ts: number; // epoch ms
  stream: "stdout" | "stderr";
  line: string;
}

export interface EngineProvider {
  // `args` is the engine's own CLI contract (e.g. a worker's `--coordinator
  // host:port`, resource caps). The pairing/orchestration layer builds them;
  // starting a worker with no coordinator exits quickly and the supervisor's
  // crash/backoff handling surfaces that honestly in the UI.
  start(role: EngineRole, args?: string[]): Promise<void>;
  stop(): Promise<void>;
  status(): Promise<EngineStatus>;
  logs(maxLines?: number): Promise<EngineLogLine[]>;
  // "Clear my cache now" (acceptance P5): asks the engine to wipe its on-disk
  // KV warm cache. Rejects if the engine reports failure — never fakes success.
  // `kvDir` is settings.kvDir; empty/undefined = the engine's platform default.
  clearKvCache(kvDir?: string): Promise<void>;
  // One-click diagnostics bundle (for support): the facts about the machine, the
  // engine and the cluster, turning "it will not start here" into something that
  // can be investigated. Settings are merged in by the caller against an
  // allowlist -- this deliberately touches no tokens.
  diagnostics(baseUrl?: string): Promise<Record<string, unknown>>;
  onStatus(cb: (s: EngineStatus) => void): () => void;
  onLog(cb: (l: EngineLogLine) => void): () => void;
}

function runningInTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function getEngineProvider(): EngineProvider {
  return runningInTauri() ? engineTauriProvider : engineFixtureProvider;
}
