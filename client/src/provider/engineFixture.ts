// Development-only engine simulation, used when the client runs in a plain
// browser (no Tauri, no native engine). Exercises the same lifecycle the Rust
// supervisor drives — starting → running, stop, log lines — so the engine card
// can be built and reviewed offline. Everything it reports is labeled
// `source: "dev-fixture"` and the log lines say so; this must never read as a
// real engine (philosophy 9/15).
import type { EngineLogLine, EngineProvider, EngineRole, EngineStatus } from "./engine";

// Inlined (not imported from engine.ts) — this module is evaluated during
// engine.ts's own import, so a value import back into it hits the TDZ.
const STOPPED_STATUS: EngineStatus = {
  state: "stopped",
  role: null,
  pid: null,
  startedAt: null,
  restarts: 0,
  lastExitCode: null,
  refusedReason: null,
  source: "dev-fixture",
};

class EngineFixture implements EngineProvider {
  private st: EngineStatus = { ...STOPPED_STATUS };
  private lines: EngineLogLine[] = [];
  private statusSubs = new Set<(s: EngineStatus) => void>();
  private logSubs = new Set<(l: EngineLogLine) => void>();
  private timers: ReturnType<typeof setTimeout>[] = [];
  private ticker: ReturnType<typeof setInterval> | null = null;

  private emitStatus() {
    const snap = { ...this.st };
    this.statusSubs.forEach((cb) => cb(snap));
  }

  private log(line: string, stream: EngineLogLine["stream"] = "stdout") {
    const l: EngineLogLine = { ts: Date.now(), stream, line };
    this.lines.push(l);
    if (this.lines.length > 500) this.lines.shift();
    this.logSubs.forEach((cb) => cb(l));
  }

  private clearTimers() {
    this.timers.forEach(clearTimeout);
    this.timers = [];
    if (this.ticker) clearInterval(this.ticker);
    this.ticker = null;
  }

  async start(role: EngineRole, _args?: string[]): Promise<void> {
    this.clearTimers();
    this.st = { ...this.st, state: "starting", role, pid: null, startedAt: null };
    this.emitStatus();
    this.log(`[dev-fixture] simulated ${role} starting (no native engine in the browser)`);
    this.timers.push(
      setTimeout(() => {
        this.st = { ...this.st, state: "running", pid: 424242, startedAt: Date.now() };
        this.emitStatus();
        this.log(`[dev-fixture] simulated ${role} up; listening for cluster traffic`);
        this.ticker = setInterval(() => this.log("[dev-fixture] heartbeat ok"), 4000);
      }, 600)
    );
  }

  async stop(): Promise<void> {
    this.clearTimers();
    if (this.st.state !== "stopped") this.log("[dev-fixture] simulated engine stopped");
    this.st = { ...this.st, state: "stopped", pid: null, startedAt: null, lastExitCode: 0 };
    this.emitStatus();
  }

  async status(): Promise<EngineStatus> {
    return { ...this.st };
  }

  async logs(maxLines = 200): Promise<EngineLogLine[]> {
    return this.lines.slice(-maxLines);
  }

  async clearKvCache(_kvDir?: string): Promise<void> {
    // Simulated: there is no real cache in the browser. The log line keeps the
    // simulation honest; the real provider rejects on engine failure.
    await new Promise((r) => setTimeout(r, 400));
    this.log("[dev-fixture] simulated KV cache clear");
  }

  async diagnostics(_baseUrl?: string): Promise<Record<string, unknown>> {
    // In a browser there is no engine to ask. **Label it a fixture explicitly**:
    // a fake report that looks real would have support reasoning from wrong
    // facts, which is worse than having no report at all.
    return {
      schema: "idletoken-diagnostics/1",
      note: "dev fixture - no engine on this machine",
      app: { version: "dev", os: "browser", arch: "n/a" },
      probe: { error: "not available in the browser fixture" },
    };
  }

  onStatus(cb: (s: EngineStatus) => void): () => void {
    this.statusSubs.add(cb);
    return () => this.statusSubs.delete(cb);
  }

  onLog(cb: (l: EngineLogLine) => void): () => void {
    this.logSubs.add(cb);
    return () => this.logSubs.delete(cb);
  }
}

export const engineFixtureProvider: EngineProvider = new EngineFixture();
