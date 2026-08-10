// Real engine provider: thin RPC shim over the Rust supervisor in
// src-tauri/src/engine.rs, which owns the sidecar process (spawn, crash
// detection, backoff restart, log ring buffer). State changes and log lines
// are pushed as Tauri events so the UI stays live without polling.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { EngineLogLine, EngineProvider, EngineRole, EngineStatus } from "./engine";

// Event names must match src-tauri/src/engine.rs.
const EV_STATUS = "engine:status";
const EV_LOG = "engine:log";

type RawStatus = Omit<EngineStatus, "source">;

export const engineTauriProvider: EngineProvider = {
  async start(role: EngineRole, args?: string[]): Promise<void> {
    await invoke("engine_start", { role, args: args ?? [] });
  },

  async stop(): Promise<void> {
    await invoke("engine_stop");
  },

  async status(): Promise<EngineStatus> {
    const raw = await invoke<RawStatus>("engine_status");
    return { ...raw, source: "engine" };
  },

  async logs(maxLines = 200): Promise<EngineLogLine[]> {
    return await invoke<EngineLogLine[]>("engine_logs", { maxLines });
  },

  async clearKvCache(kvDir?: string): Promise<void> {
    // The Rust side runs `idletoken-worker --kv-clear [--kv-dir <dir>]` one-shot
    // and rejects on a non-zero exit, so a failure (e.g. engine build without
    // KV support yet) surfaces in the UI instead of pretending the cache is
    // gone. kvDir = settings.kvDir; empty = the engine's platform default.
    await invoke("clear_kv_cache", { kvDir: kvDir || null });
  },

  async diagnostics(baseUrl?: string): Promise<Record<string, unknown>> {
    return await invoke<Record<string, unknown>>("collect_diagnostics", {
      baseUrl: baseUrl || null,
    });
  },

  onStatus(cb: (s: EngineStatus) => void): () => void {
    const un = listen<RawStatus>(EV_STATUS, (e) => cb({ ...e.payload, source: "engine" }));
    return () => {
      un.then((f) => f());
    };
  },

  onLog(cb: (l: EngineLogLine) => void): () => void {
    const un = listen<EngineLogLine>(EV_LOG, (e) => cb(e.payload));
    return () => {
      un.then((f) => f());
    };
  },
};
