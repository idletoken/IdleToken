import { invoke } from "@tauri-apps/api/core";
import type { NodeSnapshot, ProbeJson } from "../types";
import type { ProbeOptions, ResourceProvider } from "./index";

// Real provider: asks the Rust backend to run the native worker's `--probe-json`
// as a sidecar and return the parsed hardware report. Usage caps are forwarded
// so the engine clamps its reported usable values. The Rust command is
// `probe_resources` (see src-tauri/src/main.rs).
export const tauriProvider: ResourceProvider = {
  async probe(opts?: ProbeOptions): Promise<NodeSnapshot> {
    const raw = await invoke<ProbeJson>("probe_resources", {
      maxVramMb: opts?.maxVramMb ?? 0,
      maxRamMb: opts?.maxRamMb ?? 0,
    });
    return { ...raw, source: "engine", probedAt: Date.now() };
  },
};
