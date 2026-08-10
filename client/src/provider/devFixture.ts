import type { NodeSnapshot } from "../types";
import type { ProbeOptions, ResourceProvider } from "./index";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

// Development-only placeholder, used when the client runs in a plain browser
// (no Tauri, no engine). Modeled on win-a (RTX 5060 Ti 16G) so the layout can
// be reviewed with realistic magnitudes. `source: "dev-fixture"` makes the UI
// show a DEV FIXTURE badge — this data must never read as a real probe.
// The fixture honors usage caps too, so the settings panel visibly takes effect
// during development exactly as it will against the real engine.
export const devFixtureProvider: ResourceProvider = {
  async probe(opts?: ProbeOptions): Promise<NodeSnapshot> {
    // Small delay so loading states are exercised during development.
    await new Promise((r) => setTimeout(r, 350));
    let vramUsable = Math.round(11.7 * GiB);
    let ramUsable = Math.round(16.6 * GiB);
    if (opts?.maxVramMb) vramUsable = Math.min(vramUsable, opts.maxVramMb * MiB);
    if (opts?.maxRamMb) ramUsable = Math.min(ramUsable, opts.maxRamMb * MiB);
    return {
      source: "dev-fixture",
      probedAt: Date.now(),
      hostname: "win-pc-01",
      os: "windows",
      cpu_count: 20,
      gpu_name: "NVIDIA GeForce RTX 5060 Ti",
      cc_major: 12,
      cc_minor: 0,
      unified_memory: false,
      vram_total: Math.round(16 * GiB),
      vram_used_other: Math.round(2.8 * GiB),
      vram_usable: vramUsable,
      ram_total: Math.round(32 * GiB),
      ram_used_other: Math.round(11.4 * GiB),
      ram_usable: ramUsable,
      disk_avail: Math.round(412 * GiB),
    };
  },
};
