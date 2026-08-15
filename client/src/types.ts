// Mirrors the engine's `idletoken-worker --probe-json` output (src/common/resource.c).
// All byte counts are raw integers.
export interface ProbeJson {
  hostname: string;
  // "macos" since the Metal backend landed (2026-08-12). The engine
  // (src/common/resource.c) has emitted it from the start; this union simply
  // never listed it, so the type was claiming a Mac probe is impossible.
  os: "windows" | "linux" | "macos" | "unknown";
  cpu_count: number;
  gpu_name: string;
  cc_major: number;
  cc_minor: number;
  unified_memory: boolean;
  vram_total: number;
  vram_used_other: number;
  vram_usable: number;
  ram_total: number;
  ram_used_other: number;
  ram_usable: number;
  disk_avail: number;
  // Hardware floor verdict from the engine (src/common/resource.c
  // `idletoken_hw_check`). 0 = OK; anything else means this machine cannot serve
  // layers and the UI must say so instead of showing a healthy-looking card.
  // The engine owns the rule; the client only renders its verdict.
  driver_version?: string;
  hw_status?: HwStatus;
  hw_reason?: string;
}

// Mirrors `idletoken_hw_status` in include/idletoken_resource.h.
export const HW_OK = 0;
export const HW_NO_GPU = 1;
export const HW_CC_TOO_LOW = 2;
export const HW_DRIVER_TOO_OLD = 3;
export const HW_VRAM_TOO_SMALL = 4;
// A GPU we have no backend for — an Intel Mac's AMD card, say. Added with the
// Metal backend (2026-08-12); before that the enum stopped at 4 and this status
// fell through the UI's chain of comparisons, leaving a **healthy-looking card
// on a machine the engine had already refused**.
export const HW_GPU_UNSUPPORTED = 5;
// HISTORICAL (v2 unseal, 2026-08-14): macOS compute is no longer sealed —
// Macs serve models through llama.cpp Metal (docs/v2-rebuild-plan-2026-08.md
// §1.3, G-MACSEAL retired). The constant stays because an OUTDATED engine
// build can still report it; the UI renders it as "update the engine on this
// Mac", never as "your hardware is bad".
export const HW_MACOS_SEALED = 6;
export type HwStatus = 0 | 1 | 2 | 3 | 4 | 5 | 6;

// Where a snapshot came from. `engine` = a real probe from the native worker;
// `dev-fixture` = a placeholder used only when the client runs outside Tauri
// (browser dev). The UI badges dev-fixture data so it is never mistaken for real.
export type SnapshotSource = "engine" | "dev-fixture";

export interface NodeSnapshot extends ProbeJson {
  source: SnapshotSource;
  probedAt: number; // epoch ms
}

export type ClusterState = "standalone" | "joining" | "ready";
