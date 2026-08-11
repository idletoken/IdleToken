// Engine/app settings that must really take effect (design philosophy 15).
// Split into a Simple set (what a home user needs) and an Advanced set (precise
// control). Persisted to localStorage and restored on launch. Theme + language
// live separately (pure UI state) but are surfaced in the Simple tab.
import { DEFAULT_MODEL_ID, defaultQuant, isAvailable, quantOptions } from "./models";

const MiB = 1024 ** 2;

// Context/performance tiers (docs/architecture.md §5). id -> context window + label key.
export interface Tier {
  id: 1 | 2 | 3 | 4 | 5;
  ctx: number;
}
export const TIERS: Tier[] = [
  { id: 1, ctx: 8192 },
  { id: 2, ctx: 32768 },
  { id: 3, ctx: 131072 },
  { id: 4, ctx: 524288 },
  { id: 5, ctx: 1048576 },
];

export type ResourcePreset = "conservative" | "balanced" | "max" | "custom";
// Fraction of a machine's total the preset lets IdleToken use (max = no cap).
export const PRESET_FRACTION: Record<Exclude<ResourcePreset, "custom">, number> = {
  conservative: 0.5,
  balanced: 0.75,
  max: 1,
};

export type ComputeMode = "auto" | "gpu_only" | "hybrid";
export type WeightsSource = "auto" | "local";
export type KvEviction = "lru" | "fifo";
export type LogLevel = "error" | "info" | "debug";
export type Density = "comfortable" | "compact";
export type Accent = "amber" | "teal" | "violet" | "rose";
export type UpdateChannel = "stable" | "beta";

export interface AppSettings {
  // ---- simple ----
  modelId: string;
  quant: string; // selected precision ("" = model's default variant / single-precision models)
  tier: Tier["id"];
  resourcePreset: ResourcePreset;
  // ---- advanced: resources (precise; used when resourcePreset === "custom") ----
  maxVramMb: number; // 0 = no cap
  maxRamMb: number; // 0 = no cap
  computeMode: ComputeMode; // reserved — engine auto-determines GPU_ONLY/HYBRID, no override flag yet
  // ---- advanced: model / weights ----
  weightsSource: WeightsSource;
  ggufPath: string;
  // ---- advanced: API exposure (P6) ----
  apiHost: string;
  apiPort: number;
  apiOpenAI: boolean;
  apiAnthropic: boolean;
  apiToken: string; // empty = no auth (LAN default)
  // ---- advanced: default sampling (interface-reserved; sampler is argmax today) ----
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  /** Bumped when a stored value must be discarded; see loadSettings. */
  schemaVersion: number;
  // ---- advanced: KV warm-cache policy (acceptance P5) ----
  // Honesty note (2026-07 audit): today the engine's ONLY real KV-disk surface
  // is maintenance — `idletoken-worker --kv-clear [--kv-dir DIR]`. So kvDir is
  // real (it targets the clear action); kvMaxMb / kvTtlDays / kvEviction /
  // kvOffload have NO engine implementation yet (no size bound, no TTL/LRU
  // eviction, no live offload) and stay `reserved` in the panel until the
  // engine grows them. Do not fake flags for them (design philosophy 15).
  kvOffload: boolean; // reserved — engine has no live KV offload
  kvDir: string; // real: passed to `--kv-clear --kv-dir` (empty = platform dir)
  kvMaxMb: number; // reserved — engine enforces no size bound yet
  kvTtlDays: number; // reserved — engine has no TTL eviction yet
  kvEviction: KvEviction; // reserved — engine has no eviction policy yet
  // ---- advanced: network ----
  bindNic: string; // "auto" or a specific IP
  interStagePort: number;
  // ---- advanced: startup (Tauri autostart plugin) ----
  autostart: boolean;
  autoRejoin: boolean;
  // ---- advanced: privacy / diagnostics ----
  telemetry: boolean; // default off — local-first (philosophy 16)
  logLevel: LogLevel;
  experimental: boolean;

  // ---- appearance / UI details (client-side, real) ----
  uiScale: number; // 0 = auto (follow the window); otherwise a fixed factor, 1.0 = 100%
  density: Density;
  reduceMotion: boolean;
  accent: Accent;

  // ---- models & storage ----
  modelDir: string;
  autoDownload: boolean;
  verifySha: boolean;
  idleUnload: boolean;
  idleUnloadMin: number;

  // ---- cluster & discovery ----
  mdns: boolean;
  discoveryPort: number;
  manualPeers: string; // comma-separated IPs
  clusterName: string;
  heartbeatSec: number;
  preferCoordinator: boolean;
  sameSubnetOnly: boolean;

  // ---- API hardening ----
  apiStreaming: boolean;
  apiCors: string;
  apiRateLimit: number; // requests/min, 0 = off
  apiTimeoutSec: number;
  apiLocalOnly: boolean;
  apiRequestLog: boolean;

  // ---- power / thermal / scheduling (home shared machines) ----
  pauseOnGpuBusy: boolean; // pause contributing while a foreground app uses the GPU (gaming)
  scheduleEnabled: boolean;
  scheduleFrom: string; // "23:00"
  scheduleTo: string; // "07:00"
  powerLimitPct: number; // GPU power cap %, 100 = off
  tempLimitC: number; // throttle above °C, 0 = off
  pauseOnBattery: boolean;

  // ---- notifications ----
  notifyEnabled: boolean;
  notifyNodeChange: boolean;
  notifyReady: boolean;
  notifyErrors: boolean;
  notifyDownload: boolean;
  notifySound: boolean;

  // ---- window / tray ----
  closeToTray: boolean;
  startMinimized: boolean;
  rememberWindow: boolean;
  trayIcon: boolean;

  // ---- updates ----
  autoUpdate: boolean;
  updateChannel: UpdateChannel;

  // ---- data ----
  dataDir: string;

  // ---- account / platform (P2 cloud auth + P3 account-mode pairing) ----
  // Base URL of the platform gateway. Empty = local identity, fully offline
  // (code-mode pairing still works). Set = email auth against the platform.
  platformUrl: string;

  // ---- privacy protection (docs/privacy-design.md, gate G-PRIV) ----
  // Envelope encryption to the provider's cluster: workers only see hidden
  // states, never your text. On the marketplace the platform sees plaintext
  // (moderation + metering); on your own cluster this is not needed.
  privacyEncrypt: boolean; // enforced default; kept as config for the engine pipeline
  privacyEncryptAtRest: boolean; // encrypt KV/snapshots on disk (coordinator)
  privacyLockMemory: boolean; // mlock the plaintext buffer (coordinator)
  privacyPadding: boolean; // Obfuscator: pad request length (reserved)
  privacyDpNoise: boolean; // Obfuscator: differential-privacy noise (reserved)
  privacyDummyTokens: boolean; // Obfuscator: dummy tokens (reserved)
}

// The default platform address injected at build time (VITE_PLATFORM_URL). When
// nothing is injected it is the empty string, meaning a local offline identity.
// Release builds (pnpm build:release / tauri build) inject the production address
// through .env.release; a plain pnpm build (dev, acceptance gates) injects
// nothing, so gates such as P2_auth stay offline and deterministic.
// This only affects the default for a fresh install; settings the user has saved
// (including a deliberately cleared one) are never overwritten.
const BUILT_IN_PLATFORM_URL: string =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_PLATFORM_URL) || "";

export const DEFAULT_SETTINGS: AppSettings = {
  modelId: DEFAULT_MODEL_ID,
  quant: defaultQuant(DEFAULT_MODEL_ID),
  tier: 2,
  resourcePreset: "balanced",
  maxVramMb: 0,
  maxRamMb: 0,
  computeMode: "auto",
  weightsSource: "auto",
  ggufPath: "",
  apiHost: "0.0.0.0",
  apiPort: 8000,
  apiOpenAI: true,
  apiAnthropic: true,
  apiToken: "",
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  // Governs BOTH the engine's ceiling (--max-decode at launch) and what the
  // chat sends per request.
  //
  // 0 = no ceiling but the context, matching llama.cpp/Ollama. Chosen over a
  // finite default because truncation is a SILENT wrong answer — the user just
  // sees a reply that stops mid-sentence and concludes the model is bad —
  // whereas a long generation is visible, streaming, and interruptible (Stop).
  // The cost is real and accepted: a model that never emits EOS runs until the
  // context fills, which at ~13 tok/s and a 1M window is many hours of the
  // cluster. Anyone who wants a bound sets one here, or sends max_tokens.
  maxTokens: 0,
  // Literal, not SCHEMA_VERSION: that const is declared further down and this
  // object is built at module init. Keep the two in step by hand.
  schemaVersion: 2,
  kvOffload: false,
  kvDir: "",
  kvMaxMb: 1024,
  kvTtlDays: 3,
  kvEviction: "lru",
  bindNic: "auto",
  interStagePort: 14101,
  autostart: false,
  autoRejoin: false,
  telemetry: false,
  logLevel: "info",
  experimental: false,

  uiScale: 0, // auto
  density: "comfortable",
  reduceMotion: false,
  accent: "amber",

  modelDir: "",
  autoDownload: true,
  verifySha: true,
  idleUnload: false,
  idleUnloadMin: 15,

  mdns: true,
  discoveryPort: 14099,
  manualPeers: "",
  clusterName: "home",
  heartbeatSec: 5,
  preferCoordinator: false,
  sameSubnetOnly: true,

  apiStreaming: true,
  apiCors: "*",
  apiRateLimit: 0,
  apiTimeoutSec: 120,
  apiLocalOnly: false,
  apiRequestLog: false,

  pauseOnGpuBusy: false,
  scheduleEnabled: false,
  scheduleFrom: "23:00",
  scheduleTo: "07:00",
  powerLimitPct: 100,
  tempLimitC: 0,
  pauseOnBattery: true,

  notifyEnabled: true,
  notifyNodeChange: true,
  notifyReady: true,
  notifyErrors: true,
  notifyDownload: true,
  notifySound: true,

  closeToTray: true,
  startMinimized: false,
  rememberWindow: true,
  trayIcon: true,

  autoUpdate: true,
  updateChannel: "stable",

  dataDir: "",

  platformUrl: BUILT_IN_PLATFORM_URL,

  privacyEncrypt: true,
  privacyEncryptAtRest: true,
  privacyLockMemory: true,
  privacyPadding: false,
  privacyDpNoise: false,
  privacyDummyTokens: false,
};

export const APP_VERSION = "0.1.0-pre";

const KEY = "idletoken.settings";

// Bump when a stored value must be discarded rather than merged. Absent in
// blobs written before versioning existed, which reads as 0.
const SCHEMA_VERSION = 2;

// ---- UI scale --------------------------------------------------------------
// The fixed factors the panel offers. `0` means auto; anything else must be one
// of these so the select can round-trip the stored value.
export const UI_SCALE_STEPS = [0.9, 1, 1.15, 1.3] as const;

// Auto scale from the window size. Two rules decide the shape of this:
//   - it only ever grows. Narrow windows are already handled by the responsive
//     CSS (breakpoints down to 420px); zooming *out* on top of that would just
//     make text tiny on a phone-sized control client.
//   - it is banded, not continuous. A continuous function re-lays-out the whole
//     document on every resize tick — which is precisely the flicker the old
//     drag-a-slider control produced.
// CSS px already include the OS display scaling, so DPI is not ours to correct.
//
// Bands are cut on ONE number — how much room the window has relative to the
// reference layout below — because the hysteresis has to compare against a
// single quantity. (Thresholds on width AND height separately look equivalent
// and are not: a 1600x900 window is comfortably past the width edge but only
// 4.6% past the height one, so a both-dimensions margin could never confirm the
// step up and the scale stuck at 1.0 forever.)
const REF_W = 1500;
const REF_H = 860;
const SCALE_BANDS: { fit: number; scale: number }[] = [
  { fit: 1.25, scale: 1.2 },
  { fit: 1, scale: 1.1 },
  { fit: 0, scale: 1 },
];
const fitOf = (w: number, h: number) => Math.min(w / REF_W, h / REF_H);

/** Scale to use when `uiScale === 0`. Pass the previous result to get the
 *  hysteresis: a window edge parked exactly on a band boundary must not
 *  oscillate while it is dragged. */
export function autoUiScale(w: number, h: number, prev?: number): number {
  const fit = fitOf(w, h);
  const at = (margin: number) => SCALE_BANDS.find((b) => fit >= b.fit * margin)!.scale;
  const next = at(1);
  if (prev === undefined || next === prev) return next;
  // Leave the current band only once the window is 2% clear of the edge (~30px
  // at the 1500px reference) — enough dead zone that a resize drag cannot flip
  // back and forth, small enough that a window can still reach the next band
  // when only one dimension has room to spare.
  return at(next > prev ? 1.02 : 0.98) === next ? next : prev;
}

// Merge stored settings over defaults so older saved shapes gain new fields.
export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    const merged = { ...DEFAULT_SETTINGS, ...parsed };
    // v0 → v1: maxTokens was a hollow setting — the engine's ceiling was a
    // compiled-in 4096 and nothing read this field, so every stored value is a
    // stale default (512, then 4096, then 8192), never a deliberate choice.
    // Merging one would silently cap generation on exactly the machines that
    // have been running longest. Reset it once; from v1 on it is honoured.
    if ((parsed.schemaVersion ?? 0) < 1) merged.maxTokens = DEFAULT_SETTINGS.maxTokens;
    // v1 → v2: uiScale went from a free 0.8–1.4 slider to auto (0) + four fixed
    // steps. Exactly 1.0 was the old default, so it carries no intent — those
    // machines move to auto. A value the user actually dragged to is intent, so
    // it is kept, snapped to the nearest step it can now round-trip.
    if ((parsed.schemaVersion ?? 0) < 2 && typeof parsed.uiScale === "number") {
      merged.uiScale =
        parsed.uiScale === 1 || parsed.uiScale <= 0
          ? 0
          : UI_SCALE_STEPS.reduce((a, b) => (Math.abs(b - parsed.uiScale!) < Math.abs(a - parsed.uiScale!) ? b : a));
    }
    // The picker only lists models the engine can run, so a stored id outside
    // that set (hand-edited storage, a model withdrawn between releases) would
    // leave the list with nothing selected and no way to select anything.
    // Fall back rather than render a dead panel.
    if (!isAvailable(merged.modelId)) {
      merged.modelId = DEFAULT_MODEL_ID;
      merged.quant = defaultQuant(DEFAULT_MODEL_ID);
    }
    // Same for precision: qwen3-8b's BF16 row was removed on 2026-08-11 (the
    // file it named 404s), so a machine that had it selected would render a
    // <select> with no matching option — blank, and unchanged until touched.
    if (merged.quant && !quantOptions(merged.modelId).some((v) => v.quant === merged.quant)) {
      merged.quant = defaultQuant(merged.modelId);
    }
    merged.schemaVersion = SCHEMA_VERSION;
    return merged;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: AppSettings): void {
  localStorage.setItem(KEY, JSON.stringify(s));
}

// Engine tuning passed with pairing_create / pairing_join (task 1.2:
// AppSettings → engine CLI args). Serializable mirror of the Rust `Tuning`
// struct in src-tauri/src/pairing.rs — every field has a real engine/pairing
// destination:
//   apiHost + apiPort → coord `--api-bind host:port` (apiPort is also
//                       broadcast via the roster so joiners poll the right
//                       status port and show the right baseUrl)
//   apiToken          → coord `--api-token` (401 without it on the inference
//                       endpoints; empty = open on the LAN)
//   interStagePort    → worker `--bind 0.0.0.0:port` (the coordinator's
//                       co-located worker binds port+1 to avoid collision)
//   discoveryPort     → the pairing layer's UDP beacon port (both sides must
//                       be configured alike to find each other)
//   modelId           → coord `--model-id` (engine model registry; the
//                       coordinator decides the cluster's model and joiners
//                       adopt it via the roster broadcast)
//   quant             → coord `--quant` (selected precision; "" = the model's
//                       default variant. Joiners adopt it via the roster too)
//   ctxSize           → coord `--ctx-size` (tier → context window; feeds mode
//                       decision + per-node overhead in the layer split)
// Settings without a real engine implementation (KV size/TTL/eviction,
// computeMode, sampling defaults, ...) are deliberately NOT carried here —
// they stay `reserved` in the panel instead of being silently dropped.
export interface EngineTuning {
  apiHost: string;
  apiPort: number;
  apiToken: string;
  interStagePort: number;
  discoveryPort: number;
  modelId: string;
  quant: string;
  ctxSize: number;
  /** Per-request generation ceiling → coord `--max-decode`. 0 = context-bound. */
  maxDecode: number;
}

export function tierCtx(tier: Tier["id"]): number {
  return TIERS.find((t) => t.id === tier)?.ctx ?? 8192;
}

export function engineTuning(s: AppSettings): EngineTuning {
  return {
    apiHost: s.apiHost || "0.0.0.0",
    apiPort: s.apiPort || 8000,
    apiToken: s.apiToken,
    interStagePort: s.interStagePort || 14101,
    discoveryPort: s.discoveryPort || 14099,
    modelId: s.modelId || DEFAULT_MODEL_ID,
    quant: s.quant ?? "",
    ctxSize: tierCtx(s.tier),
    // The engine's per-request ceiling comes from the same setting the chat
    // sends, so the number the user typed is the number that governs — for
    // third-party API clients too, not just our own chat.
    maxDecode: s.maxTokens,
  };
}

// Resolve the effective usable caps (MiB) the engine should enforce, given the
// machine's totals (bytes). Presets derive from a fraction of total; "custom"
// uses the precise sliders; "max"/unknown totals => no cap.
export function effectiveCaps(
  s: AppSettings,
  totals: { vram_total: number; ram_total: number } | null
): { maxVramMb: number; maxRamMb: number } {
  if (s.resourcePreset === "custom") {
    return { maxVramMb: s.maxVramMb, maxRamMb: s.maxRamMb };
  }
  const f = PRESET_FRACTION[s.resourcePreset];
  if (!totals || f >= 1) return { maxVramMb: 0, maxRamMb: 0 };
  return {
    maxVramMb: Math.floor((totals.vram_total * f) / MiB),
    maxRamMb: Math.floor((totals.ram_total * f) / MiB),
  };
}

/** Whether settings have ever been saved on this machine. Automatic model
 *  selection on first start happens only when they have **not** -- a model the
 *  user picked deliberately must never be overwritten by what we think fits
 *  better. */
export function settingsWerePersisted(): boolean {
  try { return localStorage.getItem(KEY) !== null; } catch { return false; }
}
