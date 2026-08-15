// Engine/app settings that must really take effect (design philosophy 15).
// Split into a Simple set (what a home user needs) and an Advanced set (precise
// control). Persisted to localStorage and restored on launch. Theme + language
// live separately (pure UI state) but are surfaced in the Simple tab.
import { DEFAULT_MODEL_ID, LOCAL_GGUF_ID, defaultQuant, isAvailable, quantOptions } from "./models";

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
export type KvEviction = "lru" | "fifo";
export type Density = "comfortable" | "compact";
export type Accent = "amber" | "teal" | "violet" | "rose";
export type UpdateChannel = "stable" | "beta";

export interface AppSettings {
  // ---- simple ----
  modelId: string;
  quant: string; // selected precision ("" = model's default variant / single-precision models)
  // ---- open model intake (v2 WS-D1; modelId === LOCAL_GGUF_ID) -------------
  // Where the user-supplied GGUF comes from. Unlike the removed v3 "GGUF file
  // path" box, these are never handed to the engine unchecked: the file path
  // comes from a native dialog (a real, existing file), and the HF pair goes
  // through the same verified download machinery as curated weights.
  customSource: "file" | "hf";
  customGgufPath: string; // absolute path picked in the native dialog ("" = none)
  customHfRepo: string; // e.g. "unsloth/Qwen3.5-4B-GGUF" ("" = none)
  customHfFile: string; // exact .gguf file name inside the repo
  tier: Tier["id"];
  resourcePreset: ResourcePreset;
  // ---- advanced: resources (precise; used when resourcePreset === "custom") ----
  maxVramMb: number; // 0 = no cap
  maxRamMb: number; // 0 = no cap
  computeMode: ComputeMode; // reserved — engine auto-determines GPU_ONLY/HYBRID, no override flag yet
  // No "weights source" / "GGUF file path" any more (2026-08-13). Resolution is
  // policy, not preference — see resolveLocalWeights: a complete local copy is
  // used, a joiner streams its layers from the coordinator, everyone else
  // downloads. The removed "local file" box handed an unchecked path to the
  // engine, so a typo (or the empty box you get the moment you pick it) read as
  // "weights ready" and failed at load. Point `modelDir` at an existing folder
  // instead; that one is verified.
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
  // No `telemetry` field (removed 2026-08-13): nothing in this product collects
  // or sends anything, so there was nothing for it to gate. See SettingsPanel's
  // privacy category.
  // No `logLevel` / `experimental` (removed 2026-08-13): nothing read either.
  // The log level was worse than inert — it rode into the diagnostics bundle,
  // describing an engine that had never been told about it.

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
  // Storage key kept as `mdns` (renaming it would drop the stored value for no
  // gain); everything user-facing and the engine tuning call it what it is —
  // LAN auto-discovery over a UDP broadcast beacon. It was never mDNS.
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
  /** The one-time "still running in the tray" notice has been shown. Not a
   *  user-facing control — it only keeps the notice from repeating. */
  trayHintShown: boolean;

  // ---- updates ----
  autoUpdate: boolean;
  updateChannel: UpdateChannel;

  // No `dataDir` (removed 2026-08-13): the app-data location is Tauri's, and
  // the box never moved anything.

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
  customSource: "file",
  customGgufPath: "",
  customHfRepo: "",
  customHfFile: "",
  tier: 2,
  resourcePreset: "balanced",
  maxVramMb: 0,
  maxRamMb: 0,
  computeMode: "auto",
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
  schemaVersion: 4,
  kvOffload: false,
  kvDir: "",
  kvMaxMb: 1024,
  kvTtlDays: 3,
  kvEviction: "lru",
  bindNic: "auto",
  interStagePort: 14101,
  autostart: false,
  autoRejoin: false,

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
  // 1s, not the 5 it used to say: these six were hollow until 2026-08-13, so
  // the stored numbers described nothing. Now that the poll really uses it, the
  // default has to be the interval the client has always polled at — otherwise
  // wiring the setting would silently make every roster five times laggier.
  heartbeatSec: 1,
  preferCoordinator: false,
  // Off, for the same reason: enforcing it now would start rejecting the
  // cross-subnet meshes (Tailscale et al.) that pair fine today. It is a
  // restriction you opt into, not one that appears on upgrade.
  sameSubnetOnly: false,

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
  trayHintShown: false,

  autoUpdate: true,
  updateChannel: "stable",


  platformUrl: BUILT_IN_PLATFORM_URL,

  privacyEncrypt: true,
  privacyEncryptAtRest: true,
  privacyLockMemory: true,
  privacyPadding: false,
  privacyDpNoise: false,
  privacyDummyTokens: false,
};

// Displayed version. Source of truth is src-tauri/tauri.conf.json (`version`),
// which package.json mirrors — keep this literal in step with it. (Not read
// from the Tauri API because it renders synchronously in the About note and
// must also work in the browser dev build, where there is no shell to ask.)
export const APP_VERSION = "0.1.0";

const KEY = "idletoken.settings";

// Bump when a stored value must be discarded rather than merged. Absent in
// blobs written before versioning existed, which reads as 0.
const SCHEMA_VERSION = 4;

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
/**
 * A fresh local API token: 32 hex chars from the platform CSPRNG.
 *
 * Not Math.random: this is the only thing between a machine that can spend
 * Sparks and everyone else on the LAN (docs/api-surface.md §5.3). Falls back to
 * Math.random ONLY where crypto is unavailable, which in Tauri and every
 * browser we ship to is nowhere — the branch exists so a test harness without
 * webcrypto degrades instead of throwing on launch.
 */
export function generateApiToken(): string {
  const c: Crypto | undefined = typeof crypto !== "undefined" ? crypto : undefined;
  if (c?.getRandomValues) {
    const b = new Uint8Array(16);
    c.getRandomValues(b);
    return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  }
  let s = "";
  while (s.length < 32) s += Math.floor(Math.random() * 16).toString(16);
  return s.slice(0, 32);
}

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    // Fresh install: give the local API a token instead of leaving it open on
    // the LAN (docs/api-surface.md §3 decision 3). Generated here rather than
    // in DEFAULT_SETTINGS because that object is a shared constant — one token
    // baked into it would be the same token on every machine, which is worse
    // than none.
    if (!raw) return { ...DEFAULT_SETTINGS, apiToken: generateApiToken() };
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
    // v2 → v3: the six pairing/discovery settings (mdns, manualPeers,
    // heartbeatSec, preferCoordinator, sameSubnetOnly, bindNic) became real on
    // 2026-08-13. Until then nothing read them, so a stored value is a stale
    // default and not a choice — the same reasoning as maxTokens in v0 → v1,
    // and the same fix. Honouring them would be worse than ignoring them ever
    // was: a machine that has been running since v1 would suddenly poll every
    // 5s and refuse peers from another subnet, neither of which anyone asked
    // for. Reset once; from v3 on they are obeyed.
    if ((parsed.schemaVersion ?? 0) < 3) {
      merged.mdns = DEFAULT_SETTINGS.mdns;
      merged.manualPeers = DEFAULT_SETTINGS.manualPeers;
      merged.heartbeatSec = DEFAULT_SETTINGS.heartbeatSec;
      merged.preferCoordinator = DEFAULT_SETTINGS.preferCoordinator;
      merged.sameSubnetOnly = DEFAULT_SETTINGS.sameSubnetOnly;
      merged.bindNic = DEFAULT_SETTINGS.bindNic;
    }
    // v3 → v4: "weights source" is gone. Someone running with "local file" had
    // a GGUF somewhere the automatic path does not look, so send the folder
    // along instead of quietly telling them their weights are missing — the
    // directory is what the resolver searches now, and unlike the old free-text
    // path it is verified before anything is claimed about it.
    if ((parsed.schemaVersion ?? 0) < 4) {
      const legacy = parsed as { weightsSource?: string; ggufPath?: string };
      if (legacy.weightsSource === "local" && legacy.ggufPath && !merged.modelDir) {
        const cut = Math.max(legacy.ggufPath.lastIndexOf("/"), legacy.ggufPath.lastIndexOf("\\"));
        if (cut > 0) merged.modelDir = legacy.ggufPath.slice(0, cut);
      }
    }
    // The picker only lists models the engine can run, so a stored id outside
    // that set (hand-edited storage, a model withdrawn between releases) would
    // leave the list with nothing selected and no way to select anything.
    // Fall back rather than render a dead panel. The open-intake sentinel is
    // valid exactly when its source details survived — a bare sentinel with no
    // file/repo would render a selection that cannot start anything.
    const customOk =
      merged.modelId === LOCAL_GGUF_ID &&
      (merged.customSource === "hf"
        ? !!(merged.customHfRepo && merged.customHfFile)
        : !!merged.customGgufPath);
    if (!isAvailable(merged.modelId) && !customOk) {
      merged.modelId = DEFAULT_MODEL_ID;
      merged.quant = defaultQuant(DEFAULT_MODEL_ID);
    }
    // Same for precision: qwen3-8b's BF16 row was removed on 2026-08-11 (the
    // file it named 404s), so a machine that had it selected would render a
    // <select> with no matching option — blank, and unchanged until touched.
    if (merged.quant && !quantOptions(merged.modelId).some((v) => v.quant === merged.quant)) {
      merged.quant = defaultQuant(merged.modelId);
    }
    // Deliberately NOT migrated: an existing install with an empty apiToken
    // keeps it empty. Filling one in on upgrade would 401 every curl, script
    // and Claude Code config that machine already had working, and the user
    // would have no idea why — the app they left running overnight simply
    // stopped answering. New installs are closed by default (above); old ones
    // get closed the moment it actually matters, when overflow routing is
    // switched on and the box can spend Sparks (docs/api-surface.md §5.3).
    merged.schemaVersion = SCHEMA_VERSION;
    return merged;
  } catch {
    // Unreadable storage is a fresh start in every way that matters, so it gets
    // a token too — otherwise a corrupt file silently reopens the API.
    return { ...DEFAULT_SETTINGS, apiToken: generateApiToken() };
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
  /** This machine's usage caps (MiB, 0 = no cap) → worker `--max-vram-mb` /
   *  `--max-ram-mb`. Per-machine, unlike the model/ctx settings: it is the
   *  answer to "how much of MY computer may IdleToken use", so each node
   *  passes its own and joiners never adopt the creator's. */
  maxVramMb: number;
  maxRamMb: number;
  // ---- pairing behaviour (client-side, consumed by src-tauri/src/pairing.rs) --
  /** Announce/listen for the UDP discovery beacon. Off = this machine is found
   *  (or finds others) only through `manualPeers`. */
  lanDiscovery: boolean;
  /** Comma-separated IPs to try directly when the beacon finds nothing. */
  manualPeers: string;
  /** Roster poll interval, seconds. Lower = the member list and the per-machine
   *  progress update faster; higher = less LAN chatter. */
  heartbeatSec: number;
  /** Ask the creator to hand this machine the coordinator role on join. */
  preferCoordinator: boolean;
  /** Refuse peers whose IPv4 is outside this machine's /24. */
  sameSubnetOnly: boolean;
  /** "auto" or an IPv4: which interface cluster traffic binds to and which
   *  address this machine advertises to the others. */
  bindNic: string;
}

export function tierCtx(tier: Tier["id"]): number {
  return TIERS.find((t) => t.id === tier)?.ctx ?? 8192;
}

/**
 * The settings the engine is actually told about.
 *
 * `caps` is a REQUIRED argument rather than something derived from `s` here,
 * because the presets are a fraction of the machine's totals and only the
 * caller knows them (they come from the probe). Making it optional would let a
 * call site quietly launch an uncapped engine, which is exactly the bug this
 * parameter was added to fix: until 2026-08-13 the caps reached the probe and
 * nothing else, so "This machine's usage" moved the numbers on the dashboard
 * while the running cluster helped itself to the whole machine.
 */
export function engineTuning(
  s: AppSettings,
  caps: { maxVramMb: number; maxRamMb: number }
): EngineTuning {
  return {
    maxVramMb: caps.maxVramMb,
    maxRamMb: caps.maxRamMb,
    lanDiscovery: s.mdns,
    manualPeers: s.manualPeers,
    // Clamped where it is read (pairing.rs) too; here it just keeps a 0 from a
    // hand-edited store out of a sleep loop.
    heartbeatSec: Math.max(1, Math.min(60, s.heartbeatSec || 1)),
    preferCoordinator: s.preferCoordinator,
    sameSubnetOnly: s.sameSubnetOnly,
    bindNic: s.bindNic,
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
