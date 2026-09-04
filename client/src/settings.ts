// Engine/app settings that must really take effect (design philosophy 15).
// Split into a Simple set (what a home user needs) and an Advanced set (precise
// control). Persisted to localStorage and restored on launch. Theme + language
// live separately (pure UI state) but are surfaced in the Simple tab.
import { version as OFFICIAL_CLIENT_VERSION } from "../package.json";
import { DEFAULT_MODEL_ID, defaultQuant, getManifest, isAvailable, quantOptions } from "./models";

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
  { id: 4, ctx: 262144 },
  // Stored-schema compatibility only: the tier selector is retired and v8+
  // migrates to tier 0. A stale tier 5 must not enable 1M without the checkbox.
  { id: 5, ctx: 262144 },
];

/** The three product contexts (docs/ctx-tiers-2026-09.md). 128K is the default;
 * 256K and 1M are explicit choices that change estimation, runtime admission
 * and marketplace service identity together. There is no automatic down-sizing
 * between them — a window that does not fit is refused, never shrunk.
 *
 * Why 128K and not 256K (2026-09-02): llama.cpp allocates the whole `-c` block
 * at load and cannot grow it, so the window is PRE-PAID whether a session uses
 * it or not. Claude Code's own window is 200K and it auto-compacts around
 * 150-165K, so the core scenario never reaches 256K; the average agentic prompt
 * is single-digit thousands of tokens. 256K remains one click away for the
 * sessions that genuinely need it.
 *
 * MUST match IDLETOKEN_CTX_TIER_* in include/idletoken_plan.h: the planner has
 * measurements for exactly these windows and refuses anything else. */
export const CONTEXT_TIERS = [131072, 262144, 1048576] as const;
export type ContextTier = (typeof CONTEXT_TIERS)[number];
export const DEFAULT_CONTEXT_TOKENS: ContextTier = 131072;
export const LONG_CONTEXT_TOKENS: ContextTier = 1048576;
export const PRODUCT_CONTEXT_CAP = LONG_CONTEXT_TOKENS;

/** A stored/queried value snapped to a real tier. Anything unrecognised falls
 * back to the default rather than reaching the engine, which would refuse it —
 * the picker is the only place a window is chosen, so a bad stored value is a
 * corrupted setting, not a user intent to honour. */
export function asContextTier(v: unknown): ContextTier {
  return (CONTEXT_TIERS as readonly number[]).includes(Number(v))
    ? (Number(v) as ContextTier)
    : DEFAULT_CONTEXT_TOKENS;
}

// ---- context bounded by the model -----------------------------------------
// `tier` is stored-schema compatibility only. Runtime context is now exactly
// 256K or the explicit 1M opt-in; the coordinator never silently sizes down.
export const MODEL_DEFAULT_TIER = 0;

/** The window this model would really launch for the requested tier: the tier
 * itself, or the model's own ceiling when that is lower. The planner's
 * `idletoken_llama_ctx_tier_of()` maps a below-tier ceiling back to the slot
 * holding its measurement, so a clamped value is still budgeted correctly. */
export function modelCtxMax(modelId: string, want: number = DEFAULT_CONTEXT_TOKENS): number {
  try {
    const m = getManifest(modelId || DEFAULT_MODEL_ID);
    return Math.min(want, Math.max(m.context_max || 8192, m.context_yarn_max || 0));
  } catch {
    return 8192;
  }
}

/** Which tiers this model can actually be asked for. A tier above the model's
 * ceiling is not offered — the picker hides it rather than showing an option
 * that silently clamps to something else. */
export function contextTiersFor(modelId: string): ContextTier[] {
  const ceil = modelCtxCeil(modelId);
  const usable = CONTEXT_TIERS.filter((t) => t <= ceil);
  // Every model reaches at least the smallest tier in practice, but a manifest
  // with a tiny ceiling must still offer one choice rather than an empty row.
  return usable.length ? [...usable] : [CONTEXT_TIERS[0]];
}

/** Alias used by explicit legacy tiers; mirrors modelCtxMax(). */
export function modelCtxCeil(modelId: string): number {
  try {
    const m = getManifest(modelId || DEFAULT_MODEL_ID);
    return Math.min(
      PRODUCT_CONTEXT_CAP,
      Math.max(m.context_max || 8192, m.context_yarn_max || 0),
    );
  } catch {
    return 8192;
  }
}

export function modelSupportsLongContext(modelId: string): boolean {
  return modelCtxCeil(modelId) >= LONG_CONTEXT_TOKENS;
}

/** The exact context sent to the engine. The caller picks one of the three
 * tiers; model metadata is a safety ceiling, never a reason to invent an
 * intermediate runtime window. */
export function effectiveCtx(s: Pick<AppSettings, "modelId" | "ctxTokens">): number {
  return modelCtxMax(s.modelId, asContextTier(s.ctxTokens));
}

// ---- KV cache precision ----------------------------------------------------
// Retired from the UI on 2026-08-25 (docs/ctx-kv-simplification-2026-08.md):
// the KV dtype is the COORDINATOR's automatic rule now — q4_0 for 1-2 bit
// weights, q8_0 for every other quantized tier (3-15 bit, Q8 included since
// 2026-09-02), f16 only for unquantized BF16/F16 and unreadable quant names.
// The client passes nothing; the escape hatch
// for measurements is the coordinator's IDLETOKEN_KV_CACHE_TYPE env, on
// purpose not a setting. The dtype table, the per-token estimator and
// recommendKvCache() that used to live here duplicated the coordinator's
// planner and are gone with the selector that consumed them.

export type ResourcePreset = "conservative" | "balanced" | "max" | "custom";
// Fraction of a machine's total the preset lets IdleToken use (max = no cap).
export const PRESET_FRACTION: Record<Exclude<ResourcePreset, "custom">, number> = {
  conservative: 0.5,
  balanced: 0.75,
  max: 1,
};

export type KvEviction = "lru" | "fifo";
export type Density = "comfortable" | "compact";
export type Accent = "amber" | "teal" | "violet" | "rose";
export type UpdateChannel = "stable" | "beta";

export interface AppSettings {
  // ---- simple ----
  modelId: string;
  quant: string; // selected precision ("" = model's default variant / single-precision models)
  // No open-intake fields any more (customSource/customGgufPath/customHfRepo/
  // customHfFile, removed 2026-08-15 with the open model intake): the curated
  // registry is the whole selectable set, and a stored "local-gguf" selection
  // migrates back to the default model below.
  /** Retired context-tier schema field. Runtime context comes exclusively from
   *  `ctxTokens`. */
  tier: Tier["id"] | 0;
  /** The chosen context window, one of CONTEXT_TIERS. Default 131072.
   *  Replaced the `longContext` boolean on 2026-09-02 when the product went
   *  from two windows to three (docs/ctx-tiers-2026-09.md). */
  ctxTokens: ContextTier;
  resourcePreset: ResourcePreset;
  // ---- advanced: resources (precise; used when resourcePreset === "custom") ----
  maxVramMb: number; // 0 = no cap
  maxRamMb: number; // legacy stored field; serving capacity is GPU memory only
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
  // Empty = no auth, and that is the shipped default: the API serves loopback
  // only (2026-08-15) and browser-origin requests are refused by the
  // coordinator's Origin check, so a token guards nothing a local program
  // could not read from this same store. An operator who sets one still gets
  // it enforced.
  apiToken: string;
  // ---- marketplace: two independent choices ------------------------------
  // Providing and borrowing used to share one switch. They are intentionally
  // independent now: a machine may earn Sparks without spending them, or ask
  // for help without accepting other people's prompts.
  providerEnabled: boolean;
  overflowEnabled: boolean;
  /** Stored-schema compatibility for the coordinator's optional operator cap.
   *  The product always sends 0: the account balance is the spend gate. */
  overflowDailyCapMilli: number;
  /** Internal coordinator compatibility field. The public product policy is
   *  fixed at 0: a locally queued request starts asking for shared compute
   *  immediately, then keeps both local and marketplace completion available. */
  overflowWaitS: number;
  /** The provider name this machine registered on the platform, written by
   *  the sharing toggle: `cluster-<N>`, N = the smallest free number
   *  among the account's providers. Per-machine because the gateway dedupes
   *  providers by (account, name) — two machines sharing the bare cluster
   *  name upserted into ONE row and fought over its pubkey (found
   *  2026-08-21: an account's second machine never appeared in the portal).
   *  The suffix is a NUMBER, not the hostname: this name shows on the public
   *  marketplace, and hostnames carry real names and machine identities
   *  (owner's call, same night). Chosen once, then reused verbatim by resume
   *  and delist. Not rendered. */
  providerName: string;
  /** The platform API key borrowing is billed to. Minted on first use and kept
   *  so the switch does not create a new key every time it is flipped -- an
   *  account slowly filling with abandoned keys is worse than one key the user
   *  can see and revoke. Stored beside apiToken, which is already a secret in
   *  the same store. */
  overflowKey: string;
  // ---- advanced: default sampling (interface-reserved; sampler is argmax today) ----
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  /** Bumped when a stored value must be discarded; see loadSettings. */
  schemaVersion: number;
  // ---- retired KV-cache settings (stored-schema compatibility only) ----
  // The product has no persistent disk KV cache and exposes none of these in
  // the UI. The running engine may reuse hot prefixes in-process; stopping the
  // service destroys that state. A future host-RAM LRU needs an explicit byte
  // cap, eviction tests, and stop-time clearing before any field can return.
  kvOffload: boolean;
  kvDir: string;
  // ---- KV cache precision: RETIRED 2026-08-25 (ctx-kv-simplification) ----
  /** Stored for schema compatibility only; the v8 migration pins both to ""
   *  and nothing reads them. The dtype is the coordinator's automatic
   *  weight-tier rule; the escape hatch is its IDLETOKEN_KV_CACHE_TYPE env. */
  kvCacheK: string;
  kvCacheV: string;
  kvMaxMb: number;
  kvTtlDays: number;
  kvEviction: KvEviction;
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
  //
  // NONE of the fields in this block is wired to anything (checked 2026-08-19:
  // no reader outside this file, and SettingsPanel renders an explicit key list
  // that does not include them). They are kept, not rendered: persisted
  // settings from older builds already carry them, and dropping the fields
  // would mean a schema migration to buy nothing.
  //
  // pauseOnGpuBusy specifically: the honest implementation is worker-side --
  // detect foreground GPU use and stop accepting work -- and only Windows has a
  // reliable foreground-GPU signal (DXGI / performance counters). That is its
  // own piece of work, scheduled with the sharing side, not a checkbox. Do not
  // surface any of these in the UI until the behaviour behind them exists: a
  // toggle that does nothing is worse than an absent one.
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

/** Zero is the coordinator's explicit spelling for no additional local cap. */
export const OVERFLOW_UNCAPPED_MILLI = 0;

export const DEFAULT_SETTINGS: AppSettings = {
  modelId: DEFAULT_MODEL_ID,
  quant: defaultQuant(DEFAULT_MODEL_ID),
  // Exact 256K by default. Long context is a separate explicit service choice.
  tier: MODEL_DEFAULT_TIER,
  ctxTokens: DEFAULT_CONTEXT_TOKENS,
  // Full power by default (2026-08-15, was "balanced"): the product's whole
  // promise is using this machine's idle capacity, and a fresh install that
  // silently keeps 25% back both underuses the hardware and misreports what
  // the machine could serve. Whoever needs headroom turns it down knowingly.
  resourcePreset: "max",
  maxVramMb: 0,
  maxRamMb: 0,
  apiHost: "127.0.0.1",
  apiPort: 8000,
  apiOpenAI: true,
  apiAnthropic: true,
  apiToken: "",
  // Both directions are explicit, independent opt-ins.
  providerEnabled: false,
  overflowEnabled: false,
  // No hidden local spend ceiling. The platform balance is the hard gate.
  overflowDailyCapMilli: OVERFLOW_UNCAPPED_MILLI,
  overflowWaitS: 0,
  overflowKey: "",
  providerName: "",
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
  // context fills, which at ~13 tok/s and a 256K window is still many hours of the
  // cluster. Anyone who wants a bound sets one here, or sends max_tokens.
  maxTokens: 0,
  // Literal, not SCHEMA_VERSION: that const is declared further down and this
  // object is built at module init. Keep the two in step by hand.
  schemaVersion: 12,
  kvOffload: false,
  kvDir: "",
  kvCacheK: "",
  kvCacheV: "",
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

// The About note renders synchronously and must also work in the browser dev
// build, so it shares the browser build's existing release metadata instead
// of maintaining a second handwritten version string.
export const APP_VERSION = OFFICIAL_CLIENT_VERSION;

const KEY = "idletoken.settings";

// Bump when a stored value must be discarded rather than merged. Absent in
// blobs written before versioning existed, which reads as 0.
const SCHEMA_VERSION = 13;

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
    // A fresh install gets NO token (2026-08-21). It used to mint one here
    // "instead of leaving it open on the LAN" — but the API has not been
    // reachable from the LAN since 2026-08-16, when the coordinator started
    // rewriting any non-loopback bind to 127.0.0.1. What the token was left
    // guarding was a program on this machine, which it cannot guard against:
    // that program can read this very file. The cost was real — a new user
    // pointing Claude Code, Codex or a WebUI at their own machine got 401 and
    // had to go find a key — so the local API is now open on its port, the way
    // every other local inference server works.
    // The browser, the one caller that can reach a loopback port without being
    // able to read this file, is handled where it can be: api_origin_ok() in
    // the coordinator refuses requests that carry an Origin header.
    // `apiToken` stays a setting: an operator who wants one sets it (or
    // IDLETOKEN_API_TOKEN) and the coordinator still enforces it.
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
    // v4 → v5: adopt the built-in platform address when the stored one is
    // empty. The "never overwrite a saved setting" rule above assumed the user
    // could have cleared it deliberately — but there has never been a
    // platformUrl field anywhere in the UI, so an empty stored value is a
    // pre-feature default, not a choice (the same reasoning as maxTokens in
    // v0 → v1). The cost of honouring it was total: on every upgraded install
    // the sign-in silently used the LOCAL identity backend (valid platform
    // credentials answered "wrong email or password") and the whole
    // sharing/earnings section was hidden (found on a test machine,
    // 2026-08-21).
    // A NON-empty stored URL is still kept: someone running a self-hosted
    // gateway put it there by hand-editing storage or importing settings, and
    // that IS deliberate.
    if ((parsed.schemaVersion ?? 0) < 5 && !merged.platformUrl.trim()) {
      merged.platformUrl = DEFAULT_SETTINGS.platformUrl;
    }
    // v5 → v6: the daily borrow cap stopped being a setting (owner's call,
    // 2026-08-21 — borrowing may spend the whole balance; the balance is the
    // ceiling). Stored values came from a cap field that no longer exists
    // anywhere in the UI, so they are stale mechanics, not choices — same
    // reasoning as maxTokens in v0 → v1. Overwrite unconditionally.
    if ((parsed.schemaVersion ?? 0) < 6) {
      merged.overflowDailyCapMilli = OVERFLOW_UNCAPPED_MILLI;
    }
    // v6 → v7: the one build that derived providerName from the HOSTNAME
    // (0.1.16, test machines only) put that hostname on the public
    // marketplace. Clear it so the next toggle regenerates a random-suffix
    // identity; the machine re-registers under the new name.
    if ((parsed.schemaVersion ?? 0) < 7 && merged.providerName) {
      merged.providerName = "";
    }
    // v7 → v8: the context tier ladder and the KV precision selectors left
    // the UI (docs/ctx-kv-simplification-2026-08.md). The KV dtype is the
    // coordinator's automatic weight-tier rule.
    // A stored tier or dtype was set
    // against knobs that no longer exist — honouring it would silently pin
    // yesterday's trade-off on exactly the machines that ran the old build.
    if ((parsed.schemaVersion ?? 0) < 8) {
      merged.tier = MODEL_DEFAULT_TIER;
      merged.kvCacheK = "";
      merged.kvCacheV = "";
    }
    // v8 → v9: cluster names left the product. Remove the legacy property
    // unconditionally so imported or hand-edited settings cannot silently
    // bring the retired pairing split back. The account pairing protocol keeps
    // its own fixed compatibility bytes; no user setting feeds them.
    delete (merged as AppSettings & { clusterName?: unknown }).clusterName;
    // v9 → v10: accepting work and requesting help are separate choices. An
    // existing ON switch enabled both directions, so preserve that behaviour
    // on upgrade; explicitly present new fields win for development builds
    // that wrote the split shape before the schema bump.
    const legacyMarket = parsed as Partial<AppSettings> & { sharingEnabled?: unknown };
    if ((parsed.schemaVersion ?? 0) < 10) {
      const legacyOn = legacyMarket.sharingEnabled === true;
      merged.providerEnabled =
        typeof legacyMarket.providerEnabled === "boolean" ? legacyMarket.providerEnabled : legacyOn;
      merged.overflowEnabled =
        typeof legacyMarket.overflowEnabled === "boolean" ? legacyMarket.overflowEnabled : legacyOn;
    }
    // v10 → v11: context became an explicit two-state product choice. Existing
    // installs stay on the safe/default 256K service until the user checks 1M.
    // v11 → v12: two windows became three and the default dropped to 128K
    // (docs/ctx-tiers-2026-09.md). The boolean is translated, not discarded: a
    // user who explicitly asked for 1M keeps 1M. Everyone else lands on the new
    // 128K default rather than the old 256K one — a SILENT REDUCTION, which is
    // why the picker shows the current window instead of hiding it behind a
    // checkbox that only says "long".
    const legacyLongCtx = parsed as Partial<AppSettings> & { longContext?: unknown };
    if ((parsed.schemaVersion ?? 0) < 12) {
      merged.ctxTokens =
        legacyLongCtx.longContext === true ? LONG_CONTEXT_TOKENS : DEFAULT_CONTEXT_TOKENS;
    }
    // v12 → v13: remove the compatibility-era pseudo-unlimited value
    // (2,000,000,000 milli-credits). The coordinator now represents the
    // product rule exactly: zero means no local cap; the Spark balance is the
    // only default spend gate. This field never had a UI, so no user choice is
    // overwritten.
    if ((parsed.schemaVersion ?? 0) < 13) {
      merged.overflowDailyCapMilli = OVERFLOW_UNCAPPED_MILLI;
    }
    // Unconditional: a hand-edited or imported file can carry any number, and
    // a window the engine has no measurement for is refused at launch.
    merged.ctxTokens = asContextTier(merged.ctxTokens);
    delete (merged as AppSettings & { longContext?: unknown }).longContext;
    merged.overflowWaitS = 0;
    delete (merged as AppSettings & { sharingEnabled?: unknown }).sharingEnabled;
    // The local cap is not a product setting. Keep imported/current-schema
    // values from silently restoring the retired shadow limit.
    merged.overflowDailyCapMilli = OVERFLOW_UNCAPPED_MILLI;
    // Long context stopped being a separate model SKU on 2026-08-29. Preserve
    // the selected weights/precision while folding stored `*-1m` ids back into
    // the one real model. This is unconditional because users may already have
    // saved the current schema version.
    if (merged.modelId.endsWith("-1m")) {
      const baseId = merged.modelId.slice(0, -3);
      if (isAvailable(baseId)) merged.modelId = baseId;
    }
    // The picker only lists models the engine can run, so a stored id outside
    // that set (hand-edited storage, a model withdrawn between releases, or
    // the "local-gguf" sentinel of the open intake removed 2026-08-15) would
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
    // Deliberately NOT migrated: an existing install with an empty apiToken
    // keeps it empty. Filling one in on upgrade would 401 every curl, script
    // and Claude Code config that machine already had working, and the user
    // would have no idea why — the app they left running overnight simply
    // stopped answering. Nor are new installs any different: since 2026-08-21
    // nothing mints a token at all (see the block above), so this is not a
    // migration exception, it is the same open-by-default posture reached from
    // the other direction. What guards the port is api_origin_ok() in the
    // coordinator; platform spending is bounded by the account balance, with
    // an optional per-key cap only when the user explicitly configures one.
    merged.schemaVersion = SCHEMA_VERSION;
    return merged;
  } catch {
    // Unreadable storage is a fresh start in every way that matters, and a
    // fresh start no longer means a token — see the branch above.
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
//   apiHost + apiPort → coord `--api-bind host:port`. The host is always
//                       127.0.0.1 since 2026-08-15 — the API serves its own
//                       machine only (coord enforces it).
//   apiToken          → coord `--api-token` (401 without it on the inference
//                       endpoints; empty = open, which only loopback callers
//                       can reach anyway)
//   interStagePort    → worker `--bind 0.0.0.0:port` (the coordinator's
//                       co-located worker binds port+1 to avoid collision)
//   discoveryPort     → the pairing layer's UDP beacon port (both sides must
//                       be configured alike to find each other)
//   modelId           → coord `--model-id` (engine model registry; the
//                       coordinator decides the cluster's model and joiners
//                       adopt it via the roster broadcast)
//   quant             → coord `--quant` (selected precision; "" = the model's
//                       default variant. Joiners adopt it via the roster too)
//   ctxSize           → coord `--ctx-size` (exact 256K or explicit 1M; feeds
//                       GPU admission and KV sizing without a fallback ladder)
// Settings without a real engine implementation (KV size/TTL/eviction,
// sampling defaults, ...) are deliberately NOT carried here —
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
  /** KV cache dtypes → coord env IDLETOKEN_KV_CACHE_TYPE / _V. Always "" since
   *  2026-08-25: the coordinator decides from the weight tier — q4_0 at 1-2 bit,
   *  q8_0 at every other quantized tier, f16 only for unquantized/unknown
   *  (docs/ctx-kv-simplification-2026-08.md). The fields stay so the Rust
   *  Tuning struct keeps its shape. */
  kvCacheK: string;
  kvCacheV: string;
  /** Per-request generation ceiling → coord `--max-decode`. 0 = context-bound. */
  maxDecode: number;
  /** This machine's usage cap (MiB, 0 = no cap) → worker `--max-vram-mb`.
   *  Unlike the model/ctx settings, the VRAM cap is the
   *  answer to "how much of MY computer may IdleToken use", so each node
   *  passes its own and joiners never adopt the creator's. */
  maxVramMb: number;
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
  // ---- overflow (borrow when full) -> coord --overflow-* ------------------
  //
  // Launch parameters, not live controls: the coordinator reads them once, at
  // start. Turning Request help on therefore restarts the engine, which is the same
  // thing switching models does and is described to the user in the same words
  // -- there is no hot swap and pretending otherwise would leave the switch
  // showing a state the engine is not in.
  /** Platform base URL, or "" for off. Empty here means no --overflow-* flags
   *  reach the coordinator at all. */
  overflowUrl: string;
  /** The account key the borrowed time is billed to. Sealed inside the
   *  envelope by the coordinator, never sent as a header. */
  overflowKey: string;
  overflowWaitS: number;
  overflowDailyCapMilli: number;
}

/** The launch-time subset that may change while a roster is already forming.
 *
 * Pairing freezes the rest of EngineTuning when the cluster is created: model,
 * ports, NIC and resource caps are already part of the roster contract. The
 * top-right Request help button may change after the roster forms, so these
 * fields must be refreshed at the instant the creator starts the engines. */
export interface OverflowTuning {
  /** Explicit backend gate. When false, stale credentials are ignored. */
  enabled: boolean;
  overflowUrl: string;
  overflowKey: string;
  overflowWaitS: number;
  overflowDailyCapMilli: number;
}

export function overflowTuning(s: AppSettings): OverflowTuning {
  const enabled = s.overflowEnabled && !!s.overflowKey;
  return {
    enabled,
    // Keep credentials empty in the off state as defence in depth; Rust also
    // enforces `enabled` so a stale or hand-written payload cannot route.
    overflowUrl: enabled ? s.platformUrl.trim().replace(/\/+$/, "") : "",
    overflowKey: enabled ? s.overflowKey : "",
    // Start shared-compute attempts as soon as this request enters the local queue.
    overflowWaitS: 0,
    overflowDailyCapMilli: OVERFLOW_UNCAPPED_MILLI,
  };
}

export function tierCtx(tier: Tier["id"] | 0): number {
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
  caps: { maxVramMb: number }
): EngineTuning {
  const overflow = overflowTuning(s);
  return {
    maxVramMb: caps.maxVramMb,
    lanDiscovery: s.mdns,
    manualPeers: s.manualPeers,
    // Clamped where it is read (pairing.rs) too; here it just keeps a 0 from a
    // hand-edited store out of a sleep loop.
    heartbeatSec: Math.max(1, Math.min(60, s.heartbeatSec || 1)),
    preferCoordinator: s.preferCoordinator,
    sameSubnetOnly: s.sameSubnetOnly,
    bindNic: s.bindNic,
    // Always loopback (2026-08-15): the API serves its own machine only, and
    // both the Rust spawn path and the coordinator itself normalize anything
    // else — the stored value (old installs may hold "0.0.0.0") is ignored.
    apiHost: "127.0.0.1",
    apiPort: s.apiPort || 8000,
    apiToken: s.apiToken,
    interStagePort: s.interStagePort || 14101,
    discoveryPort: s.discoveryPort || 14099,
    modelId: s.modelId || DEFAULT_MODEL_ID,
    quant: s.quant ?? "",
    // Exact selected service window: 256K by default or explicit 1M. Runtime
    // may refuse it for insufficient VRAM but never silently reduces it.
    ctxSize: effectiveCtx(s),
    // Always empty (2026-08-25): the coordinator's auto rule decides the KV
    // dtype where the memory plan is made, so the plan and the engine's
    // -ctk/-ctv can never disagree. See the field comment above.
    kvCacheK: "",
    kvCacheV: "",
    // The engine's per-request ceiling comes from the same setting the chat
    // sends, so the number the user typed is the number that governs — for
    // third-party API clients too, not just our own chat.
    maxDecode: s.maxTokens,
    // Overflow reaches the coordinator only when Request help is on AND a key
    // has been minted for it. Providing work is deliberately unrelated.
    //
    // This stays the URL as the user configured it (usually https://). The
    // coordinator has no TLS client, so the spawn layer translates it to the
    // gateway's plaintext spelling (engine_platform_url in engine.rs) — the
    // translation lives in Rust because both launch paths (coord overflow and
    // the platform agent) pass through there, and a second copy here would be
    // the kind that drifts.
    overflowUrl: overflow.overflowUrl,
    overflowKey: overflow.overflowKey,
    overflowWaitS: overflow.overflowWaitS,
    overflowDailyCapMilli: overflow.overflowDailyCapMilli,
  };
}

// Resolve the effective usable caps (MiB) the engine should enforce, given the
// machine's totals (bytes). Presets derive from a fraction of total; "custom"
// uses the precise sliders; "max"/unknown totals => no cap.
export function effectiveCaps(
  s: AppSettings,
  totals: { vram_total: number; ram_total: number } | null
): { maxVramMb: number } {
  if (s.resourcePreset === "custom") {
    return { maxVramMb: s.maxVramMb };
  }
  const f = PRESET_FRACTION[s.resourcePreset];
  if (!totals || f >= 1) return { maxVramMb: 0 };
  return {
    maxVramMb: Math.floor((totals.vram_total * f) / MiB),
  };
}

/** Whether settings have ever been saved on this machine. Automatic model
 *  selection on first start happens only when they have **not** -- a model the
 *  user picked deliberately must never be overwritten by what we think fits
 *  better. */
export function settingsWerePersisted(): boolean {
  try { return localStorage.getItem(KEY) !== null; } catch { return false; }
}
