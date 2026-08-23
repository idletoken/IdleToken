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
  // Empty = no auth, and that is the shipped default: the API serves loopback
  // only (2026-08-15) and browser-origin requests are refused by the
  // coordinator's Origin check, so a token guards nothing a local program
  // could not read from this same store. An operator who sets one still gets
  // it enforced.
  apiToken: string;
  // ---- sharing: lend when idle, borrow when busy (one switch) ----
  //
  // One setting, because it is one deal: this machine takes other people's work
  // when it is free and hands its own overflow to someone else when it is full.
  // Splitting it into two switches would offer "only earn" and "only spend" as
  // if they were separate products; they are the two halves of the same
  // exchange, and the pricing only works if most machines do both.
  sharingEnabled: boolean;
  /** What borrowing may cost in one UTC day, in milli-credits. NOT a user
   *  setting any more (owner's call, 2026-08-21): borrowing may spend the
   *  whole balance, and the balance itself is the ceiling — the platform
   *  refuses past zero. The field stays because the coordinator's flag needs
   *  a positive number (it reads 0 as "use my own default"); it is pinned to
   *  OVERFLOW_UNCAPPED_MILLI by the v6 migration and no UI edits it. */
  overflowDailyCapMilli: number;
  /** Only borrow once this machine's estimated wait reaches this many seconds.
   *  0 = borrow as soon as it is full. Advanced: the useful default is 0 and
   *  the setting only matters to someone who would rather queue than pay. */
  overflowWaitS: number;
  /** The provider name this machine registered on the platform, written by
   *  the sharing toggle: `<clusterName>-<N>`, N = the smallest free number
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

/** The daily borrow cap as shipped: effectively NONE (owner's call,
 * 2026-08-21) — borrowing may spend the whole account balance, and the real
 * ceiling is the balance itself (the platform refuses past zero). The value is
 * not 0 because the coordinator reads 0 as "use my own 50-credit default" and
 * has no spelling for "no ceiling" at all; and it stays under 2^31 because the
 * flag is parsed with atol(), which is 32-bit on Windows. 2e9 milli = 2M
 * credits, orders of magnitude past any real balance. */
export const OVERFLOW_UNCAPPED_MILLI = 2_000_000_000;

export const DEFAULT_SETTINGS: AppSettings = {
  modelId: DEFAULT_MODEL_ID,
  quant: defaultQuant(DEFAULT_MODEL_ID),
  tier: 2,
  // Full power by default (2026-08-15, was "balanced"): the product's whole
  // promise is using this machine's idle capacity, and a fresh install that
  // silently keeps 25% back both underuses the hardware and misreports what
  // the machine could serve. Whoever needs headroom turns it down knowingly.
  resourcePreset: "max",
  maxVramMb: 0,
  maxRamMb: 0,
  computeMode: "auto",
  apiHost: "127.0.0.1",
  apiPort: 8000,
  apiOpenAI: true,
  apiAnthropic: true,
  apiToken: "",
  // Off by default. Sharing sends this machine other people's prompts and
  // spends this account's credits; both are decisions to be made, not defaults
  // to be discovered.
  sharingEnabled: false,
  // 50 credits/day. Matches the coordinator's own default
  // (IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI) so the number shown here is the
  // number in force. Raised from 5 on 2026-08-19 against the measured rate card
  // (anchor-proposal-v1, D2): 5 credits bought one 27B-class conversation, so
  // the feature looked broken to anyone who switched it on. 50 is still a
  // guardrail — a default that surprises someone by refusing is recoverable in
  // a way that one which surprises them by spending is not.
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
  // context fills, which at ~13 tok/s and a 1M window is many hours of the
  // cluster. Anyone who wants a bound sets one here, or sends max_tokens.
  maxTokens: 0,
  // Literal, not SCHEMA_VERSION: that const is declared further down and this
  // object is built at module init. Keep the two in step by hand.
  schemaVersion: 7,
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
  clusterName: "IdleToken-Home",
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
export const APP_VERSION = "0.1.19";

const KEY = "idletoken.settings";

// Bump when a stored value must be discarded rather than merged. Absent in
// blobs written before versioning existed, which reads as 0.
const SCHEMA_VERSION = 7;

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
    // No migration for the cluster-name default rename ("home" →
    // "IdleToken-Home", 2026-08-15): a stored "home" MAY be a deliberate
    // choice, and rewriting a name the user could have typed is worse than
    // letting old and new defaults coexist. Machines that should pair must
    // simply agree on the name — which the field says out loud.
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
    // coordinator; what bounds a local program is the daily spend cap
    // (docs/api-surface.md §5.3).
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
  // ---- overflow (borrow when full) -> coord --overflow-* ------------------
  //
  // Launch parameters, not live controls: the coordinator reads them once, at
  // start. Turning sharing on therefore restarts the engine, which is the same
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
    ctxSize: tierCtx(s.tier),
    // The engine's per-request ceiling comes from the same setting the chat
    // sends, so the number the user typed is the number that governs — for
    // third-party API clients too, not just our own chat.
    maxDecode: s.maxTokens,
    // Overflow reaches the coordinator only when the sharing switch is on AND
    // a key has been minted for it. Anything less and the URL is left empty,
    // which is how the coordinator is told "do not enable this" -- there is no
    // separate off flag to fall out of step with the credentials.
    //
    // This stays the URL as the user configured it (usually https://). The
    // coordinator has no TLS client, so the spawn layer translates it to the
    // gateway's plaintext spelling (engine_platform_url in engine.rs) — the
    // translation lives in Rust because both launch paths (coord overflow and
    // the platform agent) pass through there, and a second copy here would be
    // the kind that drifts.
    overflowUrl: s.sharingEnabled && s.overflowKey ? s.platformUrl.trim().replace(/\/+$/, "") : "",
    overflowKey: s.sharingEnabled ? s.overflowKey : "",
    overflowWaitS: Math.max(0, s.overflowWaitS || 0),
    // Never 0: 0 means "the coordinator's own default", not "no ceiling", and
    // the coordinator has no way to express "no ceiling" at all.
    overflowDailyCapMilli: Math.max(1, s.overflowDailyCapMilli || OVERFLOW_UNCAPPED_MILLI),
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
