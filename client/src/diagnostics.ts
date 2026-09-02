// How the diagnostics bundle is assembled. **There is exactly one copy**: the
// retained acceptance channel's report-diagnostics hook goes through this
// function. The former user-facing diagnostics export was deliberately removed;
// any future replacement must call this same final boundary rather than grow a
// second implementation. This is where it is decided whether an access token
// goes out, the last place that should have two implementations.
import type { AppSettings } from "./settings";
import { readProblems } from "./problems";

// The settings fields that go into the bundle (an **allowlist**). Anything that
// could carry a credential is excluded: apiToken, the platform session, any
// key or secret. One field missing costs one extra question while debugging;
// one field too many could hand a user's token to a stranger. A missed entry in
// a denylist is a leak, a missed entry in an allowlist is just a blank field --
// the same trade-off as the public mirror.
export const DIAG_SETTING_KEYS = [
  "modelId", "quant", "tier", "longContext", "kvCacheK", "kvCacheV",
  "apiHost", "apiPort", "interStagePort", "discoveryPort",
  "resourcePreset", "maxVramMb", "maxRamMb",
  "preferCoordinator", "sameSubnetOnly", "bindNic",
  "platformUrl",
] as const satisfies readonly (keyof AppSettings)[];

/**
 * The settings that may leave this machine in a file, and only those.
 *
 * Added 2026-08-20 (audit A-P2-1) because there were two exits and only one
 * had a filter: "Export settings" serialized the whole `AppSettings` object,
 * `apiToken` and `overflowKey` included, into a file whose entire purpose is
 * to be moved to another machine or handed to someone helping. The same
 * allowlist now governs both — a settings backup that omits the local API key
 * costs one re-copy from the API page; one that includes it hands over the
 * gate to this machine's inference API.
 *
 * A superset of the diagnostics keys: a backup is meant to be restorable, so it
 * carries the preferences the diagnostics bundle has no reason to know about.
 */
export const EXPORT_SETTING_KEYS = [
  ...DIAG_SETTING_KEYS,
  "quant", "maxTokens", "tier",
  "accent", "uiScale", "density", "reduceMotion",
  "trayIcon", "closeToTray", "startMinimized", "rememberWindow", "autostart",
  "autoUpdate", "updateChannel",
  "modelDir", "mdns", "manualPeers", "heartbeatSec",
  "schemaVersion",
] as const satisfies readonly (keyof AppSettings)[];

export function exportableSettings(settings: AppSettings): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  // Deduplicated: EXPORT_SETTING_KEYS spreads DIAG_SETTING_KEYS, which already
  // holds several of the names repeated below for readability.
  for (const k of new Set<keyof AppSettings>(EXPORT_SETTING_KEYS)) out[k] = settings[k];
  return out;
}

/** Hard ceilings for an artifact assembled from native and persisted input. */
export const DIAGNOSTIC_LIMITS = Object.freeze({
  maxDepth: 6,
  maxKeyLength: 64,
  maxStringLength: 1024,
  maxArrayItems: 64,
  maxObjectKeys: 64,
  maxTotalKeys: 512,
  maxTotalNodes: 2048,
  maxTotalStringChars: 64 * 1024,
});

type DiagnosticValue = null | boolean | number | string | DiagnosticValue[] | { [key: string]: DiagnosticValue };
type DiagnosticRule =
  | { readonly type: "scalar" }
  | { readonly type: "array"; readonly item: DiagnosticRule }
  | { readonly type: "object"; readonly fields: Readonly<Record<string, DiagnosticRule>> };

const SCALAR: DiagnosticRule = Object.freeze({ type: "scalar" });
const objectRule = (fields: Readonly<Record<string, DiagnosticRule>>): DiagnosticRule => ({ type: "object", fields });
const arrayRule = (item: DiagnosticRule): DiagnosticRule => ({ type: "array", item });

const APP_RULE = objectRule({ version: SCALAR, os: SCALAR, arch: SCALAR });
const PROBE_RULE = objectRule({
  // hostname is deliberately absent: it is identity, not a hardware fact.
  os: SCALAR,
  cpu_count: SCALAR,
  gpu_name: SCALAR,
  cc_major: SCALAR,
  cc_minor: SCALAR,
  unified_memory: SCALAR,
  vram_total: SCALAR,
  vram_used_other: SCALAR,
  vram_usable: SCALAR,
  ram_total: SCALAR,
  ram_used_other: SCALAR,
  ram_usable: SCALAR,
  disk_avail: SCALAR,
  driver_version: SCALAR,
  hw_status: SCALAR,
  hw_reason: SCALAR,
  error: SCALAR,
});
const ADVISE_MODEL_RULE = objectRule({
  id: SCALAR,
  label: SCALAR,
  quant: SCALAR,
  mode: SCALAR,
  max_ctx: SCALAR,
  weight_bytes: SCALAR,
  need_bytes: SCALAR,
  shortfall_bytes: SCALAR,
  available: SCALAR,
  single_node: SCALAR,
});
const ADVISE_RULE = objectRule({
  nodes: SCALAR,
  models: arrayRule(ADVISE_MODEL_RULE),
  error: SCALAR,
});
const CLUSTER_MEMBER_RULE = objectRule({
  // hostname and rpc_endpoint are deliberately absent: neither is needed to
  // diagnose placement, and both identify a home network.
  role: SCALAR,
  state: SCALAR,
  gpu: SCALAR,
  stage: SCALAR,
  layer_lo: SCALAR,
  layer_hi: SCALAR,
});
const CLUSTER_RULE = objectRule({
  phase: SCALAR,
  engine: SCALAR,
  engine_state: SCALAR,
  cluster_size: SCALAR,
  members: arrayRule(CLUSTER_MEMBER_RULE),
  error: SCALAR,
});
const PROBLEM_DETAIL_RULE = objectRule({
  model: SCALAR,
  quant: SCALAR,
  file: SCALAR,
  turn: SCALAR,
  partialReply: SCALAR,
});
const PROBLEM_RULE = objectRule({
  at: SCALAR,
  kind: SCALAR,
  message: SCALAR,
  detail: PROBLEM_DETAIL_RULE,
});
const SETTINGS_RULE = objectRule(Object.fromEntries(DIAG_SETTING_KEYS.map((key) => [key, SCALAR])));
const BUNDLE_RULE = objectRule({
  schema: SCALAR,
  generatedAt: SCALAR,
  app: APP_RULE,
  probe: PROBE_RULE,
  advise: ADVISE_RULE,
  cluster: CLUSTER_RULE,
  settings: SETTINGS_RULE,
  recent_problems: arrayRule(PROBLEM_RULE),
});

const OMIT = Object.freeze({
  depth: "[omitted: depth limit]",
  keys: "[omitted: key limit]",
  nodes: "[omitted: node limit]",
  string: "[omitted: string limit]",
  strings: "[omitted: total string limit]",
  array: "[omitted: array limit]",
  object: "[omitted: object limit]",
  cycle: "[omitted: cyclic value]",
  unknown: "[omitted: unsupported value]",
  invalid: "[omitted: invalid value]",
});

const CONTROL_CHARS = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]/g;
const INVISIBLE_DIRECTIONAL_CHARS = /[\u00ad\u034f\u061c\u115f-\u1160\u17b4-\u17b5\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff9-\ufffb]/g;
const INVISIBLE_SUPPLEMENTARY_CHARS = /[\u{e0000}-\u{e007f}\u{e0100}-\u{e01ef}]/gu;
const URL_TEXT = /\b(?:https?|wss?):\/\/[^\s<>"'`]+/gi;
const EMAIL = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
const JWT = /\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b/g;
const BEARER = /\bbearer\s+[A-Za-z0-9._~+\/-]{8,}/gi;
const SECRET_ASSIGNMENT = /\b(api[\s_-]*(?:key|token)|access[\s_-]*token|refresh[\s_-]*token|session[\s_-]*token|overflow[\s_-]*key|provider[\s_-]*key|rendezvous[\s_-]*key|private[\s_-]*key|authorization|cookie|jwt|token|secret|password|passwd|psk)\b\s*[:=]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;]+)/gi;
const PROMPT_ASSIGNMENT = /(["']?(?:prompt|messages?|input|output|reply|completion|request[\s_-]*body|response[\s_-]*body|payload|body)["']?\s*[:=]\s*).*$/gi;
const KEY_SHAPED_SECRET = /\b(?:sk|gh[pousr])[-_][A-Za-z0-9_-]{12,}\b/g;
const WINDOWS_PATH = /(?:\b[A-Za-z]:\\|\\\\)[^\s<>"|?*]+/g;
const POSIX_PATH = /(^|[\s("'=])((?:~\/|\/)[A-Za-z0-9._@+-]+(?:\/[A-Za-z0-9._@+ -]+)+)/g;
const IPV4 = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
const IPV6_FULL = /\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b/g;
const IPV6_COMPRESSED = /(?<![A-Za-z0-9:])(?:[A-Fa-f0-9]{0,4}:){1,7}:[A-Fa-f0-9]{0,4}(?:%[A-Za-z0-9._-]+)?(?![A-Za-z0-9:])/g;

interface SanitizeContext {
  keys: number;
  nodes: number;
  stringChars: number;
  readonly seen: WeakSet<object>;
}

function isIpHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, "");
  if (/^127(?:\.\d{1,3}){3}$/.test(host) || host === "::1") return false;
  return /^(?:\d{1,3}\.){3}\d{1,3}$/.test(host) || host.includes(":");
}

function redactIpv4(address: string): string {
  return /^127(?:\.\d{1,3}){3}$/.test(address) ? address : "[redacted-ip]";
}

function redactIpv6(address: string): string {
  return address.replace(/^\[|\]$/g, "") === "::1" ? address : "[redacted-ip]";
}

/** Remove secrets and identity-bearing text while retaining short, honest error context. */
function sanitizeText(raw: string, context: SanitizeContext): string {
  if (raw.length > DIAGNOSTIC_LIMITS.maxStringLength) return OMIT.string;
  let value = raw
    .normalize("NFKC")
    .replace(INVISIBLE_DIRECTIONAL_CHARS, "")
    .replace(INVISIBLE_SUPPLEMENTARY_CHARS, "")
    .replace(CONTROL_CHARS, " ");
  if (value.length > DIAGNOSTIC_LIMITS.maxStringLength) return OMIT.string;

  value = value.replace(URL_TEXT, (candidate) => {
    try {
      const url = new URL(candidate);
      return url.username || url.password || url.search || url.hash || isIpHost(url.hostname)
        ? "[redacted-url]"
        : candidate;
    } catch {
      return "[redacted-url]";
    }
  });
  value = value
    .replace(PROMPT_ASSIGNMENT, "$1[redacted]")
    .replace(SECRET_ASSIGNMENT, (_match, label: string) => `${label}=[redacted]`)
    .replace(BEARER, "Bearer [redacted]")
    .replace(JWT, "[redacted-jwt]")
    .replace(KEY_SHAPED_SECRET, "[redacted-key]")
    .replace(EMAIL, "[redacted-email]")
    .replace(WINDOWS_PATH, "[redacted-path]")
    .replace(POSIX_PATH, (_match, prefix: string) => `${prefix}[redacted-path]`)
    .replace(IPV4, redactIpv4)
    .replace(IPV6_FULL, redactIpv6)
    .replace(IPV6_COMPRESSED, redactIpv6)
    .replace(/\s+/g, " ")
    .trim();

  if (value.length > DIAGNOSTIC_LIMITS.maxStringLength) return OMIT.string;
  if (context.stringChars + value.length > DIAGNOSTIC_LIMITS.maxTotalStringChars) return OMIT.strings;
  context.stringChars += value.length;
  return value;
}

function plainObject(value: object): boolean {
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function isDataDescriptor(descriptor: PropertyDescriptor): boolean {
  return Object.prototype.hasOwnProperty.call(descriptor, "value");
}

function sanitizeValue(
  value: unknown,
  rule: DiagnosticRule,
  context: SanitizeContext,
  depth: number,
): DiagnosticValue {
  if (depth > DIAGNOSTIC_LIMITS.maxDepth) return OMIT.depth;
  context.nodes += 1;
  if (context.nodes > DIAGNOSTIC_LIMITS.maxTotalNodes) return OMIT.nodes;

  if (value === null) return null;
  if (typeof value === "string") return sanitizeText(value, context);
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : OMIT.invalid;
  if (typeof value !== "object") return OMIT.unknown;
  if (rule.type === "scalar") return OMIT.unknown;

  if (context.seen.has(value)) return OMIT.cycle;
  context.seen.add(value);
  try {
    if (rule.type === "array") {
      if (!Array.isArray(value)) return OMIT.unknown;
      if (value.length > DIAGNOSTIC_LIMITS.maxArrayItems) return OMIT.array;
      return value.map((item) => sanitizeValue(item, rule.item, context, depth + 1));
    }

    if (Array.isArray(value) || !plainObject(value)) return OMIT.unknown;
    const descriptors = Object.getOwnPropertyDescriptors(value);
    const ownKeys = Reflect.ownKeys(descriptors);
    if (ownKeys.length > DIAGNOSTIC_LIMITS.maxObjectKeys) return OMIT.object;
    if (ownKeys.some((key) => typeof key !== "string" || key.length > DIAGNOSTIC_LIMITS.maxKeyLength)) {
      return OMIT.keys;
    }
    context.keys += ownKeys.length;
    if (context.keys > DIAGNOSTIC_LIMITS.maxTotalKeys) return OMIT.keys;
    if (Object.values(descriptors).some((descriptor) => !isDataDescriptor(descriptor))) {
      return OMIT.unknown;
    }

    const output: { [key: string]: DiagnosticValue } = {};
    for (const [key, childRule] of Object.entries(rule.fields)) {
      const descriptor = descriptors[key];
      if (!descriptor?.enumerable || !isDataDescriptor(descriptor)) continue;
      output[key] = sanitizeValue(descriptor.value, childRule, context, depth + 1);
    }
    return output;
  } catch {
    return OMIT.unknown;
  } finally {
    context.seen.delete(value);
  }
}

function ownDataValue(source: Record<string, unknown>, key: string): unknown {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(source, key);
    return descriptor?.enumerable && isDataDescriptor(descriptor) ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

/** The engine-side report plus the allowlisted settings = the final artifact boundary. */
export function buildDiagnosticsBundle(
  report: Record<string, unknown>,
  settings: AppSettings,
): Record<string, unknown> {
  const picked: Record<string, unknown> = {};
  for (const key of DIAG_SETTING_KEYS) {
    const value = ownDataValue(settings as unknown as Record<string, unknown>, key);
    if (value !== undefined) picked[key] = value;
  }
  // Recent problems (chat, downloads, cluster). The bundle exists so a user does
  // not have to open a console, and it used to carry hardware and settings but
  // not one word about what actually went wrong — the error lived in a banner
  // the next send wiped, so "what did it say?" had no answer. No prompts or
  // replies go in: see problems.ts for what is deliberately excluded.
  //
  // Always included (2026-08-15, the opt-out switch is gone): exporting the
  // bundle is itself the explicit act of sharing — it downloads a file the
  // user then sends by hand — and a diagnostics bundle without the failures
  // is the one kind that cannot diagnose anything.
  const untrusted: Record<string, unknown> = {
    settings: picked,
    recent_problems: readProblems(),
  };
  // Native diagnostics are JSON today, but this boundary does not trust that
  // implementation detail. Copy only known top-level facts without invoking
  // getters; the recursive rule then constrains every nested shape and value.
  for (const key of ["schema", "generatedAt", "app", "probe", "advise", "cluster"] as const) {
    const value = ownDataValue(report, key);
    if (value !== undefined) untrusted[key] = value;
  }
  const sanitized = sanitizeValue(untrusted, BUNDLE_RULE, {
    keys: 0,
    nodes: 0,
    stringChars: 0,
    seen: new WeakSet<object>(),
  }, 0);
  // BUNDLE_RULE is an object rule and `untrusted` is a fresh plain object, so
  // only a programming error could make this non-object. Fail closed anyway.
  return sanitized !== null && typeof sanitized === "object" && !Array.isArray(sanitized)
    ? sanitized
    : {};
}

/** The filename carries a UTC timestamp: users often send several in a row, and identically named files have overwritten each other. */
export function diagnosticsFileName(now: Date = new Date()): string {
  return `idletoken-diagnostics-${now.toISOString().slice(0, 19).replace(/[:T]/g, "")}.json`;
}
