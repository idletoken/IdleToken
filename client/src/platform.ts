// Platform (marketplace) client — integration plan 3.1/3.2. Two halves:
//
//  1. Minimal account API: the in-memory balance pill, provider identity needed
//     to start the local agent, and the agent's overflow credential.
//  2. Agent control: local RPC to the Rust supervisor (engine.rs) that runs
//     `idletoken-platform-agent --relay ...` as a sidecar. Relay mode dials OUT
//     to the platform, so the home side opens no inbound port; the agent
//     self-registers the provider and waits for the local coordinator.
//
// In a plain browser (no Tauri) the console API still works — it is ordinary
// fetch — but agent control is unavailable and the UI must say so honestly
// (same philosophy as the DEV FIXTURE badge).
import { loadSettings } from "./settings";
import { getAuthProvider, type Session } from "./auth";
import { platformRequest, replyJson } from "./platformHttp";
import { AGENT_TOKEN_KEY, getSecret, setSecret } from "./secrets";

const FETCH_TIMEOUT_MS = 10_000;

// ---- response shapes (mirror platform/packages/gateway, do not invent) -----
export interface PlatformMe {
  id: string;
  email: string;
  role: string;
  balanceMilli: number;
  balanceCredits: number;
  // Public identity layer (the portal's profile page writes these). The gateway
  // has always returned them from GET /me; they were simply not declared here,
  // so the desktop client could not show the user as the account they set up.
  // avatarHue is never null on the wire — the gateway derives one from the id
  // when the user has not chosen a colour.
  username?: string | null;
  displayName?: string | null;
  avatarHue?: number;
  emailVerified?: boolean;
}

export interface ProviderInfo {
  id: string;
  name: string;
  endpoint: string;
  status: string; // ONLINE | OFFLINE | SUSPENDED
  listed: boolean; // whether it is listed on the marketplace (off by default; only then can others call it and earn credits)
  lastBeat: string | null;
  createdAt: string;
}

export interface CreatedApiKey {
  id: string;
  apiKey: string; // plaintext — shown exactly once
  prefix: string;
}

// ---- connection gate -------------------------------------------------------
// The panel needs both: a platform URL in settings AND a session issued by the
// platform (provider === "cloud" carries the gateway JWT). A local session's
// token is a random string the platform would 401.
export type PlatformGate =
  | { ok: true; url: string; session: Session }
  | { ok: false; reason: "no-url" | "no-session" | "local-session" };

export function platformGate(): PlatformGate {
  const url = loadSettings().platformUrl.trim();
  if (!url) return { ok: false, reason: "no-url" };
  const session = getAuthProvider().currentSession();
  if (!session) return { ok: false, reason: "no-session" };
  if (session.provider !== "cloud") return { ok: false, reason: "local-session" };
  return { ok: true, url: url.replace(/\/+$/, ""), session };
}

// ---- console API -----------------------------------------------------------
async function req<T>(path: string, init?: { method?: string; body?: string }): Promise<T> {
  const gate = platformGate();
  if (!gate.ok) throw new Error(`platform not connected (${gate.reason})`);
  let res;
  try {
    // Native transport in the desktop app — see platformHttp.ts. Every call in
    // this file was CORS-blocked in the webview for the same reason sign-in
    // was, so the whole console reported "can't reach the platform server"
    // against a healthy gateway.
    res = await platformRequest(gate.url + path, {
      method: init?.method ?? "GET",
      body: init?.body,
      bearer: gate.session.token,
      timeoutMs: FETCH_TIMEOUT_MS,
    });
  } catch (e) {
    // The cause is worth carrying: "dns error" and "connection refused" send
    // the reader to different places, and the old message named neither.
    throw new Error(`network: can't reach the platform server (${e instanceof Error ? e.message : e})`);
  }
  if (res.status === 401) {
    // The gateway rejected the session token (expired or revoked). Keeping the
    // dead session makes every panel show `HTTP 401: invalid token` forever
    // with no way back (field report 2026-08-22) — drop it and tell the shell,
    // which reverts to the signed-out state and opens the login screen.
    getAuthProvider().signOut();
    window.dispatchEvent(new CustomEvent("idletoken:session-expired"));
    throw new Error("session expired — sign in again");
  }
  if (!res.ok) {
    // Surface the server's own message when it sends one (Nest error bodies).
    const body = replyJson<{ message?: string | string[] }>(res);
    const m = body?.message;
    const detail = Array.isArray(m) ? m.join("; ") : m || "";
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  const parsed = replyJson<T>(res);
  if (parsed === null) throw new Error(`the platform sent a non-JSON reply to ${path}`);
  return parsed;
}

export function getMe(): Promise<PlatformMe> {
  return req<PlatformMe>("/me");
}

export function getProviders(): Promise<ProviderInfo[]> {
  return req<ProviderInfo[]>("/providers");
}

export function createApiKey(opts?: { label?: string; dailyCapMilli?: number | null }): Promise<CreatedApiKey> {
  // The plaintext key comes back exactly once, so whoever calls this owns
  // keeping it. `label` matters more than it looks: without one, a revoke page
  // shows a list of prefixes and no way to tell which is which.
  const body = opts ? JSON.stringify(opts) : undefined;
  return req<CreatedApiKey>("/me/api-keys", { method: "POST", ...(body ? { body } : {}) });
}

// ---- model usage leaderboard (public, no session) --------------------------
// The platform's scoreboard of how many tokens each curated model actually
// served, over a rolling window. The picker uses it to ORDER the list; it is
// never a gate on what you can run.
export interface LeaderboardRow {
  model: string;
  requests: number;
  inTokens: number;
  outTokens: number;
  cachedTokens: number;
  totalTokens: number;
}

export interface LeaderboardPayload {
  window: "day" | "week" | "month" | "all";
  /** `seeded` = the platform has too little traffic to rank; the order is editorial. */
  source: "measured" | "seeded";
  generatedAt: string;
  rows: LeaderboardRow[];
}

/** How long the picker waits before deciding the platform is not answering. */
const LEADERBOARD_TIMEOUT_MS = 5_000;

/**
 * Public, unauthenticated: it needs a platform URL and nothing else — no
 * session, no API key. Deliberately NOT routed through `req()`, whose gate
 * demands a cloud session: a signed-out user opening the picker should still
 * see the ordering, and a failure here must stay a non-event.
 *
 * Throws on any failure (unreachable, timeout, non-200). Callers fall back to
 * the static manifest order — see ModelPicker.
 */
export async function fetchLeaderboard(
  window: "day" | "week" | "month" | "all" = "week",
): Promise<LeaderboardPayload> {
  const base = loadSettings().platformUrl.trim().replace(/\/+$/, "");
  if (!base) throw new Error("platform not configured");
  const res = await platformRequest(`${base}/leaderboard/models?window=${encodeURIComponent(window)}`, {
    timeoutMs: LEADERBOARD_TIMEOUT_MS,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const parsed = replyJson<LeaderboardPayload>(res);
  if (parsed === null) throw new Error("the platform sent a non-JSON leaderboard");
  return parsed;
}

// ---- agent control (Tauri only) --------------------------------------------
// Same wire shape as the engine supervisor's EngineStatus, but for the
// platform-agent slot only (events on "platform-agent:status").
export interface AgentStatus {
  state: "stopped" | "starting" | "running" | "restarting" | "crashed";
  role: string | null;
  pid: number | null;
  startedAt: number | null;
  restarts: number;
  lastExitCode: number | null;
}

export interface AgentLogLine {
  ts: number;
  stream: "stdout" | "stderr";
  line: string;
}

export function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

// The supervisor's shared ring buffer tags every line with its role.
const AGENT_LOG_TAG = "[platform-agent]";

/**
 * The credential the platform agent actually runs with (2026-08-30, threat
 * register HOST-03/HOST-04/CHAIN-01).
 *
 * The agent used to be handed the full console JWT. That token sits in the
 * process arguments and memory of a machine that is powered on unattended for
 * weeks, and it can transfer this account's credits, mint API keys, and issue a
 * cluster-join credential — so stealing it is a complete account takeover. The
 * agent itself only ever calls `/providers*` (register, heartbeat, relay poll
 * and result, cache-state), so it gets a `scope=agent` token whose server-side
 * route allowlist refuses everything else.
 *
 * Cached in the 0600 credential store and reused until it is close to expiry:
 * minting one per launch would fill the account's session list with a row per
 * restart, and the list is the thing the user is supposed to be able to read.
 *
 * A gateway too old to know the endpoint answers 404. That case falls back to
 * the console JWT with a loud warning rather than refusing to share — a client
 * that silently stops earning against an older self-hosted gateway is a worse
 * failure, and the fallback is exactly the status quo it replaces.
 */
const AGENT_TOKEN_RENEW_BEFORE_MS = 24 * 3600 * 1000;

function storedAgentToken(): string | null {
  const raw = getSecret(AGENT_TOKEN_KEY);
  if (!raw) return null;
  try {
    const payload = JSON.parse(atob(raw.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    if (typeof payload.exp !== "number") return null;
    // Renew a day early: a token that expires while the agent is mid-shift takes
    // the machine off the market until someone notices.
    if (payload.exp * 1000 - Date.now() < AGENT_TOKEN_RENEW_BEFORE_MS) return null;
    return raw;
  } catch {
    return null;
  }
}

export async function agentCredential(fallbackJwt: string): Promise<string> {
  const cached = storedAgentToken();
  if (cached) return cached;
  try {
    const issued = await req<{ token?: string; scope?: string }>("/auth/agent-token", {
      method: "POST",
      body: JSON.stringify({ label: "sharing agent" }),
    });
    if (issued?.token && issued.scope === "agent") {
      setSecret(AGENT_TOKEN_KEY, issued.token);
      return issued.token;
    }
    console.warn("platform: /auth/agent-token returned an unexpected body; falling back to the console session token");
  } catch (e) {
    console.warn(
      `platform: could not obtain a scope-restricted agent token (${e instanceof Error ? e.message : e}). ` +
        "Falling back to the console session token — the agent will run with more authority than it needs. " +
        "Upgrade the platform gateway to close this."
    );
  }
  return fallbackJwt;
}

export async function agentStart(opts: {
  platformUrl: string;
  /** Console session token. Used only to MINT the agent's scoped token, and as
   *  the fallback when the gateway is too old to issue one. */
  jwt: string;
  name: string;
  coordApiPort: number;
  /** settings.apiToken — the coordinator 401s dispatched jobs without it. */
  coordToken: string;
  /** settings.modelId — what this machine serves. Without it the agent
   *  registers under its own default ("dsv4-flash", a DSv4 relic) and the
   *  marketplace routes this machine's real model around it. */
  modelId: string;
  /** settings.quant — precision-aware routing (may be ""). */
  quant: string;
}): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  // Swap the console session for the scope-restricted agent token HERE rather
  // than at each call site: both callers (the sharing toggle and the launch-time
  // resume) pass `gate.session.token`, and a rule that has to be remembered
  // twice is a rule that will be followed once.
  const jwt = await agentCredential(opts.jwt);
  await invoke("platform_agent_start", {
    platformUrl: opts.platformUrl,
    jwt,
    name: opts.name,
    coordApiPort: opts.coordApiPort,
    coordToken: opts.coordToken,
    modelId: opts.modelId,
    quant: opts.quant,
  });
}

export async function agentStop(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("platform_agent_stop");
}

/** Resume lending on launch. The provider switch is a STANDING choice: the
 *  user made it once, and every later launch on this machine keeps sharing
 *  without being asked again — until 0.1.10 a restarted client showed the
 *  switch on while no agent ran, so the machine silently stopped earning.
 *  Called from App's mount effect (Tauri only).
 *
 *  Deliberately quiet about the cases it cannot act on: signed out or no
 *  platform URL means the agent CANNOT register (it needs the JWT), and the
 *  sharing panel already explains both states to the user. An agent that is
 *  already running or restarting is left alone — this resumes, it never
 *  restarts. */
export async function resumeSharingAgent(): Promise<string> {
  if (!inTauri()) return "skipped: not the desktop app";
  const s = loadSettings();
  if (!s.providerEnabled) return "skipped: sharing is off";
  const gate = platformGate();
  if (!gate.ok) return `skipped: ${gate.reason}`;
  const st = await agentStatus();
  if (st.state === "running" || st.state === "starting" || st.state === "restarting")
    return "skipped: agent already up";
  // The EXACT name this machine registered under (see ShareToggleButton):
  // the gateway dedupes by (account, name), so resuming under a different
  // spelling would register a second provider row for the same machine. No
  // name yet = sharing was never toggled on this identity scheme; the next
  // manual toggle mints one, and resuming under a guess would be worse.
  if (!s.providerName) return "skipped: no provider identity yet";
  await agentStart({
    platformUrl: gate.url,
    jwt: gate.session.token,
    name: s.providerName,
    coordApiPort: s.apiPort || 8000,
    coordToken: s.apiToken,
    modelId: s.modelId,
    quant: s.quant,
  });
  return "started";
}

export async function agentStatus(): Promise<AgentStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return await invoke<AgentStatus>("platform_agent_status");
}

export function onAgentStatus(cb: (s: AgentStatus) => void): () => void {
  let disposed = false;
  let unlisten: (() => void) | null = null;
  import("@tauri-apps/api/event").then(({ listen }) =>
    listen<AgentStatus>("platform-agent:status", (e) => cb(e.payload)).then((un) => {
      if (disposed) un();
      else unlisten = un;
    })
  );
  return () => {
    disposed = true;
    unlisten?.();
  };
}

/** Tail of the agent's own log lines (filtered out of the shared ring). */
export async function agentLogs(maxLines: number): Promise<AgentLogLine[]> {
  const { invoke } = await import("@tauri-apps/api/core");
  const all = await invoke<AgentLogLine[]>("engine_logs", { maxLines: 500 });
  return all.filter((l) => l.line.startsWith(AGENT_LOG_TAG)).slice(-maxLines);
}

export function onAgentLog(cb: (l: AgentLogLine) => void): () => void {
  let disposed = false;
  let unlisten: (() => void) | null = null;
  import("@tauri-apps/api/event").then(({ listen }) =>
    listen<AgentLogLine>("engine:log", (e) => {
      if (e.payload.line.startsWith(AGENT_LOG_TAG)) cb(e.payload);
    }).then((un) => {
      if (disposed) un();
      else unlisten = un;
    })
  );
  return () => {
    disposed = true;
    unlisten?.();
  };
}
