// Platform (marketplace) client — integration plan 3.1/3.2. Two halves:
//
//  1. Console API: real HTTP calls to the platform gateway with the cloud
//     session's JWT (balance /me, ledger /me/ledger, my providers /providers,
//     API keys /me/api-keys). No mock data: every function either returns the
//     server's answer or throws with the real failure (philosophy 15).
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

const FETCH_TIMEOUT_MS = 10_000;

// ---- response shapes (mirror platform/packages/gateway, do not invent) -----
export interface PlatformMe {
  id: string;
  email: string;
  role: string;
  balanceMilli: number;
  balanceCredits: number;
}

export interface LedgerEntry {
  id: string;
  type: string; // GRANT | SPEND | EARN | ...
  deltaMilli: number;
  reason: string | null;
  ref: string | null;
  createdAt: string;
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

export interface ApiKeyInfo {
  id: string;
  prefix: string;
  revoked: boolean;
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
async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const gate = platformGate();
  if (!gate.ok) throw new Error(`platform not connected (${gate.reason})`);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(gate.url + path, {
      ...init,
      headers: {
        authorization: `Bearer ${gate.session.token}`,
        "content-type": "application/json",
        ...(init?.headers ?? {}),
      },
      signal: ctrl.signal,
    });
  } catch {
    throw new Error("network: can't reach the platform server");
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    // Surface the server's own message when it sends one (Nest error bodies).
    let detail = "";
    try {
      const body = (await res.json()) as { message?: string | string[] };
      const m = body?.message;
      detail = Array.isArray(m) ? m.join("; ") : m || "";
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  return (await res.json()) as T;
}

export function getMe(): Promise<PlatformMe> {
  return req<PlatformMe>("/me");
}

export function getLedger(): Promise<LedgerEntry[]> {
  return req<LedgerEntry[]>("/me/ledger");
}

export function getProviders(): Promise<ProviderInfo[]> {
  return req<ProviderInfo[]>("/providers");
}

/** List or unlist on the marketplace (the sharing switch). Only with listed=true can others call it and earn credits. */
export function setProviderListing(id: string, listed: boolean): Promise<{ id: string; listed: boolean }> {
  return req<{ id: string; listed: boolean }>(`/providers/${encodeURIComponent(id)}/listing`, {
    method: "POST",
    body: JSON.stringify({ listed }),
  });
}

export function listApiKeys(): Promise<ApiKeyInfo[]> {
  return req<ApiKeyInfo[]>("/me/api-keys");
}

export function createApiKey(): Promise<CreatedApiKey> {
  return req<CreatedApiKey>("/me/api-keys", { method: "POST" });
}

export function revokeApiKey(id: string): Promise<{ ok: boolean }> {
  return req<{ ok: boolean }>(`/me/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/** The rendezvous pairing token (for the engine's account mode `--account-token`; see gateway 4.7). */
export interface RendezvousToken {
  token: string;
  expiresInSec: number;
}

/** Fetch a 30-day scope=rendezvous token: the engine speaks over a plaintext link and should hold nothing but this restricted token. */
export function createRendezvousToken(): Promise<RendezvousToken> {
  return req<RendezvousToken>("/auth/rendezvous-token", { method: "POST" });
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

export async function agentStart(opts: {
  platformUrl: string;
  jwt: string;
  name: string;
  coordApiPort: number;
}): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("platform_agent_start", {
    platformUrl: opts.platformUrl,
    jwt: opts.jwt,
    name: opts.name,
    coordApiPort: opts.coordApiPort,
  });
}

export async function agentStop(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("platform_agent_stop");
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
