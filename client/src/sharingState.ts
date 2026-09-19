import type { AgentLogLine, AgentStatus, ProviderInfo } from "./platform";

export type SharingPhase = "off" | "opening" | "reconnecting" | "sharing";
export interface AgentRegistration { startedAt: number; providerId: string }
export interface SharingObservation {
  agent: AgentStatus;
  provider: ProviderInfo | null;
  registration: AgentRegistration | null;
  refusal: string | null;
  serviceReady: boolean;
  platformError?: unknown;
}

export function agentRefusal({ line }: AgentLogLine): string | null {
  const text = line.replace(/^\[platform-agent\]\s*/, "").trim();
  // The summary follows the useful HTTP error. Do not overwrite that cause.
  if (/registration failed; refusing to start/i.test(text)) return null;
  if (!/register:|cannot be listed|too old to be listed|refus|spawn failed:/i.test(text)) return null;
  const brace = text.indexOf("{");
  if (brace >= 0) {
    try {
      const body = JSON.parse(text.slice(brace));
      if (typeof body?.message === "string" && body.message) return body.message.slice(0, 400);
    } catch { /* A truncated reply still has a useful plain-text explanation. */ }
  }
  return text.replace(/^platform-agent:\s*/, "").slice(0, 400) || null;
}

export function agentProviderId(line: string): string | null {
  return /\bprovider id\s*:\s*([\w-]+)/.exec(line)?.[1]
    ?? /\bprovider id changed \S+ -> ([\w-]+)/.exec(line)?.[1]
    ?? /\bnow advertising .+ \(provider ([\w-]+)\)/.exec(line)?.[1]
    ?? null;
}

/** Names can be reassigned by the gateway; only this run's provider ID counts. */
export function sharingObservation(
  agent: AgentStatus, logs: AgentLogLine[], providers: ProviderInfo[],
  previous: AgentRegistration | null, serviceReady: boolean,
): SharingObservation {
  let registration = previous?.startedAt === agent.startedAt ? previous : null;
  let refusal = agent.refusedReason ?? null;
  for (const log of logs) {
    if (agent.startedAt == null || log.ts < agent.startedAt) continue;
    const providerId = agentProviderId(log.line);
    if (providerId) registration = { startedAt: agent.startedAt, providerId };
    refusal = agentRefusal(log) ?? refusal;
  }
  return {
    agent, registration, refusal, serviceReady,
    provider: providers.find((p) => p.id === registration?.providerId) ?? null,
  };
}

export function isSharing(s: SharingObservation): boolean {
  return !s.platformError && s.serviceReady && s.agent.state === "running"
    && s.provider?.listed === true && s.provider.online === true;
}

/** Only positive evidence can stop an enabled service; a failed read cannot. */
export class SharingStoppedError extends Error {}

export function sharingFailure(s: SharingObservation): SharingStoppedError | null {
  if (s.agent.state === "stopped" || s.agent.state === "crashed") {
    return new SharingStoppedError(s.refusal || "[SHARE_AGENT_STOPPED] The sharing agent stopped.");
  }
  if (!s.platformError && s.provider && !s.provider.listed) {
    return new SharingStoppedError(s.provider.unlistedReason || "[SHARE_NOT_LISTED] The platform did not list this service.");
  }
  if (!s.platformError && s.provider && ["SUSPENDED", "BLACKLISTED"].includes(s.provider.status)) {
    return new SharingStoppedError(s.provider.unlistedReason || "[SHARE_SUSPENDED] The platform suspended this service.");
  }
  return null;
}

/** Spawning a process starts an attempt; listing with a fresh heartbeat ends it. */
export async function waitForSharing(
  read: () => Promise<SharingObservation>,
  options: {
    signal: AbortSignal;
    pending?: () => void;
    timeoutMs?: number;
    now?: () => number;
    pause?: () => Promise<void>;
  },
): Promise<SharingObservation> {
  const now = options.now ?? Date.now;
  const deadline = now() + (options.timeoutMs ?? 60_000);
  const pause = options.pause ?? (() => new Promise<void>((resolve) => setTimeout(resolve, 1_000)));
  let failure: unknown;
  while (true) {
    options.signal.throwIfAborted();
    let s: SharingObservation | null = null;
    try { s = await read(); } catch (e) { failure = e; }
    options.signal.throwIfAborted();
    if (s) {
      if (isSharing(s)) return s;
      const terminal = sharingFailure(s);
      if (terminal) throw terminal;
      failure = s.platformError || (s.refusal ? new Error(s.refusal) : null);
    }
    options.pending?.();
    if (now() >= deadline) {
      if (failure) throw failure;
      throw new Error("[SHARE_START_TIMEOUT] The platform did not confirm an online sharing service within 60 seconds. Try again.");
    }
    await pause();
  }
}
