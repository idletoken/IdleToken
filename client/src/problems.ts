// A short, persisted log of things that went wrong on this machine.
//
// Why this exists: a failure used to live only in the UI element that showed it
// — a red banner the next send cleared, a download row the next model switch
// replaced. By the time anyone asked "what did it say?", the one string that
// identified the problem was gone and the user had nothing to report but the
// colour. This is the record that survives, and it is what the diagnostics
// bundle carries when someone asks for help or files an issue.
//
// It grew out of chatErrors.ts (chat-only, 2026-08-11) on 2026-08-13: the same
// question — "what has been failing on this machine?" — was being answered for
// one surface and not for the others, and two parallel logs would have been the
// usual two-copies mistake. Old chat-only entries are migrated in, so nobody
// loses the history they already had.
//
// **Nothing here is sent anywhere on its own.** It is a local record; the
// export is the only way it travels, and a switch in Settings controls whether
// it is included in that export.
//
// Deliberately NOT recorded: prompt text, replies, tokens, paths outside the
// model folder — anything that is the user's content or a credential. This file
// is designed to be handed to a stranger.

const KEY = "idletoken.problems.v1";
const LEGACY_CHAT_KEY = "idletoken.chat.errors.v1";
const MAX = 30;

/** Which part of the product failed. Kept coarse: the point is to spot "this
 *  keeps happening on downloads", not to build a taxonomy. */
export type ProblemKind = "chat" | "download" | "cluster";

export interface Problem {
  /** ISO-8601 UTC, so records from several machines can be lined up. */
  at: string;
  kind: ProblemKind;
  /** Verbatim message shown to the user — the thing that used to be lost. */
  message: string;
  /** Small structured facts about the moment: model, precision, turn number,
   *  whether anything had streamed in. Values only ever come from the client's
   *  own state, never from user text. */
  detail?: Record<string, string | number | boolean>;
}

/**
 * Consent to SHARE the log with us — not consent to keep it.
 *
 * The distinction matters, and the first version had it wrong: switching off
 * deleted everything, so someone who turned it back on later found their
 * history gone. Recording is local and unconditional. It is also the user's own
 * record of what keeps failing, and confiscating that is not what "do not send
 * this to the developers" asks for.
 *
 * What "off" governs today is the diagnostics bundle, which is the only way any
 * of this reaches us. When an upload path exists it will gate that too — same
 * switch, same meaning.
 *
 * Deleting is a separate, explicit action (Clear the log).
 */
const NO_SHARE_KEY = "idletoken.problems.noshare";

export function problemsShared(): boolean {
  try {
    return localStorage.getItem(NO_SHARE_KEY) !== "1";
  } catch {
    return true;
  }
}

export function setProblemsShared(on: boolean): void {
  try {
    if (on) localStorage.removeItem(NO_SHARE_KEY);
    else localStorage.setItem(NO_SHARE_KEY, "1");
  } catch {
    /* a blocked localStorage must not break the setting */
  }
}

/** Always records — see problemsShared() for what the switch actually does. */
export function recordProblem(p: Problem): void {
  try {
    const all = readProblems();
    all.unshift(p);
    localStorage.setItem(KEY, JSON.stringify(all.slice(0, MAX)));
  } catch {
    /* a full or blocked localStorage must never break what the user was doing */
  }
}

export function readProblems(): Problem[] {
  let out: Problem[] = [];
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as Problem[]) : [];
    if (Array.isArray(parsed)) out = parsed;
  } catch {
    /* corrupt storage: treat as empty rather than break the settings page */
  }
  // Chat failures recorded before this file existed. Read-only and merged, not
  // rewritten: if the user downgrades, their old client still finds them.
  try {
    const legacy = localStorage.getItem(LEGACY_CHAT_KEY);
    if (legacy) {
      const old = JSON.parse(legacy) as {
        at: string; message: string; modelId?: string; quant?: string; turn?: number; hadPartialReply?: boolean;
      }[];
      if (Array.isArray(old)) {
        out = out.concat(
          old.map((o) => ({
            at: o.at,
            kind: "chat" as const,
            message: o.message,
            detail: {
              model: o.modelId ?? "",
              quant: o.quant ?? "",
              turn: o.turn ?? 0,
              partialReply: !!o.hadPartialReply,
            },
          }))
        );
      }
    }
  } catch {
    /* ignore a corrupt legacy log */
  }
  // Newest first across both sources.
  out.sort((a, b) => (a.at < b.at ? 1 : a.at > b.at ? -1 : 0));
  return out.slice(0, MAX);
}

export function clearProblems(): void {
  try {
    localStorage.removeItem(KEY);
    localStorage.removeItem(LEGACY_CHAT_KEY);
  } catch {
    /* nothing to do */
  }
}
