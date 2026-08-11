// A short, persisted log of chat failures.
//
// Why this exists: a failed generation used to appear only as a red banner
// under the thread, which the next send cleared. It was never stored and never
// reached the diagnostics bundle, so by the time anyone asked "what did it
// say?", the one string that identified the failure was gone — the user had
// nothing to report but the colour.
//
// Deliberately NOT the message text, the prompt, or the reply: those are the
// user's conversation and this file is designed to be handed to someone else.
// Only the failure itself, plus what was being asked of the cluster at the time.

const KEY = "idletoken.chat.errors.v1";
const MAX = 20;

export interface ChatFailure {
  /** ISO-8601 UTC, so timestamps from several machines can be lined up. */
  at: string;
  /** Verbatim message shown to the user — the thing that used to be lost. */
  message: string;
  /** What the client believed it was talking to when this happened. */
  modelId: string;
  quant: string;
  /** Turn number within the conversation (1 = first). A failure that only ever
   *  happens from turn 2 onward points somewhere very different from one that
   *  happens on turn 1. */
  turn: number;
  /** Whether any text had streamed in before it failed. */
  hadPartialReply: boolean;
}

export function recordChatFailure(f: ChatFailure): void {
  try {
    const all = readChatFailures();
    all.unshift(f);
    localStorage.setItem(KEY, JSON.stringify(all.slice(0, MAX)));
  } catch {
    /* a full or blocked localStorage must never break chatting */
  }
}

export function readChatFailures(): ChatFailure[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatFailure[];
    return Array.isArray(parsed) ? parsed.slice(0, MAX) : [];
  } catch {
    return [];
  }
}

export function clearChatFailures(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* nothing to do */
  }
}
