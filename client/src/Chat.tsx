// Chat — the real multi-turn chat surface (2026-07 IA pass). The engine is
// stateless per request, so every send resends that conversation's full message
// list (the Rust relay takes the messages array verbatim).
//
// History is device-local only (localStorage, capped) — nothing leaves the
// machine except the API call to your own cluster.
//
// 2026-08-11: conversations + a sidebar. One rolling transcript meant the only
// way to start a clean topic was "clear history", which threw the old one away.
// The dashboard's one-shot launcher box is gone too — chatting has one home.
import { useEffect, useMemo, useRef, useState } from "react";
import Markdown from "./Markdown";
import { UserAvatar, type UserIdentity } from "./Avatar";
import { useI18n } from "./i18n";
import { inTauri } from "./platform";
import type { ClusterApi } from "./pairing";
import { recordProblem } from "./problems";
import { useClusterStats, servedModelOf } from "./clusterStats";

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  sim?: boolean; // browser dev-sim reply (labeled, never mistaken for real)
  /** This turn failed. Kept IN the transcript rather than shown as a banner the
   *  next send wipes: the message is the only identification of the failure,
   *  and a user asked "what did it say?" could only answer "it was red". */
  error?: string;
  stats?: {
    ttftMs: number;
    totalMs: number;
    tokens: number;
    /** Prompt tokens served from the cluster's KV cache, and the prompt's total
     *  length. Recorded per reply so "is the cache working?" is answerable by
     *  looking at the transcript, not by catching a progress line as it goes by. */
    reused?: number;
    promptTokens?: number;
  };
}

interface Conversation {
  id: string;
  title: string;
  msgs: ChatMsg[];
  updatedAt: number;
}

/** A generation in flight, one per conversation.
 *
 *  `phase` is the honest answer to "what is happening right now":
 *    waiting   — the request is out and the cluster has not said a word. Most
 *                often it is queued behind another turn; nothing is being
 *                generated, so nothing may be animated as though it were.
 *    prefill   — the cluster is reading the prompt (ticks arrive, `prefill` is set)
 *    streaming — tokens are landing in the transcript
 */
interface LiveGen {
  /** Passed to `api_chat_cancel`, and what tags this stream's `api-chat` events. */
  reqId: string;
  phase: "waiting" | "prefill" | "streaming";
  prefill: { done: number; total: number; reused: number } | null;
  startedAt: number;
}

const STORE_KEY = "idletoken.chat.v2";
const LEGACY_KEY = "idletoken.chat.v1"; // single rolling transcript
const MAX_STORED = 50; // messages kept per conversation
const MAX_CONVOS = 40;
/** How many conversations may generate at once.
 *
 *  Not a setting (and not derived from the cluster): the coordinator serves
 *  requests serially in single-machine llama.cpp mode today, so past a small
 *  number the extra turns are pure queue — a ceiling that let a user open ten
 *  would be a ceiling that lies about what the machine can do. Three is enough
 *  to work in one thread while two others think, and small enough that the
 *  queue behind it stays comprehensible. Raise it when the engine grows real
 *  sequence slots (see the engine-multislot-cachehit plan), not before.
 */
const CLIENT_MAX_INFLIGHT = 3;
/** How long changes may sit unwritten before history hits localStorage. The
 *  whole table is serialized on every write, so with three streams appending
 *  deltas several times a second an unbatched write is a full re-serialize per
 *  token. Flushed on unmount and on `beforeunload`, so the debounce can only
 *  ever cost the last few hundred ms of an unclean kill. */
const STORE_DEBOUNCE_MS = 400;
/** …and the ceiling on that wait. A plain debounce is WRONG here: deltas land
 *  every few tens of milliseconds, so each one re-arms the timer and a reply
 *  that streams for two minutes writes nothing for two minutes — a crash
 *  mid-generation would then lose the whole conversation, which is worse than
 *  the write cost the debounce was meant to save. This makes it a throttle:
 *  under continuous change, history still reaches disk this often. */
const STORE_MAX_WAIT_MS = 2000;

const newId = () => `c-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/** Drop a trailing assistant turn that has neither text nor an error.
 *
 *  That message is the empty placeholder a generation writes before its first
 *  token; the run that was going to fill it did not survive the reload (or the
 *  process was killed). In-flight state is deliberately not persisted, so
 *  without this the transcript ends on a bubble that is permanently about to
 *  start speaking — the "half-generating" residue §5 of the plan forbids. */
function settleTail(msgs: ChatMsg[]): ChatMsg[] {
  const out = [...(msgs ?? [])];
  while (out.length) {
    const last = out[out.length - 1];
    if (last.role !== "assistant" || last.text || last.error) break;
    out.pop();
  }
  return out;
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Conversation[];
      // Re-derive the titles rather than trusting the stored ones: everything
      // saved before 2026-08-11 was cut at 34 raw characters, which for CJK is
      // twice the width the sidebar has. The title is derived data with no user
      // input path (there is no rename), so recomputing it can only ever be
      // more correct — and it means the oldest conversations, the ones most
      // likely to look wrong, get fixed instead of being grandfathered in.
      if (Array.isArray(parsed)) {
        return parsed.slice(0, MAX_CONVOS).map((c) => {
          const msgs = settleTail(c.msgs);
          return { ...c, msgs, title: msgs.length ? titleOf(msgs) : c.title };
        });
      }
    }
    // Migrate the old single transcript rather than dropping it on the floor.
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const msgs = settleTail(JSON.parse(legacy) as ChatMsg[]);
      if (Array.isArray(msgs) && msgs.length > 0) {
        return [{ id: newId(), title: titleOf(msgs), msgs: msgs.slice(-MAX_STORED), updatedAt: Date.now() }];
      }
    }
  } catch {
    /* corrupt storage must not block chatting */
  }
  return [];
}

// Roughly how wide a character is, in half-width units. CJK, kana and
// full-width punctuation take about twice the space of a Latin letter at the
// same font size, so counting raw characters gives two different results for
// the same visual length — 34 CJK characters is over 440px in a 232px column.
function displayWidth(s: string): number {
  let w = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    w += (c >= 0x1100 && (c <= 0x115f ||                       // Hangul jamo
          c === 0x2329 || c === 0x232a ||
          (c >= 0x2e80 && c <= 0xa4cf && c !== 0x303f) ||      // CJK radicals..Yi
          (c >= 0xac00 && c <= 0xd7a3) ||                      // Hangul syllables
          (c >= 0xf900 && c <= 0xfaff) ||                      // CJK compat ideographs
          (c >= 0xfe30 && c <= 0xfe6f) ||                      // CJK compat forms
          (c >= 0xff00 && c <= 0xff60) ||                      // full-width forms
          (c >= 0xffe0 && c <= 0xffe6) ||
          (c >= 0x20000 && c <= 0x3fffd)))                     // CJK ext B..
      ? 2 : 1;
  }
  return w;
}

/** Cut to `budget` half-width units, preferring a clause boundary in the last
 *  third so the title ends somewhere a human would stop rather than mid-word. */
function clip(text: string, budget: number): string {
  if (displayWidth(text) <= budget) return text;
  let out = "";
  let w = 0;
  for (const ch of text) {
    const cw = displayWidth(ch);
    if (w + cw > budget) break;
    out += ch;
    w += cw;
  }
  // A boundary at least 60% in. Higher and common phrasings miss it by a
  // character and get cut mid-word anyway ("How does pipeline parallelis…");
  // much lower and the title throws away half of what would have fit.
  const KEEP = budget * 0.6;
  const m = out.match(/^([\s\S]*[。！？；，、.!?;,:])[\s\S]*$/);
  if (m && displayWidth(m[1]) >= KEEP) return m[1].replace(/[，、,:;；]$/, "") + "…";
  const sp = out.lastIndexOf(" ");
  if (sp > 0 && displayWidth(out.slice(0, sp)) >= KEEP) return out.slice(0, sp) + "…";
  return out + "…";
}

/** A conversation is named after what the user opened it with.
 *
 *  Note on the alternative: ChatGPT asks a model for a real summary. We could
 *  too — the cluster is right there — but a title would then cost a whole
 *  generation, and on a home cluster running DSv4 at ~30 s/token that is
 *  minutes of the machine for a sidebar label. This stays deterministic, free
 *  and instant; the only thing it gives up is a genuinely abstractive title.
 *  Code fences and markdown noise are stripped so a pasted snippet does not
 *  become a title of backticks. */
export function titleOf(msgs: ChatMsg[]): string {
  const first = msgs.find((m) => m.role === "user")?.text ?? "";
  const cleaned = first
    .replace(/```[\s\S]*?```/g, " ")      // fenced code
    .replace(/`([^`]*)`/g, "$1")          // inline code, keep the text
    .replace(/^#{1,6}\s+/gm, "")          // heading markers
    .replace(/[*_>]/g, "")                // emphasis / quote markers
    .replace(/\s+/g, " ")
    .trim();
  // 28 half-widths ~ the 232px column at 13px: 28 Latin characters or 14 CJK,
  // which come out the same length on screen.
  return clip(cleaned, 28);
}

/**
 * Split a reply into reasoning and answer.
 *
 * Reasoning models (Qwen3.x here) wrap their scratchpad in <think>…</think> and
 * the engine passes it through verbatim — so the raw tags were landing in the
 * bubble, which is what made replies look mangled. Parsed on every render
 * because it must also work MID-STREAM: an opened-but-unclosed <think> means
 * everything after it is still reasoning.
 */
export function splitThink(text: string): { reasoning: string; answer: string; thinking: boolean } {
  const open = text.indexOf("<think>");
  if (open === -1) return { reasoning: "", answer: text, thinking: false };
  const close = text.indexOf("</think>", open);
  if (close === -1) {
    return { reasoning: text.slice(open + 7), answer: text.slice(0, open), thinking: true };
  }
  const reasoning = text.slice(open + 7, close);
  const answer = text.slice(0, open) + text.slice(close + 8);
  return { reasoning: reasoning.trim(), answer: answer.trim(), thinking: false };
}

/** Per-message actions under a finished turn: copy, and regenerate on the
 *  latest assistant reply. Copy takes the ANSWER only — the reasoning block is
 *  scratchpad, and pasting it along with the answer is never what "copy this
 *  reply" means. Regenerate doubles as retry: an errored turn renders the same
 *  button, and re-running it is exactly what retrying is. */
function MsgActions(props: { m: ChatMsg; canRegen: boolean; onRegen: () => void }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const text = props.m.role === "assistant" ? splitThink(props.m.text).answer || props.m.text : props.m.text;
  return (
    <div className="chat-msg__actions">
      {text ? (
        <button
          className="msgbtn"
          onClick={() => {
            void navigator.clipboard?.writeText(text).then(
              () => { setCopied(true); setTimeout(() => setCopied(false), 1400); },
              () => {},   // clipboard denied: stay silent rather than claim success
            );
          }}
        >
          <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
            {copied ? (
              <path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2.2"
                    strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <>
                <rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.8" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" fill="none"
                      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </>
            )}
          </svg>
          {copied ? t("chat.copied") : t("chat.copy")}
        </button>
      ) : null}
      {props.canRegen ? (
        <button className="msgbtn" onClick={props.onRegen}>
          <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
            <path d="M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6" fill="none" stroke="currentColor"
                  strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {t("chat.regen")}
        </button>
      ) : null}
    </div>
  );
}

function Bubble(props: {
  m: ChatMsg;
  /** A reply is being generated into THIS conversation. */
  live: boolean;
  last: boolean;
  /** How far that generation has got, or null when nothing is in flight here. */
  phase: LiveGen["phase"] | null;
  /** Another conversation's generation has already been picked up by the
   *  cluster, so this one is genuinely behind it. Kept separate from "we have
   *  simply not heard back yet": both look identical from here, and only one of
   *  them is evidence that something is ahead in the queue. */
  behind: boolean;
  /** Prefill progress, while the cluster is still reading the prompt. */
  prefill: { done: number; total: number; reused: number } | null;
}) {
  const { t } = useI18n();
  const [openReasoning, setOpenReasoning] = useState(false);
  const { reasoning, answer, thinking } = useMemo(() => splitThink(props.m.text), [props.m.text]);

  if (props.m.role === "user") {
    return <div className="chat-msg__bubble">{props.m.text}</div>;
  }
  const waiting = props.live && props.last && !props.m.text;
  // Nothing is being generated yet: the request is out and the cluster has not
  // said a word, which on a coordinator that serves one request at a time
  // usually means this turn is behind another one.
  const queued = waiting && props.phase === "waiting";
  return (
    <div className="chat-msg__bubble">
      {reasoning ? (
        <div className={`think${openReasoning ? " is-open" : ""}`}>
          <button className="think__toggle" onClick={() => setOpenReasoning(!openReasoning)}>
            {thinking ? t("chat.thinking") : t("chat.reasoning")}
            {/* SVG, not "▾": the display font has no triangle glyph, so the
                text version rendered as a bare dot in the shipped app. */}
            <svg className="think__chev" viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">
              <path d={openReasoning ? "M6 15l6-6 6 6" : "M6 9l6 6 6-6"} fill="none" stroke="currentColor"
                    strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          {openReasoning ? <div className="think__body">{reasoning}</div> : null}
        </div>
      ) : null}
      {answer ? <Markdown text={answer} /> : null}
      {/* On a LAN cluster prefill is minutes. Without this the bubble is blank
          the whole time and the app looks hung — which is how the read-timeout
          bug was experienced even before it errored out. */}
      {/* Deliberately a plain line and no animated ellipsis: nothing is being
          typed. Animating a queued turn would be the app claiming progress it
          has no evidence for — the same rule that keeps a seeded leaderboard
          from showing a confident 0. And it says WHICH kind of silence this is:
          "queued behind another reply" is a claim, and it is only made when
          another generation has actually been picked up. */}
      {queued ? (
        <span className="chat-msg__phase">{t(props.behind ? "chat.queuedBehind" : "chat.queued")}</span>
      ) : null}
      {waiting && props.prefill ? (
        <span className="chat-msg__phase">
          {/* Say which of the two it is. "120/135" and "0/135" scroll past
              looking the same, and the difference between them is the entire
              question of whether the KV cache is doing anything. */}
          {props.prefill.reused > 0
            ? t("chat.prefillCached", {
                reused: props.prefill.reused,
                fresh: Math.max(0, props.prefill.total - props.prefill.reused),
                done: props.prefill.done,
                total: props.prefill.total,
              })
            : t("chat.prefill", { done: props.prefill.done, total: props.prefill.total })}
        </span>
      ) : null}
      {(waiting && !queued) || (thinking && !answer) ? <span className="chat-msg__dots">…</span> : null}
      {props.m.error ? (
        <div className="chat-msg__err">
          <span className="chat-msg__err-text">{props.m.error}</span>
          <button
            className="linkbtn"
            onClick={() => void navigator.clipboard?.writeText(props.m.error!).catch(() => {})}
          >
            {t("chat.copyError")}
          </button>
        </div>
      ) : null}
      {props.m.sim && props.m.text ? <span className="tryit__sim"> {t("tryit.browserNote")}</span> : null}
    </div>
  );
}

export default function Chat(props: {
  api: ClusterApi | null;
  /** Where `api` comes from, so the browser dev-sim can answer "what is loaded"
   *  the way a coordinator would. Defaults to the real engine. */
  source?: "engine" | "dev-sim";
  apiToken: string;
  modelId: string;
  /** Selected precision — stamped on a recorded failure so a report says which
   *  configuration produced it, not just which model. */
  quant: string;
  /** 0 = no limit (generate until EOS or the context runs out). */
  maxTokens: number;
  /** Who the user is, for the avatar and the name over their turn. null when
   *  signed out or on a local-only identity — then a neutral glyph is used. */
  identity: UserIdentity | null;
  onGoCluster: () => void;
  /** Machines in the running cluster (0 when nothing is running). */
  machines?: number;
  /** Display label for modelId, supplied by the caller. */
  modelLabel: string;
}) {
  const { t, tErr } = useI18n();
  const [convos, setConvos] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(() => loadConversations()[0]?.id ?? null);
  const [input, setInput] = useState("");
  // Every conversation with a generation in flight, keyed by conversation id.
  // Until 2026-08-18 this was a single global `busy` flag: one reply at a time,
  // and on a LAN cluster a reply is minutes — so the sidebar full of
  // conversations could be read but never used. Conversations are now
  // independent, up to CLIENT_MAX_INFLIGHT of them; within one conversation a
  // second send is still refused (the two turns would race the same transcript).
  const [liveMap, setLiveMap] = useState<Map<string, LiveGen>>(() => new Map());
  // The AUTHORITATIVE record of what is running; `liveMap` above only drives
  // rendering. The guard in generate() used to read React state, i.e. whatever
  // the last COMMITTED render captured. Press regenerate several times faster
  // than React commits — easy on a long transcript, where re-rendering Markdown
  // takes longer than the gap between clicks — and every press passes the guard
  // and starts its own generation.
  //
  // That is not merely wasteful. The coordinator executes one request at a time
  // (the intake queue in src/coord/coord_main.c hands the executor thread a
  // single request), so the extra turns sit in its queue emitting NOTHING, and
  // whichever generation finishes first runs the tail below and clears the
  // state for all of them — leaving a silent UI, an empty bubble and a stop
  // button that no longer knows what to stop. A ref updates synchronously, so
  // the second press is refused in the same tick as the first.
  //
  // It maps conversation id -> request id rather than being a bare set of ids:
  // stop() needs the request id SYNCHRONOUSLY, and reading that out of
  // `liveMap` would reintroduce exactly the stale-state read this ref exists to
  // avoid.
  const liveRef = useRef<Map<string, string>>(new Map());
  // Browser dev-sim only: request ids whose simulated stream has been stopped.
  // The Tauri path cancels through the Rust registry, which the sim has no
  // access to — and a stop button that works in one build but not the other is
  // how the sim stops being somewhere this feature can be tested.
  const simStopRef = useRef<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  // "Follow the stream." A ref, not state: every delta reads it, and it must be
  // the value as of THIS scroll event, not as of the last render.
  const stickRef = useRef(true);
  // The same fact as state, for the one thing that has to re-render on it: the
  // jump-to-latest button.
  const [atBottom, setAtBottom] = useState(true);
  const online = props.api?.status === "online";

  // Who is actually answering. `modelOnly` stops the poll the moment the
  // coordinator has said — the model is fixed when it loads, so one answer is
  // the whole truth, and a repeating poll here would compete with generation on
  // a coordinator that serves requests serially.
  const stats = useClusterStats(props.api ?? null, props.source ?? "engine", {
    modelOnly: true,
    simModel: { id: props.modelId, label: props.modelLabel, quant: props.quant },
  });
  const served = servedModelOf(stats);
  // Falling back to the local setting is a WEAKER claim, so it is labeled
  // differently: the cluster loads its model at startup, so after changing the
  // setting the local value names a model nothing is running. Older engines do
  // not report one at all — then "selected" is all we can honestly say.
  const shownModel = served ?? { id: props.modelId, label: props.modelLabel, quant: props.quant };
  // No mismatch warning on the chip any more (2026-08-15): selecting a model
  // IS the switch, everywhere — the setting and the served model can only
  // disagree for the moments a rebuild is in flight.

  const active = convos.find((c) => c.id === activeId) ?? null;
  const msgs = active?.msgs ?? [];
  /** The generation running in the conversation on screen, if any. Every
   *  in-progress affordance below asks THIS question rather than "is anything
   *  generating" — answering the second one everywhere is what used to put the
   *  spinner on whichever conversation you happened to open. */
  const here = activeId ? liveMap.get(activeId) ?? null : null;
  const liveHere = here !== null;
  /** No more generations may be started until one of the running ones ends.
   *  Deliberately NOT a local queue: a queue the user cannot see is a promise
   *  the UI never shows itself keeping, and "why has nothing happened for two
   *  minutes" has no answer on screen. */
  const atLimit = liveMap.size >= CLIENT_MAX_INFLIGHT;
  /** Some OTHER conversation's generation has already been picked up by the
   *  cluster. That is the only evidence we have that a silent turn here is
   *  queued rather than merely not answered yet, and the copy below says so
   *  only when it holds. */
  const othersStarted =
    activeId !== null && [...liveMap].some(([id, g]) => id !== activeId && g.phase !== "waiting");

  // History is written on a debounce. The whole table is re-serialized per
  // write, and with several streams appending deltas that is a full JSON
  // encode per token — measurable on a long transcript, and pure waste since
  // only the last state of the batch is ever read back.
  const convosRef = useRef(convos);
  convosRef.current = convos;
  const lastWriteRef = useRef(0);
  const flushStore = useRef(() => {
    lastWriteRef.current = Date.now();
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(convosRef.current.slice(0, MAX_CONVOS)));
    } catch {
      /* storage full/blocked: history is a convenience, chatting still works */
    }
  });
  useEffect(() => {
    // Debounce, but never past STORE_MAX_WAIT_MS since the last write — see the
    // constant: a pure debounce never fires at all while a reply is streaming.
    const waited = Date.now() - lastWriteRef.current;
    const delay = Math.max(0, Math.min(STORE_DEBOUNCE_MS, STORE_MAX_WAIT_MS - waited));
    const h = setTimeout(() => flushStore.current(), delay);
    return () => clearTimeout(h);
  }, [convos]);
  // The two ways the debounce could otherwise swallow the last few hundred
  // milliseconds: closing the window, and navigating off the chat page.
  useEffect(() => {
    const onClose = () => flushStore.current();
    window.addEventListener("beforeunload", onClose);
    return () => {
      window.removeEventListener("beforeunload", onClose);
      flushStore.current();
    };
  }, []);

  // Opening a conversation always starts at its newest message, whatever the
  // previous one's scroll position was. Declared BEFORE the pin effect so the
  // re-attach lands before the pin that follows it in the same commit.
  useEffect(() => {
    stickRef.current = true;
    setAtBottom(true);
  }, [activeId]);

  // Keep the newest message in view while streaming — but only while the reader
  // is ALREADY at the bottom. Scrolling up during a generation is how you re-read
  // what was just said, and this effect used to slam the view back down on every
  // delta, i.e. several times a second: the thread could not be read at all until
  // the reply finished. Following the stream is a default, not a policy.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // The check is INSIDE pin, not around the effect: the deferred pin below
    // runs a frame later, and by then the reader may have scrolled away. When
    // that pin fired anyway it scrolled to the bottom, the browser reported a
    // scroll at the bottom, and this component read that as "the reader is
    // following again" — one stale frame was enough to re-arm the auto-scroll
    // permanently, which is the bug the whole check exists to prevent.
    const pin = () => {
      if (!stickRef.current) return;
      el.scrollTop = el.scrollHeight;
    };
    pin();
    // …and once more after layout settles. On the first paint of a restored
    // conversation the scroller can still be zero-height (flex sizing, web
    // fonts), so the assignment above clamps to 0 and a long thread opens at
    // its OLDEST message — the one place nobody wants to be.
    const raf = requestAnimationFrame(pin);
    return () => cancelAnimationFrame(raf);
    // `liveHere`, not "anything is generating": a stream landing in a
    // conversation you are not reading changes nothing about this scroller, and
    // re-pinning on it would fight the reader for no reason.
  }, [msgs, liveHere]);

  /** Re-arm following and go there. */
  const jumpToLatest = () => {
    const el = scrollRef.current;
    stickRef.current = true;
    setAtBottom(true);
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  // Whether we follow the stream is decided here, by where the reader is.
  // The threshold is generous on purpose: a line of text is ~26px, so "within
  // one or two lines of the end" still counts as watching the live text, and a
  // stream that grows the page by a line does not detach the view by itself.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight <= 96;
    stickRef.current = near;
    setAtBottom((prev) => (prev === near ? prev : near));
  };

  // Grow the composer with its content, up to the max-height CSS sets (then it
  // scrolls). Reset to "auto" first or scrollHeight only ever ratchets upwards.
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  /** Update the active conversation's messages, keeping it at the top of the list. */
  const setActiveMsgs = (fn: (prev: ChatMsg[]) => ChatMsg[], id: string) => {
    setConvos((cs) =>
      cs.map((c) => (c.id === id ? { ...c, msgs: fn(c.msgs).slice(-MAX_STORED), updatedAt: Date.now() } : c))
    );
  };

  const newChat = () => {
    setInput("");
    // An empty conversation already at the top is the same thing — reuse it
    // instead of stacking blanks every time the button is pressed.
    const blank = convos.find((c) => c.msgs.length === 0);
    if (blank) {
      setActiveId(blank.id);
      return;
    }
    const c: Conversation = { id: newId(), title: "", msgs: [], updatedAt: Date.now() };
    setConvos((cs) => [c, ...cs].slice(0, MAX_CONVOS));
    setActiveId(c.id);
  };

  const deleteChat = (id: string) => {
    setConvos((cs) => {
      const out = cs.filter((c) => c.id !== id);
      if (id === activeId) setActiveId(out[0]?.id ?? null);
      return out;
    });
  };

  /** May a fresh generation start in `convoId` right now?
   *
   *  Two independent refusals, both read from the ref so the answer is the one
   *  as of THIS tick: the conversation already has a reply in flight (a second
   *  send would race the first over the same transcript), and the client-wide
   *  ceiling. Both are also reflected in the composer, so reaching either of
   *  them here should be rare — but "rare" is not "never", and a guard that
   *  only the UI enforces is not a guard. */
  const canGenerate = (convoId: string) =>
    !liveRef.current.has(convoId) && liveRef.current.size < CLIENT_MAX_INFLIGHT;

  const send = async () => {
    const q = input.trim();
    if (!q || !online || !props.api) return;
    // Checked BEFORE the conversation is created and before the box is cleared:
    // a refused send that had already emptied the composer would look like the
    // app swallowing the message.
    if (liveRef.current.size >= CLIENT_MAX_INFLIGHT) return;

    // Sending from a fresh window creates the conversation.
    let id = activeId;
    if (!id || !convos.some((c) => c.id === id)) {
      const c: Conversation = { id: newId(), title: "", msgs: [], updatedAt: Date.now() };
      setConvos((cs) => [c, ...cs].slice(0, MAX_CONVOS));
      setActiveId(c.id);
      id = c.id;
    }
    if (!canGenerate(id)) return;
    setInput("");
    const history = [...(convos.find((c) => c.id === id)?.msgs ?? []), { role: "user" as const, text: q }];
    await generate(id, history);
  };

  /** Stream one assistant reply into `convoId`, whose transcript becomes
   *  `history` + the reply. Shared by send (history = old msgs + the new user
   *  turn) and regenerate (history = old msgs cut back to the last user turn) —
   *  the engine is stateless per request, so both are just "replay this list". */
  const generate = async (convoId: string, history: ChatMsg[]) => {
    if (!canGenerate(convoId) || !online || !props.api) return;
    const api = props.api;

    // One id per generation, minted before anything else so it can be recorded
    // synchronously: it is what routes this stream's `api-chat` events back to
    // THIS conversation (the Rust bridge tags every event with it, and
    // app.emit broadcasts to the whole webview — without the filter two
    // conversations' deltas land in one bubble), and what api_chat_cancel
    // takes. In the browser sim it is only a cancellation handle.
    const reqId = `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    liveRef.current.set(convoId, reqId);
    // Starts in "waiting": the request is out and the cluster has said nothing.
    setLiveMap((m) => new Map(m).set(convoId, { reqId, phase: "waiting", prefill: null, startedAt: Date.now() }));
    /** Move this conversation's generation on. A no-op once it has ended —
     *  a late event from a stream we already cleaned up must not resurrect it. */
    const setPhase = (phase: LiveGen["phase"], prefill: LiveGen["prefill"]) =>
      setLiveMap((m) => {
        const cur = m.get(convoId);
        if (!cur || cur.reqId !== reqId) return m;
        return new Map(m).set(convoId, { ...cur, phase, prefill });
      });
    // Starting a generation is an explicit "I am at the live end again", so it
    // re-arms following even if you had scrolled up to re-read something first.
    // Safe to do unconditionally: generation is only ever started from the
    // conversation on screen (send and regenerate both act on the active one).
    stickRef.current = true;
    setAtBottom(true);
    setConvos((cs) =>
      cs.map((c) =>
        c.id === convoId
          ? {
              ...c,
              title: c.title || titleOf(history),
              msgs: [...history, { role: "assistant" as const, text: "", sim: !inTauri() }].slice(-MAX_STORED),
              updatedAt: Date.now(),
            }
          : c
      )
    );

    try {
      if (inTauri()) {
        const { listen } = await import("@tauri-apps/api/event");
        const { invoke } = await import("@tauri-apps/api/core");
        // Failed turns are not conversation: sending an assistant message that
        // never happened (or is half a sentence) asks the model to continue
        // from a reply it never gave.
        const wire = history
          .filter((m) => !m.error && (m.role === "user" || m.text))
          .map((m) => ({ role: m.role, content: m.text }));
        const t0 = performance.now();
        let tFirst = 0;
        let nDeltas = 0;
        let lastPrefill: { done: number; total: number; reused: number } | null = null;
        await new Promise<void>((resolve, reject) => {
          let un: (() => void) | null = null;
          listen<{ id: string; kind: string; text?: string; message?: string; done?: number; total?: number; reused?: number }>("api-chat", (ev) => {
            const p = ev.payload;
            if (p.id !== reqId) return;
            if (p.kind === "progress") {
              lastPrefill = { done: p.done ?? 0, total: p.total ?? 0, reused: p.reused ?? 0 };
              setPhase("prefill", lastPrefill);
            }
            if (p.kind === "delta" && p.text) {
              if (nDeltas === 0) {
                tFirst = performance.now();
                setPhase("streaming", null); // prefill is over the moment real text arrives
              }
              nDeltas++;
              setActiveMsgs((m) => {
                const out = [...m];
                out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text + p.text };
                return out;
              }, convoId);
            }
            if (p.kind === "done") {
              un?.();
              if (nDeltas > 0) {
                const stats = {
                  ttftMs: tFirst - t0,
                  totalMs: performance.now() - t0,
                  tokens: nDeltas,
                  reused: lastPrefill?.reused,
                  promptTokens: lastPrefill?.total,
                };
                setActiveMsgs((m) => {
                  const out = [...m];
                  out[out.length - 1] = { ...out[out.length - 1], stats };
                  return out;
                }, convoId);
              }
              resolve();
            }
            if (p.kind === "error") {
              un?.();
              reject(new Error(p.message || "request failed"));
            }
          }).then((u) => {
            un = u;
            invoke("api_chat_stream", {
              id: reqId,
              baseUrl: api.baseUrl,
              messages: wire,
              token: props.apiToken,
              model: props.modelId,
              maxTokens: props.maxTokens,
            }).catch((e) => {
              un?.();
              reject(e as Error);
            });
          });
        });
      } else {
        // Browser dev build: simulated streaming echo, clearly labeled per-message.
        // Each call has its own loop and its own timers, so several of these run
        // side by side exactly as several real streams do — which is what makes
        // the dev channel a place this feature can be hand-tested without a
        // cluster. Nothing here is serialized: the sim has no queue to model.
        const q = [...history].reverse().find((m) => m.role === "user")?.text ?? "";
        const reply = `“${q}” ✓`;
        let first = true;
        for (const ch of reply.split("")) {
          await new Promise((ok) => setTimeout(ok, 30));
          if (simStopRef.current.has(reqId)) break; // stop() pressed on this conversation
          if (first) {
            first = false;
            setPhase("streaming", null);
          }
          setActiveMsgs((m) => {
            const out = [...m];
            out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text + ch };
            return out;
          }, convoId);
        }
      }
    } catch (e) {
      const raw = String((e as Error)?.message ?? e);
      // Client-authored errors (the "[CODE] detail" convention, e.g. the two
      // stream timeouts) render localized; engine verbatim text passes through.
      const msg = tErr(raw);
      // Mark the turn as failed and KEEP it, partial text and all. It used to be
      // deleted, which threw away both the evidence and the fact that anything
      // had been attempted.
      let partial = false;
      setActiveMsgs((m) => {
        const out = [...m];
        const last = out[out.length - 1];
        if (last?.role === "assistant") {
          partial = !!last.text;
          out[out.length - 1] = { ...last, error: msg };
        }
        return out;
      }, convoId);
      recordProblem({
        at: new Date().toISOString(),
        kind: "chat",
        // The raw (English, code-prefixed) form: the problem log is a
        // diagnostic artifact and must not vary with the UI language.
        message: raw,
        detail: {
          model: props.modelId,
          quant: props.quant,
          // Turn number within the conversation (1 = first). A failure that only
          // ever happens from turn 2 onward points somewhere very different from
          // one that happens on turn 1.
          turn: Math.ceil(history.length / 2),
          partialReply: partial,
        },
      });
    } finally {
      // MUST be `finally`. This used to be straight-line code after the
      // try/catch, so anything that threw on the way out — including inside the
      // catch block itself — skipped it and left the guard stuck at "running".
      // From then on EVERY later generation in this conversation is refused,
      // silently: no text, and a stop button that has no id to stop. Permanent,
      // and it survives switching conversations. One escape hatch being
      // conditional on nothing having gone wrong is not an escape hatch.
      //
      // It clears ONLY this conversation's row. Clearing the lot — which the
      // single-flag version effectively did — would strand every other stream:
      // still running in Rust, no longer stoppable from here.
      clearLive(convoId, reqId);
    }
  };

  /** Forget one conversation's generation. Safe to run twice: generate()'s own
   *  finally does exactly this.
   *
   *  `reqId` guards against clearing a row that a LATER generation in the same
   *  conversation has already replaced — a stop pressed just as the previous
   *  run ended would otherwise wipe the new one's state. */
  const clearLive = (convoId: string, reqId?: string) => {
    if (reqId && liveRef.current.get(convoId) !== reqId) return;
    liveRef.current.delete(convoId);
    if (reqId) simStopRef.current.delete(reqId);
    setLiveMap((m) => {
      if (!m.has(convoId)) return m;
      const next = new Map(m);
      next.delete(convoId);
      return next;
    });
  };

  /** Stop one conversation's generation, keeping whatever has already streamed
   *  in. Every other conversation's stream is untouched: cancellation is keyed
   *  by request id all the way down to the Rust registry, which holds one
   *  cancel flag per id and drops only that stream's socket. */
  const stop = async (convoId: string) => {
    const reqId = liveRef.current.get(convoId);
    if (reqId) {
      if (inTauri()) {
        const { invoke } = await import("@tauri-apps/api/core");
        // api_chat_cancel answers whether it FOUND a stream to cancel. The answer
        // used to be discarded, so "there is nothing by that id" and "cancelled"
        // were indistinguishable and a press that reached neither was silent.
        const found = await invoke<boolean>("api_chat_cancel", { id: reqId }).catch(() => false);
        if (found) return; // it will stop itself, and generate()'s finally cleans up
      } else {
        // Browser sim: its loop checks this set between characters and then
        // falls into the same finally.
        simStopRef.current.add(reqId);
        return;
      }
    }
    // Either nothing is registered under this id or there is no id at all: this
    // view is showing a generation that is not running. Recovering the state is
    // the only useful thing left — a stop button that provably does nothing is
    // worse than no button, because the user keeps pressing it.
    clearLive(convoId);
  };

  /** Re-run the last exchange: cut the trailing assistant turn (reply, error,
   *  or both — a stopped reply then a retry can stack two) and replay what is
   *  left. The old reply is not kept as an alternative; on a home cluster a
   *  reply costs minutes, and someone regenerating has already judged it. */
  const regen = async () => {
    const cur = convos.find((c) => c.id === activeId);
    if (!cur || !canGenerate(cur.id) || !online || !props.api) return;
    let end = cur.msgs.length;
    while (end > 0 && cur.msgs[end - 1].role === "assistant") end--;
    if (end === 0) return; // nothing but assistant turns: no prompt to replay
    await generate(cur.id, cur.msgs.slice(0, end));
  };

  // Offline is a SENDING problem, not a reading one. History lives in this
  // machine's localStorage and needs no cluster to display, so the page renders
  // in full either way — only the composer is swapped for the reason you can't
  // send. The old behaviour replaced the whole view with "go start a cluster",
  // which read as "your conversations are gone".
  return (
    <main className="main chat chat--withbar">
      <aside className="chatbar">
        {/* Switching and starting conversations stay live during a generation.
            They used to be disabled by `busy`, which on a LAN cluster meant the
            whole sidebar locked up for minutes — you could not read anything
            else while waiting. Nothing about the stream needs the lock:
            setActiveMsgs writes to an explicit conversation id, so deltas keep
            landing in the right thread no matter what is on screen. */}
        <button className="btn-secondary chatbar__new" onClick={newChat}>
          + {t("chat.new")}
        </button>
        <div className="chatbar__list">
          {convos.length === 0 ? <p className="chatbar__empty">{t("chat.noConvos")}</p> : null}
          {convos.map((c) => {
            const gen = liveMap.get(c.id) ?? null;
            // A queued turn gets a still dot, a running one a pulsing dot.
            // Same rule as the bubble: motion means something is happening.
            const label = gen?.phase === "waiting" ? t("chat.queued") : t("chat.generating");
            return (
            <div key={c.id} className={`chatbar__item${c.id === activeId ? " is-on" : ""}`}>
              <button className="chatbar__pick" onClick={() => setActiveId(c.id)}>
                <span className="chatbar__title">{c.title || t("chat.untitled")}</span>
                {/* Where the replies are landing, for when you have navigated
                    away. Outside the truncating span, or a long title would
                    ellipsis it away exactly when it matters most. Now one per
                    generating conversation, not one for the app. */}
                {gen ? (
                  <span
                    className={`chatbar__live${gen.phase === "waiting" ? " chatbar__live--queued" : ""}`}
                    title={label}
                    aria-label={label}
                  />
                ) : null}
              </button>
              <button
                className="chatbar__del"
                onClick={() => deleteChat(c.id)}
                // Deleting the thread being written to would drop the reply on
                // the floor and leave Stop pointing at nothing. Only THIS
                // conversation is locked; the others stay deletable however
                // many of them are generating.
                disabled={!!gen}
                aria-label={t("chat.delete")}
                title={t("chat.delete")}
              >
                ✕
              </button>
            </div>
            );
          })}
        </div>
      </aside>

      <div className="chat__main">
        {/* Which model is answering. It used to be nowhere on this page: the
            reply's tone, speed and quality all come from a choice made on
            another screen, and going back to Settings was the only way to find
            out which one — a saved conversation could not even be read back
            knowing what wrote it. The two claims are kept apart: "serving" is
            the coordinator reporting what it loaded, "selected" is this
            machine's setting when the cluster has not said. */}
        <div className="chat__head">
          {/* Display only (2026-08-15 split): chatting happens against
              whatever is running, and switching models is a cluster operation
              that lives on the Cluster page. The chip stopped being a picker —
              a page for talking is not a place to tear the cluster down. */}
          <div
            className={`modelchip${served ? " modelchip--live" : ""}`}
            title={served ? t("chat.model.servingTitle") : t("chat.model.selectedTitle")}
          >
            <span className="modelchip__label">{t(served ? "model.serving" : "model.selected")}</span>
            <span className="modelchip__name">{shownModel.label}</span>
            {shownModel.quant ? <span className="modelchip__quant">{shownModel.quant}</span> : null}
          </div>
        </div>
        <div className="chat__scroll" ref={scrollRef} onScroll={onScroll}>
          <div className="chat__thread">
            {msgs.map((m, i) => (
              <div key={i} className={`chat-msg chat-msg--${m.role}${i === msgs.length - 1 ? " chat-msg--tail" : ""}`}>
                {m.role === "user"
                  ? <UserAvatar identity={props.identity} />
                  : <div className="chat-avatar chat-avatar--assistant" aria-hidden="true" />}
                <div className="chat-msg__body">
                  <div className="chat-msg__who">
                    {m.role === "user" ? props.identity?.name ?? t("chat.you") : t("chat.assistant")}
                  </div>
                  <Bubble
                    m={m}
                    live={liveHere}
                    last={i === msgs.length - 1}
                    phase={here?.phase ?? null}
                    behind={othersStarted}
                    prefill={here?.prefill ?? null}
                  />
                  {m.stats ? (
                    <div className="chat-msg__stats">
                      {t("tryit.stats", {
                        ttft: (m.stats.ttftMs / 1000).toFixed(1),
                        n: m.stats.tokens,
                        rate:
                          m.stats.totalMs > m.stats.ttftMs && m.stats.tokens > 1
                            ? (((m.stats.tokens - 1) * 1000) / (m.stats.totalMs - m.stats.ttftMs)).toFixed(1)
                            : "—",
                      })}
                      {/* The cache verdict, per reply. TTFT climbing turn over
                          turn is expected even on a hit (attention spans more
                          history each time), so the counter is the thing that
                          actually answers "is the cache working" — not the clock. */}
                      {m.stats.promptTokens ? (
                        <span className={m.stats.reused ? "chat-msg__kv" : "chat-msg__kv chat-msg__kv--miss"}>
                          {" · "}
                          {m.stats.reused
                            ? t("chat.kvHit", {
                                reused: m.stats.reused,
                                total: m.stats.promptTokens,
                              })
                            : t("chat.kvMiss", { total: m.stats.promptTokens })}
                        </span>
                      ) : null}
                    </div>
                  ) : null}
                  {/* Actions only once the turn is settled — while text is
                      still streaming into it, copy would take half a reply.
                      Regenerate lives on the last message only: re-running an
                      EARLIER turn would have to throw away everything after
                      it, which is a different (destructive) feature. */}
                  {!(liveHere && i === msgs.length - 1) ? (
                    <MsgActions
                      m={m}
                      canRegen={i === msgs.length - 1 && m.role === "assistant" && online && !liveHere && !atLimit}
                      onRegen={() => void regen()}
                    />
                  ) : null}
                </div>
              </div>
            ))}
            {/* No banner: the failure is rendered on the turn it belongs to,
                where it persists and can be copied. A banner here would say the
                same thing twice and then delete its copy on the next send. */}
          </div>
        </div>
        {/* The way back. Once you have scrolled up, the stream no longer drags
            the view along — so there has to be one click that returns to it,
            or catching up means scrolling against text that keeps growing.
            Only shown when detached AND there is something to come back to. */}
        {!atBottom && msgs.length > 0 ? (
          <button className={`tolatest${liveHere ? " tolatest--live" : ""}`} onClick={jumpToLatest}>
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
              <path d="M12 5v14M5 12l7 7 7-7" fill="none" stroke="currentColor" strokeWidth="2.2"
                    strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {t(liveHere ? "chat.toLive" : "chat.toLatest")}
          </button>
        ) : null}
        {!online ? (
          <div className="chat__composer">
            <div className="chat-offline">
              <p className="chat-offline__msg">{t("chat.needCluster")}</p>
              <button className="btn-primary" onClick={props.onGoCluster}>
                {t("chat.goCluster")} →
              </button>
            </div>
          </div>
        ) : (
        <div className="chat__composer">
          <div className="composer">
            <textarea
              ref={taRef}
              className="composer__input"
              value={input}
              rows={1}
              placeholder={t("chat.placeholder")}
              // Live while OTHER conversations generate — that is the whole
              // point of the change. Only two things close it: this thread is
              // already answering, and the client-wide ceiling.
              disabled={liveHere || atLimit}
              onChange={(e) => setInput(e.target.value)}
              // Enter sends, Shift+Enter breaks the line — the convention every
              // chat client shares. isComposing guards IME candidate selection,
              // where Enter means "accept this candidate", not "send".
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            {/* Stop belongs to the conversation being generated. Showing it
                while reading a different thread would offer to cancel
                something not on screen. */}
            {liveHere && activeId ? (
              <button
                className="composer__btn composer__btn--stop"
                onClick={() => void stop(activeId)}
                /* Deliberately never disabled. It used to be `!liveId`, i.e.
                 * the button went dead in exactly the state where the user most
                 * needs it — the view says "generating" but no id is on record.
                 * stop() now treats that case as "recover this view". */
                title={t("chat.stop")}
                aria-label={t("chat.stop")}
              >
                <span className="composer__stopmark" />
              </button>
            ) : (
              <button
                className="composer__btn"
                disabled={atLimit || !input.trim()}
                onClick={() => send()}
                title={t("tryit.send")}
                aria-label={t("tryit.send")}
              >
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path d="M12 19V5M12 5l-6 6M12 5l6 6" fill="none" stroke="currentColor" strokeWidth="2"
                        strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            )}
          </div>
          {/* Why the composer is dead — otherwise it just looks broken. The
              old "another conversation is generating, one at a time" hint is
              gone with the restriction it explained. */}
          {atLimit ? (
            <p className="composer__hint">{t("chat.limit", { n: CLIENT_MAX_INFLIGHT })}</p>
          ) : (
            <p className="composer__hint">{t("chat.sendHint")}</p>
          )}
        </div>
        )}
      </div>
    </main>
  );
}
