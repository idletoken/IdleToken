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
import { recordChatFailure } from "./chatErrors";

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

const STORE_KEY = "idletoken.chat.v2";
const LEGACY_KEY = "idletoken.chat.v1"; // single rolling transcript
const MAX_STORED = 50; // messages kept per conversation
const MAX_CONVOS = 40;

const newId = () => `c-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

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
        return parsed.slice(0, MAX_CONVOS).map((c) => ({ ...c, title: c.msgs?.length ? titleOf(c.msgs) : c.title }));
      }
    }
    // Migrate the old single transcript rather than dropping it on the floor.
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const msgs = JSON.parse(legacy) as ChatMsg[];
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

function Bubble(props: {
  m: ChatMsg;
  busy: boolean;
  last: boolean;
  /** Prefill progress, while the cluster is still reading the prompt. */
  prefill: { done: number; total: number; reused: number } | null;
}) {
  const { t } = useI18n();
  const [openReasoning, setOpenReasoning] = useState(false);
  const { reasoning, answer, thinking } = useMemo(() => splitThink(props.m.text), [props.m.text]);

  if (props.m.role === "user") {
    return <div className="chat-msg__bubble">{props.m.text}</div>;
  }
  const waiting = props.busy && props.last && !props.m.text;
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
      {waiting || (thinking && !answer) ? <span className="chat-msg__dots">…</span> : null}
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
}) {
  const { t } = useI18n();
  const [convos, setConvos] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(() => loadConversations()[0]?.id ?? null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // Which conversation the in-flight generation belongs to. Reading the
  // sidebar while a reply streams is allowed, so "is something generating"
  // (busy) and "is it generating HERE" are different questions — every
  // in-progress affordance below asks the second one. Answering the first
  // everywhere is what made the spinner appear on whichever conversation you
  // happened to open.
  const [liveConvo, setLiveConvo] = useState<string | null>(null);
  // Id of the generation currently streaming, so it can be stopped. A reply can
  // run for minutes; without this the only way out is quitting the app.
  const [liveId, setLiveId] = useState<string | null>(null);
  const [prefill, setPrefill] = useState<{ done: number; total: number; reused: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const online = props.api?.status === "online";

  const active = convos.find((c) => c.id === activeId) ?? null;
  const msgs = active?.msgs ?? [];
  /** A reply is streaming into the conversation currently on screen. */
  const liveHere = busy && activeId === liveConvo;

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(convos.slice(0, MAX_CONVOS)));
    } catch {
      /* storage full/blocked: history is a convenience, chatting still works */
    }
  }, [convos]);

  // Keep the newest message in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const pin = () => {
      el.scrollTop = el.scrollHeight;
    };
    pin();
    // …and once more after layout settles. On the first paint of a restored
    // conversation the scroller can still be zero-height (flex sizing, web
    // fonts), so the assignment above clamps to 0 and a long thread opens at
    // its OLDEST message — the one place nobody wants to be.
    const raf = requestAnimationFrame(pin);
    return () => cancelAnimationFrame(raf);
  }, [msgs, busy]);

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

  const send = async () => {
    const q = input.trim();
    if (!q || busy || !online || !props.api) return;
    const api = props.api;

    // Sending from a fresh window creates the conversation.
    let id = activeId;
    if (!id || !convos.some((c) => c.id === id)) {
      const c: Conversation = { id: newId(), title: "", msgs: [], updatedAt: Date.now() };
      setConvos((cs) => [c, ...cs].slice(0, MAX_CONVOS));
      setActiveId(c.id);
      id = c.id;
    }
    const convoId = id;

    setBusy(true);
    setLiveConvo(convoId);
    setInput("");
    const history = [...(convos.find((c) => c.id === convoId)?.msgs ?? []), { role: "user" as const, text: q }];
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
        const reqId = `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        setLiveId(reqId);
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
              setPrefill({ done: p.done ?? 0, total: p.total ?? 0, reused: p.reused ?? 0 });
              lastPrefill = { done: p.done ?? 0, total: p.total ?? 0, reused: p.reused ?? 0 };
            }
            if (p.kind === "delta" && p.text) {
              if (nDeltas === 0) {
                tFirst = performance.now();
                setPrefill(null); // prefill is over the moment real text arrives
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
        const reply = `“${q}” ✓`;
        for (const ch of reply.split("")) {
          await new Promise((ok) => setTimeout(ok, 30));
          setActiveMsgs((m) => {
            const out = [...m];
            out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text + ch };
            return out;
          }, convoId);
        }
      }
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
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
      recordChatFailure({
        at: new Date().toISOString(),
        message: msg,
        modelId: props.modelId,
        quant: props.quant,
        turn: Math.ceil(history.length / 2),
        hadPartialReply: partial,
      });
    }
    setLiveId(null);
    setLiveConvo(null);
    setPrefill(null);
    setBusy(false);
  };

  /** Stop the running generation, keeping whatever has already streamed in. */
  const stop = async () => {
    if (!liveId) return;
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("api_chat_cancel", { id: liveId }).catch(() => {});
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
          {convos.map((c) => (
            <div key={c.id} className={`chatbar__item${c.id === activeId ? " is-on" : ""}`}>
              <button className="chatbar__pick" onClick={() => setActiveId(c.id)}>
                <span className="chatbar__title">{c.title || t("chat.untitled")}</span>
                {/* Where the reply is landing, for when you have navigated away.
                    Outside the truncating span, or a long title would ellipsis
                    it away exactly when it matters most. */}
                {busy && c.id === liveConvo ? (
                  <span className="chatbar__live" title={t("chat.generating")} aria-label={t("chat.generating")} />
                ) : null}
              </button>
              <button
                className="chatbar__del"
                onClick={() => deleteChat(c.id)}
                // Deleting the thread being written to would drop the reply on
                // the floor and leave Stop pointing at nothing. Every other
                // conversation is fair game.
                disabled={busy && c.id === liveConvo}
                aria-label={t("chat.delete")}
                title={t("chat.delete")}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      <div className="chat__main">
        <div className="chat__scroll" ref={scrollRef}>
          <div className="chat__thread">
            {msgs.length === 0 ? <p className="chat-hint">{t("chat.hint")}</p> : null}
            {msgs.map((m, i) => (
              <div key={i} className={`chat-msg chat-msg--${m.role}`}>
                {m.role === "user"
                  ? <UserAvatar identity={props.identity} />
                  : <div className="chat-avatar chat-avatar--assistant" aria-hidden="true" />}
                <div className="chat-msg__body">
                  <div className="chat-msg__who">
                    {m.role === "user" ? props.identity?.name ?? t("chat.you") : t("chat.assistant")}
                  </div>
                  <Bubble m={m} busy={liveHere} last={i === msgs.length - 1} prefill={liveHere ? prefill : null} />
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
                </div>
              </div>
            ))}
            {/* No banner: the failure is rendered on the turn it belongs to,
                where it persists and can be copied. A banner here would say the
                same thing twice and then delete its copy on the next send. */}
          </div>
        </div>
        {!online ? (
          <div className="chat__composer">
            <div className="chat-offline">
              <p className="chat-offline__msg">{t("chat.needCluster")}</p>
              <button className="btn-primary" onClick={props.onGoCluster}>
                {t("chat.goCluster")} →
              </button>
            </div>
            <p className="composer__hint">{t("chat.offlineHistory")}</p>
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
              disabled={busy}
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
            {liveHere ? (
              <button
                className="composer__btn composer__btn--stop"
                onClick={() => void stop()}
                disabled={!liveId}
                title={t("chat.stop")}
                aria-label={t("chat.stop")}
              >
                <span className="composer__stopmark" />
              </button>
            ) : (
              <button
                className="composer__btn"
                // Still one generation at a time: the coordinator serves a
                // single sequence slot by default, so a second send would only
                // queue behind the first (or 429).
                disabled={busy || !input.trim()}
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
          {/* Why the composer is dead while you are reading another thread —
              otherwise it just looks broken. */}
          {busy && !liveHere ? (
            <p className="composer__hint">
              {t("chat.otherBusy")}{" "}
              <button className="linkbtn" onClick={() => liveConvo && setActiveId(liveConvo)}>
                {t("chat.goLive")}
              </button>
            </p>
          ) : (
            <p className="composer__hint">{t("chat.sendHint")}</p>
          )}
        </div>
        )}
      </div>
    </main>
  );
}
