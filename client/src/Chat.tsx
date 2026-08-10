// Chat — the real multi-turn chat surface (2026-07 IA pass). The dashboard's
// old one-shot "try it" box looked like chat but wasn't; this is the place the
// sidebar's Chat entry leads to: history, streaming, bottom input -- the shape users
// expect. The engine is stateless per request, so every send resends the full
// conversation (the Rust relay takes the messages array verbatim).
//
// History is device-local only (localStorage, capped) — nothing leaves the
// machine except the API call to your own cluster.
import { useEffect, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { inTauri } from "./platform";
import type { ClusterApi } from "./pairing";

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  sim?: boolean; // browser dev-sim reply (labeled, never mistaken for real)
  stats?: { ttftMs: number; totalMs: number; tokens: number };
}

const STORE_KEY = "idletoken.chat.v1";
const MAX_STORED = 50;

function loadHistory(): ChatMsg[] {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMsg[];
    return Array.isArray(parsed) ? parsed.slice(-MAX_STORED) : [];
  } catch {
    return [];
  }
}

export default function Chat(props: {
  api: ClusterApi | null;
  apiToken: string;
  modelId: string;
  initialPrompt: string | null;
  onInitialConsumed: () => void;
  onGoCluster: () => void;
}) {
  const { t } = useI18n();
  const [msgs, setMsgs] = useState<ChatMsg[]>(loadHistory);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const online = props.api?.status === "online";

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(msgs.slice(-MAX_STORED)));
    } catch {
      /* storage full/blocked: history is a convenience, chatting still works */
    }
  }, [msgs]);

  // Keep the newest message in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs, busy]);

  const send = async (raw?: string) => {
    const q = (raw ?? input).trim();
    if (!q || busy || !online || !props.api) return;
    const api = props.api;
    setBusy(true);
    setErr(null);
    setInput("");
    const history = [...msgs, { role: "user" as const, text: q }];
    setMsgs([...history, { role: "assistant", text: "", sim: !inTauri() }]);
    try {
      if (inTauri()) {
        const id = `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const { listen } = await import("@tauri-apps/api/event");
        const { invoke } = await import("@tauri-apps/api/core");
        const wire = history.map((m) => ({ role: m.role, content: m.text }));
        const t0 = performance.now();
        let tFirst = 0;
        let nDeltas = 0;
        await new Promise<void>((resolve, reject) => {
          let un: (() => void) | null = null;
          listen<{ id: string; kind: string; text?: string; message?: string }>("api-chat", (ev) => {
            const p = ev.payload;
            if (p.id !== id) return;
            if (p.kind === "delta" && p.text) {
              if (nDeltas === 0) tFirst = performance.now();
              nDeltas++;
              setMsgs((m) => {
                const out = [...m];
                out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text + p.text };
                return out;
              });
            }
            if (p.kind === "done") {
              un?.();
              if (nDeltas > 0) {
                const stats = { ttftMs: tFirst - t0, totalMs: performance.now() - t0, tokens: nDeltas };
                setMsgs((m) => {
                  const out = [...m];
                  out[out.length - 1] = { ...out[out.length - 1], stats };
                  return out;
                });
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
              id,
              baseUrl: api.baseUrl,
              messages: wire,
              token: props.apiToken,
              model: props.modelId,
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
          setMsgs((m) => {
            const out = [...m];
            out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text + ch };
            return out;
          });
        }
      }
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
      // Drop the empty assistant bubble; keep the user's message for retry.
      setMsgs((m) => (m[m.length - 1]?.role === "assistant" && !m[m.length - 1].text ? m.slice(0, -1) : m));
    }
    setBusy(false);
  };

  // Launched from the cluster card's quick box: auto-send once when online.
  const initialSent = useRef(false);
  useEffect(() => {
    if (props.initialPrompt && online && !initialSent.current) {
      initialSent.current = true;
      const p = props.initialPrompt;
      props.onInitialConsumed();
      void send(p);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.initialPrompt, online]);

  if (!online) {
    return (
      <main className="main chat chat--empty">
        <div className="chat-empty">
          <p className="chat-empty__msg">{t("chat.needCluster")}</p>
          <button className="btn-primary" onClick={props.onGoCluster}>
            {t("chat.goCluster")} →
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="main chat">
      <div className="chat__scroll" ref={scrollRef}>
        {msgs.length === 0 ? <p className="chat-hint">{t("chat.hint")}</p> : null}
        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg chat-msg--${m.role}`}>
            <div className="chat-msg__bubble">
              {m.text || (busy && i === msgs.length - 1 ? "…" : "")}
              {m.sim && m.text ? <span className="tryit__sim"> {t("tryit.browserNote")}</span> : null}
            </div>
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
              </div>
            ) : null}
          </div>
        ))}
        {err ? <div className="engine-meta engine-meta--bad">{err}</div> : null}
      </div>
      <div className="chat__inputrow">
        <input
          className="tryit__input"
          value={input}
          placeholder={t("chat.placeholder")}
          disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && send()}
        />
        <button className="btn-primary tryit__send" disabled={busy || !input.trim()} onClick={() => send()}>
          {busy ? t("tryit.busy") : t("tryit.send")}
        </button>
        {msgs.length > 0 ? (
          <button className="linkbtn" onClick={() => setMsgs([])} disabled={busy}>
            {t("chat.clear")}
          </button>
        ) : null}
      </div>
    </main>
  );
}
