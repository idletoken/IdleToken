// "Connect a client here" — the block that tells the user their own base URL
// and token (docs/api-surface.md §7).
//
// It exists because the client never said either one. You could start a
// cluster, watch it go green, and still have to go find the README to point
// Claude Code at the thing you had just started. The two scenarios in §1 differ
// by exactly two facts — which address, which key — so both are shown side by
// side, each with the one sentence that says when it is the right one.
import { useEffect, useState } from "react";
import { useI18n, type Lang } from "./i18n";
import type { AppSettings } from "./settings";
import { getPairingProvider } from "./pairing";

type Bi = { en: string; zh: string };
const L = (b: Bi, lang: Lang) => b[lang];

/** Copy-to-clipboard button that reports success in place, like the roster's. */
function CopyBtn(props: { text: string; lang: Lang }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="linkbtn"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(props.text);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          /* the value is on screen either way — nothing to recover */
        }
      }}
    >
      {done ? L({ en: "Copied ✓", zh: "已复制 ✓" }, props.lang) : L({ en: "Copy", zh: "复制" }, props.lang)}
    </button>
  );
}

function Row(props: { label: string; value: string; mono?: boolean; lang: Lang }) {
  return (
    <div className="endpoint-row">
      <span className="endpoint-row__label">{props.label}</span>
      <code className={props.mono === false ? undefined : "endpoint-row__value"}>{props.value}</code>
      <CopyBtn text={props.value} lang={props.lang} />
    </div>
  );
}

export function EndpointsPanel(props: { settings: AppSettings }) {
  const { lang } = useI18n();
  const s = props.settings;
  const [liveBase, setLiveBase] = useState<string | null>(null);

  // Prefer the address the cluster is actually serving on: apiHost is often
  // 0.0.0.0, which is a bind address and not something anyone can type into a
  // client. Fall back to the configured values when no cluster is up, so the
  // block is still useful before you press start.
  useEffect(() => {
    let alive = true;
    try {
      const off = getPairingProvider().subscribe((snap) => {
        if (!alive) return;
        setLiveBase(snap.api?.status === "online" && snap.api.baseUrl ? snap.api.baseUrl : null);
      });
      return () => {
        alive = false;
        off();
      };
    } catch {
      // No provider (plain browser, no Tauri): fall back to the configured
      // values below rather than rendering nothing.
      return () => {
        alive = false;
      };
    }
  }, []);

  const host =
    s.apiHost && s.apiHost !== "0.0.0.0"
      ? s.apiHost
      : L({ en: "<this machine's LAN IP>", zh: "<本机局域网 IP>" }, lang);
  const localBase = liveBase ?? `http://${host}:${s.apiPort}`;
  const localKey = s.apiToken || "";
  const cloudBase = (s.platformUrl || "https://api.idletoken.ai").replace(/\/+$/, "");

  const localExport =
    `export ANTHROPIC_BASE_URL=${localBase}\n` +
    `export ANTHROPIC_API_KEY=${localKey || "unused"}\n` +
    `claude`;
  const cloudExport =
    `export ANTHROPIC_BASE_URL=${cloudBase}\n` +
    `export ANTHROPIC_API_KEY=<your API key>\n` +
    `claude`;

  return (
    <div className="setting-group endpoints">
      <div className="setting-group__label">{L({ en: "Connect a client", zh: "接入客户端" }, lang)}</div>

      <div className="endpoint-card">
        <div className="endpoint-card__title">{L({ en: "This machine", zh: "本机" }, lang)}</div>
        <p className="setting-hint">
          {L(
            {
              en: "Use this while you have your own cluster running. Requests are served by your hardware and cost nothing.",
              zh: "当你自己的集群正在运行时使用。请求由你自己的硬件完成，不消耗火花。",
            },
            lang,
          )}
        </p>
        <Row label={L({ en: "Base URL", zh: "服务地址" }, lang)} value={localBase} lang={lang} />
        {localKey ? (
          <Row label={L({ en: "Token", zh: "访问令牌" }, lang)} value={localKey} lang={lang} />
        ) : (
          // An empty token is a real state (upgraded installs keep it), and it
          // is exactly the state that becomes dangerous once this machine can
          // spend Sparks — so say so here rather than only in the field's hint.
          <p className="setting-hint setting-hint--warn">
            {L(
              {
                en: "No token set — any device on your LAN can call this API. Set one under “Access token” below.",
                zh: "未设置访问令牌，局域网内的设备均可调用此 API；可在下方「访问令牌」中设置。",
              },
              lang,
            )}
          </p>
        )}
        <pre className="endpoint-snippet">{localExport}</pre>
        <CopyBtn text={localExport} lang={lang} />
        {liveBase ? null : (
          <p className="setting-hint">
            {L(
              {
                en: "The cluster is not serving yet — these are configured values, not confirmed ones.",
                zh: "集群尚未开始服务，以上为配置值，不代表当前可用。",
              },
              lang,
            )}
          </p>
        )}
      </div>

      <div className="endpoint-card">
        <div className="endpoint-card__title">{L({ en: "IdleToken platform", zh: "IdleToken 平台" }, lang)}</div>
        <p className="setting-hint">
          {L(
            {
              en: "For devices without a cluster of their own. Requests run on shared capacity and are billed in Sparks; create the key under “Sharing & earnings”.",
              zh: "用于没有自建集群的设备。请求由平台共享算力完成，按火花计费；密钥在「共享与收益」中创建。",
            },
            lang,
          )}
        </p>
        <Row label={L({ en: "Base URL", zh: "服务地址" }, lang)} value={cloudBase} lang={lang} />
        <pre className="endpoint-snippet">{cloudExport}</pre>
        <CopyBtn text={cloudExport} lang={lang} />
      </div>

      <p className="setting-hint">
        {L(
          {
            en: "Both expose the same OpenAI- and Anthropic-compatible API; only the address and key differ.",
            zh: "两个入口提供相同的 OpenAI / Anthropic 兼容 API，仅地址与密钥不同。",
          },
          lang,
        )}
      </p>
    </div>
  );
}
