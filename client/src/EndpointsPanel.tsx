// "Connect a client here" — the block that tells the user their own base URL
// and the key that opens it (docs/api-surface.md §7).
//
// It exists because the client never said it. You could start a cluster,
// watch it go green, and still have to go find the README to point a client
// at the thing you had just started.
//
// 2026-08-15: the API serves its own machine only (127.0.0.1, enforced by the
// coordinator), so this block no longer shows a LAN address or warns about the
// LAN at all. The address is derived from the SETTINGS, not from the running
// cluster's report: it used to prefer the live base URL, which meant editing
// the port field left the shown address on the old port until the next
// restart — the card looked broken exactly while the user was configuring it.
// A restart applies the port anyway, so the configured value is the one worth
// copying.
//
// 2026-08-20 (audit A-P1-1): the API key is back on screen. Removing the token
// FIELD was right — nobody should have to invent one — but removing the token
// itself was never on the table: a fresh install mints one, hands it to the
// coordinator, and the coordinator answers 401 without it. With the row gone,
// Claude Code (the core scenario) got 401 and the client offered no way to find
// out why. The gate stays; the key is simply shown to the person who owns the
// machine.
import { useState } from "react";
import { useI18n, type Lang } from "./i18n";
import type { AppSettings } from "./settings";

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

/** The key, masked until asked for. Shoulder-surfing is the only threat left
 *  once the API is loopback-only, and it is a real one on a shared desk. */
function KeyRow(props: { token: string; lang: Lang }) {
  const [shown, setShown] = useState(false);
  const label = L({ en: "API key", zh: "API 密钥" }, props.lang);
  return (
    <div className="endpoint-row">
      <span className="endpoint-row__label">{label}</span>
      <code className="endpoint-row__value">{shown ? props.token : "•".repeat(24)}</code>
      <button className="linkbtn" onClick={() => setShown((v) => !v)}>
        {shown ? L({ en: "Hide", zh: "隐藏" }, props.lang) : L({ en: "Show", zh: "显示" }, props.lang)}
      </button>
      <CopyBtn text={props.token} lang={props.lang} />
    </div>
  );
}

export function EndpointsPanel(props: { settings: AppSettings }) {
  const { lang } = useI18n();
  const s = props.settings;
  const localBase = `http://127.0.0.1:${s.apiPort || 8000}`;
  const token = s.apiToken.trim();

  return (
    <div className="setting-group endpoints">
      <div className="setting-group__label">{L({ en: "Connect a client", zh: "接入客户端" }, lang)}</div>

      <div className="endpoint-card">
        <div className="endpoint-card__title">{L({ en: "This machine", zh: "本机" }, lang)}</div>
        <Row label={L({ en: "Base URL", zh: "服务地址" }, lang)} value={localBase} lang={lang} />
        {token ? (
          <>
            <KeyRow token={token} lang={lang} />
            <p className="setting-hint">
              {L(
                {
                  en: "The API answers this machine only. Send the key as the Authorization header:",
                  zh: "该 API 只应答本机。请求需带上 Authorization 头：",
                },
                lang
              )}
            </p>
            <Row
              label={L({ en: "Header", zh: "请求头" }, lang)}
              value={`Authorization: Bearer ${token}`}
              lang={lang}
            />
          </>
        ) : (
          // An install that predates the minted key (or one whose settings were
          // imported without it) really does run an open loopback API. Say so
          // rather than print an empty key row.
          <p className="setting-hint">
            {L(
              {
                en: "This machine's API needs no key. It answers this machine only.",
                zh: "本机 API 无需密钥。它只应答本机。",
              },
              lang
            )}
          </p>
        )}
        {/* A-P2-4: a local HTTP proxy is the one thing that reliably breaks
            this. Clash and friends capture 127.0.0.1 too, and they swallow the
            end-of-stream marker on a streaming reply, so the third-party client
            hangs forever with no error to show. Two lines here cost less than
            the support thread. */}
        <p className="setting-hint">
          {L(
            {
              en: "If a proxy runs on this machine (Clash and similar), exclude the loopback address or streaming replies never finish: set NO_PROXY=127.0.0.1,localhost for the client you are connecting.",
              zh: "本机若运行代理（Clash 等），需把回环地址排除在外，否则流式回复不会结束：为接入的客户端设置 NO_PROXY=127.0.0.1,localhost。",
            },
            lang
          )}
        </p>
      </div>

      {/* No "IdleToken platform" card here (2026-08-15): the platform side is
          not live for users yet, and a settings page must not display things
          that do nothing. It returns together with the platform features. */}
    </div>
  );
}
