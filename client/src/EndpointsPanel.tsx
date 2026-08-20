// "Connect a client here" — the block that tells the user their own base URL
// (docs/api-surface.md §7).
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

export function EndpointsPanel(props: { settings: AppSettings }) {
  const { lang } = useI18n();
  const s = props.settings;
  const localBase = `http://127.0.0.1:${s.apiPort || 8000}`;

  return (
    <div className="setting-group endpoints">
      <div className="setting-group__label">{L({ en: "Connect a client", zh: "接入客户端" }, lang)}</div>

      <div className="endpoint-card">
        <div className="endpoint-card__title">{L({ en: "This machine", zh: "本机" }, lang)}</div>
        {/* One fact: the address. No token row (the loopback-only API gates
            nothing a local caller could not already do), no client snippet —
            configuring third-party tools is their docs' job. */}
        <Row label={L({ en: "Base URL", zh: "服务地址" }, lang)} value={localBase} lang={lang} />
      </div>

      {/* No "IdleToken platform" card here (2026-08-15): the platform side is
          not live for users yet, and a settings page must not display things
          that do nothing. It returns together with the platform features. */}
    </div>
  );
}
