// "Connect a client here" — the address a third-party client points at.
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
// 2026-08-21: cut to the address and nothing else (user's call). The API key
// row and the three explanatory paragraphs (Authorization header, key-less
// installs, the NO_PROXY warning) are gone.
// ⚠ The key itself is NOT gone — a fresh install still mints `apiToken`, hands
// it to the coordinator, and the coordinator still answers 401 without it. It
// now lives only in the settings file. If "Claude Code gets 401 and there is no
// way to find out why" comes back as a report, that is this change, and the fix
// is to stop minting the token for a loopback-only API — not to re-add prose.
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

export function EndpointsPanel(props: { settings: AppSettings }) {
  const { lang } = useI18n();
  const s = props.settings;
  const localBase = `http://127.0.0.1:${s.apiPort || 8000}`;

  return (
    <div className="setting-group endpoints">
      <div className="endpoint-card">
        <div className="endpoint-row">
          <span className="endpoint-row__label">{L({ en: "Base URL", zh: "服务地址" }, lang)}</span>
          <code className="endpoint-row__value">{localBase}</code>
          <CopyBtn text={localBase} lang={lang} />
        </div>
      </div>
    </div>
  );
}
