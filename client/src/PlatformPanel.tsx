// Platform panel — "share compute / marketplace console" (integration plan
// 3.1 + 3.2). Everything shown here is a real platform call with the cloud
// session's JWT; failures render as-is, nothing is faked. Agent start/stop is
// local RPC to the Rust supervisor (platform_agent_start/stop) and is disabled
// in a plain browser, where no sidecar can exist (DEV-FIXTURE philosophy).
import { useCallback, useEffect, useState } from "react";
import { fmtCredits as fmtCreditsBase } from "@idletoken/shared-ui";
import { useI18n } from "./i18n";
import type { AppSettings } from "./settings";
import type { Session } from "./auth";
import { getPairingProvider } from "./pairing";
import {
  agentLogs,
  agentStart,
  agentStatus,
  agentStop,
  createApiKey,
  createRendezvousToken,
  getLedger,
  getMe,
  getProviders,
  inTauri,
  listApiKeys,
  onAgentLog,
  onAgentStatus,
  platformGate,
  revokeApiKey,
  setProviderListing,
  type AgentLogLine,
  type AgentStatus,
  type ApiKeyInfo,
  type CreatedApiKey,
  type LedgerEntry,
  type PlatformMe,
  type ProviderInfo,
  type RendezvousToken,
} from "./platform";

const LOG_TAIL = 6;

/** milli-credits -> signed credit string, e.g. +100 / -2.35 (shared implementation, see @idletoken/shared-ui) */
const fmtCredits = (milli: number): string => fmtCreditsBase(milli, { signed: true });

/** One remote dataset: either data, or the real error — never placeholder rows. */
interface Remote<T> {
  data: T | null;
  err: string | null;
}
const EMPTY: Remote<never> = { data: null, err: null };

// Bare content — the host draws the frame. It had a full-page wrapper and a
// modal wrapper while the marketplace was its own place; both went unreachable
// on 2026-08-10 when this became Settings → Sharing & earnings, and an
// unreachable branch is a branch nobody notices rotting.
export default function PlatformPanel(props: {
  settings: AppSettings;
  session: Session | null;
  onOpenSettings: () => void;
  onSignIn: () => void;
}) {
  const gate = platformGate();
  return gate.ok ? (
    <Console settings={props.settings} jwt={gate.session.token} url={gate.url} />
  ) : (
    <Guide reason={gate.reason} onOpenSettings={props.onOpenSettings} onSignIn={props.onSignIn} />
  );
}

// ---- not connected: honest guidance, no data ------------------------------
function Guide(props: {
  reason: "no-url" | "no-session" | "local-session";
  onOpenSettings: () => void;
  onSignIn: () => void;
}) {
  const { t } = useI18n();
  const key =
    props.reason === "no-url"
      ? "platform.guide.noUrl"
      : props.reason === "no-session"
        ? "platform.guide.needSignIn"
        : "platform.guide.needCloud";
  // Not btn-block: this renders in the wide settings column now, where a
  // full-bleed button stretches to ~1100px and stops reading as a button.
  return (
    <div className="plat-guide">
      <p className="auth-note">{t(key)}</p>
      {props.reason === "no-url" ? (
        <button className="btn-primary" onClick={props.onOpenSettings}>
          {t("platform.goSettings")}
        </button>
      ) : (
        <button className="btn-primary" onClick={props.onSignIn}>
          {t("platform.goSignIn")}
        </button>
      )}
    </div>
  );
}

// ---- connected console -----------------------------------------------------
function Console(props: { settings: AppSettings; jwt: string; url: string }) {
  const { t, lang } = useI18n();
  const [me, setMe] = useState<Remote<PlatformMe>>(EMPTY);
  const [ledger, setLedger] = useState<Remote<LedgerEntry[]>>(EMPTY);
  const [providers, setProviders] = useState<Remote<ProviderInfo[]>>(EMPTY);
  const [keys, setKeys] = useState<Remote<ApiKeyInfo[]>>(EMPTY);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    const grab = <T,>(p: Promise<T>, set: (r: Remote<T>) => void) =>
      p.then((data) => set({ data, err: null })).catch((e) => set({ data: null, err: String((e as Error)?.message ?? e) }));
    Promise.allSettled([
      grab(getMe(), setMe),
      grab(getLedger(), setLedger),
      grab(getProviders(), setProviders),
      grab(listApiKeys(), setKeys),
    ]).then(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Relative time for heartbeats/ledger rows, localized via the string table.
  const rel = (iso: string | null): string => {
    if (!iso) return t("time.never");
    const ms = Date.now() - new Date(iso).getTime();
    const min = Math.floor(ms / 60_000);
    if (min < 1) return t("time.now");
    if (min < 60) return t("time.min", { n: min });
    const h = Math.floor(min / 60);
    if (h < 24) return t("time.hour", { n: h });
    return t("time.day", { n: Math.floor(h / 24) });
  };

  return (
    <>
      {/* balance */}
      <div className="setting-group__label plat-label">
        {t("platform.balance")}
        <button className="linkbtn" onClick={refresh} disabled={loading}>
          {loading ? t("platform.loading") : t("platform.refresh")}
        </button>
      </div>
      {me.err ? <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: me.err })}</div> : null}
      {me.data ? (
        <div className="plat-balance">
          <span className="plat-balance__v">
            {me.data.balanceCredits.toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits: 2 })}
          </span>
          <span className="plat-balance__u">{t("platform.balance.unit")}</span>
          <span className="plat-balance__email">{me.data.email}</span>
        </div>
      ) : null}

      <ShareCard settings={props.settings} jwt={props.jwt} url={props.url} providers={providers} rel={rel} onChanged={refresh} />

      {/* ledger */}
      <div className="setting-group__label plat-label">{t("platform.ledger")}</div>
      {ledger.err ? <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: ledger.err })}</div> : null}
      {ledger.data ? (
        ledger.data.length === 0 ? (
          <p className="auth-note">{t("platform.ledger.empty")}</p>
        ) : (
          <div className="plat-ledger">
            {ledger.data.slice(0, 8).map((l) => (
              <div key={l.id} className="plat-ledger__row">
                <span className="plat-ledger__type">{l.type}</span>
                <span className="plat-ledger__reason">{l.reason || l.ref || "—"}</span>
                <span className="plat-ledger__when">{rel(l.createdAt)}</span>
                <span className={`plat-ledger__delta${l.deltaMilli < 0 ? " is-neg" : ""}`}>{fmtCredits(l.deltaMilli)}</span>
              </div>
            ))}
          </div>
        )
      ) : null}

      <ApiKeysCard keys={keys} rel={rel} onChanged={refresh} />

      <RendezvousTokenCard />
    </>
  );
}

// ---- rendezvous pairing token (engine account-mode networking) -------------
// The engine's rendezvous HTTP client has no TLS, so this fetches a restricted
// scope=rendezvous token over HTTPS for the CLI and engine to use as
// --account-token: leaking it exposes only the ability to exchange addresses.
function RendezvousTokenCard() {
  const { t } = useI18n();
  const [issued, setIssued] = useState<RendezvousToken | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const issue = async () => {
    setBusy(true);
    setErr(null);
    try {
      setIssued(await createRendezvousToken());
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
    setBusy(false);
  };

  const copy = async () => {
    if (!issued) return;
    try {
      await navigator.clipboard.writeText(issued.token);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked; the token is shown regardless */
    }
  };

  return (
    <>
      <div className="setting-group__label plat-label">
        {t("platform.rdv")}
        <button className="linkbtn" onClick={issue} disabled={busy}>
          {t("platform.rdv.new")}
        </button>
      </div>
      <p className="setting-row__hint">{t("platform.rdv.hint")}</p>
      {err ? <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: err })}</div> : null}
      {issued ? (
        <div className="plat-newkey">
          <div className="plat-newkey__row">
            <code className="api-card__url">{issued.token}</code>
            <button className="iconbtn" onClick={copy}>
              {copied ? t("pairing.copied") : t("pairing.copy")}
            </button>
          </div>
          <p className="plat-newkey__warn">{t("platform.rdv.warn", { days: Math.round(issued.expiresInSec / 86400) })}</p>
        </div>
      ) : null}
    </>
  );
}

// ---- share compute (platform agent lifecycle) ------------------------------
function ShareCard(props: {
  settings: AppSettings;
  jwt: string;
  url: string;
  providers: Remote<ProviderInfo[]>;
  rel: (iso: string | null) => string;
  onChanged: () => void;
}) {
  const { t } = useI18n();
  const tauri = inTauri();
  const [st, setSt] = useState<AgentStatus | null>(null);
  const [lines, setLines] = useState<AgentLogLine[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [clusterReady, setClusterReady] = useState(false);
  const [listBusy, setListBusy] = useState<string | null>(null);

  // List or unlist on the marketplace: the last inch of the sharing loop.
  // Registration defaults to listed=false, so without this switch a provider
  // never receives anyone's traffic and never earns credits.
  const toggleListing = async (p: ProviderInfo) => {
    setListBusy(p.id);
    setErr(null);
    try {
      await setProviderListing(p.id, !p.listed);
      props.onChanged();
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally {
      setListBusy(null);
    }
  };

  useEffect(() => {
    if (!tauri) return;
    let live = true;
    agentStatus().then((s) => live && setSt(s)).catch(() => {});
    agentLogs(LOG_TAIL).then((ls) => live && setLines(ls)).catch(() => {});
    const unStatus = onAgentStatus((s) => setSt(s));
    const unLog = onAgentLog((l) => setLines((prev) => [...prev.slice(-(LOG_TAIL - 1)), l]));
    return () => {
      live = false;
      unStatus();
      unLog();
    };
  }, [tauri]);

  // Honest readiness hint: the local cluster state comes from the same pairing
  // subscription the topbar pill uses. Not ready ≠ blocked — the agent waits.
  useEffect(() => {
    return getPairingProvider().subscribe((s) => setClusterReady(s.phase === "ready"));
  }, []);

  const active = st !== null && st.state !== "stopped" && st.state !== "crashed";
  const toggle = async () => {
    if (busy || !tauri) return;
    setBusy(true);
    setErr(null);
    try {
      if (active) {
        await agentStop();
      } else {
        await agentStart({
          platformUrl: props.url,
          jwt: props.jwt,
          name: props.settings.clusterName || "home",
          coordApiPort: props.settings.apiPort || 8000,
        });
      }
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
    setBusy(false);
  };

  const state = st?.state ?? "stopped";
  return (
    <>
      <div className="setting-group__label plat-label">{t("platform.share")}</div>
      <div className="plat-share">
        <div className="engine-card__head">
          <span className={`engine-state engine-state--${state}`}>
            <span className="engine-state__dot" />
            {t(`engine.state.${state}` as const)}
          </span>
          <button className="btn-secondary engine-card__btn" disabled={busy || !tauri} onClick={toggle}>
            {active ? t("platform.share.stop") : t("platform.share.start")}
          </button>
        </div>
        <p className="setting-row__hint">{t("platform.share.explain")}</p>
        {!tauri ? <p className="engine-meta">{t("platform.share.browser")}</p> : null}
        {tauri && !active && !clusterReady ? (
          <p className="engine-meta">{t("platform.share.notReady", { port: props.settings.apiPort || 8000 })}</p>
        ) : null}
        {st?.state === "crashed" ? (
          <p className="engine-meta engine-meta--bad">{t("engine.crashedHint", { code: st.lastExitCode ?? "?" })}</p>
        ) : null}
        {err ? <p className="engine-meta engine-meta--bad">{err}</p> : null}
        {lines.length > 0 ? (
          <pre className="engine-log" aria-label={t("platform.agent.log")}>
            {lines.map((l) => l.line).join("\n")}
          </pre>
        ) : null}
      </div>

      {/* my providers, as the platform sees them */}
      <div className="setting-group__label plat-label">{t("platform.providers")}</div>
      {props.providers.err ? (
        <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: props.providers.err })}</div>
      ) : null}
      {props.providers.data ? (
        props.providers.data.length === 0 ? (
          <p className="auth-note">{t("platform.providers.empty")}</p>
        ) : (
          <div className="peer-list">
            {props.providers.data.map((p) => (
              <div key={p.id} className="peer">
                <span className={`peer__dot${p.status === "ONLINE" ? " plat-dot--on" : ""}`} />
                <div className="peer__id">
                  <span className="peer__host">{p.name}</span>
                  <span className="peer__gpu">{t("platform.provider.lastBeat", { t: props.rel(p.lastBeat) })}</span>
                </div>
                <span className={`peer__stage${p.status === "ONLINE" ? " peer__stage--ready" : ""}`}>
                  {p.status === "ONLINE"
                    ? t("platform.provider.online")
                    : p.status === "SUSPENDED"
                      ? t("platform.provider.suspended")
                      : t("platform.provider.offline")}
                </span>
                <button
                  className={`sw${p.listed ? " is-on" : ""}`}
                  disabled={listBusy === p.id || p.status === "SUSPENDED"}
                  onClick={() => toggleListing(p)}
                  role="switch"
                  aria-checked={p.listed}
                  title={p.listed ? t("platform.listing.on") : t("platform.listing.off")}
                >
                  <span className="sw__k" />
                </button>
              </div>
            ))}
            <p className="auth-note plat-listing-hint">
              {t("platform.listing.hint")}
            </p>
          </div>
        )
      ) : null}
    </>
  );
}

// ---- API keys (consumer side, minimal) ------------------------------------
function ApiKeysCard(props: { keys: Remote<ApiKeyInfo[]>; rel: (iso: string | null) => string; onChanged: () => void }) {
  const { t } = useI18n();
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const make = async () => {
    setBusy(true);
    setErr(null);
    try {
      setCreated(await createApiKey());
      props.onChanged();
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
    setBusy(false);
  };

  const revoke = async (id: string) => {
    setErr(null);
    try {
      await revokeApiKey(id);
      props.onChanged();
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
  };

  const copy = async () => {
    if (!created) return;
    try {
      await navigator.clipboard.writeText(created.apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked; the key is shown regardless */
    }
  };

  return (
    <>
      <div className="setting-group__label plat-label">
        {t("platform.keys")}
        <button className="linkbtn" onClick={make} disabled={busy}>
          {t("platform.keys.new")}
        </button>
      </div>
      <p className="setting-row__hint">{t("platform.keys.hint")}</p>
      {err ? <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: err })}</div> : null}
      {created ? (
        <div className="plat-newkey">
          <div className="plat-newkey__row">
            <code className="api-card__url">{created.apiKey}</code>
            <button className="iconbtn" onClick={copy}>
              {copied ? t("pairing.copied") : t("pairing.copy")}
            </button>
          </div>
          <p className="plat-newkey__warn">{t("platform.keys.plainWarn")}</p>
        </div>
      ) : null}
      {props.keys.err ? <div className="engine-meta engine-meta--bad">{t("platform.err", { msg: props.keys.err })}</div> : null}
      {props.keys.data ? (
        props.keys.data.length === 0 && !created ? (
          <p className="auth-note">{t("platform.keys.empty")}</p>
        ) : (
          <div className="plat-ledger">
            {props.keys.data.map((k) => (
              <div key={k.id} className="plat-ledger__row">
                <code className="plat-key__prefix">{k.prefix}…</code>
                <span className="plat-ledger__when">{props.rel(k.createdAt)}</span>
                {k.revoked ? (
                  <span className="plat-key__revoked">{t("platform.keys.revoked")}</span>
                ) : (
                  <button className="linkbtn" onClick={() => revoke(k.id)}>
                    {t("platform.keys.revoke")}
                  </button>
                )}
              </div>
            ))}
          </div>
        )
      ) : null}
    </>
  );
}
