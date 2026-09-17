import { useEffect, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { useDialog } from "./useDialog";
import { loadSettings, saveSettings } from "./settings";
import { RELEASES_URL } from "./links";
import { currentVersionStanding } from "./release";
import {
  agentLogs,
  agentStart,
  agentStatus,
  agentStop,
  createApiKey,
  getProviders,
  inTauri,
  onAgentLog,
  onAgentStatus,
  platformGate,
  setLocalOverflow,
  setProviderListed,
} from "./platform";

import {
  agentProviderId, agentRefusal, sharingObservation, waitForSharing,
  type AgentRegistration, type SharingObservation, type SharingPhase,
} from "./sharingState";

const MARKETPLACE_CHANGED = "idletoken:marketplace-settings-changed";

function publishMarketplaceChange(): void {
  window.dispatchEvent(new CustomEvent(MARKETPLACE_CHANGED));
}

/**
 * This build is too old to be listed on the marketplace.
 *
 * Its own class rather than a message string because it is the one failure here
 * that has a REMEDY the user can act on, so it gets a dialog with a link rather
 * than a tooltip. Everything else the toggle can fail with — no session, agent
 * would not start, coordinator not ready — is either transient or already
 * explained elsewhere.
 */
class VersionTooOldToShare extends Error {
  constructor(readonly installed: string, readonly floor: string) {
    super(`client ${installed} is below the marketplace floor ${floor}`);
    this.name = "VersionTooOldToShare";
  }
}

/**
 * Shown when someone turns sharing on from a build below the listing floor.
 *
 * The copy has one job beyond "download this": say that nothing else is
 * affected. A person who reads "your version is too old" on a machine that has
 * been happily running a 200 GB model all week will otherwise conclude the
 * whole product just stopped working, and that conclusion is wrong — the floor
 * is about serving strangers and nothing else.
 */
function ShareVersionBlockedDialog(
  { installed, floor, onClose }: { installed: string; floor: string; onClose: () => void },
) {
  const { t } = useI18n();
  const ref = useDialog(onClose);
  const open = async () => {
    if (inTauri()) {
      const { open: openUrl } = await import("@tauri-apps/plugin-shell");
      await openUrl(RELEASES_URL);
    } else {
      window.open(RELEASES_URL, "_blank", "noopener,noreferrer");
    }
  };
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div ref={ref} className="modal modal--share-version" role="dialog" aria-modal="true"
           onClick={(e) => e.stopPropagation()}>
        <div className="modal__head">
          <h2>{t("update.share.title")}</h2>
          <button className="iconbtn" onClick={onClose} aria-label={t("a11y.close")}>✕</button>
        </div>
        {/* The two versions side by side. A sentence with numbers buried in it
            makes the reader parse; two labelled values make the gap the first
            thing they see, which is the whole content of this dialog. */}
        <div className="share-version__versions">
          <div className="share-version__cell">
            <span className="share-version__label">{t("update.share.yours")}</span>
            <span className="share-version__value">{installed}</span>
          </div>
          <span className="share-version__arrow" aria-hidden="true">→</span>
          <div className="share-version__cell share-version__cell--need">
            <span className="share-version__label">{t("update.share.needed")}</span>
            <span className="share-version__value">{floor}</span>
          </div>
        </div>
        <p className="share-version__lead">{t("update.share.body")}</p>
        <p className="share-version__note">{t("update.share.unaffected")}</p>
        <div className="modal__foot">
          <button className="btn-secondary" onClick={onClose}>{t("update.share.notNow")}</button>
          <button className="btn-primary" onClick={() => void open()}>
            {t("update.openDownloads")}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Provider-side control: off -> opening -> confirmed sharing. */
export function ShareToggleButton({ serviceReady, onNeedLogin }: {
  serviceReady: boolean;
  onNeedLogin?: () => void;
}) {
  const { t, tErr } = useI18n();
  const [on, setOn] = useState(() => loadSettings().providerEnabled);
  const [phase, setPhase] = useState<SharingPhase>("off");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [blocked, setBlocked] = useState<VersionTooOldToShare | null>(null);
  const busyRef = useRef(false);
  const resumed = useRef(false);
  const confirmed = useRef(false);
  const registration = useRef<AgentRegistration | null>(null);
  const observation = useRef<SharingObservation | null>(null);
  const operation = useRef<AbortController | null>(null);
  const monitor = useRef<AbortController | null>(null);
  const ready = useRef(serviceReady);
  ready.current = serviceReady;
  const gate = platformGate();

  useEffect(() => () => {
    operation.current?.abort();
    monitor.current?.abort();
  }, []);

  const read = async (signal: AbortSignal): Promise<SharingObservation> => {
    signal.throwIfAborted();
    const [logs, providers] = await Promise.all([agentLogs(500), getProviders()]);
    // Read liveness AFTER the network round trip: it may have exited meanwhile.
    const agent = await agentStatus();
    signal.throwIfAborted();
    const next = sharingObservation(agent, logs, providers, registration.current, ready.current);
    registration.current = next.registration;
    observation.current = next;
    return next;
  };

  const stop = async () => {
    const provider = observation.current?.provider;
    try {
      if (provider?.listed) await setProviderListed(provider.id, false);
    } catch { /* An unreachable gateway still ages out the stopped heartbeat. */ }
    await agentStop();
    confirmed.current = false;
    registration.current = null;
    observation.current = null;
    saveSettings({ ...loadSettings(), providerEnabled: false });
    setOn(false);
    setPhase("off");
    publishMarketplaceChange();
  };

  const start = async (restore = false) => {
    if (busyRef.current || !ready.current || !gate.ok) return;
    busyRef.current = true;
    setBusy(true);
    monitor.current?.abort();
    const attempt = new AbortController();
    operation.current = attempt;
    setErr(null);
    setBlocked(null);
    setPhase("opening");
    try {
      // Manual activation and startup restoration use the SAME admission path.
      const standing = await currentVersionStanding(true);
      attempt.signal.throwIfAborted();
      if (standing.belowShareFloor) {
        throw new VersionTooOldToShare(standing.installed, standing.shareFloor!);
      }
      let providerName = loadSettings().providerName;
      if (!/^cluster-\d+$/.test(providerName)) {
        const taken = new Set((await getProviders()).map((p) => p.name));
        let n = 1;
        while (taken.has(`cluster-${n}`)) n++;
        providerName = `cluster-${n}`;
      }
      attempt.signal.throwIfAborted();
      const status = await agentStatus();
      if (!restore || !["running", "starting", "restarting"].includes(status.state)) {
        registration.current = null;
        observation.current = null;
        const s = loadSettings();
        await agentStart({
          platformUrl: gate.url, jwt: gate.session.token, name: providerName,
          coordApiPort: s.apiPort || 8000, coordToken: s.apiToken,
          modelId: s.modelId, quant: s.quant,
        });
      }
      const result = await waitForSharing(() => read(attempt.signal), { signal: attempt.signal });
      attempt.signal.throwIfAborted();
      confirmed.current = true;
      saveSettings({ ...loadSettings(), providerEnabled: true, providerName: result.provider!.name });
      setOn(true);
      setPhase("sharing");
      publishMarketplaceChange();
    } catch (e) {
      // A failed attempt cannot keep registering behind an "off" button.
      await stop();
      if (!attempt.signal.aborted) {
        if (e instanceof VersionTooOldToShare) setBlocked(e);
        else setErr(tErr(String((e as Error)?.message ?? e)));
      }
    } finally {
      if (operation.current === attempt) operation.current = null;
      busyRef.current = false;
      setBusy(false);
    }
  };

  // Check at STARTUP, including a sign-in that happens after the app mounted.
  // Waiting for the local service avoids racing App's old fire-and-forget resume.
  const sessionToken = gate.ok ? gate.session.token : "";
  useEffect(() => {
    if (!inTauri() || !serviceReady || !sessionToken || resumed.current) return;
    resumed.current = true;
    if (loadSettings().providerEnabled) void start(true);
  }, [serviceReady, sessionToken]);

  useEffect(() => {
    if (!inTauri() || !on || busy || !confirmed.current) return;
    const watch = new AbortController();
    monitor.current = watch;
    let checking = false;
    const check = async () => {
      if (checking || busyRef.current || watch.signal.aborted) return;
      checking = true;
      try {
        await waitForSharing(() => read(watch.signal), {
          signal: watch.signal,
          pending: () => { if (!watch.signal.aborted) setPhase("opening"); },
        });
        if (!watch.signal.aborted) setPhase("sharing");
      } catch (e) {
        if (!watch.signal.aborted) {
          // Retrying from the off label must START, never toggle a stale wish off.
          busyRef.current = true;
          setBusy(true);
          try {
            await stop();
            setErr(tErr(String((e as Error)?.message ?? e)));
          } finally { busyRef.current = false; setBusy(false); }
        }
      } finally { checking = false; }
    };
    const timer = setInterval(() => void check(), 5_000);
    const offStatus = onAgentStatus(() => void check());
    const offLog = onAgentLog((log) => {
      if (agentProviderId(log.line) || agentRefusal(log)) void check();
    });
    void check();
    return () => {
      watch.abort();
      clearInterval(timer);
      offStatus();
      offLog();
    };
  }, [on, busy]);

  if (!inTauri()) return null;
  const opening = phase === "opening";
  const active = phase === "sharing";
  const title = err ?? (!serviceReady ? t("share.needService")
    : !gate.ok ? t(gate.reason === "no-url" ? "platform.err.noUrl" : "share.needLogin")
    : t(opening ? "share.opening" : active ? "share.on" : "share.off"));
  const toggle = async () => {
    if (busyRef.current || opening) return;
    if (!gate.ok) { if (serviceReady) onNeedLogin?.(); return; }
    if (!active) { await start(); return; }
    busyRef.current = true;
    setBusy(true);
    monitor.current?.abort();
    try { await stop(); }
    catch (e) { setErr(tErr(String((e as Error)?.message ?? e))); }
    finally { busyRef.current = false; setBusy(false); }
  };
  return (
    <>
      <button
        className={`pill pill--market pill--share pill--${active ? "ready" : "standalone"}`}
        data-share-state={phase}
        disabled={busy || opening || (!gate.ok && gate.reason === "no-url")}
        onClick={() => void toggle()}
        aria-pressed={active}
        aria-busy={busy || opening}
        aria-label={t(opening ? "share.opening" : active ? "share.on" : "share.off")}
        title={title}
      >
        <span className="pill__dot" />
        <span className="pill__label">{opening ? "…" : t(active ? "share.on" : "share.off")}</span>
      </button>
      {blocked ? <ShareVersionBlockedDialog installed={blocked.installed} floor={blocked.floor}
        onClose={() => setBlocked(null)} /> : null}
      {err ? <ShareFailureDialog message={err} onClose={() => setErr(null)} /> : null}
    </>
  );
}

function ShareFailureDialog({ message, onClose }: { message: string; onClose: () => void }) {
  const { t } = useI18n();
  const ref = useDialog(onClose);
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div ref={ref} className="modal" role="alertdialog" aria-modal="true"
        aria-labelledby="share-failure-title" onClick={(e) => e.stopPropagation()}>
        <div className="modal__head"><h2 id="share-failure-title">{t("share.failed")}</h2></div>
        <p>{message}</p>
        <div className="modal__foot"><button className="btn-primary" onClick={onClose}>
          {t("a11y.close")}
        </button></div>
      </div>
    </div>
  );
}

/** Consumer-side control: route an occupied local slot to another provider. */
export function OverflowToggleButton({
  serviceReady,
  onNeedLogin,
}: {
  serviceReady: boolean;
  onNeedLogin?: () => void;
}) {
  const { t } = useI18n();
  const [on, setOn] = useState(() => loadSettings().overflowEnabled);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const sync = () => setOn(loadSettings().overflowEnabled);
    window.addEventListener(MARKETPLACE_CHANGED, sync);
    return () => window.removeEventListener(MARKETPLACE_CHANGED, sync);
  }, []);

  if (!inTauri()) return null;
  const gate = platformGate();

  const toggle = async () => {
    if (busy || !serviceReady) return;
    // Turning paid routing OFF is always local and must remain possible after
    // sign-out or during a platform outage. Only enabling needs a session.
    if (on) {
      setBusy(true);
      setErr(null);
      try {
        saveSettings({ ...loadSettings(), overflowEnabled: false });
        setOn(false);
        // Tell the coordinator now. OFF is the direction that spends money, so
        // it must not wait for the next model start — which is what it used to
        // do, silently, while the switch read as off.
        await setLocalOverflow(false).catch(() => { /* not running yet: the preference covers it */ });
        publishMarketplaceChange();
      } catch (e) {
        setErr(String((e as Error)?.message ?? e));
      } finally {
        setBusy(false);
      }
      return;
    }
    if (!gate.ok) {
      if (gate.reason !== "no-url") onNeedLogin?.();
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const s = loadSettings();
      const key = s.overflowKey
        || (await createApiKey({
          label: "request help when local slot is busy",
          // Account balance is the spend gate. Automatic overflow must not
          // silently add a second per-key daily ceiling.
          dailyCapMilli: null,
        })).apiKey;
      saveSettings({ ...loadSettings(), overflowEnabled: true, overflowKey: key, overflowWaitS: 0 });
      setOn(true);
      // A coordinator started while this was off already holds the key and is
      // simply not using it, so switching on lands on the next busy request.
      // One that predates the key (or is not running) starts from the stored
      // preference instead — hence best-effort rather than a failure.
      await setLocalOverflow(true).catch(() => { /* older coordinator: next start covers it */ });
      publishMarketplaceChange();
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
      saveSettings({ ...loadSettings(), overflowEnabled: false });
      setOn(false);
      publishMarketplaceChange();
    } finally {
      setBusy(false);
    }
  };

  if (!gate.ok && !on) {
    const noUrl = gate.reason === "no-url";
    return (
      <button
        className="pill pill--market pill--standalone"
        disabled={noUrl}
        onClick={() => { if (!noUrl && serviceReady) onNeedLogin?.(); }}
        title={
          !serviceReady
            ? t("platform.overflow.needService")
            : noUrl
              ? t("platform.err.noUrl")
              : t("platform.overflow.needLogin")
        }
      >
        <span className="pill__dot" /><span className="pill__label">{t("platform.overflow.name")}</span>
      </button>
    );
  }

  // Green means THIS CAN ROUTE RIGHT NOW, not "the preference is on"
  // (2026-09-16, same rule as the sharing pill).
  //
  // An enabled overflow preference cannot route anything until the local
  // service is ready to receive the request in the first place — and it also
  // cannot route without a live session or the key it pays with. Signing out,
  // or a settings file that lost the key, used to leave this green while every
  // borrowed request would have failed.
  const routable = gate.ok && !!loadSettings().overflowKey;
  const active = on && serviceReady && routable;

  return (
    <button
      className={`pill pill--market pill--${active ? "ready" : "standalone"}`}
      disabled={busy}
      onClick={() => void toggle()}
      aria-pressed={active}
      title={
        !serviceReady
          ? t("platform.overflow.needService")
          : on && !routable
            ? t("platform.overflow.notReady")
            : err ?? t("platform.overflow.hint")
      }
    >
      <span className="pill__dot" />
      <span className="pill__label">
        {busy ? "…" : t(active ? "platform.overflow.on" : "platform.overflow.name")}
      </span>
    </button>
  );
}
