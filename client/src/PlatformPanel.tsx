import { useEffect, useState } from "react";
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
  type AgentLogLine,
  type AgentStatus,
} from "./platform";

const MARKETPLACE_CHANGED = "idletoken:marketplace-settings-changed";

function publishMarketplaceChange(): void {
  window.dispatchEvent(new CustomEvent(MARKETPLACE_CHANGED));
}

/**
 * The agent's own explanation for why it is not selling, or null.
 *
 * Reads the lines the agent already prints; nothing new is invented and no
 * status is guessed. Two shapes exist in the wild and both must be caught: the
 * generic `register: platform said <code>: {json}` that every build has, and
 * the multi-line block a 0.1.78+ agent prints for the marketplace floor.
 *
 * When the line carries a server JSON body, the server's `message` is what is
 * shown — it is written for a person ("...must be stable version X or newer.
 * Your machine keeps working for your own use either way"), while the envelope
 * around it is not. Pasting raw JSON into a tooltip is the same failure as
 * saying nothing, one step later.
 */
function agentRefusal(l: AgentLogLine): string | null {
  const line = l.line.replace(/^\[platform-agent\]\s*/, "").trim();
  if (!/register:|cannot be listed|too old to be listed|refus/i.test(line)) return null;
  const brace = line.indexOf("{");
  if (brace >= 0) {
    try {
      const body = JSON.parse(line.slice(brace));
      if (typeof body?.message === "string" && body.message) return body.message.slice(0, 400);
    } catch {
      // Not JSON after all, or truncated by the ring buffer. Fall through to
      // the raw line, which is still the agent's own sentence.
    }
  }
  return line.replace(/^platform-agent:\s*/, "").slice(0, 400) || null;
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
      <div ref={ref} className="modal modal--auth" role="dialog" aria-modal="true"
           onClick={(e) => e.stopPropagation()}>
        <div className="modal__head">
          <h2>{t("update.share.title")}</h2>
          <button className="iconbtn" onClick={onClose} aria-label={t("a11y.close")}>✕</button>
        </div>
        <p>{t("update.share.body", { installed, floor })}</p>
        <p className="field__hint">{t("update.share.unaffected")}</p>
        <div className="modal__actions">
          <button className="btn btn--primary" onClick={() => void open()}>
            {t("update.openDownloads")}
          </button>
          <button className="btn" onClick={onClose}>{t("update.share.notNow")}</button>
        </div>
      </div>
    </div>
  );
}

/** Provider-side control: accept other users' work and earn Sparks. */
export function ShareToggleButton({
  serviceReady,
  onNeedLogin,
}: {
  serviceReady: boolean;
  onNeedLogin?: () => void;
}) {
  const { t } = useI18n();
  const [on, setOn] = useState(() => loadSettings().providerEnabled);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [blocked, setBlocked] = useState<VersionTooOldToShare | null>(null);
  const [agent, setAgent] = useState<AgentStatus | null>(null);
  const [agentErr, setAgentErr] = useState<string | null>(null);

  useEffect(() => {
    const sync = () => setOn(loadSettings().providerEnabled);
    window.addEventListener(MARKETPLACE_CHANGED, sync);
    return () => window.removeEventListener(MARKETPLACE_CHANGED, sync);
  }, []);

  // Watch the agent itself, not just the stored preference (2026-09-15).
  //
  // Until now this pill was `providerEnabled && serviceReady` — the user's
  // saved choice and the LOCAL engine. Neither says whether the thing that
  // actually sells anything is alive. A 0.1.76 client with the switch on sat
  // green all evening while its agent registered, was refused 426 by the
  // marketplace floor, and exited; the machine was on nobody's discovery page
  // and the interface never said a word.
  //
  // The version case now has its own check before the agent is even started,
  // but that only closed one instance of the shape. An agent can also die on an
  // expired session, a registration rate limit, a suspended account or no
  // network at all, and every one of those used to look exactly like success.
  useEffect(() => {
    if (!inTauri()) return;
    let alive = true;
    void agentStatus().then((s) => { if (alive) setAgent(s); }).catch(() => { /* not running yet */ });
    void agentLogs(60).then((ls) => {
      if (!alive) return;
      const last = ls.map(agentRefusal).filter(Boolean).pop();
      if (last) setAgentErr(last);
    }).catch(() => { /* no ring buffer yet */ });
    const offStatus = onAgentStatus((s) => {
      setAgent(s);
      // A clean start clears the previous run's complaint; leaving it would
      // explain a healthy agent with a stale reason.
      if (s.state === "running") setAgentErr(null);
    });
    const offLog = onAgentLog((l) => {
      const why = agentRefusal(l);
      if (why) setAgentErr(why);
    });
    return () => { alive = false; offStatus(); offLog(); };
  }, []);

  if (!inTauri()) return null;
  const gate = platformGate();
  if (!gate.ok) {
    if (gate.reason === "no-url") {
      return (
        <button className="pill pill--market pill--standalone" disabled title={t("platform.err.noUrl")}>
          <span className="pill__dot" /><span className="pill__label">{t("share.off")}</span>
        </button>
      );
    }
    return (
      <button
        className="pill pill--market pill--standalone"
        onClick={() => { if (serviceReady) onNeedLogin?.(); }}
        title={serviceReady ? t("share.needLogin") : t("share.needService")}
      >
        <span className="pill__dot" /><span className="pill__label">{t("share.off")}</span>
      </button>
    );
  }

  const toggle = async () => {
    if (busy || !serviceReady) return;
    setBusy(true);
    setErr(null);
    try {
      if (on) {
        await agentStop();
        saveSettings({ ...loadSettings(), providerEnabled: false });
        setOn(false);
      } else {
        // The one moment a version genuinely blocks something, and the one
        // moment it is fair to say so (2026-09-14). Turning sharing ON is the
        // user stepping out to serve strangers — nothing of theirs is running
        // yet, they are actively doing this, and the gateway is about to refuse
        // the registration anyway. Catching it here turns an opaque 426 in a
        // headless agent's log into a sentence with a download link.
        //
        // Everything else in this client stays unconditional. There is no
        // version check on starting a cluster, loading a model, or serving the
        // local API, and there must not be: an old build there can only affect
        // the person who chose to run it.
        const standing = await currentVersionStanding();
        if (standing.belowShareFloor) {
          throw new VersionTooOldToShare(standing.installed, standing.shareFloor!);
        }
        const scheme = /^cluster-\d+$/;
        const stored = loadSettings().providerName;
        let providerName = stored && scheme.test(stored) ? stored : "";
        if (!providerName) {
          const taken = new Set((await getProviders()).map((p) => p.name));
          let n = 1;
          while (taken.has(`cluster-${n}`)) n++;
          providerName = `cluster-${n}`;
        }
        const s = loadSettings();
        await agentStart({
          platformUrl: gate.url,
          jwt: gate.session.token,
          name: providerName,
          coordApiPort: s.apiPort || 8000,
          coordToken: s.apiToken,
          modelId: s.modelId,
          quant: s.quant,
        });
        // Provider identity belongs only to lending. Request-help credentials
        // are created and controlled by the adjacent independent button.
        saveSettings({ ...loadSettings(), providerEnabled: true, providerName });
        setOn(true);
      }
      publishMarketplaceChange();
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
      setErr(msg);
      // The one failure with a remedy gets a dialog. It is also the only one
      // raised BEFORE the agent was asked to start, so there is nothing to
      // unwind — the rollback below is for a start that got partway.
      if (e instanceof VersionTooOldToShare) setBlocked(e);
      if (!on) {
        try { await agentStop(); } catch { /* already stopped */ }
        saveSettings({ ...loadSettings(), providerEnabled: false });
        setOn(false);
        publishMarketplaceChange();
      }
    } finally {
      setBusy(false);
    }
  };

  // The setting records the user's persistent choice. Green is reserved for
  // a choice that can actually serve work right now — which means the agent is
  // up, not merely that the switch is on and the local engine loaded.
  //
  // `starting` and `restarting` are deliberately NOT down: the supervisor is
  // mid-attempt and calling that a failure would flicker red on every ordinary
  // start. Unknown (null, e.g. the status call has not answered yet) is not
  // down either — this pill reports what it knows and never guesses.
  const agentDown = on && (agent?.state === "stopped" || agent?.state === "crashed");
  const active = on && serviceReady && !agentDown;
  const title = !serviceReady
    ? t("share.needService")
    : agentDown
      // The agent's own sentence when we have one. It is the only thing that
      // knows WHY, and the whole point of this pill is not to swallow it.
      ? `${t("share.agentDown")}${agentErr ? `\n\n${agentErr}` : ""}`
      : err ?? t(active ? "share.on" : "share.off");

  return (
    <>
      <button
        className={`pill pill--market pill--${active ? "ready" : "standalone"}`}
        disabled={busy}
        onClick={() => void toggle()}
        aria-pressed={active}
        title={title}
      >
        <span className="pill__dot" />
        <span className="pill__label">{busy ? "…" : t(active ? "share.on" : "share.off")}</span>
      </button>
      {blocked ? (
        <ShareVersionBlockedDialog
          installed={blocked.installed}
          floor={blocked.floor}
          onClose={() => setBlocked(null)}
        />
      ) : null}
    </>
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

  // An enabled overflow preference cannot route anything until the local
  // service is ready to receive the request in the first place.
  const active = on && serviceReady;

  return (
    <button
      className={`pill pill--market pill--${active ? "ready" : "standalone"}`}
      disabled={busy}
      onClick={() => void toggle()}
      aria-pressed={active}
      title={!serviceReady ? t("platform.overflow.needService") : err ?? t("platform.overflow.hint")}
    >
      <span className="pill__dot" />
      <span className="pill__label">
        {busy ? "…" : t(active ? "platform.overflow.on" : "platform.overflow.name")}
      </span>
    </button>
  );
}
