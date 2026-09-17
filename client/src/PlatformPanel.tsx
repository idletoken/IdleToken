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
  setLocalOverflow,
  setProviderListed,
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
  /**
   * What the MARKETPLACE says about this machine — the only authority on
   * whether it is actually on sale. `null` = not asked yet / could not ask.
   */
  const [shelf, setShelf] = useState<{ listed: boolean; reason: string | null } | null>(null);

  useEffect(() => {
    const sync = () => setOn(loadSettings().providerEnabled);
    window.addEventListener(MARKETPLACE_CHANGED, sync);
    return () => window.removeEventListener(MARKETPLACE_CHANGED, sync);
  }, []);

  // Check at STARTUP, not only when the switch is clicked (2026-09-15).
  //
  // `providerEnabled` is persisted, so a client that was sharing yesterday comes
  // up with the switch already on and nothing re-asks whether it still may. On
  // a machine whose version fell below the floor overnight that meant a green
  // switch selling nothing, and the refusal only appeared if the user happened
  // to toggle it off and on again — which is not a thing anyone does to find
  // out whether they are earning.
  //
  // Below the floor the switch is turned OFF, not merely greyed: it cannot
  // serve, so leaving it on would be recording an intention the product cannot
  // honour. The dialog says what happened and how to get back.
  useEffect(() => {
    if (!inTauri() || !loadSettings().providerEnabled) return;
    let alive = true;
    // force: an admission decision is never served from the advisory cache.
    void currentVersionStanding(true).then((s) => {
      if (!alive || !s.belowShareFloor) return;
      void agentStop().catch(() => { /* may never have started */ });
      saveSettings({ ...loadSettings(), providerEnabled: false });
      setOn(false);
      publishMarketplaceChange();
      setBlocked(new VersionTooOldToShare(s.installed, s.shareFloor!));
    });
    return () => { alive = false; };
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

  // Ask the marketplace whether this machine is actually on sale (2026-09-16).
  //
  // Everything else this pill could look at is local: a saved preference, the
  // local engine, a live agent process. All three can be perfectly healthy on a
  // machine the platform has taken off the market — which is exactly what the
  // version floor does, and what a suspension, a rate limit or a manual unlist
  // do too. The switch is not only a switch, it is the status light for "am I
  // selling?", and only the platform knows the answer.
  //
  // Polled rather than pushed because delisting happens platform-side, on the
  // agent's next keep-alive, with nothing flowing back to this process.
  useEffect(() => {
    if (!inTauri()) return;
    let alive = true;
    const read = () => {
      if (!loadSettings().providerEnabled) { setShelf(null); return; }
      void getProviders().then((ps) => {
        if (!alive) return;
        const mine = ps.find((p) => p.name === loadSettings().providerName);
        // No row at all is not "listed: false" — it is "the platform has never
        // heard of this machine", which is what a refused registration looks
        // like. Both are "not selling", and neither may show green.
        setShelf(mine ? { listed: mine.listed, reason: mine.unlistedReason ?? null } : { listed: false, reason: null });
      }).catch(() => { /* offline or signed out: claim nothing */ });
    };
    // A read triggered by the switch itself runs BEFORE the machine is on sale:
    // flipping it only starts the agent, and registering is a round trip after
    // that. One read at that instant sees no row — correctly, for another second
    // or two — and the next steady read is 30 s away, so the pill stayed dark on
    // a machine that had in fact registered and listed within seconds.
    //
    // Dark reads as "sharing failed", and the reasonable response to that is to
    // flip the switch off and on again. That mints a fresh provider identity
    // every time (registrations are one-shot) and restarts the same wait, so the
    // display is what keeps the loop going. Measured 2026-09-16: four identities
    // in five minutes, every one of them registered and listed by the platform.
    //
    // So a change is followed by a short burst of re-reads that stops as soon as
    // the platform answers. The steady poll is unchanged and still owns the
    // other direction — delisting happens platform-side with nothing flowing
    // back here, and no burst can be scheduled for an event we never hear about.
    let burst: ReturnType<typeof setTimeout>[] = [];
    const clearBurst = () => { burst.forEach(clearTimeout); burst = []; };
    const readSoon = () => {
      clearBurst();
      read();
      burst = [1_000, 2_500, 5_000, 9_000, 15_000].map((ms) => setTimeout(read, ms));
    };

    read();
    const timer = setInterval(read, 30_000);
    window.addEventListener(MARKETPLACE_CHANGED, readSoon);
    const offStatus = onAgentStatus(() => readSoon());
    return () => {
      alive = false;
      clearInterval(timer);
      clearBurst();
      window.removeEventListener(MARKETPLACE_CHANGED, readSoon);
      offStatus();
    };
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
        // Unlist BEFORE stopping the agent, and best-effort: the platform must
        // stop sending strangers here the moment the user says stop, not 60
        // seconds later when the heartbeat ages out — that window could only
        // produce failed requests against an agent that had already gone. It
        // must not be able to block turning sharing off, though, so a platform
        // that cannot be reached costs the old behaviour and nothing more.
        try {
          const mine = (await getProviders()).find((p) => p.name === loadSettings().providerName);
          if (mine?.listed) await setProviderListed(mine.id, false);
        } catch { /* offline or signed out: the heartbeat timeout still covers it */ }
        await agentStop();
        saveSettings({ ...loadSettings(), providerEnabled: false });
        setOn(false);
        setShelf(null);
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
        const standing = await currentVersionStanding(true);
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
  // Green means SELLING, and the marketplace is the only thing that knows.
  //
  // Not `on` (a saved wish), not `serviceReady` (the local engine), not merely
  // a live agent: a machine can have all three and be unlisted. `shelf === null`
  // — not asked yet, offline, signed out — is NOT green either, because this
  // pill states a fact and has no business guessing one. That also removes the
  // startup flash where an agent that had never come up showed green until its
  // status arrived.
  const active = on && serviceReady && !agentDown && shelf?.listed === true;
  const title = !serviceReady
    ? t("share.needService")
    : agentDown
      // The agent's own sentence when we have one. It is the only thing that
      // knows WHY, and the whole point of this pill is not to swallow it.
      ? `${t("share.agentDown")}${agentErr ? `\n\n${agentErr}` : ""}`
      // Unlisted BY THE PLATFORM, with its reason. This is the case that used
      // to be invisible: switch on, agent up, nothing on sale.
      : on && shelf && !shelf.listed
        ? `${t("share.notListed")}${shelf.reason ? `\n\n${shelf.reason}` : ""}`
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
