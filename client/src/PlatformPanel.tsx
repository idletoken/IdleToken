import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { loadSettings, saveSettings } from "./settings";
import {
  agentStart,
  agentStop,
  createApiKey,
  getProviders,
  inTauri,
  platformGate,
} from "./platform";

const MARKETPLACE_CHANGED = "idletoken:marketplace-settings-changed";

function publishMarketplaceChange(): void {
  window.dispatchEvent(new CustomEvent(MARKETPLACE_CHANGED));
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

  useEffect(() => {
    const sync = () => setOn(loadSettings().providerEnabled);
    window.addEventListener(MARKETPLACE_CHANGED, sync);
    return () => window.removeEventListener(MARKETPLACE_CHANGED, sync);
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
  // a choice that can actually serve work right now.
  const active = on && serviceReady;

  return (
    <button
      className={`pill pill--market pill--${active ? "ready" : "standalone"}`}
      disabled={busy}
      onClick={() => void toggle()}
      aria-pressed={active}
      title={!serviceReady ? t("share.needService") : err ?? t(active ? "share.on" : "share.off")}
    >
      <span className="pill__dot" />
      <span className="pill__label">{busy ? "…" : t(active ? "share.on" : "share.off")}</span>
    </button>
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
        || (await createApiKey({ label: "request help when local slot is busy" })).apiKey;
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
