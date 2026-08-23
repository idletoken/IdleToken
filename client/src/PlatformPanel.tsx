import { useState } from "react";
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

/** Local emergency run control. Commercial listing and pricing live only in the portal. */
export function ShareToggleButton({ onNeedLogin }: { onNeedLogin?: () => void }) {
  const { t } = useI18n();
  const [on, setOn] = useState(() => loadSettings().sharingEnabled);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!inTauri()) return null;
  const gate = platformGate();
  if (!gate.ok) {
    if (gate.reason === "no-url") return null;
    return <button className="pill pill--standalone" onClick={onNeedLogin} title={t("share.needLogin")}><span className="pill__dot" />{t("share.off")}</button>;
  }

  const toggle = async () => {
    if (busy) return;
    setBusy(true); setErr(null);
    try {
      if (on) {
        await agentStop();
        saveSettings({ ...loadSettings(), sharingEnabled: false });
        setOn(false);
      } else {
        const SCHEME = /^cluster-\d+$/;
        const stored = loadSettings().providerName;
        let clusterName = stored && SCHEME.test(stored) ? stored : "";
        if (!clusterName) {
          const taken = new Set((await getProviders()).map((p) => p.name));
          let n = 1;
          while (taken.has(`cluster-${n}`)) n++;
          clusterName = `cluster-${n}`;
        }
        const s = loadSettings();
        let key = s.overflowKey;
        if (!key) key = (await createApiKey({ label: "overflow (borrow when busy)" })).apiKey;
        await agentStart({
          platformUrl: gate.url, jwt: gate.session.token, name: clusterName,
          coordApiPort: s.apiPort || 8000, coordToken: s.apiToken,
          modelId: s.modelId, quant: s.quant,
        });
        // This persists only the device owner's local run decision and the
        // credential required by the agent. No market/account snapshot is stored.
        saveSettings({ ...loadSettings(), sharingEnabled: true, overflowKey: key, providerName: clusterName });
        setOn(true);
      }
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
      setErr(msg);
      if (!on) {
        try { await agentStop(); } catch { /* already stopped */ }
        saveSettings({ ...loadSettings(), sharingEnabled: false });
        setOn(false);
      }
    } finally { setBusy(false); }
  };

  return <button className={`pill pill--${on ? "ready" : "standalone"}`} disabled={busy} onClick={() => void toggle()} title={err ?? t(on ? "share.on" : "share.off")}>
    <span className="pill__dot" />{busy ? "…" : t(on ? "share.on" : "share.off")}
  </button>;
}
