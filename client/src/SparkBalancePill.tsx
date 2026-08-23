import { useCallback, useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { portalUrl } from "./links";
import { getMe, inTauri, platformGate } from "./platform";

/**
 * Ephemeral balance indicator. The value exists only in React state: there is
 * no AppSettings key, browser storage, native config, database, or log path.
 */
export default function SparkBalancePill() {
  const { t, lang } = useI18n();
  const gate = platformGate();
  const [balance, setBalance] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const refresh = useCallback(() => {
    setBalance(null); setFailed(false);
    getMe().then((me) => setBalance(me.balanceCredits)).catch(() => { setBalance(null); setFailed(true); });
  }, [gate.ok ? gate.session.token : ""]);

  useEffect(() => {
    if (!gate.ok) { setBalance(null); setFailed(false); return; }
    refresh();
    const focus = () => refresh();
    window.addEventListener("focus", focus);
    window.addEventListener("idletoken:platform-request-complete", focus);
    const timer = window.setInterval(refresh, 60_000);
    return () => {
      window.removeEventListener("focus", focus);
      window.removeEventListener("idletoken:platform-request-complete", focus);
      window.clearInterval(timer);
    };
  }, [gate.ok ? gate.session.token : "", refresh]);

  if (!gate.ok) return null;
  const target = portalUrl();
  const openSpark = async () => {
    if (!target) return;
    const url = `${target}/market?view=wallet`;
    if (inTauri()) { const { open } = await import("@tauri-apps/plugin-shell"); await open(url); }
    else window.open(url, "_blank", "noopener,noreferrer");
  };
  const text = failed || balance == null
    ? t("spark.balance.unknown")
    : `✦ ${balance.toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits: 1 })} ${t("spark.balance.unit")}`;
  return <button className="spark-balance-pill" onClick={() => void openSpark()} title={failed ? t("spark.balance.failed") : t("spark.balance.open")}>
    {text}
  </button>;
}
