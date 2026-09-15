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
  // The failure's own words, not just the fact of one (2026-09-15). This used
  // to be a bare `setFailed(true)` with the error discarded in the catch, so a
  // machine whose every platform call died on a dead proxy showed the same
  // dash as one that had simply not answered yet — and the message that said
  // `127.0.0.1:7897` existed nowhere a user could reach.
  const [failure, setFailure] = useState<string | null>(null);
  const refresh = useCallback(() => {
    setBalance(null); setFailure(null);
    getMe()
      .then((me) => {
        // A 200 with no number is NOT a failed request. Keeping them apart
        // matters because they send you to different places: one is the
        // network, the other is this client and the gateway disagreeing about
        // the shape of /me.
        if (typeof me.balanceCredits === "number") setBalance(me.balanceCredits);
        else setFailure("the platform answered without a balance");
      })
      .catch((e) => { setBalance(null); setFailure(String((e as Error)?.message ?? e)); });
  }, [gate.ok ? gate.session.token : ""]);

  useEffect(() => {
    if (!gate.ok) { setBalance(null); setFailure(null); return; }
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
  // Three states, three appearances. They used to be two: `failed || balance
  // == null` rendered one dash for both "still asking" and "asked and it
  // broke", so a permanently dead link was indistinguishable from a pill that
  // was about to fill in — and nothing anywhere said why.
  const text = failure
    ? t("spark.balance.unavailable")
    : balance == null
      ? t("spark.balance.unknown")
      : `✦ ${balance.toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits: 1 })} ${t("spark.balance.unit")}`;
  // On failure the tooltip carries the underlying message verbatim. It is the
  // only place the actual cause exists — "network: can't reach the platform
  // server (error sending request ... 127.0.0.1:7897)" names the dead proxy
  // that a summary like "temporarily unavailable" would hide.
  const title = failure
    ? `${t("spark.balance.failed")}\n\n${failure}`
    : t("spark.balance.open");
  return <button
    className={`spark-balance-pill${failure ? " spark-balance-pill--failed" : ""}`}
    onClick={() => void openSpark()}
    title={title}
  >
    {text}
  </button>;
}
