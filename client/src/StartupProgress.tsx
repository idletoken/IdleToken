import { useEffect, useState } from "react";
import { useI18n } from "./i18n";

function elapsedLabel(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

/**
 * Long startup work does not expose a trustworthy byte or layer percentage.
 * Show the real current stage, a live elapsed time, and an explicitly
 * indeterminate bar instead of inventing a completion estimate.
 */
export default function StartupProgress(props: {
  label: string;
  detail: string;
  compact?: boolean;
}) {
  const { t } = useI18n();
  const [startedAt] = useState(() => Date.now());
  const [now, setNow] = useState(startedAt);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const elapsed = elapsedLabel(Math.max(0, Math.floor((now - startedAt) / 1000)));
  return (
    <div className={`startup-progress${props.compact ? " startup-progress--compact" : ""}`}>
      <div className="startup-progress__head">
        <span className="startup-progress__label" role="status" aria-live="polite">{props.label}</span>
        <span className="startup-progress__elapsed">{t("startup.elapsed", { time: elapsed })}</span>
      </div>
      <div className="startup-progress__track" role="progressbar" aria-label={props.label}>
        <i />
      </div>
      <p className="startup-progress__detail">{props.detail}</p>
    </div>
  );
}
