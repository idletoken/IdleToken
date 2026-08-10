// Weight presence + download control, rendered next to whatever the user just
// pressed. Shared by the model list (Settings → Quick) and the "run on this
// machine" row (Cluster), because BOTH of them start a download.
//
// This used to be one floating bar for the whole app. Removing it fixed the
// interruption but created a worse bug: pressing "Run it here" on the Cluster
// page kicked off a 4.7 GB download whose progress — and whose failures — only
// rendered on a different page. The button looked dead. Feedback has to live
// where the action is, so this component goes to both places rather than back
// into a bar that covers everything.
import { useI18n } from "./i18n";
import { fmtBytes } from "./format";

export interface WeightsInfo {
  needs: boolean;
  path: string;
  dl: { have: number; total: number; endpoint?: string; note?: string; error?: string } | null;
  onDownload: () => void;
  onCancel: () => void;
}

/**
 * Four states, and the button says what pressing it DOES in each:
 *   downloading -> progress + Cancel      failed -> reason + Retry
 *   missing     -> Download               present -> "ready", no button
 *
 * `idle` hides the resting "not downloaded yet" case — the Cluster row passes
 * it, because there the download is implied by the button next to it and the
 * standing fact would just be noise.
 */
export default function WeightsRow(props: { w: WeightsInfo; idle?: "hide" | "show" }) {
  const { t } = useI18n();
  const d = props.w.dl;
  // In Settings this sits inside a <label> that owns a radio; without this,
  // clicking Download would also re-select the model.
  const stop = (e: React.MouseEvent) => e.preventDefault();

  if (d?.error) {
    return (
      <span className="wrow wrow--err" onClick={stop}>
        <span className="wrow__msg">{t("weights.failed")} — {d.error}</span>
        <button className="linkbtn" onClick={props.w.onDownload}>{t("weights.retry")}</button>
      </span>
    );
  }
  if (d) {
    const pct = d.total > 0 ? Math.min(100, Math.round((d.have / d.total) * 100)) : 0;
    return (
      <span className="wrow" onClick={stop}>
        <span className="wrow__bar"><i style={{ width: `${pct}%` }} /></span>
        <span className="wrow__msg">
          {fmtBytes(d.have)}{d.total > 0 ? ` / ${fmtBytes(d.total)}` : ""}
          {d.endpoint ? ` · ${t("weights.from")} ${new URL(d.endpoint).host}` : ""}
          {d.note ? ` · ${d.note}` : ""}
        </span>
        <button className="linkbtn" onClick={props.w.onCancel}>{t("weights.cancel")}</button>
      </span>
    );
  }
  if (props.w.needs) {
    if (props.idle === "hide") return null;
    return (
      <span className="wrow" onClick={stop}>
        <span className="wrow__msg">{t("weights.needed")}</span>
        <button className="linkbtn" onClick={props.w.onDownload}>{t("weights.download")}</button>
      </span>
    );
  }
  if (props.idle === "hide") return null;
  return <span className="wrow wrow--ok">{t("weights.ready")}</span>;
}
