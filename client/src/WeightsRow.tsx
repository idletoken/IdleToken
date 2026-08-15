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
  dl: { have: number; total: number; note?: string } | null;
  /** Bytes of an unfinished copy on disk; the next attempt resumes from it. */
  partialBytes: number;
  /** Why the last attempt stopped. A footnote on the "not here yet" row — not
   *  a state of its own. */
  lastError: string | null;
  onDownload: () => void;
  onCancel: () => void;
}

/**
 * Three states, and the button says what pressing it DOES in each:
 *   downloading -> progress + Cancel
 *   missing     -> what is missing (or how much is already here) + Download
 *   present     -> "ready", no button
 *
 * There is deliberately no "failed" state (2026-08-13). A download that did not
 * finish leaves the machine exactly where it was — without the weights — and
 * that is one situation, not two. Announcing "Download failed" instead made a
 * dead end out of it: the row stopped offering the thing that fixes it and had
 * to be cleared first, and the sentence was misleading anyway, because what is
 * on disk is kept and the next attempt continues from there. The reason for the
 * last failure rides along as a quiet note, so a download that keeps dying is
 * still diagnosable.
 *
 * `idle` hides the resting "not downloaded yet" case — the Cluster row passes
 * it, because there the download is implied by the button next to it and the
 * standing fact would just be noise.
 */
export default function WeightsRow(props: { w: WeightsInfo; idle?: "hide" | "show" }) {
  // tErr: notes and failure reasons from the Rust download path carry a
  // "[CODE] detail" prefix and render localized; anything else passes through.
  const { t, tErr } = useI18n();
  const d = props.w.dl;
  // In Settings this sits inside a <label> that owns a radio; without this,
  // clicking Download would also re-select the model.
  const stop = (e: React.MouseEvent) => e.preventDefault();

  if (d) {
    const pct = d.total > 0 ? Math.min(100, Math.round((d.have / d.total) * 100)) : 0;
    return (
      <span className="wrow" onClick={stop}>
        <span className="wrow__bar"><i style={{ width: `${pct}%` }} /></span>
        <span className="wrow__msg">
          {fmtBytes(d.have)}{d.total > 0 ? ` / ${fmtBytes(d.total)}` : ""}
          {d.note ? ` · ${tErr(d.note)}` : ""}
        </span>
        <button className="linkbtn" onClick={props.w.onCancel}>{t("weights.cancel")}</button>
      </span>
    );
  }
  if (props.w.needs) {
    // The Cluster row hides the resting state — but not when there is something
    // to say about it. A part-downloaded model and a failed attempt are both
    // things the button next to it does NOT imply.
    const partial = props.w.partialBytes > 0;
    if (props.idle === "hide" && !partial && !props.w.lastError) return null;
    return (
      <span className="wrow" onClick={stop}>
        {/* The row is one ellipsised line, so a long reason gets cut — the
            tooltip is where the whole sentence lives. */}
        <span className="wrow__msg" title={props.w.lastError ? tErr(props.w.lastError) : undefined}>
          {partial ? t("weights.partial", { have: fmtBytes(props.w.partialBytes) }) : t("weights.needed")}
          {/* Why the last one stopped, in the same muted voice as the rest of
              the row. Not red, not a headline: it is context for the button,
              not a verdict on the machine. */}
          {props.w.lastError ? (
            <span className="wrow__note"> · {t("weights.lastStop", { why: tErr(props.w.lastError) })}</span>
          ) : null}
        </span>
        <button className="linkbtn" onClick={props.w.onDownload}>
          {partial ? t("weights.resume") : t("weights.download")}
        </button>
      </span>
    );
  }
  if (props.idle === "hide") return null;
  return <span className="wrow wrow--ok">{t("weights.ready")}</span>;
}
