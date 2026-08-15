// The update prompt, and the two other answers a check can give.
//
// It is a modal rather than a banner because an update replaces the running
// application and, on a machine that is serving a cluster, takes the engine
// down with it. That is worth one interruption — and it is worth making
// "Later" exactly as easy to hit as "Update now".
//
// All three outcomes are rendered here on purpose. "Checked, you are current"
// and "could not check" are different facts, and a checker that shows the
// first when it means the second is how machines stay months behind believing
// they are up to date. The automatic check at startup only opens this when
// something was found; a check the user asked for always opens it, because a
// button that answers nothing reads as broken.
import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { useDialog } from "./useDialog";
import Markdown from "./Markdown";
import { fmtBytes } from "./format";
import { getUpdateProvider, type UpdateInfo } from "./provider/update";

export type UpdateResult =
  | { kind: "found"; info: UpdateInfo }
  | { kind: "upToDate"; current: string }
  | { kind: "error"; message: string };

type Phase = "ask" | "downloading" | "installing" | "error";

export default function UpdateDialog(props: {
  result: UpdateResult;
  /** "Later" / Close / Escape. Not offered once the install has started. */
  onClose: () => void;
  /** Reported so the acceptance channel can assert what the user would see. */
  onOutcome?: (outcome: { phase: Phase; bytes?: number; error?: string }) => void;
}) {
  const { t } = useI18n();
  const ref = useDialog(props.onClose);
  const [phase, setPhase] = useState<Phase>("ask");
  const [progress, setProgress] = useState<{ done: number; total: number | null }>({ done: 0, total: null });
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const off = getUpdateProvider().onProgress((p) =>
      setProgress({ done: p.downloaded, total: p.total })
    );
    return off;
  }, []);

  // Nothing to install: a short acknowledgement with one button.
  if (props.result.kind !== "found") {
    const r = props.result;
    return (
      <div className="modal-scrim" onClick={props.onClose}>
        <div ref={ref} className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
          <div className="modal__head">
            <h2>{r.kind === "upToDate" ? t("update.upToDate", { current: r.current }) : t("update.checkFailed")}</h2>
          </div>
          {r.kind === "error" ? (
            <p className="update__error" role="alert">
              {r.message}
            </p>
          ) : null}
          <div className="modal__foot">
            <button className="btn-primary" onClick={props.onClose}>
              {t("update.later")}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const info = props.result.info;

  const run = async () => {
    const provider = getUpdateProvider();
    setPhase("downloading");
    let bytes = 0;
    try {
      bytes = await provider.download();
    } catch (e) {
      // Includes the one failure that matters most: a signature that does not
      // verify. It is shown as a failed update, never worked around.
      setError(String(e));
      setPhase("error");
      props.onOutcome?.({ phase: "error", error: String(e) });
      return;
    }
    setPhase("installing");
    props.onOutcome?.({ phase: "installing", bytes });
    try {
      // On success the process is replaced, so nothing after this runs.
      await provider.install();
    } catch (e) {
      setError(String(e));
      setPhase("error");
      props.onOutcome?.({ phase: "error", error: String(e) });
    }
  };

  const pct =
    progress.total && progress.total > 0
      ? Math.min(100, Math.round((progress.done / progress.total) * 100))
      : null;

  return (
    <div className="modal-scrim" onClick={phase === "ask" || phase === "error" ? props.onClose : undefined}>
      <div ref={ref} className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal__head">
          <div>
            <h2>{t("update.title", { version: info.version })}</h2>
            <p className="update__meta">
              {t("update.from", { current: info.currentVersion })}
              {info.channel === "beta" ? ` · ${t("update.betaChannel")}` : ""}
              {info.date ? ` · ${info.date.slice(0, 10)}` : ""}
            </p>
          </div>
        </div>

        {info.notes ? (
          <div className="update__notes">
            <Markdown text={info.notes} />
          </div>
        ) : null}

        {phase === "downloading" ? (
          <div className="update__progress">
            <span className="wrow__bar">
              <i style={{ width: `${pct ?? 0}%` }} />
            </span>
            <span className="wrow__msg">
              {fmtBytes(progress.done)}
              {progress.total ? ` / ${fmtBytes(progress.total)}` : ""}
            </span>
          </div>
        ) : null}

        {phase === "installing" ? <p className="update__meta">{t("update.installing")}</p> : null}

        {phase === "error" ? (
          <p className="update__error" role="alert">
            {t("update.failed")} {error}
          </p>
        ) : null}

        <div className="modal__foot">
          {phase === "ask" || phase === "error" ? (
            <button className="btn-secondary" onClick={props.onClose}>
              {t("update.later")}
            </button>
          ) : null}
          {phase === "ask" ? (
            <button className="btn-primary" onClick={run}>
              {t("update.install")}
            </button>
          ) : null}
          {phase === "error" ? (
            <button className="btn-primary" onClick={run}>
              {t("update.retry")}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
