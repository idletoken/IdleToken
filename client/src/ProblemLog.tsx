// "What has gone wrong on this machine" — the improvement loop, and the honest
// bound on it.
//
// This is what replaced the "Share anonymous telemetry" switch (2026-08-13).
// That switch gated nothing: there is no telemetry client anywhere in this
// product, so turning it off bought the careful user precisely nothing. What
// the product actually needs in order to improve is the failures themselves,
// and what a user can honestly be offered is this: they are written down here,
// they never leave the machine on their own, and exporting them is how they
// reach us.
//
// The switch is about SHARING, not recording. Off = the log is left out of the
// diagnostics export (and of any future upload); the records stay, because they
// are also how the user themselves answers "what keeps going wrong here", and
// because someone who switches sharing back on should not find their history
// deleted. Throwing them away is the button at the bottom, said out loud.
import { useCallback, useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { clearProblems, problemsShared, readProblems, setProblemsShared, type Problem } from "./problems";

/** Local time, short. These are read next to "did that just happen?", so the
 *  date only appears when it is not today. */
function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const today = new Date();
  const sameDay =
    d.getFullYear() === today.getFullYear() && d.getMonth() === today.getMonth() && d.getDate() === today.getDate();
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return sameDay ? hm : `${d.getMonth() + 1}/${d.getDate()} ${hm}`;
}

function detailOf(p: Problem): string {
  if (!p.detail) return "";
  return Object.entries(p.detail)
    .filter(([, v]) => v !== "" && v !== 0 && v !== false)
    .map(([k, v]) => `${k}=${v}`)
    .join(" · ");
}

export default function ProblemLog() {
  const { t } = useI18n();
  const [share, setShare] = useState(problemsShared);
  const [items, setItems] = useState<Problem[]>([]);
  const [expanded, setExpanded] = useState(false);

  const reload = useCallback(() => setItems(readProblems()), []);
  useEffect(() => reload(), [reload]);

  // Off means "do not send these to the developers", and nothing more. The log
  // stays: it is the user's own record of what keeps failing, and someone who
  // turns sharing back on later should still have their history. Deleting is
  // the separate button below, where a destructive action belongs.
  const toggle = (next: boolean) => {
    setProblemsShared(next);
    setShare(next);
  };

  const shown = expanded ? items : items.slice(0, 3);

  return (
    <div className="problems">
      <div className="setting-row setting-row--inline">
        <div className="setting-row__label">
          <span className="setting-row__k">{t("prob.share")}</span>
        </div>
        <div className="setting-row__control">
          <button
            role="switch"
            aria-checked={share}
            aria-label={t("prob.share")}
            className={`switch${share ? " is-on" : ""}`}
            onClick={() => toggle(!share)}
          >
            <span className="switch__knob" />
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <p className="setting-hint">{t("prob.none")}</p>
      ) : (
        <>
          <ul className="problems__list">
            {shown.map((p, i) => (
              <li key={`${p.at}-${i}`} className="problems__item">
                <span className="problems__at">{when(p.at)}</span>
                <span className={`problems__kind problems__kind--${p.kind}`}>{t(`prob.kind.${p.kind}` as const)}</span>
                {/* The message verbatim — it is the one string that identifies
                    the failure, and paraphrasing it would defeat the log. */}
                <span className="problems__msg" title={`${p.message}${detailOf(p) ? `\n${detailOf(p)}` : ""}`}>
                  {p.message}
                </span>
              </li>
            ))}
          </ul>
          <div className="problems__foot">
            {items.length > 3 ? (
              <button className="linkbtn" onClick={() => setExpanded(!expanded)}>
                {expanded ? t("prob.less") : t("prob.more", { n: items.length - 3 })}
              </button>
            ) : null}
            <button
              className="linkbtn"
              onClick={() => {
                clearProblems();
                reload();
              }}
            >
              {t("prob.clear")}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
