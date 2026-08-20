// "What has gone wrong on this machine" — the improvement loop, and the honest
// bound on it: failures are written down here, they never leave the machine on
// their own, and exporting a diagnostics bundle is how they reach us.
//
// No sharing switch (2026-08-15; its predecessor, the telemetry switch, went
// on 2026-08-13). Exporting the bundle is itself the explicit act of sharing —
// it produces a file the user sends by hand — so a second opt-out ahead of it
// gated nothing a user could feel, and a bundle without the failures is the
// one kind that cannot diagnose anything.
import { useCallback, useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { clearProblems, readProblems, type Problem } from "./problems";

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
  const [items, setItems] = useState<Problem[]>([]);
  const [expanded, setExpanded] = useState(false);

  const reload = useCallback(() => setItems(readProblems()), []);
  useEffect(() => reload(), [reload]);

  const shown = expanded ? items : items.slice(0, 3);

  return (
    <div className="problems">
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
