// "What is everyone running" — the platform's 7-day usage ranking, on the page
// where the question is asked.
//
// Written 2026-08-20 (audit A-P1-3). The ranking already existed, but only
// inside the model-picker popover: you had to decide to change models, open the
// dialog and read a grey line before the product would tell you which models
// people actually use. That is the wrong order — for a new user "which model
// should I pick" comes BEFORE "let me change the model" — so the same data is
// rendered on the cluster page, where the first choice is made.
//
// Same honesty rule as the picker, and it matters more here because this block
// is about nothing else: the numbers are shown only when the platform reports
// `source: "measured"`. No platform, offline, or too little traffic to rank
// (`seeded`) renders NOTHING. A popularity list built from a seed table would
// be an invention, and this one is small enough that its absence costs the page
// nothing.
import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { getModel } from "./models";
import { fetchLeaderboard } from "./platform";

/** 12_345_678 → "12M". Shared shape with the picker's chips. */
function shortCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

const TOP_N = 5;

export default function PopularModels(props: {
  /** The model this machine would start now, highlighted in the list. */
  selectedId: string;
  /** Clicking a row selects that model (same effect as picking it). */
  onPick?: (id: string) => void;
}) {
  const { t } = useI18n();
  const [rows, setRows] = useState<{ model: string; totalTokens: number }[] | null>(null);

  useEffect(() => {
    let alive = true;
    fetchLeaderboard("week")
      .then((p) => {
        if (!alive || p.source !== "measured") return;
        setRows(p.rows.slice(0, TOP_N));
      })
      .catch(() => {
        /* enhancement only: no ranking is a fine answer, an invented one is not */
      });
    return () => {
      alive = false;
    };
  }, []);

  if (!rows || rows.length === 0) return null;

  return (
    <div className="popular">
      <div className="popular__head">{t("model.rank.byUsage")}</div>
      <ul className="popular__list">
        {rows.map((r, i) => {
          // An id the platform bills that this build has no manifest for still
          // gets a row: dropping it would silently renumber the ranking.
          const spec = getModel(r.model);
          const label = spec.id === r.model ? spec.label : r.model;
          return (
            <li key={r.model} className={`popular__row${r.model === props.selectedId ? " is-on" : ""}`}>
              <span className="popular__rank">{i + 1}</span>
              <button className="linkbtn popular__name" onClick={() => props.onPick?.(r.model)}>
                {label}
              </button>
              <span className="popular__n">{t("model.rank.tokens", { n: shortCount(r.totalTokens) })}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
