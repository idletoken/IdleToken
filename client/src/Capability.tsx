// "What can my machines run?" — the capability table (acceptance G-ADVISE).
//
// The verdicts are NOT computed here. The engine's advisor (src/common/advise.c)
// asks the planner and returns rows; this component only renders them. That is
// deliberate: a table that re-derives the fit rule in TypeScript drifts from the
// planner and starts promising models the cluster then refuses to load.
//
// Source of rows:
//   - paired cluster  → coordinator GET /v1/capability (the whole pool)
//   - standalone      → `idletoken-worker --advise-json` via the Tauri sidecar
import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { fmtBytes } from "./format";

export type CapabilityMode = "gpu_only" | "hybrid" | "no" | "unavailable";

export interface CapabilityRow {
  id: string;
  label: string;
  quant: string;
  mode: CapabilityMode;
  max_ctx: number;
  weight_bytes: number;
  shortfall_bytes: number;
  available: boolean;
}

export interface CapabilityReport {
  nodes: number;
  models: CapabilityRow[];
}

function ctxLabel(ctx: number): string {
  if (!ctx) return "—";
  if (ctx >= 1024 * 1024) return `${Math.round(ctx / (1024 * 1024))}M`;
  return `${Math.round(ctx / 1024)}K`;
}

/** Load the report: cluster endpoint when paired, local engine otherwise. */
export async function loadCapability(apiBaseUrl?: string | null): Promise<CapabilityReport> {
  if (apiBaseUrl) {
    const res = await fetch(`${apiBaseUrl.replace(/\/+$/, "")}/v1/capability`);
    if (!res.ok) throw new Error(`capability ${res.status}`);
    return (await res.json()) as CapabilityReport;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return (await invoke("advise_capability")) as CapabilityReport;
}

export default function Capability(props: { apiBaseUrl?: string | null }) {
  const { t } = useI18n();
  const [rep, setRep] = useState<CapabilityReport | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    let alive = true;
    loadCapability(props.apiBaseUrl)
      .then((r) => alive && setRep(r))
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [props.apiBaseUrl]);

  if (err) return <section className="card"><div className="cap-err">{t("cap.error")}: {err}</div></section>;
  if (!rep) return <section className="card"><div className="cap-loading">{t("cap.loading")}</div></section>;

  // Runnable first, then "almost" (smallest shortfall), then unavailable — the
  // order a user scans when deciding what to try.
  const rows = [...rep.models].sort((a, b) => {
    const rank = (r: CapabilityRow) =>
      r.mode === "gpu_only" ? 0 : r.mode === "hybrid" ? 1 : r.mode === "no" ? 2 : 3;
    return rank(a) - rank(b) || a.shortfall_bytes - b.shortfall_bytes;
  });

  const verdict = (r: CapabilityRow) => {
    if (r.mode === "gpu_only") return <span className="cap-yes">{t("cap.yesGpu")}</span>;
    if (r.mode === "hybrid") return <span className="cap-hybrid">{t("cap.yesHybrid")}</span>;
    if (r.mode === "unavailable") return <span className="cap-na">{t("cap.notInBuild")}</span>;
    return <span className="cap-no">{t("cap.no")}</span>;
  };

  return (
    <section className="card cap-card">
      <header className="cap-head">
        <h3>{t("cap.title")}</h3>
        <span className="cap-sub">{t("cap.forNodes").replace("{n}", String(rep.nodes))}</span>
      </header>
      <table className="cap-table">
        <thead>
          <tr>
            <th>{t("cap.model")}</th>
            <th>{t("cap.quant")}</th>
            <th>{t("cap.size")}</th>
            <th>{t("cap.canRun")}</th>
            <th>{t("cap.maxCtx")}</th>
            <th>{t("cap.note")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            return (
              <tr key={`${r.id}:${r.quant}`} className={r.mode === "no" || r.mode === "unavailable" ? "cap-row--dim" : ""}>
                <td>{r.label}</td>
                <td>{r.quant || "—"}</td>
                <td>{fmtBytes(r.weight_bytes)}</td>
                <td>{verdict(r)}</td>
                <td>{ctxLabel(r.max_ctx)}</td>
                <td className="cap-note">
                  {r.mode === "no" && r.shortfall_bytes > 0
                    ? t("cap.needMore").replace("{gb}", String(Math.ceil(r.shortfall_bytes / 1024 ** 3)))
                    : r.mode === "hybrid"
                      ? t("cap.hybridNote")
                      : r.mode === "unavailable"
                        ? t("cap.notInBuildNote")
                        : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="cap-foot">{t("cap.footer")}</p>
    </section>
  );
}
