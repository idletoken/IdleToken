// "What can my machines run?" — the capability table (acceptance G-ADVISE).
//
// The verdicts are NOT computed here. The engine's advisor (src/common/advise.c)
// asks the planner and returns rows; this component only renders them. That is
// deliberate: a table that re-derives the fit rule in TypeScript drifts from the
// planner and starts promising models the cluster then refuses to load.
//
// Source of rows:
//   - paired cluster  → coordinator GET /idletoken/v1/capability (the whole pool)
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
  /** Memory the cluster needs to serve this row — weights + per-node shared
   *  weights + per-node inference overhead, from the planner
   *  (`idletoken_needed_bytes_quant`). Optional because a coordinator built before
   *  2026-08-21 does not send it; absent renders as "—" rather than as a
   *  number this file made up. */
  need_bytes?: number;
  shortfall_bytes: number;
  available: boolean;
  // Engine-side flag (src/common/advise.c): this model is served by ONE
  // machine, so its verdict was measured against the best single node rather
  // than the roster's pooled memory. Optional because an older coordinator
  // does not send it — absent reads as "clusterable", which is what every
  // model meant before the field existed.
  single_node?: boolean;
}

export interface CapabilityReport {
  nodes: number;
  models: CapabilityRow[];
}

/**
 * Load the report: cluster endpoint when paired, local engine otherwise.
 *
 * The cluster call goes through Rust (`api_capability`), NOT the webview's
 * fetch. The engine serves plain LAN HTTP with no CORS headers, so a direct
 * fetch is refused before a byte moves — which surfaced as
 * "读取能力报告失败: TypeError: Failed to fetch" the moment a cluster came up.
 * Every other engine call here already went through Rust for this reason
 * (api_stats, api_chat, diagnostics); this one was the exception.
 *
 * In a plain browser there is no Tauri and no cluster to reach; the caller
 * renders the error, which is the honest outcome there.
 */
export async function loadCapability(apiBaseUrl?: string | null): Promise<CapabilityReport> {
  const { invoke } = await import("@tauri-apps/api/core");
  if (apiBaseUrl) {
    // The cluster endpoint can be unreachable even when the pairing snapshot
    // says "online": the cluster just stopped, or — since 2026-08-15 — the
    // API serves only the coordinator's own machine, so every OTHER machine
    // gets connection-refused here by design. A dead endpoint must not take
    // the table down when the local advisor can still answer for THIS machine
    // (the table then honestly says "based on 1 machine").
    try {
      return (await invoke("api_capability", { baseUrl: apiBaseUrl })) as CapabilityReport;
    } catch {
      return (await invoke("advise_capability")) as CapabilityReport;
    }
  }
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

  // Four columns as of 2026-08-21 (user's call): model, precision, download
  // size, memory needed. "Can run" / "Max context" / "Note" and the footer
  // about pooling a new machine's memory are gone.
  //
  // What survives without a column of its own: the runnable-first SORT above
  // and the dimming below, both driven by `mode`. They are not columns and
  // they are the difference between a table you can scan and 150 identical
  // rows. The verdict is still readable from the numbers — the memory column
  // is what the row needs, and the reader knows what they have.
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
            <th>{t("cap.need")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.id}:${r.quant}`} className={r.mode === "no" || r.mode === "unavailable" ? "cap-row--dim" : ""}>
              <td>{r.label}</td>
              <td>{r.quant || "—"}</td>
              <td>{fmtBytes(r.weight_bytes)}</td>
              {/* An engine older than this column sends no need_bytes. Say so
                  with a dash — a table that fills a gap with a plausible
                  number is the failure mode this whole component was written
                  to avoid (see the header). */}
              <td>{r.need_bytes ? fmtBytes(r.need_bytes) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
