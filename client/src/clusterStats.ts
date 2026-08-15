// Cluster serving counters (engine GET /idletoken/v1/stats), shared by every surface that
// needs to say something about the running cluster.
//
// Lifted out of App.tsx on 2026-08-13 because the chat page needs one fact from
// it — WHICH MODEL is answering — and the alternative was for chat to display
// the local setting instead. Those are two different claims: the model is fixed
// when the coordinator loads it, so after changing the setting the local value
// names a model nothing is running.
import { useEffect, useState } from "react";
import { inTauri } from "./platform";
import type { ClusterApi } from "./pairing";

export interface ClusterStats {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_hits?: number; // KV prefix reuse hits (undefined on older engines)
  cached_tokens?: number; // cumulative prefill tokens saved
  uptime_s: number;
  last_tok_per_s: number;
  // What the cluster is ACTUALLY serving, straight from the coordinator that
  // loaded it. Absent on older engines — the UI then says nothing rather than
  // falling back to the local setting, which is a different claim.
  model?: string;
  model_label?: string;
  quant?: string;
  // v2 llamacpp mode: the coordinator supervises a llama-server child and
  // mirrors its health here (append-only fields on /idletoken/v1/stats — see
  // results/llamacpp-b1-sidecar-20260814.md). Chat 503s until engine_state is
  // "ready"; the UI must show that instead of a healthy-looking panel over a
  // 503ing API. Absent on the legacy cluster path.
  engine?: string; // "llamacpp"
  engine_state?: "starting" | "ready" | "restarting" | "failed";
  engine_restarts?: number;
}

/** What the coordinator reports it loaded, once it has reported anything. */
export interface ServedModel {
  id: string;
  label: string;
  quant: string;
}

/** Dev-sim: the model the simulated cluster "loaded". Module scope, not effect
 *  scope, because a real coordinator latches this once at startup and keeps
 *  reporting it — navigating between pages must not silently re-latch it, or
 *  the setting and the cluster can never disagree in the browser build and the
 *  mismatch path goes untested until it reaches a real machine. */
let g_simLoadedModel: ServedModel | null = null;

/** Dev-sim: the simulated coordinator died (leave / model switch). Clearing the
 *  latch is what makes the browser build re-report the NEW model after a
 *  restart — without it a switch looked like it had silently failed, which is
 *  the opposite of the lie the latch exists to prevent. */
export function forgetSimLoadedModel(): void {
  g_simLoadedModel = null;
}

/**
 * Poll the cluster's serving counters.
 *
 * The coordinator executes requests **serially**, so a poll is not free — it is
 * another connection competing with generation. Two rules follow:
 *   - one poll per screen (the cluster card shares its result between the
 *     activity numbers and the served-model row);
 *   - `modelOnly` for screens that want nothing but the model name. It stops
 *     polling the moment the coordinator has reported one, so the chat page
 *     does not keep a 5 s connection alive underneath a running generation.
 */
export function useClusterStats(
  api: ClusterApi | null,
  source: "engine" | "dev-sim",
  opts?: {
    /** Dev-sim only: what to claim is loaded. Latched on the first tick, the way
     *  a real coordinator latches it at startup — so changing the setting
     *  afterwards diverges in the browser build exactly as it does on a machine. */
    simModel?: ServedModel;
    /** Stop as soon as `model_label` is known (see above). */
    modelOnly?: boolean;
  }
): ClusterStats | null {
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const simModel = opts?.simModel;
  const modelOnly = !!opts?.modelOnly;
  const status = api?.status;
  const baseUrl = api?.baseUrl;

  useEffect(() => {
    // Offline: forget what we knew. These numbers — and above all the served
    // model — describe a coordinator that is answering; carrying the last value
    // through a restart is how a switch ends up displaying the model it just
    // replaced.
    if (!baseUrl || status !== "online") {
      setStats(null);
      return;
    }
    let live = true;
    let timer: ReturnType<typeof setInterval> | null = null;
    let simBase = { requests: 0, tokens: 0 };
    const done = () => {
      if (timer !== null) clearInterval(timer);
      timer = null;
    };
    const tick = async () => {
      if (!live) return;
      if (inTauri()) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          const v = await invoke<ClusterStats>("api_stats", { baseUrl });
          if (!live) return;
          setStats(v);
          // Asked for the model and got it: nothing left to poll for.
          if (modelOnly && v.model_label) done();
        } catch {
          /* stats are best-effort; the row just stays as-is */
        }
      } else if (source === "dev-sim") {
        simBase = { requests: simBase.requests + (Math.random() < 0.4 ? 1 : 0), tokens: simBase.tokens + 40 };
        if (!g_simLoadedModel && simModel) g_simLoadedModel = { ...simModel };
        if (!live) return;
        setStats({
          requests: simBase.requests,
          input_tokens: Math.floor(simBase.tokens * 0.6),
          output_tokens: Math.floor(simBase.tokens * 0.4),
          uptime_s: Math.floor(performance.now() / 1000) + 60,
          last_tok_per_s: 9.5,
          model: g_simLoadedModel?.id,
          model_label: g_simLoadedModel?.label,
          quant: g_simLoadedModel?.quant,
        });
        if (modelOnly && g_simLoadedModel) done();
      }
    };
    tick();
    timer = setInterval(tick, 5000);
    return () => {
      live = false;
      done();
    };
  }, [status, baseUrl, source, modelOnly]);

  return stats;
}

/** The served model as a value, or null while the coordinator has not said. */
export function servedModelOf(stats: ClusterStats | null): ServedModel | null {
  return stats?.model_label ? { id: stats.model ?? "", label: stats.model_label, quant: stats.quant ?? "" } : null;
}
