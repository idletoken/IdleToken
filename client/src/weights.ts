// Weight fetching, front-end half -- the UI side of B1.
//
// Division of labour: **manifest parsing here, downloading in Rust**
// (src-tauri/src/weights.rs). The front end resolves the repo and file name
// because it already imports models/*.json; writing a second model registry in
// Rust would be a third hand-maintained copy, and this repo has already been
// bitten by copies drifting apart (scripts/model_manifest_check.py exists for
// exactly that reason).
//
// The resolution rules correspond line for line with the python block in
// scripts/model_fetch.sh -- both sides must select the same file, or the script
// and the client would download two different sets of weights, with symptoms
// appearing only at load time.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ModelManifest } from "./models";
import { inTauri } from "./platform";

export interface DownloadTarget {
  repo: string;
  file: string;
  /** Byte count declared by the manifest (layers + shared). 0 = not stated, in
   *  which case the server's value wins. */
  expectBytes: number;
}

/**
 * Select the file to download from a manifest plus a precision.
 *
 * Same precedence as model_fetch.sh: an explicit quant, then default_quant, then
 * the first variant. Large models without variants (the MLA-MoE family) use the
 * top-level sources[0].repo plus default_gguf.
 */
export function resolveDownload(man: ModelManifest, quant?: string): DownloadTarget | null {
  const variants = man.variants ?? [];
  let v = null as (typeof variants)[number] | null;
  if (variants.length > 0) {
    v = (quant ? variants.find((x) => x.quant === quant) : null) ?? null;
    if (!v) {
      const dq = man.default_quant;
      v = (dq ? variants.find((x) => x.quant === dq) : null) ?? variants[0] ?? null;
    }
  }
  const repo = v?.repo ?? man.sources?.[0]?.repo ?? "";
  const file = v?.gguf ?? man.default_gguf ?? "";
  if (!repo || !file) return null;
  const layer = v?.layer_weight_bytes ?? man.layer_weight_bytes ?? 0;
  const shared = v?.shared_weight_bytes ?? man.shared_weight_bytes ?? 0;
  return { repo, file, expectBytes: layer + shared };
}

export interface WeightsState {
  path: string;
  complete: boolean;
  have_bytes: number;
}

export async function defaultModelDir(): Promise<string> {
  return invoke<string>("weights_default_dir");
}

export async function weightsState(
  destDir: string,
  file: string,
  expectBytes: number,
): Promise<WeightsState> {
  return invoke<WeightsState>("weights_state", { destDir, file, expectBytes });
}

export interface FetchProgress {
  id: string;
  // "cancelled" is distinct from "error" on purpose: the user asking to stop is
  // not a failure, and reporting it as one made the client apologise for doing
  // what it was told (see weights.rs).
  kind: "probe" | "progress" | "done" | "error" | "cancelled";
  endpoint?: string;
  have?: number;
  total?: number;
  mirror?: boolean;
  note?: string;
  path?: string;
  message?: string;
}

/** Subscribe to download progress. Returns the unsubscribe function. */
export async function onFetchProgress(cb: (p: FetchProgress) => void): Promise<() => void> {
  const un = await listen<FetchProgress>("weights-fetch", (e) => cb(e.payload));
  return () => un();
}

/**
 * Download weights. Leaving `endpoints` empty tries huggingface.co first and
 * then hf-mirror.com; passing endpoints uses only those (mirroring the script's
 * HF_ENDPOINT: when the user names one, do not silently try elsewhere).
 */
export async function fetchWeights(args: {
  id: string;
  target: DownloadTarget;
  destDir: string;
  endpoints?: string[];
}): Promise<void> {
  return invoke("weights_fetch", {
    id: args.id,
    repo: args.target.repo,
    file: args.target.file,
    destDir: args.destDir,
    expectBytes: args.target.expectBytes,
    endpoints: args.endpoints ?? [],
  });
}

export async function cancelFetch(id: string): Promise<boolean> {
  return invoke<boolean>("weights_cancel", { id });
}

/**
 * "Where is this machine's copy of the weights" -- called by the startup flow.
 *
 * This is what B2 fixed. `weightsSource: "auto"` used to pass **an empty string**
 * to the engine in every case, and on the engine side an empty string takes the
 * mock branch -- which no longer falls back automatically (worker_main.c requires
 * IDLETOKEN_ALLOW_MOCK). So a freshly installed client was guaranteed to fail to
 * load once a model was selected.
 *
 * "auto" now has three real paths, in order:
 *   1. A complete local copy already exists (downloaded by the script, or
 *      finished last time) -> use it directly.
 *   2. Joining someone else's cluster -> still pass an empty string and let the
 *      coordinator's shard service feed the layers (an existing capability; this
 *      path was always correct, it was just the only one).
 *   3. Acting as the coordinator, or running standalone -> the weights must be
 *      downloaded first, and the caller initiates that.
 */
export async function resolveLocalWeights(args: {
  weightsSource: "auto" | "local";
  ggufPath: string;
  modelDir: string;
  manifest: ModelManifest;
  quant?: string;
}): Promise<{ path: string; needsDownload: boolean; target: DownloadTarget | null }> {
  // Browser dev build: no engine, no filesystem, so every call below throws on
  // the missing Tauri bridge — which blocked "run on this machine alone", a
  // headline mode, from being reachable in the dev sim at all. Report the
  // weights as present. inTauri() is false ONLY under `vite dev`/`preview`;
  // the packaged app always takes the real path below.
  if (!inTauri()) return { path: "(dev-sim)", needsDownload: false, target: null };
  if (args.weightsSource === "local") {
    return { path: args.ggufPath, needsDownload: false, target: null };
  }
  const target = resolveDownload(args.manifest, args.quant);
  if (!target) return { path: "", needsDownload: false, target: null };
  const dir = args.modelDir || (await defaultModelDir());
  const st = await weightsState(dir, target.file, target.expectBytes);
  return { path: st.complete ? st.path : "", needsDownload: !st.complete, target };
}
