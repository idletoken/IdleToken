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
  /** Which endpoint is serving the bytes. Diagnostics only — the UI
   *  deliberately never shows where a download comes from. */
  endpoint?: string;
  have?: number;
  total?: number;
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
 * The user stopped this download. Thrown instead of a plain error so callers
 * can tell "you asked me to stop" from "it broke" — the two look identical on
 * the wire (the command rejects either way, because in both cases there are no
 * usable weights and the caller must not carry on), and treating the first as
 * the second is what made the client report **"Download failed — cancelled"**
 * seconds after the user pressed Cancel.
 */
export class WeightsCancelled extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WeightsCancelled";
  }
}

export function isWeightsCancelled(e: unknown): boolean {
  return e instanceof WeightsCancelled;
}

/** Ids whose cancel WE asked for, so the rejection that follows can be
 *  classified as intent rather than failure.
 *
 *  Kept here rather than in the caller because of an ordering trap: the engine
 *  emits its `cancelled` progress event and returns the command error at
 *  roughly the same moment, and a caller that clears its own flag on the event
 *  is no longer holding it when the rejection lands. Nothing clears this set
 *  except the rejection it exists to explain. */
const cancelRequested = new Set<string>();

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
  cancelRequested.delete(args.id);
  try {
    await invoke("weights_fetch", {
      id: args.id,
      repo: args.target.repo,
      file: args.target.file,
      destDir: args.destDir,
      expectBytes: args.target.expectBytes,
      endpoints: args.endpoints ?? [],
    });
  } catch (e) {
    if (cancelRequested.delete(args.id)) throw new WeightsCancelled(String(e));
    throw e;
  }
}

/** A weight file on this machine, as found by scanning the model folder. */
export interface StoredWeights {
  /** File name without `.part` — the name a manifest would use. */
  file: string;
  bytes: number;
  /** An unfinished download; continuing resumes from these bytes. */
  partial: boolean;
}

/** Everything in the model folder, largest first. */
export async function listWeights(destDir: string): Promise<StoredWeights[]> {
  return invoke<StoredWeights[]>("weights_list", { destDir });
}

/** Delete one model's weights (finished file plus any leftover `.part`).
 *  Returns the bytes freed. */
export async function deleteWeights(destDir: string, file: string): Promise<number> {
  return invoke<number>("weights_delete", { destDir, file });
}

export async function cancelFetch(id: string): Promise<boolean> {
  cancelRequested.add(id);
  return invoke<boolean>("weights_cancel", { id });
}

// ---- open model intake (v2 WS-D1) ------------------------------------------
// A user-supplied GGUF instead of a curated manifest. Two sources, one result
// shape (the same as resolveLocalWeights, so the serve flow does not fork):
//   file -> the picked path IS the weights; verified to still exist.
//   hf   -> repo + exact file name, through the SAME download machinery as
//           curated weights (weights_fetch takes repo/file/expectBytes and is
//           manifest-agnostic; expectBytes 0 = the server's Content-Length is
//           authoritative).

export interface CustomModelSource {
  source: "file" | "hf";
  path: string; // file source: absolute GGUF path
  repo: string; // hf source
  file: string; // hf source: exact .gguf name
}

/** The file name a custom selection would load — for display and for scanning
 *  the model folder. */
export function customGgufName(c: CustomModelSource): string {
  if (c.source === "hf") return c.file;
  const cut = Math.max(c.path.lastIndexOf("/"), c.path.lastIndexOf("\\"));
  return cut >= 0 ? c.path.slice(cut + 1) : c.path;
}

export async function resolveCustomWeights(args: {
  modelDir: string;
  custom: CustomModelSource;
}): Promise<{ path: string; needsDownload: boolean; target: DownloadTarget | null; haveBytes: number }> {
  if (!inTauri()) return { path: "(dev-sim)", needsDownload: false, target: null, haveBytes: 0 };
  const c = args.custom;
  if (c.source === "file") {
    // The dialog guaranteed existence when it was picked; re-verify now because
    // files get moved/deleted, and handing a stale path to the engine turns a
    // one-sentence answer here into a load-time failure there.
    const cut = Math.max(c.path.lastIndexOf("/"), c.path.lastIndexOf("\\"));
    const dir = cut > 0 ? c.path.slice(0, cut) : "";
    const file = cut >= 0 ? c.path.slice(cut + 1) : c.path;
    if (!dir || !file) return { path: "", needsDownload: false, target: null, haveBytes: 0 };
    const st = await weightsState(dir, file, 0);
    return {
      path: st.complete ? c.path : "",
      needsDownload: false, // a fixed local path is not something we can fetch
      target: null,
      haveBytes: 0,
    };
  }
  const target: DownloadTarget = { repo: c.repo, file: c.file, expectBytes: 0 };
  const dir = args.modelDir || (await defaultModelDir());
  const st = await weightsState(dir, c.file, 0);
  return {
    path: st.complete ? st.path : "",
    needsDownload: !st.complete,
    target,
    haveBytes: st.complete ? 0 : st.have_bytes,
  };
}

/**
 * "Where is this machine's copy of the weights" -- called by the startup flow.
 *
 * This is what B2 fixed. The client used to pass **an empty string** to the
 * engine in every case, and on the engine side an empty string takes the mock
 * branch -- which no longer falls back automatically (worker_main.c requires
 * IDLETOKEN_ALLOW_MOCK). So a freshly installed client was guaranteed to fail to
 * load once a model was selected.
 *
 * There is no "weights source" setting any more (2026-08-13): this one function
 * IS the policy, and it has three real paths, in order:
 *   1. A complete local copy already exists (downloaded by the script, or
 *      finished last time) -> use it directly.
 *   2. Joining someone else's cluster -> still pass an empty string and let the
 *      coordinator's shard service feed the layers (an existing capability; this
 *      path was always correct, it was just the only one).
 *   3. Acting as the coordinator, or running standalone -> the weights must be
 *      downloaded first, and the caller initiates that.
 */
export async function resolveLocalWeights(args: {
  modelDir: string;
  manifest: ModelManifest;
  quant?: string;
}): Promise<{
  path: string;
  needsDownload: boolean;
  target: DownloadTarget | null;
  /** Bytes of an unfinished copy already on disk (the `.part`). The next
   *  attempt resumes from here, so the UI can say so instead of implying the
   *  whole file has to come down again. 0 = nothing to resume. */
  haveBytes: number;
}> {
  // Browser dev build: no engine, no filesystem, so every call below throws on
  // the missing Tauri bridge — which blocked "run on this machine alone", a
  // headline mode, from being reachable in the dev sim at all. Report the
  // weights as present. inTauri() is false ONLY under `vite dev`/`preview`;
  // the packaged app always takes the real path below.
  if (!inTauri()) return { path: "(dev-sim)", needsDownload: false, target: null, haveBytes: 0 };
  const target = resolveDownload(args.manifest, args.quant);
  if (!target) return { path: "", needsDownload: false, target: null, haveBytes: 0 };
  const dir = args.modelDir || (await defaultModelDir());
  const st = await weightsState(dir, target.file, target.expectBytes);
  return {
    path: st.complete ? st.path : "",
    needsDownload: !st.complete,
    target,
    haveBytes: st.complete ? 0 : st.have_bytes,
  };
}
