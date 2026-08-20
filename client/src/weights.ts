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
import type { ModelManifest, SplitPart } from "./models";
import { inTauri } from "./platform";

export interface DownloadTarget {
  repo: string;
  file: string;
  /** Byte count declared by the manifest (layers + shared). 0 = not stated, in
   *  which case the server's value wins. */
  expectBytes: number;
  /** SHA-256 pinned by the manifest. "" = not pinned (curation gap): the
   *  integrity gate then has nothing to check — it never invents a hash. */
  sha256: string;
  /** HF repo commit to download from. "" = not pinned; the engine then falls
   *  back to the moving branch. */
  revision: string;
  /** Split GGUF: the parts after `file`. Empty = single-file. The engine is
   *  handed `file`; every part must nonetheless land complete and verified. */
  parts: SplitPart[];
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
  // The hash follows the same precedence as the file it certifies: a selected
  // variant's hash never falls back to the top-level one (that would verify
  // file A against file B's hash and reject a perfectly good download).
  const sha256 = (v ? v.sha256 : man.sha256) ?? "";
  const revision = (v ? v.revision : man.revision) ?? "";
  const parts = (v ? v.parts : man.parts) ?? [];
  // The declared size must cover the WHOLE model, parts included — otherwise a
  // 434 GB download would report "done" at part one's 36 GB.
  const partBytes = parts.reduce((n: number, p: SplitPart) => n + (p.bytes ?? 0), 0);
  return { repo, file, expectBytes: layer + shared + partBytes, sha256, revision, parts };
}

export interface WeightsState {
  path: string;
  complete: boolean;
  have_bytes: number;
  /** Passed the SHA-256 gate (or nothing was pinned). `complete && !verified`
   *  means: run verifyWeights before serving this file. */
  verified: boolean;
}

export async function defaultModelDir(): Promise<string> {
  return invoke<string>("weights_default_dir");
}

export async function weightsState(
  destDir: string,
  file: string,
  expectBytes: number,
  expectSha256: string,
): Promise<WeightsState> {
  return invoke<WeightsState>("weights_state", { destDir, file, expectBytes, expectSha256 });
}

/**
 * Verify an already-downloaded file against the manifest hash (the serve-path
 * gate for files that came from model_fetch.sh or predate the gate). Slow on
 * big files — progress arrives on the same weights-fetch channel under `id`.
 * A mismatch DELETES the file and rejects: red, no automatic retry (decision
 * 2026-08-15); recovery is a fresh download.
 */
export async function verifyWeights(args: {
  id: string;
  destDir: string;
  file: string;
  sha256: string;
}): Promise<void> {
  await invoke("weights_verify", {
    id: args.id,
    destDir: args.destDir,
    file: args.file,
    expectSha256: args.sha256,
  });
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
      expectSha256: args.target.sha256,
      revision: args.target.revision,
      parts: args.target.parts,
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
  /** Complete on disk but never checked against the manifest hash (script
   *  download, or a client from before the gate). The serve path must run
   *  verifyWeights first — using the file anyway would make the hash in the
   *  manifest decorative. */
  needsVerify: boolean;
}> {
  // Browser dev build: no engine, no filesystem, so every call below throws on
  // the missing Tauri bridge — which blocked "run on this machine alone", a
  // headline mode, from being reachable in the dev sim at all. Report the
  // weights as present. inTauri() is false ONLY under `vite dev`/`preview`;
  // the packaged app always takes the real path below.
  if (!inTauri()) return { path: "(dev-sim)", needsDownload: false, target: null, haveBytes: 0, needsVerify: false };
  const target = resolveDownload(args.manifest, args.quant);
  if (!target) return { path: "", needsDownload: false, target: null, haveBytes: 0, needsVerify: false };
  const dir = args.modelDir || (await defaultModelDir());
  const st = await weightsState(dir, target.file, target.expectBytes, target.sha256);
  return {
    path: st.complete ? st.path : "",
    needsDownload: !st.complete,
    target,
    haveBytes: st.complete ? 0 : st.have_bytes,
    needsVerify: st.complete && !st.verified,
  };
}
