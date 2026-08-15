// Pluggable model interface (design philosophy 14 + acceptance P5 +
// docs/multi-model-design.md §3.1): the client selects a model through this
// registry, it never hard-codes any model into the splitting/API core.
// The registry is GENERATED from the repo-root models/<id>.json manifests —
// the same single source of truth the engine-side registry
// (src/common/model.c) mirrors. Adding a model = adding a manifest (+ an
// engine backend that can run it); no UI or planner changes.
import dsv4 from "../../models/deepseek-v4-flash.json";
import qwen38b from "../../models/qwen3-8b.json";
import qwen3508b from "../../models/qwen3.5-0.8b.json";
import qwen354b from "../../models/qwen3.5-4b.json";
import qwen359b from "../../models/qwen3.5-9b.json";
import qwen3527b from "../../models/qwen3.5-27b.json";
import qwen3535ba3b from "../../models/qwen3.5-35b-a3b.json";
import glm52 from "../../models/glm-5.2.json";
import kimiK25 from "../../models/kimi-k2.5.json";
import kimiK3 from "../../models/kimi-k3.json";

// One selectable precision of a model (small-model-design.md §3.1). The
// top-level layer_weight_bytes/shared_weight_bytes mirror the default variant,
// so quant-unaware code (toSpec/capacity) keeps working unchanged.
export interface ModelVariant {
  quant: string; // "Q4_K_M" .. "BF16"; matches the engine ASSIGN_PLAN.quant
  layer_weight_bytes: number;
  shared_weight_bytes: number;
  repo: string;
  gguf: string;
}

// Shape of a models/<id>.json manifest (planning-time metadata; the GGUF is
// runtime truth — the engine cross-checks on load).
export interface ModelManifest {
  id: string;
  label: string;
  family: string;
  backend: "ds4" | "ds4x";
  arch: string;
  available: boolean;
  // "cluster" = may be spread over a homogeneous LAN cluster; "single-node" =
  // served by one machine, and the coordinator REFUSES --num-workers > 1 for it
  // (src/common/model.c idletoken_model_may_cluster). Mirrors the engine
  // registry; model_manifest_check.py fails the build if the two disagree.
  deployment: "single-node" | "cluster";
  license: string;
  params_summary: string;
  n_layers: number;
  n_embd: number;
  hc_streams: number;
  n_vocab: number;
  layer_weight_bytes: number;
  shared_weight_bytes: number;
  context_max: number;
  split: { boundary_multiple: number };
  kv: { kind: string; bytes_per_token_per_layer: number };
  overhead_base_bytes: number;
  default_gguf: string;
  chat_template: string;
  sources: { repo: string; quant: string }[];
  // Small dense models (GQA) carry a precision menu; large MLA-MoE models omit
  // these (single implicit variant = the top-level scalars).
  attn?: { kind: string; n_head: number; n_head_kv: number; head_dim: number; qk_norm: boolean };
  rope_theta?: number;
  default_quant?: string;
  variants?: ModelVariant[];
  note?: string;
}

export interface ModelSpec {
  id: string;
  label: string;
  params: string; // human summary, e.g. "304B · 13B active"
  totalLayers: number;
  approxWeightsBytes: number;
  available: boolean; // false = shown greyed out, backend not implemented yet
  backend: "ds4" | "ds4x";
  contextMax: number;
  singleNode: boolean; // served by one machine; clustering it is refused
  note?: string;
}

// WARNING: this list must agree with the engine registry (src/common/model.c).
// On 2026-08-08 it turned out to be missing four models: the engine could run
// qwen3.5-4b/9b/27b/35b-a3b, but they were absent from the client's dropdown, so
// users could not select them even though `idletoken-worker --advise` listed them
// as runnable. That is this repo's recurring "two hand-maintained copies drift"
// problem, except this time the client's copy drifted
// (model_manifest_check.py compares models/*.json against model.c and does not
// cover this file).
const MANIFESTS = [
  dsv4,
  qwen3508b, qwen354b, qwen38b, qwen359b, qwen3527b, qwen3535ba3b,
  glm52, kimiK25, kimiK3,
] as ModelManifest[];

function toSpec(m: ModelManifest): ModelSpec {
  return {
    id: m.id,
    label: m.label,
    params: m.params_summary,
    totalLayers: m.n_layers,
    approxWeightsBytes: m.layer_weight_bytes + m.shared_weight_bytes,
    available: m.available,
    backend: m.backend,
    contextMax: m.context_max,
    singleNode: m.deployment !== "cluster",
    note: m.note,
  };
}

export const MODELS: ModelSpec[] = MANIFESTS.map(toSpec);

/**
 * The models a user can actually pick.
 *
 * `MODELS` still holds every manifest, because `getModel`/`getManifest` must be
 * able to resolve an id that is no longer offered (a stored setting, a cluster
 * reporting what it serves). But nothing user-facing should list a model the
 * engine cannot run: a greyed-out row with a "coming soon" badge is a promise
 * in the settings panel, and it pushed the four models that DO run below the
 * fold. Roadmap belongs in the docs, not in a picker.
 */
export const AVAILABLE_MODELS: ModelSpec[] = MODELS.filter((m) => m.available);

export const DEFAULT_MODEL_ID = "deepseek-v4-flash";

// ---- open model intake (v2 rebuild WS-D1) ----------------------------------
// The registry above is the CURATED list — default recommendations. Since the
// llama.cpp pivot (docs/v2-rebuild-plan-2026-08.md §1.6) any GGUF the engine
// can load is selectable: the user points at a local file or an HF repo+file,
// and the COORDINATOR builds the manifest from the GGUF header (WS-B4) — the
// client deliberately knows nothing about such a model beyond where it lives.
// One sentinel id marks that selection; the file/repo details ride in settings
// (customGguf*), not here, because there is no manifest to register.

/** settings.modelId value meaning "a user-supplied GGUF, not a curated model". */
export const LOCAL_GGUF_ID = "local-gguf";

/** Is this the open-intake selection (user-supplied GGUF)? */
export function isLocalGguf(id: string): boolean {
  return id === LOCAL_GGUF_ID;
}

/**
 * A display-only ModelSpec for the open-intake selection, so components that
 * render "the selected model" need no second code path. Capacity fields are
 * zero on purpose: the client has no manifest for this model, and the honest
 * fit answer is the coordinator's startup verdict (shown on the engine card),
 * not a client-side estimate built from another model's numbers.
 */
export function localGgufSpec(label: string): ModelSpec {
  return {
    id: LOCAL_GGUF_ID,
    label: label || "Local GGUF",
    params: "GGUF",
    totalLayers: 0,
    approxWeightsBytes: 0,
    available: true,
    backend: "ds4x", // unused on this path; the coordinator drives llama.cpp
    contextMax: 0,
    singleNode: true, // networked serving of open models arrives with WS-C
  };
}

/** Is this id something the engine can run today? Unknown ids are not. */
export function isAvailable(id: string): boolean {
  return MODELS.some((m) => m.id === id && m.available);
}

export function getModel(id: string): ModelSpec {
  return MODELS.find((m) => m.id === id) ?? MODELS[0];
}

export function getManifest(id: string): ModelManifest {
  return MANIFESTS.find((m) => m.id === id) ?? MANIFESTS[0];
}

/**
 * Put a human name on a GGUF file found on disk.
 *
 * Searches every manifest AND every variant, not just the models currently
 * offered: the files that pile up in the model folder are precisely the ones
 * nobody has selected in a while. Returns null for a file no manifest claims —
 * the caller shows the raw name rather than hiding it, because it is occupying
 * the disk regardless of whether we can explain it.
 */
export function describeGguf(file: string): { label: string; quant?: string } | null {
  for (const m of MANIFESTS) {
    for (const v of m.variants ?? []) {
      if (v.gguf === file) return { label: m.label, quant: v.quant };
    }
    if (m.default_gguf === file) return { label: m.label };
  }
  return null;
}

/**
 * Is this model served by a single machine?
 *
 * Small models are: they fit one node, and splitting one over a LAN spends more
 * time on pipeline round-trips than on compute. The coordinator enforces it
 * (it exits rather than accept a second worker), so the UI must not offer a
 * path that ends in that refusal.
 */
export function isSingleNode(id: string): boolean {
  return getManifest(id).deployment !== "cluster";
}

// ---- precision (quant) selection -------------------------------------------
// Small models expose a precision menu; large MLA-MoE models have one implicit
// variant. These helpers give the UI a uniform view either way.

// True when the model offers a user-selectable precision menu.
export function hasQuantChoice(id: string): boolean {
  const v = getManifest(id).variants;
  return !!v && v.length > 1;
}

// The default precision for a model ("" when it has no explicit menu).
export function defaultQuant(id: string): string {
  const m = getManifest(id);
  return m.default_quant ?? m.variants?.[0]?.quant ?? "";
}

// The precision menu (empty when the model has no explicit variants).
export function quantOptions(id: string): ModelVariant[] {
  return getManifest(id).variants ?? [];
}

// Resolve a variant by quant name, falling back to the default. Returns
// undefined only for models with no variant table (use the ModelSpec scalars).
export function getVariant(id: string, quant?: string): ModelVariant | undefined {
  const vs = getManifest(id).variants;
  if (!vs || vs.length === 0) return undefined;
  if (quant) {
    const hit = vs.find((v) => v.quant === quant);
    if (hit) return hit;
  }
  const def = defaultQuant(id);
  return vs.find((v) => v.quant === def) ?? vs[0];
}

// Whole-model weight bytes at the selected precision (variant if present, else
// the ModelSpec scalar). Drives the single-node-vs-cluster judgment.
export function weightsBytesForQuant(model: ModelSpec, quant?: string): number {
  const v = getVariant(model.id, quant);
  return v ? v.layer_weight_bytes + v.shared_weight_bytes : model.approxWeightsBytes;
}

// How many of a model's layers a node with `usableVramBytes` could hold.
// A capacity *estimate* for the UI; the real split comes from the coordinator's
// resource-proportional planner after networking. Clamped to [0, totalLayers].
export function estimateHostableLayers(usableVramBytes: number, model: ModelSpec): number {
  if (usableVramBytes <= 0 || model.totalLayers <= 0) return 0;
  const perLayer = model.approxWeightsBytes / model.totalLayers;
  const n = Math.floor(usableVramBytes / perLayer);
  return Math.max(0, Math.min(model.totalLayers, n));
}

// ---- cluster capacity guidance ---------------------------------------------
// Answers the question the dashboard must answer BEFORE pairing: "is my
// hardware enough, and if not, how far off am I?" Mirrors the engine's
// estimate shape (src/common/model.c idletoken_model_overhead + plan.c
// idletoken_needed_bytes), driven by the model's manifest: DSv4 uses the
// per-tier table, MLA-KV models (GLM/Kimi) use base + bytes/token/layer.
// Same caveat as engine-side: ESTIMATES pending real-machine calibration.
const GiB = 1024 ** 3;

function overheadBytes(m: ModelManifest, ctx: number, layersOnNode: number): number {
  if (m.kv.kind === "mla") {
    const kv = m.kv.bytes_per_token_per_layer * ctx * Math.max(1, layersOnNode);
    return (m.overhead_base_bytes + kv) * 1.1; // +10% margin, engine parity
  }
  // dsv4: calibrated per-tier table (docs/architecture.md §5)
  if (ctx <= 8192) return 1.5 * GiB;
  if (ctx <= 32768) return 2 * GiB;
  if (ctx <= 131072) return 3 * GiB;
  if (ctx <= 524288) return 6 * GiB;
  return 9 * GiB;
}

export interface CapacityEstimate {
  needBytes: number; // whole-cluster requirement at this tier / node count
  haveBytes: number; // this machine's usable contribution (VRAM+RAM aware)
  gapBytes: number; // max(0, need - have)
  hostableLayers: number; // layers THIS machine could hold (incl. RAM offload)
  // Node count the estimate was actually computed for. Equals the `nNodes`
  // argument for cluster models and is always 1 for single-node ones — the UI
  // must quote THIS number, not what it asked for, or it would tell the user
  // "needs 5 GB across 3 machines" about a model the engine will only ever run
  // on one.
  nodes: number;
}

// `nNodes`: the known cluster size when paired; pass the nominal typical
// cluster (e.g. 3) when standalone — the per-node overhead term is small, so
// the guidance stays honest either way.
export function estimateClusterCapacity(
  model: ModelSpec,
  mem: { vram_usable: number; ram_usable: number; unified_memory: boolean },
  ctx: number,
  nNodes: number,
  quant?: string // selected precision; changes the weight bytes → feasibility
): CapacityEstimate {
  const man = getManifest(model.id);
  // A single-node model is sized for ONE machine no matter how many are
  // paired — pooling their memory would promise a configuration the
  // coordinator refuses to start (engine parity: src/common/advise.c).
  const n = man.deployment === "cluster" ? Math.max(1, nNodes) : 1;
  const avgLayers = man.n_layers > 0 ? Math.ceil(man.n_layers / n) : 1;
  const overhead = overheadBytes(man, ctx, avgLayers);
  // Size the SELECTED precision (falls back to the manifest scalars when the
  // model has no variant menu) so the guidance tracks the quant dropdown.
  const v = getVariant(model.id, quant);
  const layerBytes = v ? v.layer_weight_bytes : man.layer_weight_bytes;
  const sharedBytes = v ? v.shared_weight_bytes : man.shared_weight_bytes;
  const needBytes = layerBytes + n * (sharedBytes + overhead);
  // Unified memory is one physical pool — never count it twice (plan.c rule).
  const haveBytes = mem.unified_memory
    ? Math.max(mem.vram_usable, mem.ram_usable)
    : mem.vram_usable + mem.ram_usable;
  const perLayer = man.n_layers > 0 ? layerBytes / man.n_layers : 0;
  const usableForLayers = haveBytes - sharedBytes - overhead;
  const hostableLayers =
    perLayer > 0
      ? Math.max(0, Math.min(man.n_layers, Math.floor(usableForLayers / perLayer)))
      : 0;
  return {
    needBytes,
    haveBytes,
    gapBytes: Math.max(0, needBytes - haveBytes),
    hostableLayers,
    nodes: n,
  };
}

/**
 * Pick a default model on first start that **this machine can actually run**.
 *
 * The default used to be hardcoded to `deepseek-v4-flash` (80.76 GiB). The first
 * thing a new user with an 8 GB card saw after installing was "this model and
 * precision do not fit on one machine -- please build a cluster", with exactly
 * one machine to their name. The product has six models that run on a single
 * machine, and the first screen led with one that could not.
 *
 * How it picks: among `available` models, try them from largest weights down and
 * take the first that **fits in VRAM** (nNodes=1, comparing vram_usable only).
 * Usage comes from `estimateClusterCapacity`'s needBytes, **the same source as
 * the capability panel and the planner** -- no second estimator.
 *
 * WARNING: the test must be GPU_ONLY, not that function's `gapBytes` -- its
 * "have" is VRAM+RAM, i.e. HYBRID, where weights spill into host memory. The
 * first version did exactly that and ended up recommending 27B for an 8 GB card
 * and 9B for a 4 GB one: technically "it fits", measured at **0.77 tok/s**.
 * Recommending a configuration slow enough to make someone uninstall is far
 * worse than recommending a small model.
 *
 * When nothing fits in VRAM it returns the **smallest** one: the capability panel
 * still explains how much is missing, and that beats leading with a 304B model,
 * which only makes people think the software is not for them.
 */
export function pickBestFittingModel(
  mem: { vram_usable: number; ram_usable: number; unified_memory: boolean },
  ctx: number
): { modelId: string; quant: string } {
  const usable = MODELS.filter((m) => getManifest(m.id).available);
  if (usable.length === 0) return { modelId: DEFAULT_MODEL_ID, quant: defaultQuant(DEFAULT_MODEL_ID) };
  const bytes = (m: ModelSpec) => weightsBytesForQuant(m, defaultQuant(m.id));
  const desc = [...usable].sort((a, b) => bytes(b) - bytes(a));
  // On unified memory (DGX/Grace) VRAM and RAM are one pool, and vram_usable
  // already represents it.
  const vram = mem.vram_usable;
  for (const m of desc) {
    const q = defaultQuant(m.id);
    if (estimateClusterCapacity(m, mem, ctx, 1, q).needBytes <= vram) {
      return { modelId: m.id, quant: q };
    }
  }
  const smallest = desc[desc.length - 1];
  return { modelId: smallest.id, quant: defaultQuant(smallest.id) };
}
