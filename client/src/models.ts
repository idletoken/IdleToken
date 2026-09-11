// Pluggable model interface (design philosophy 14 + acceptance P5 +
// docs/multi-model-design.md §3.1): the client selects a model through this
// registry, it never hard-codes any model into the splitting/API core.
// The registry is GENERATED from the repo-root models/<id>.json manifests —
// the same single source of truth the engine-side registry
// (src/common/model.c) mirrors. Adding a model = adding a manifest (+ an
// engine backend that can run it); no UI or planner changes.
import dsv4 from "../../models/deepseek-v4-flash.json";
import dsv4pro from "../../models/deepseek-v4-pro.json";
import qwen38b from "../../models/qwen3-8b.json";
import qwen3508b from "../../models/qwen3.5-0.8b.json";
import qwen352b from "../../models/qwen3.5-2b.json";
import qwen354b from "../../models/qwen3.5-4b.json";
import qwen359b from "../../models/qwen3.5-9b.json";
import qwen3527b from "../../models/qwen3.5-27b.json";
import qwen3827b from "../../models/qwen3.8-27b.json";
import qwen3535ba3b from "../../models/qwen3.5-35b-a3b.json";
import qwen35122ba10b from "../../models/qwen3.5-122b-a10b.json";
import qwen35397ba17b from "../../models/qwen3.5-397b-a17b.json";
import glm52 from "../../models/glm-5.2.json";
import kimiK25 from "../../models/kimi-k2.5.json";
import gptOss20b from "../../models/gpt-oss-20b.json";
import gptOss120b from "../../models/gpt-oss-120b.json";

// One selectable precision of a model (small-model-design.md §3.1). The
// top-level layer_weight_bytes/shared_weight_bytes mirror the default variant,
// so quant-unaware code (toSpec/capacity) keeps working unchanged.
export interface ModelVariant {
  quant: string; // "Q4_K_M" .. "BF16"; matches the engine ASSIGN_PLAN.quant
  layer_weight_bytes: number;
  shared_weight_bytes: number;
  repo: string;
  gguf: string;
  // Content hash of the GGUF (integrity gate; scripts/manifest_sha256.py fills
  // it from the HF API). Absent only during curation — a shipped variant
  // without a hash downloads unverified, which the gate treats as "nothing to
  // verify", not as failure.
  sha256?: string;
  // HF repo commit the hash was observed at. Downloads resolve against it, so
  // a force-push cannot swap the bytes under the same name. Absent = use the
  // moving branch (`main`).
  revision?: string;
  /** Split GGUF (2026-08-15): the parts AFTER `gguf`, in repo order. Every
   *  model past ~50 GB ships this way on HF (`*-00002-of-00006.gguf` …), so a
   *  precision is "present" only when all of its parts are. `gguf` stays the
   *  first part — the one the engine is handed; llama.cpp finds the rest by
   *  name in the same directory. Absent/empty = a single-file precision. */
  parts?: SplitPart[];
}

/** One extra file of a split GGUF. `bytes` is that part's own size (0 =
 *  unknown, the server's length decides); `sha256` gates that part alone. */
export interface SplitPart {
  file: string;
  bytes?: number;
  sha256?: string;
}

// Shape of a models/<id>.json manifest (planning-time metadata; the GGUF is
// runtime truth — the engine cross-checks on load).
export interface ModelManifest {
  id: string;
  label: string;
  family: string;
  backend: "ds4" | "ds4x" | "llamacpp";
  arch: string;
  available: boolean;
  // Technical placement capability, not the default. "cluster" means the
  // model may run either locally or across machines; a future "single-node"
  // value is reserved for a backend that genuinely cannot be split. The UI's
  // default is decided from the selected precision and current hardware.
  deployment: "single-node" | "cluster";
  license: string;
  /** False keeps a locally runnable model out of the paid service-sharing
   * marketplace when its weight licence does not permit that use. Missing is
   * the backwards-compatible true value. The gateway enforces the actual
   * fail-closed catalogue gate; the client keeps the metadata for display. */
  marketplace_allowed?: boolean;
  params_summary: string;
  n_layers: number;
  n_embd: number;
  hc_streams: number;
  n_vocab: number;
  layer_weight_bytes: number;
  shared_weight_bytes: number;
  context_max: number;
  /** Curated/validated YaRN-extended window (Qwen publishes 4x configs);
   *  absent/0 = the trained window is the hard ceiling. */
  context_yarn_max?: number;
  /** MEASURED graph workspace at the two product context tiers, in bytes,
   *  from llama.cpp's own no_alloc dry-run (scripts/measure_model_memory.sh).
   *  Mirrors idletoken_model_spec.compute_bytes_* one for one; absent/0 means
   *  "not measured", which the estimate must surface rather than treat as 0 —
   *  the old closed form was 9.6x low on one curated model at 256K.
   *
   *  Per BACKEND: identical across GPUs and WEIGHT quantizations, but GLM-5.2
   *  at 256K is 1.50 GiB on CUDA and 33.3 GiB on Metal
   *  (results/memory-need-measured-20260901.md).
   *
   *  Per KV TIER (2026-09-02): one entry per KV cache dtype, in KV_TIERS order
   *  [f16, q8_0, q4_0] — the same order as idletoken_kv_tier in C, because both
   *  sides index these arrays. The workspace is not KV-dtype independent on
   *  every architecture: Qwen3.5-0.8B at 256K on Metal measures 489.00 MiB with
   *  an f16 cache and 745.28 MiB with a quantized one.
   *
   *  Per CONTEXT TIER (2026-09-02): three product windows, 128K the default
   *  (docs/ctx-tiers-2026-09.md). NOT interpolatable — the low tiers are
   *  dominated by an n_ubatch floor and only the high end is linear in ctx. */
  compute_bytes_128k_cuda?: number[];
  compute_bytes_256k_cuda?: number[];
  compute_bytes_1m_cuda?: number[];
  compute_bytes_128k_metal?: number[];
  compute_bytes_256k_metal?: number[];
  compute_bytes_1m_metal?: number[];
  split: { boundary_multiple: number };
  kv: {
    kind: string;
    bytes_per_token_per_layer: number;
    state_bytes_per_layer?: number;
    full_attention_interval?: number;
    /** DeepSeek4 whole-model f16 cache geometry from the GGUF plus the pinned
     * engine's padded raw/CSA/HCA allocation. */
    raw_bytes_per_cell?: number;
    csa_bytes_per_cell?: number;
    hca_bytes_per_cell?: number;
    fixed_bytes_per_sequence?: number;
  };
  moe?: { n_expert: number; n_expert_used: number };
  overhead_base_bytes: number;
  default_gguf: string;
  // Content hash of default_gguf for models WITHOUT a variants table (the
  // large MLA-MoE family). Variant models carry the hash per variant instead.
  sha256?: string;
  revision?: string;
  /** Split GGUF parts for `default_gguf` (models without a variants table). */
  parts?: SplitPart[];
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
  /** Manifest family ("qwen3.5", "kimi", …) — the raw grouping key. */
  family: string;
  params: string; // human summary, e.g. "304B · 13B active"
  totalLayers: number;
  approxWeightsBytes: number;
  available: boolean; // false = shown greyed out; backend or validation is not ready
  backend: "ds4" | "ds4x" | "llamacpp";
  contextMax: number;
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
  dsv4, dsv4pro,
  qwen3508b, qwen352b, qwen354b, qwen359b, qwen3527b, qwen3535ba3b,
  qwen35122ba10b, qwen35397ba17b, qwen38b, qwen3827b,
  glm52, kimiK25, gptOss20b, gptOss120b,
] as ModelManifest[];

/** Parameter counts only — "304B · 13B active", nothing else. The manifest's
 *  params_summary also carries layer counts (rendered separately by every list
 *  that wants them), architecture jargon ("hybrid linear attention", "GQA",
 *  "KDA+MLA") and quantized sizes; none of that helps a home user pick a
 *  model, and it made the list read like a spec sheet (2026-08-15). Segments
 *  are kept iff they lead with a parameter count (12B / 2.8T), so "total /
 *  active" MoE splits survive and everything else drops.
 *
 *  `label` suppresses the whole thing when the NAME already carries the size
 *  (2026-08-16): every Qwen is named after its parameter count, so the row read
 *  "Qwen3 8B … 8.2B" and "Qwen3.5 35B-A3B … 35B total / 3B active" — the same
 *  fact twice, the second time in more words. Models whose name says nothing
 *  about size (DeepSeek V4 Flash, GLM-5.2, Kimi K2.5) keep it, because there it
 *  is the only place the number appears. */
function paramsOnly(summary: string, label?: string): string {
  const kept = summary
    .split("·")
    .map((seg) => seg.replace(/\bdense\b/, "").replace(/\s+/g, " ").trim())
    .filter((seg) => /^[\d.]+\s*[BT]\b/.test(seg));
  if (kept.length === 0) return summary;
  // Redundant with the name? Compare the leading size token only, ignoring
  // separators and case: "35B" is in "Qwen3.5 35B-A3B", and "8.2B" counts as
  // covered by "Qwen3 8B" — the tenth of a billion is not what anyone reads
  // that column for.
  const lead = kept[0].match(/^([\d.]+)\s*([BT])\b/);
  if (label && lead) {
    const norm = label.toLowerCase().replace(/[\s\-_.]/g, "");
    const n = lead[1];
    const unit = lead[2].toLowerCase();
    const forms = new Set([n + unit, Math.round(Number(n)) + unit]);
    for (const f of forms) if (norm.includes(f.replace(/[.]/g, ""))) return "";
  }
  return kept.join(" · ");
}

function toSpec(m: ModelManifest): ModelSpec {
  return {
    id: m.id,
    label: m.label,
    family: m.family,
    params: paramsOnly(m.params_summary, m.label),
    totalLayers: m.n_layers,
    approxWeightsBytes: m.layer_weight_bytes + m.shared_weight_bytes,
    available: m.available,
    backend: m.backend,
    contextMax: Math.max(m.context_max, m.context_yarn_max || 0),
    note: m.note,
  };
}

export const MODELS: ModelSpec[] = MANIFESTS.map(toSpec);

/**
 * Families withdrawn from the client (product decision). 2026-08-25: the
 * original Qwen3 generation. Qwen3.5 was briefly withdrawn on 2026-09-02 and
 * restored on 2026-09-05; the complete eight-model family is selectable.
 *
 * Delisted, NOT deleted: the manifests stay imported and the engine registry is
 * untouched, so these can come back by removing a string from the set below.
 * (Contrast qwen3.8-2.4t-a95b, removed outright on 2026-09-02 — its shipped
 * quant does not load on the pinned engine and it is not coming back.)
 * The manifests stay imported on purpose — weights already on disk keep their
 * names (describeGguf) and a stored selection still resolves via getManifest —
 * but nothing user-facing lists them, and isAvailable() saying "no" makes the
 * settings loader migrate a stored selection of a delisted model back to the
 * default model.
 */
const DELISTED_FAMILIES = new Set(["qwen3"]);

/**
 * The models a user can actually pick.
 *
 * `MODELS` still holds every manifest, because `getModel`/`getManifest` must be
 * able to resolve an id that is no longer offered (a stored setting, a cluster
 * reporting what it serves). But nothing user-facing should list a model the
 * engine cannot run: a greyed-out row with a "coming soon" badge is a promise
 * in the settings panel, and it pushed the four models that DO run below the
 * fold. Roadmap belongs in the docs, not in a picker. Delisted families are
 * excluded here for the same reason, even though the engine CAN run them.
 */
export const AVAILABLE_MODELS: ModelSpec[] = MODELS.filter(
  (m) => m.available && !DELISTED_FAMILIES.has(m.family)
);

/**
 * Models grouped into picker cards (2026-08-15; regrouped 2026-08-21).
 *
 * Flat, the catalogue is now 16 models × up to 26 precisions — a single list
 * ran off the screen and buried the choice that actually matters first ("which
 * model?"), then second ("how big?"), then last ("how precise?").
 *
 * The grouping key is the manifest's `family`, i.e. the GENERATION, not the
 * vendor. Until 2026-08-21 every Qwen shared one card on the reasoning that
 * "qwen3" and "qwen3.5" are one vendor to a reader — but a generation is not a
 * size, and folding three of them together made the size row inside the card
 * read as one ladder when it is really three: "0.8B, 4B, 8B, 9B, 27B, 27B,
 * 35B-A3B" has two 27Bs in it that are different models a year apart. Three
 * cards, one per generation, and the sizes inside each are comparable again.
 */
/** The model's name WITHOUT its brand: "DeepSeek V4 Flash" inside the DeepSeek
 *  card is just "V4 Flash", "Qwen3.5 4B" inside Qwen is "3.5 4B". The brand is
 *  already the card's title, and repeating it on every chip is noise the eye
 *  has to filter (2026-08-15). */
export function shortModelLabel(label: string, brandLabel: string): string {
  const l = label.trim();
  const b = brandLabel.trim();
  if (b && l.toLowerCase().startsWith(b.toLowerCase())) {
    const rest = l.slice(b.length).replace(/^[\s·:-]+/, "").trim();
    if (rest) return rest;
  }
  return l;
}

export interface ModelBrand {
  /** Stable key for React lists and the open/closed state. */
  id: string;
  label: string;
  /** Smallest first — the ladder a user scans to find something that fits. */
  models: ModelSpec[];
}

// One entry per family, in the order the cards appear. Newest generation of a
// vendor first, and the vendors ordered as the list has always shown them.
// A family missing from this table still gets a card (see brandOf) — it just
// sorts to the end, which is the right default for a manifest that landed
// after this table was last edited.
const BRAND_OF: { family: string; id: string; label: string }[] = [
  { family: "qwen3.8", id: "qwen3.8", label: "Qwen3.8" },
  { family: "qwen3.5", id: "qwen3.5", label: "Qwen3.5" },
  { family: "deepseek", id: "deepseek-v4", label: "DeepSeek V4" },
  { family: "gpt-oss", id: "gpt-oss", label: "GPT-OSS" },
  { family: "kimi", id: "kimi", label: "Kimi" },
  { family: "glm", id: "glm", label: "GLM" },
];

export function brandOf(family: string): { id: string; label: string } {
  // Exact family match, NOT a prefix test. A prefix test is what made "qwen3"
  // swallow "qwen3.5" and "qwen3.8"; it would do the same to any future
  // "qwen4" vs "qwen4.5" pair, silently, the day the manifest lands.
  const hit = BRAND_OF.find((b) => b.family === family);
  // An unknown family becomes its own card rather than a catch-all "Other":
  // a new generation should appear under its own name the day its manifest
  // lands, without anybody remembering to edit this table.
  return hit ?? { id: family || "other", label: family || "Other" };
}

/** The picker's data: one card per family, models smallest-first inside it. */
export const MODEL_BRANDS: ModelBrand[] = (() => {
  const by = new Map<string, ModelBrand>();
  for (const m of AVAILABLE_MODELS) {
    const b = brandOf(m.family);
    if (!by.has(b.id)) by.set(b.id, { id: b.id, label: b.label, models: [] });
    by.get(b.id)!.models.push(m);
  }
  for (const b of by.values()) b.models.sort((x, y) => x.approxWeightsBytes - y.approxWeightsBytes);
  // Card order follows BRAND_OF, not the order the manifests happen to be
  // imported in — which is why the old single-Qwen card looked stable and the
  // split one would not have: whichever generation owned the first manifest
  // would have decided where all of Qwen sat in the list.
  const rank = (id: string) => {
    const i = BRAND_OF.findIndex((b) => b.id === id);
    return i < 0 ? BRAND_OF.length : i;
  };
  return [...by.values()].sort((a, b) => rank(a.id) - rank(b.id));
})();

export const DEFAULT_MODEL_ID = "deepseek-v4-flash";

// The curated registry is the WHOLE selectable set. The open intake (WS-D1:
// pick any GGUF file / HF repo, sentinel id "local-gguf") was removed on
// 2026-08-15 — a product decision, not a technical limit: shared endpoints
// must run models whose quality we can vouch for, and serving arbitrary GGUFs
// is what generic local-inference tools already do. Users who want another
// model file a GitHub issue; adding one = adding a manifest here.
// (settings.ts still recognizes the stored sentinel and migrates it back to
// the default model.)

/** Is this id something the client offers today? Unknown and delisted ids
 *  are not — the settings loader relies on that to migrate stored selections
 *  of a withdrawn model back to the default. */
export function isAvailable(id: string): boolean {
  return AVAILABLE_MODELS.some((m) => m.id === id);
}

export function getModel(id: string): ModelSpec {
  return MODELS.find((m) => m.id === id) ?? MODELS[0];
}

export function getManifest(id: string): ModelManifest {
  return MANIFESTS.find((m) => m.id === id) ?? MANIFESTS[0];
}

/** Exact-id MoE check for placement/resource UI. Unlike getManifest(), an
 * unknown id must not inherit the default model's architecture. */
export function isMoeModel(id: string): boolean {
  return MANIFESTS.some((m) => m.id === id && (m.moe?.n_expert ?? 0) > 0);
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
  // Compare LEAF names. Manifest entries carry the repo's directory prefix
  // ("UD-IQ1_S/GLM-5.2-…-00001-of-00006.gguf") while the model folder holds
  // the bare file, so a whole-string compare recognised nothing that came from
  // a split repo — every part of a 200 GB download listed as an unknown file.
  const leaf = (p: string) => p.split(/[\\/]/).pop() ?? p;
  const want = leaf(file);
  for (const m of MANIFESTS) {
    for (const v of m.variants ?? []) {
      if (leaf(v.gguf) === want) return { label: m.label, quant: v.quant };
      // A part of a split precision is that precision — the user downloaded
      // one thing and should see one name, not six mystery files.
      for (const part of v.parts ?? []) {
        if (leaf(part.file) === want) return { label: m.label, quant: v.quant };
      }
    }
    if (leaf(m.default_gguf) === want) return { label: m.label };
    for (const part of m.parts ?? []) {
      if (leaf(part.file) === want) return { label: m.label };
    }
  }
  // Engine-produced layer shards ("L26-38.gguf": layers 26 through 38 of a
  // model, written by the weight-sharding tooling). They are not models and
  // never appear in a manifest, but they DO sit in the model folder — saying
  // what they are beats listing them as if a model had that name.
  const shard = /^L(\d+)[-_](\d+)\.gguf$/i.exec(want);
  if (shard) return { label: `Layer shard ${shard[1]}–${shard[2]}` };
  return null;
}

// ---- precision (quant) selection -------------------------------------------
// Curated models expose their registered precision rows, even when only one
// variant is available. Keeping that single row visible makes the actual
// weight format explicit instead of silently treating it as a default.

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
// One formula with the llama.cpp scheduler (src/common/plan.c): selected
// weight bytes + the exact manifest/GGUF KV geometry + the calibrated per-node
// engine allocation. Do not add a UI-only "safety margin" here; any number not
// present in the engine formula makes the card and the eventual launch disagree.
// Per-node cost, mirroring plan.c: the MEASURED CUDA context (878 MiB on the
// RTX 5060 Ti, derived from NVML free outside the engine process minus the free
// the engine reports from inside it) plus the single 100 MiB margin. Metal has
// no context analogue, so charging the CUDA figure there is conservative.
// This is per NODE because each machine runs its own engine process; weights,
// KV and the graph workspace are split by the tensor split instead.
const LLAMA_CUDA_CONTEXT_BYTES = 878 * 1024 ** 2;
const LLAMA_NODE_MARGIN_BYTES = 100 * 1024 ** 2;

/** The measured workspace for a context size on a backend, or 0 when
 *  unmeasured. Two product tiers, two measurements — nothing in between to
 *  interpolate, and interpolating is what this replaced.
 *
 *  An unknown backend takes the LARGER of the two, matching plan.c: picking one
 *  would be a guess about the machine, and on GLM-5.2 that guess is wrong by
 *  22x in the direction that OOMs. */
export type NodeBackend = "cuda" | "metal" | "unknown";

/** OS -> backend, mirroring IDLETOKEN_BACKEND_OF_OS in include/idletoken_plan.h.
 *  Sound because hard constraint #3 admits exactly two compute configurations:
 *  Windows/Linux on CUDA, macOS on Metal. An unknown OS stays "unknown", which
 *  charges the larger of the two rather than guessing the common case. */
export function backendOfOs(os: string | undefined): NodeBackend {
  if (os === "macos") return "metal";
  if (os === "windows" || os === "linux") return "cuda";
  return "unknown";
}

/** Index into every `compute_bytes_*` array. MUST match idletoken_kv_tier in
 *  include/idletoken_model.h — both sides read these arrays positionally. */
const KV_TIERS = ["f16", "q8_0", "q4_0"] as const;
export type KvTier = (typeof KV_TIERS)[number];

function quantBits(quant: string): number {
  const m = /(?:^|[^A-Za-z0-9])(?:MXFP|FP|I?Q|BF|F)(\d+)/i.exec(quant);
  return m ? Number(m[1]) : 0;
}

/** THE boundary table, mirroring `idletoken_llama_kv_tier_for_weight`
 *  (src/common/plan.c) EXACTLY. The coordinator applies the same rule to the
 *  plan it launches with, so a divergence here shows up as a card that promises
 *  a fit the engine refuses. scripts/resource_estimate_gate.sh compares the two
 *  byte for byte over every model/precision/tier/node/backend combination.
 *
 *  REVISED 2026-09-02 (user decision): q8_0 covers every quantized tier, Q8
 *  included; f16 is reserved for unquantized BF16/F16 weights and for a quant
 *  name we could not read (never guess DOWN — that under-charges the budget). */
export function kvTierForQuant(quant: string): number {
  const bits = quantBits(quant);
  if (bits >= 1 && bits <= 2) return 2; // q4_0
  if (bits >= 3 && bits <= 15) return 1; // q8_0
  return 0; // unquantized (>=16 bit) and unreadable names stay f16
}

/** KV cache dtype cost relative to f16. Ratios are block bytes per 32 elements
 *  from the engine's own dtype table (src/coord/llama_sidecar.c). */
function kvGrowthScale(quant: string): number {
  switch (kvTierForQuant(quant)) {
    case 2: return 18 / 64; // q4_0 block bytes / f16
    case 1: return 34 / 64; // q8_0 block bytes / f16
    default: return 1;
  }
}

/** The measured workspace for a context size on a backend, or 0 when
 *  unmeasured. Two product tiers, two measurements — nothing in between to
 *  interpolate, and interpolating is what this replaced.
 *
 *  `quant` selects the KV tier, because the workspace differs between cache
 *  dtypes on some architectures and the coordinator picks the dtype from the
 *  precision. Omitting it reads the f16 slot, which is only right for
 *  unquantized weights — every caller that knows the precision must pass it. */
export function computeBytesFor(
  man: ModelManifest,
  ctx: number,
  backend: NodeBackend = "unknown",
  quant = "",
): number {
  const kv = kvTierForQuant(quant);
  const at = (a: number[] | undefined) => a?.[kv] ?? 0;
  // Mirrors idletoken_llama_ctx_tier_of(): a model whose ceiling falls between
  // tiers launches at its ceiling and reads the slot above it; anything past
  // 256K that is not exactly 1M has no measurement and must refuse, not round.
  let cuda: number, metal: number;
  if (ctx > 0 && ctx <= 131072) {
    cuda = at(man.compute_bytes_128k_cuda);
    metal = at(man.compute_bytes_128k_metal);
  } else if (ctx <= 262144) {
    cuda = at(man.compute_bytes_256k_cuda);
    metal = at(man.compute_bytes_256k_metal);
  } else if (ctx === 1048576) {
    cuda = at(man.compute_bytes_1m_cuda);
    metal = at(man.compute_bytes_1m_metal);
  } else {
    return 0;
  }
  if (backend === "cuda") return cuda;
  if (backend === "metal") return metal;
  if (cuda === 0 || metal === 0) return 0;
  return Math.max(cuda, metal);
}

function roundCells256(cells: number): number {
  return Math.ceil(cells / 256) * 256;
}

/** One sequence's actual cache/state allocation for the selected context and
 * automatic KV tier. Fixed recurrent/compressor state stays f32 and is not
 * scaled with K/V dtype. */
export function kvBytesForContext(m: ModelManifest, ctx: number, quant: string): number {
  const scale = kvGrowthScale(quant);
  if (m.kv.raw_bytes_per_cell) {
    const raw = m.kv.raw_bytes_per_cell * ctx;
    const csa = (m.kv.csa_bytes_per_cell ?? 0) * roundCells256(Math.ceil(ctx / 4));
    const hca = (m.kv.hca_bytes_per_cell ?? 0) * roundCells256(Math.ceil(ctx / 128));
    return (raw + csa + hca) * scale + (m.kv.fixed_bytes_per_sequence ?? 0);
  }
  if (m.kv.kind === "hybrid") {
    const interval = Math.max(1, m.kv.full_attention_interval ?? 1);
    const full = Math.ceil(m.n_layers / interval);
    const linear = m.n_layers - full;
    return m.kv.bytes_per_token_per_layer * full * ctx * scale
      + (m.kv.state_bytes_per_layer ?? 0) * linear;
  }
  return m.kv.bytes_per_token_per_layer * m.n_layers * ctx * scale;
}

export interface CapacityEstimate {
  needBytes: number; // whole-cluster requirement at this tier / node count
  haveBytes: number; // this machine's usable VRAM contribution
  gapBytes: number; // max(0, need - have)
  hostableLayers: number; // layers THIS GPU could hold
  // Node count the estimate was actually computed for.
  nodes: number;
}

/** One machine's contribution to the GPU-only serving pool. Legacy roster
 * fields may still carry RAM for wire compatibility; estimation ignores it. */
export interface NodeMemory {
  vramFree?: number;
  ramFree?: number;
  ramExpertFree?: number;
  unifiedMemory?: boolean;
}

/**
 * Add up what a whole cluster brings (2026-08-15).
 *
 * Every machine measures its own GPU working-set budget and sends it with its
 * join, so the roster already carries the numbers — this just totals VRAM.
 * Unified-memory platforms report their one GPU budget in vramFree.
 *
 * `complete` is false when any member reported nothing (an older build). The
 * total is then a lower bound, and the caller must say "cannot tell" rather
 * than declare a shortfall that may not exist — a wrong "not enough" would
 * send someone shopping for hardware they already have.
 */
export function poolVram(nodes: NodeMemory[]): { bytes: number; complete: boolean } {
  let bytes = 0;
  let complete = nodes.length > 0;
  for (const n of nodes) {
    const v = n.vramFree ?? 0;
    if (v === 0) {
      complete = false;
      continue;
    }
    bytes += v;
  }
  return { bytes, complete };
}

/** Potential node-local MoE expert pool. Unified-memory members contribute no
 * second pool; the runtime also filters out older workers that do not advertise
 * rpc-cpu-v1 and scans the exact GGUF tensor layout before admitting Hybrid. */
export function poolRam(nodes: NodeMemory[]): { bytes: number; complete: boolean } {
  let bytes = 0;
  let complete = nodes.length > 0;
  for (const n of nodes) {
    if (n.unifiedMemory) continue;
    /* Use the member's already node-local budget, not raw free RAM. In
     * particular, every Windows member has its own WDDM fixed reserve; adding
     * ramFree first and subtracting one reserve for the cluster is wrong. */
    const r = n.ramExpertFree ?? 0;
    if (r === 0) {
      complete = false;
      continue;
    }
    bytes += r;
  }
  return { bytes, complete };
}

export type ClusterCapacityVerdict = "fits" | "short" | "unknown" | "hybrid-check";

/** Classify only what the pre-flight aggregate can prove. RAM in a MoE cluster
 * is owner-local: reaching `need` after adding it is a candidate, never proof
 * that a contiguous per-node layer plan exists. */
export function clusterCapacityVerdict(
  needBytes: number,
  gpuBytes: number,
  ramExpertBytes: number,
  isMoe: boolean,
  gpuComplete = true,
  ramComplete = true,
): ClusterCapacityVerdict {
  if (gpuBytes >= needBytes) return gpuComplete ? "fits" : "unknown";
  if (isMoe && gpuBytes + ramExpertBytes >= needBytes) return "hybrid-check";
  if (!gpuComplete || (isMoe && !ramComplete)) return "unknown";
  return "short";
}

/**
 * The RAM this machine adds to its capacity for a MoE model — the
 * single-machine `--n-cpu-moe` path (hard constraint #6/#7), where routed
 * experts may live in RAM.
 *
 * It is the native probe's node-local expert budget. This normally equals
 * usable RAM, but Windows caps it at WDDM's per-process NON-LOCAL budget: RAM
 * can be physically free while cudaHostAlloc can no longer page-lock it.
 *
 * Zero when it does not apply: a dense model (GPU_ONLY) or a unified-memory
 * machine (one pool, nothing to add). For a cluster the caller passes the
 * roster's pooled RAM (poolRam), which already leaves unified-memory members
 * out — cluster MoE Hybrid keeps each machine's experts in its own RAM, so
 * the pool is the sum of what the members can each hold. The coordinator
 * remains the hard gate at launch, on the real tensor sizes.
 */
export function moeRamExpertBudget(modelId: string, ramBytes: number, unified: boolean): number {
  if (unified || !isMoeModel(modelId)) return 0;
  return Math.max(0, ramBytes);
}

/** Exact routed-expert layout read from the selected GGUF's tensor directory.
 * Hybrid budgeting must not infer this split from parameter counts: different
 * quantizations can store their expert tensors differently even when the model
 * architecture is identical. */
export interface MoeLayoutBudget {
  expertBytesTotal: number;
  nExpert: number;
  nExpertUsed: number;
  complete: boolean;
}

export interface HybridRequirements {
  /** Non-expert resident weights + KV/workspace/node overhead + the smallest
   * device expert pool that can serve one decode batch. */
  vramNeedBytes: number;
  /** The complete routed-expert store kept in system memory. */
  ramNeedBytes: number;
  /** Expert bytes included in vramNeedBytes for the one-batch device pool. */
  transferBytes: number;
  poolExperts: number;
}

/** Split a full-GPU requirement into Hybrid's two independent resource pools.
 *
 * The minimum device pool is the selected model's routed-expert count for one
 * token (`n_expert_used`). This is not a universal constant and it is not a
 * prefetch target: every extra byte of VRAM can retain more experts and reduce
 * host-to-device traffic. */
export function hybridRequirements(
  fullGpuNeedBytes: number,
  layout: MoeLayoutBudget | null | undefined,
): HybridRequirements | null {
  if (
    !layout?.complete ||
    !Number.isSafeInteger(fullGpuNeedBytes) ||
    !Number.isSafeInteger(layout.expertBytesTotal) ||
    fullGpuNeedBytes <= 0 ||
    layout.expertBytesTotal <= 0 ||
    layout.expertBytesTotal > fullGpuNeedBytes ||
    layout.nExpert <= 0 ||
    layout.nExpertUsed <= 0 ||
    layout.nExpertUsed > layout.nExpert
  ) {
    return null;
  }
  const poolExperts = layout.nExpertUsed;
  const transferBytes = Math.ceil(
    (layout.expertBytesTotal * poolExperts) / layout.nExpert,
  );
  return {
    vramNeedBytes: fullGpuNeedBytes - layout.expertBytesTotal + transferBytes,
    ramNeedBytes: layout.expertBytesTotal,
    transferBytes,
    poolExperts,
  };
}

// `nNodes`: the known cluster size when paired; pass the nominal typical
// cluster (e.g. 3) when standalone — the per-node overhead term is small, so
// the guidance stays honest either way.
export function estimateClusterCapacity(
  model: ModelSpec,
  mem: { vram_usable: number },
  ctx: number,
  nNodes: number,
  quant?: string, // selected precision; changes the weight bytes → feasibility
  // Which backend the machines run. Not cosmetic: GLM-5.2's measured workspace
  // is 1.50 GiB on CUDA and 33.25 GiB on Metal, so charging the wrong one is a
  // 22x error. Omitted = "unknown" = charge the larger, never the cheaper.
  backend: NodeBackend = "unknown",
  // Routed-expert bytes this machine's RAM may take for a MoE model
  // (moeRamExpertBudget); 0 for everything else. Counted as capacity, never
  // as a reduction of the need: the model still has to be loaded whole.
  ramExpertBytes = 0
): CapacityEstimate {
  const man = getManifest(model.id);
  const n = Math.max(1, nNodes);
  // Size the SELECTED precision (falls back to the manifest scalars when the
  // model has no variant menu) so the guidance tracks the quant dropdown.
  const v = getVariant(model.id, quant);
  const layerBytes = v ? v.layer_weight_bytes : man.layer_weight_bytes;
  const sharedBytes = v ? v.shared_weight_bytes : man.shared_weight_bytes;
  const weightBytes = layerBytes + sharedBytes;
  // ONE resolved precision for both KV terms below. Reading the dropdown for
  // the cache size and the manifest default for the workspace would price two
  // different configurations against each other.
  const kvQuant = quant || defaultQuant(model.id);
  const kvBytes = kvBytesForContext(man, ctx, kvQuant);
  const nodeOverhead = LLAMA_CUDA_CONTEXT_BYTES + LLAMA_NODE_MARGIN_BYTES;
  // The graph workspace is charged ONCE for the cluster, like weights and KV:
  // the tensor split divides the graph. Only the CUDA context is per-node.
  // The precision goes in because it selects the KV cache dtype, and the
  // workspace differs between cache dtypes on some architectures.
  const computeBytes = computeBytesFor(man, ctx, backend, kvQuant);
  const needBytes = weightBytes + kvBytes + computeBytes + n * nodeOverhead;
  // Product capacity is GPU-addressable memory only. On unified-memory
  // machines the native probe already reports the GPU working-set budget in
  // vram_usable, so there is still exactly one number to count.
  const haveBytes = mem.vram_usable + Math.max(0, ramExpertBytes);
  // Same split basis as plan.c's tensor-split cap: weights, KV and workspace
  // all divide with the layers, so the workspace belongs in the per-layer cost.
  const perLayer =
    man.n_layers > 0 ? (weightBytes + kvBytes + computeBytes) / man.n_layers : 0;
  const usableForLayers = haveBytes - nodeOverhead;
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
 * The estimator itself is GPU-only, so its `gapBytes` now answers this exact
 * question without a second VRAM-vs-RAM interpretation.
 *
 * When nothing fits in VRAM it returns the **smallest** one: the capability panel
 * still explains how much is missing, and that beats leading with a 304B model,
 * which only makes people think the software is not for them.
 */
export function pickBestFittingModel(
  mem: { vram_usable: number; ram_usable: number; unified_memory: boolean },
  ctx: number
): { modelId: string; quant: string } {
  const usable = AVAILABLE_MODELS;
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
