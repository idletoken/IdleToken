/* idletoken_plan.h — cluster mode decision + PP layer split.
 *
 * The capacity model of docs/architecture.md §5 as pure functions: no sockets, no GPU, no
 * ds4 — so the core planning algorithms unit-test on any host
 * (src/tools/plan_test.c, runs on macOS control machine and Linux alike).
 *
 * Multi-model: every model-shaped number (layer count, weight bytes, KV
 * overhead) comes from the idletoken_model_spec passed in — the planner itself
 * knows no model (docs/multi-model-design.md §3.3).
 */
#ifndef IDLETOKEN_PLAN_H
#define IDLETOKEN_PLAN_H

#include <stdint.h>
#include <stddef.h>

#include "idletoken_model.h"

typedef enum {
    IDLETOKEN_MODE_REFUSE   = 0,
    IDLETOKEN_MODE_GPU_ONLY = 1,   /* value matches the ASSIGN_PLAN mode byte */
} idletoken_mode;

/* The product's exact context windows (docs/ctx-tiers-2026-09.md). 128K is the
 * client default; 256K and 1M are explicit choices. Context never changes
 * automatically — a window that does not fit is refused, never shrunk.
 *
 * These are the ONLY values the planner has measurements for. A ctx that is not
 * one of them is refused rather than rounded: rounding is how a request for
 * 100000 tokens silently got priced at the 256K workspace. */
#define IDLETOKEN_CTX_TIER_128K  131072u
#define IDLETOKEN_CTX_TIER_256K  262144u
#define IDLETOKEN_CTX_TIER_1M   1048576u
#define IDLETOKEN_PRODUCT_CONTEXT_CAP IDLETOKEN_CTX_TIER_1M
#define IDLETOKEN_DEFAULT_CONTEXT_TOKENS IDLETOKEN_CTX_TIER_128K

uint32_t idletoken_llama_product_ctx_ceiling(
    const idletoken_model_spec *model);

/* The tier a context size belongs to, or 0 when it is not a product tier.
 * A model whose ceiling falls between tiers is launched at its ceiling, so the
 * caller passes the ceiling-clamped value and this maps it to the slot that
 * holds its measurement: <=128K -> 128K slot, <=256K -> 256K slot, exactly 1M
 * -> 1M slot. Anything above 256K that is not 1M has no measurement and gets 0.
 */
uint32_t idletoken_llama_ctx_tier_of(uint32_t ctx_size);

/* Is this EXACTLY one of the three product windows?
 *
 * Distinct from idletoken_llama_ctx_tier_of() on purpose, and mixing them up is
 * a silent bug. That one answers "which measurement slot does this launched
 * window read?" and therefore maps anything at or below a tier onto it — which
 * is required, because a model whose ceiling falls between tiers (qwen3-8b at
 * 163840) really is launched at its ceiling. This one answers "may a CALLER ask
 * for this?", where the only acceptable answers are the three the product
 * offers. Using the former to validate an argument lets `--ctx-size 100000`
 * through and serves a window nobody asked for at a price nobody measured. */
int idletoken_llama_is_ctx_tier(uint32_t ctx_size);

/* Which compute backend a node runs. Needed because the graph workspace is
 * NOT backend-independent: measured 2026-09-01, GLM-5.2 at 256K reserves
 * 1.50 GiB on CUDA and 33.3 GiB on Metal — 22x — while the Qwen family agrees
 * byte for byte. One number per model would therefore be right for some models
 * and 22x wrong for others.
 *
 * UNKNOWN is 0 so that a zero-initialized row (which the node_mem contract
 * below requires) cannot silently claim a backend it never reported. The
 * planner charges the LARGER of the two backends for an unknown node — the
 * safe direction, and visibly conservative rather than quietly wrong. */
typedef enum {
    IDLETOKEN_NODE_BACKEND_UNKNOWN = 0,
    IDLETOKEN_NODE_BACKEND_CUDA    = 1,
    IDLETOKEN_NODE_BACKEND_METAL   = 2,
} idletoken_node_backend;

/* OS family -> backend, using the idletoken_os_family values. Sound because
 * hard constraint #3 admits exactly two compute configurations: Windows/Linux
 * on CUDA, macOS on Metal. Derived from a field the roster already carries, so
 * no wire change was needed to know this. An unknown OS stays UNKNOWN rather
 * than defaulting to the common case. */
#define IDLETOKEN_BACKEND_OF_OS(os) \
    ((os) == 3 /* IDLETOKEN_OS_MACOS */ ? IDLETOKEN_NODE_BACKEND_METAL \
   : ((os) == 1 || (os) == 2)           ? IDLETOKEN_NODE_BACKEND_CUDA  \
   :                                      IDLETOKEN_NODE_BACKEND_UNKNOWN)

typedef struct {
    uint64_t vram_usable;
    uint64_t ram_usable;
    /* Legacy probe fields retained for roster compatibility. Serving capacity
     * is GPU-only; neither field contributes to planning. */
    uint64_t ram_pinnable;
    uint8_t  unified;   /* 1 = unified memory host (vram aliases ram) */
    /* idletoken_node_backend. Derived from the probe's gpu_vendor; unified
     * memory is NOT a proxy for it (the DGX is CUDA and unified). */
    uint8_t  backend;
    /* Which machine this row is (hostname / node id). Empty = unnamed, and the
     * planner then says "node #i" instead.
     *
     * Why the planner needs a name at all: every number above is a machine's
     * own DECLARATION. When one of them is wrong (a stale cap, a joiner
     * reporting memory it does not have), the cluster refuses to start — and a
     * refusal that says "a node has only 2.1 GiB" leaves the owner of five
     * machines with no idea which one to look at. Same standard as the
     * engine-version gate, which names the machine that must upgrade.
     *
     * Keep the name inline rather than borrowing a pointer into a roster. The
     * capability endpoint is served by a thread pool and outlives several of
     * the stack adapters that create these rows; a borrowed or uninitialized
     * pointer here used to crash the coordinator while a chat stream was live.
     *
     * ⚠ **Zero-initialize this struct** (`= {0}` / designated initializers /
     * memset) before filling fields one by one. This also protects every
     * future scalar field from becoming stack garbage. */
    char label[64];
} idletoken_node_mem;

/* Whole-cluster admission bytes for the selected precision:
 *
 *   exact measured GGUF weight bytes
 * + exact engine-shaped KV/recurrent state for `ctx_size`
 * + the MEASURED graph workspace for `ctx_size` on `backend`
 * + n_nodes * (measured CUDA context + the single 100 MiB margin)
 *
 * Every term is measured, none is a slope on the weights: the closed form that
 * used to sit here (768 MiB + weight_bytes/64) was 8.2x low on GLM-5.2 at 256K
 * and 22x too high if applied with the wrong backend
 * (results/memory-need-measured-20260901.md). Cache-cell padding still matches
 * the compiled llama.cpp planner byte-for-byte. */
uint64_t idletoken_needed_bytes(const idletoken_model_spec *model,
                             uint32_t ctx_size, int n_nodes, uint8_t backend);

/* Same, for an explicitly chosen precision. Small models ship several quants
 * whose weight bytes differ by 3-4x, so "can I run this?" is only answerable
 * per (model, quant). `quant` NULL/unknown falls back to the model default,
 * making this a strict superset of idletoken_needed_bytes. */
uint64_t idletoken_needed_bytes_quant(const idletoken_model_spec *model,
                                   const char *quant,
                                   uint32_t ctx_size, int n_nodes,
                                   uint8_t backend);

/* Mode decision: Σ usable_vram >= exact need → GPU_ONLY, otherwise REFUSE.
 * System RAM is never serving capacity. On unified-memory machines the probe
 * reports the GPU working-set budget in vram_usable, so it is still one pool.
 * `why` (optional) receives a short human-readable reason. */
/* Quant-aware mode decision + the shortfall when the answer is REFUSE.
 * `idletoken_mode_decide` below is this with quant=NULL and no shortfall — one
 * implementation, so the advisor's verdict can never drift from the planner's
 * (that drift is the easiest way for a capability table to lie). */
idletoken_mode idletoken_mode_decide_quant(const idletoken_model_spec *model,
                                     const char *quant,
                                     const idletoken_node_mem *nodes, int n,
                                     uint32_t ctx_size,
                                     uint64_t *shortfall_out,
                                     char *why, size_t whylen);

idletoken_mode idletoken_mode_decide(const idletoken_model_spec *model,
                               const idletoken_node_mem *nodes, int n,
                               uint32_t ctx_size, char *why, size_t whylen);

/* Resource-proportional contiguous split of model->n_layers across n nodes
 * (callers pass nodes sorted strongest-first). Every node gets ≥1 layer;
 * remainders land on the strongest nodes. out_counts[i] = node i's layer
 * count, Σ == n_layers. Returns 0, or -1 when n <= 0 or n > n_layers.
 * `mode` is retained in this pure helper's ABI and must be GPU_ONLY. Splits use
 * usable VRAM only, with every layer resident on a GPU.
 * `ctx_size` feeds the per-node capacity cap: after the proportional pass the
 * split is repaired so no node is assigned more layers than
 * (its usable memory − shared weights − per-tier overhead) can hold — the
 * Per-node capacity-cap bin-packing repair. Excess moves to nodes with headroom,
 * strongest first. When even the caps can't fit n_layers (coarse estimates,
 * borderline clusters) the repair degrades gracefully back toward the
 * proportional split instead of failing — mode_decide already admitted the
 * cluster by total capacity, so refusing here on estimates would brick it.
 * When model->split_boundary_multiple > 1 a final pass nudges stage
 * boundaries onto multiples of it where capacity allows (best-effort — e.g.
 * GLM-5.2's 4-layer shared-indexer groups); it never violates caps or the
 * ≥1-layer floor. */
int idletoken_plan_layers(const idletoken_model_spec *model,
                       const idletoken_node_mem *nodes, int n,
                       uint32_t ctx_size, int *out_counts, idletoken_mode mode);

/* ===== llama.cpp-engine scheduling (v2 rebuild WS-B2) ======================
 *
 * The v2 mandate (docs/v2-rebuild-plan-2026-08.md §4 WS-B2, §5 invariants
 * 1/5): given a model's REAL byte size (from the GGUF header, WS-B4) and the
 * probed node memories, decide single-machine vs cluster, produce per-node
 * --tensor-split ratios, pin layer 0 to the coordinator, and refuse with a
 * sentence a human can act on. Pure functions — unit-tested in plan_test.c.
 *
 * Usable memory per node (the formula; calibration constants are estimates):
 *
 *   usable = vram_usable on every platform. On Apple Silicon / Grace this is
 *   the probed GPU working-set budget for the unified pool, counted once.
 *   ram_usable remains telemetry only and never enlarges serving capacity.
 */

#define IDLETOKEN_LLPLAN_MAX_NODES 16
#define IDLETOKEN_LLPLAN_MAX_DEVICES (IDLETOKEN_LLPLAN_MAX_NODES * 2 - 1)

typedef enum {
    IDLETOKEN_LLPLAN_SINGLE  = 0,  /* run on one machine, no RPC */
    IDLETOKEN_LLPLAN_CLUSTER = 1,  /* span nodes[] via ggml-RPC + tensor-split */
    IDLETOKEN_LLPLAN_REFUSE  = 2,  /* cannot run; `why` says what is missing */
} idletoken_llplan_kind;

/* What the scheduler needs to know about a model — from
 * idletoken_model_from_gguf() (WS-B4) for open models, or the registry. */
typedef struct {
    uint64_t total_bytes;        /* whole GGUF on disk */
    uint32_t n_layers;           /* transformer blocks */
    uint64_t kv_bytes_per_token; /* whole-model KV bytes per context token;
                                  * 0 = unknown (charged as unknown, not free) */
    /* Context-independent recurrent/compressor state for one sequence. This is
     * NOT scaled when K/V cache dtype changes: Qwen hybrid state and DeepSeek4
     * compressor state are f32 allocations in the pinned engine. */
    uint64_t kv_fixed_bytes_per_seq;
    /* DeepSeek4 exact cache geometry. Nonzero raw selects the padded cache
     * formula in idletoken_llama_kv_bytes(); all are whole-model f16 bytes per
     * cache cell. CSA/HCA cell counts are ceil(ctx/4) and ceil(ctx/128), then
     * each is rounded up to 256 exactly like llama-kv-cache-dsv4.cpp. */
    uint64_t dsv4_raw_bytes_per_cell;
    uint64_t dsv4_csa_bytes_per_cell;
    uint64_t dsv4_hca_bytes_per_cell;
    /* MoE: experts present and experts consulted per token (GGUF
     * `expert_count` / `expert_used_count`; both 0 on a dense model).
     *
     * They exist because the WORKING SET — the bytes actually touched per
     * token, i.e. what decides speed — is a fraction of the weights on an MoE
     * model. Measured 2026-08-16: GLM-5.2 (744B total / 40B active, 256/8
     * experts) generated at 0.91 tok/s on ONE 119 GiB machine with a 222 GiB
     * file, because each token walks ~12 GiB and that stays in page cache.
     * A dense model of the same size would have to read all 222 GiB per token.
     * Same size, different machine requirement — one number cannot say both. */
    uint32_t n_expert;
    uint32_t n_expert_used;
    /* MEASURED graph workspace at the two context tiers — llama.cpp's own
     * no_alloc dry-run (`scripts/measure_model_memory.sh`), NOT an estimate.
     * 0 = not measured for this model; the planner then refuses to guess.
     *
     * WHY MEASURED (2026-09-01, results/memory-need-measured-20260901.md):
     * the closed form this replaces (`768 MiB + weights/64`) had no context
     * term at all, and the workspace turns out to grow at a rate set by the
     * model's ARCHITECTURE. Measured at 256K it was 59% high on Qwen3.5-0.8B
     * and 9.6x LOW on DeepSeek-V2-Lite (9764 MiB actual vs 1013 MiB charged) —
     * low enough to admit a run that then OOMs after a full load. No single
     * formula covers both, so there is no formula here any more.
     *
     * Independent of the WEIGHT quantization: measured byte-identical across
     * Metal / CUDA-unified / CUDA-discrete for a given backend, and across
     * every precision in a menu (Q4_K_M..BF16 all 489.00 MiB on Qwen3.5-0.8B
     * at 256K). It is a property of the compute graph, not of the GPU or of
     * the weights.
     *
     * ⚠ Re-measure when scripts/llamacpp-patches/UPSTREAM moves: the graph
     * belongs to the engine. Same rule as the perplexity baselines.
     *
     * ⚠ Per BACKEND, not one number. Qwen3.5-0.8B and Qwen3.5-9B measure
     * byte-identical on Metal and CUDA, which is what made a single value look
     * safe; GLM-5.2 measures 33.3 GiB on Metal against 1.50 GiB on CUDA. Two
     * agreeing models of one architecture family are not evidence about the
     * rest of the list.
     *
     * ⚠ Per KV CACHE DTYPE too (2026-09-02). These are ALREADY REDUCED to the
     * tier this model+precision will launch with — idletoken_model_size_
     * resolve() picks the index; the spec carries all three. Qwen3.5-0.8B at
     * 256K on Metal: 489.00 MiB f16 vs 745.28 MiB quantized.
     *
     * Three CONTEXT tiers since 2026-09-02 (docs/ctx-tiers-2026-09.md). Not
     * interpolatable — see idletoken_model_spec for the measured curve. */
    uint64_t compute_bytes_128k_cuda;
    uint64_t compute_bytes_256k_cuda;
    uint64_t compute_bytes_1m_cuda;
    uint64_t compute_bytes_128k_metal;
    uint64_t compute_bytes_256k_metal;
    uint64_t compute_bytes_1m_metal;
    /* Which IDLETOKEN_KV_TIER_* the four values above were taken from, i.e.
     * the cache dtype the engine will be launched with. Diagnostic: the budget
     * is already reduced. Kept so a log line can state the configuration it
     * priced instead of leaving the reader to re-derive it from the quant. */
    uint8_t kv_tier;
} idletoken_llm_model_size;

/* The measured workspace for `ctx_size` on `backend`
 * (idletoken_node_backend), or 0 when this model has no measurement for it.
 * UNKNOWN yields the larger of the two. Exposed so the coordinator and the
 * capability table ask the same question the planner does. */
uint64_t idletoken_llama_compute_bytes(const idletoken_llm_model_size *model,
                                       uint32_t ctx_size, uint8_t backend);

/* The backend a homogeneous roster runs on, or UNKNOWN if the rows disagree or
 * any row never reported one. Heterogeneous clusters are out of scope for the
 * workspace budget (2026-09-01); UNKNOWN makes the planner charge the larger
 * backend rather than pick one and be quietly wrong on the other machines. */
uint8_t idletoken_llama_roster_backend(const idletoken_node_mem *nodes, int n);

/* Exact one-sequence cache/state allocation for this model at `ctx_size`.
 * Generic attention is linear KV + fixed recurrent state; DeepSeek4 follows
 * the pinned engine's three-cache padded allocation. Saturates on overflow. */
uint64_t idletoken_llama_kv_bytes(const idletoken_llm_model_size *model,
                                  uint32_t ctx_size);

/* Apply the chosen K/V dtype to growth caches only. Fixed recurrent state is
 * f32 and deliberately unchanged. Used by the coordinator for both explicit
 * and automatic KV precision so plan and spawned engine stay identical. */
void idletoken_llama_model_kv_scale(idletoken_llm_model_size *model,
                                    double scale);

/* Bytes that must be RESIDENT (allocated, not evictable) for the model to run
 * at all, on `n_nodes` machines at `ctx_size`: KV cache + per-node engine
 * overhead. Deliberately NOT the weights: llama.cpp mmaps them, so they are
 * page cache the kernel reclaims under pressure — measured 2026-08-16, a
 * 222 GiB model served from a 119 GiB machine with MemAvailable never below
 * 113 GiB (docs/resource-budget-rethink-2026-08.md §5). */
uint64_t idletoken_llama_hard_need(const idletoken_llm_model_size *model,
                                   uint32_t ctx_size, int n_nodes,
                                   uint8_t backend);

/* Bytes touched per token — what memory has to CACHE for full speed. Dense:
 * the whole file. MoE: shared/attention layers plus only the experts a token
 * consults. Not a feasibility bound; it decides fast vs slow. */
uint64_t idletoken_llama_working_set(const idletoken_llm_model_size *model);

typedef struct {
    idletoken_llplan_kind kind;

    /* SINGLE: the chosen node (index into the caller's nodes[]). */
    int single_node;

    /* CLUSTER: participating nodes in tensor-split order. order[0] is ALWAYS
     * the coordinator — llama.cpp assigns the first split slice (layer 0 and
     * the embedding lookup with it) to the first device, which is how hard
     * invariant #1 (layer 0 + embedding stay on the coordinator) is enforced
     * by construction rather than by hope. */
    int    n_nodes;
    int    order[IDLETOKEN_LLPLAN_MAX_NODES];        /* indices into nodes[] */
    double tensor_split[IDLETOKEN_LLPLAN_MAX_NODES]; /* proportional, Σ = 1.0 */
    int    layer0_node;   /* == order[0] == coordinator (CLUSTER); == single_node (SINGLE) */

    /* GPU-only admission always keeps the complete working set resident. */
    int    working_set_fits;
    uint64_t hard_need_bytes;    /* what must be resident (KV + per-node overhead) */
    uint64_t working_set_bytes;  /* what memory should cache for full speed */

    char why[512];        /* human-readable decision / refusal reason */
} idletoken_llama_plan;

/* Mirror llama.cpp's LLAMA_SPLIT_MODE_LAYER placement for one contiguous
 * interval of devices [dev_lo, dev_hi).  The returned [layer_lo, layer_hi)
 * contains only repeating transformer layers; GGUF shared tensors are handled
 * separately by the model-cache layer.  A device interval which owns no
 * repeating layer returns [n_layers, n_layers).
 *
 * This is deliberately a public planning primitive: coordinator cache plans,
 * worker cache plans, and heterogeneous-topology tests must all use the exact
 * same rounding rule as the engine. */
int idletoken_llama_device_layer_range(
    unsigned n_layers, int n_gpu_layers,
    const double *shares, int n_devices,
    int dev_lo, int dev_hi,
    unsigned *layer_lo, unsigned *layer_hi);

/* Decide how to run `model` on `nodes` (n of them, coordinator = index of the
 * node this coordinator process runs on).
 *
 *   - Fits the coordinator → SINGLE by default. The coordinator is the only
 *     machine that can legally run single: layer 0 + embedding may not move to
 *     a worker, so a strong worker never substitutes. `force_cluster` records
 *     an explicit user choice and keeps CLUSTER even when SINGLE would fit.
 *   - Needs several nodes → CLUSTER with tensor_split; coordinator first and
 *     holding at least one layer's worth. A coordinator with no usable local
 *     compute memory → REFUSE (layer 0 + embedding may not leave it).
 *   - Total memory insufficient → REFUSE; `why` names the need, the have and
 *     the shortfall in GiB plus what to do about it.
 *
 * Returns 0 (out fully filled, including REFUSE) or -1 on invalid arguments. */
int idletoken_plan_llamacpp(const idletoken_llm_model_size *model,
                            const idletoken_node_mem *nodes, int n,
                            int coordinator, uint32_t ctx_size,
                            int force_cluster,
                            idletoken_llama_plan *out);

/* Largest context that fits `usable` bytes next to the weights + fixed
 * overhead, capped at ctx_want. Returns the granted context (a multiple of
 * 1024, ≥ ctx_floor), or 0 when even ctx_floor does not fit — the caller must
 * then refuse loudly, never silently serve a smaller window (a Claude Code
 * session needs ~13K of input before the first reply; a quietly shrunken
 * context fails mid-conversation instead of at startup, which is worse).
 * kv_bytes_per_token == 0 (unknown KV shape) grants ctx_want unchanged:
 * inventing a KV cost would refuse machines that are actually fine, and the
 * engine itself still fails loudly if it cannot allocate. */
/* `backend` is idletoken_node_backend: the workspace term differs per backend
 * and this helper has no roster to derive it from. */
uint32_t idletoken_llama_fit_ctx(uint64_t usable_bytes,
                                 const idletoken_llm_model_size *model,
                                 uint32_t ctx_want, uint32_t ctx_floor,
                                 uint8_t backend);

/* Leading bit count of a weight-quant name ("IQ2_XXS"→2, "Q4_K_M"→4,
 * "Q8_0"→8, "BF16"/"F16"→16, unknown/empty→0). Drives the tiered KV rule:
 * the weight noise floor bounds what KV precision can possibly matter, so
 * 1-2 bit weights take q4_0/q4_0, 3-15 bit take q8_0/q8_0, and only
 * unquantized weights (>=16 bit) keep f16. 0 = cannot tell — treated as
 * unquantized (the conservative direction), never guessed low. */
int idletoken_quant_weight_bits(const char *quant);

/* Same answer read from a GGUF file NAME (the client launches the coordinator
 * with only --llama-gguf; curated downloads carry the variant token in the
 * filename). Last quant-shaped token wins; no match -> 0 (conservative). */
int idletoken_quant_bits_from_path(const char *path);

/* THE boundary table for the automatic KV cache dtype: q4_0 at 1-2 bit weights,
 * q8_0 at every other quantized tier (3-15 bit, Q8 included), f16 only for
 * unquantized weights and for an unreadable quant name. Returns an
 * IDLETOKEN_KV_TIER_* index, which is also how the measured `compute_bytes_*`
 * arrays are indexed — one function decides both what the engine runs and what
 * the planner charges for it. */
int idletoken_llama_kv_tier_for_weight(int weight_bits);

/* Engine dtype name for a tier ("f16" / "q8_0" / "q4_0"). Out-of-range is
 * reported as f16 rather than as an error: the caller is building a log line
 * or an env value, and the safe answer is the engine's own default. */
const char *idletoken_llama_kv_tier_name(int tier);

/* The inverse: which tier an engine dtype NAME belongs to, or -1 when it is
 * outside the three the product measures. "" and "bf16" map to f16, the dtype
 * the f16 workspace was measured under. -1 is not an error — it is the honest
 * answer for a dtype reachable only through the IDLETOKEN_KV_CACHE_TYPE escape
 * hatch, and the caller must charge the largest measured tier rather than pick
 * a neighbour. */
int idletoken_llama_kv_tier_of_name(const char *name);

/* The dtype to pass as -ctk/-ctv, or NULL when the tier is f16 and the engine's
 * own default is what we want. Derived from the tier above; there is no second
 * boundary table. K and V deliberately use the same dtype: mixed CUDA
 * flash-attention pairs were measured falling back to CPU on the pinned engine.
 * This rule is identical in private and shared mode. */
const char *idletoken_llama_kv_type_for_weight(int weight_bits);

/* Legacy diagnostic rounding helper. Product startup uses the exact selected
 * 262144 or 1048576 tokens; it never walks this tier list automatically. */
uint32_t idletoken_llama_ctx_display_tier(uint32_t max_ctx);

/* GPU-addressable serving pool, exposed so coordinator and planner cannot
 * drift. System RAM is deliberately ignored. */
uint64_t idletoken_llama_node_usable(const idletoken_node_mem *node);

/* The pool the KV cache actually LIVES IN on this node — which is NOT the same
 * as what the model can be laid out across (idletoken_llama_node_usable).
 *
 * All platforms → vram_usable only. Unified-memory probes must place their GPU
 * working-set budget in this field; host RAM is never added separately.
 *
 * Despite the name it is the general answer to "what can the engine backend on
 * this node allocate", and the CLUSTER split asks the same question of the
 * weight slice (2026-08-20): a worker's rpc-server is started `-d CUDA0`, so
 * on a discrete card the host RAM cannot hold a layer either. There is
 * deliberately no second function — one number, one place to be wrong.
 *
 * WHY THIS EXISTS (2026-08-18, measured on a Windows desktop): the slot count
 * was budgeted against vram+ram, so a 16 GiB card was told it had 80 GiB and
 * opened 4 slots
 * × 40960 ctx ≈ 22 GiB of KV. Windows WDDM does not fail that allocation — it
 * pages VRAM out to system memory, and because the desktop compositor shares
 * the card, the whole machine froze. A budget charged to memory the allocation
 * never touches is not a conservative estimate, it is a wrong one. */
uint64_t idletoken_llama_kv_pool(const idletoken_node_mem *node);

/* Ceiling on auto-derived sequence slots. Four is where a home GPU stops
 * buying throughput with KV memory: past it the batch is already wide enough
 * to keep the matmuls busy, and each further slot is another full context's
 * worth of memory that the ONE user of this machine is unlikely to need at
 * once. Raise it only with a measured curve (results/llamacpp-multislot-*). */
#define IDLETOKEN_LLAMA_SLOT_CAP 4

/* How many independent sequences this node can hold at `ctx_size` —
 * scheduler-design §4.5b, contract 1. Each slot is one more request that can
 * be in flight at the SAME context length, so the cost is one more
 * `kv_bytes_per_token × ctx_size` in the pool the KV lives in.
 *
 *   pool       = idletoken_llama_kv_pool(node)   (discrete: VRAM only)
 *   kv_per_seq = kv_bytes_per_token × ctx_size × layer_share
 *   free       = pool − (weights × layer_share) − per-node overhead
 *   slots      = clamp(⌊free / kv_per_seq⌋, 1, cap)
 *
 * It takes the NODE rather than a byte count on purpose (changed 2026-08-18):
 * whether the KV budget is VRAM or a unified pool is a property of the machine,
 * and a caller that has to pick the right number is a caller that can pick the
 * wrong one — which is exactly the bug this signature ends. See
 * idletoken_llama_kv_pool.
 *
 * The weights are charged to the SAME pool, and that carries the slow-tier
 * rule: on a discrete card whose VRAM cannot hold the weights, `free` goes
 * negative and the answer is 1. That is deliberate. Weights spilling to host
 * RAM means every token crosses PCIe already; adding parallel sequences to a
 * machine in that state buys nothing and costs another full context of the
 * scarcest memory it has.
 *
 * `layer_share` is this node's share of the model's layers (1.0 on a single
 * machine; the plan's tensor_split entry in a cluster) — under PP each node
 * holds KV only for the layers it owns, and the CLUSTER-wide answer is the
 * MINIMUM over nodes, not the sum: every node must hold its slice of the same
 * sequence, so the tightest one decides.
 *
 * Never returns 0: one slot is what the engine does anyway, and reporting 0
 * would tell the platform this machine cannot serve at all. `cap <= 0` means
 * IDLETOKEN_LLAMA_SLOT_CAP. `kv_bytes_per_token == 0` (unknown KV shape)
 * returns 1 — the opposite direction from fit_ctx's grant, and deliberately
 * so: not knowing the KV cost is a reason to open FEWER slots, while there it
 * was a reason not to refuse a machine outright. */
int idletoken_llama_seq_slots(const idletoken_node_mem *node,
                              const idletoken_llm_model_size *model,
                              uint32_t ctx_size, double layer_share, int cap);

#endif /* IDLETOKEN_PLAN_H */
