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
    IDLETOKEN_MODE_HYBRID   = 2,
} idletoken_mode;

typedef struct {
    uint64_t vram_usable;
    uint64_t ram_usable;
    /* Measured ceiling on pinned host memory; 0 = unknown/unconstrained.
     * HYBRID spills into pinned memory, so this — not ram_usable — bounds the
     * host side of a node's shard. See idletoken_resource.h for why it must be
     * measured rather than derived. */
    uint64_t ram_pinnable;
    uint8_t  unified;   /* 1 = unified memory host (vram aliases ram) */
    /* Which machine this row is (hostname / node id). NULL = unnamed, and the
     * planner then says "node #i" instead.
     *
     * Why the planner needs a name at all: every number above is a machine's
     * own DECLARATION. When one of them is wrong (a stale cap, a joiner
     * reporting memory it does not have), the cluster refuses to start — and a
     * refusal that says "a node has only 2.1 GiB" leaves the owner of five
     * machines with no idea which one to look at. Same standard as the
     * engine-version gate, which names the machine that must upgrade.
     *
     * ⚠ **Zero-initialize this struct** (`= {0}` / designated initializers /
     * memset) before filling fields one by one. A pointer field left
     * uninitialized on the stack is not a wrong name, it is a crash — which is
     * exactly how adding this field broke plan_test on 2026-08-15. The struct
     * already carries a scar from a field insertion (see `ram_pinnable` and
     * the comment in src/tools/plan_test.c); this is its pointer-shaped twin. */
    const char *label;
} idletoken_node_mem;

/* needed = model layer weights + per-node duplicated shared weights (every
 * stage loads embd/head) + per-node inference overhead (KV + compressed
 * state + activations + CUDA workspace + comms — idletoken_model_overhead()
 * with an average layers-per-node estimate).
 * ⚠ Overheads are ESTIMATES pending real-machine calibration — Spark's
 * unified memory can't produce valid numbers. */
uint64_t idletoken_needed_bytes(const idletoken_model_spec *model,
                             uint32_t ctx_size, int n_nodes);

/* Same, for an explicitly chosen precision. Small models ship several quants
 * whose weight bytes differ by 3-4x, so "can I run this?" is only answerable
 * per (model, quant). `quant` NULL/unknown falls back to the model default,
 * making this a strict superset of idletoken_needed_bytes. */
uint64_t idletoken_needed_bytes_quant(const idletoken_model_spec *model,
                                   const char *quant,
                                   uint32_t ctx_size, int n_nodes);

/* Mode decision (docs/architecture.md §5):
 *   Σ usable_vram              >= needed  → GPU_ONLY
 *   Σ (usable_vram+usable_ram) >= needed  → HYBRID, but every node must keep
 *                                           ≥4 GiB usable VRAM (CUDA context
 *                                           + workspace + ≥1 GPU layer)
 *   else                                  → REFUSE
 * Unified-memory nodes contribute max(vram, ram) exactly once — the pool is
 * the same physical memory, summing both would double-count.
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
 * `mode` sets the sizing weight: GPU_ONLY splits by usable VRAM (all layers in
 * fast device memory); HYBRID splits by total usable capacity (VRAM+RAM) so a
 * node holds layers up to its combined memory and offloads the overflow.
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
 *   unified == 0 (discrete GPU): usable = vram_usable + ram_usable.
 *       llama.cpp keeps -ngl layers in VRAM and the remainder in host RAM;
 *       both pools genuinely hold weights, so both count. The probe already
 *       subtracted other processes + safety margins (idletoken_resource.h).
 *
 *   unified == 1 (Apple Silicon / Grace): usable = max(vram_usable,
 *       ram_usable), counted ONCE — one physical pool, summing would
 *       double-count. On macOS the probe reports vram_usable =
 *       min(Metal recommendedMaxWorkingSetSize, ram_usable) − workspace
 *       reserve (src/common/resource.c, Darwin probe_gpu): the working-set
 *       recommendation IS Apple's "total minus OS reserve" number (measured
 *       ~74% of physical on the 16 GiB M4). Exact per-machine calibration of
 *       the workspace constant is TODO (v2 plan §1.3) — the structure is here,
 *       the constant is a reasoned reuse of the CUDA workspace reserve.
 */

#define IDLETOKEN_LLPLAN_MAX_NODES 16

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
} idletoken_llm_model_size;

/* Bytes that must be RESIDENT (allocated, not evictable) for the model to run
 * at all, on `n_nodes` machines at `ctx_size`: KV cache + per-node engine
 * overhead. Deliberately NOT the weights: llama.cpp mmaps them, so they are
 * page cache the kernel reclaims under pressure — measured 2026-08-16, a
 * 222 GiB model served from a 119 GiB machine with MemAvailable never below
 * 113 GiB (docs/resource-budget-rethink-2026-08.md §5). */
uint64_t idletoken_llama_hard_need(const idletoken_llm_model_size *model,
                                   uint32_t ctx_size, int n_nodes);

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

    /* Expected speed, decided separately from feasibility (2026-08-16).
     * 1 = the working set fits in memory, so weights are read from cache.
     * 0 = it runs, but pages stream from disk each token — measured 0.91 tok/s
     *     for GLM-5.2 at 1.9x over-subscription. Honest to offer, dishonest to
     *     present as equivalent, and it USED to be reported as a plain refusal. */
    int    working_set_fits;
    uint64_t hard_need_bytes;    /* what must be resident (KV + per-node overhead) */
    uint64_t working_set_bytes;  /* what memory should cache for full speed */

    char why[512];        /* human-readable decision / refusal reason */
} idletoken_llama_plan;

/* Decide how to run `model` on `nodes` (n of them, coordinator = index of the
 * node this coordinator process runs on).
 *
 *   - Fits the coordinator → SINGLE (hard invariant #5: fits → don't
 *     cluster). The coordinator is the only machine that can legally run
 *     single: layer 0 + embedding may not move to a worker, so a strong
 *     worker never substitutes. `allow_small_cluster` (the acceptance
 *     harness's IDLETOKEN_ALLOW_SMALL_CLUSTER=1) forces CLUSTER even then —
 *     test vehicle only; callers plumb the env var, this function reads none.
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
                            int allow_small_cluster,
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
uint32_t idletoken_llama_fit_ctx(uint64_t usable_bytes,
                                 const idletoken_llm_model_size *model,
                                 uint32_t ctx_want, uint32_t ctx_floor);

/* The usable-memory formula documented above, exposed so the coordinator's
 * preflight and the planner can never drift apart. */
uint64_t idletoken_llama_node_usable(const idletoken_node_mem *node);

/* The pool the KV cache actually LIVES IN on this node — which is NOT the same
 * as what the model can be laid out across (idletoken_llama_node_usable).
 *
 *   unified == 1 → the usable pool. One physical memory; the two answers
 *                  coincide, which is why this distinction stayed invisible on
 *                  every Mac we develop on.
 *   unified == 0 → vram_usable ONLY. We spawn the engine with -ngl 99, so the
 *                  KV cache is allocated on the GPU; host RAM holds none of it.
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
