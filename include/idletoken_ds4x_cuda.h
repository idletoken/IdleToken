/* idletoken_ds4x_cuda.h — CUDA acceleration for the ds4x forward.
 *
 * Scope (small-model-design.md §S-C): the ds4x CPU forward is the numeric
 * REFERENCE and stays the oracle. This module moves the part that actually
 * costs time — dequantize-and-matvec over the big projections (q/k/v/attn_out
 * and the FFN gate/up/down) — onto the GPU, keeping each weight resident in
 * VRAM in its ON-DISK QUANTIZED form (a Q4_K_M 8B model is ~4.7 GB, so it fits
 * a 12-16 GB card whole; dequant happens per block inside the kernel).
 *
 * Norms / rope / softmax / the attention reduction stay on the CPU for now:
 * they are a few percent of the time and keeping them shared with the
 * reference path means the GPU can never silently diverge on them.
 *
 * Contract: a CUDA matvec must match the CPU matvec within fp32 tolerance
 * (src/tools/ds4x_cuda_test.cu is that gate). If CUDA is unavailable at
 * runtime every entry point degrades to "not available" and the caller keeps
 * using the CPU path — never a silent wrong answer.
 *
 * C-callable (extern "C"); implementation is .cu compiled by nvcc.
 */
#ifndef IDLETOKEN_DS4X_CUDA_H
#define IDLETOKEN_DS4X_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>   /* getenv for ds4x_gpu_wanted */

#ifdef __cplusplus
extern "C" {
#endif

/* 1 = a usable CUDA device is present (and this build has CUDA). */
int ds4x_cuda_available(void);

/* Should the ds4x backend use the GPU? YES whenever a usable device exists —
 * opt out with IDLETOKEN_DS4X_CPU=1. This used to be opt-IN (IDLETOKEN_DS4X_CUDA=1),
 * which silently left every small model on the CPU for anyone who did not know
 * to set an env var, GUI users included. The old variable is still accepted (and
 * harmless) so existing scripts and docs keep working.
 *
 * `static inline` so both the model loader and the runner share one rule; the
 * two disagreeing is how the recurrence once stayed on the CPU while the
 * projections went to VRAM. */
static inline int ds4x_gpu_wanted(void) {
    const char *cpu = getenv("IDLETOKEN_DS4X_CPU");
    return !(cpu && cpu[0] == '1');
}

/* Cap on how many bytes of weights this process may keep in VRAM. 0 = no cap
 * (the standalone CLI tools, which own the whole card). The worker always sets
 * it from its probed usable VRAM minus headroom, so the per-machine limit the
 * user typed in the client's settings panel actually binds: before this existed
 * the cap only shrank the number the worker REPORTED, while the upload loop
 * below happily filled the card until cudaMalloc failed. Uploads past the cap
 * are skipped (that weight stays on the CPU — correct, just slower). */
void ds4x_cuda_set_budget(uint64_t bytes);

/* Opaque handle to one weight matrix resident in device memory. */
typedef struct ds4x_cuda_wt ds4x_cuda_wt;

/* Upload a (possibly quantized) weight matrix to VRAM.
 *   host    : the raw bytes exactly as ds4x_wt.data holds them
 *   type    : ggml type id (0=F32, 1=F16, 8=Q8_0, 2=Q4_0, 12=Q4_K, 13=Q5_K, 14=Q6_K)
 *   n_out   : rows, n_in: columns (n_in must be a multiple of the block size)
 * Returns NULL on OOM / unsupported type (caller falls back to CPU). */
ds4x_cuda_wt *ds4x_cuda_upload(const void *host, uint32_t type,
                               uint32_t n_out, uint32_t n_in);
void ds4x_cuda_free(ds4x_cuda_wt *w);

/* y[0..n_out) = W · x[0..n_in). Host pointers for x/y (copied in/out).
 * Returns 0, or -1 if the handle is NULL / a CUDA call failed. */
int ds4x_cuda_matvec(const ds4x_cuda_wt *w, const float *x, float *y);

/* Matvec over a SUB-RANGE of the same weight: rows [elem_off/n_in, +n_out).
 * Lets one stacked [n_expert][...] MoE tensor serve every expert from a SINGLE
 * device buffer — a 128-expert, 48-layer model would otherwise want ~18k
 * handles (and 3x that many cudaMallocs) to put its experts in VRAM.
 * elem_off must be a whole number of rows. Returns 0, or -1. */
int ds4x_cuda_matvec_off(const ds4x_cuda_wt *w, uint64_t elem_off,
                         uint32_t n_out, const float *x, float *y);

/* Y = X · Wᵀ for a WHOLE CHUNK: X is [n_tokens][n_in], Y is [n_tokens][n_out],
 * both host, both row-major and contiguous.
 *
 * This exists because per-token matvec re-streams the entire model once per
 * token. Measured (design doc §4g, Qwen3.5-0.8B, 540-token prefill): 112,535
 * matvec calls, 261 GB of weight reads, ~87 GB/s — not even bandwidth-bound,
 * just launch- and traffic-bound. Here each weight row is fetched once and
 * used for every token in the block, so weight traffic falls by the block
 * size instead of the token count.
 *
 * Bit-exactness across chunk sizes is a REQUIREMENT, not a nicety: a token's
 * accumulation path must not depend on how many other tokens share the launch,
 * or one-shot prefill would stop matching token-by-token decode. Each token
 * gets its own accumulator and its own reduction; nothing is shared.
 * Returns 0, or -1. */
int ds4x_cuda_matmul(const ds4x_cuda_wt *w, const float *X, float *Y,
                     uint32_t n_tokens);

/* Batched-matmul time, split like ds4x_cuda_stats. `rows` counts token-rows so
 * it is comparable with the matvec call count. */
void ds4x_cuda_matmul_stats(double *ms_kernel, double *ms_total,
                            uint64_t *calls, uint64_t *rows);

/* ---- Gated DeltaNet recurrence (linear-attention layers) -----------------
 * The delta-rule recurrence is strictly sequential in t and, once the
 * projections moved to the GPU, became the dominant term of a linear layer
 * (measured, design doc §4f: 45.7% of the linear path after the CPU loop was
 * already reordered). It is memory-bound on a [k_dim][v_dim] state per head,
 * so the win comes from keeping that state RESIDENT IN VRAM across tokens and
 * chunks — one handle per linear layer, living as long as the layer's cache.
 *
 * Parallel shape: one block per v-head, one thread per v_dim element. Each
 * thread owns column d of the state, so both passes over S are coalesced
 * across the block and no cross-thread reduction is needed. The t loop stays
 * inside the kernel — a launch per token would be pure overhead.
 *
 * The CPU implementation (gdn_recur_cpu in ds4x_forward.c) remains the
 * reference; IDLETOKEN_DS4X_GDN_CHECK=1 runs both and diffs them on real weights.
 */
typedef struct ds4x_cuda_gdn ds4x_cuda_gdn;

/* Allocate one layer's state [v_heads][k_dim][v_dim], zeroed. v_heads must be
 * a multiple of k_heads (linear-path GQA sharing). NULL on OOM/no device. */
ds4x_cuda_gdn *ds4x_cuda_gdn_create(uint32_t k_heads, uint32_t v_heads,
                                    uint32_t k_dim, uint32_t v_dim);
void ds4x_cuda_gdn_free(ds4x_cuda_gdn *g);

/* Zero the state — a new sequence. A recurrent state has no position index, so
 * skipping this leaks the previous request into the next one (real bug,
 * 2026-07-28: same prompt at temperature 0 gave different tokens). */
int ds4x_cuda_gdn_zero(ds4x_cuda_gdn *g);

/* Advance the state over `n_tokens` and write the attention core out.
 *   cnv  : [n_tokens][k_heads*k_dim*2 + v_heads*v_dim] — per token
 *          [q̂ | k̂ | v], q,k ALREADY L2-normalised per k-head
 *   beta : [n_tokens][v_heads]   sigmoid(b)
 *   decay: [n_tokens][v_heads]   exp(g)
 *   core : [n_tokens][v_heads*v_dim] output, already scaled by 1/sqrt(v_dim)
 * Host pointers; staged through pinned buffers. Returns 0, or -1 (state
 * untouched-or-partial → caller must treat the layer as failed, not fall back
 * silently: the state may have advanced). */
int ds4x_cuda_gdn_run(ds4x_cuda_gdn *g, uint32_t n_tokens, const float *cnv,
                      const float *beta, const float *decay, float *core);

/* Copy the state to/from host memory (tests, debugging, future migration). */
int ds4x_cuda_gdn_get_state(const ds4x_cuda_gdn *g, float *state);
int ds4x_cuda_gdn_set_state(ds4x_cuda_gdn *g, const float *state);

/* Recurrence time, split like ds4x_cuda_stats: kernel vs wall (copies+sync). */
void ds4x_cuda_gdn_stats(double *ms_kernel, double *ms_total, uint64_t *calls);

/* Total bytes currently resident in VRAM through this module (reporting). */
uint64_t ds4x_cuda_bytes_resident(void);

/* Where matvec time actually goes, accumulated since the last reset. Split so
 * "is it the kernel or the transfers?" is measured, not guessed — that
 * question was mis-answered three times by inspection alone.
 *   ms_kernel : GPU time inside the kernel (cudaEvent)
 *   ms_total  : wall time inside ds4x_cuda_matvec (copies + launch + sync)
 *   calls     : number of matvec invocations */
void ds4x_cuda_stats(double *ms_kernel, double *ms_total, uint64_t *calls);
void ds4x_cuda_stats_reset(void);

/* Startup cost split: cudaMalloc vs host→device copy, accumulated over every
 * ds4x_cuda_upload. Model load is ~all upload, so this says which half to fix. */
void ds4x_cuda_upload_stats(double *ms_malloc, double *ms_h2d);

/* Human-readable device name + last error (diagnostics; never NULL). */
const char *ds4x_cuda_device_name(void);
const char *ds4x_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_DS4X_CUDA_H */
