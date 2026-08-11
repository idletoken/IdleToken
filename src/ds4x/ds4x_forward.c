/* clock_gettime under -std=c99 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
/* ds4x_forward.c — CPU reference forward for the generic MLA-MoE layer
 * (Phase B.2). Correctness-first fp32; the CUDA path lands after this matches
 * scripts/ds4x_ref.py (see the math spec there — the two files mirror each
 * other operation for operation). No allocations in the hot path beyond one
 * scratch block sized from the config. */
#include "idletoken_ds4x.h"
#include "idletoken_ds4x_quant.h"
#ifdef IDLETOKEN_DS4X_CUDA
#include "idletoken_ds4x_cuda.h"
#endif

#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4X_EPS 1e-6f

/* Why the last layer forward returned -1.
 *
 * A bare -1 costs a debugging round every time: the caller can only report
 * "forward failed", and the branches behind it are qualitatively different —
 * "this cache slot is NULL" means the config or the plan is wrong, while
 * "malloc failed" means the box is out of memory and nothing about the model
 * is broken. The unexplained win-a transient (design doc §4e,
 * `ds4x_runner_run(pos0=0,n=19) failed`) is exactly the case this exists for:
 * without it, telling those two apart needs source reading, and by then the
 * machine has usually been rebooted. */
static char g_fwd_err[192];

const char *ds4x_forward_last_error(void) {
    return g_fwd_err[0] ? g_fwd_err : "(no detail recorded)";
}

/* Record a reason and return -1, so failure sites stay one line. */
static int fwd_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_fwd_err, sizeof g_fwd_err, fmt, ap);
    va_end(ap);
    return -1;
}

static float vec_dot(const float *a, const float *b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

/* y[o] = Σ_i W[o][i] x[i]   (W row-major [n_out][n_in], plain fp32). */
static void matvec_f32(const float *W, const float *x, float *y,
                       uint32_t n_out, uint32_t n_in) {
    for (uint32_t o = 0; o < n_out; o++)
        y[o] = vec_dot(W + (size_t)o * n_in, x, n_in);
}

/* Row `o` of a possibly-quantized weight, into buf (fp32 rows return in place,
 * no copy). Quant rows require n_in to be a block multiple — always true for
 * real GGUF matrices (n_embd / ff / lora ranks are ×32). */
static const float *wt_row(ds4x_wt w, size_t o, uint32_t n_in, float *buf) {
    if (w.type == 0) return (const float *)w.data + o * (size_t)n_in;
    const uint32_t bc = ds4x_type_block_count(w.type);
    const uint64_t bb = ds4x_type_block_bytes(w.type);
    const uint8_t *p = (const uint8_t *)w.data + (o * (n_in / bc)) * bb;
    ds4x_dequant_row(w.type, p, buf, n_in, NULL, 0);
    return buf;
}

/* Quant-aware matvec: dequantizes ONE ROW at a time into rowbuf[n_in].
 * When the weight has a VRAM mirror (CUDA build + upload succeeded) the whole
 * matvec runs on the GPU instead; the CPU path below remains the reference the
 * GPU is validated against (src/tools/ds4x_cuda_test.c). */
static void matvec_q(ds4x_wt w, const float *x, float *y,
                     uint32_t n_out, uint32_t n_in, float *rowbuf) {
#ifdef IDLETOKEN_DS4X_CUDA
    if (w.dev && ds4x_cuda_matvec_off((const ds4x_cuda_wt *)w.dev,
                                      w.dev_elem_off, n_out, x, y) == 0) return;
#endif
    for (uint32_t o = 0; o < n_out; o++)
        y[o] = vec_dot(wt_row(w, o, n_in, rowbuf), x, n_in);
}

/* Same weight, a WHOLE CHUNK of tokens: X [n_tokens][n_in] → Y [n_tokens][n_out].
 *
 * On the CPU this is the same arithmetic as n_tokens calls to matvec_q — every
 * output element is the same dot product summed in the same order, so results
 * are bit-identical — but each quantized row is DEQUANTIZED ONCE instead of
 * once per token. On the GPU it becomes one launch that reads each weight once
 * for the whole chunk (see ds4x_cuda_matmul): per-token matvec re-streamed the
 * entire model once per token, which measured as 112,535 launches and 261 GB
 * of weight traffic for a 540-token prefill (design doc §4g).
 *
 * Callers must hand it CONTIGUOUS [n_tokens][n_in] input and [n_tokens][n_out]
 * output — that requirement is why a few call sites below grew a chunk buffer
 * where they used to reuse one row of scratch. */
static void matmul_q(ds4x_wt w, const float *X, float *Y, uint32_t n_tokens,
                     uint32_t n_out, uint32_t n_in, float *rowbuf) {
#ifdef IDLETOKEN_DS4X_CUDA
    /* Whole tensors only: the batched kernel has no sub-range form, and MoE
     * experts do not batch anyway (each token routes to different ones). */
    if (w.dev && w.dev_elem_off == 0 &&
        ds4x_cuda_matmul((const ds4x_cuda_wt *)w.dev, X, Y, n_tokens) == 0)
        return;
#endif
    for (uint32_t o = 0; o < n_out; o++) {
        const float *row = wt_row(w, o, n_in, rowbuf);
        for (uint32_t t = 0; t < n_tokens; t++)
            Y[(size_t)t * n_out + o] = vec_dot(row, X + (size_t)t * n_in, n_in);
    }
}

/* Sub-tensor starting `elem_off` elements in (block-aligned for quant). Used to
 * pick expert e out of a stacked [n_expert][...] weight. */
static ds4x_wt wt_slice(ds4x_wt w, size_t elem_off) {
    ds4x_wt r = w;
    /* The VRAM view has to move with the CPU view. Inheriting `dev` unchanged
     * (what a plain struct copy does) would point every expert at row 0 of the
     * whole stack AND write n_expert*n_ff floats into an n_ff buffer. */
    r.dev_elem_off = w.dev_elem_off + (uint64_t)elem_off;
    if (w.type == 0) { r.data = (const float *)w.data + elem_off; return r; }
    const uint32_t bc = ds4x_type_block_count(w.type);
    const uint64_t bb = ds4x_type_block_bytes(w.type);
    r.data = (const uint8_t *)w.data + (elem_off / bc) * bb;
    return r;
}

static void rmsnorm(const float *x, const float *w, float *y, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + DS4X_EPS);
    for (uint32_t i = 0; i < n; i++) y[i] = x[i] * w[i] * inv;
}

/* neox-style rope in place: rotate pairs (i, i+d/2). */
static void rope_neox(float *v, uint32_t d, uint32_t pos, float base) {
    const uint32_t half = d / 2;
    for (uint32_t i = 0; i < half; i++) {
        const float theta = (float)pos * powf(base, -2.0f * (float)i / (float)d);
        const float c = cosf(theta), s = sinf(theta);
        const float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* swiglu FFN: y = Wd · (silu(Wg·x) ⊙ (Wu·x)), scratch needs 2*ff floats;
 * rowbuf holds one dequantized weight row (size ≥ max(n_embd, ff)). */
static void swiglu(ds4x_wt Wg, ds4x_wt Wu, ds4x_wt Wd,
                   const float *x, float *y, uint32_t ff, uint32_t n_embd,
                   float *scratch, float *rowbuf) {
#ifdef IDLETOKEN_DS4X_CUDA
    /* All three weights resident and whole (dev_elem_off 0 — a sliced MoE
     * expert has no fused form) → one GPU call for the whole block instead of
     * three matmuls with a host round trip between each. */
    if (ds4x_fuse_wanted() && Wg.dev && Wu.dev && Wd.dev &&
        !Wg.dev_elem_off && !Wu.dev_elem_off && !Wd.dev_elem_off &&
        ds4x_cuda_swiglu((const ds4x_cuda_wt *)Wg.dev, (const ds4x_cuda_wt *)Wu.dev,
                         (const ds4x_cuda_wt *)Wd.dev, x, y, 1) == 0) return;
#endif
    float *g = scratch, *u = scratch + ff;
    matvec_q(Wg, x, g, ff, n_embd, rowbuf);
    matvec_q(Wu, x, u, ff, n_embd, rowbuf);
    for (uint32_t i = 0; i < ff; i++) g[i] = silu(g[i]) * u[i];
    matvec_q(Wd, g, y, n_embd, ff, rowbuf);
}

/* Same swiglu over a whole chunk. A dense FFN is entirely token-independent —
 * no cache, no position — so it batches with nothing to prove beyond the
 * matmul contract itself. Scratch is [n_tokens][2*ff]; the caller owns it
 * because the chunk sizes it, not the config.
 * (The MoE branch stays per-token on purpose: each token routes to a different
 * set of experts, so there is no shared weight to amortize.) */
static void swiglu_chunk(ds4x_wt Wg, ds4x_wt Wu, ds4x_wt Wd,
                         const float *X, float *Y, uint32_t n_tokens,
                         uint32_t ff, uint32_t n_embd,
                         float *scratch, float *rowbuf) {
#ifdef IDLETOKEN_DS4X_CUDA
    if (ds4x_fuse_wanted() && Wg.dev && Wu.dev && Wd.dev &&
        !Wg.dev_elem_off && !Wu.dev_elem_off && !Wd.dev_elem_off &&
        ds4x_cuda_swiglu((const ds4x_cuda_wt *)Wg.dev, (const ds4x_cuda_wt *)Wu.dev,
                         (const ds4x_cuda_wt *)Wd.dev, X, Y, n_tokens) == 0) return;
#endif
    float *g = scratch, *u = scratch + (size_t)n_tokens * ff;
    matmul_q(Wg, X, g, n_tokens, ff, n_embd, rowbuf);
    matmul_q(Wu, X, u, n_tokens, ff, n_embd, rowbuf);
    for (size_t i = 0; i < (size_t)n_tokens * ff; i++) g[i] = silu(g[i]) * u[i];
    matmul_q(Wd, g, Y, n_tokens, n_embd, ff, rowbuf);
}

static float softplus_f(float x) {
    /* stable: log1p(exp(-|x|)) + max(x,0) */
    return log1pf(expf(-fabsf(x))) + (x > 0.0f ? x : 0.0f);
}

/* Gated DeltaNet attention sublayer, transcribed from
 * docs/linear-attention-design.md §2 (verified against transformers
 * Qwen3NextGatedDeltaNet). Reads pre-normed hidden `normed` [n_tokens][n_embd],
 * writes the attention output into `attn_out`, and advances the recurrent
 * state + conv window in `cache`.
 *
 * Two passes on purpose: the causal conv needs every RAW channel row of the
 * chunk (plus the carried window), so raws are materialised first and the
 * per-token recurrence runs after. Carrying conv_win + state is what makes
 * one-shot prefill identical to token-by-token decode. */
/* IDLETOKEN_GDN_DUMP=1 prints per-op sums for layer 0 so they can be diffed
 * against llama.cpp's llama-eval-callback on the same GGUF + prompt. This is
 * the only reliable oracle for the GDN path (design doc §7.1). */
/* IDLETOKEN_DS4X_PROF=1 accumulates wall time in the FOUR stages of a linear
 * layer, because "the recurrence" used to be timed together with the causal
 * conv and the out_proj matvec and so over-reported itself. Decide whether a
 * stage is worth a kernel by MEASURING — four performance guesses in this
 * project were wrong before measurement.
 *   proj : the four input matvecs (GPU-eligible, already on GPU)
 *   conv : causal depthwise conv + silu + gates + L2 norm  (CPU)
 *   rec  : the delta-rule recurrence itself                (CPU or GPU)
 *   post : gated RMSNorm + out_proj matvec                 (matvec on GPU) */
static double g_t_proj = 0.0, g_t_conv = 0.0, g_t_rec = 0.0, g_t_post = 0.0;
static int gdn_prof_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("IDLETOKEN_DS4X_PROF"); v = (e && *e == '1'); }
    return v;
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
void ds4x_prof_report(double *proj_s, double *conv_s, double *rec_s, double *post_s) {
    if (proj_s) *proj_s = g_t_proj;
    if (conv_s) *conv_s = g_t_conv;
    if (rec_s)  *rec_s  = g_t_rec;
    if (post_s) *post_s = g_t_post;
}

static const ds4x_layer_weights *g_dump_layer = NULL;
static int gdn_dump_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("IDLETOKEN_GDN_DUMP"); v = (e && *e == '1'); }
    return v;
}
static void gdn_sum(const char *tag, const float *p, size_t n) {
    double s = 0.0; for (size_t i = 0; i < n; i++) s += p[i];
    /* first elements too: a plain sum hides errors behind cancellation. */
    fprintf(stderr, "  GDN %-22s sum = %12.6f  [%.4f %.4f %.4f]\n",
            tag, s, p[0], n > 1 ? p[1] : 0.0f, n > 2 ? p[2] : 0.0f);
}

/* IDLETOKEN_DS4X_GDN_CHECK=1 runs the CPU recurrence ALONGSIDE the GPU one on real
 * weights and reports the worst per-element divergence seen so far. The two
 * keep SEPARATE states (host vs device) from the same zeroed start, so drift
 * accumulates instead of being re-synced every chunk — a wrong kernel cannot
 * hide behind one lucky chunk. Costs the CPU time it saves; diagnostics only. */
static int gdn_check_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("IDLETOKEN_DS4X_GDN_CHECK"); v = (e && *e == '1'); }
    return v;
}
static double g_gdn_check_max = 0.0;
static uint64_t g_gdn_check_n = 0;
static void gdn_check_report(const float *ref, const float *got, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)ref[i] - (double)got[i];
        if (d < 0) d = -d;
        if (d > mx) mx = d;
    }
    g_gdn_check_n++;
    if (mx > g_gdn_check_max) {
        g_gdn_check_max = mx;
        fprintf(stderr, "  GDN cpu/gpu recurrence: new max |Δ| = %.3e "
                        "(chunk %llu)\n", mx, (unsigned long long)g_gdn_check_n);
    }
}
void ds4x_gdn_check_report(double *max_abs, uint64_t *chunks) {
    if (max_abs) *max_abs = g_gdn_check_max;
    if (chunks)  *chunks  = g_gdn_check_n;
}

/* The delta-rule recurrence for one chunk — the whole of it, and nothing else.
 * `cnv` holds per token [q̂(kd) | k̂(kd) | v(vd)] with q,k already L2-normalised
 * per k-head, `bet`/`dec` the per-v-head scalars, and S the [vh][kdim][vdim]
 * state, advanced in place. Everything else (conv, gates, norms, projections)
 * happens outside, so this stage — and only this stage — is what the GPU
 * kernel replaces (ds4x_cuda_gdn_run takes exactly these arguments).
 *
 * Loop order is i-outer / d-inner: S is [kdim][vdim] row-major, so this touches
 * it contiguously. The per-element summation order (i ascending) is what makes
 * the result reproducible; do not reorder it without re-running the oracle. */
void ds4x_gdn_recur_cpu(uint32_t n_tokens, uint32_t kh, uint32_t vh,
                        uint32_t kdim, uint32_t vdim, float *S,
                        const float *cnv, const float *bet, const float *dec,
                        float *core_all, float *memv, float *updv) {
    float *owned = NULL;
    if (!memv || !updv) {
        owned = (float *)malloc((size_t)vdim * 2 * sizeof(float));
        if (!owned) return;
        memv = owned; updv = owned + vdim;
    }
    const uint32_t kd = kh * kdim, vd = vh * vdim;
    const uint32_t stride = kd * 2u + vd;
    const float oscale = 1.0f / sqrtf((float)vdim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *base = cnv + (size_t)t * stride;
        const float *bt = bet + (size_t)t * vh, *dt = dec + (size_t)t * vh;
        for (uint32_t h_v = 0; h_v < vh; h_v++) {
            /* Which key head serves value head h_v when k_heads < v_heads.
             * It is STRIDED (h_v % k_heads), not repeat-interleaved
             * (h_v / grp) — value heads h and h+k_heads share key head h.
             *
             * This was measured, not assumed. With a zero state and one token
             * the rule collapses to out[h] = (q̂·k̂)_{h_k}·β_h·v_h/√d_v, so the
             * two candidate mappings give two computable totals: 0.8365 for
             * h_v/grp and 0.8986 for h_v%k_heads, against llama.cpp's 0.8968
             * on the same GGUF (Qwen3.5-4B, prompt "Hello"). 0.2% apart is
             * Q4_K noise; 6.7% is a different answer.
             *
             * Head 0 maps to key head 0 under BOTH rules, which is why every
             * cheap check passed: the 0.8B has k_heads == v_heads so the two
             * are identical there, and on the 4B the first head's numbers
             * matched the oracle while the model quietly talked nonsense. */
            const uint32_t h_k = h_v % kh;
            const float beta = bt[h_v], decay = dt[h_v];
            const float *qh   = base + (size_t)h_k * kdim;
            const float *kh_p = base + kd + (size_t)h_k * kdim;
            const float *vv   = base + kd * 2u + (size_t)h_v * vdim;
            float *Sh = S + (size_t)h_v * kdim * vdim;
            float *outh = core_all + (size_t)t * vd + (size_t)h_v * vdim;

            /* Delta rule, in llama.cpp's exact order (ggml_compute_forward_
             * gated_delta_net):
             *   1. decay the WHOLE state:  S *= exp(g)
             *   2. mem  = S_decayed^T k
             *   3. S   += k ⊗ beta*(v - mem)
             *   4. out  = S^T q / sqrt(v_head_dim)
             * Two things here were wrong before and only the oracle caught them:
             * `mem` must come from the DECAYED state (invisible at t=0 where
             * S=0, wrong from t=1 on), and the output carries a 1/sqrt(d_v)
             * scale that the transformers reference folds in elsewhere.
             * Since decay is a per-head scalar, step 1 folds into step 2 as
             * mem = decay * (S^T k) — no extra pass over S. */
            for (uint32_t d = 0; d < vdim; d++) memv[d] = 0.0f;
            for (uint32_t i = 0; i < kdim; i++) {
                const float ki = kh_p[i];
                const float *Si = Sh + (size_t)i * vdim;
                for (uint32_t d = 0; d < vdim; d++) memv[d] += Si[d] * ki;
            }
            for (uint32_t d = 0; d < vdim; d++) {
                updv[d] = beta * (vv[d] - memv[d] * decay);
                outh[d] = 0.0f;
            }
            for (uint32_t i = 0; i < kdim; i++) {
                const float ki = kh_p[i], qi = qh[i];
                float *Si = Sh + (size_t)i * vdim;
                for (uint32_t d = 0; d < vdim; d++) {
                    const float sv = Si[d] * decay + ki * updv[d];
                    Si[d] = sv;
                    outh[d] += sv * qi;                /* out = S^T q */
                }
            }
            for (uint32_t d = 0; d < vdim; d++) outh[d] *= oscale;
        }
    }
    free(owned);
}

static int gdn_attn_cpu(const ds4x_config *cfg, const ds4x_layer_weights *w,
                        const float *normed, uint32_t n_tokens,
                        ds4x_kv_cache *cache, float *attn_out) {
    const uint32_t n_embd = cfg->n_embd;
    const uint32_t kh = cfg->lin_k_heads, vh = cfg->lin_v_heads;
    const uint32_t kdim = cfg->lin_k_dim, vdim = cfg->lin_v_dim;
    const uint32_t kd = kh * kdim, vd = vh * vdim;
    const uint32_t conv_ch = kd * 2 + vd;
    const uint32_t K = cfg->conv_kernel;
    if (!cache->state || !cache->conv_win || kh == 0 || vh % kh)
        return fwd_fail("linear layer unusable: state=%s conv_win=%s k_heads=%u "
                        "v_heads=%u (v_heads must be a non-zero multiple of k_heads)",
                        cache->state ? "y" : "NULL", cache->conv_win ? "y" : "NULL",
                        kh, vh);

    const size_t rowmax = (n_embd > vd ? n_embd : vd);
    /* per-chunk buffers */
    float *raw  = (float *)malloc((size_t)n_tokens * conv_ch * sizeof(float)); /* pre-conv q,k,v */
    float *cnv  = (float *)malloc((size_t)n_tokens * conv_ch * sizeof(float)); /* post-conv q̂,k̂,v */
    float *zbuf = (float *)malloc((size_t)n_tokens * vd * sizeof(float));      /* z gate         */
    /* b and a are separate buffers, not two halves of one row: a batched
     * matmul writes [n_tokens][n_out] contiguously, so interleaving them per
     * token would need two strided writes. */
    float *bbuf = (float *)malloc((size_t)n_tokens * vh * sizeof(float));      /* b logits       */
    float *abuf = (float *)malloc((size_t)n_tokens * vh * sizeof(float));      /* a logits       */
    float *bet  = (float *)malloc((size_t)n_tokens * vh * sizeof(float));      /* sigmoid(b)     */
    float *dec  = (float *)malloc((size_t)n_tokens * vh * sizeof(float));      /* exp(g)         */
    float *core = (float *)malloc((size_t)n_tokens * vd * sizeof(float));
    float *rowbuf = (float *)malloc(rowmax * sizeof(float));
    float *memv = (float *)malloc((size_t)vdim * sizeof(float));   /* S^T k   */
    float *updv = (float *)malloc((size_t)vdim * sizeof(float));   /* β(v−mem) */
    if (!raw || !cnv || !zbuf || !bbuf || !abuf || !bet || !dec || !core ||
        !rowbuf || !memv || !updv) {
        free(raw); free(cnv); free(zbuf); free(bbuf); free(abuf); free(bet);
        free(dec); free(core); free(rowbuf); free(memv); free(updv);
        return fwd_fail("oom: linear-layer chunk buffers for %u tokens "
                        "(~%.1f MB)", n_tokens,
                        (double)((size_t)n_tokens * (conv_ch * 2 + vd * 2 + vh * 4)
                                 * sizeof(float)) / 1048576.0);
    }

    /* ---- pass 1: projections; stash raw conv channels + z + (b,a) ---- */
    const int prof = gdn_prof_on();
    const double t0 = prof ? now_s() : 0.0;
    /* All four read the same `normed`, so fused they cost one upload and one
     * download instead of four of each. */
    int projected = 0;
#ifdef IDLETOKEN_DS4X_CUDA
    {
        const ds4x_wt src[4] = { w->in_proj_qkv, w->in_proj_z, w->in_proj_b, w->in_proj_a };
        float *dst[4] = { raw, zbuf, bbuf, abuf };
        const ds4x_cuda_wt *dw[4];
        int ok = ds4x_fuse_wanted();
        for (int i = 0; i < 4 && ok; i++) {
            if (!src[i].dev || src[i].dev_elem_off) ok = 0;
            else dw[i] = (const ds4x_cuda_wt *)src[i].dev;
        }
        if (ok && ds4x_cuda_proj_fanout(dw, dst, 4, normed, n_tokens) == 0)
            projected = 1;
    }
#endif
    if (!projected) {
        matmul_q(w->in_proj_qkv, normed, raw,  n_tokens, conv_ch, n_embd, rowbuf);
        matmul_q(w->in_proj_z,   normed, zbuf, n_tokens, vd,      n_embd, rowbuf);
        matmul_q(w->in_proj_b,   normed, bbuf, n_tokens, vh,      n_embd, rowbuf);
        matmul_q(w->in_proj_a,   normed, abuf, n_tokens, vh,      n_embd, rowbuf);
    }

    const int dump = gdn_dump_on() && w == g_dump_layer;
    if (dump) { gdn_sum("normed(attn_norm)", normed, n_embd);
                gdn_sum("raw qkv(node_13)", raw, conv_ch); }

    if (prof) g_t_proj += now_s() - t0;
    const double t1 = prof ? now_s() : 0.0;

    /* ---- pass 2: causal conv + gates + L2 norm, for every token ----
     * All of this is per-token-independent given `raw`, so it runs over the
     * whole chunk before the (strictly sequential) recurrence. That split is
     * what lets the recurrence — and only the recurrence — move to the GPU. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *conv = cnv + (size_t)t * conv_ch;
        /* causal depthwise conv: tap j reads chunk-relative position
         * t-(K-1-j); anything before the chunk comes from the carried window
         * (conv_win[i] holds the raw row (K-1-i) steps before the chunk). */
        for (uint32_t c = 0; c < conv_ch; c++) {
            float acc = w->conv1d_b ? w->conv1d_b[c] : 0.0f;
            for (uint32_t j = 0; j < K; j++) {
                const int rel = (int)t - (int)(K - 1 - j);
                const float src = (rel >= 0)
                    ? raw[(size_t)rel * conv_ch + c]
                    : cache->conv_win[(size_t)(K - 1 + rel) * conv_ch + c];
                acc += w->conv1d_w[(size_t)c * K + j] * src;
            }
            conv[c] = silu(acc);
        }
        float *qs = conv, *ks = conv + kd;
        const float *bt_l = bbuf + (size_t)t * vh, *at_l = abuf + (size_t)t * vh;
        /* dump BEFORE the L2 norm below rewrites q,k in place — these sums are
         * compared against llama.cpp's pre-norm conv output. */
        if (dump && t == 0) { gdn_sum("conv_silu", conv, conv_ch);
                              gdn_sum("q_conv", qs, kd); gdn_sum("k_conv", ks, kd);
                              gdn_sum("v_conv", conv + kd * 2, vd); }

        for (uint32_t h_v = 0; h_v < vh; h_v++) {
            /* gates: beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias) */
            bet[(size_t)t * vh + h_v] = 1.0f / (1.0f + expf(-bt_l[h_v]));
            /* ssm_a is used DIRECTLY — it already stores the negative
             * coefficient, there is no exp() on it. Verified against
             * llama.cpp's graph: gate = MUL(softplus(...), ssm_a), then the
             * kernel takes expf(g). The transformers reference writes
             * -exp(A_log), i.e. its checkpoint stores log|A| while the GGUF
             * converter stores -A. */
            const float gval = w->A_log[h_v] * softplus_f(at_l[h_v] + w->dt_bias[h_v]);
            dec[(size_t)t * vh + h_v] = expf(gval);
        }
        /* per-head L2 norm on q,k (eps 1e-6), in place — the recurrence then
         * reads q̂,k̂ directly instead of re-deriving them per output element. */
        for (uint32_t h_k = 0; h_k < kh; h_k++) {
            float *qh = qs + (size_t)h_k * kdim, *kp = ks + (size_t)h_k * kdim;
            float qn = 0.0f, kn = 0.0f;
            for (uint32_t i = 0; i < kdim; i++) { qn += qh[i]*qh[i]; kn += kp[i]*kp[i]; }
            qn = 1.0f / sqrtf(qn + 1e-6f);
            kn = 1.0f / sqrtf(kn + 1e-6f);
            for (uint32_t i = 0; i < kdim; i++) { qh[i] *= qn; kp[i] *= kn; }
        }
    }
    if (prof) g_t_conv += now_s() - t1;

    /* ---- pass 3: the delta-rule recurrence (GPU when the layer has a
     * device-resident state, else CPU — the CPU stays the reference) ---- */
    const double t2 = prof ? now_s() : 0.0;
    int on_gpu = 0;
#ifdef IDLETOKEN_DS4X_CUDA
    if (cache->dev_state &&
        ds4x_cuda_gdn_run((ds4x_cuda_gdn *)cache->dev_state, n_tokens,
                          cnv, bet, dec, core) == 0)
        on_gpu = 1;
#endif
    if (!on_gpu || gdn_check_on()) {
        /* In check mode both paths run from their own state (host vs device),
         * both starting zeroed, so any drift accumulates and shows up. */
        float *ref = core;
        if (on_gpu) {
            ref = (float *)malloc((size_t)n_tokens * vd * sizeof(float));
            if (!ref) ref = NULL;
        }
        if (ref) {
            ds4x_gdn_recur_cpu(n_tokens, kh, vh, kdim, vdim, cache->state,
                          cnv, bet, dec, ref, memv, updv);
            if (on_gpu) { gdn_check_report(ref, core, (size_t)n_tokens * vd); free(ref); }
        }
    }
    if (prof) g_t_rec += now_s() - t2;

    /* ---- pass 4: gated RMSNorm + out_proj, token by token ---- */
    const double t3 = prof ? now_s() : 0.0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *ct = core + (size_t)t * vd;
        if (dump && t == 0) gdn_sum("attn_output(core)", ct, vd);
        /* Gated RMSNorm — PER HEAD over v_head_dim. The real GGUF's
         * ssm_norm.weight is only v_head_dim long (shared across heads);
         * normalising over the whole v_dim is wrong (design doc §4b). */
        const float *zt = zbuf + (size_t)t * vd;
        for (uint32_t h_v = 0; h_v < vh; h_v++) {
            float *ch = ct + (size_t)h_v * vdim;
            float ss = 0.0f;
            for (uint32_t i = 0; i < vdim; i++) ss += ch[i] * ch[i];
            const float inv = 1.0f / sqrtf(ss / (float)vdim + DS4X_EPS);
            const float *zh = zt + (size_t)h_v * vdim;
            for (uint32_t i = 0; i < vdim; i++)
                ch[i] = ch[i] * w->ssm_norm[i] * inv * silu(zh[i]);
        }
        if (dump && t == 0) gdn_sum("final_output", ct, vd);
    }
    matmul_q(w->out_proj, core, attn_out, n_tokens, n_embd, vd, rowbuf);
    if (dump) gdn_sum("linear_attn_out", attn_out, n_embd);
    if (prof) g_t_post += now_s() - t3;

    /* ---- carry the last K-1 raw rows for the next chunk ---- */
    if (K > 1) {
        for (uint32_t i = 0; i < K - 1; i++) {
            const int rel = (int)n_tokens - (int)(K - 1) + (int)i;
            float *dst = cache->conv_win + (size_t)i * conv_ch;
            if (rel >= 0) memcpy(dst, raw + (size_t)rel * conv_ch, conv_ch * sizeof(float));
            else memmove(dst, cache->conv_win + (size_t)(K - 1 + rel) * conv_ch,
                         conv_ch * sizeof(float));
        }
    }

    free(raw); free(cnv); free(zbuf); free(bbuf); free(abuf); free(bet);
    free(dec); free(core); free(rowbuf); free(memv); free(updv);
    return 0;
}

int ds4x_layer_forward_cpu(const ds4x_config *cfg, uint32_t il,
                           const ds4x_layer_weights *w,
                           const float *x, uint32_t n_tokens, uint32_t pos0,
                           ds4x_kv_cache *cache, float *out) {
    g_fwd_err[0] = '\0';
    if (!cfg || !w || !x || !out || !cache || n_tokens == 0)
        return fwd_fail("bad args (cfg=%p w=%p x=%p out=%p cache=%p n=%u)",
                        (const void *)cfg, (const void *)w, (const void *)x,
                        (void *)out, (void *)cache, n_tokens);
    const uint32_t n_embd = cfg->n_embd;
    const uint32_t nope = cfg->qk_nope_head_dim, rope_d = cfg->qk_rope_head_dim;
    const uint32_t v_dim = cfg->v_head_dim, qd = nope + rope_d;
    const uint32_t kv_rank = cfg->kv_lora_rank;
    /* dense FFN when this is a leading dense block OR the model has no experts
     * (fully-dense small models like Qwen3-8B). */
    const int is_dense = (il < cfg->n_dense_lead) || (cfg->n_expert == 0);

    /* One scratch block: laid out so nothing overlaps within a token step. */
    const uint32_t ff_max = is_dense ? cfg->n_ff_dense : cfg->n_ff_exp;
    /* Largest matvec input dim → size of the one-row dequant scratch. */
    uint32_t rowmax = n_embd;
    if (cfg->q_lora_rank > rowmax) rowmax = cfg->q_lora_rank;
    if (cfg->n_head * v_dim > rowmax) rowmax = cfg->n_head * v_dim;
    if (ff_max > rowmax) rowmax = ff_max;
    const size_t scr_n =
        (size_t)n_embd                      /* cur (normed input) */
        + cfg->q_lora_rank                  /* qa */
        + (size_t)cfg->n_head * qd * 2      /* q (+ gate for qwen35) */
        + (kv_rank + rope_d)                /* kv_a out */
        + (nope > v_dim ? nope : v_dim)     /* k_nope / v reconstruction row */
        + (size_t)cfg->n_head * v_dim       /* attn head outputs */
        + (size_t)pos0 + n_tokens           /* attention scores (≤ ctx) */
        + n_embd                            /* ffn y accumulator */
        + n_embd                            /* one expert's output */
        + cfg->n_expert                     /* router probs */
        + 2u * ff_max                       /* swiglu gate/up */
        + rowmax;                           /* one dequantized weight row */
    float *scr = (float *)malloc(scr_n * sizeof(float));
    if (!scr) return fwd_fail("oom: layer scratch (%.1f MB)",
                              (double)(scr_n * sizeof(float)) / 1048576.0);
    float *cur    = scr;
    float *qa     = cur + n_embd;
    float *q      = qa + cfg->q_lora_rank;   /* holds [q|gate] when gated */
    float *kva    = q + (size_t)cfg->n_head * qd * 2;
    float *recon  = kva + (kv_rank + rope_d);
    float *hout   = recon + (nope > v_dim ? nope : v_dim);
    float *scores = hout + (size_t)cfg->n_head * v_dim;
    float *ffn_y  = scores + (size_t)pos0 + n_tokens;
    float *eout   = ffn_y + n_embd;
    float *probs  = eout + n_embd;
    float *sw     = probs + cfg->n_expert;
    float *rowbuf = sw + 2u * ff_max;

    if (out != x) memcpy(out, x, (size_t)n_tokens * n_embd * sizeof(float));

    /* Per-layer attention kind: hybrid models (Qwen3.5 = 3 linear : 1 full)
     * override the model-level default per layer. 0 = not set → default. */
    const uint8_t lt = (il < DS4X_MAX_LAYERS && cfg->layer_types[il])
                       ? cfg->layer_types[il] : cfg->attn_kind;

  if (lt == DS4X_ATTN_LINEAR) {
    /* Gated DeltaNet: norm → GDN → residual. Uses its own scratch. */
    float *normed = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));
    float *aout   = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));
    if (!normed || !aout) {
        free(normed); free(aout); free(scr);
        return fwd_fail("oom: linear-layer hidden buffers (%u tokens, %.1f MB)",
                        n_tokens, (double)((size_t)n_tokens * n_embd * 2
                                           * sizeof(float)) / 1048576.0);
    }
    for (uint32_t t = 0; t < n_tokens; t++)
        rmsnorm(out + (size_t)t * n_embd, w->attn_norm,
                normed + (size_t)t * n_embd, n_embd);
    if (il == 0) g_dump_layer = w;
    if (gdn_dump_on()) { char t[32]; snprintf(t, sizeof t, "l_in-%u", il);
                         gdn_sum(t, x, cfg->n_embd); }
    const int rc = gdn_attn_cpu(cfg, w, normed, n_tokens, cache, aout);
    if (rc == 0)
        for (uint32_t t = 0; t < n_tokens; t++) {
            float *xt = out + (size_t)t * n_embd;
            const float *at = aout + (size_t)t * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) xt[i] += at[i];
        }
    free(normed); free(aout);
    if (rc != 0) { free(scr); return -1; }   /* reason already recorded */
  } else if (lt == DS4X_ATTN_GQA) {
    /* ---- standard GQA attention (qwen3/llama): full K/V cache, optional
     * per-head Q/K RMSNorm (Qwen3), neox rope on the whole head_dim, KV heads
     * broadcast to n_head/n_head_kv query heads. Reuses the same scratch (q,
     * hout, scores, rowbuf) — kv_lora_rank/q_lora_rank are 0 so the MLA
     * regions collapse to nothing. */
    const uint32_t hdim   = cfg->head_dim;
    const uint32_t kv_dim = cfg->n_head_kv * hdim;
    const uint32_t grp    = cfg->n_head / cfg->n_head_kv;
    const float gscale    = 1.0f / sqrtf((float)hdim);
    /* qwen35 Gated Attention: q_proj emits q and its gate INTERLEAVED PER
     * HEAD — [q_h(hdim) | gate_h(hdim)] repeated n_head times, not two
     * contiguous halves. The tell in llama.cpp's graph is that the gate
     * view needs a CONT (contiguous copy) while the q view does not; a
     * plain second half would already be contiguous. `qstride` is the
     * per-head stride and is 2*hdim exactly when the gate is present. */
    const uint32_t qstride = cfg->attn_out_gate ? hdim * 2u : hdim;
    /* Chunk buffers for the batched projections. The K and V projections need
     * none: cache->k rows are kv_dim wide and consecutive positions are
     * adjacent, so cache->k + pos0*kv_dim IS the [n_tokens][kv_dim] output. */
    const size_t qn_all = (size_t)cfg->n_head * qstride;
    float *nrm_all = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));
    float *q_all   = (float *)malloc((size_t)n_tokens * qn_all * sizeof(float));
    float *ho_all  = (float *)malloc((size_t)n_tokens * cfg->n_head * hdim * sizeof(float));
    if (!nrm_all || !q_all || !ho_all) {
        free(nrm_all); free(q_all); free(ho_all); free(scr);
        return fwd_fail("oom: attention chunk buffers (%u tokens, %.1f MB)",
                        n_tokens, (double)((size_t)n_tokens
                            * (n_embd + qn_all + (size_t)cfg->n_head * hdim)
                            * sizeof(float)) / 1048576.0);
    }
    for (uint32_t t = 0; t < n_tokens; t++)
        rmsnorm(out + (size_t)t * n_embd, w->attn_norm,
                nrm_all + (size_t)t * n_embd, n_embd);
    /* q/k/v all read nrm_all, so they fan out from ONE upload. k and v write
     * straight into the KV cache, which is fine: the fan-out scatters to N
     * independent host destinations. */
    int qkv_projected = 0;
#ifdef IDLETOKEN_DS4X_CUDA
    {
        const ds4x_wt src[3] = { w->q_proj, w->k_proj, w->v_proj };
        float *dst[3] = { q_all, cache->k + (size_t)pos0 * kv_dim,
                                 cache->v + (size_t)pos0 * kv_dim };
        const ds4x_cuda_wt *dw[3];
        int ok = ds4x_fuse_wanted();
        for (int i = 0; i < 3 && ok; i++) {
            if (!src[i].dev || src[i].dev_elem_off) ok = 0;
            else dw[i] = (const ds4x_cuda_wt *)src[i].dev;
        }
        if (ok && ds4x_cuda_proj_fanout(dw, dst, 3, nrm_all, n_tokens) == 0)
            qkv_projected = 1;
    }
#endif
    if (!qkv_projected) {
        matmul_q(w->q_proj, nrm_all, q_all, n_tokens, (uint32_t)qn_all, n_embd, rowbuf);
        matmul_q(w->k_proj, nrm_all, cache->k + (size_t)pos0 * kv_dim,
                 n_tokens, kv_dim, n_embd, rowbuf);
        matmul_q(w->v_proj, nrm_all, cache->v + (size_t)pos0 * kv_dim,
                 n_tokens, kv_dim, n_embd, rowbuf);
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos = pos0 + t;
        /* qt/ht are this token's slices of the chunk buffers — deliberately
         * NOT the `q`/`hout` scratch views, which stay untouched for the MLA
         * branch below. */
        float *qt = q_all + (size_t)t * qn_all;
        float *ht = ho_all + (size_t)t * cfg->n_head * hdim;
        float *k_t = cache->k + (size_t)pos * kv_dim;
        /* per-head qk-norm (before rope) then rope */
        /* Partial rotary: qwen35 rotates only the first rope_dim of a 256-wide
         * head (rope.dimension_count=64); dims above it are left untouched.
         * rope_dim_partial == 0 means "rope the whole head" (plain qwen3). */
        const uint32_t rdim = cfg->rope_dim_partial ? cfg->rope_dim_partial : hdim;
        for (uint32_t hd = 0; hd < cfg->n_head; hd++) {
            float *q_h = qt + (size_t)hd * qstride;
            if (w->q_norm) rmsnorm(q_h, w->q_norm, q_h, hdim);
            rope_neox(q_h, rdim, pos, cfg->rope_theta);
        }
        for (uint32_t hk = 0; hk < cfg->n_head_kv; hk++) {
            float *k_h = k_t + (size_t)hk * hdim;
            if (w->k_norm) rmsnorm(k_h, w->k_norm, k_h, hdim);
            rope_neox(k_h, rdim, pos, cfg->rope_theta);
        }
        for (uint32_t hd = 0; hd < cfg->n_head; hd++) {
            const uint32_t kvh = hd / grp;
            const float *q_h = qt + (size_t)hd * qstride;
            float smax = -1e30f;
            for (uint32_t u = 0; u <= pos; u++) {
                const float *kk = cache->k + (size_t)u * kv_dim + (size_t)kvh * hdim;
                const float s = vec_dot(q_h, kk, hdim) * gscale;
                scores[u] = s;
                if (s > smax) smax = s;
            }
            float ssum = 0.0f;
            for (uint32_t u = 0; u <= pos; u++) { scores[u] = expf(scores[u] - smax); ssum += scores[u]; }
            float *o_h = ht + (size_t)hd * hdim;
            memset(o_h, 0, hdim * sizeof(float));
            for (uint32_t u = 0; u <= pos; u++) {
                const float p = scores[u] / ssum;
                const float *vv = cache->v + (size_t)u * kv_dim + (size_t)kvh * hdim;
                for (uint32_t i = 0; i < hdim; i++) o_h[i] += p * vv[i];
            }
        }
        /* Gated Attention: elementwise gate on the concatenated heads. The gate
         * is SIGMOID, not silu — verified in llama.cpp's graph for qwen35
         * (gate_sigmoid = SIGMOID(...); attn_gated = MUL(...)). The GDN z-gate
         * in the very same model IS silu; do not conflate the two. */
        if (cfg->attn_out_gate)
            for (uint32_t hd = 0; hd < cfg->n_head; hd++) {
                const float *g_h = qt + (size_t)hd * qstride + hdim;
                float *o_h = ht + (size_t)hd * hdim;
                for (uint32_t i = 0; i < hdim; i++)
                    o_h[i] *= 1.0f / (1.0f + expf(-g_h[i]));
            }
    }
    /* One output projection for the whole chunk, into nrm_all (done with it). */
    matmul_q(w->attn_out, ho_all, nrm_all, n_tokens, n_embd,
             cfg->n_head * hdim, rowbuf);
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *xt = out + (size_t)t * n_embd;
        const float *at = nrm_all + (size_t)t * n_embd;
        for (uint32_t i = 0; i < n_embd; i++) xt[i] += at[i];
    }
    free(nrm_all); free(q_all); free(ho_all);
  } else {
    /* ---- attention sublayer (sequential tokens, causal over the cache) -- */
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos = pos0 + t;
        float *xt = out + (size_t)t * n_embd;
        rmsnorm(xt, w->attn_norm, cur, n_embd);

        /* q: either LoRA (down → rmsnorm → up) or a full projection when the
         * model has no q compression (q_lora_rank == 0, e.g. DeepSeek-V2-Lite).
         * Then per-head rope on the tail dims (below). */
        if (cfg->q_lora_rank > 0) {
            matvec_q(w->q_a, cur, qa, cfg->q_lora_rank, n_embd, rowbuf);
            rmsnorm(qa, w->q_a_norm, qa, cfg->q_lora_rank);
            matvec_q(w->q_b, qa, q, cfg->n_head * qd, cfg->q_lora_rank, rowbuf);
        } else {
            matvec_q(w->q_proj, cur, q, cfg->n_head * qd, n_embd, rowbuf);
        }

        /* kv: down-project, split latent + shared rope key, cache both */
        matvec_q(w->kv_a, cur, kva, kv_rank + rope_d, n_embd, rowbuf);
        float *lat_t = cache->latent + (size_t)pos * kv_rank;
        float *rop_t = cache->k_rope + (size_t)pos * rope_d;
        rmsnorm(kva, w->kv_a_norm, lat_t, kv_rank);
        memcpy(rop_t, kva + kv_rank, rope_d * sizeof(float));
        rope_neox(rop_t, rope_d, pos, cfg->rope_theta);

        const float scale = 1.0f / sqrtf((float)qd);
        /* kv_b is kept fp32 at load (assembled/small) — index its sub-blocks. */
        const float *kv_b_f = (const float *)w->kv_b.data;
        for (uint32_t hd = 0; hd < cfg->n_head; hd++) {
            float *q_h = q + (size_t)hd * qd;
            rope_neox(q_h + nope, rope_d, pos, cfg->rope_theta);
            const float *kvb_k = kv_b_f + (size_t)hd * (nope + v_dim) * kv_rank;
            const float *kvb_v = kvb_k + (size_t)nope * kv_rank;

            /* scores over cache 0..pos: k_nope reconstructed per position */
            float smax = -1e30f;
            for (uint32_t u = 0; u <= pos; u++) {
                const float *lat_u = cache->latent + (size_t)u * kv_rank;
                matvec_f32(kvb_k, lat_u, recon, nope, kv_rank);
                float s = vec_dot(q_h, recon, nope) +
                          vec_dot(q_h + nope,
                                  cache->k_rope + (size_t)u * rope_d, rope_d);
                s *= scale;
                scores[u] = s;
                if (s > smax) smax = s;
            }
            float ssum = 0.0f;
            for (uint32_t u = 0; u <= pos; u++) {
                scores[u] = expf(scores[u] - smax);
                ssum += scores[u];
            }
            /* out_h = Σ p(u) · v(u), v reconstructed from the latent */
            float *o_h = hout + (size_t)hd * v_dim;
            memset(o_h, 0, v_dim * sizeof(float));
            for (uint32_t u = 0; u <= pos; u++) {
                const float p = scores[u] / ssum;
                const float *lat_u = cache->latent + (size_t)u * kv_rank;
                matvec_f32(kvb_v, lat_u, recon, v_dim, kv_rank);
                for (uint32_t i = 0; i < v_dim; i++) o_h[i] += p * recon[i];
            }
        }
        /* project concat(heads) back and add the residual */
        matvec_q(w->attn_out, hout, cur, n_embd, cfg->n_head * v_dim, rowbuf);
        for (uint32_t i = 0; i < n_embd; i++) xt[i] += cur[i];
    }
  }

    /* ---- FFN sublayer (dense lead or MoE) ------------------------------- */
    if (is_dense && n_tokens > 1) {
        /* Whole chunk at once. This is the single biggest matvec consumer in a
         * dense model (3 of the ~8 projections per layer, and the widest), so
         * it is where batching pays most. Falls back to the per-token path for
         * n_tokens == 1 (decode) where there is nothing to amortize and the
         * chunk buffers would just be malloc churn. */
        float *fin = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));
        float *fout = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));
        float *fsw = (float *)malloc((size_t)n_tokens * 2u * cfg->n_ff_dense * sizeof(float));
        if (!fin || !fout || !fsw) {
            free(fin); free(fout); free(fsw); free(scr);
            return fwd_fail("oom: dense-FFN chunk buffers (%u tokens, %.1f MB)",
                            n_tokens, (double)((size_t)n_tokens
                                * (n_embd * 2 + 2u * cfg->n_ff_dense)
                                * sizeof(float)) / 1048576.0);
        }
        for (uint32_t t = 0; t < n_tokens; t++)
            rmsnorm(out + (size_t)t * n_embd, w->ffn_norm,
                    fin + (size_t)t * n_embd, n_embd);
        swiglu_chunk(w->gate, w->up, w->down, fin, fout, n_tokens,
                     cfg->n_ff_dense, n_embd, fsw, rowbuf);
        for (uint32_t t = 0; t < n_tokens; t++) {
            float *xt = out + (size_t)t * n_embd;
            const float *yt = fout + (size_t)t * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) xt[i] += yt[i];
        }
        free(fin); free(fout); free(fsw);
        free(scr);
        return 0;
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *xt = out + (size_t)t * n_embd;
        rmsnorm(xt, w->ffn_norm, cur, n_embd);
        if (is_dense) {
            swiglu(w->gate, w->up, w->down, cur, ffn_y, cfg->n_ff_dense, n_embd, sw, rowbuf);
        } else {
            /* MoE router — exact llama.cpp build_moe_ffn order:
             *   probs = sigmoid|softmax(logits)
             *   select top-k by (probs + exp_probs_b) [bias only for selection]
             *   combine weights = UNBIASED probs of the selected experts
             *   if expert_weights_norm: weights /= max(Σweights, 6.1e-5)
             *   if expert_weights_scale ∉ {0,1}: weights *= scale
             *   out = Σ weights_i · expert_i(x) + shared(x) */
            matvec_q(w->router, cur, probs, cfg->n_expert, n_embd, rowbuf);
            if (cfg->gating_func == DS4X_GATE_SIGMOID) {
                for (uint32_t e = 0; e < cfg->n_expert; e++)
                    probs[e] = 1.0f / (1.0f + expf(-probs[e]));
            } else {
                float mx = probs[0];
                for (uint32_t e = 1; e < cfg->n_expert; e++) if (probs[e] > mx) mx = probs[e];
                float sum = 0.0f;
                for (uint32_t e = 0; e < cfg->n_expert; e++) { probs[e] = expf(probs[e] - mx); sum += probs[e]; }
                for (uint32_t e = 0; e < cfg->n_expert; e++) probs[e] /= sum;
            }

            /* top-k by selection score (probs + optional bias), repeated-max. */
            uint32_t top[64];
            const uint32_t k = cfg->n_expert_used < 64 ? cfg->n_expert_used : 64;
            for (uint32_t j = 0; j < k; j++) {
                int best = -1;
                float best_s = -1e30f;
                for (uint32_t e = 0; e < cfg->n_expert; e++) {
                    int taken = 0;
                    for (uint32_t q2 = 0; q2 < j; q2++) if (top[q2] == e) { taken = 1; break; }
                    if (taken) continue;
                    float s = probs[e] + (w->e_score_bias ? w->e_score_bias[e] : 0.0f);
                    if (best < 0 || s > best_s) { best = (int)e; best_s = s; }
                }
                top[j] = (uint32_t)best;
            }
            /* combine weights from UNBIASED probs; optional norm then scale. */
            float wt[64];
            float wsum = 0.0f;
            for (uint32_t j = 0; j < k; j++) { wt[j] = probs[top[j]]; wsum += wt[j]; }
            if (cfg->expert_weights_norm) {
                const float denom = wsum > 6.103515625e-5f ? wsum : 6.103515625e-5f;
                for (uint32_t j = 0; j < k; j++) wt[j] /= denom;
            }
            if (cfg->expert_weights_scale != 0.0f && cfg->expert_weights_scale != 1.0f)
                for (uint32_t j = 0; j < k; j++) wt[j] *= cfg->expert_weights_scale;

            memset(ffn_y, 0, n_embd * sizeof(float));
            const size_t gu_stride = (size_t)cfg->n_ff_exp * n_embd;
            const size_t dn_stride = (size_t)n_embd * cfg->n_ff_exp;
            for (uint32_t j = 0; j < k; j++) {
                const uint32_t e = top[j];
                swiglu(wt_slice(w->e_gate, (size_t)e * gu_stride),
                       wt_slice(w->e_up,   (size_t)e * gu_stride),
                       wt_slice(w->e_down, (size_t)e * dn_stride),
                       cur, eout, cfg->n_ff_exp, n_embd, sw, rowbuf);
                for (uint32_t i = 0; i < n_embd; i++) ffn_y[i] += wt[j] * eout[i];
            }
            /* Shared expert: on for every token when the model HAS one, added
             * directly with no scaling. Must mirror the loader, which only
             * fills s_gate/s_up/s_down when n_expert_shared > 0 — calling it
             * unconditionally dereferenced NULL on any MoE without a shared
             * expert (Qwen3-MoE is one). DSv4 has n_expert_shared=1, which is
             * why this never fired. */
            if (cfg->n_expert_shared > 0 && w->s_gate.data) {
                swiglu(w->s_gate, w->s_up, w->s_down, cur, eout,
                       cfg->n_ff_exp * cfg->n_expert_shared, n_embd, sw, rowbuf);
                /* qwen35moe scales the shared expert by a SCALAR sigmoid gate
                 * (llama.cpp: shared_expert_gate = MUL_MAT(ffn_gate_inp_shexp,
                 * x) → SIGMOID → MUL). Models without that tensor (DSv4) add it
                 * ungated, so the weight being absent is the switch. */
                float sg = 1.0f;
                if (w->s_gate_inp.data) {
                    matvec_q(w->s_gate_inp, cur, &sg, 1, n_embd, rowbuf);
                    sg = 1.0f / (1.0f + expf(-sg));
                }
                for (uint32_t i = 0; i < n_embd; i++) ffn_y[i] += sg * eout[i];
            }
        }
        for (uint32_t i = 0; i < n_embd; i++) xt[i] += ffn_y[i];
    }

    free(scr);
    return 0;
}
