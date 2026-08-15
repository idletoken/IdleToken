/* ds4x_cuda.cu — GPU dequantize-and-matvec for the ds4x forward.
 * See include/idletoken_ds4x_cuda.h for scope and the correctness contract.
 *
 * The block layouts below MIRROR src/ds4x/ds4x_quant.c exactly. Keep the two
 * in sync — in particular get_scale_min_k4's asymmetry (the scale's high bits
 * come from scales[j-4], the min's from scales[j]); getting that wrong makes
 * every min in sub-blocks 4..7 garbage, which on real weights shows up as the
 * model emitting one constant token (real bug, 2026-07-27).
 *
 * Kernel shape: one thread block per output row; the row's quant-blocks are
 * strided across threads; a warp/block reduction sums the partial dots. This
 * is a correctness-first design — coalescing/tensor-core work comes later, and
 * only after the parity gate stays green.
 */
#include "idletoken_ds4x_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>   /* QueryPerformanceCounter (no clock_gettime under MSVC) */
#endif

#define QK_K 256

enum { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q2_K = 10,
       T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_BF16 = 30 };

/* Sub-4-bit types, needed for the big MoE builds (GLM-5.2's GGUF is 225
 * IQ2_XXS tensors + 3 Q2_K). Layouts mirror ds4x_quant.c, which is itself
 * gated bit-exact against ggml's own to_float. */
#include "ds4x_iq2_tables_cuda.inc"

static char g_err[256] = "";
static char g_dev[128] = "";
static unsigned long long g_bytes = 0;
/* 0 = uncapped. See ds4x_cuda_set_budget in idletoken_ds4x_cuda.h. */
static unsigned long long g_budget = 0;

struct ds4x_cuda_wt {
    void    *d_data;     /* raw (quantized) bytes on device */
    uint32_t type;
    uint32_t n_out, n_in;
    float   *d_x;        /* [n_in]  input staging  */
    float   *d_y;        /* [n_out] output staging */
    size_t   bytes;
};

static void set_err(const char *what, cudaError_t e) {
    /* Always include the numeric code. cudaGetErrorString() itself returns
     * "(null)" when the runtime/driver pair is mismatched — exactly the case
     * you most need the code for. Real instance (2026-07-28, RTX 2070 box):
     * "cudaGetDeviceCount: (null)" told us nothing and cost two rounds of
     * guessing; the code would have named cudaErrorInsufficientDriver at once.
     * 35 = insufficient driver (DLL built against a newer CUDA than the driver
     * supports) — the single most likely cause on a mixed-driver home cluster. */
    const char *str = cudaGetErrorString(e);
    const char *name = cudaGetErrorName(e);
    snprintf(g_err, sizeof(g_err), "%s: %s (%s, code %d)%s", what,
             (str && *str) ? str : "no description",
             (name && *name) ? name : "unknown",
             (int)e,
             e == cudaErrorInsufficientDriver
               ? " — the driver is older than the CUDA the DLL was built with;"
                 " rebuild ds4xcuda against this machine's CUDA or update the driver"
               : "");
}

/* bf16 → fp32 is a pure shift (bf16 IS the high half of an fp32), so it is
 * exact and needs no exponent rebias — unlike f16_to_f32_d above. */
__device__ __forceinline__ float bf16_to_f32_d(unsigned short b) {
    const unsigned int bits = (unsigned int)b << 16;
    return __int_as_float((int)bits);
}

/* ---- host-side block geometry (mirrors ds4x_quant.c) -------------------- */
static uint32_t blk_count(uint32_t t) {
    switch (t) {
        case T_F32: case T_F16: case T_BF16: return 1;
        case T_Q8_0: case T_Q4_0: return 32;
        case T_Q2_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ2_XXS: return QK_K;
        default: return 0;
    }
}
static uint64_t blk_bytes(uint32_t t) {
    switch (t) {
        case T_F32:  return 4;
        case T_F16:  return 2;
        case T_BF16: return 2;
        case T_Q8_0: return 34;
        case T_Q4_0: return 18;
        case T_Q2_K: return 84;
        case T_Q4_K: return 144;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_IQ2_XXS: return 66;
        default: return 0;
    }
}

/* ---- device helpers ----------------------------------------------------- */
/* Bit-exact mirror of ds4x_f16_to_f32 (ds4x_quant.c) — done by hand rather
 * than via cuda_fp16.h so the GPU and CPU agree bit-for-bit on denormals and
 * so this file needs no half-precision headers. */
__device__ __forceinline__ float f16_to_f32_d(unsigned short h) {
    const unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return __int_as_float((int)bits);
}

__device__ __forceinline__ void get_scale_min_k4_d(int j, const unsigned char *q,
                                                   unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        /* asymmetric on purpose — see file header */
        *d = (unsigned char)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (unsigned char)((q[j + 4] >>   4) | ((q[j    ] >> 6) << 4));
    }
}

/* ---- unit decomposition -------------------------------------------------
 * The first kernel gave each thread a whole quant block, which for Q4_K at
 * n_in=4096 left only 16 threads busy per row (4096/256) — under half a warp.
 * Instead every type is decomposed into 32-ELEMENT UNITS, which is the natural
 * granularity of the K-quant layouts: a Q4_K/Q5_K 64-element sub-block is two
 * halves with their own (scale,min), and Q6_K's 128-element group is four 32s.
 * n_units = n_in/32, so a 4096-wide row now feeds 128 threads and a 12288-wide
 * FFN row feeds 384. */
#define UNIT 32

/* Token rows one matmul block serves. The long note above matmul_q4k_kernel
 * explains why this is not a free tuning knob. */
#define TOK_TILE 8

/* Dot of ONE 32-element unit `u` of a row with the matching slice of x. */
__device__ float unit_dot(uint32_t type, const unsigned char *rowp,
                          uint32_t u, const float *x) {
    float acc = 0.0f;
    const float *xv = x + (size_t)u * UNIT;
    switch (type) {
        case T_F32: {
            const float *w = (const float *)(rowp + (size_t)u * UNIT * 4);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) acc += w[i] * xv[i];
            return acc;
        }
        case T_F16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) acc += f16_to_f32_d(w[i]) * xv[i];
            return acc;
        }
        case T_BF16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) acc += bf16_to_f32_d(w[i]) * xv[i];
            return acc;
        }
        case T_Q8_0: {   /* one 32-elem block == one unit */
            const unsigned char *p = rowp + (size_t)u * 34;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const signed char *qs = (const signed char *)(p + 2);
            #pragma unroll
            for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * xv[i];
            return acc;
        }
        case T_Q4_0: {   /* one 32-elem block == one unit (lo j, hi j+16) */
            const unsigned char *p = rowp + (size_t)u * 18;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const unsigned char *qs = p + 2;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                acc += d * (float)((qs[j] & 0x0F) - 8) * xv[j];
                acc += d * (float)((qs[j] >>   4) - 8) * xv[j + 16];
            }
            return acc;
        }
        case T_Q2_K: {
            /* 8 units per 256-block. Element index u*32+i sits at
             * half = u>>2 (which 128-group), j = u&3 (which shift), and the
             * two 16-element halves of the unit use CONSECUTIVE scale bytes
             * with q[l] and q[l+16] — the same "separated by shift, not by
             * adjacency" layout as the CPU path. */
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 84;
            const unsigned char *sc = p;
            const unsigned char *q  = p + 16 + (size_t)(uw >> 2) * 32;
            const float d    = f16_to_f32_d(*(const unsigned short *)(p + 80));
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 82));
            const int shift = 2 * (int)(uw & 3);
            const int is = (int)(uw >> 2) * 8 + (int)(uw & 3) * 2;
            const float dl0 = d * (float)(sc[is] & 0xF), ml0 = dmin * (float)(sc[is] >> 4);
            const float dl1 = d * (float)(sc[is + 1] & 0xF), ml1 = dmin * (float)(sc[is + 1] >> 4);
            #pragma unroll
            for (int l = 0; l < 16; l++) {
                acc += (dl0 * (float)((q[l] >> shift) & 3) - ml0) * xv[l];
                acc += (dl1 * (float)((q[l + 16] >> shift) & 3) - ml1) * xv[l + 16];
            }
            return acc;
        }
        case T_IQ2_XXS: {
            /* One 32-element group per unit exactly (QK_K/32 == 8 units). */
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 66;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            unsigned int aux32[2];
            memcpy(aux32, p + 2 + (size_t)uw * 8, 8);
            const unsigned char *aux8 = (const unsigned char *)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const unsigned char *grid =
                    (const unsigned char *)(ds4x_iq2xxs_grid + aux8[l]);
                const unsigned char signs = ds4x_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc += db * (float)grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f)
                              * xv[l * 8 + j];
            }
            return acc;
        }
        case T_Q4_K: {
            const uint32_t qb = u >> 3, uw = u & 7;      /* 8 units per 256-blk */
            const unsigned char *p = rowp + (size_t)qb * 144;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);   /* scale idx == uw */
            const float dd = d * sc, m = dmin * mm;
            /* 32-bit loads instead of 32 byte loads: the byte-wise version ran
             * at ~32 GB/s against ~270 GB/s of device bandwidth. Blocks are
             * 144 B and cudaMalloc is 256 B-aligned, so `q` is 4 B-aligned. */
            const unsigned char *q = p + 16 + (size_t)(uw >> 1) * 32;
            const unsigned int *qw = (const unsigned int *)q;
            const int sh = (uw & 1) ? 4 : 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const unsigned int wv = qw[k];
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    acc += (dd * (float)((wv >> (8 * b + sh)) & 0xFu) - m) * xv[k * 4 + b];
            }
            return acc;
        }
        case T_Q5_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 176;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
            const float dd = d * sc, m = dmin * mm;
            const unsigned char *qh = p + 16;
            const unsigned char *ql = p + 48 + (size_t)(uw >> 1) * 32;
            /* qh bit pair advances by 2 per 64-elem group; low half uses the
             * even bit, high half the odd one (mirrors u1/u2 in the CPU loop) */
            const unsigned char bit = (unsigned char)(1u << ((uw >> 1) * 2 + (uw & 1)));
            if ((uw & 1) == 0) {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    acc += (dd * (float)((ql[l] & 0xF) + ((qh[l] & bit) ? 16 : 0)) - m) * xv[l];
            } else {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    acc += (dd * (float)((ql[l] >>  4) + ((qh[l] & bit) ? 16 : 0)) - m) * xv[l];
            }
            return acc;
        }
        case T_Q6_K: {
            const uint32_t qb = u >> 3, uw = u & 7;      /* 8 units per 256-blk */
            const uint32_t nn = uw >> 2, g = uw & 3;     /* 128-group, 32-quarter */
            const unsigned char *p = rowp + (size_t)qb * 210;
            const unsigned char *ql = p + (size_t)nn * 64;
            const unsigned char *qh = p + 128 + (size_t)nn * 32;
            const signed char   *sc = (const signed char *)(p + 192) + (size_t)nn * 8;
            const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
            /* The per-g selection is loop-invariant: g picks which half of ql
             * and which 2-bit field of qh, so hoist it out of the loop.
             * NOTE: unlike Q4_K (144 B blocks) a Q6_K block is 210 B — NOT a
             * multiple of 4 — so every other block starts 2 B off and 32-bit
             * loads fault with "misaligned address". Byte loads it is. */
            const unsigned char *qlb = ql + (size_t)(g & 1) * 32;
            const int lsh = (g >= 2) ? 4 : 0;     /* low vs high nibble  */
            const int hsh = 2 * (int)g;           /* which qh bit pair   */
            const float s0 = d * (float)sc[0 + 2 * g];   /* l <  16 */
            const float s1 = d * (float)sc[1 + 2 * g];   /* l >= 16 */
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int lo = (int)((qlb[l] >> lsh) & 0xFu);
                const int hi = (int)((qh[l]  >> hsh) & 0x3u);
                acc += ((l < 16) ? s0 : s1) * (float)((lo | (hi << 4)) - 32) * xv[l];
            }
            return acc;
        }
        default: return 0.0f;
    }
}

/* Batched counterpart of unit_dot: dequantize a unit ONCE, apply it to every
 * token row in the tile.
 *
 * unit_dot decodes a unit's 32 weights and immediately dots them with one token,
 * so calling it once per token — which is what matmul_kernel used to do — repeats
 * the entire dequant TOK_TILE times. For the K-quants that dequant *is* most of the
 * instruction count, and the repeat is pure waste: the weights are identical for
 * every token in the tile. Measured on a GB10 (2026-08-13): the generic path cost
 * 13.1 ms/call against 3.06 ms for matmul_q4k_kernel, which already dequantized
 * once, and nsys attributed 77% of all prefill kernel time to this one function.
 * It is also why raising TOK_TILE made prefill *slower* — more dequant repeats,
 * while the bandwidth it was supposed to save was never the constraint.
 *
 * The token loop is innermost rather than "dequantize into a scratch array, then
 * dot" on purpose: several layouts (Q4_0, Q2_K, IQ2_XXS) accumulate in an order
 * that is not index-ascending, and rewriting them to ascending order would change
 * each token's floating-point summation order. Keeping the original order per token
 * is what holds cudatest's `chunk==1-by-1` assertion, which is md == 0.0 — a
 * bit-identity requirement, not a tolerance. */
__device__ __forceinline__ void unit_dot_tile(uint32_t type, const unsigned char *rowp,
                                              uint32_t u, const float *X, uint32_t n_in,
                                              uint32_t t0, uint32_t n_tok, float *acc) {
    /* Row bases for the tile. Clamped so the address stays inside X even for the
     * masked-off lanes of a partial tile; those lanes never dereference it. */
    const float *xb[TOK_TILE];
    #pragma unroll
    for (int tt = 0; tt < TOK_TILE; tt++) {
        const uint32_t t = t0 + (uint32_t)tt;
        xb[tt] = X + (size_t)(t < n_tok ? t : (n_tok - 1u)) * n_in + (size_t)u * UNIT;
    }

#define FMA_T(WV, IDX)                                          \
    do {                                                        \
        const float w_ = (WV);                                  \
        _Pragma("unroll")                                       \
        for (int tt_ = 0; tt_ < TOK_TILE; tt_++) {              \
            if (t0 + (uint32_t)tt_ >= n_tok) break;             \
            acc[tt_] += w_ * xb[tt_][(IDX)];                    \
        }                                                       \
    } while (0)

    switch (type) {
        case T_F32: {
            const float *w = (const float *)(rowp + (size_t)u * UNIT * 4);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) FMA_T(w[i], i);
            return;
        }
        case T_F16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) FMA_T(f16_to_f32_d(w[i]), i);
            return;
        }
        case T_BF16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) FMA_T(bf16_to_f32_d(w[i]), i);
            return;
        }
        case T_Q8_0: {
            const unsigned char *p = rowp + (size_t)u * 34;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const signed char *qs = (const signed char *)(p + 2);
            #pragma unroll
            for (int i = 0; i < 32; i++) FMA_T(d * (float)qs[i], i);
            return;
        }
        case T_Q4_0: {
            const unsigned char *p = rowp + (size_t)u * 18;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const unsigned char *qs = p + 2;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                FMA_T(d * (float)((qs[j] & 0x0F) - 8), j);
                FMA_T(d * (float)((qs[j] >>   4) - 8), j + 16);
            }
            return;
        }
        case T_Q2_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 84;
            const unsigned char *sc = p;
            const unsigned char *q  = p + 16 + (size_t)(uw >> 2) * 32;
            const float d    = f16_to_f32_d(*(const unsigned short *)(p + 80));
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 82));
            const int shift = 2 * (int)(uw & 3);
            const int is = (int)(uw >> 2) * 8 + (int)(uw & 3) * 2;
            const float dl0 = d * (float)(sc[is] & 0xF), ml0 = dmin * (float)(sc[is] >> 4);
            const float dl1 = d * (float)(sc[is + 1] & 0xF), ml1 = dmin * (float)(sc[is + 1] >> 4);
            #pragma unroll
            for (int l = 0; l < 16; l++) {
                FMA_T(dl0 * (float)((q[l] >> shift) & 3) - ml0, l);
                FMA_T(dl1 * (float)((q[l + 16] >> shift) & 3) - ml1, l + 16);
            }
            return;
        }
        case T_IQ2_XXS: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 66;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            unsigned int aux32[2];
            memcpy(aux32, p + 2 + (size_t)uw * 8, 8);
            const unsigned char *aux8 = (const unsigned char *)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const unsigned char *grid =
                    (const unsigned char *)(ds4x_iq2xxs_grid + aux8[l]);
                const unsigned char signs = ds4x_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    FMA_T(db * (float)grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f),
                          l * 8 + j);
            }
            return;
        }
        case T_Q4_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 144;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
            const float dd = d * sc, m = dmin * mm;
            const unsigned char *q = p + 16 + (size_t)(uw >> 1) * 32;
            const unsigned int *qw = (const unsigned int *)q;
            const int sh = (uw & 1) ? 4 : 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const unsigned int wv = qw[k];
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    FMA_T(dd * (float)((wv >> (8 * b + sh)) & 0xFu) - m, k * 4 + b);
            }
            return;
        }
        case T_Q5_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 176;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
            const float dd = d * sc, m = dmin * mm;
            const unsigned char *qh = p + 16;
            const unsigned char *ql = p + 48 + (size_t)(uw >> 1) * 32;
            const unsigned char bit = (unsigned char)(1u << ((uw >> 1) * 2 + (uw & 1)));
            if ((uw & 1) == 0) {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    FMA_T(dd * (float)((ql[l] & 0xF) + ((qh[l] & bit) ? 16 : 0)) - m, l);
            } else {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    FMA_T(dd * (float)((ql[l] >>  4) + ((qh[l] & bit) ? 16 : 0)) - m, l);
            }
            return;
        }
        case T_Q6_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const uint32_t nn = uw >> 2, g = uw & 3;
            const unsigned char *p = rowp + (size_t)qb * 210;
            const unsigned char *ql = p + (size_t)nn * 64;
            const unsigned char *qh = p + 128 + (size_t)nn * 32;
            const signed char   *sc = (const signed char *)(p + 192) + (size_t)nn * 8;
            const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
            const unsigned char *qlb = ql + (size_t)(g & 1) * 32;
            const int lsh = (g >= 2) ? 4 : 0;
            const int hsh = 2 * (int)g;
            const float s0 = d * (float)sc[0 + 2 * g];
            const float s1 = d * (float)sc[1 + 2 * g];
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int lo = (int)((qlb[l] >> lsh) & 0xFu);
                const int hi = (int)((qh[l]  >> hsh) & 0x3u);
                FMA_T(((l < 16) ? s0 : s1) * (float)((lo | (hi << 4)) - 32), l);
            }
            return;
        }
        default: return;
    }
#undef FMA_T
}

/* Dequantize ONE unit of a weight row into a strided destination.
 *
 * Case for case this mirrors unit_dot_tile: wherever that one multiplies a weight
 * into xv[IDX], this one stores it at dst[IDX * stride]. Reusing the index
 * expression is the point — the exotic layouts (Q4_0's lo/hi halves, Q2_K's shift
 * groups, IQ2_XXS's grid lookup) map unit-local positions to element positions in
 * ways that are easy to restate wrongly. **Keep the two functions in sync.**
 *
 * Used by the tiled GEMM to stage a weight tile in shared memory, where the layout
 * wanted is Bs[k][n] — hence the stride rather than a contiguous write. */
template <typename T> __device__ __forceinline__ T dq_cast(float v);
template <> __device__ __forceinline__ float dq_cast<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 dq_cast<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ void unit_dequant_row(uint32_t type, const unsigned char *rowp,
                                                 uint32_t u, T *dst, uint32_t stride) {
#define DEQ_W(WV, IDX) dst[(size_t)(IDX) * stride] = dq_cast<T>(WV)
    switch (type) {
        case T_F32: {
            const float *w = (const float *)(rowp + (size_t)u * UNIT * 4);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) DEQ_W(w[i], i);
            return;
        }
        case T_F16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) DEQ_W(f16_to_f32_d(w[i]), i);
            return;
        }
        case T_BF16: {
            const unsigned short *w = (const unsigned short *)(rowp + (size_t)u * UNIT * 2);
            #pragma unroll
            for (int i = 0; i < UNIT; i++) DEQ_W(bf16_to_f32_d(w[i]), i);
            return;
        }
        case T_Q8_0: {
            const unsigned char *p = rowp + (size_t)u * 34;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const signed char *qs = (const signed char *)(p + 2);
            #pragma unroll
            for (int i = 0; i < 32; i++) DEQ_W(d * (float)qs[i], i);
            return;
        }
        case T_Q4_0: {
            const unsigned char *p = rowp + (size_t)u * 18;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const unsigned char *qs = p + 2;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                DEQ_W(d * (float)((qs[j] & 0x0F) - 8), j);
                DEQ_W(d * (float)((qs[j] >>   4) - 8), j + 16);
            }
            return;
        }
        case T_Q2_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 84;
            const unsigned char *sc = p;
            const unsigned char *q  = p + 16 + (size_t)(uw >> 2) * 32;
            const float d    = f16_to_f32_d(*(const unsigned short *)(p + 80));
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 82));
            const int shift = 2 * (int)(uw & 3);
            const int is = (int)(uw >> 2) * 8 + (int)(uw & 3) * 2;
            const float dl0 = d * (float)(sc[is] & 0xF), ml0 = dmin * (float)(sc[is] >> 4);
            const float dl1 = d * (float)(sc[is + 1] & 0xF), ml1 = dmin * (float)(sc[is + 1] >> 4);
            #pragma unroll
            for (int l = 0; l < 16; l++) {
                DEQ_W(dl0 * (float)((q[l] >> shift) & 3) - ml0, l);
                DEQ_W(dl1 * (float)((q[l + 16] >> shift) & 3) - ml1, l + 16);
            }
            return;
        }
        case T_IQ2_XXS: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 66;
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            unsigned int aux32[2];
            memcpy(aux32, p + 2 + (size_t)uw * 8, 8);
            const unsigned char *aux8 = (const unsigned char *)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const unsigned char *grid =
                    (const unsigned char *)(ds4x_iq2xxs_grid + aux8[l]);
                const unsigned char signs = ds4x_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    DEQ_W(db * (float)grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f),
                          l * 8 + j);
            }
            return;
        }
        case T_Q4_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 144;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
            const float dd = d * sc, m = dmin * mm;
            const unsigned char *q = p + 16 + (size_t)(uw >> 1) * 32;
            const unsigned int *qw = (const unsigned int *)q;
            const int sh = (uw & 1) ? 4 : 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const unsigned int wv = qw[k];
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    DEQ_W(dd * (float)((wv >> (8 * b + sh)) & 0xFu) - m, k * 4 + b);
            }
            return;
        }
        case T_Q5_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const unsigned char *p = rowp + (size_t)qb * 176;
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
            const float dd = d * sc, m = dmin * mm;
            const unsigned char *qh = p + 16;
            const unsigned char *ql = p + 48 + (size_t)(uw >> 1) * 32;
            const unsigned char bit = (unsigned char)(1u << ((uw >> 1) * 2 + (uw & 1)));
            if ((uw & 1) == 0) {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    DEQ_W(dd * (float)((ql[l] & 0xF) + ((qh[l] & bit) ? 16 : 0)) - m, l);
            } else {
                #pragma unroll
                for (int l = 0; l < 32; l++)
                    DEQ_W(dd * (float)((ql[l] >>  4) + ((qh[l] & bit) ? 16 : 0)) - m, l);
            }
            return;
        }
        case T_Q6_K: {
            const uint32_t qb = u >> 3, uw = u & 7;
            const uint32_t nn = uw >> 2, g = uw & 3;
            const unsigned char *p = rowp + (size_t)qb * 210;
            const unsigned char *ql = p + (size_t)nn * 64;
            const unsigned char *qh = p + 128 + (size_t)nn * 32;
            const signed char   *sc = (const signed char *)(p + 192) + (size_t)nn * 8;
            const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
            const unsigned char *qlb = ql + (size_t)(g & 1) * 32;
            const int lsh = (g >= 2) ? 4 : 0;
            const int hsh = 2 * (int)g;
            const float s0 = d * (float)sc[0 + 2 * g];
            const float s1 = d * (float)sc[1 + 2 * g];
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int lo = (int)((qlb[l] >> lsh) & 0xFu);
                const int hi = (int)((qh[l]  >> hsh) & 0x3u);
                DEQ_W(((l < 16) ? s0 : s1) * (float)((lo | (hi << 4)) - 32), l);
            }
            return;
        }
        default: {
            #pragma unroll
            for (int i = 0; i < UNIT; i++) DEQ_W(0.0f, i);
            return;
        }
    }
#undef DEQ_W
}

/* Dot of ONE quant block (blk_count(type) elements) with x[]. */
__device__ float block_dot(uint32_t type, const unsigned char *p, const float *x) {
    float acc = 0.0f;
    switch (type) {
        case T_F32: {
            acc = (*(const float *)p) * x[0];
            break;
        }
        case T_F16: {
            acc = f16_to_f32_d(*(const unsigned short *)p) * x[0];
            break;
        }
        case T_BF16: {
            acc = bf16_to_f32_d(*(const unsigned short *)p) * x[0];
            break;
        }
        case T_Q8_0: {
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const signed char *qs = (const signed char *)(p + 2);
            #pragma unroll
            for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * x[i];
            break;
        }
        case T_Q4_0: {
            const float d = f16_to_f32_d(*(const unsigned short *)p);
            const unsigned char *qs = p + 2;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const int lo = (qs[j] & 0x0F) - 8;
                const int hi = (qs[j] >>   4) - 8;
                acc += d * (float)lo * x[j];
                acc += d * (float)hi * x[j + 16];
            }
            break;
        }
        case T_Q4_K: {
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            const unsigned char *scales = p + 4;
            const unsigned char *q = p + 16;
            int is = 0, oi = 0;
            unsigned char sc, mm;
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4_d(is + 0, scales, &sc, &mm);
                const float d1 = d * sc, m1 = dmin * mm;
                get_scale_min_k4_d(is + 1, scales, &sc, &mm);
                const float d2 = d * sc, m2 = dmin * mm;
                for (int l = 0; l < 32; l++) acc += (d1 * (float)(q[l] & 0xF) - m1) * x[oi + l];
                for (int l = 0; l < 32; l++) acc += (d2 * (float)(q[l] >>  4) - m2) * x[oi + 32 + l];
                q += 32; is += 2; oi += 64;
            }
            break;
        }
        case T_Q5_K: {
            const float d    = f16_to_f32_d(*(const unsigned short *)p);
            const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
            const unsigned char *scales = p + 4;
            const unsigned char *qh = p + 16;
            const unsigned char *ql = p + 48;
            int is = 0, oi = 0;
            unsigned char sc, mm, u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4_d(is + 0, scales, &sc, &mm);
                const float d1 = d * sc, m1 = dmin * mm;
                get_scale_min_k4_d(is + 1, scales, &sc, &mm);
                const float d2 = d * sc, m2 = dmin * mm;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * x[oi + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2) * x[oi + 32 + l];
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2; oi += 64;
            }
            break;
        }
        case T_Q6_K: {
            const unsigned char *ql = p;
            const unsigned char *qh = p + 128;
            const signed char   *sc = (const signed char *)(p + 192);
            const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
            int oi = 0;
            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; l++) {
                    const int is = l / 16;
                    const signed char q1 = (signed char)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const signed char q2 = (signed char)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const signed char q3 = (signed char)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const signed char q4 = (signed char)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * (float)sc[is + 0] * (float)q1 * x[oi + l +  0];
                    acc += d * (float)sc[is + 2] * (float)q2 * x[oi + l + 32];
                    acc += d * (float)sc[is + 4] * (float)q3 * x[oi + l + 64];
                    acc += d * (float)sc[is + 6] * (float)q4 * x[oi + l + 96];
                }
                ql += 64; qh += 32; sc += 8; oi += 128;
            }
            break;
        }
        default: break;
    }
    return acc;
}

/* Q4_K specialization: COALESCED. The generic unit kernel gives thread t the
 * t-th 32-element unit, so a warp's 32 threads scatter across 4 quant blocks
 * (576 B); each load instruction uses 64 useful bytes but drags in ~16 sectors
 * — an 8x waste that measured as 32 GB/s against ~270 GB/s of device
 * bandwidth. Here the whole block cooperates on ONE quant block at a time and
 * thread i reads qs[i], so consecutive threads touch consecutive bytes.
 * Byte i of qs encodes two elements: low nibble → element (i/32)*64 + i%32
 * (scale 2j), high nibble → that +32 (scale 2j+1). */
__global__ void matvec_q4k_kernel(const unsigned char *W, uint32_t n_in,
                                  const float *x, float *y) {
    const uint32_t row  = blockIdx.x;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 144;

    float acc = 0.0f;
    for (uint32_t qb = 0; qb < nblk; qb++) {
        const unsigned char *p = rowp + (size_t)qb * 144;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        const unsigned char *scales = p + 4;
        const unsigned char *qs = p + 16;
        const float *xb = x + (size_t)qb * QK_K;
        for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) {
            const uint32_t j = i >> 5, l = i & 31u;
            unsigned char sc, mm;
            get_scale_min_k4_d((int)(2 * j),     scales, &sc, &mm);
            const float d1 = d * sc, m1 = dmin * mm;
            get_scale_min_k4_d((int)(2 * j + 1), scales, &sc, &mm);
            const float d2 = d * sc, m2 = dmin * mm;
            const unsigned char b = qs[i];
            acc += (d1 * (float)(b & 0xF) - m1) * xb[j * 64 + l];
            acc += (d2 * (float)(b >>  4) - m2) * xb[j * 64 + 32 + l];
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    extern __shared__ float smq[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) smq[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? smq[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[row] = v;
    }
}

/* Rows per block for the one-warp-per-row kernels. */
#define MV_WARPS 4

/* Q4_K for a NARROW matrix: one WARP per output row, MV_WARPS rows per block.
 * Same coalescing as matvec_q4k_kernel (a warp still walks qs[] consecutively),
 * but no shared memory, no __syncthreads, and only the intra-warp shuffle.
 *
 * Why a second Q4_K kernel. The block-per-row version does nblk*128 = n_in/2
 * half-byte positions spread over 128 threads, i.e. n_in/256 iterations PER
 * THREAD. At n_in = 4096 that is 16 and the reduction is well amortized; at
 * n_in = 1024 — every projection of a 1024-wide model — it is FOUR. Four
 * iterations of ~8 flops, then ~10 shuffle steps plus a shared round trip plus
 * __syncthreads to reduce them. The reduction costs more than the arithmetic.
 *
 * One warp per row gives each lane 4x the work (16 iterations at n_in=1024) and
 * cuts the reduction to a single 5-step shuffle with no barrier.
 *
 * This is the hypothesis that failed the first time it was tried, because it
 * was aimed at the OUTPUT HEAD — which turned out to be Q6_K (tied embedding),
 * and which the generic path was already running one-warp-per-row. Aimed at the
 * Q4_K projections instead it targets a shape that genuinely has the defect.
 * The lesson is in the git history: verify the hot tensor's TYPE and DIMS
 * before reshaping a kernel for it.
 *
 * Summation order differs from the block-per-row kernel (32 partials rather
 * than 128), so it is gated on shape and checked by the numeric gate. */
__global__ void matvec_q4k_narrow_kernel(const unsigned char *W, uint32_t n_in,
                                         uint32_t n_out, const float *x, float *y) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * MV_WARPS + warp;
    if (row >= n_out) return;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 144;

    float acc = 0.0f;
    for (uint32_t qb = 0; qb < nblk; qb++) {
        const unsigned char *p = rowp + (size_t)qb * 144;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        const unsigned char *scales = p + 4;
        const unsigned char *qs = p + 16;
        const float *xb = x + (size_t)qb * QK_K;
        for (uint32_t i = lane; i < 128; i += 32u) {
            const uint32_t j = i >> 5, l = i & 31u;
            unsigned char sc, mm;
            get_scale_min_k4_d((int)(2 * j),     scales, &sc, &mm);
            const float d1 = d * sc, m1 = dmin * mm;
            get_scale_min_k4_d((int)(2 * j + 1), scales, &sc, &mm);
            const float d2 = d * sc, m2 = dmin * mm;
            const unsigned char b = qs[i];
            acc += (d1 * (float)(b & 0xF) - m1) * xb[j * 64 + l];
            acc += (d2 * (float)(b >>  4) - m2) * xb[j * 64 + 32 + l];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

/* Q6_K specialization: COALESCED, for exactly the reason Q4_K got one.
 *
 * This matters more than the type's share of a Q4_K_M model suggests, because
 * of where Q6_K actually sits: llama.cpp keeps the embedding/output matrix at
 * higher precision, and Qwen3.5 TIES them — so `token_embd.weight` is a single
 * Q6_K [248320][1024], and it is re-read for every decoded token as the output
 * head. Measured on an RTX 5060 Ti it was 2.96 ms/token, 26% of ALL kernel
 * time, against a ~0.47 ms memory roofline for the ~209 MB it reads.
 *
 * The generic unit kernel gives thread t the t-th 32-element unit, so a warp
 * scatters across quant blocks and each load drags in far more sectors than it
 * uses. Here the whole block cooperates on ONE 256-element block at a time and
 * thread i reads ql[i] — consecutive threads, consecutive bytes.
 *
 * Layout (210 B/block, mirrors ds4x_quant.c): ql[0..127] low nibbles,
 * qh[128..191] 2-bit highs, sc[192..207] 16 int8 scales, d at 208.
 * Byte ql[i] carries TWO elements. With nn = i>>6 (128-group), h = (i>>5)&1,
 * l = i&31: the low nibble belongs to quarter g = h, the high nibble to
 * g = h+2. Both take their 2 high bits from qh[nn*32 + l] at bit pair 2g, and
 * their scale from sc[nn*8 + 2g + (l >= 16)].
 *
 * Byte loads only, deliberately: a Q6_K block is 210 B, NOT a multiple of 4, so
 * every other block starts 2 B off and 32-bit loads fault with "misaligned
 * address" (the generic path's comment records the same trap). */
__global__ void matvec_q6k_kernel(const unsigned char *W, uint32_t n_in,
                                  const float *x, float *y) {
    const uint32_t row  = blockIdx.x;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 210;

    float acc = 0.0f;
    for (uint32_t qb = 0; qb < nblk; qb++) {
        const unsigned char *p  = rowp + (size_t)qb * 210;
        const unsigned char *ql = p;
        const unsigned char *qh = p + 128;
        const signed char   *sc = (const signed char *)(p + 192);
        const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
        const float *xb = x + (size_t)qb * QK_K;
        for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) {
            const uint32_t nn = i >> 6, h = (i >> 5) & 1u, l = i & 31u;
            const uint32_t g0 = h, g1 = h + 2u;
            const unsigned char b  = ql[i];
            const unsigned char hb = qh[nn * 32u + l];
            const uint32_t shalf = (l >= 16u) ? 1u : 0u;
            const int v0 = (int)((uint32_t)(b & 0x0Fu) | (((uint32_t)(hb >> (2u * g0)) & 3u) << 4)) - 32;
            const int v1 = (int)((uint32_t)(b >> 4)    | (((uint32_t)(hb >> (2u * g1)) & 3u) << 4)) - 32;
            const float s0 = d * (float)sc[nn * 8u + 2u * g0 + shalf];
            const float s1 = d * (float)sc[nn * 8u + 2u * g1 + shalf];
            acc += s0 * (float)v0 * xb[nn * 128u + g0 * 32u + l];
            acc += s1 * (float)v1 * xb[nn * 128u + g1 * 32u + l];
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    extern __shared__ float smq6[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) smq6[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? smq6[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[row] = v;
    }
}

/* Should this Q4_K matvec use one warp per row instead of one block per row?
 *
 * The block-per-row kernel gives each of its 128 threads n_in/256 iterations.
 * Below n_in = 2048 that is <= 8, at which point the block-wide reduction (a
 * shared round trip and a __syncthreads on top of two shuffle chains) costs
 * more than the work it reduces. 1024-wide models — Qwen3.5-0.8B and every
 * other n_embd=1024 build — sit at 4.
 *
 * Threshold on n_in, not n_out: the defect is "too little work per thread",
 * which n_in alone decides. */
static inline int q4k_narrow(uint32_t type, uint32_t n_in) {
    /* IDLETOKEN_DS4X_NO_NARROW=1 forces the block-per-row kernel, so this
     * choice can be A/B'd IN ONE BINARY like every other change in this file.
     * It exists because the kernel was committed on a kernel-time improvement
     * that never showed up in total time, and settling that needs a paired
     * measurement rather than two builds an hour apart. */
    static int off = -1;
    if (off < 0) { const char *e = getenv("IDLETOKEN_DS4X_NO_NARROW");
                   off = (e && e[0] == '1'); }
    return !off && type == T_Q4_K && n_in <= 2048u && (n_in % QK_K) == 0;
}

/* One block per output row; threads stride over the row's 32-element units.
 * Reduction is warp-shuffle first (no shared traffic inside a warp), then one
 * shared slot per warp. */
__global__ void matvec_kernel(const unsigned char *W, uint32_t type,
                              uint32_t n_in, uint32_t bc, uint32_t bb,
                              const float *x, float *y) {
    const uint32_t row     = blockIdx.x;
    const uint32_t n_units = n_in / UNIT;
    const size_t   rowbytes = (size_t)(n_in / bc) * bb;
    const unsigned char *rowp = W + (size_t)row * rowbytes;

    float acc = 0.0f;
    for (uint32_t u = threadIdx.x; u < n_units; u += blockDim.x)
        acc += unit_dot(type, rowp, u, x);

    /* intra-warp reduction */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    /* one partial per warp → shared → first warp reduces */
    extern __shared__ float sm[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) sm[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? sm[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[row] = v;
    }
}

/* ---- batched matmul (whole chunk) ---------------------------------------
 * TOK_TILE token-rows share one block, so the block dequantizes each weight
 * byte ONCE and applies it to TOK_TILE tokens. That is the whole point: with
 * per-token matvec a 540-token prefill re-reads the model 540 times.
 *
 * Every token keeps its own accumulator and its own reduction, so a token's
 * result does not depend on how many others were in the launch — one-shot
 * prefill stays bit-identical to token-by-token decode.
 *
 * TOK_TILE is the weight-reuse factor: a prefill of N tokens reads the whole
 * model ceil(N/TOK_TILE) times. It is NOT the prefill bottleneck, which is worth
 * recording because it looks like it should be. Raising it 8 -> 32 on a GB10
 * (2026-08-13) left prefill unchanged (BF16 20.7 -> 20.1 tok/s) and made Q4_K_M
 * worse (61.4 -> 52.0), because weight bandwidth is not what prefill is waiting
 * on: at TOK_TILE=8 a 128-token BF16 prefill moves 125 GB in 6.3 s, i.e. 20 GB/s
 * against 273 GB/s of LPDDR5x. Reducing an unsaturated cost buys nothing, and the
 * extra acc[] registers cost occupancy. 32 is also the ceiling for this block
 * shape: tile_store assigns one lane per token (`lane < TOK_TILE`), a warp has 32.
 *
 * Defined further up, next to UNIT: unit_dot_tile() needs it. */

/* Reduce acc[TOK_TILE] across the block and store to Y[(t0+tt)*n_out + row].
 * sm is [nwarps][TOK_TILE]. */
__device__ __forceinline__ void tile_store(float *acc, float *sm, float *Y,
                                           uint32_t row, uint32_t n_out,
                                           uint32_t t0, uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    #pragma unroll
    for (int tt = 0; tt < TOK_TILE; tt++) {
        float v = acc[tt];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) sm[warp * TOK_TILE + tt] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < TOK_TILE) {
        const uint32_t tt = lane;
        if (t0 + tt < n_tok) {
            float s = 0.0f;
            for (uint32_t wq = 0; wq < nwarps; wq++) s += sm[wq * TOK_TILE + tt];
            Y[(size_t)(t0 + tt) * n_out + row] = s;
        }
    }
}

/* Q4_K specialization — the dominant type in a Q4_K_M model. Same coalesced
 * access as matvec_q4k_kernel (thread i reads qs[i]); the only change is that
 * the two unpacked weights are multiplied into TOK_TILE token rows. */
__global__ void matmul_q4k_kernel(const unsigned char *W, uint32_t n_in,
                                  uint32_t n_out, uint32_t n_tok,
                                  const float *X, float *Y) {
    const uint32_t row = blockIdx.x, t0 = blockIdx.y * TOK_TILE;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 144;
    float acc[TOK_TILE];
    #pragma unroll
    for (int tt = 0; tt < TOK_TILE; tt++) acc[tt] = 0.0f;

    for (uint32_t qb = 0; qb < nblk; qb++) {
        const unsigned char *p = rowp + (size_t)qb * 144;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        const unsigned char *scales = p + 4;
        const unsigned char *qs = p + 16;
        for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) {
            const uint32_t j = i >> 5, l = i & 31u;
            unsigned char sc, mm;
            get_scale_min_k4_d((int)(2 * j),     scales, &sc, &mm);
            const float d1 = d * sc, m1 = dmin * mm;
            get_scale_min_k4_d((int)(2 * j + 1), scales, &sc, &mm);
            const float d2 = d * sc, m2 = dmin * mm;
            const unsigned char b = qs[i];
            const float wlo = d1 * (float)(b & 0xF) - m1;
            const float whi = d2 * (float)(b >>  4) - m2;
            const uint32_t o1 = qb * QK_K + j * 64 + l, o2 = o1 + 32;
            #pragma unroll
            for (int tt = 0; tt < TOK_TILE; tt++) {
                if (t0 + (uint32_t)tt >= n_tok) break;
                const float *xr = X + (size_t)(t0 + tt) * n_in;
                acc[tt] += wlo * xr[o1] + whi * xr[o2];
            }
        }
    }
    extern __shared__ float smm[];
    tile_store(acc, smm, Y, row, n_out, t0, n_tok);
}

/* Generic types: unit_dot_tile() dequantizes each unit once and applies it to all
 * TOK_TILE token rows. Calling unit_dot() per token instead (what this did until
 * 2026-08-13) repeated the dequant TOK_TILE times and was 77% of prefill kernel
 * time on a GB10 — the K-quant dequant, not the weight fetch, is the cost. The
 * Q4_K path below is still specialized because it can also use 32-bit loads. */
__global__ void matmul_kernel(const unsigned char *W, uint32_t type,
                              uint32_t n_in, uint32_t n_out, uint32_t bc,
                              uint32_t bb, uint32_t n_tok,
                              const float *X, float *Y) {
    const uint32_t row = blockIdx.x, t0 = blockIdx.y * TOK_TILE;
    const uint32_t n_units = n_in / UNIT;
    const size_t rowbytes = (size_t)(n_in / bc) * bb;
    const unsigned char *rowp = W + (size_t)row * rowbytes;
    float acc[TOK_TILE];
    #pragma unroll
    for (int tt = 0; tt < TOK_TILE; tt++) acc[tt] = 0.0f;

    for (uint32_t u = threadIdx.x; u < n_units; u += blockDim.x)
        unit_dot_tile(type, rowp, u, X, n_in, t0, n_tok, acc);

    extern __shared__ float smg[];
    tile_store(acc, smg, Y, row, n_out, t0, n_tok);
}

/* ---- tiled GEMM ----------------------------------------------------------
 * The prefill path proper. matmul_kernel above is a *batched matvec*: one block
 * produces TOK_TILE outputs for a single output row, then spends warp shuffles
 * reducing along K. Measured on a GB10 (2026-08-13) that ran at 430 GFLOP/s and
 * 18 GB/s — one to two percent of both peaks, at 62% occupancy. The warps were
 * resident and idle: 65536 blocks each doing 256 FMAs per thread before a
 * reduction, so launch and reduction overhead swamped the arithmetic.
 *
 * This kernel is the standard fix. Each block owns a BM x BN tile of the output
 * and walks K in UNIT-sized steps, staging both operands in shared memory:
 *   - As[k][m]: BM token rows of the activation,
 *   - Bs[k][n]: BN weight rows, DEQUANTIZED ON THE WAY IN.
 * Every thread then owns a TM x TN sub-tile held in registers, so each shared
 * value loaded is reused TM (or TN) times and **there is no cross-thread
 * reduction at all** — the K sum lives entirely in one thread's accumulator.
 *
 * Fusing the dequant into the GEMM is deliberate: handing the weights to cuBLAS
 * would mean materialising an FP16/FP32 copy of every tensor first, which is both
 * an extra pass over memory and (at BF16) a second multi-GB buffer. This is what
 * llama.cpp's MMQ kernels do for the same reason.
 *
 * NUMERICS: k advances 0..n_in-1 in a fixed order and each output's whole sum
 * stays in one accumulator, so a token's result does NOT depend on n_tok, on how
 * the tiles were cut, or on how many other tokens were in the launch. That is
 * what cudatest's `chunk==1-by-1` assertion (md == 0.0) checks, and it is why the
 * accumulation could not simply be split across threads for more parallelism.
 * The order does differ from matvec_kernel's (unit-local sums plus a shuffle
 * reduction), so the `vs cpu` comparison moves within its 1e-5 tolerance. */
/* Measured on a GB10, Qwen3.5-4B prefill (2026-08-13). Two "obvious" widenings
 * were tried and both LOST, so they are recorded here rather than left for the
 * next person to re-derive:
 *   64x64  TM4 TN4  -> BF16 345.9 / Q4 376.7 tok/s   (this one)
 *   64x128 TM4 TN8  -> BF16 330.8 / Q4 285.2         (bigger register tile)
 *   float4 operand loads (needs BN+TN padding) -> BF16 301.1
 * The register tile is not the binding constraint at this size; widening BN just
 * leaves more of the 256 threads idle during the dequant-into-shared step (only
 * BN of them have a row to decode) and costs occupancy in shared memory.
 *
 * BM is the one that matters, and for a different reason: the weight tile is
 * re-read AND re-dequantized once per M block, i.e. ceil(n_tok/BM) times over a
 * prefill. At BM=64 the per-token cost was flat in prompt length (7.3 us/token at
 * 128, 7.0 at 256) — which is exactly why prefill did not speed up on longer
 * prompts the way a real GEMM does. Doubling BM halves that repetition. */
#define GEMM_BM 128
#define GEMM_BN 64
#define GEMM_TM 8
#define GEMM_TN 4
#define GEMM_THREADS ((GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN))   /* 16*16 = 256 */

static_assert(GEMM_BM % GEMM_TM == 0 && GEMM_BN % GEMM_TN == 0,
              "tile dimensions must divide evenly");

__global__ __launch_bounds__(GEMM_THREADS)
void matmul_tiled_kernel(const unsigned char *W, uint32_t type,
                         uint32_t n_in, uint32_t n_out, uint32_t bc, uint32_t bb,
                         uint32_t n_tok, const float *X, float *Y) {
    const uint32_t n0 = blockIdx.x * GEMM_BN;
    const uint32_t m0 = blockIdx.y * GEMM_BM;
    const size_t rowbytes = (size_t)(n_in / bc) * bb;

    /* +1 column of padding so consecutive k rows start on different banks.
     *
     * Widening this to +GEMM_TN so the rows would be 16 B aligned (to read each
     * operand as one float4) was tried on 2026-08-13 and was SLOWER: 345.9 -> 301.1
     * tok/s prefill. The tn*TN stride already costs a 2-way bank conflict, and a
     * 128-bit access widens it rather than amortising it. Scalar reads stay. */
    __shared__ float As[UNIT][GEMM_BM];
    __shared__ float Bs[UNIT][GEMM_BN + 1];

    const uint32_t tid = threadIdx.x;
    const uint32_t tm = tid / (GEMM_BN / GEMM_TN);      /* 0..15 */
    const uint32_t tn = tid % (GEMM_BN / GEMM_TN);      /* 0..15 */

    float acc[GEMM_TM][GEMM_TN];
    #pragma unroll
    for (int i = 0; i < GEMM_TM; i++)
        #pragma unroll
        for (int j = 0; j < GEMM_TN; j++) acc[i][j] = 0.0f;

    const uint32_t n_units = n_in / UNIT;
    for (uint32_t u = 0; u < n_units; u++) {
        /* Activation tile, transposed into [k][m] so the inner loop reads a row. */
        for (uint32_t idx = tid; idx < UNIT * GEMM_BM; idx += GEMM_THREADS) {
            const uint32_t k = idx / GEMM_BM, m = idx % GEMM_BM;
            const uint32_t mm = m0 + m;
            As[k][m] = (mm < n_tok)
                     ? X[(size_t)mm * n_in + (size_t)u * UNIT + k] : 0.0f;
        }
        /* Weight tile: one thread per output row dequantizes that row's unit u
         * straight into shared memory. Rows past n_out are zeroed so the masked
         * lanes contribute nothing instead of reading past the tensor. */
        for (uint32_t n = tid; n < GEMM_BN; n += GEMM_THREADS) {
            const uint32_t nn = n0 + n;
            if (nn < n_out)
                unit_dequant_row<float>(type, W + (size_t)nn * rowbytes, u,
                                        &Bs[0][n], GEMM_BN + 1);
            else
                #pragma unroll
                for (int k = 0; k < UNIT; k++) Bs[k][n] = 0.0f;
        }
        __syncthreads();

        /* ncu put l1tex throughput at 65.7% here, so the shared-memory traffic is
         * the limit, not the arithmetic. The lever that works is the register tile:
         * TM x TN FMAs per (TM + TN) loads, so a wider TN buys ratio directly. */
        #pragma unroll
        for (uint32_t k = 0; k < UNIT; k++) {
            float a[GEMM_TM], b[GEMM_TN];
            #pragma unroll
            for (int i = 0; i < GEMM_TM; i++) a[i] = As[k][tm * GEMM_TM + i];
            #pragma unroll
            for (int j = 0; j < GEMM_TN; j++) b[j] = Bs[k][tn * GEMM_TN + j];
            #pragma unroll
            for (int i = 0; i < GEMM_TM; i++)
                #pragma unroll
                for (int j = 0; j < GEMM_TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < GEMM_TM; i++) {
        const uint32_t m = m0 + tm * GEMM_TM + i;
        if (m >= n_tok) continue;
        #pragma unroll
        for (int j = 0; j < GEMM_TN; j++) {
            const uint32_t n = n0 + tn * GEMM_TN + j;
            if (n < n_out) Y[(size_t)m * n_out + n] = acc[i][j];
        }
    }
}

/* One staging arena shared by every weight, not one per handle: at 540 tokens
 * a per-handle [n_tokens][n_in] buffer would cost ~1.4 GB across 187 weights.
 * Matmuls are serial on the default stream, so one arena is safe. */
static struct {
    float *d_x, *d_y, *h_x, *h_y;
    size_t n_x, n_y;                /* floats */
} g_stage = { NULL, NULL, NULL, NULL, 0, 0 };

/* Tokens per launch. Bounds the arena (256 x 12288 floats = 12 MB) without
 * costing anything: the weight is still read once per block of TOK_TILE. */
#define MM_BLOCK 256

/* ---- Gated DeltaNet recurrence ------------------------------------------
 * Mirrors gdn_recur_cpu() in ds4x_forward.c operation for operation; that CPU
 * loop is the reference this must match (IDLETOKEN_DS4X_GDN_CHECK=1 diffs them on
 * real weights).
 *
 * One block per v-head, thread d owning column d of S[k_dim][v_dim]:
 *   - both passes read S[i*vdim + d] with i fixed across the block, so the 32
 *     threads of a warp touch 32 CONSECUTIVE floats — fully coalesced;
 *   - mem[d] and out[d] are per-column sums, so there is no cross-thread
 *     reduction anywhere in the kernel;
 *   - q̂ and k̂ are read by every thread, so they are staged in shared memory
 *     once per token instead of re-read v_dim times.
 * The t loop stays inside the kernel: decode is one token per layer, and a
 * launch per token would cost more than the work. */
__global__ void gdn_recur_kernel(float *S, uint32_t n_tokens,
                                 uint32_t kh, uint32_t vh,
                                 uint32_t kdim, uint32_t vdim,
                                 const float *cnv, const float *bet,
                                 const float *dec, float *core, float oscale) {
    const uint32_t h_v = blockIdx.x;
    /* STRIDED key-head sharing — mirrors gdn_recur_cpu, see the long comment
     * there for how the mapping was measured against llama.cpp. */
    const uint32_t h_k = h_v % kh;
    const uint32_t kd = kh * kdim, vd = vh * vdim;
    const uint32_t stride = kd * 2u + vd;
    float *Sh = S + (size_t)h_v * kdim * vdim;

    extern __shared__ float shq[];          /* [kdim] q̂ | [kdim] k̂ */
    float *q_s = shq, *k_s = shq + kdim;

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *base = cnv + (size_t)t * stride;
        const float *q = base + (size_t)h_k * kdim;
        const float *k = base + kd + (size_t)h_k * kdim;
        const float *v = base + kd * 2u + (size_t)h_v * vdim;
        for (uint32_t i = threadIdx.x; i < kdim; i += blockDim.x) {
            q_s[i] = q[i]; k_s[i] = k[i];
        }
        __syncthreads();
        const float beta  = bet[(size_t)t * vh + h_v];
        const float decay = dec[(size_t)t * vh + h_v];
        for (uint32_t d = threadIdx.x; d < vdim; d += blockDim.x) {
            float mem = 0.0f;
            for (uint32_t i = 0; i < kdim; i++)
                mem += Sh[(size_t)i * vdim + d] * k_s[i];
            const float upd = beta * (v[d] - mem * decay);
            float out = 0.0f;
            for (uint32_t i = 0; i < kdim; i++) {
                float *p = Sh + (size_t)i * vdim + d;
                const float sv = *p * decay + k_s[i] * upd;
                *p = sv;
                out += sv * q_s[i];
            }
            core[(size_t)t * vd + (size_t)h_v * vdim + d] = out * oscale;
        }
        /* q_s/k_s are overwritten by the next token — everyone must be done. */
        __syncthreads();
    }
}

/* Same recurrence, but with the per-head state resident in shared memory for the
 * whole token loop.
 *
 * The state is the hot data: every token reads S[i][d] twice and writes it once,
 * so a 128-token prefill drags the 64 KB state through global memory hundreds of
 * times per head. Measured on a GB10 (2026-08-13, docs/progress.md) the recurrence
 * was 86% of all prefill kernel time and ran at 1.8 GFLOP/s — that is latency-bound,
 * not bandwidth-bound. One block per v-head puts only vh*vdim threads in flight
 * (4096 here, against 48 SMs x 1536), which is nowhere near enough to hide a
 * global-memory round trip, and the recurrence cannot be parallelised over tokens
 * to fix that. Staging S in shared memory attacks the other half instead: it makes
 * each access cheap enough that low occupancy stops mattering.
 *
 * The arithmetic is untouched — same operations in the same order — so results stay
 * bit-identical to gdn_recur_kernel, to gdn_recur_cpu, and between prefill and
 * decode. cudatest's `decode==prefill` assertion (md == 0.0, not a tolerance) is
 * what holds that down.
 *
 * Usable only when the state fits the device's opt-in shared-memory limit, which
 * the caller checks; otherwise it falls back to gdn_recur_kernel above. */
__global__ void gdn_recur_smem_kernel(float *S, uint32_t n_tokens,
                                      uint32_t kh, uint32_t vh,
                                      uint32_t kdim, uint32_t vdim,
                                      const float *cnv, const float *bet,
                                      const float *dec, float *core, float oscale) {
    const uint32_t h_v = blockIdx.x;
    const uint32_t h_k = h_v % kh;
    const uint32_t kd = kh * kdim, vd = vh * vdim;
    const uint32_t stride = kd * 2u + vd;
    float *Sg = S + (size_t)h_v * kdim * vdim;

    extern __shared__ float sgd[];
    float *S_s = sgd;                                  /* [kdim][vdim] */
    float *q_s = sgd + (size_t)kdim * vdim;            /* [kdim] */
    float *k_s = q_s + kdim;                           /* [kdim] */

    /* Coalesced: consecutive threads take consecutive elements. */
    for (uint32_t idx = threadIdx.x; idx < kdim * vdim; idx += blockDim.x)
        S_s[idx] = Sg[idx];
    __syncthreads();

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *base = cnv + (size_t)t * stride;
        const float *q = base + (size_t)h_k * kdim;
        const float *k = base + kd + (size_t)h_k * kdim;
        const float *v = base + kd * 2u + (size_t)h_v * vdim;
        for (uint32_t i = threadIdx.x; i < kdim; i += blockDim.x) {
            q_s[i] = q[i]; k_s[i] = k[i];
        }
        __syncthreads();
        const float beta  = bet[(size_t)t * vh + h_v];
        const float decay = dec[(size_t)t * vh + h_v];
        for (uint32_t d = threadIdx.x; d < vdim; d += blockDim.x) {
            float mem = 0.0f;
            for (uint32_t i = 0; i < kdim; i++)
                mem += S_s[(size_t)i * vdim + d] * k_s[i];
            const float upd = beta * (v[d] - mem * decay);
            float out = 0.0f;
            for (uint32_t i = 0; i < kdim; i++) {
                float *p = S_s + (size_t)i * vdim + d;
                const float sv = *p * decay + k_s[i] * upd;
                *p = sv;
                out += sv * q_s[i];
            }
            core[(size_t)t * vd + (size_t)h_v * vdim + d] = out * oscale;
        }
        __syncthreads();
    }

    for (uint32_t idx = threadIdx.x; idx < kdim * vdim; idx += blockDim.x)
        Sg[idx] = S_s[idx];
}

/* Pick the shared-memory recurrence when the state fits, else the global one.
 *
 * The opt-in limit is queried and the attribute set once: cudaFuncSetAttribute on
 * every launch would add an API call to a path that runs twice per layer. A device
 * that cannot take the full state (or a model with a larger head) simply keeps the
 * original kernel, which is why this returns void and never fails. */
static void gdn_launch_recur(float *d_state, uint32_t n_tokens,
                             uint32_t kh, uint32_t vh, uint32_t kdim, uint32_t vdim,
                             const float *d_cnv, const float *d_bet,
                             const float *d_dec, float *d_core, float oscale) {
    uint32_t threads = vdim < 256u ? vdim : 256u;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;

    const size_t smem_small = 2u * (size_t)kdim * sizeof(float);
    const size_t smem_full  = (size_t)kdim * vdim * sizeof(float) + smem_small;

    /* Query and raise the cap once, but decide per call: the limit is a property of
     * the device, the requirement is a property of the model's head shape, and one
     * process can load more than one model. Caching the decision instead of the cap
     * would let a small first model green-light a later one that does not fit. */
    static int optin_max = -1;
    static int attr_set = 0;
    if (optin_max < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   dev) != cudaSuccess)
            optin_max = 0;
        if (optin_max > 0)
            attr_set = (cudaFuncSetAttribute(gdn_recur_smem_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             optin_max) == cudaSuccess);
        cudaGetLastError();   /* a refused opt-in must not surface as the next launch's error */
    }

    /* Staging the state costs one read and one write of the whole 64 KB regardless
     * of how many tokens follow, and asking for that much shared memory caps the
     * block at one per SM. Decode (n_tokens == 1) touches the state once, so it pays
     * the staging and gets nothing back: measured 12.70 -> 12.52 tok/s until this
     * threshold was added. Only take the shared path once there are enough tokens
     * for the saved global traffic to dominate. */
    if (attr_set && smem_full <= (size_t)optin_max && n_tokens >= 4u)
        gdn_recur_smem_kernel<<<vh, threads, smem_full>>>(
            d_state, n_tokens, kh, vh, kdim, vdim, d_cnv, d_bet, d_dec, d_core, oscale);
    else
        gdn_recur_kernel<<<vh, threads, smem_small>>>(
            d_state, n_tokens, kh, vh, kdim, vdim, d_cnv, d_bet, d_dec, d_core, oscale);
}

struct ds4x_cuda_gdn {
    float   *d_state;                  /* [vh][kdim][vdim] */
    float   *d_cnv, *d_bet, *d_dec, *d_core;
    float   *h_cnv, *h_bet, *h_dec, *h_core;   /* pinned staging */
    uint32_t kh, vh, kdim, vdim;
    uint32_t cap_tokens;               /* staging capacity, grown on demand */
    size_t   state_n;
    /* Pass-2 (conv + gates) inputs, for ds4x_cuda_gdn_pre_run.
     *
     * The four WEIGHT arrays are uploaded ONCE and kept: there is one handle
     * per linear layer, so they are constant for the handle's lifetime. Sending
     * them per call would dominate the transfer — conv1d_w alone is
     * conv_ch*K floats (98 KB for Qwen3.5-0.8B), which at one call per layer
     * per token is ~210 MB per 118 decoded tokens, more than the activations
     * this whole exercise is trying to stop moving. */
    float   *d_raw, *d_win, *d_bb, *d_ab;      /* per-call inputs */
    float   *h_pre;                            /* pinned staging for the above */
    size_t   pre_cap;                          /* floats in h_pre */
    float   *d_cw, *d_cb, *d_alog, *d_dt;      /* per-layer weights, uploaded once */
    float   *d_ssm;                            /* pass-4 gated-norm weight, likewise */
    uint32_t conv_ch, K;                       /* 0 = weights not uploaded yet */
};

static double g_gdn_ms_kernel = 0.0, g_gdn_ms_total = 0.0;
static unsigned long long g_gdn_calls = 0;
static cudaEvent_t g_gev0 = NULL, g_gev1 = NULL;

/* --- timing accumulators (see ds4x_cuda_stats) --- */
static double g_ms_kernel = 0.0, g_ms_total = 0.0;
static double g_ms_malloc = 0.0, g_ms_h2d = 0.0;   /* VRAM upload split */
static unsigned long long g_calls = 0;
static cudaEvent_t g_ev0 = NULL, g_ev1 = NULL;

static double host_ms(void) {
#ifdef _WIN32
    /* nvcc on Windows drives MSVC, which has no clock_gettime. */
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

extern "C" void ds4x_cuda_stats(double *ms_kernel, double *ms_total, uint64_t *calls) {
    if (ms_kernel) *ms_kernel = g_ms_kernel;
    if (ms_total)  *ms_total  = g_ms_total;
    if (calls)     *calls     = (uint64_t)g_calls;
}
extern "C" void ds4x_cuda_stats_reset(void) {
    g_ms_kernel = g_ms_total = 0.0; g_calls = 0;
}

extern "C" void ds4x_cuda_upload_stats(double *ms_malloc, double *ms_h2d) {
    if (ms_malloc) *ms_malloc = g_ms_malloc;
    if (ms_h2d)    *ms_h2d    = g_ms_h2d;
}

/* ---- public API ---------------------------------------------------------- */
extern "C" int ds4x_cuda_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) { set_err("cudaGetDeviceCount", e); return 0; }
    if (!g_dev[0]) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, 0) == cudaSuccess)
            snprintf(g_dev, sizeof(g_dev), "%s (cc%d.%d)", p.name, p.major, p.minor);
    }
    return 1;
}

extern "C" void ds4x_cuda_set_budget(uint64_t bytes) { g_budget = bytes; }

extern "C" ds4x_cuda_wt *ds4x_cuda_upload(const void *host, uint32_t type,
                                          uint32_t n_out, uint32_t n_in) {
    const uint32_t bc = blk_count(type);
    const uint64_t bb = blk_bytes(type);
    if (!host || bc == 0 || bb == 0 || n_in % bc) {
        snprintf(g_err, sizeof(g_err), "unsupported type %u or n_in %u not block-aligned", type, n_in);
        return NULL;
    }
    ds4x_cuda_wt *w = (ds4x_cuda_wt *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->type = type; w->n_out = n_out; w->n_in = n_in;
    w->bytes = (size_t)((uint64_t)n_out * (n_in / bc) * bb);

    /* Budget check BEFORE cudaMalloc: refusing here is what makes the user's
     * VRAM limit real. The x/y scratch buffers are counted too — they are small
     * per weight but there is one pair per matrix, so ignoring them would let a
     * many-layer model drift past the cap. */
    const uint64_t need = (uint64_t)w->bytes
                        + (uint64_t)n_in * sizeof(float)
                        + (uint64_t)n_out * sizeof(float);
    if (g_budget && g_bytes + need > g_budget) {
        snprintf(g_err, sizeof(g_err),
                 "VRAM budget %.2f GB reached (%.2f GB resident) — weight stays on the CPU",
                 (double)g_budget / 1073741824.0, (double)g_bytes / 1073741824.0);
        free(w);
        return NULL;
    }

    cudaError_t e;
    const double m0 = host_ms();
    if ((e = cudaMalloc(&w->d_data, w->bytes)) != cudaSuccess) { set_err("cudaMalloc(W)", e); free(w); return NULL; }
    const double m1 = host_ms();
    if ((e = cudaMemcpy(w->d_data, host, w->bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(W)", e); cudaFree(w->d_data); free(w); return NULL;
    }
    g_ms_malloc += m1 - m0;
    g_ms_h2d    += host_ms() - m1;
    if ((e = cudaMalloc(&w->d_x, (size_t)n_in  * sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&w->d_y, (size_t)n_out * sizeof(float))) != cudaSuccess) {
        set_err("cudaMalloc(xy)", e);
        cudaFree(w->d_data); cudaFree(w->d_x); cudaFree(w->d_y); free(w); return NULL;
    }
    g_bytes += need;
    return w;
}

/* Same expression as the budget check in ds4x_cuda_upload — weight + x/y
 * scratch. Kept in one place so alloc and free can never disagree. */
static uint64_t wt_acct_bytes(const ds4x_cuda_wt *w) {
    return (uint64_t)w->bytes
         + (uint64_t)w->n_in  * sizeof(float)
         + (uint64_t)w->n_out * sizeof(float);
}

extern "C" void ds4x_cuda_free(ds4x_cuda_wt *w) {
    if (!w) return;
    const uint64_t acct = wt_acct_bytes(w);
    if (acct <= g_bytes) g_bytes -= acct;
    cudaFree(w->d_data); cudaFree(w->d_x); cudaFree(w->d_y);
    free(w);
}

/* Matvec over a SUB-RANGE of an uploaded weight: rows [elem_off/n_in,
 * elem_off/n_in + n_out) of the same matrix. This is how a stacked
 * [n_expert][n_ff][n_embd] MoE tensor is used without uploading each expert
 * separately — one device buffer per tensor instead of n_expert of them (a
 * 128-expert × 48-layer model would otherwise need ~18k handles and 3x that
 * many cudaMallocs).
 *
 * `elem_off == 0 && n_out == w->n_out` is exactly ds4x_cuda_matvec. The scratch
 * buffers are the parent's: d_x holds n_in floats (unchanged) and d_y holds the
 * parent's n_out, which is >= any slice's. */
/* THE matvec dispatch, singular. It used to exist in three copies -- one in
 * launch_matvec (the fused FFN path), one in ds4x_cuda_matvec_off and one in
 * ds4x_cuda_matvec -- and a fast path added to fewer than all three was silently
 * unreached. That is not a hypothetical: the parity gate goes through the third
 * copy, so it reported all green twice in a row (2026-08-13) while the new kernel
 * never executed once. Now there is one copy and no way to add half a path.
 *
 * Defined further down, next to the integer-domain kernels it selects. */
static void dispatch_matvec(uint32_t type, const unsigned char *base,
                            uint32_t n_in, uint32_t n_out,
                            const float *d_x, float *d_y);

extern "C" int ds4x_cuda_matvec_off(const ds4x_cuda_wt *w, uint64_t elem_off,
                                    uint32_t n_out, const float *x, float *y) {
    if (!w || !x || !y || n_out == 0) return -1;
    if (elem_off % w->n_in) return -1;            /* must start on a row */
    if (elem_off / w->n_in + n_out > w->n_out) return -1;   /* past the end */
    const double t0 = host_ms();
    if (!g_ev0) { cudaEventCreate(&g_ev0); cudaEventCreate(&g_ev1); }
    const uint32_t bc = blk_count(w->type);
    const uint32_t bb = (uint32_t)blk_bytes(w->type);
    const unsigned char *base = (const unsigned char *)w->d_data
                              + (size_t)(elem_off / bc) * bb;

    cudaError_t e;
    if ((e = cudaMemcpy(w->d_x, x, (size_t)w->n_in * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(x)", e); return -1;
    }
    cudaEventRecord(g_ev0);
    dispatch_matvec(w->type, base, w->n_in, n_out, w->d_x, w->d_y);
    cudaEventRecord(g_ev1);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("kernel launch", e); return -1; }
    if ((e = cudaMemcpy(y, w->d_y, (size_t)n_out * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        set_err("cudaMemcpy(y)", e); return -1;
    }
    float kms = 0.0f;
    if (cudaEventElapsedTime(&kms, g_ev0, g_ev1) == cudaSuccess) g_ms_kernel += kms;
    g_ms_total += host_ms() - t0;
    g_calls++;
    return 0;
}

extern "C" int ds4x_cuda_matvec(const ds4x_cuda_wt *w, const float *x, float *y) {
    if (!w || !x || !y) return -1;
    const double t0 = host_ms();
    if (!g_ev0) { cudaEventCreate(&g_ev0); cudaEventCreate(&g_ev1); }
    cudaError_t e;
    if ((e = cudaMemcpy(w->d_x, x, (size_t)w->n_in * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(x)", e); return -1;
    }
    cudaEventRecord(g_ev0);
    dispatch_matvec(w->type, (const unsigned char *)w->d_data,
                    w->n_in, w->n_out, w->d_x, w->d_y);
    cudaEventRecord(g_ev1);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("kernel launch", e); return -1; }
    if ((e = cudaMemcpy(y, w->d_y, (size_t)w->n_out * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        set_err("cudaMemcpy(y)", e); return -1;
    }
    float kms = 0.0f;
    if (cudaEventElapsedTime(&kms, g_ev0, g_ev1) == cudaSuccess) g_ms_kernel += kms;
    g_ms_total += host_ms() - t0;
    g_calls++;
    return 0;
}

/* ---- batched matmul API --------------------------------------------------- */
static double g_mm_ms_kernel = 0.0, g_mm_ms_total = 0.0;
static unsigned long long g_mm_calls = 0, g_mm_rows = 0;
static cudaEvent_t g_mev0 = NULL, g_mev1 = NULL;

static int stage_reserve(size_t nx, size_t ny) {
    cudaError_t e;
    if (nx > g_stage.n_x) {
        cudaFree(g_stage.d_x); cudaFreeHost(g_stage.h_x);
        g_stage.d_x = NULL; g_stage.h_x = NULL; g_stage.n_x = 0;
        if ((e = cudaMalloc(&g_stage.d_x, nx * sizeof(float))) != cudaSuccess ||
            (e = cudaHostAlloc(&g_stage.h_x, nx * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
            set_err("cudaMalloc(matmul X)", e); return -1;
        }
        g_stage.n_x = nx;
    }
    if (ny > g_stage.n_y) {
        cudaFree(g_stage.d_y); cudaFreeHost(g_stage.h_y);
        g_stage.d_y = NULL; g_stage.h_y = NULL; g_stage.n_y = 0;
        if ((e = cudaMalloc(&g_stage.d_y, ny * sizeof(float))) != cudaSuccess ||
            (e = cudaHostAlloc(&g_stage.h_y, ny * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
            set_err("cudaMalloc(matmul Y)", e); return -1;
        }
        g_stage.n_y = ny;
    }
    return 0;
}

/* Launch the matmul kernel for `nt` token-rows, DEVICE POINTER IN, DEVICE
 * POINTER OUT. No copies, no synchronize, no timing — the caller owns all
 * three.
 *
 * Extracted from ds4x_cuda_matmul so that a fused block (see
 * ds4x_cuda_swiglu below) can chain several matmuls with the intermediate
 * activations never leaving VRAM. That chaining is the whole point: measured
 * on an RTX 5060 Ti, each host<->device round trip costs ~57 us regardless of
 * how small the transfer is, and the per-call overhead was ~48% of all GPU
 * path time for a 0.8B model (docs/linear-attention-design.md §4m-bis). */
/* Single-token counterpart of launch_matmul: same device-in/device-out shape,
 * but the MATVEC kernel (one block per output row, no token tiling).
 *
 * Needed because the matmul kernel tiles TOK_TILE token-rows per block, so a
 * 1-token launch leaves all but one lane masked off. ds4x_cuda_matmul's comment
 * claims that costs what a matvec did; measured on a 5060 Ti it does not — the
 * first cut of the fused FFN routed decode through the matmul kernel and paid
 * +33 ms of kernel time per 32 tokens, eating half of what fusing had just
 * saved in round trips.
 *
 * It also keeps the numerics where they were: the unfused decode FFN went
 * through matvec_q, so using the matvec kernel here reproduces the old path
 * exactly rather than introducing a new reduction order. */
/* ---- integer-domain matvec (DP4A) ----------------------------------------
 * Decode's counterpart to the tensor-core prefill path, and the same idea: stop
 * converting the weights to float at all.
 *
 * Every matvec kernel above dequantizes each weight to fp32 and multiplies it by
 * an fp32 activation. That spends the arithmetic unit on unpacking rather than on
 * the dot product, and decode is where it hurts most because the whole token costs
 * one pass over the weights. Measured against llama.cpp on the SAME Q4_K_M file
 * (GB10, 2026-08-13): 26.70 tok/s vs 68.2, i.e. 68 GB/s of effective bandwidth
 * against their 174 out of 273 available.
 *
 * llama.cpp's answer, reproduced here: quantize the ACTIVATION to int8 as well,
 * then contract with DP4A (four int8 products and an int32 accumulate in one
 * instruction), and apply the scales once at the end. For Q4_K the identity is
 *
 *     sum(w*x) = d8 * ( d*sc * dot(q, xq) - dmin*mm * sum(xq) )
 *
 * so a sub-block needs exactly two DP4A chains: one against the weights, one
 * against 0x01010101 to get sum(xq) for the min term. Our Q4_K sub-block is 32
 * weights and llama.cpp's q8_1 activation block is also 32, so they line up with
 * no regrouping.
 *
 * NUMERICS: the activation now carries ~7 bits, so this cannot match the fp32
 * kernels bit for bit. It is a separate, switchable path for that reason
 * (IDLETOKEN_DS4X_NO_DP4A=1), and the parity gate checks it against a tolerance
 * rather than folding it into the exact assertions. */
static int8_t *g_q8_x = NULL;
static float  *g_q8_d = NULL;
static size_t  g_q8_cap = 0;

/* One warp per 32-element block: shuffle for the max, then write the int8s. */
__global__ void quantize_x_q8_kernel(const float *x, int8_t *qs, float *ds,
                                     uint32_t n_blocks) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const uint32_t lane = threadIdx.x & 31u;
    const float v = x[(size_t)b * 32 + lane];

    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    qs[(size_t)b * 32 + lane] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) ds[b] = d;
}

/* Warp shuffle, then one shared slot per warp, then the first warp finishes —
 * the tail every block-per-row matvec kernel needs. Written once so the three
 * integer-domain kernels below cannot drift apart in how they reduce. `sm` is the
 * caller's dynamic shared block (one float per warp); it stays a parameter
 * because `extern __shared__` has to be declared in the kernel itself. */
__device__ __forceinline__ void matvec_block_reduce(float acc, float *sm,
                                                    float *y, uint32_t row) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) sm[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < nwarps) ? sm[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) y[row] = s;
    }
}

__global__ void matvec_q4k_dp4a_kernel(const unsigned char *W, uint32_t n_in,
                                       const int8_t *xq, const float *xd, float *y) {
    const uint32_t row = blockIdx.x;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 144;
    const uint32_t n_units = n_in / 32u;          /* Q4_K sub-blocks == q8 blocks */

    float acc = 0.0f;
    for (uint32_t u = threadIdx.x; u < n_units; u += blockDim.x) {
        const uint32_t qb = u >> 3, uw = u & 7u;
        const unsigned char *p = rowp + (size_t)qb * 144;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        unsigned char sc, mm;
        get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);

        /* Same addressing as unit_dot's Q4_K case: 32 bytes hold two sub-blocks,
         * the even one in the low nibbles and the odd one in the high nibbles. */
        const unsigned int *qw = (const unsigned int *)(p + 16 + (size_t)(uw >> 1) * 32);
        const int sh = (uw & 1u) ? 4 : 0;
        const int *xu = (const int *)(xq + (size_t)u * 32);

        int sumi = 0, sumq = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int vi = (int)((qw[k] >> sh) & 0x0F0F0F0Fu);
            sumi = __dp4a(vi, xu[k], sumi);
            sumq = __dp4a(0x01010101, xu[k], sumq);   /* sum(xq) for the min term */
        }
        acc += xd[u] * (d * (float)sc * (float)sumi - dmin * (float)mm * (float)sumq);
    }
    extern __shared__ float smv4[];
    matvec_block_reduce(acc, smv4, y, row);
}

/* A Q6_K block is 210 B, so only every other one starts 4 B-aligned and a plain
 * 32-bit load faults with "misaligned address" (the fp32 kernels above record the
 * same trap). Two 16-bit loads are always safe here because every offset we take
 * into a block is even. llama.cpp solves it the same way (get_int_b2). */
__device__ __forceinline__ unsigned int ld_u32_align2(const unsigned char *p) {
    const unsigned short *h = (const unsigned short *)p;
    return (unsigned int)h[0] | ((unsigned int)h[1] << 16);
}

/* Q5_K, integer domain. Same identity as Q4_K -- a weight is d*sc*q - dmin*mm
 * with q in [0,31] -- the fifth bit just arrives from a separate 32-byte plane,
 * one bit per element, bit index == the sub-block index.
 *
 * THE DECOMPOSITION DIFFERS FROM matvec_q4k_dp4a_kernel ABOVE, deliberately.
 * That one hands a thread a whole 32-element unit, which is the densest possible
 * arithmetic (one float combine per 32 weights) but makes a warp read ql at a
 * 32-byte stride. Decode is bandwidth-bound -- 68 GB/s of the 273 this machine
 * has -- so coalescing is worth more than arithmetic density. Here a thread takes
 * ONE 4-byte ql word, so lanes 0..31 of a warp read 128 CONSECUTIVE bytes, at the
 * cost of one float combine per 8 weights. llama.cpp makes the same trade (its
 * VDR_*_MMVQ widths) and reaches 174 GB/s on this file.
 *
 * Q4_K keeps the unit-per-thread shape because that one is measured (+30% decode)
 * and changing it is a separate experiment, not a free ride on this one. */
__global__ void matvec_q5k_dp4a_kernel(const unsigned char *W, uint32_t n_in,
                                       const int8_t *xq, const float *xd, float *y) {
    const uint32_t row  = blockIdx.x;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 176;
    const uint32_t nwords = nblk * 32u;      /* 128 B of ql per block, 4 B per word */

    float acc = 0.0f;
    for (uint32_t w = threadIdx.x; w < nwords; w += blockDim.x) {
        const uint32_t qb = w >> 5, m = w & 31u;
        const uint32_t uw = (m >> 3) << 1;   /* the even sub-block; +1 is the odd one */
        const uint32_t l4 = m & 7u;          /* which 4 of the sub-block's 32 elements */
        const unsigned char *p = rowp + (size_t)qb * 176;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        /* 176 is a multiple of 4 and the base is cudaMalloc-aligned, so unlike
         * Q6_K below these can be plain 32-bit loads. */
        const unsigned int qlw = *(const unsigned int *)(p + 48 + (size_t)m * 4);
        const unsigned int qhw = *(const unsigned int *)(p + 16 + (size_t)l4 * 4);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned char sc, mm;
            get_scale_min_k4_d((int)uw + i, p + 4, &sc, &mm);
            /* low nibbles are the even sub-block, high nibbles the odd one */
            const int vi = (int)(((qlw >> (4 * i)) & 0x0F0F0F0Fu)
                                 | (((qhw >> (uw + (uint32_t)i)) & 0x01010101u) << 4));
            const uint32_t b = qb * 8u + uw + (uint32_t)i;      /* q8 block index */
            const int xw = *(const int *)(xq + (size_t)b * 32u + (size_t)l4 * 4u);
            const int sumi = __dp4a(vi, xw, 0);
            const int sumq = __dp4a(0x01010101, xw, 0);   /* sum(xq) for the min term */
            acc += xd[b] * (d * (float)sc * (float)sumi - dmin * (float)mm * (float)sumq);
        }
    }
    extern __shared__ float smv5[];
    matvec_block_reduce(acc, smv5, y, row);
}

/* Q6_K, integer domain. No min term here: a weight is d*sc*(q-32) with q in
 * [0,63], so one DP4A chain per scale group and __vsubss4 to bias the packed
 * bytes into int8 range.
 *
 * The output head lands here and it is the single largest matvec in a Q4_K_M
 * model -- Qwen3.5 ties the embedding, so token_embd stays Q6_K and is re-read in
 * full for every decoded token (12.6 ms of 21.4 ms of fp32 matvec time before
 * this kernel existed).
 *
 * Word-per-thread like Q5_K above; the extra reason here is that a Q6_K byte
 * carries TWO elements (low nibble -> quarter g, high nibble -> quarter g+2), so
 * one 4-byte load feeds eight weights and both quarters share it. */
__global__ void matvec_q6k_dp4a_kernel(const unsigned char *W, uint32_t n_in,
                                       const int8_t *xq, const float *xd, float *y) {
    const uint32_t row  = blockIdx.x;
    const uint32_t nblk = n_in / QK_K;
    const unsigned char *rowp = W + (size_t)row * nblk * 210;
    const uint32_t nwords = nblk * 32u;      /* 128 B of ql per block, 4 B per word */

    float acc = 0.0f;
    for (uint32_t w = threadIdx.x; w < nwords; w += blockDim.x) {
        const uint32_t qb = w >> 5, j = w & 31u;
        const unsigned char *p = rowp + (size_t)qb * 210;
        const uint32_t nn = j >> 4;             /* which 128-element group */
        const uint32_t h  = (j >> 3) & 1u;      /* which quarter pair (g = h, h+2) */
        const uint32_t l4 = j & 7u;             /* which 4 of the quarter's 32 */
        const uint32_t half = (l4 >= 4u) ? 1u : 0u;   /* scales split at element 16 */
        const signed char *sc = (const signed char *)(p + 192);
        const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
        const unsigned int qlw = ld_u32_align2(p + (size_t)j * 4);
        const unsigned int qhw = ld_u32_align2(p + 128 + (size_t)nn * 32 + (size_t)l4 * 4);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint32_t g = h + 2u * (uint32_t)i;
            const int vl = (int)((qlw >> (4 * i)) & 0x0F0F0F0Fu);
            const int vh = (int)(((qhw >> (2u * g)) & 0x03030303u) << 4);
            /* q in [0,63]; bias to [-32,31] so the bytes read as signed int8.
             * Byte-wise, so a borrow cannot leak into the next element. */
            const int vi = (int)__vsubss4((unsigned int)(vl | vh), 0x20202020u);
            const uint32_t b = qb * 8u + nn * 4u + g;        /* q8 block index */
            const int xw = *(const int *)(xq + (size_t)b * 32u + (size_t)l4 * 4u);
            acc += xd[b] * d * (float)sc[nn * 8u + 2u * g + half]
                 * (float)__dp4a(vi, xw, 0);
        }
    }
    extern __shared__ float smv6[];
    matvec_block_reduce(acc, smv6, y, row);
}

/* 0 on success, -1 to fall back to the fp32 kernels. Takes an explicit base and
 * n_out rather than a ds4x_cuda_wt because ds4x_cuda_matvec_off slices rows out
 * of a stacked MoE weight. */
static int launch_matvec_dp4a(uint32_t type, const unsigned char *base,
                              uint32_t n_in, uint32_t n_out,
                              const float *d_x, float *d_y) {
    if (n_in % QK_K) return -1;
    if (type != T_Q4_K && type != T_Q5_K && type != T_Q6_K) return -1;
    /* Reports the first call PER TYPE that the path actually accepts. Printing
     * on entry instead would say nothing about whether the kernel ran, which is
     * the exact failure this diagnostic exists to catch. */
    if (getenv("IDLETOKEN_DS4X_DP4A_DEBUG")) {
        static unsigned seen = 0u;
        const unsigned bit = 1u << (type & 31u);
        if (!(seen & bit)) { seen |= bit;
            fprintf(stderr, "ds4x: dp4a matvec live: type=%u n_in=%u n_out=%u\n",
                    type, n_in, n_out); }
    }
    const uint32_t n_blocks = n_in / 32u;
    if ((size_t)n_in > g_q8_cap) {
        cudaFree(g_q8_x); cudaFree(g_q8_d);
        if (cudaMalloc(&g_q8_x, (size_t)n_in) != cudaSuccess ||
            cudaMalloc(&g_q8_d, (size_t)n_blocks * sizeof(float)) != cudaSuccess) {
            g_q8_x = NULL; g_q8_d = NULL; g_q8_cap = 0; cudaGetLastError(); return -1;
        }
        g_q8_cap = n_in;
    }
    const uint32_t qth = 256;
    quantize_x_q8_kernel<<<(n_blocks * 32u + qth - 1u) / qth, qth>>>(
        d_x, g_q8_x, g_q8_d, n_blocks);

    const uint32_t th = 128;
    const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
    if (type == T_Q4_K)
        matvec_q4k_dp4a_kernel<<<n_out, th, sh>>>(base, n_in, g_q8_x, g_q8_d, d_y);
    else if (type == T_Q5_K)
        matvec_q5k_dp4a_kernel<<<n_out, th, sh>>>(base, n_in, g_q8_x, g_q8_d, d_y);
    else
        matvec_q6k_dp4a_kernel<<<n_out, th, sh>>>(base, n_in, g_q8_x, g_q8_d, d_y);
    return 0;
}

/* Declared far above; see the note there on why there is exactly one of these. */
static void dispatch_matvec(uint32_t type, const unsigned char *base,
                            uint32_t n_in, uint32_t n_out,
                            const float *d_x, float *d_y) {
    if (q4k_narrow(type, n_in)) {
        const uint32_t th = MV_WARPS * 32u;
        matvec_q4k_narrow_kernel<<<(n_out + MV_WARPS - 1u) / MV_WARPS, th>>>(
            base, n_in, n_out, d_x, d_y);
        return;
    }
    if (ds4x_dp4a_wanted() &&
        launch_matvec_dp4a(type, base, n_in, n_out, d_x, d_y) == 0) return;

    const uint32_t th = 128;
    const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
    if (type == T_Q4_K) {   /* coalesced specialization: dominant type in Q4_K_M */
        matvec_q4k_kernel<<<n_out, th, sh>>>(base, n_in, d_x, d_y);
        return;
    }
    if (type == T_Q6_K) {   /* the tied output head, if DP4A is switched off */
        matvec_q6k_kernel<<<n_out, th, sh>>>(base, n_in, d_x, d_y);
        return;
    }
    /* Generic: size the block to the row's 32-element units (cap 256), rounded up
     * to a whole warp — the old sizing (one thread per QUANT BLOCK) left Q4_K rows
     * with 16 busy threads out of 256. Shared = one float per warp. */
    const uint32_t n_units = n_in / UNIT;
    uint32_t threads = n_units < 256 ? n_units : 256;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;
    const uint32_t shmem = ((threads + 31u) / 32u) * (uint32_t)sizeof(float);
    matvec_kernel<<<n_out, threads, shmem>>>(
        base, type, n_in, blk_count(type), (uint32_t)blk_bytes(type), d_x, d_y);
}

static int launch_matvec(const ds4x_cuda_wt *w, const float *d_x, float *d_y) {
    dispatch_matvec(w->type, (const unsigned char *)w->d_data,
                    w->n_in, w->n_out, d_x, d_y);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { set_err("matvec launch", e); return -1; }
    return 0;
}

/* ---- cuBLAS / tensor-core path -------------------------------------------
 * Only the arithmetic unit is worth changing here, not the library. Measured on
 * a GB10 with this model's real shape (N=8192, K=2560), cuBLAS in FP32 is
 * 5.87 TFLOP/s at M=128 — the same as matmul_tiled_kernel above (5.8). Handing
 * the work to a library buys nothing on its own. What buys something is the
 * tensor cores, which need a 16-bit input type:
 *
 *     M=128   FP32 5.87   TF32 11.49   BF16 25.13  TFLOP/s
 *     M=256   FP32 9.57   TF32 15.77   BF16 46.96
 *     M=512   FP32 11.46  TF32 19.57   BF16 72.42
 *
 * That table is also the answer to "why does PyTorch get faster on longer prompts
 * and we don't": the FP32 column barely doubles across the range, the BF16 one
 * nearly triples.
 *
 * A BF16 GGUF stores each row as n_in consecutive bfloat16 values (blk_count 1,
 * blk_bytes 2), which is exactly a CUDA_R_16BF matrix — so for these weights the
 * tensor-core path needs no dequantization at all, only the activations converted.
 *
 * NUMERICS: inputs are rounded to 8 mantissa bits (accumulation stays FP32 via
 * CUBLAS_COMPUTE_32F), so results move by ~1e-3 relative, not the ~1e-7 of the
 * hand-written kernels, and cuBLAS picks a different algorithm per M, so a token's
 * result is no longer independent of how many tokens shared its launch. This path
 * therefore CANNOT satisfy `chunk==1-by-1`'s md == 0.0. It is opt-in per weight
 * type for that reason, and the bit-exact kernels remain the default for
 * everything else. */
static cublasHandle_t g_blas = NULL;
static __nv_bfloat16 *g_bf_x = NULL;
static size_t g_bf_x_cap = 0;
static __nv_bfloat16 *g_bf_w = NULL;
static size_t g_bf_w_cap = 0;

/* Below this many tokens the dequantize-to-bf16 pass costs more than the tensor
 * cores save, because it is a fixed cost in the weight's size while the GEMM it
 * feeds shrinks with M. Measured shape (N=8192, K=2560) on a GB10: the expansion
 * is ~0.21 ms, the tiled kernel does M=16 in ~0.12 ms outright. Weights already
 * stored as bf16 skip the pass and so have no threshold. */
#define TC_MIN_TOKENS 64

__global__ void f32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

/* Expand a quantized weight matrix into a dense bf16 one, one thread per unit.
 * Reuses unit_dequant_row so the element mapping cannot drift from the kernels
 * that read the same layout. */
__global__ void dequant_to_bf16_kernel(const unsigned char *W, uint32_t type,
                                       uint32_t n_in, uint32_t n_out,
                                       uint32_t bc, uint32_t bb,
                                       __nv_bfloat16 *out) {
    const uint32_t n_units = n_in / UNIT;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_out * n_units) return;
    const uint32_t row = (uint32_t)(idx / n_units), u = (uint32_t)(idx % n_units);
    const size_t rowbytes = (size_t)(n_in / bc) * bb;
    /* Straight into the destination. Staging through a float[UNIT] first cost
     * 128 bytes of local memory per thread across ~650k threads, and that spill
     * made this kernel 67% of prefill time (0.75 ms/call against a ~0.21 ms
     * bandwidth floor) -- it was slower than the GEMM it was feeding. */
    unit_dequant_row<__nv_bfloat16>(type, W + row * rowbytes, u,
                                    out + (size_t)row * n_in + (size_t)u * UNIT, 1);
}

/* Returns 0 on success, -1 to mean "fall back to the hand-written kernel". */
static int tc_bail(const char *why) {
    static int said = 0;
    if (!said && getenv("IDLETOKEN_DS4X_TC_DEBUG")) {
        said = 1;
        fprintf(stderr, "ds4x: tensor-core matmul declined: %s\n", why);
    }
    return -1;
}

static int launch_matmul_tc(const ds4x_cuda_wt *w, const float *d_x, float *d_y,
                            uint32_t nt) {
    const int need_expand = (w->type != T_BF16);
    if (need_expand && nt < TC_MIN_TOKENS) return tc_bail("too few tokens to amortize the dequant");
    if (w->n_in % UNIT) return tc_bail("n_in not a multiple of UNIT");
    if (!g_blas && cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) { g_blas = NULL; return tc_bail("cublasCreate failed"); }

    const size_t nx = (size_t)nt * w->n_in;
    if (nx > g_bf_x_cap) {
        cudaFree(g_bf_x);
        if (cudaMalloc(&g_bf_x, nx * sizeof(__nv_bfloat16)) != cudaSuccess) {
            g_bf_x = NULL; g_bf_x_cap = 0; cudaGetLastError();
            return tc_bail("cudaMalloc for the bf16 activation scratch failed");
        }
        g_bf_x_cap = nx;
    }
    const uint32_t th = 256;
    f32_to_bf16_kernel<<<(uint32_t)((nx + th - 1) / th), th>>>(d_x, g_bf_x, nx);

    const void *dW = w->d_data;
    if (need_expand) {
        /* One scratch buffer sized to the largest weight seen, not a bf16 copy of
         * the whole model: keeping every tensor expanded would undo the point of
         * shipping quantized weights (2.55 GB -> 8 GB for this 4B model). The cost
         * is re-expanding a tensor once per prefill chunk. */
        const size_t nw = (size_t)w->n_out * w->n_in;
        if (nw > g_bf_w_cap) {
            cudaFree(g_bf_w);
            if (cudaMalloc(&g_bf_w, nw * sizeof(__nv_bfloat16)) != cudaSuccess) {
                g_bf_w = NULL; g_bf_w_cap = 0; cudaGetLastError();
                return tc_bail("cudaMalloc for the bf16 weight scratch failed");
            }
            g_bf_w_cap = nw;
        }
        const size_t units = (size_t)w->n_out * (w->n_in / UNIT);
        dequant_to_bf16_kernel<<<(uint32_t)((units + th - 1) / th), th>>>(
            (const unsigned char *)w->d_data, w->type, w->n_in, w->n_out,
            blk_count(w->type), (uint32_t)blk_bytes(w->type), g_bf_w);
        dW = g_bf_w;
    }

    /* Row-major Y = X * W^T is col-major Y^T = W^T_col * X_col. */
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t s = cublasGemmEx(
        g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)w->n_out, (int)nt, (int)w->n_in, &alpha,
        dW, CUDA_R_16BF, (int)w->n_in,
        g_bf_x,    CUDA_R_16BF, (int)w->n_in,
        &beta, d_y, CUDA_R_32F, (int)w->n_out,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) return tc_bail("cublasGemmEx failed");
    return 0;
}

/* ---- MMQ: batched matmul in the integer domain ---------------------------
 * Prefill's counterpart to the DP4A matvecs, and the way out of the wall the
 * previous attempt hit. That attempt expanded each quantized weight into bf16 and
 * handed it to cuBLAS: the GEMM itself got 8.6x faster (231.7 -> 27 ms) and the
 * expansion cost 197 ms, for a net of zero. The cost had nowhere to amortise --
 * a weight is used exactly ONCE per prefill, so expanding it can never pay.
 *
 * MMQ does not expand anything. The ACTIVATION is quantized to int8 instead (one
 * cheap pass over a small matrix), the weights go into shared memory as int8 in
 * their own layout, and the contraction happens in integers with the scales
 * applied per 16-element group at the end. For Q4_K/Q5_K a group's real dot is
 *
 *     d8*ds*dot(q, xq) - dm*(d8*sum(xq))
 *
 * with ds = d*sc and dm = dmin*mm from the weight, and both d8 terms precomputed
 * per token during activation quantization (that is what q8_1 stores in
 * llama.cpp). Q6_K has no min term, just d*sc*(q-32) with a scale per 16.
 *
 * Two things get 4x cheaper at once, and ncu said both were binding on the fp32
 * tiled GEMM (l1tex 65.7%, sm__throughput 21%): the shared-memory tiles shrink
 * from float to int8, and one DP4A replaces four FMAs.
 *
 * 16 ELEMENTS IS THE UNIT OF SCALE, not 32. Q4_K and Q5_K carry one (scale, min)
 * per 32 and simply repeat it across the two halves, but Q6_K's scale changes
 * every 16 -- so a K step is 32 (one activation block) split into two halves that
 * flush independently. Sizing the flush at 32 would have quietly excluded Q6_K,
 * which is the tied output head.
 *
 * NUMERICS: the activation carries ~7 bits, so this is not bit-exact against the
 * fp32 kernels (`vs cpu` lands at ~2e-3, same order as every other quantized
 * path). It DOES keep `chunk==1-by-1` exact -- see the note in the header. */
#define MMQ_BK 32                      /* K elements per shared step == one q8 block */
#define MMQ_KW (MMQ_BK / 4)            /* ...packed as this many int8-quad words */
#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_TM 8
#define MMQ_TN 4
#define MMQ_THREADS ((MMQ_BM / MMQ_TM) * (MMQ_BN / MMQ_TN))   /* 16*16 = 256 */
/* Left to itself the compiler took 178 registers/thread here, which is one block
 * per SM and 16.6% of peak warps — the fp32 GEMM this replaces ran at 62%. The
 * accumulators alone need 64 (TM*TN floats plus TM*TN ints), so the budget is
 * genuinely tight; capping it makes ptxas trade re-loads for residency instead of
 * hoisting everything. Override to A/B it without editing: */
#ifndef MMQ_BLOCKS_PER_SM
#define MMQ_BLOCKS_PER_SM 2
#endif

static_assert(MMQ_BN * 2 <= MMQ_THREADS,
              "the weight tile is filled by two threads per row");

/* Activation quantization for MMQ. Same int8 rounding as quantize_x_q8_kernel,
 * but it also emits the two per-16 sums the min term needs, because those are a
 * property of the ACTIVATION alone and would otherwise be recomputed once per
 * weight row. A __shfl_xor with offsets 8..1 never crosses lane 16, so the two
 * halves reduce independently in the same instruction stream. */
__global__ void quantize_x_q8h_kernel(const float *x, int8_t *qs,
                                      float *sd, float *ss, uint32_t n_blocks) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const uint32_t lane = threadIdx.x & 31u;
    const float v = x[(size_t)b * 32 + lane];

    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    const int   q  = __float2int_rn(v * id);
    qs[(size_t)b * 32 + lane] = (int8_t)q;

    int s = q;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0)  { sd[(size_t)b * 2 + 0] = d; ss[(size_t)b * 2 + 0] = d * (float)s; }
    if (lane == 16) { sd[(size_t)b * 2 + 1] = d; ss[(size_t)b * 2 + 1] = d * (float)s; }
}

/* Pack the 16 weights of unit `u`, half `h`, of one row into four int8-quad words
 * and hand back the two factors that turn an integer dot into the real one:
 * `ds` multiplies the dot, `dm` multiplies the activation's own sum.
 *
 * This is the same bit unpacking as unit_dot/the matvec kernels, with the ONE
 * difference that it stops before multiplying by anything. Kept in one function
 * per type rather than templated on the caller, because the three layouts share
 * no useful structure past "16 consecutive k live in four 32-bit words". */
template<uint32_t TYPE>
__device__ __forceinline__ void mmq_pack16(const unsigned char *rowp,
                                           uint32_t u, uint32_t h,
                                           int *out, float *ds, float *dm) {
    const uint32_t qb = u >> 3, uw = u & 7u;
    if (TYPE == T_Q4_K) {
        const unsigned char *p = rowp + (size_t)qb * 144;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        unsigned char sc, mm;
        get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
        *ds = d * (float)sc;  *dm = dmin * (float)mm;
        const unsigned int *qw = (const unsigned int *)(p + 16 + (size_t)(uw >> 1) * 32);
        const int sh = (uw & 1u) ? 4 : 0;
        #pragma unroll
        for (int k = 0; k < 4; k++)
            out[k] = (int)((qw[h * 4 + (uint32_t)k] >> sh) & 0x0F0F0F0Fu);
    } else if (TYPE == T_Q5_K) {
        const unsigned char *p = rowp + (size_t)qb * 176;
        const float d    = f16_to_f32_d(*(const unsigned short *)p);
        const float dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
        unsigned char sc, mm;
        get_scale_min_k4_d((int)uw, p + 4, &sc, &mm);
        *ds = d * (float)sc;  *dm = dmin * (float)mm;
        const unsigned int *ql = (const unsigned int *)(p + 48 + (size_t)(uw >> 1) * 32);
        const unsigned int *qh = (const unsigned int *)(p + 16);
        const int sh = (uw & 1u) ? 4 : 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t w = h * 4 + (uint32_t)k;
            out[k] = (int)(((ql[w] >> sh) & 0x0F0F0F0Fu)
                           | (((qh[w] >> uw) & 0x01010101u) << 4));
        }
    } else {   /* T_Q6_K */
        const uint32_t nn = uw >> 2, g = uw & 3u;
        const unsigned char *p = rowp + (size_t)qb * 210;
        const float d = f16_to_f32_d(*(const unsigned short *)(p + 208));
        const signed char *sc = (const signed char *)(p + 192);
        *ds = d * (float)sc[nn * 8u + 2u * g + h];  *dm = 0.0f;
        /* 210 is not a multiple of 4 -- two 16-bit loads, as in the matvec. */
        const unsigned char *qlb = p + (size_t)nn * 64 + (size_t)(g & 1u) * 32 + (size_t)h * 16;
        const unsigned char *qhb = p + 128 + (size_t)nn * 32 + (size_t)h * 16;
        const int lsh = (g >= 2u) ? 4 : 0;
        const uint32_t hsh = 2u * g;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned int ql = ld_u32_align2(qlb + (size_t)k * 4);
            const unsigned int qh = ld_u32_align2(qhb + (size_t)k * 4);
            out[k] = (int)__vsubss4(((ql >> lsh) & 0x0F0F0F0Fu)
                                    | (((qh >> hsh) & 0x03030303u) << 4), 0x20202020u);
        }
    }
}

/* HALVES is the number of times a K step has to stop and apply scales.
 *
 * Q6_K genuinely needs 2 — its scale changes every 16 elements. Q4_K and Q5_K
 * carry one (scale, min) per 32 and were paying for a split they do not need:
 * the flush is ~128 FFMA + 48 shared loads against the 256 DP4A it interrupts,
 * so halving how often it runs takes a third of the instruction count off the
 * dominant type in a Q4_K_M model. That is the whole reason this kernel is
 * templated rather than switching on a runtime `type`. */
/* int8 tensor core, one 8x8 output tile over k=16 per instruction.
 *
 * m8n8k16 and NOT m16n8k32, which is twice the work per instruction: the latter
 * needs sm_80 and this project builds with nvcc's default architecture (sm_75
 * under CUDA 13), deliberately -- the hardware floor is "NVIDIA with >= 4 GB and
 * CUDA", and one of the repo's own test machines is an RTX 2070. Raising the
 * floor to Ampere to get a 2x on one instruction would drop a node that
 * currently passes the cross-machine gates. Moving up means adding -gencode for
 * several arches, not swapping the shape.
 *
 * Fragment layout (PTX ISA, .m8n8k16 with .s8), gid = lane>>2, tig = lane&3:
 *   A 8x16 : one b32 = A[gid][tig*4 .. +3]
 *   B 16x8 : one b32 = B[tig*4 .. +3][gid]
 *   C 8x8  : two b32 = C[gid][tig*2], C[gid][tig*2+1]
 * Both operand fragments are "four consecutive k for one row", which is exactly
 * how MMQ already packs its shared tiles for DP4A -- so the tiles, the fills and
 * the scales are untouched and only the contraction changes. */
__device__ __forceinline__ void mma_m8n8k16_s8(int *c, int a, int b) {
#if __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
#else
    /* No tensor cores here. The host must not have selected this path; failing
     * loudly beats returning zeros that look like a very confident model. */
    (void)a; (void)b; c[0] = 0x7fffffff; c[1] = 0x7fffffff;
#endif
}

#define MMA_WM 4                      /* warps across M */
#define MMA_WN 2                      /* warps across N */
#define MMA_MI (MMQ_BM / (MMA_WM * 8))  /* 8-row mma tiles per warp   = 4 */
#define MMA_NI (MMQ_BN / (MMA_WN * 8))  /* 8-col mma tiles per warp   = 4 */
static_assert(MMA_WM * MMA_WN * 32 == MMQ_THREADS, "warp grid must cover the block");

template<uint32_t TYPE, int MMA>
__global__ __launch_bounds__(MMQ_THREADS, MMQ_BLOCKS_PER_SM)
void matmul_mmq_kernel(const unsigned char *W,
                       uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                       const int8_t *Xq, const float *Xsd, const float *Xss,
                       float *Y) {
    constexpr int HALVES = (TYPE == T_Q6_K) ? 2 : 1;
    constexpr uint32_t KPF = (uint32_t)(MMQ_KW / HALVES);   /* words per flush */
    constexpr uint32_t BB  = (TYPE == T_Q4_K) ? 144u : (TYPE == T_Q5_K) ? 176u : 210u;
    const uint32_t n0 = blockIdx.x * MMQ_BN;
    const uint32_t m0 = blockIdx.y * MMQ_BM;
    const size_t rowbytes = (size_t)(n_in / QK_K) * BB;

    /* +1 int of padding: without it the tn stride (MMQ_TN rows of MMQ_KW ints)
     * is a multiple of 32 words and every lane of a warp hits the same bank. */
    __shared__ int   As[MMQ_BM][MMQ_KW + 1];
    __shared__ int   Bs[MMQ_BN][MMQ_KW + 1];
    __shared__ float Asd[MMQ_BM][2], Ass[MMQ_BM][2];
    __shared__ float Bsd[MMQ_BN][2], Bsm[MMQ_BN][2];

    const uint32_t tid = threadIdx.x;
    /* DP4A indexing: a thread owns a TM x TN block of outputs. */
    const uint32_t tm = tid / (MMQ_BN / MMQ_TN);        /* 0..15 */
    const uint32_t tn = tid % (MMQ_BN / MMQ_TN);        /* 0..15 */
    /* MMA indexing: a WARP owns MMA_MI x MMA_NI tiles of 8x8. */
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const uint32_t wm = warp / MMA_WN, wn = warp % MMA_WN;
    const uint32_t gid = lane >> 2, tig = lane & 3u;
    const uint32_t mbase = wm * (MMA_MI * 8u), nbase = wn * (MMA_NI * 8u);

    float acc[MMQ_TM][MMQ_TN];
    float facc[MMA_MI][MMA_NI][2];
    if (MMA) {
        #pragma unroll
        for (int mi = 0; mi < MMA_MI; mi++)
            #pragma unroll
            for (int ni = 0; ni < MMA_NI; ni++) facc[mi][ni][0] = facc[mi][ni][1] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < MMQ_TM; i++)
            #pragma unroll
            for (int j = 0; j < MMQ_TN; j++) acc[i][j] = 0.0f;
    }

    const uint32_t n_units = n_in / MMQ_BK;
    for (uint32_t u = 0; u < n_units; u++) {
        for (uint32_t idx = tid; idx < MMQ_BM * MMQ_KW; idx += MMQ_THREADS) {
            const uint32_t m = idx / MMQ_KW, kw = idx % MMQ_KW;
            const uint32_t mm = m0 + m;
            As[m][kw] = (mm < n_tok)
                ? ((const int *)(Xq + (size_t)mm * n_in + (size_t)u * MMQ_BK))[kw]
                : 0;
        }
        for (uint32_t idx = tid; idx < MMQ_BM * 2u; idx += MMQ_THREADS) {
            const uint32_t m = idx >> 1, hh = idx & 1u;
            const uint32_t mm = m0 + m;
            const size_t hi = (size_t)mm * (n_in / 16u) + (size_t)u * 2u + hh;
            Asd[m][hh] = (mm < n_tok) ? Xsd[hi] : 0.0f;
            Ass[m][hh] = (mm < n_tok) ? Xss[hi] : 0.0f;
        }
        /* Two threads per weight row, one per 16-element scale group. The fp32
         * GEMM gave a whole row to one thread and left 3/4 of the block idle
         * here; that step is unchanged in cost while everything around it got 4x
         * cheaper, so it would now dominate. */
        if (tid < MMQ_BN * 2u) {
            const uint32_t n = tid >> 1, hh = tid & 1u;
            const uint32_t nn = n0 + n;
            int q[4] = {0, 0, 0, 0};
            float ds = 0.0f, dm = 0.0f;
            if (nn < n_out)
                mmq_pack16<TYPE>(W + (size_t)nn * rowbytes, u, hh, q, &ds, &dm);
            #pragma unroll
            for (int k = 0; k < 4; k++) Bs[n][hh * 4u + (uint32_t)k] = q[k];
            Bsd[n][hh] = ds; Bsm[n][hh] = dm;
        }
        __syncthreads();

        /* KG = how many k=16 mma groups fit between two scale flushes: both of
         * them at HALVES == 1 (Q4_K/Q5_K, one scale per 32), one at HALVES == 2
         * (Q6_K). The flush is the expensive part -- ~3 float ops per
         * accumulator against 4 accumulators per mma -- so this ratio, not the
         * tile shape, is what bounds the tensor-core win. */
        constexpr uint32_t KG = 2u / (uint32_t)HALVES;
        #pragma unroll
        for (uint32_t hh = 0; MMA && hh < (uint32_t)HALVES; hh++) {
            int c[MMA_MI][MMA_NI][2];
            #pragma unroll
            for (int mi = 0; mi < MMA_MI; mi++)
                #pragma unroll
                for (int ni = 0; ni < MMA_NI; ni++) c[mi][ni][0] = c[mi][ni][1] = 0;

            #pragma unroll
            for (uint32_t g = 0; g < KG; g++) {
                const uint32_t kw = (hh * KG + g) * 4u + tig;
                int af[MMA_MI], bf[MMA_NI];
                #pragma unroll
                for (int mi = 0; mi < MMA_MI; mi++)
                    af[mi] = As[mbase + (uint32_t)mi * 8u + gid][kw];
                #pragma unroll
                for (int ni = 0; ni < MMA_NI; ni++)
                    bf[ni] = Bs[nbase + (uint32_t)ni * 8u + gid][kw];
                #pragma unroll
                for (int mi = 0; mi < MMA_MI; mi++)
                    #pragma unroll
                    for (int ni = 0; ni < MMA_NI; ni++)
                        mma_m8n8k16_s8(c[mi][ni], af[mi], bf[ni]);
            }

            /* One accumulator's row is mbase+mi*8+gid; its two columns are
             * nbase+ni*8+tig*2 and +1. Same scale algebra as the DP4A flush. */
            #pragma unroll
            for (int mi = 0; mi < MMA_MI; mi++) {
                const uint32_t r = mbase + (uint32_t)mi * 8u + gid;
                const float asd = Asd[r][hh];
                const float ass = (HALVES == 1) ? (Ass[r][0] + Ass[r][1]) : Ass[r][hh];
                #pragma unroll
                for (int ni = 0; ni < MMA_NI; ni++) {
                    const uint32_t cc = nbase + (uint32_t)ni * 8u + tig * 2u;
                    #pragma unroll
                    for (int e = 0; e < 2; e++)
                        facc[mi][ni][e] += asd * Bsd[cc + (uint32_t)e][hh]
                                               * (float)c[mi][ni][e]
                                         - ass * Bsm[cc + (uint32_t)e][hh];
                }
            }
        }

        #pragma unroll
        for (uint32_t hh = 0; !MMA && hh < (uint32_t)HALVES; hh++) {
            int iacc[MMQ_TM][MMQ_TN];
            #pragma unroll
            for (int i = 0; i < MMQ_TM; i++)
                #pragma unroll
                for (int j = 0; j < MMQ_TN; j++) iacc[i][j] = 0;

            #pragma unroll
            for (uint32_t kk = 0; kk < KPF; kk++) {
                int a[MMQ_TM], b[MMQ_TN];
                #pragma unroll
                for (int i = 0; i < MMQ_TM; i++) a[i] = As[tm * MMQ_TM + i][hh * KPF + kk];
                #pragma unroll
                for (int j = 0; j < MMQ_TN; j++) b[j] = Bs[tn * MMQ_TN + j][hh * KPF + kk];
                #pragma unroll
                for (int i = 0; i < MMQ_TM; i++)
                    #pragma unroll
                    for (int j = 0; j < MMQ_TN; j++)
                        iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }

            /* At HALVES == 1 the two stored halves hold the same d and the same
             * (ds, dm) -- only the activation's own sum differs, so the min term
             * takes both. Worst case for the int accumulator is Q6_K at 32 * 32 *
             * 127, four orders inside int32. */
            float asd[MMQ_TM], ass[MMQ_TM], bsd[MMQ_TN], bsm[MMQ_TN];
            #pragma unroll
            for (int i = 0; i < MMQ_TM; i++) {
                asd[i] = Asd[tm * MMQ_TM + i][hh];
                ass[i] = (HALVES == 1) ? (Ass[tm * MMQ_TM + i][0] + Ass[tm * MMQ_TM + i][1])
                                       : Ass[tm * MMQ_TM + i][hh];
            }
            #pragma unroll
            for (int j = 0; j < MMQ_TN; j++) {
                bsd[j] = Bsd[tn * MMQ_TN + j][hh];
                bsm[j] = Bsm[tn * MMQ_TN + j][hh];
            }
            #pragma unroll
            for (int i = 0; i < MMQ_TM; i++)
                #pragma unroll
                for (int j = 0; j < MMQ_TN; j++)
                    acc[i][j] += asd[i] * bsd[j] * (float)iacc[i][j] - ass[i] * bsm[j];
        }
        __syncthreads();
    }

    if (MMA) {
        #pragma unroll
        for (int mi = 0; mi < MMA_MI; mi++) {
            const uint32_t m = m0 + mbase + (uint32_t)mi * 8u + gid;
            if (m >= n_tok) continue;
            #pragma unroll
            for (int ni = 0; ni < MMA_NI; ni++) {
                const uint32_t c0 = n0 + nbase + (uint32_t)ni * 8u + tig * 2u;
                #pragma unroll
                for (int e = 0; e < 2; e++)
                    if (c0 + (uint32_t)e < n_out)
                        Y[(size_t)m * n_out + c0 + (uint32_t)e] = facc[mi][ni][e];
            }
        }
        return;
    }
    #pragma unroll
    for (int i = 0; i < MMQ_TM; i++) {
        const uint32_t m = m0 + tm * MMQ_TM + i;
        if (m >= n_tok) continue;
        #pragma unroll
        for (int j = 0; j < MMQ_TN; j++) {
            const uint32_t n = n0 + tn * MMQ_TN + j;
            if (n < n_out) Y[(size_t)m * n_out + n] = acc[i][j];
        }
    }
}

/* ---- MMQ, K-slab form (Q4_K) --------------------------------------------
 * Second MMQ kernel, written after reading llama.cpp's (ggml-cuda/mmq.cuh,
 * mmq-load-tiles.cuh, mmq-vec-dot.cuh). Three structural differences from the
 * kernel above, all of which the measurements had already pointed at and none of
 * which is a tuning constant:
 *
 * 1. A K STEP IS A WHOLE 256-ELEMENT QUANT BLOCK, not 32. The weight tile is
 *    staged once per block and the activation tile in two 128-element halves, so
 *    a 256-element slab costs FOUR __syncthreads instead of sixteen. ncu put the
 *    barrier stall at 3.5-4.3 per issued instruction, second only to memory.
 *
 * 2. THIRTY-TWO THREADS COOPERATE ON ONE WEIGHT ROW, reading 32 consecutive
 *    ints of its `qs`. The old kernel gave a row to 2 threads reading scattered
 *    16-byte pieces, which is why 78% of every fetched sector was wasted. I had
 *    tried to fix that by staging raw block bytes in shared and it came out 11.5%
 *    SLOWER -- the fix is the thread mapping, not another buffer.
 *
 * 3. SCALES ARE RESOLVED AT LOAD TIME into (d*sc, -dmin*mm) per 32-element
 *    sub-block, once per row per slab, with the negation folded in so the min
 *    term becomes an fma instead of a separate multiply and subtract. The old
 *    kernel re-ran get_scale_min_k4_d and both float multiplies for every
 *    16-element group of every K step.
 *
 * What is NOT done here yet, and is the reason llama.cpp is still ahead: their
 * inner loop hoists the weight FRAGMENTS and their scales into registers with
 * ldmatrix and then sweeps all J tokens against them, so a shared load is
 * amortised over the whole token tile. This kernel still reads both operands
 * from shared on every step. That is the next change, and it is why the
 * measurement of this one has to be taken before it and not after.
 *
 * Q4_K only for now: it is 42% of prefill GPU time, and proving the structure on
 * the dominant type before porting Q5_K/Q6_K keeps the blast radius small. The
 * other two keep the kernel above, which the gate still asserts. */
/* int8 tensor core, 16x8 output over k=32 — four times the work per instruction
 * of the m8n8k16 used elsewhere in this file, and the reason the build now names
 * its architectures (m16n8k32 is sm_80+; below that this falls back).
 *
 * Fragment layout (PTX ISA, .m16n8k32 with .s8), gid = lane>>2, tig = lane&3,
 * with k counted in INTS (4 packed int8 each), so a k=32 group is 8 ints:
 *   A 16x32 : a0,a1 = rows gid, gid+8 at int tig ; a2,a3 = same rows at int 4+tig
 *   B 32x8  : b0 = int tig of column gid ; b1 = int 4+tig of column gid
 *   C 16x8  : c0,c1 = row gid, cols tig*2, tig*2+1 ; c2,c3 = row gid+8, same cols
 * A is the WEIGHTS and B the TOKENS, so one C tile is 16 weight rows x 8 tokens. */
__device__ __forceinline__ void mma_m16n8k32_s8(int *c, const int *a, const int *b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    /* Turing: two m8n8k16 halves per k=32, accumulating into the same C. The
     * 16-row A tile splits into rows [0,8) and [8,16), which land in c0/c1 and
     * c2/c3 respectively — the same halves the m16n8k32 layout puts them in. */
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[2]), "r"(b[1]));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[2]), "+r"(c[3]) : "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[2]), "+r"(c[3]) : "r"(a[3]), "r"(b[1]));
#endif
}

/* One ldmatrix.x4 in place of four scalar shared loads. `p` must be the address
 * THIS lane contributes: ldmatrix gathers eight row pointers per 8x8 matrix from
 * lanes 0-7, 8-15, 16-23, 24-31, so the caller computes it per lane rather than
 * passing a tile base. 16-byte alignment is required (PTX). */
__device__ __forceinline__ void ldmatrix_x4(int *r, const int *p) {
#if __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "l"(__cvta_generic_to_shared(p)));
#else
    r[0] = p[0]; r[1] = p[1]; r[2] = p[2]; r[3] = p[3];
#endif
}

#define MMQ2_BN 64                    /* weight rows per block */
#define MMQ2_BM 128                   /* tokens per block */
#define MMQ2_TM 8
#define MMQ2_TN 4
#define MMQ2_THREADS ((MMQ2_BM / MMQ2_TM) * (MMQ2_BN / MMQ2_TN))   /* 256 */

template<int MMA>
__global__ __launch_bounds__(MMQ2_THREADS, 1)
void matmul_mmq_k256_kernel(const unsigned char *W,
                            uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                            const int8_t *Xq, const float *Xsd, const float *Xss,
                            float *Y) {
    /* +1 int of padding on the k axis: without it the tn stride is a multiple of
     * 32 words and every lane of a warp lands in one bank. */
    __shared__ int    x_qs[MMQ2_BN][64 + 1];   /* 256 quants per row, 4 per int */
    __shared__ float2 x_dm[MMQ2_BN][8];        /* (d*sc, -dmin*mm) per 32 */
    __shared__ int    y_qs[MMQ2_BM][32 + 1];   /* 128 quants per token (one half) */
    __shared__ float2 y_ds[MMQ2_BM][4];        /* (d8, d8*sum(xq)) per 32 */

    const uint32_t n0 = blockIdx.x * MMQ2_BN;
    const uint32_t m0 = blockIdx.y * MMQ2_BM;
    const uint32_t tid = threadIdx.x;
    const uint32_t tm = tid / (MMQ2_BN / MMQ2_TN);
    const uint32_t tn = tid % (MMQ2_BN / MMQ2_TN);
    /* MMA indexing: 8 warps as 4 (weight rows, 16 each) x 2 (tokens, 64 each). */
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const uint32_t wi = warp & 3u, wj = warp >> 2;
    const uint32_t gid = lane >> 2, tig = lane & 3u;
    const uint32_t wrow = wi * 16u, wtok = wj * 64u;
    const size_t rowbytes = (size_t)(n_in / QK_K) * 144;
    const uint32_t n_qblk = n_in / QK_K;
    const uint32_t nsub32 = n_in / 32u;        /* activation scale blocks per row */

    float acc[MMQ2_TM][MMQ2_TN];
    float facc[8][4];
    if (MMA) {
        #pragma unroll
        for (int nj = 0; nj < 8; nj++)
            #pragma unroll
            for (int l = 0; l < 4; l++) facc[nj][l] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < MMQ2_TM; i++)
            #pragma unroll
            for (int j = 0; j < MMQ2_TN; j++) acc[i][j] = 0.0f;
    }

    for (uint32_t qb = 0; qb < n_qblk; qb++) {
        __syncthreads();                       /* previous slab's reads are done */

        /* --- weight tile: 32 threads per row, 32 consecutive ints each ------ */
        for (uint32_t idx = tid; idx < MMQ2_BN * 32u; idx += MMQ2_THREADS) {
            const uint32_t n = idx >> 5, txi = idx & 31u;
            const uint32_t nn = n0 + n;
            const unsigned char *p = W + (size_t)nn * rowbytes + (size_t)qb * 144;
            const unsigned int qs0 = (nn < n_out)
                ? ((const unsigned int *)(p + 16))[txi] : 0u;
            /* txi indexes ql words; word w feeds sub-block 2*(w/8) through its
             * low nibbles and 2*(w/8)+1 through its high ones, at the same
             * position within each. Same de-interleave llama.cpp does on load. */
            const uint32_t base = 16u * (txi >> 3) + (txi & 7u);
            x_qs[n][base + 0] = (int)((qs0 >> 0) & 0x0F0F0F0Fu);
            x_qs[n][base + 8] = (int)((qs0 >> 4) & 0x0F0F0F0Fu);
        }
        for (uint32_t idx = tid; idx < MMQ2_BN * 8u; idx += MMQ2_THREADS) {
            const uint32_t n = idx >> 3, j = idx & 7u;
            const uint32_t nn = n0 + n;
            float d = 0.0f, dmin = 0.0f;
            unsigned char sc = 0, mm = 0;
            if (nn < n_out) {
                const unsigned char *p = W + (size_t)nn * rowbytes + (size_t)qb * 144;
                d    = f16_to_f32_d(*(const unsigned short *)p);
                dmin = f16_to_f32_d(*(const unsigned short *)(p + 2));
                get_scale_min_k4_d((int)j, p + 4, &sc, &mm);
            }
            x_dm[n][j] = make_float2(d * (float)sc, -dmin * (float)mm);
        }

        for (uint32_t half = 0; half < 2u; half++) {
            /* --- activation tile: 32 consecutive ints per token ------------- */
            for (uint32_t idx = tid; idx < MMQ2_BM * 32u; idx += MMQ2_THREADS) {
                const uint32_t m = idx >> 5, ki = idx & 31u;
                const uint32_t mm = m0 + m;
                y_qs[m][ki] = (mm < n_tok)
                    ? ((const int *)(Xq + (size_t)mm * n_in
                                        + (size_t)qb * QK_K + half * 128u))[ki]
                    : 0;
            }
            for (uint32_t idx = tid; idx < MMQ2_BM * 4u; idx += MMQ2_THREADS) {
                const uint32_t m = idx >> 2, sub = idx & 3u;
                const uint32_t mm = m0 + m;
                float sd = 0.0f, ss = 0.0f;
                if (mm < n_tok) {
                    /* the activation carries a scale per 16; a Q4_K sub-block is
                     * 32, so the two halves' sums add. */
                    const size_t b = (size_t)mm * nsub32 + qb * 8u + half * 4u + sub;
                    sd = Xsd[b * 2u];
                    ss = Xss[b * 2u] + Xss[b * 2u + 1u];
                }
                y_ds[m][sub] = make_float2(sd, ss);
            }
            __syncthreads();

            /* MMA: the weight fragment is read ONCE per 32-element group and
             * swept against all eight token tiles this warp owns. That reuse is
             * the whole point of the shape -- the dp4a loop below re-reads both
             * operands from shared on every step, which is what left 88% of the
             * issued instructions as loads and reductions. */
            #pragma unroll
            for (uint32_t sub = 0; MMA && sub < 4u; sub++) {
                const uint32_t k8 = half * 32u + sub * 8u;
                int a[4];
                a[0] = x_qs[wrow + gid     ][k8 + tig];
                a[1] = x_qs[wrow + gid + 8u][k8 + tig];
                a[2] = x_qs[wrow + gid     ][k8 + 4u + tig];
                a[3] = x_qs[wrow + gid + 8u][k8 + 4u + tig];
                const float2 xd0 = x_dm[wrow + gid     ][half * 4u + sub];
                const float2 xd1 = x_dm[wrow + gid + 8u][half * 4u + sub];

                #pragma unroll
                for (int nj = 0; nj < 8; nj++) {
                    const uint32_t tb = wtok + (uint32_t)nj * 8u;
                    int b[2];
                    b[0] = y_qs[tb + gid][sub * 8u + tig];
                    b[1] = y_qs[tb + gid][sub * 8u + 4u + tig];
                    int c[4] = { 0, 0, 0, 0 };
                    mma_m16n8k32_s8(c, a, b);
                    /* C column is tig*2 + {0,1}; C row is gid for c0/c1 and
                     * gid+8 for c2/c3 -- a different lane mapping from the one
                     * that SUPPLIED the fragments, which is normal for mma. */
                    const float2 yd0 = y_ds[tb + tig * 2u     ][sub];
                    const float2 yd1 = y_ds[tb + tig * 2u + 1u][sub];
                    facc[nj][0] += yd0.x * xd0.x * (float)c[0] + yd0.y * xd0.y;
                    facc[nj][1] += yd1.x * xd0.x * (float)c[1] + yd1.y * xd0.y;
                    facc[nj][2] += yd0.x * xd1.x * (float)c[2] + yd0.y * xd1.y;
                    facc[nj][3] += yd1.x * xd1.x * (float)c[3] + yd1.y * xd1.y;
                }
            }

            #pragma unroll
            for (uint32_t sub = 0; !MMA && sub < 4u; sub++) {
                int iacc[MMQ2_TM][MMQ2_TN];
                #pragma unroll
                for (int i = 0; i < MMQ2_TM; i++)
                    #pragma unroll
                    for (int j = 0; j < MMQ2_TN; j++) iacc[i][j] = 0;

                #pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t ki = sub * 8u + kk;
                    int a[MMQ2_TM], b[MMQ2_TN];
                    #pragma unroll
                    for (int i = 0; i < MMQ2_TM; i++) a[i] = y_qs[tm * MMQ2_TM + i][ki];
                    #pragma unroll
                    for (int j = 0; j < MMQ2_TN; j++)
                        b[j] = x_qs[tn * MMQ2_TN + j][half * 32u + ki];
                    #pragma unroll
                    for (int i = 0; i < MMQ2_TM; i++)
                        #pragma unroll
                        for (int j = 0; j < MMQ2_TN; j++)
                            iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
                }

                float2 yd[MMQ2_TM], xd[MMQ2_TN];
                #pragma unroll
                for (int i = 0; i < MMQ2_TM; i++) yd[i] = y_ds[tm * MMQ2_TM + i][sub];
                #pragma unroll
                for (int j = 0; j < MMQ2_TN; j++)
                    xd[j] = x_dm[tn * MMQ2_TN + j][half * 4u + sub];
                /* min term is an ADD because the negation was folded into x_dm. */
                #pragma unroll
                for (int i = 0; i < MMQ2_TM; i++)
                    #pragma unroll
                    for (int j = 0; j < MMQ2_TN; j++)
                        acc[i][j] += yd[i].x * xd[j].x * (float)iacc[i][j]
                                   + yd[i].y * xd[j].y;
            }
            __syncthreads();
        }
    }

    if (MMA) {
        #pragma unroll
        for (int nj = 0; nj < 8; nj++) {
            const uint32_t tb = wtok + (uint32_t)nj * 8u;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const uint32_t r = wrow + ((l < 2) ? gid : gid + 8u);
                const uint32_t t = tb + tig * 2u + (uint32_t)(l & 1);
                const uint32_t m = m0 + t, n = n0 + r;
                if (m < n_tok && n < n_out) Y[(size_t)m * n_out + n] = facc[nj][l];
            }
        }
        return;
    }
    #pragma unroll
    for (int i = 0; i < MMQ2_TM; i++) {
        const uint32_t m = m0 + tm * MMQ2_TM + i;
        if (m >= n_tok) continue;
        #pragma unroll
        for (int j = 0; j < MMQ2_TN; j++) {
            const uint32_t n = n0 + tn * MMQ2_TN + j;
            if (n < n_out) Y[(size_t)m * n_out + n] = acc[i][j];
        }
    }
}

/* Activation-side scratch. One arena for every weight, like g_stage above: the
 * matmuls are serial on the default stream and a per-handle copy would cost
 * hundreds of MB across a layer's weights. */
static int8_t *g_mmq_q  = NULL;
static float  *g_mmq_sd = NULL, *g_mmq_ss = NULL;
static size_t  g_mmq_cap = 0;                       /* in activation elements */

static int mmq_supported(uint32_t type, uint32_t n_in) {
    return (type == T_Q4_K || type == T_Q5_K || type == T_Q6_K) && (n_in % QK_K) == 0;
}

/* 0 on success, -1 to fall back. */
static int launch_matmul_mmq(const ds4x_cuda_wt *w, const float *d_x, float *d_y,
                             uint32_t nt) {
    if (!mmq_supported(w->type, w->n_in)) return -1;
    const size_t nelem = (size_t)nt * w->n_in;
    if (nelem > g_mmq_cap) {
        cudaFree(g_mmq_q); cudaFree(g_mmq_sd); cudaFree(g_mmq_ss);
        g_mmq_q = NULL; g_mmq_sd = NULL; g_mmq_ss = NULL; g_mmq_cap = 0;
        if (cudaMalloc(&g_mmq_q, nelem) != cudaSuccess ||
            cudaMalloc(&g_mmq_sd, (nelem / 16) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g_mmq_ss, (nelem / 16) * sizeof(float)) != cudaSuccess) {
            cudaFree(g_mmq_q); cudaFree(g_mmq_sd); cudaFree(g_mmq_ss);
            g_mmq_q = NULL; g_mmq_sd = NULL; g_mmq_ss = NULL;
            cudaGetLastError(); return -1;
        }
        g_mmq_cap = nelem;
    }
    /* n_in is a multiple of 256 here, so no 32-element block ever straddles two
     * token rows and the whole activation matrix quantizes as one flat array. */
    const uint32_t nq = (uint32_t)(nelem / 32);
    const uint32_t qth = 256;
    quantize_x_q8h_kernel<<<(nq * 32u + qth - 1u) / qth, qth>>>(
        d_x, g_mmq_q, g_mmq_sd, g_mmq_ss, nq);

    const dim3 grid((w->n_out + MMQ_BN - 1u) / MMQ_BN, (nt + MMQ_BM - 1u) / MMQ_BM);
    const unsigned char *dW = (const unsigned char *)w->d_data;
    #define MMQ_LAUNCH(T, M) matmul_mmq_kernel<T, M><<<grid, MMQ_THREADS>>>( \
        dW, w->n_in, w->n_out, nt, g_mmq_q, g_mmq_sd, g_mmq_ss, d_y)
    /* OPT-IN, not default: measured 2.7% slower on its own (1145.8 vs 1177.9
     * tok/s, 502-token prefill, same binary). Its 41.5 KB of shared forces one
     * block per SM where the kernel below gets two, and without the register-
     * blocked inner loop there is not enough ILP to pay for that. Kept, gated
     * and asserted, because it is the scaffolding the register-blocked loop
     * needs -- see the header above. IDLETOKEN_DS4X_K256=1 selects it. */
    {
        const char *nk = getenv("IDLETOKEN_DS4X_K256");
        if (getenv("IDLETOKEN_DS4X_K256_DEBUG")) {
            static int said = 0;
            if (!said) { said = 1;
                fprintf(stderr, "ds4x: k256 dispatch reached: type=%u nk=%s\n",
                        w->type, nk ? nk : "(unset)"); }
        }
        if (w->type == T_Q4_K && (nk && nk[0] == '1')) {
            const dim3 g2((w->n_out + MMQ2_BN - 1u) / MMQ2_BN,
                          (nt + MMQ2_BM - 1u) / MMQ2_BM);
            if (ds4x_mma_wanted())
                matmul_mmq_k256_kernel<1><<<g2, MMQ2_THREADS>>>(
                    dW, w->n_in, w->n_out, nt, g_mmq_q, g_mmq_sd, g_mmq_ss, d_y);
            else
                matmul_mmq_k256_kernel<0><<<g2, MMQ2_THREADS>>>(
                    dW, w->n_in, w->n_out, nt, g_mmq_q, g_mmq_sd, g_mmq_ss, d_y);
            return 0;
        }
    }
    if (ds4x_mma_wanted()) {
        if      (w->type == T_Q4_K) MMQ_LAUNCH(T_Q4_K, 1);
        else if (w->type == T_Q5_K) MMQ_LAUNCH(T_Q5_K, 1);
        else                        MMQ_LAUNCH(T_Q6_K, 1);
    } else {
        if      (w->type == T_Q4_K) MMQ_LAUNCH(T_Q4_K, 0);
        else if (w->type == T_Q5_K) MMQ_LAUNCH(T_Q5_K, 0);
        else                        MMQ_LAUNCH(T_Q6_K, 0);
    }
    #undef MMQ_LAUNCH
    return 0;
}

static int launch_matmul(const ds4x_cuda_wt *w, const float *d_x, float *d_y,
                         uint32_t nt) {
    /* MMQ first: for the K-quants it beats the cuBLAS route outright, because it
     * does not pay the expansion pass that route's tensor cores never recovered. */
    if (ds4x_mmq_wanted() && launch_matmul_mmq(w, d_x, d_y, nt) == 0) {
        cudaError_t me = cudaGetLastError();
        if (me != cudaSuccess) { set_err("matmul mmq launch", me); return -1; }
        return 0;
    }
    if (ds4x_tc_wanted() && launch_matmul_tc(w, d_x, d_y, nt) == 0) {
        cudaError_t te = cudaGetLastError();
        if (te != cudaSuccess) { set_err("matmul tensor-core launch", te); return -1; }
        return 0;
    }
    /* Every quant type goes through the tiled GEMM, including Q4_K which has a
     * hand-written batched-matvec of its own. Splitting by type here would mean
     * matmul(1) and matmul(13) could take different code paths for different
     * tensors, and `chunk==1-by-1` is a bit-identity assertion — it only means
     * something if the path is the same for every token count. */
    const dim3 grid((w->n_out + GEMM_BN - 1u) / GEMM_BN,
                    (nt + GEMM_BM - 1u) / GEMM_BM);
    matmul_tiled_kernel<<<grid, GEMM_THREADS>>>(
        (const unsigned char *)w->d_data, w->type, w->n_in, w->n_out,
        blk_count(w->type), (uint32_t)blk_bytes(w->type), nt, d_x, d_y);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { set_err("matmul launch", e); return -1; }
    return 0;
}

extern "C" int ds4x_cuda_matmul(const ds4x_cuda_wt *w, const float *X, float *Y,
                                uint32_t n_tokens) {
    if (!w || !X || !Y || n_tokens == 0) return -1;
    /* n_tokens == 1 deliberately goes through the SAME kernel rather than
     * shortcutting to ds4x_cuda_matvec: the two reduce differently, and routing
     * decode elsewhere would make "decode == prefill" untestable on the GPU
     * exactly where it matters. The masked-off tile lanes exit early, so a
     * 1-token call costs what the matvec did. */
    const double t0 = host_ms();
    if (!g_mev0) { cudaEventCreate(&g_mev0); cudaEventCreate(&g_mev1); }

    for (uint32_t base = 0; base < n_tokens; base += MM_BLOCK) {
        const uint32_t nt = (n_tokens - base) < MM_BLOCK ? (n_tokens - base) : MM_BLOCK;
        const size_t nx = (size_t)nt * w->n_in, ny = (size_t)nt * w->n_out;
        if (stage_reserve(nx, ny) != 0) return -1;
        memcpy(g_stage.h_x, X + (size_t)base * w->n_in, nx * sizeof(float));
        cudaError_t e;
        if ((e = cudaMemcpyAsync(g_stage.d_x, g_stage.h_x, nx * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(matmul X)", e); return -1;
        }
        cudaEventRecord(g_mev0);
        if (launch_matmul(w, g_stage.d_x, g_stage.d_y, nt) != 0) return -1;
        cudaEventRecord(g_mev1);
        if ((e = cudaMemcpyAsync(g_stage.h_y, g_stage.d_y, ny * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(matmul Y)", e); return -1;
        }
        if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("matmul sync", e); return -1; }
        memcpy(Y + (size_t)base * w->n_out, g_stage.h_y, ny * sizeof(float));
        float kms = 0.0f;
        if (cudaEventElapsedTime(&kms, g_mev0, g_mev1) == cudaSuccess) g_mm_ms_kernel += kms;
        g_mm_calls++;
        g_mm_rows += nt;
    }
    g_mm_ms_total += host_ms() - t0;
    return 0;
}

/* ---- fused SwiGLU FFN ----------------------------------------------------
 * y = Wd . (silu(Wg.x) (*) (Wu.x)), the whole thing on the device.
 *
 * Unfused this is three ds4x_cuda_matmul calls with the elementwise silu*mul
 * on the host in between: 3 H2D + 3 D2H per FFN per token. Fused it is 1 + 1.
 * A dense FFN is entirely token-independent (no cache, no position), so there
 * is nothing to prove here beyond the matmul contract and the elementwise op.
 *
 * The CPU swiglu() in ds4x_forward.c stays the numeric reference. */
static struct { float *d_g, *d_u; size_t n; } g_ffn = { NULL, NULL, 0 };
static double g_ffn_ms_kernel = 0.0, g_ffn_ms_total = 0.0;
static unsigned long long g_ffn_calls = 0, g_ffn_rows = 0;

static int ffn_reserve(size_t n) {
    if (n <= g_ffn.n) return 0;
    cudaError_t e;
    cudaFree(g_ffn.d_g); cudaFree(g_ffn.d_u);
    g_ffn.d_g = NULL; g_ffn.d_u = NULL; g_ffn.n = 0;
    if ((e = cudaMalloc(&g_ffn.d_g, n * sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&g_ffn.d_u, n * sizeof(float))) != cudaSuccess) {
        set_err("cudaMalloc(ffn scratch)", e); return -1;
    }
    g_ffn.n = n;
    return 0;
}

/* g[i] = silu(g[i]) * u[i]. Mirrors the CPU loop in swiglu() exactly; expf is
 * left as expf (not __expf) so the only fast-math divergence is the one the
 * whole translation unit already has from --use_fast_math. */
__global__ void silu_mul_kernel(float *g, const float *u, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    g[i] = (v / (1.0f + expf(-v))) * u[i];
}

extern "C" int ds4x_cuda_swiglu(const ds4x_cuda_wt *Wg, const ds4x_cuda_wt *Wu,
                                const ds4x_cuda_wt *Wd, const float *X, float *Y,
                                uint32_t n_tokens) {
    if (!Wg || !Wu || !Wd || !X || !Y || n_tokens == 0) return -1;
    /* Shape agreement is the caller's contract, but a mismatch here would read
     * out of bounds on the device, so check rather than trust: Wg and Wu must
     * share both dimensions, and Wd must consume what they produce. */
    if (Wg->n_in != Wu->n_in || Wg->n_out != Wu->n_out || Wd->n_in != Wg->n_out) {
        snprintf(g_err, sizeof(g_err),
                 "swiglu: shape mismatch g=[%ux%u] u=[%ux%u] d=[%ux%u]",
                 Wg->n_out, Wg->n_in, Wu->n_out, Wu->n_in, Wd->n_out, Wd->n_in);
        return -1;
    }
    const double t0 = host_ms();
    if (!g_mev0) { cudaEventCreate(&g_mev0); cudaEventCreate(&g_mev1); }
    const uint32_t ff = Wg->n_out, n_embd_in = Wg->n_in, n_embd_out = Wd->n_out;

    for (uint32_t base = 0; base < n_tokens; base += MM_BLOCK) {
        const uint32_t nt = (n_tokens - base) < MM_BLOCK ? (n_tokens - base) : MM_BLOCK;
        const size_t nx = (size_t)nt * n_embd_in, ny = (size_t)nt * n_embd_out;
        const size_t nff = (size_t)nt * ff;
        if (stage_reserve(nx, ny) != 0 || ffn_reserve(nff) != 0) return -1;
        memcpy(g_stage.h_x, X + (size_t)base * n_embd_in, nx * sizeof(float));
        cudaError_t e;
        if ((e = cudaMemcpyAsync(g_stage.d_x, g_stage.h_x, nx * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(swiglu X)", e); return -1;
        }
        cudaEventRecord(g_mev0);
        /* The three projections and the elementwise op are all on the default
         * stream, so they serialize in issue order with no explicit sync.
         *
         * FFN_PROJ picks the kernel the UNFUSED path would have used at this
         * chunk size: swiglu() (decode, 1 token) went through matvec_q, and
         * swiglu_chunk() (prefill) through matmul_q. Keeping that split matters
         * twice over — it reproduces the old reduction order exactly instead of
         * inventing a new one, and the matmul kernel's token tiling would leave
         * all but one lane masked off on a 1-token launch (measured at TOK_TILE=8: +33 ms kernel
         * per 32 tokens, half the round-trip saving handed straight back).
         *
         * Deliberately NOT folded into launch_matmul: ds4x_cuda_matmul routes
         * n_tokens==1 through the matmul kernel on purpose, so that decode and
         * prefill stay comparable through that API. */
#define FFN_PROJ(W, DX, DY) \
    (nt == 1 ? launch_matvec((W), (DX), (DY)) : launch_matmul((W), (DX), (DY), nt))
        if (FFN_PROJ(Wg, g_stage.d_x, g_ffn.d_g) != 0) return -1;
        if (FFN_PROJ(Wu, g_stage.d_x, g_ffn.d_u) != 0) return -1;
        {
            const uint32_t th = 256;
            const size_t blocks = (nff + th - 1) / th;
            silu_mul_kernel<<<(uint32_t)blocks, th>>>(g_ffn.d_g, g_ffn.d_u, nff);
            if ((e = cudaGetLastError()) != cudaSuccess) { set_err("silu_mul launch", e); return -1; }
        }
        if (FFN_PROJ(Wd, g_ffn.d_g, g_stage.d_y) != 0) return -1;
#undef FFN_PROJ
        cudaEventRecord(g_mev1);
        if ((e = cudaMemcpyAsync(g_stage.h_y, g_stage.d_y, ny * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(swiglu Y)", e); return -1;
        }
        if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("swiglu sync", e); return -1; }
        memcpy(Y + (size_t)base * n_embd_out, g_stage.h_y, ny * sizeof(float));
        float kms = 0.0f;
        if (cudaEventElapsedTime(&kms, g_mev0, g_mev1) == cudaSuccess) g_ffn_ms_kernel += kms;
        g_ffn_calls++;
        g_ffn_rows += nt;
    }
    g_ffn_ms_total += host_ms() - t0;
    return 0;
}

/* h[i] += o[i] — the residual add, on device so the value never has to come
 * back just to be added to. */
__global__ void residual_add_kernel(float *h, const float *o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += o[i];
}

/* y[t] = (h[t] / rms(h[t])) * w   — one block per token row.
 * Mirrors rmsnorm() in ds4x_forward.c including the eps placement (inside the
 * sqrt, on the mean), which is the detail that silently changes results if
 * copied wrong. */
__global__ void rmsnorm_kernel(const float *h, const float *w, float *y,
                               uint32_t n, float eps) {
    const float *hr = h + (size_t)blockIdx.x * n;
    float *yr = y + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += hr[i] * hr[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    extern __shared__ float smn[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) smn[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (uint32_t k = 0; k < nwarps; k++) s += smn[k];
        smn[0] = 1.0f / sqrtf(s / (float)n + eps);
    }
    __syncthreads();
    const float inv = smn[0];
    /* Multiply order is x*w*inv, matching rmsnorm() in ds4x_forward.c exactly.
     * fp32 multiplication is not associative, so x*inv*w is a DIFFERENT number;
     * writing it the other way round is a divergence that no compiler warns
     * about and that only shows up as a flipped token much later. */
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = hr[i] * w[i] * inv;
}

extern "C" void ds4x_cuda_ffn_stats(double *ms_kernel, double *ms_total,
                                    uint64_t *calls, uint64_t *rows) {
    if (ms_kernel) *ms_kernel = g_ffn_ms_kernel;
    if (ms_total)  *ms_total  = g_ffn_ms_total;
    if (calls)     *calls     = (uint64_t)g_ffn_calls;
    if (rows)      *rows      = (uint64_t)g_ffn_rows;
}

/* ---- Gated-DeltaNet pass 2 on the device ---------------------------------
 * conv + silu, the two gates, and the per-head L2 norm — everything between
 * the input projections and the recurrence. All of it is per-token
 * independent, which is exactly why it used to sit on the host: it was cheap
 * arithmetic. It is not cheap POSITIONALLY, because it splits the layer into
 * two device calls with a host round trip between them.
 *
 * Three small kernels rather than one: conv output is conv_ch floats per token
 * (6144 for Qwen3.5-0.8B = 24 KB) which does not fit comfortably in shared
 * memory, so it goes through global and the L2 norm re-reads it. Three launches
 * inside ONE call still cost one round trip, which is the thing being paid for.
 *
 * Each mirrors the CPU loop in gdn_attn_cpu operation for operation; that loop
 * stays the reference. */

/* conv[c] = silu(bias[c] + Σ_j w[c][j] · src(t-(K-1-j), c)), where positions
 * before the chunk come from the carried window. */
__global__ void gdn_conv_kernel(const float *raw, const float *win,
                                const float *cw, const float *cb,
                                float *cnv, uint32_t n_tokens,
                                uint32_t conv_ch, uint32_t K) {
    const uint32_t t = blockIdx.y;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_ch || t >= n_tokens) return;
    float acc = cb ? cb[c] : 0.0f;
    for (uint32_t j = 0; j < K; j++) {
        const int rel = (int)t - (int)(K - 1 - j);
        const float src = (rel >= 0)
            ? raw[(size_t)rel * conv_ch + c]
            : win[(size_t)(K - 1 + rel) * conv_ch + c];
        acc += cw[(size_t)c * K + j] * src;
    }
    cnv[(size_t)t * conv_ch + c] = acc / (1.0f + expf(-acc));   /* silu */
}

/* Per-k-head L2 norm on q and k, in place. One block per (token, head-slot);
 * head-slot < kh is a q head, >= kh is the matching k head. eps 1e-6 inside the
 * sqrt, matching the CPU. */
__global__ void gdn_l2norm_kernel(float *cnv, uint32_t conv_ch,
                                  uint32_t kh, uint32_t kdim) {
    const uint32_t t = blockIdx.y;
    const uint32_t slot = blockIdx.x;
    const uint32_t is_k = slot >= kh;
    const uint32_t h = is_k ? (slot - kh) : slot;
    float *v = cnv + (size_t)t * conv_ch + (size_t)(is_k ? kh * kdim : 0)
             + (size_t)h * kdim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < kdim; i += blockDim.x) ss += v[i] * v[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    extern __shared__ float sml[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) sml[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (uint32_t k = 0; k < nwarps; k++) s += sml[k];
        sml[0] = 1.0f / sqrtf(s + 1e-6f);
    }
    __syncthreads();
    const float inv = sml[0];
    for (uint32_t i = threadIdx.x; i < kdim; i += blockDim.x) v[i] *= inv;
}

/* beta = sigmoid(b); decay = exp(A_log · softplus(a + dt_bias)).
 * softplus is the stable form log1p(exp(-|x|)) + max(x,0), same as the CPU.
 * A_log is used DIRECTLY — it already stores the negative coefficient; there is
 * no exp() on it (the comment in gdn_attn_cpu records why). */
__global__ void gdn_gates_kernel(const float *bb, const float *ab,
                                 const float *A_log, const float *dt_bias,
                                 float *beta, float *decay,
                                 uint32_t n_tokens, uint32_t vh) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n_tokens * vh) return;
    const uint32_t h = (uint32_t)(i % vh);
    const float b = bb[i], a = ab[i];
    beta[i] = 1.0f / (1.0f + expf(-b));
    const float xa = a + dt_bias[h];
    const float sp = log1pf(expf(-fabsf(xa))) + (xa > 0.0f ? xa : 0.0f);
    decay[i] = expf(A_log[h] * sp);
}

/* Gated RMSNorm, PER v-HEAD over v_dim: core = core·ssm_norm·inv·silu(z).
 *
 * ssm_norm is only v_dim long and shared across heads — normalising over the
 * whole vd would be wrong (design doc §4b). Multiply order matches the CPU
 * exactly (c * ssm * inv * silu(z)); fp32 multiplication is not associative,
 * so a different order is a different number. One block per (token, v-head). */
__global__ void gdn_gatednorm_kernel(float *core, const float *z,
                                     const float *ssm_norm,
                                     uint32_t vh, uint32_t vdim, float eps) {
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    float *c = core + (size_t)t * vh * vdim + (size_t)h * vdim;
    const float *zh = z + (size_t)t * vh * vdim + (size_t)h * vdim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < vdim; i += blockDim.x) ss += c[i] * c[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    extern __shared__ float smg[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) / 32u;
    if (lane == 0) smg[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t k = 0; k < nwarps; k++) v += smg[k];
        smg[0] = 1.0f / sqrtf(v / (float)vdim + eps);
    }
    __syncthreads();
    const float inv = smg[0];
    for (uint32_t i = threadIdx.x; i < vdim; i += blockDim.x) {
        const float zv = zh[i];
        c[i] = c[i] * ssm_norm[i] * inv * (zv / (1.0f + expf(-zv)));
    }
}

/* ---- fused attention-output + FFN tail -----------------------------------
 * The whole back half of a layer, in one call:
 *
 *     h = x + Wo·attn_in            (attention output projection + residual)
 *     n = rmsnorm(h) * ffn_norm_w
 *     x = h + Wd·(silu(Wg·n) ⊙ (Wu·n))
 *
 * Unfused this is TWO device calls (the output projection, then the fused FFN)
 * with the residual and the norm on the host in between — so the hidden state
 * makes a full round trip purely to have two elementwise operations done to it.
 * Fused it is one upload of (attn_in, x) and one download of x.
 *
 * That middle round trip is the whole `matmuls` bucket in a dense model: 2856
 * calls per 118 tokens, 350 ms, 55% of it not kernel time. Folding it in here
 * makes the bucket disappear rather than shrink.
 *
 * Works for both attention kinds because both end the same way — GQA passes
 * concat(heads) with Wo = attn_out, a linear layer passes the recurrence output
 * `core` with Wo = out_proj. Only the input width differs, and that is Wo->n_in.
 *
 * Dense FFN only: a MoE layer routes each token to different experts, so there
 * is no single (Wg,Wu,Wd) to fuse with. Those keep the unfused path. */
static struct { float *d_attn, *d_h, *d_n, *d_w, *h_io; size_t n_attn, n_h, n_w, n_io; } g_tail =
    { NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0 };
static double g_tail_ms_kernel = 0.0, g_tail_ms_total = 0.0;
static unsigned long long g_tail_calls = 0, g_tail_rows = 0;

static int tail_reserve(size_t n_attn, size_t n_h, size_t n_w, size_t n_io) {
    cudaError_t e;
    if (n_attn > g_tail.n_attn) {
        cudaFree(g_tail.d_attn); g_tail.d_attn = NULL; g_tail.n_attn = 0;
        if ((e = cudaMalloc(&g_tail.d_attn, n_attn * sizeof(float))) != cudaSuccess) {
            set_err("cudaMalloc(tail attn)", e); return -1; }
        g_tail.n_attn = n_attn;
    }
    if (n_h > g_tail.n_h) {
        cudaFree(g_tail.d_h); cudaFree(g_tail.d_n);
        g_tail.d_h = NULL; g_tail.d_n = NULL; g_tail.n_h = 0;
        if ((e = cudaMalloc(&g_tail.d_h, n_h * sizeof(float))) != cudaSuccess ||
            (e = cudaMalloc(&g_tail.d_n, n_h * sizeof(float))) != cudaSuccess) {
            set_err("cudaMalloc(tail h)", e); return -1; }
        g_tail.n_h = n_h;
    }
    if (n_w > g_tail.n_w) {
        cudaFree(g_tail.d_w); g_tail.d_w = NULL; g_tail.n_w = 0;
        if ((e = cudaMalloc(&g_tail.d_w, n_w * sizeof(float))) != cudaSuccess) {
            set_err("cudaMalloc(tail w)", e); return -1; }
        g_tail.n_w = n_w;
    }
    if (n_io > g_tail.n_io) {
        cudaFreeHost(g_tail.h_io); g_tail.h_io = NULL; g_tail.n_io = 0;
        if ((e = cudaHostAlloc(&g_tail.h_io, n_io * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
            set_err("cudaHostAlloc(tail io)", e); return -1; }
        g_tail.n_io = n_io;
    }
    return 0;
}

/* The tail's DEVICE-SIDE sequence, every buffer already resident:
 *   d_h  in: the residual   out: the finished layer output
 *   d_n  scratch, n_tokens x n_embd
 *   g_ffn.d_g / d_u must already be reserved for n_tokens x ff.
 * No copies, no sync, no timing — the caller owns all three.
 *
 * Extracted so the linear-layer path can run the tail on a `core` that is
 * already in VRAM rather than carrying a second copy of these eight launches. */
static int launch_attn_ffn_tail(const ds4x_cuda_wt *Wo, const ds4x_cuda_wt *Wg,
                                const ds4x_cuda_wt *Wu, const ds4x_cuda_wt *Wd,
                                const float *d_attn, float *d_h, float *d_n,
                                const float *d_normw, uint32_t nt,
                                uint32_t n_embd, uint32_t ff, float eps) {
    const size_t nh = (size_t)nt * n_embd, nff = (size_t)nt * ff;
    cudaError_t e;
        /* h = x + Wo·attn_in : project into d_n as scratch, then add into d_h */
        if ((nt == 1 ? launch_matvec(Wo, d_attn, d_n)
                     : launch_matmul(Wo, d_attn, d_n, nt)) != 0) return -1;
        {
            const uint32_t th = 256;
            residual_add_kernel<<<(uint32_t)((nh + th - 1) / th), th>>>(d_h, d_n, nh);
            if ((e = cudaGetLastError()) != cudaSuccess) { set_err("residual launch", e); return -1; }
        }
        /* n = rmsnorm(h) * ffn_norm_w */
        {
            const uint32_t th = 256;
            const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
            rmsnorm_kernel<<<nt, th, sh>>>(d_h, d_normw, d_n, n_embd, eps);
            if ((e = cudaGetLastError()) != cudaSuccess) { set_err("rmsnorm launch", e); return -1; }
        }
        /* swiglu(n) into d_attn (reused as scratch: it is >= nh only when
         * ain >= n_embd, so use the ffn arena's d_g result path instead) */
        if ((nt == 1 ? launch_matvec(Wg, d_n, g_ffn.d_g)
                     : launch_matmul(Wg, d_n, g_ffn.d_g, nt)) != 0) return -1;
        if ((nt == 1 ? launch_matvec(Wu, d_n, g_ffn.d_u)
                     : launch_matmul(Wu, d_n, g_ffn.d_u, nt)) != 0) return -1;
        {
            const uint32_t th = 256;
            silu_mul_kernel<<<(uint32_t)((nff + th - 1) / th), th>>>(g_ffn.d_g, g_ffn.d_u, nff);
            if ((e = cudaGetLastError()) != cudaSuccess) { set_err("silu_mul launch", e); return -1; }
        }
        /* x = h + Wd·g : project into d_n, add into d_h, download d_h */
        if ((nt == 1 ? launch_matvec(Wd, g_ffn.d_g, d_n)
                     : launch_matmul(Wd, g_ffn.d_g, d_n, nt)) != 0) return -1;
        {
            const uint32_t th = 256;
            residual_add_kernel<<<(uint32_t)((nh + th - 1) / th), th>>>(d_h, d_n, nh);
            if ((e = cudaGetLastError()) != cudaSuccess) { set_err("residual2 launch", e); return -1; }
        }
    return 0;
}

extern "C" int ds4x_cuda_attn_ffn_tail(const ds4x_cuda_wt *Wo,
                                       const ds4x_cuda_wt *Wg, const ds4x_cuda_wt *Wu,
                                       const ds4x_cuda_wt *Wd,
                                       const float *attn_in, const float *ffn_norm_w,
                                       float *x, uint32_t n_tokens, float eps) {
    if (!Wo || !Wg || !Wu || !Wd || !attn_in || !ffn_norm_w || !x || n_tokens == 0)
        return -1;
    const uint32_t n_embd = Wo->n_out, ff = Wg->n_out, ain = Wo->n_in;
    if (Wg->n_in != n_embd || Wu->n_in != n_embd || Wu->n_out != ff ||
        Wd->n_in != ff || Wd->n_out != n_embd) {
        snprintf(g_err, sizeof(g_err),
                 "attn_ffn_tail: shape mismatch o=[%ux%u] g=[%ux%u] u=[%ux%u] d=[%ux%u]",
                 Wo->n_out, Wo->n_in, Wg->n_out, Wg->n_in,
                 Wu->n_out, Wu->n_in, Wd->n_out, Wd->n_in);
        return -1;
    }
    const double t0 = host_ms();
    if (!g_mev0) { cudaEventCreate(&g_mev0); cudaEventCreate(&g_mev1); }

    for (uint32_t base = 0; base < n_tokens; base += MM_BLOCK) {
        const uint32_t nt = (n_tokens - base) < MM_BLOCK ? (n_tokens - base) : MM_BLOCK;
        const size_t na = (size_t)nt * ain, nh = (size_t)nt * n_embd;
        const size_t nff = (size_t)nt * ff;
        /* One pinned buffer holding BOTH inputs back to back, so the two uploads
         * are issued without a synchronize between them. Staging them through
         * the same region one after the other would need a sync to know the
         * first copy had drained before overwriting it — i.e. it would put back
         * exactly the round trip this function exists to remove. */
        if (tail_reserve(na, nh, n_embd, na + nh) != 0) return -1;
        if (ffn_reserve(nff) != 0) return -1;
        cudaError_t e;
        memcpy(g_tail.h_io,      attn_in + (size_t)base * ain,    na * sizeof(float));
        memcpy(g_tail.h_io + na, x       + (size_t)base * n_embd, nh * sizeof(float));
        if ((e = cudaMemcpyAsync(g_tail.d_attn, g_tail.h_io, na * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(tail attn_in)", e); return -1; }
        if ((e = cudaMemcpyAsync(g_tail.d_h, g_tail.h_io + na, nh * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(tail x)", e); return -1; }
        if (base == 0) {
            if ((e = cudaMemcpyAsync(g_tail.d_w, ffn_norm_w, n_embd * sizeof(float),
                                     cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
                set_err("cudaMemcpyAsync(tail norm_w)", e); return -1; }
        }
        cudaEventRecord(g_mev0);
        if (launch_attn_ffn_tail(Wo, Wg, Wu, Wd, g_tail.d_attn, g_tail.d_h,
                                 g_tail.d_n, g_tail.d_w, nt, n_embd, ff, eps) != 0)
            return -1;
        cudaEventRecord(g_mev1);
        if ((e = cudaMemcpyAsync(g_tail.h_io, g_tail.d_h, nh * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(tail out)", e); return -1; }
        if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("tail sync", e); return -1; }
        memcpy(x + (size_t)base * n_embd, g_tail.h_io, nh * sizeof(float));
        float kms = 0.0f;
        if (cudaEventElapsedTime(&kms, g_mev0, g_mev1) == cudaSuccess) g_tail_ms_kernel += kms;
        g_tail_calls++;
        g_tail_rows += nt;
    }
    g_tail_ms_total += host_ms() - t0;
    return 0;
}

extern "C" void ds4x_cuda_tail_stats(double *ms_kernel, double *ms_total,
                                     uint64_t *calls, uint64_t *rows) {
    if (ms_kernel) *ms_kernel = g_tail_ms_kernel;
    if (ms_total)  *ms_total  = g_tail_ms_total;
    if (calls)     *calls     = (uint64_t)g_tail_calls;
    if (rows)      *rows      = (uint64_t)g_tail_rows;
}

/* ---- fused projection fan-out --------------------------------------------
 * N projections that all read the SAME input, in one call.
 *
 * Two places in the forward have this shape and between them they are most of
 * the remaining calls:
 *   - Gated DeltaNet: qkv / z / b / a all read `normed`   (4 projections)
 *   - GQA attention:  q / k / v      all read `nrm_all`   (3 projections)
 *
 * Unfused that is N ds4x_cuda_matmul calls: the identical input uploaded N
 * times, N downloads, N synchronizes. Fused it is one upload, N kernels, ONE
 * download, one synchronize — the outputs are written into slices of a single
 * contiguous device arena and come back in a single transfer, then get
 * scattered to the caller's N destinations (which need not be contiguous with
 * each other: GQA writes k and v straight into the KV cache).
 *
 * Deliberately ONE function rather than one per call site. The first cut had a
 * GDN-specific version and the GQA one would have been a 60-line near-copy —
 * and this repository's standing failure mode is two copies of the same logic
 * drifting apart (see the header of scripts/testbed-lib.sh for the last three
 * times).
 *
 * Outputs stay host-side for now: the conv/gates/recurrence and the attention
 * reduction that consume them are still CPU code. When those move over, these
 * downloads disappear rather than being rewritten. */
static struct { float *d, *h; size_t n; } g_proj = { NULL, NULL, 0 };
static double g_proj_ms_kernel = 0.0, g_proj_ms_total = 0.0;
static unsigned long long g_proj_calls = 0, g_proj_rows = 0;

static int proj_reserve(size_t n) {
    if (n <= g_proj.n) return 0;
    cudaError_t e;
    cudaFree(g_proj.d); cudaFreeHost(g_proj.h);
    g_proj.d = NULL; g_proj.h = NULL; g_proj.n = 0;
    if ((e = cudaMalloc(&g_proj.d, n * sizeof(float))) != cudaSuccess ||
        (e = cudaHostAlloc(&g_proj.h, n * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
        set_err("cudaMalloc(gdn proj arena)", e); return -1;
    }
    g_proj.n = n;
    return 0;
}

extern "C" int ds4x_cuda_proj_fanout(const ds4x_cuda_wt *const *W,
                                     float *const *Y, uint32_t n_proj,
                                     const float *X, uint32_t n_tokens) {
    if (!W || !Y || !X || n_tokens == 0 || n_proj == 0 ||
        n_proj > DS4X_PROJ_FANOUT_MAX) return -1;
    const uint32_t n_in = W[0] ? W[0]->n_in : 0;
    for (uint32_t i = 0; i < n_proj; i++) {
        if (!W[i] || !Y[i]) return -1;
        if (W[i]->n_in != n_in) {
            /* The whole premise is that they share one upload; a disagreeing
             * n_in means the caller grouped the wrong weights together. */
            snprintf(g_err, sizeof(g_err),
                     "proj_fanout: projection %u wants n_in=%u, group is %u",
                     i, W[i]->n_in, n_in);
            return -1;
        }
    }
    const double t0 = host_ms();
    if (!g_mev0) { cudaEventCreate(&g_mev0); cudaEventCreate(&g_mev1); }

    for (uint32_t base = 0; base < n_tokens; base += MM_BLOCK) {
        const uint32_t nt = (n_tokens - base) < MM_BLOCK ? (n_tokens - base) : MM_BLOCK;
        const size_t nx = (size_t)nt * n_in;
        size_t off[DS4X_PROJ_FANOUT_MAX], total = 0;
        for (uint32_t i = 0; i < n_proj; i++) {
            off[i] = total;
            total += (size_t)nt * W[i]->n_out;
        }
        if (stage_reserve(nx, 1) != 0 || proj_reserve(total) != 0) return -1;
        memcpy(g_stage.h_x, X + (size_t)base * n_in, nx * sizeof(float));
        cudaError_t e;
        if ((e = cudaMemcpyAsync(g_stage.d_x, g_stage.h_x, nx * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(proj_fanout X)", e); return -1;
        }
        cudaEventRecord(g_mev0);
        for (uint32_t i = 0; i < n_proj; i++) {
            /* Same kernel-choice rule as the fused FFN: decode (1 token) uses
             * the matvec kernel the unfused path used, so neither the reduction
             * order nor the tile occupancy changes. */
            const int rc = (nt == 1)
                ? launch_matvec(W[i], g_stage.d_x, g_proj.d + off[i])
                : launch_matmul(W[i], g_stage.d_x, g_proj.d + off[i], nt);
            if (rc != 0) return -1;
        }
        cudaEventRecord(g_mev1);
        /* ONE download for every output — that is the point of packing them
         * into a single arena. */
        if ((e = cudaMemcpyAsync(g_proj.h, g_proj.d, total * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(proj_fanout out)", e); return -1;
        }
        if ((e = cudaStreamSynchronize(0)) != cudaSuccess) {
            set_err("proj_fanout sync", e); return -1;
        }
        for (uint32_t i = 0; i < n_proj; i++)
            memcpy(Y[i] + (size_t)base * W[i]->n_out, g_proj.h + off[i],
                   (size_t)nt * W[i]->n_out * sizeof(float));
        float kms = 0.0f;
        if (cudaEventElapsedTime(&kms, g_mev0, g_mev1) == cudaSuccess) g_proj_ms_kernel += kms;
        g_proj_calls++;
        g_proj_rows += nt;
    }
    g_proj_ms_total += host_ms() - t0;
    return 0;
}

extern "C" void ds4x_cuda_proj_stats(double *ms_kernel, double *ms_total,
                                     uint64_t *calls, uint64_t *rows) {
    if (ms_kernel) *ms_kernel = g_proj_ms_kernel;
    if (ms_total)  *ms_total  = g_proj_ms_total;
    if (calls)     *calls     = (uint64_t)g_proj_calls;
    if (rows)      *rows      = (uint64_t)g_proj_rows;
}

extern "C" void ds4x_cuda_matmul_stats(double *ms_kernel, double *ms_total,
                                       uint64_t *calls, uint64_t *rows) {
    if (ms_kernel) *ms_kernel = g_mm_ms_kernel;
    if (ms_total)  *ms_total  = g_mm_ms_total;
    if (calls)     *calls     = (uint64_t)g_mm_calls;
    if (rows)      *rows      = (uint64_t)g_mm_rows;
}

/* ---- GQA attention -------------------------------------------------------
 * One WARP per (token, head). Lane `l` owns dims {l, l+32, l+64, ...} of the
 * head, which is the layout that makes every K and V read coalesced: for a
 * fixed dim-group the 32 lanes read 32 consecutive floats, one 128 B
 * transaction. The obvious alternative (lane l owns a contiguous run of dims)
 * makes each lane read its own 32 B and wastes 7/8 of every sector.
 *
 * A warp, not a block, because the score for one (token, head, position) is a
 * reduction over hdim and that reduction happens once PER POSITION -- 512 times
 * for a 512-token prefill. At warp width it is five shuffles with no
 * __syncthreads; at block width it would be a shared-memory round trip and a
 * barrier per position.
 *
 * Softmax runs in the online form: keep a running max m and denominator l, and
 * rescale the accumulator by exp(m_old - m_new) whenever the max moves. The
 * host path instead scans for the max, then exponentiates, then weights -- three
 * passes and an O(n_kv) scratch array per query. Same value either way, but the
 * host's order cannot be reproduced bit for bit, so the gate asserts a
 * tolerance rather than equality.
 *
 * m starts at -1e30f rather than -INFINITY on purpose: the first position
 * computes exp(m - max(m, s)), and with -INFINITY that is inf - inf = NaN the
 * moment a score is itself -inf. The host path uses the same -1e30f sentinel. */
#define ATTN_WARPS   4                      /* warps per block */
#define ATTN_MAX_ND 16                      /* hdim <= 16*32 = 512 */

__global__ void attn_gqa_kernel(const float *Q, const float *K, const float *V,
                                float *O, uint32_t n_tok, uint32_t pos0,
                                uint32_t n_head, uint32_t grp, uint32_t hdim,
                                uint32_t kv_dim, uint32_t qstride, float scale,
                                int gated) {
    const uint32_t w = blockIdx.x * ATTN_WARPS + (threadIdx.x >> 5);
    if (w >= n_tok * n_head) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t t = w / n_head, hd = w % n_head;
    const uint32_t kvh = hd / grp;
    const uint32_t nd = hdim >> 5;
    const uint32_t pos = pos0 + t;

    const float *qp = Q + (size_t)t * n_head * qstride + (size_t)hd * qstride;
    float qv[ATTN_MAX_ND], acc[ATTN_MAX_ND];
    for (uint32_t j = 0; j < nd; j++) { qv[j] = qp[j * 32u + lane]; acc[j] = 0.0f; }

    float m = -1e30f, l = 0.0f;
    for (uint32_t u = 0; u <= pos; u++) {
        const float *kp = K + (size_t)u * kv_dim + (size_t)kvh * hdim;
        float p = 0.0f;
        for (uint32_t j = 0; j < nd; j++) p += qv[j] * kp[j * 32u + lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p += __shfl_xor_sync(0xffffffffu, p, off);   /* xor: every lane ends with the sum */

        const float s = p * scale;
        const float mn = fmaxf(m, s);
        const float corr = expf(m - mn), e = expf(s - mn);
        l = l * corr + e;
        const float *vp = V + (size_t)u * kv_dim + (size_t)kvh * hdim;
        for (uint32_t j = 0; j < nd; j++) acc[j] = acc[j] * corr + e * vp[j * 32u + lane];
        m = mn;
    }

    float *op = O + (size_t)t * n_head * hdim + (size_t)hd * hdim;
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    /* Gated Attention's sigmoid gate, applied here rather than in a host loop
     * afterwards. Its input is the second half of this head's q — already
     * uploaded, and indexed by exactly the (token, head, dim) this thread owns —
     * so it costs one extra load and a sigmoid on a value that is otherwise
     * written and immediately re-read. On the host it was 36 ms of a 463 ms
     * prefill (502 tokens, 8 layers, ~16M sigmoids). */
    for (uint32_t j = 0; j < nd; j++) {
        float r = acc[j] * inv;
        if (gated) r *= 1.0f / (1.0f + expf(-qp[hdim + j * 32u + lane]));
        op[j * 32u + lane] = r;
    }
}

/* Scratch arena, same reasoning as g_stage: attention runs once per full-
 * attention layer per chunk and the buffers are shaped by the chunk, not by the
 * layer, so one growable arena serves all of them. */
/* One capacity per buffer. Sharing a single g_at_nkv between K and V looked
 * harmless -- they are always the same size -- but the first reserve set the cap
 * and the second then saw `want <= cap` and returned success WITHOUT ever
 * allocating V, so the memcpy went to a null pointer. */
static float *g_at_q = NULL, *g_at_k = NULL, *g_at_v = NULL, *g_at_o = NULL;
static size_t g_at_nq = 0, g_at_nk = 0, g_at_nv = 0, g_at_no = 0;

static int at_reserve(float **p, size_t *cap, size_t want) {
    if (want <= *cap) return 0;
    cudaFree(*p); *p = NULL; *cap = 0;
    if (cudaMalloc(p, want * sizeof(float)) != cudaSuccess) {
        *p = NULL; cudaGetLastError(); return -1;
    }
    *cap = want;
    return 0;
}

extern "C" int ds4x_cuda_attn_gqa(const float *q, const float *k, const float *v,
                                  float *o, uint32_t n_tok, uint32_t pos0,
                                  uint32_t n_head, uint32_t n_head_kv, uint32_t hdim,
                                  uint32_t qstride, float scale, int gated) {
    if (!q || !k || !v || !o) return -1;
    if (!n_tok || !n_head || !n_head_kv || !hdim) return -1;
    if (hdim % 32u || (hdim >> 5) > ATTN_MAX_ND) return -1;
    if (n_head % n_head_kv) return -1;
    if (qstride < hdim) return -1;
    /* The gate reads q[hdim .. 2*hdim) of each head; without room for it there
     * is no gate to apply and asking for one is a caller bug, not a fallback. */
    if (gated && qstride < 2u * hdim) return -1;

    const uint32_t n_kv = pos0 + n_tok;
    const uint32_t kv_dim = n_head_kv * hdim;
    const size_t nq = (size_t)n_tok * n_head * qstride;
    const size_t nkv = (size_t)n_kv * kv_dim;
    const size_t no = (size_t)n_tok * n_head * hdim;
    if (at_reserve(&g_at_q, &g_at_nq, nq) != 0) return -1;
    if (at_reserve(&g_at_k, &g_at_nk, nkv) != 0) return -1;
    if (at_reserve(&g_at_v, &g_at_nv, nkv) != 0) return -1;
    if (at_reserve(&g_at_o, &g_at_no, no) != 0) return -1;

    cudaError_t e;
    if ((e = cudaMemcpy(g_at_q, q, nq * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMemcpy(g_at_k, k, nkv * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMemcpy(g_at_v, v, nkv * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("attn cudaMemcpy(in)", e); return -1;
    }
    const uint32_t warps = n_tok * n_head;
    const uint32_t grid = (warps + ATTN_WARPS - 1u) / ATTN_WARPS;
    attn_gqa_kernel<<<grid, ATTN_WARPS * 32u>>>(
        g_at_q, g_at_k, g_at_v, g_at_o, n_tok, pos0, n_head,
        n_head / n_head_kv, hdim, kv_dim, qstride, scale, gated);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("attn launch", e); return -1; }
    if ((e = cudaMemcpy(o, g_at_o, no * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        set_err("attn cudaMemcpy(out)", e); return -1;
    }
    return 0;
}

/* ---- GDN recurrence API -------------------------------------------------- */
extern "C" ds4x_cuda_gdn *ds4x_cuda_gdn_create(uint32_t k_heads, uint32_t v_heads,
                                               uint32_t k_dim, uint32_t v_dim) {
    if (!k_heads || !v_heads || !k_dim || !v_dim || (v_heads % k_heads)) {
        snprintf(g_err, sizeof(g_err),
                 "gdn_create: bad shape kh=%u vh=%u kdim=%u vdim=%u",
                 k_heads, v_heads, k_dim, v_dim);
        return NULL;
    }
    ds4x_cuda_gdn *g = (ds4x_cuda_gdn *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->kh = k_heads; g->vh = v_heads; g->kdim = k_dim; g->vdim = v_dim;
    g->state_n = (size_t)v_heads * k_dim * v_dim;
    /* The recurrent state counts against the same budget as the weights — it is
     * per LINEAR layer and grows with v_heads·k_dim·v_dim, so a hybrid model
     * with many linear layers is not a rounding error. Over budget → NULL, and
     * the caller keeps the recurrence on the CPU (ds4x_runner handles that). */
    if (g_budget && g_bytes + (uint64_t)g->state_n * sizeof(float) > g_budget) {
        snprintf(g_err, sizeof(g_err),
                 "VRAM budget %.2f GB reached — GDN state stays on the CPU",
                 (double)g_budget / 1073741824.0);
        free(g);
        return NULL;
    }
    cudaError_t e = cudaMalloc(&g->d_state, g->state_n * sizeof(float));
    if (e != cudaSuccess) { set_err("cudaMalloc(gdn state)", e); free(g); return NULL; }
    if ((e = cudaMemset(g->d_state, 0, g->state_n * sizeof(float))) != cudaSuccess) {
        set_err("cudaMemset(gdn state)", e); cudaFree(g->d_state); free(g); return NULL;
    }
    g_bytes += g->state_n * sizeof(float);
    return g;
}

/* Grow the per-chunk staging buffers to hold n_tokens. Pinned host memory: the
 * copies here are small (24 KB/token) and latency-bound, and pageable copies
 * would add a staging memcpy inside the driver on every one of them. */
static int gdn_reserve(ds4x_cuda_gdn *g, uint32_t n_tokens) {
    if (n_tokens <= g->cap_tokens) return 0;
    const uint32_t kd = g->kh * g->kdim, vd = g->vh * g->vdim;
    const size_t cn = (size_t)n_tokens * (kd * 2u + vd) * sizeof(float);
    const size_t bn = (size_t)n_tokens * g->vh * sizeof(float);
    const size_t on = (size_t)n_tokens * vd * sizeof(float);
    cudaFree(g->d_cnv);  cudaFree(g->d_bet);  cudaFree(g->d_dec);  cudaFree(g->d_core);
    cudaFreeHost(g->h_cnv); cudaFreeHost(g->h_bet);
    cudaFreeHost(g->h_dec); cudaFreeHost(g->h_core);
    g->d_cnv = g->d_bet = g->d_dec = g->d_core = NULL;
    g->h_cnv = g->h_bet = g->h_dec = g->h_core = NULL;
    g->cap_tokens = 0;
    cudaError_t e;
    if ((e = cudaMalloc(&g->d_cnv,  cn)) != cudaSuccess ||
        (e = cudaMalloc(&g->d_bet,  bn)) != cudaSuccess ||
        (e = cudaMalloc(&g->d_dec,  bn)) != cudaSuccess ||
        (e = cudaMalloc(&g->d_core, on)) != cudaSuccess ||
        (e = cudaHostAlloc(&g->h_cnv,  cn, cudaHostAllocDefault)) != cudaSuccess ||
        (e = cudaHostAlloc(&g->h_bet,  bn, cudaHostAllocDefault)) != cudaSuccess ||
        (e = cudaHostAlloc(&g->h_dec,  bn, cudaHostAllocDefault)) != cudaSuccess ||
        (e = cudaHostAlloc(&g->h_core, on, cudaHostAllocDefault)) != cudaSuccess) {
        set_err("cudaMalloc(gdn staging)", e);
        return -1;
    }
    g->cap_tokens = n_tokens;
    return 0;
}

extern "C" void ds4x_cuda_gdn_free(ds4x_cuda_gdn *g) {
    if (!g) return;
    const size_t sb = g->state_n * sizeof(float);
    if (sb <= g_bytes) g_bytes -= sb;
    cudaFree(g->d_state);
    cudaFree(g->d_cnv); cudaFree(g->d_bet); cudaFree(g->d_dec); cudaFree(g->d_core);
    cudaFreeHost(g->h_cnv); cudaFreeHost(g->h_bet);
    cudaFreeHost(g->h_dec); cudaFreeHost(g->h_core);
    cudaFree(g->d_raw); cudaFree(g->d_win); cudaFree(g->d_bb); cudaFree(g->d_ab);
    cudaFree(g->d_cw);  cudaFree(g->d_cb);  cudaFree(g->d_alog); cudaFree(g->d_dt);
    cudaFree(g->d_ssm);
    cudaFreeHost(g->h_pre);
    free(g);
}

extern "C" int ds4x_cuda_gdn_zero(ds4x_cuda_gdn *g) {
    if (!g) return -1;
    cudaError_t e = cudaMemset(g->d_state, 0, g->state_n * sizeof(float));
    if (e != cudaSuccess) { set_err("cudaMemset(gdn zero)", e); return -1; }
    return 0;
}

extern "C" int ds4x_cuda_gdn_get_state(const ds4x_cuda_gdn *g, float *state) {
    if (!g || !state) return -1;
    cudaError_t e = cudaMemcpy(state, g->d_state, g->state_n * sizeof(float),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { set_err("cudaMemcpy(gdn get)", e); return -1; }
    return 0;
}

extern "C" int ds4x_cuda_gdn_set_state(ds4x_cuda_gdn *g, const float *state) {
    if (!g || !state) return -1;
    cudaError_t e = cudaMemcpy(g->d_state, state, g->state_n * sizeof(float),
                               cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_err("cudaMemcpy(gdn set)", e); return -1; }
    return 0;
}

/* Upload the four per-layer pass-2 weights, once per handle. conv_ch doubles as
 * the "already uploaded" flag. Shared by gdn_pre_run and gdn_layer. */
static int gdn_pre_weights(ds4x_cuda_gdn *g, const float *cw, const float *cb,
                           const float *A_log, const float *dt_bias,
                           uint32_t conv_ch, uint32_t K) {
    if (g->conv_ch != 0) return 0;
    cudaError_t e;
    const size_t cwn = (size_t)conv_ch * K * sizeof(float);
    if ((e = cudaMalloc(&g->d_cw, cwn)) != cudaSuccess ||
        (e = cudaMalloc(&g->d_cb, conv_ch * sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&g->d_alog, g->vh * sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&g->d_dt, g->vh * sizeof(float))) != cudaSuccess) {
        set_err("cudaMalloc(gdn pre weights)", e); return -1;
    }
    if ((e = cudaMemcpy(g->d_cw, cw, cwn, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMemcpy(g->d_alog, A_log, g->vh * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMemcpy(g->d_dt, dt_bias, g->vh * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(gdn pre weights)", e); return -1;
    }
    if (cb) {
        if ((e = cudaMemcpy(g->d_cb, cb, conv_ch * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
            set_err("cudaMemcpy(gdn conv bias)", e); return -1; }
    } else {
        if ((e = cudaMemset(g->d_cb, 0, conv_ch * sizeof(float))) != cudaSuccess) {
            set_err("cudaMemset(gdn conv bias)", e); return -1; }
    }
    g->conv_ch = conv_ch; g->K = K;
    return 0;
}

/* pass-4's ssm_norm, cached like the rest (v_dim floats, shared across heads). */
static int gdn_ssm_weight(ds4x_cuda_gdn *g, const float *ssm_norm) {
    if (g->d_ssm) return 0;
    cudaError_t e;
    if ((e = cudaMalloc(&g->d_ssm, g->vdim * sizeof(float))) != cudaSuccess ||
        (e = cudaMemcpy(g->d_ssm, ssm_norm, g->vdim * sizeof(float),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMalloc(gdn ssm_norm)", e); return -1; }
    return 0;
}

/* One call for pass 2 + pass 3: conv/silu, gates, L2 norm, then the recurrence.
 *
 * The conv writes straight into g->d_cnv — the buffer the recurrence kernel
 * already reads — so the intermediate never touches the host. That is the whole
 * point: unfused, pass 2 ran on the CPU between two device calls, which is one
 * host round trip per linear layer per token spent on arithmetic that is
 * per-token independent and trivially parallel.
 *
 * Per-layer weights are uploaded on the first call and kept (see the struct).
 * `conv_win_out` receives the last K-1 raw rows for the next chunk, computed
 * host-side by the caller as before — it is a few KB and needs the raw rows,
 * which the caller already has. */
extern "C" int ds4x_cuda_gdn_pre_run(ds4x_cuda_gdn *g, uint32_t n_tokens,
                                     const float *raw, const float *win,
                                     const float *cw, const float *cb,
                                     const float *bb, const float *ab,
                                     const float *A_log, const float *dt_bias,
                                     uint32_t conv_ch, uint32_t K, float *core) {
    if (!g || !raw || !win || !cw || !bb || !ab || !A_log || !dt_bias || !core ||
        n_tokens == 0 || conv_ch == 0 || K == 0) return -1;
    const uint32_t kd = g->kh * g->kdim, vd = g->vh * g->vdim;
    if (conv_ch != kd * 2u + vd) {
        snprintf(g_err, sizeof(g_err), "gdn_pre_run: conv_ch %u != 2*%u+%u",
                 conv_ch, kd, vd);
        return -1;
    }
    const double t0 = host_ms();
    if (gdn_reserve(g, n_tokens) != 0) return -1;
    if (!g_gev0) { cudaEventCreate(&g_gev0); cudaEventCreate(&g_gev1); }
    cudaError_t e;

    if (gdn_pre_weights(g, cw, cb, A_log, dt_bias, conv_ch, K) != 0) return -1;

    /* Per-call inputs, packed into one pinned buffer so they upload without a
     * synchronize between them. */
    const size_t n_raw = (size_t)n_tokens * conv_ch;
    const size_t n_win = (size_t)(K - 1) * conv_ch;
    const size_t n_g   = (size_t)n_tokens * g->vh;
    const size_t need  = n_raw + n_win + 2 * n_g;
    if (need > g->pre_cap) {
        cudaFree(g->d_raw); cudaFree(g->d_win); cudaFree(g->d_bb); cudaFree(g->d_ab);
        cudaFreeHost(g->h_pre);
        g->d_raw = g->d_win = g->d_bb = g->d_ab = NULL; g->h_pre = NULL; g->pre_cap = 0;
        if ((e = cudaMalloc(&g->d_raw, n_raw * sizeof(float))) != cudaSuccess ||
            (e = cudaMalloc(&g->d_win, (n_win ? n_win : 1) * sizeof(float))) != cudaSuccess ||
            (e = cudaMalloc(&g->d_bb, n_g * sizeof(float))) != cudaSuccess ||
            (e = cudaMalloc(&g->d_ab, n_g * sizeof(float))) != cudaSuccess ||
            (e = cudaHostAlloc(&g->h_pre, need * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
            set_err("cudaMalloc(gdn pre staging)", e); return -1;
        }
        g->pre_cap = need;
    }
    memcpy(g->h_pre,                       raw, n_raw * sizeof(float));
    if (n_win) memcpy(g->h_pre + n_raw,    win, n_win * sizeof(float));
    memcpy(g->h_pre + n_raw + n_win,       bb,  n_g   * sizeof(float));
    memcpy(g->h_pre + n_raw + n_win + n_g, ab,  n_g   * sizeof(float));
    if ((e = cudaMemcpyAsync(g->d_raw, g->h_pre, n_raw * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess ||
        (n_win && (e = cudaMemcpyAsync(g->d_win, g->h_pre + n_raw, n_win * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess) ||
        (e = cudaMemcpyAsync(g->d_bb, g->h_pre + n_raw + n_win, n_g * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess ||
        (e = cudaMemcpyAsync(g->d_ab, g->h_pre + n_raw + n_win + n_g, n_g * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
        set_err("cudaMemcpyAsync(gdn pre in)", e); return -1;
    }

    const size_t on = (size_t)n_tokens * vd * sizeof(float);
    cudaEventRecord(g_gev0);
    {
        const uint32_t th = 256;
        dim3 grid((conv_ch + th - 1u) / th, n_tokens);
        gdn_conv_kernel<<<grid, th>>>(g->d_raw, g->d_win, g->d_cw, g->d_cb,
                                      g->d_cnv, n_tokens, conv_ch, K);
    }
    {
        const uint32_t th = 128;
        const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
        dim3 grid(2u * g->kh, n_tokens);
        gdn_l2norm_kernel<<<grid, th, sh>>>(g->d_cnv, conv_ch, g->kh, g->kdim);
    }
    {
        const uint32_t th = 128;
        const size_t n = (size_t)n_tokens * g->vh;
        gdn_gates_kernel<<<(uint32_t)((n + th - 1) / th), th>>>(
            g->d_bb, g->d_ab, g->d_alog, g->d_dt, g->d_bet, g->d_dec,
            n_tokens, g->vh);
    }
    gdn_launch_recur(g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
                     g->d_cnv, g->d_bet, g->d_dec, g->d_core,
                     1.0f / sqrtf((float)g->vdim));
    cudaEventRecord(g_gev1);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("gdn pre launch", e); return -1; }
    if ((e = cudaMemcpyAsync(g->h_core, g->d_core, on, cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
        set_err("cudaMemcpyAsync(gdn pre out)", e); return -1;
    }
    if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("gdn pre sync", e); return -1; }
    memcpy(core, g->h_core, on);
    float kms = 0.0f;
    if (cudaEventElapsedTime(&kms, g_gev0, g_gev1) == cudaSuccess) g_gdn_ms_kernel += kms;
    g_gdn_ms_total += host_ms() - t0;
    g_gdn_calls++;
    return 0;
}

/* Widest prefill chunk the whole-layer fused path will take. Matches ds4's
 * prefill chunk cap (ds4_prefill_chunk_cap_for_ctx), which is what actually
 * bounds n_tokens on this path, so in practice nothing is ever refused. */
#define GDN_LAYER_MAX_TOKENS 2048u

/* A WHOLE linear layer's attention in one call: the four input projections,
 * conv/silu, the gates, the L2 norm, and the recurrence.
 *
 * This is the last host round trip inside a linear layer. proj_fanout already
 * wrote qkv/z/b/a into one contiguous device arena; gdn_pre_run then downloaded
 * three of those slices only to upload them again. Here the conv reads the qkv
 * slice where it already sits.
 *
 * What still comes back, and why:
 *   core : pass 4 (the gated RMSNorm) is host code
 *   z    : pass 4 needs it
 *   raw  : the conv window carry needs the last K-1 pre-conv rows, and that
 *          bookkeeping is host-side. raw and z are ADJACENT slices of the arena,
 *          so both ride one download.
 * b and a are consumed entirely on the device and never return. */
extern "C" int ds4x_cuda_gdn_layer(ds4x_cuda_gdn *g,
                                   const ds4x_cuda_wt *Wqkv, const ds4x_cuda_wt *Wz,
                                   const ds4x_cuda_wt *Wb, const ds4x_cuda_wt *Wa,
                                   const float *normed, const float *win,
                                   const float *cw, const float *cb,
                                   const float *A_log, const float *dt_bias,
                                   uint32_t n_tokens, uint32_t conv_ch, uint32_t K,
                                   float *raw_out, float *z_out, float *core_out,
                                   const ds4x_gdn_tail *tail) {
    /* The cap used to be MM_BLOCK (256), which quietly disabled this entire path
     * for any prefill chunk bigger than that -- i.e. for every prompt longer than
     * a couple of sentences. Measured on Qwen3.5-4B (GB10, 2026-08-13): a
     * 178-token prompt fused and ran at 943 tok/s, a 502-token one fell back to
     * the fanout + pre_run pair and ran at 782. A 21% cliff at a boundary nobody
     * could see, on the more common side of it.
     *
     * MM_BLOCK was the wrong bound to borrow: it is the token block that the
     * PUBLIC ds4x_cuda_matmul loops over, and nothing below depends on it.
     * Everything here is sized from n_tokens and grows -- stage_reserve,
     * proj_reserve, gdn_reserve, g->pre_cap -- and launch_matmul takes any token
     * count. The real bound is memory, so it is stated as one: a chunk is capped
     * at ds4_prefill_chunk_cap_for_ctx (2048), and at that width the staging is
     * tens of MB. */
    if (!g || !Wqkv || !Wz || !Wb || !Wa || !normed || !win || !raw_out ||
        !z_out || !core_out || n_tokens == 0 ||
        n_tokens > GDN_LAYER_MAX_TOKENS) return -1;
    const uint32_t kd = g->kh * g->kdim, vd = g->vh * g->vdim;
    const uint32_t n_in = Wqkv->n_in;
    if (conv_ch != kd * 2u + vd || Wqkv->n_out != conv_ch || Wz->n_out != vd ||
        Wb->n_out != g->vh || Wa->n_out != g->vh ||
        Wz->n_in != n_in || Wb->n_in != n_in || Wa->n_in != n_in) {
        snprintf(g_err, sizeof(g_err), "gdn_layer: shape mismatch");
        return -1;
    }
    const double t0 = host_ms();
    if (gdn_reserve(g, n_tokens) != 0) return -1;
    if (gdn_pre_weights(g, cw, cb, A_log, dt_bias, conv_ch, K) != 0) return -1;
    if (tail && gdn_ssm_weight(g, tail->ssm_norm) != 0) return -1;
    if (!g_gev0) { cudaEventCreate(&g_gev0); cudaEventCreate(&g_gev1); }
    cudaError_t e;

    /* Arena slices, in the order the download wants them: raw | z | b | a. */
    const size_t n_raw = (size_t)n_tokens * conv_ch, n_z = (size_t)n_tokens * vd;
    const size_t n_g = (size_t)n_tokens * g->vh;
    const size_t total = n_raw + n_z + 2 * n_g;
    const size_t nx = (size_t)n_tokens * n_in;
    const size_t n_win = (size_t)(K - 1) * conv_ch;
    if (stage_reserve(nx, 1) != 0 || proj_reserve(total) != 0) return -1;
    if (n_win + nx > g->pre_cap) {
        cudaFree(g->d_win); cudaFreeHost(g->h_pre);
        g->d_win = NULL; g->h_pre = NULL; g->pre_cap = 0;
        if ((e = cudaMalloc(&g->d_win, (n_win ? n_win : 1) * sizeof(float))) != cudaSuccess ||
            (e = cudaHostAlloc(&g->h_pre, (n_win + nx) * sizeof(float), cudaHostAllocDefault)) != cudaSuccess) {
            set_err("cudaMalloc(gdn layer staging)", e); return -1;
        }
        g->pre_cap = n_win + nx;
    }
    /* normed and the carried window in one pinned buffer, two async uploads. */
    memcpy(g->h_pre, normed, nx * sizeof(float));
    if (n_win) memcpy(g->h_pre + nx, win, n_win * sizeof(float));
    if ((e = cudaMemcpyAsync(g_stage.d_x, g->h_pre, nx * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess ||
        (n_win && (e = cudaMemcpyAsync(g->d_win, g->h_pre + nx, n_win * sizeof(float),
                             cudaMemcpyHostToDevice, 0)) != cudaSuccess)) {
        set_err("cudaMemcpyAsync(gdn layer in)", e); return -1;
    }

    float *d_raw = g_proj.d, *d_z = d_raw + n_raw;
    float *d_b = d_z + n_z, *d_a = d_b + n_g;
    cudaEventRecord(g_gev0);
    {
        const ds4x_cuda_wt *W[4] = { Wqkv, Wz, Wb, Wa };
        float *Y[4] = { d_raw, d_z, d_b, d_a };
        for (int i = 0; i < 4; i++) {
            const int rc = (n_tokens == 1)
                ? launch_matvec(W[i], g_stage.d_x, Y[i])
                : launch_matmul(W[i], g_stage.d_x, Y[i], n_tokens);
            if (rc != 0) return -1;
        }
    }
    {
        const uint32_t th = 256;
        dim3 grid((conv_ch + th - 1u) / th, n_tokens);
        gdn_conv_kernel<<<grid, th>>>(d_raw, g->d_win, g->d_cw, g->d_cb,
                                      g->d_cnv, n_tokens, conv_ch, K);
    }
    {
        const uint32_t th = 128;
        const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
        dim3 grid(2u * g->kh, n_tokens);
        gdn_l2norm_kernel<<<grid, th, sh>>>(g->d_cnv, conv_ch, g->kh, g->kdim);
    }
    {
        const uint32_t th = 128;
        gdn_gates_kernel<<<(uint32_t)((n_g + th - 1) / th), th>>>(
            d_b, d_a, g->d_alog, g->d_dt, g->d_bet, g->d_dec, n_tokens, g->vh);
    }
    gdn_launch_recur(g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
                     g->d_cnv, g->d_bet, g->d_dec, g->d_core,
                     1.0f / sqrtf((float)g->vdim));
    if (tail) {
        /* Pass 4 and the layer tail, still on the device: `core` is already
         * here, so the only thing that has to come back is the finished layer
         * output (and `raw`, for the conv window carry). */
        const uint32_t n_embd = tail->Wo->n_out, ff = tail->Wg->n_out;
        const size_t nh = (size_t)n_tokens * n_embd;
        if (tail_reserve(n_z, nh, n_embd, nh) != 0) return -1;
        if (ffn_reserve((size_t)n_tokens * ff) != 0) return -1;
        /* ffn_norm and ssm_norm: ssm_norm is per-layer and cached; ffn_norm
         * rides the per-call staging (it is n_embd floats). */
        if ((e = cudaMemcpyAsync(g_tail.d_w, tail->ffn_norm_w, n_embd * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(gdn tail norm)", e); return -1; }
        if ((e = cudaMemcpyAsync(g_tail.d_h, tail->x, nh * sizeof(float),
                                 cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(gdn tail resid)", e); return -1; }
        {
            const uint32_t th = 128;
            const uint32_t sh = ((th + 31u) / 32u) * (uint32_t)sizeof(float);
            dim3 grid(g->vh, n_tokens);
            gdn_gatednorm_kernel<<<grid, th, sh>>>(g->d_core, d_z, g->d_ssm,
                                                   g->vh, g->vdim, tail->eps);
        }
        if (launch_attn_ffn_tail(tail->Wo, tail->Wg, tail->Wu, tail->Wd,
                                 g->d_core, g_tail.d_h, g_tail.d_n, g_tail.d_w,
                                 n_tokens, n_embd, ff, tail->eps) != 0) return -1;
    }
    cudaEventRecord(g_gev1);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("gdn layer launch", e); return -1; }

    if (tail) {
        const uint32_t n_embd = tail->Wo->n_out;
        const size_t nh = (size_t)n_tokens * n_embd;
        if ((e = cudaMemcpyAsync(g_proj.h, d_raw, n_raw * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess ||
            (e = cudaMemcpyAsync(g_tail.h_io, g_tail.d_h, nh * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
            set_err("cudaMemcpyAsync(gdn layer tail out)", e); return -1; }
        if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("gdn layer sync", e); return -1; }
        memcpy(raw_out,  g_proj.h,     n_raw * sizeof(float));
        memcpy(tail->x,  g_tail.h_io,  nh * sizeof(float));
        float kms0 = 0.0f;
        if (cudaEventElapsedTime(&kms0, g_gev0, g_gev1) == cudaSuccess) g_gdn_ms_kernel += kms0;
        g_gdn_ms_total += host_ms() - t0;
        g_gdn_calls++;
        return 0;
    }

    /* raw|z ride one download (adjacent slices); core is a separate buffer. */
    const size_t n_rz = n_raw + n_z;
    if ((e = cudaMemcpyAsync(g_proj.h, g_proj.d, n_rz * sizeof(float),
                             cudaMemcpyDeviceToHost, 0)) != cudaSuccess ||
        (e = cudaMemcpyAsync(g->h_core, g->d_core, n_z * sizeof(float),
                             cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
        set_err("cudaMemcpyAsync(gdn layer out)", e); return -1;
    }
    if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("gdn layer sync", e); return -1; }
    memcpy(raw_out,  g_proj.h,         n_raw * sizeof(float));
    memcpy(z_out,    g_proj.h + n_raw, n_z   * sizeof(float));
    memcpy(core_out, g->h_core,        n_z   * sizeof(float));
    float kms = 0.0f;
    if (cudaEventElapsedTime(&kms, g_gev0, g_gev1) == cudaSuccess) g_gdn_ms_kernel += kms;
    g_gdn_ms_total += host_ms() - t0;
    g_gdn_calls++;
    return 0;
}

extern "C" int ds4x_cuda_gdn_run(ds4x_cuda_gdn *g, uint32_t n_tokens,
                                 const float *cnv, const float *bet,
                                 const float *dec, float *core) {
    if (!g || !cnv || !bet || !dec || !core || n_tokens == 0) return -1;
    const double t0 = host_ms();
    if (gdn_reserve(g, n_tokens) != 0) return -1;
    if (!g_gev0) { cudaEventCreate(&g_gev0); cudaEventCreate(&g_gev1); }

    const uint32_t kd = g->kh * g->kdim, vd = g->vh * g->vdim;
    const size_t cn = (size_t)n_tokens * (kd * 2u + vd) * sizeof(float);
    const size_t bn = (size_t)n_tokens * g->vh * sizeof(float);
    const size_t on = (size_t)n_tokens * vd * sizeof(float);
    memcpy(g->h_cnv, cnv, cn);
    memcpy(g->h_bet, bet, bn);
    memcpy(g->h_dec, dec, bn);

    cudaError_t e;
    if ((e = cudaMemcpyAsync(g->d_cnv, g->h_cnv, cn, cudaMemcpyHostToDevice, 0)) != cudaSuccess ||
        (e = cudaMemcpyAsync(g->d_bet, g->h_bet, bn, cudaMemcpyHostToDevice, 0)) != cudaSuccess ||
        (e = cudaMemcpyAsync(g->d_dec, g->h_dec, bn, cudaMemcpyHostToDevice, 0)) != cudaSuccess) {
        set_err("cudaMemcpyAsync(gdn in)", e); return -1;
    }

    cudaEventRecord(g_gev0);
    gdn_launch_recur(g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
                     g->d_cnv, g->d_bet, g->d_dec, g->d_core,
                     1.0f / sqrtf((float)g->vdim));
    cudaEventRecord(g_gev1);
    if ((e = cudaGetLastError()) != cudaSuccess) { set_err("gdn kernel launch", e); return -1; }
    if ((e = cudaMemcpyAsync(g->h_core, g->d_core, on, cudaMemcpyDeviceToHost, 0)) != cudaSuccess) {
        set_err("cudaMemcpyAsync(gdn out)", e); return -1;
    }
    if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { set_err("gdn sync", e); return -1; }
    memcpy(core, g->h_core, on);

    float kms = 0.0f;
    if (cudaEventElapsedTime(&kms, g_gev0, g_gev1) == cudaSuccess) g_gdn_ms_kernel += kms;
    g_gdn_ms_total += host_ms() - t0;
    g_gdn_calls++;
    return 0;
}

extern "C" void ds4x_cuda_gdn_stats(double *ms_kernel, double *ms_total, uint64_t *calls) {
    if (ms_kernel) *ms_kernel = g_gdn_ms_kernel;
    if (ms_total)  *ms_total  = g_gdn_ms_total;
    if (calls)     *calls     = (uint64_t)g_gdn_calls;
}

extern "C" uint64_t ds4x_cuda_bytes_resident(void) { return g_bytes; }
extern "C" const char *ds4x_cuda_device_name(void) { return g_dev[0] ? g_dev : "(no device)"; }
extern "C" const char *ds4x_cuda_last_error(void) { return g_err[0] ? g_err : "(none)"; }
