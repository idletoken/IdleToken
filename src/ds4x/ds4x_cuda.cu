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
static inline int q4k_narrow_shape(const ds4x_cuda_wt *w) {
    /* IDLETOKEN_DS4X_NO_NARROW=1 forces the block-per-row kernel, so this
     * choice can be A/B'd IN ONE BINARY like every other change in this file.
     * It exists because the kernel was committed on a kernel-time improvement
     * that never showed up in total time, and settling that needs a paired
     * measurement rather than two builds an hour apart. */
    static int off = -1;
    if (off < 0) { const char *e = getenv("IDLETOKEN_DS4X_NO_NARROW");
                   off = (e && e[0] == '1'); }
    return !off && w->type == T_Q4_K && w->n_in <= 2048u && (w->n_in % QK_K) == 0;
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
 * prefill stays bit-identical to token-by-token decode. */
#define TOK_TILE 8

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

/* Generic types: unit_dot() is called once per (unit, token). The weight bytes
 * stay in L1 across the token loop, so this still collapses 540 launches into
 * one and reuses each fetched byte — just without the explicit dequant-once of
 * the Q4_K path above. Worth specializing further only if a non-Q4_K model
 * becomes the hot case; measure first. */
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
        #pragma unroll
        for (int tt = 0; tt < TOK_TILE; tt++) {
            if (t0 + (uint32_t)tt >= n_tok) break;
            acc[tt] += unit_dot(type, rowp, u, X + (size_t)(t0 + tt) * n_in);
        }

    extern __shared__ float smg[];
    tile_store(acc, smg, Y, row, n_out, t0, n_tok);
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
    const uint32_t n_units = w->n_in / UNIT;
    uint32_t threads = n_units < 256 ? n_units : 256;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;

    cudaError_t e;
    if ((e = cudaMemcpy(w->d_x, x, (size_t)w->n_in * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(x)", e); return -1;
    }
    const uint32_t shmem = ((threads + 31u) / 32u) * (uint32_t)sizeof(float);
    cudaEventRecord(g_ev0);
    if (q4k_narrow_shape(w)) {
        const uint32_t th = MV_WARPS * 32u;
        matvec_q4k_narrow_kernel<<<(n_out + MV_WARPS - 1u) / MV_WARPS, th>>>(
            base, w->n_in, n_out, w->d_x, w->d_y);
    } else if (w->type == T_Q4_K) {
        const uint32_t th = 128;
        matvec_q4k_kernel<<<n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            base, w->n_in, w->d_x, w->d_y);
    } else if (w->type == T_Q6_K) {
        const uint32_t th = 128;
        matvec_q6k_kernel<<<n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            base, w->n_in, w->d_x, w->d_y);
    } else {
        matvec_kernel<<<n_out, threads, shmem>>>(
            base, w->type, w->n_in, bc, bb, w->d_x, w->d_y);
    }
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
    const uint32_t bc = blk_count(w->type);
    const uint32_t bb = (uint32_t)blk_bytes(w->type);
    /* Size the block to the row's 32-element units (cap 256), rounded up to a
     * whole warp — the old sizing (one thread per QUANT BLOCK) left Q4_K rows
     * with 16 busy threads out of 256. */
    const uint32_t n_units = w->n_in / UNIT;
    uint32_t threads = n_units < 256 ? n_units : 256;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;

    cudaError_t e;
    if ((e = cudaMemcpy(w->d_x, x, (size_t)w->n_in * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        set_err("cudaMemcpy(x)", e); return -1;
    }
    /* shared = one float per warp */
    const uint32_t shmem = ((threads + 31u) / 32u) * (uint32_t)sizeof(float);
    cudaEventRecord(g_ev0);
    if (q4k_narrow_shape(w)) {
        const uint32_t th = MV_WARPS * 32u;
        matvec_q4k_narrow_kernel<<<(w->n_out + MV_WARPS - 1u) / MV_WARPS, th>>>(
            (const unsigned char *)w->d_data, w->n_in, w->n_out, w->d_x, w->d_y);
    } else if (w->type == T_Q4_K) {
        /* coalesced specialization (dominant type in a Q4_K_M model) */
        const uint32_t th = 128;
        matvec_q4k_kernel<<<w->n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            (const unsigned char *)w->d_data, w->n_in, w->d_x, w->d_y);
    } else if (w->type == T_Q6_K) {
        /* The output head lands here: a Q4_K_M model keeps the tied
         * embedding/output matrix at Q6_K, and it is the largest single matvec
         * in the model, re-read every decoded token. */
        const uint32_t th = 128;
        matvec_q6k_kernel<<<w->n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            (const unsigned char *)w->d_data, w->n_in, w->d_x, w->d_y);
    } else {
        matvec_kernel<<<w->n_out, threads, shmem>>>(
            (const unsigned char *)w->d_data, w->type, w->n_in, bc, bb, w->d_x, w->d_y);
    }
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
 * Needed because the matmul kernel tiles TOK_TILE=8 token-rows per block, so a
 * 1-token launch leaves 7 of 8 lanes masked off. ds4x_cuda_matmul's comment
 * claims that costs what a matvec did; measured on a 5060 Ti it does not — the
 * first cut of the fused FFN routed decode through the matmul kernel and paid
 * +33 ms of kernel time per 32 tokens, eating half of what fusing had just
 * saved in round trips.
 *
 * It also keeps the numerics where they were: the unfused decode FFN went
 * through matvec_q, so using the matvec kernel here reproduces the old path
 * exactly rather than introducing a new reduction order. */
static int launch_matvec(const ds4x_cuda_wt *w, const float *d_x, float *d_y) {
    const uint32_t bc = blk_count(w->type);
    const uint32_t bb = (uint32_t)blk_bytes(w->type);
    const uint32_t n_units = w->n_in / UNIT;
    uint32_t threads = n_units < 256 ? n_units : 256;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;
    if (q4k_narrow_shape(w)) {
        const uint32_t th = MV_WARPS * 32u;
        matvec_q4k_narrow_kernel<<<(w->n_out + MV_WARPS - 1u) / MV_WARPS, th>>>(
            (const unsigned char *)w->d_data, w->n_in, w->n_out, d_x, d_y);
    } else if (w->type == T_Q4_K) {
        const uint32_t th = 128;
        matvec_q4k_kernel<<<w->n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            (const unsigned char *)w->d_data, w->n_in, d_x, d_y);
    } else if (w->type == T_Q6_K) {
        const uint32_t th = 128;
        matvec_q6k_kernel<<<w->n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
            (const unsigned char *)w->d_data, w->n_in, d_x, d_y);
    } else {
        const uint32_t shmem = ((threads + 31u) / 32u) * (uint32_t)sizeof(float);
        matvec_kernel<<<w->n_out, threads, shmem>>>(
            (const unsigned char *)w->d_data, w->type, w->n_in, bc, bb, d_x, d_y);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { set_err("matvec launch", e); return -1; }
    return 0;
}

static int launch_matmul(const ds4x_cuda_wt *w, const float *d_x, float *d_y,
                         uint32_t nt) {
    const uint32_t ntile = (nt + TOK_TILE - 1u) / TOK_TILE;
    if (w->type == T_Q4_K) {
        const uint32_t th = 128;
        const uint32_t sh = ((th + 31u) / 32u) * TOK_TILE * (uint32_t)sizeof(float);
        matmul_q4k_kernel<<<dim3(w->n_out, ntile), th, sh>>>(
            (const unsigned char *)w->d_data, w->n_in, w->n_out, nt, d_x, d_y);
    } else {
        uint32_t th = (w->n_in / UNIT) < 256u ? (w->n_in / UNIT) : 256u;
        th = ((th + 31u) / 32u) * 32u;
        if (th == 0) th = 32;
        const uint32_t sh = ((th + 31u) / 32u) * TOK_TILE * (uint32_t)sizeof(float);
        matmul_kernel<<<dim3(w->n_out, ntile), th, sh>>>(
            (const unsigned char *)w->d_data, w->type, w->n_in, w->n_out,
            blk_count(w->type), (uint32_t)blk_bytes(w->type), nt, d_x, d_y);
    }
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
         * inventing a new one, and the matmul kernel's TOK_TILE=8 would leave
         * 7 of 8 lanes masked off on a 1-token launch (measured: +33 ms kernel
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
    {
        uint32_t threads = g->vdim < 256u ? g->vdim : 256u;
        threads = ((threads + 31u) / 32u) * 32u;
        if (threads == 0) threads = 32;
        const uint32_t shmem = 2u * g->kdim * (uint32_t)sizeof(float);
        const float oscale = 1.0f / sqrtf((float)g->vdim);
        gdn_recur_kernel<<<g->vh, threads, shmem>>>(
            g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
            g->d_cnv, g->d_bet, g->d_dec, g->d_core, oscale);
    }
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
    if (!g || !Wqkv || !Wz || !Wb || !Wa || !normed || !win || !raw_out ||
        !z_out || !core_out || n_tokens == 0 || n_tokens > MM_BLOCK) return -1;
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
    {
        uint32_t threads = g->vdim < 256u ? g->vdim : 256u;
        threads = ((threads + 31u) / 32u) * 32u;
        if (threads == 0) threads = 32;
        const uint32_t shmem = 2u * g->kdim * (uint32_t)sizeof(float);
        const float oscale = 1.0f / sqrtf((float)g->vdim);
        gdn_recur_kernel<<<g->vh, threads, shmem>>>(
            g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
            g->d_cnv, g->d_bet, g->d_dec, g->d_core, oscale);
    }
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

    uint32_t threads = g->vdim < 256u ? g->vdim : 256u;
    threads = ((threads + 31u) / 32u) * 32u;
    if (threads == 0) threads = 32;
    const uint32_t shmem = 2u * g->kdim * (uint32_t)sizeof(float);
    const float oscale = 1.0f / sqrtf((float)g->vdim);
    cudaEventRecord(g_gev0);
    gdn_recur_kernel<<<g->vh, threads, shmem>>>(
        g->d_state, n_tokens, g->kh, g->vh, g->kdim, g->vdim,
        g->d_cnv, g->d_bet, g->d_dec, g->d_core, oscale);
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
