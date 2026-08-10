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
    if (w->type == T_Q4_K) {
        const uint32_t th = 128;
        matvec_q4k_kernel<<<n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
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
    if (w->type == T_Q4_K) {
        /* coalesced specialization (dominant type in a Q4_K_M model) */
        const uint32_t th = 128;
        matvec_q4k_kernel<<<w->n_out, th, ((th + 31u) / 32u) * (uint32_t)sizeof(float)>>>(
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
    const uint32_t bc = blk_count(w->type);
    const uint32_t bb = (uint32_t)blk_bytes(w->type);

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
        const uint32_t ntile = (nt + TOK_TILE - 1u) / TOK_TILE;
        cudaEventRecord(g_mev0);
        if (w->type == T_Q4_K) {
            const uint32_t th = 128;
            const uint32_t sh = ((th + 31u) / 32u) * TOK_TILE * (uint32_t)sizeof(float);
            matmul_q4k_kernel<<<dim3(w->n_out, ntile), th, sh>>>(
                (const unsigned char *)w->d_data, w->n_in, w->n_out, nt,
                g_stage.d_x, g_stage.d_y);
        } else {
            uint32_t th = (w->n_in / UNIT) < 256u ? (w->n_in / UNIT) : 256u;
            th = ((th + 31u) / 32u) * 32u;
            if (th == 0) th = 32;
            const uint32_t sh = ((th + 31u) / 32u) * TOK_TILE * (uint32_t)sizeof(float);
            matmul_kernel<<<dim3(w->n_out, ntile), th, sh>>>(
                (const unsigned char *)w->d_data, w->type, w->n_in, w->n_out,
                bc, bb, nt, g_stage.d_x, g_stage.d_y);
        }
        cudaEventRecord(g_mev1);
        if ((e = cudaGetLastError()) != cudaSuccess) { set_err("matmul launch", e); return -1; }
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
