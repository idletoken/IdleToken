/* ds4x_quant.c — CPU dequantization. See idletoken_ds4x_quant.h. */
#include "idletoken_ds4x_quant.h"

#include <stdio.h>
#include <string.h>

enum { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q2_K = 10,
       T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_BF16 = 30 };

/* Q2_K + IQ2_XXS are what the sub-4-bit builds of the big MoE models are made
 * of — the GLM-5.2 GGUF we target is 225 IQ2_XXS tensors + 3 Q2_K + Q8_0/F32
 * (docs/multi-model-design.md §7b). vendor/ds4 already implements both, but as
 * QUANTIZED DOT PRODUCTS against Q8_K activations; ds4x is built around
 * "dequantize one row, then a plain fp32 dot", so these are transcribed from
 * the block layout rather than lifted as-is. The lookup tables ARE lifted
 * verbatim (see the .inc). */
#include "ds4x_iq2_tables.inc"

/* K-quant super-block = 256 elements (ggml QK_K). Byte layouts below match
 * ggml-quants.c exactly (the numeric oracle is llama.cpp on real weights,
 * DGX); the known-answer unit tests pin the bit unpacking on any host. */
#define QK_K 256

float ds4x_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
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
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint32_t ds4x_type_block_count(uint32_t t) {
    switch (t) {
        case T_F32: case T_F16: case T_BF16: return 1;
        case T_Q8_0: case T_Q4_0: return 32;
        case T_Q2_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ2_XXS: return QK_K;
        default: return 0;
    }
}

uint64_t ds4x_type_block_bytes(uint32_t t) {
    switch (t) {
        case T_F32:  return 4;
        case T_F16:  return 2;
        case T_BF16: return 2;
        case T_Q8_0: return 2 + 32;      /* f16 scale + 32×int8 */
        case T_Q4_0: return 2 + 16;      /* f16 scale + 32×4bit */
        /* K-quants over a 256-element super-block: */
        case T_Q4_K: return 2 + 2 + 12 + 128;        /* d, dmin, 6-bit scales, 4-bit qs */
        case T_Q5_K: return 2 + 2 + 12 + 32 + 128;   /* + qh high bits */
        case T_Q6_K: return 128 + 64 + 16 + 2;       /* ql, qh, int8 scales, d */
        case T_Q2_K: return 16 + 64 + 2 + 2;         /* scales, 2-bit qs, d, dmin */
        case T_IQ2_XXS: return 2 + 64;               /* d, 32×uint16 (grid+signs) */
        default: return 0;
    }
}

int ds4x_type_supported(uint32_t t) {
    return t == T_F32 || t == T_F16 || t == T_BF16 || t == T_Q8_0 ||
           t == T_Q4_0 || t == T_Q2_K || t == T_Q4_K || t == T_Q5_K ||
           t == T_Q6_K || t == T_IQ2_XXS;
}

/* 6-bit packed (scale,min) for sub-block j of a Q4_K/Q5_K super-block. Verbatim
 * ggml get_scale_min_k4. `q` is the 12-byte scales[] array. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *mn) {
    if (j < 4) {
        *d  = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        /* NOTE the asymmetry (verbatim ggml): the scale's high bits come from
         * q[j-4], the min's from q[j] — NOT both from q[j-4]. Getting this
         * wrong corrupts every min in sub-blocks 4..7 (real weights only —
         * hand-built fixtures with j<4 scales never reach this branch). */
        *d  = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *mn = (uint8_t)((q[j + 4] >>   4) | ((q[j    ] >> 6) << 4));
    }
}

int ds4x_dequant_row(uint32_t t, const void *src, float *dst, uint64_t n,
                     char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)src;
    switch (t) {
        case T_F32:
            memcpy(dst, src, (size_t)n * 4);
            return 0;
        case T_F16: {
            const uint16_t *h = (const uint16_t *)src;
            for (uint64_t i = 0; i < n; i++) dst[i] = ds4x_f16_to_f32(h[i]);
            return 0;
        }
        case T_BF16: {
            /* bfloat16 IS the top 16 bits of an fp32 (same exponent field, 7
             * mantissa bits) — widening is a shift, no exponent rebias, and it
             * is exact. The manifests advertise a BF16 tier for small models,
             * so without this the "full precision" option failed to load. */
            const uint16_t *h = (const uint16_t *)src;
            for (uint64_t i = 0; i < n; i++) {
                const uint32_t bits = (uint32_t)h[i] << 16;
                float f; memcpy(&f, &bits, 4);
                dst[i] = f;
            }
            return 0;
        }
        case T_Q8_0: {
            if (n % 32) goto badlen;
            for (uint64_t b = 0; b < n / 32; b++) {
                uint16_t dh; memcpy(&dh, p, 2);
                const float d = ds4x_f16_to_f32(dh);
                const int8_t *qs = (const int8_t *)(p + 2);
                for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * (float)qs[i];
                p += 34;
            }
            return 0;
        }
        case T_Q4_0: {
            if (n % 32) goto badlen;
            for (uint64_t b = 0; b < n / 32; b++) {
                uint16_t dh; memcpy(&dh, p, 2);
                const float d = ds4x_f16_to_f32(dh);
                const uint8_t *qs = p + 2;
                /* ggml Q4_0: elem j = low nibble of qs[j]; elem j+16 = high. */
                for (int j = 0; j < 16; j++) {
                    const int lo = (qs[j] & 0x0F) - 8;
                    const int hi = (qs[j] >> 4)   - 8;
                    dst[b * 32 + j]      = d * (float)lo;
                    dst[b * 32 + j + 16] = d * (float)hi;
                }
                p += 18;
            }
            return 0;
        }
        case T_Q2_K: {
            /* 84 B super-block: scales[16] | qs[64] | d(f16) | dmin(f16).
             * Each scales byte packs BOTH a 4-bit scale (low) and a 4-bit min
             * (high) for one 16-element sub-block; qs packs four 2-bit values
             * per byte, and the four are separated by SHIFT, not by adjacency
             * — element l and element l+64 share a byte. */
            if (n % QK_K) goto badlen;
            for (uint64_t sb = 0; sb < n / QK_K; sb++) {
                const uint8_t *sc = p;              /* 16 */
                const uint8_t *q  = p + 16;         /* 64 */
                uint16_t dh, dmh;
                memcpy(&dh, p + 80, 2); memcpy(&dmh, p + 82, 2);
                const float d = ds4x_f16_to_f32(dh), dmin = ds4x_f16_to_f32(dmh);
                float *y = dst + sb * QK_K;
                int is = 0;
                for (int n0 = 0; n0 < QK_K; n0 += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        uint8_t s0 = sc[is++];
                        float dl = d * (float)(s0 & 0xF), ml = dmin * (float)(s0 >> 4);
                        for (int l = 0; l < 16; l++)
                            *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
                        s0 = sc[is++];
                        dl = d * (float)(s0 & 0xF); ml = dmin * (float)(s0 >> 4);
                        for (int l = 0; l < 16; l++)
                            *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
                        shift += 2;
                    }
                    q += 32;
                }
                p += 84;
            }
            return 0;
        }
        case T_IQ2_XXS: {
            /* 66 B super-block: d(f16) | qs[32] uint16. Every 32 elements are
             * described by two uint32: the first holds four 8-bit GRID indices
             * (each selecting 8 pre-tabulated magnitudes), the second holds
             * four 7-bit SIGN indices plus a 4-bit block scale in its top
             * nibble. This is why the tables are copied verbatim — the values
             * are not derivable, and a wrong entry looks like plausible noise. */
            if (n % QK_K) goto badlen;
            for (uint64_t sb = 0; sb < n / QK_K; sb++) {
                uint16_t dh; memcpy(&dh, p, 2);
                const float d = ds4x_f16_to_f32(dh);
                const uint8_t *q2 = p + 2;
                float *y = dst + sb * QK_K;
                for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
                    uint32_t aux32[2];
                    memcpy(aux32, q2 + (size_t)ib32 * 8, 8);
                    const uint8_t *aux8 = (const uint8_t *)aux32;
                    const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
                    for (int l = 0; l < 4; l++) {
                        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                        const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                        for (int j = 0; j < 8; j++)
                            *y++ = db * (float)grid[j] *
                                   ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                }
                p += 66;
            }
            return 0;
        }
        case T_Q6_K: {
            if (n % QK_K) goto badlen;
            for (uint64_t sb = 0; sb < n / QK_K; sb++) {
                const uint8_t *ql = p;              /* 128 */
                const uint8_t *qh = ql + 128;       /* 64  */
                const int8_t  *sc = (const int8_t *)(qh + 64); /* 16 (signed) */
                uint16_t dh; memcpy(&dh, qh + 64 + 16, 2);
                const float d = ds4x_f16_to_f32(dh);
                float *y = dst + sb * QK_K;
                for (int n0 = 0; n0 < QK_K; n0 += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                        y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                        y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                        y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                    }
                    y  += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
                p += 210;
            }
            return 0;
        }
        case T_Q4_K: {
            if (n % QK_K) goto badlen;
            for (uint64_t sb = 0; sb < n / QK_K; sb++) {
                uint16_t dh, dmh; memcpy(&dh, p, 2); memcpy(&dmh, p + 2, 2);
                const float d = ds4x_f16_to_f32(dh), dmin = ds4x_f16_to_f32(dmh);
                const uint8_t *scales = p + 4;      /* 12 */
                const uint8_t *q = p + 16;          /* 128 */
                float *y = dst + sb * QK_K;
                int is = 0; uint8_t sc, mm;
                for (int j = 0; j < QK_K; j += 64) {
                    get_scale_min_k4(is + 0, scales, &sc, &mm);
                    const float d1 = d * sc, m1 = dmin * mm;
                    get_scale_min_k4(is + 1, scales, &sc, &mm);
                    const float d2 = d * sc, m2 = dmin * mm;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >>  4) - m2;
                    q += 32; is += 2;
                }
                p += 144;
            }
            return 0;
        }
        case T_Q5_K: {
            if (n % QK_K) goto badlen;
            for (uint64_t sb = 0; sb < n / QK_K; sb++) {
                uint16_t dh, dmh; memcpy(&dh, p, 2); memcpy(&dmh, p + 2, 2);
                const float d = ds4x_f16_to_f32(dh), dmin = ds4x_f16_to_f32(dmh);
                const uint8_t *scales = p + 4;      /* 12 */
                const uint8_t *qh = p + 16;         /* 32 high bits */
                const uint8_t *ql = p + 48;         /* 128 low 4 bits */
                float *y = dst + sb * QK_K;
                int is = 0; uint8_t sc, mm, u1 = 1, u2 = 2;
                for (int j = 0; j < QK_K; j += 64) {
                    get_scale_min_k4(is + 0, scales, &sc, &mm);
                    const float d1 = d * sc, m1 = dmin * mm;
                    get_scale_min_k4(is + 1, scales, &sc, &mm);
                    const float d2 = d * sc, m2 = dmin * mm;
                    for (int l = 0; l < 32; l++)
                        *y++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++)
                        *y++ = d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                    ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
                }
                p += 176;
            }
            return 0;
        }
        default:
            if (err) snprintf(err, errlen,
                "ggml type %u unsupported (have F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K); "
                "other quants pending DGX validation vs llama.cpp", t);
            return -1;
    }
badlen:
    if (err) snprintf(err, errlen, "type %u needs a multiple of 32 elements (got %llu)",
                      t, (unsigned long long)n);
    return -1;
}
