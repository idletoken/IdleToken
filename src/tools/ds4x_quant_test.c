/* ds4x_quant_test.c — CPU dequant formula unit tests (Q8_0 / Q4_0 / F16).
 * Blocks are hand-built with known scales + quants; dequant output is checked
 * against the spec formula directly. No fixture file needed. */
#include "idletoken_ds4x_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) printf("  [ok] %s\n", what);
    else { failures++; printf("  [FAIL] %s\n", what); }
}

/* Exact-f16 scales (chosen so d has no rounding). */
#define F16_0_5  0x3800u
#define F16_2_0  0x4000u

int main(void) {
    char err[128];
    float dst[64];

    /* ---- block geometry ---- */
    ok(ds4x_type_block_count(0) == 1 && ds4x_type_block_bytes(0) == 4, "F32 geom");
    ok(ds4x_type_block_count(1) == 1 && ds4x_type_block_bytes(1) == 2, "F16 geom");
    ok(ds4x_type_block_count(8) == 32 && ds4x_type_block_bytes(8) == 34, "Q8_0 geom");
    ok(ds4x_type_block_count(2) == 32 && ds4x_type_block_bytes(2) == 18, "Q4_0 geom");
    ok(ds4x_type_block_count(12) == 256 && ds4x_type_block_bytes(12) == 144, "Q4_K geom");
    ok(ds4x_type_block_count(13) == 256 && ds4x_type_block_bytes(13) == 176, "Q5_K geom");
    ok(ds4x_type_block_count(14) == 256 && ds4x_type_block_bytes(14) == 210, "Q6_K geom");
    ok(ds4x_type_block_count(30) == 1 && ds4x_type_block_bytes(30) == 2, "BF16 geom");
    ok(ds4x_type_block_count(10) == 256 && ds4x_type_block_bytes(10) == 84, "Q2_K geom");
    ok(ds4x_type_block_count(16) == 256 && ds4x_type_block_bytes(16) == 66, "IQ2_XXS geom");
    /* Q2_K and IQ2_XXS joined the set when the big MoE builds needed them
     * (GLM-5.2 is 225 IQ2_XXS tensors + 3 Q2_K). Q3_K stays out — the point of
     * this check is that the set is EXPLICIT, so adding a type has to be a
     * deliberate edit here rather than something that slips in. */
    ok(ds4x_type_supported(8) && ds4x_type_supported(2) &&
       ds4x_type_supported(10) && ds4x_type_supported(12) && ds4x_type_supported(13) &&
       ds4x_type_supported(14) && ds4x_type_supported(16) && ds4x_type_supported(30) &&
       !ds4x_type_supported(11),
       "supported set = F32/F16/BF16/Q8_0/Q4_0/Q2_K/Q4_K/Q5_K/Q6_K/IQ2_XXS, not Q3_K");

    /* ---- BF16 (ggml type 30) — the manifests' "full precision" tier.
     * bf16 is literally the high half of an fp32, so these are exact:
     *   0x3F80 = 1.0   0x4000 = 2.0   0xBF00 = -0.5   0x3F00 = 0.5
     * Picking values with zero low mantissa bits makes the expectation
     * independent of any rounding policy. */
    { uint16_t b[4] = { 0x3F80, 0x4000, 0xBF00, 0x3F00 };
      ok(ds4x_dequant_row(30, b, dst, 4, err, sizeof(err)) == 0 &&
         dst[0] == 1.0f && dst[1] == 2.0f && dst[2] == -0.5f && dst[3] == 0.5f,
         "BF16 dequant = fp32 with the low 16 mantissa bits zeroed"); }

    /* ---- F16 ---- */
    { uint16_t h[2] = { F16_0_5, F16_2_0 };
      ok(ds4x_dequant_row(1, h, dst, 2, err, sizeof(err)) == 0 &&
         dst[0] == 0.5f && dst[1] == 2.0f, "F16 dequant"); }

    /* ---- Q8_0: dst[i] = d * qs[i], d=0.5, qs[i]=i-16 ---- */
    {
        uint8_t blk[34];
        uint16_t d = F16_0_5; memcpy(blk, &d, 2);
        int8_t *qs = (int8_t *)(blk + 2);
        for (int i = 0; i < 32; i++) qs[i] = (int8_t)(i - 16);
        int r = ds4x_dequant_row(8, blk, dst, 32, err, sizeof(err));
        int good = (r == 0);
        for (int i = 0; i < 32 && good; i++)
            if (fabsf(dst[i] - 0.5f * (float)(i - 16)) > 1e-6f) good = 0;
        ok(good, "Q8_0 dequant = d·qs over a full block");
    }

    /* ---- Q4_0: low nibble = elem j, high nibble = elem j+16, val=nibble-8,
     *      d=2.0 ---- */
    {
        uint8_t blk[18];
        uint16_t d = F16_2_0; memcpy(blk, &d, 2);
        uint8_t *qs = blk + 2;
        for (int j = 0; j < 16; j++) {
            int lo = j % 16;              /* 0..15 */
            int hi = (j + 3) % 16;
            qs[j] = (uint8_t)((hi << 4) | lo);
        }
        int r = ds4x_dequant_row(2, blk, dst, 32, err, sizeof(err));
        int good = (r == 0);
        for (int j = 0; j < 16 && good; j++) {
            float elo = 2.0f * (float)((j % 16) - 8);
            float ehi = 2.0f * (float)(((j + 3) % 16) - 8);
            if (fabsf(dst[j] - elo) > 1e-6f || fabsf(dst[j + 16] - ehi) > 1e-6f) good = 0;
        }
        ok(good, "Q4_0 dequant = d·(nibble-8), low→j high→j+16");
    }

    /* ---- K-quant known-answer tests (bit-layout pinned independently of any
     * encoder; the numeric oracle vs real weights is llama.cpp on the DGX). A
     * 256-element super-block is hand-built with a super-scale d=1, and quants
     * chosen so a few outputs have closed-form values. ---- */
    static float dstk[256];
    { /* Q6_K: d=1, all scales=1; ql[0]=0x0F qh[0]=0x03 → q1=(15|48)-32=31 */
        uint8_t blk[210]; memset(blk, 0, sizeof(blk));
        uint8_t *ql = blk, *qh = blk + 128; int8_t *sc = (int8_t *)(blk + 192);
        uint16_t d = 0x3C00u; memcpy(blk + 208, &d, 2);      /* f16 1.0 */
        for (int i = 0; i < 16; i++) sc[i] = 1;
        ql[0] = 0x0F; qh[0] = 0x03;
        int r = ds4x_dequant_row(14, blk, dstk, 256, err, sizeof(err));
        ok(r == 0 && fabsf(dstk[0] - 31.0f) < 1e-6f &&
           fabsf(dstk[1] + 32.0f) < 1e-6f &&      /* ql[1]=qh[1]=0 → -32 */
           fabsf(dstk[64] + 32.0f) < 1e-6f,       /* q3=(0|0)-32 */
           "Q6_K dequant = d·sc·(6-bit − 32)");
    }
    { /* Q4_K: d=1, dmin=0; scales[0]=1 scales[1]=2; qs[0]=0x35 →
       * y[0]=d1·(0x5)=5, y[32]=d2·(0x3)=6 */
        uint8_t blk[144]; memset(blk, 0, sizeof(blk));
        uint16_t d = 0x3C00u, dm = 0x0000u; memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
        uint8_t *scales = blk + 4, *qs = blk + 16;
        scales[0] = 1; scales[1] = 2;
        qs[0] = 0x35;
        int r = ds4x_dequant_row(12, blk, dstk, 256, err, sizeof(err));
        ok(r == 0 && fabsf(dstk[0] - 5.0f) < 1e-6f && fabsf(dstk[32] - 6.0f) < 1e-6f,
           "Q4_K dequant = d·sc·nibble − dmin·min");
    }
    { /* Q4_K sub-block j>=4: exercises the packed 6-bit scale/min split, where
       * the scale's high bits come from scales[j-4] and the min's from
       * scales[j]. Sub-block 4 covers y[256/8*4 ...] = elements 128..159.
       * scales[8] low nibble = sc low bits, scales[4] top 2 bits = sc high;
       * scales[8] high nibble = mn low bits, scales[4+4=8]... per the formula:
       *   sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
       *   mn = (scales[8] >> 4)  | ((scales[4] >> 6) << 4)
       * Pick scales[0]=0x40 (high bits=1 → sc += 16), scales[4]=0x80
       * (high bits=2 → mn += 32), scales[8]=0x21 → sc=1+16=17, mn=2+32=34.
       * With d=1, dmin=1/32 and nibble 0 → y = 17*0 - 34/32 = -1.0625. */
        uint8_t blk[144]; memset(blk, 0, sizeof(blk));
        uint16_t d = 0x3C00u;              /* f16 1.0  */
        uint16_t dm = 0x2800u;             /* f16 1/32 (exp=10 → 2^-5) */
        memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
        uint8_t *scales = blk + 4;
        scales[0] = 0x40; scales[4] = 0x80; scales[8] = 0x21;
        int r = ds4x_dequant_row(12, blk, dstk, 256, err, sizeof(err));
        /* element 128 is the first of sub-block 4 (nibble 0) → -mn*dmin */
        ok(r == 0 && fabsf(dstk[128] - (-34.0f / 32.0f)) < 1e-5f,
           "Q4_K j>=4 packed scale/min split (sc from [j-4], mn from [j])");
    }
    { /* Q5_K: d=1, dmin=0; scales[0]=1; ql[0]=0x05 qh[0]=0x01 (u1 bit) →
       * y[0]=1·(5+16)=21 */
        uint8_t blk[176]; memset(blk, 0, sizeof(blk));
        uint16_t d = 0x3C00u, dm = 0x0000u; memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
        uint8_t *scales = blk + 4, *qh = blk + 16, *ql = blk + 48;
        scales[0] = 1; qh[0] = 0x01; ql[0] = 0x05;
        int r = ds4x_dequant_row(13, blk, dstk, 256, err, sizeof(err));
        ok(r == 0 && fabsf(dstk[0] - 21.0f) < 1e-6f,
           "Q5_K dequant = d·sc·(nibble + 16·qh) − dmin·min");
    }

    /* ---- Q2_K / IQ2_XXS known answers -------------------------------------
     * Values captured from ggml's own to_float on the DGX (the independent
     * oracle; see /tmp/deqtest and docs/multi-model-design.md §7b). Embedding
     * them pins the bit unpacking on hosts without llama.cpp — the Mac gate
     * cannot link ggml, and "it matched once on one machine" is not a gate. */
    {
        uint8_t blk[84];
        memset(blk, 0, sizeof blk);
        blk[0] = 0x21;            /* sub-block 0: scale 1, min 2 */
        blk[16] = 0x1B;           /* qs[0] = 0b00011011 → 3,2,1,0 by shift */
        blk[80] = 0x00; blk[81] = 0x3C;   /* d    = 1.0 (f16) */
        blk[82] = 0x00; blk[83] = 0x38;   /* dmin = 0.5 (f16) */
        float d2[256];
        const int r = ds4x_dequant_row(10, blk, d2, 256, err, sizeof(err));
        /* element 0: dl*(qs&3) - ml = 1*3 - 0.5*2 = 2.0 */
        ok(r == 0 && fabsf(d2[0] - 2.0f) < 1e-6f,
           "Q2_K dequant = d·scale·2bit − dmin·min");
    }
    {
        /* IQ2_XXS: grid index 0 with sign index 0 → all +grid[j], scale
         * db = d*(0.5 + 0)*0.25. iq2xxs_grid[0] = 0x0808080808080808, so every
         * magnitude is 8 → 8 * 1.0 * 0.5 * 0.25 = 1.0 */
        uint8_t blk[66];
        memset(blk, 0, sizeof blk);
        blk[0] = 0x00; blk[1] = 0x3C;    /* d = 1.0 */
        float d2[256];
        const int r = ds4x_dequant_row(16, blk, d2, 256, err, sizeof(err));
        ok(r == 0 && fabsf(d2[0] - 1.0f) < 1e-6f,
           "IQ2_XXS dequant = d·(0.5+sc)·0.25·grid·sign");
    }

    /* ---- errors ---- */
    ok(ds4x_dequant_row(11, dst, dst, 256, err, sizeof(err)) == -1,
       "still-unsupported type (Q3_K) refused");
    ok(ds4x_dequant_row(8, dst, dst, 30, err, sizeof(err)) == -1,
       "Q8_0 with non-block-multiple length refused");

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("DS4X_QUANT_TEST_FAIL\n"); return 1; }
    printf("DS4X_QUANT_TEST_OK\n");
    return 0;
}
