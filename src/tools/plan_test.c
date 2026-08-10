/* plan_test.c — unit tests for the planning core (mode decision + layer
 * split) and the model registry. Pure C, runs anywhere:
 *   cc -Wall -Wextra -std=c99 -Iinclude src/common/plan.c src/common/model.c \
 *      src/tools/plan_test.c -o plan_test && ./plan_test
 * Prints PLAN_TEST_OK on success (project principle: core algorithms must
 * have unit tests). */
#include "idletoken_plan.h"
#include "idletoken_advise.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GiB (1024ull * 1024 * 1024)

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) { printf("  [ok] %s\n", what); }
    else      { failures++; printf("  [FAIL] %s\n", what); }
}

static uint64_t g(double gib) { return (uint64_t)(gib * (double)GiB); }

/* Node memory, ALWAYS by field name — never positionally.
 *
 * 2026-08-08: `idletoken_node_mem` gained `ram_pinnable` BETWEEN `ram_usable`
 * and `unified` (the pinned-capacity model). Every `{v, r, u}` in this file
 * silently became `{v, r, .ram_pinnable = u}` with `unified` defaulting to 0
 * — so the "unified 50/50" case stopped exercising unified memory at all and
 * started reporting HYBRID (50+50=100) where it must refuse. The compiler did
 * say `-Wmissing-field-initializers` 29 times; nothing treated that as fatal.
 *
 * Naming the fields makes the next insertion a compile-time no-op instead of a
 * silent semantic shift. `ram_pinnable = 0` means "unknown/unconstrained",
 * which is what these planner cases intend. */
#define NM(v, r, u) { .vram_usable = (v), .ram_usable = (r), \
                      .ram_pinnable = 0, .unified = (u) }

int main(void) {
    char why[256];
    const idletoken_model_spec *M = idletoken_model_default();  /* DSv4-Flash */

    /* ---- registry --------------------------------------------------- */
    ok(strcmp(M->id, "deepseek-v4-flash") == 0 && M->available,
       "default model is DSv4-Flash and available");
    ok(idletoken_model_get("deepseek-v4-flash") == M, "lookup by id hits default");
    ok(idletoken_model_get("no-such-model") == NULL, "unknown id → NULL");
    ok(idletoken_model_get("glm-5.2") != NULL &&
       !idletoken_model_get("glm-5.2")->available,
       "GLM-5.2 registered but not yet available");

    /* DSv4 tracks the OFFICIAL 0731 release; the preview conversions are gone.
     * The default_gguf must carry the -0731 suffix, because that name is what
     * the weight fetcher and the topology matrix actually RESOLVE — a stale
     * default silently downloads the superseded preview weights. */
    ok(strstr(M->default_gguf, "-0731.gguf") != NULL,
       "DSv4 default GGUF is the official 0731 release");
    ok(M->n_variants == 2 &&
       idletoken_model_variant_get(M, NULL)->layer_weight_bytes == M->layer_weight_bytes,
       "DSv4 has a 2-entry precision menu, scalars mirror the default");
    ok(idletoken_model_variant_get(M, "IQ2_XXS+Q4_K-L37-42")->layer_weight_bytes >
       idletoken_model_variant_get(M, "IQ2_XXS+Q2_K")->layer_weight_bytes,
       "DSv4 Q4_K-mixed variant is heavier than the Q2 default");

    /* ---- small-model precision menu (small-model-design.md §3.2) ------ */
    {
        const idletoken_model_spec *Q = idletoken_model_get("qwen3-8b");
        ok(Q != NULL && Q->kv_kind == IDLETOKEN_KV_GQA && Q->n_variants == 5,
           "qwen3-8b registered as GQA with 5 precision variants");
        /* scalars mirror the default (Q4_K_M) variant */
        const idletoken_model_variant *def = idletoken_model_variant_get(Q, NULL);
        ok(def != NULL && strcmp(def->quant, "Q4_K_M") == 0 &&
           def->layer_weight_bytes == Q->layer_weight_bytes,
           "NULL quant → default variant, mirrors spec scalars");
        ok(strcmp(idletoken_model_variant_get(Q, "Q8_0")->quant, "Q8_0") == 0,
           "quant name resolves to its variant");
        ok(idletoken_model_variant_get(Q, "does-not-exist") == def,
           "unknown quant falls back to default variant");
        ok(idletoken_model_variant_get(Q, "BF16")->layer_weight_bytes >
           idletoken_model_variant_get(Q, "Q4_K_M")->layer_weight_bytes,
           "higher precision costs more weight bytes");
        /* glm-5.2 has no variant table. This used to point at DSv4, which
         * only passed because DSv4 happened to ship a single quant — it now
         * has a precision menu (Q2 default + Q4_K-mixed), so the assertion
         * needs a model that genuinely has no variants, not an incidental one. */
        ok(idletoken_model_variant_get(idletoken_model_get("glm-5.2"), "Q4_K_M") == NULL,
           "model with no variant table → NULL (caller uses scalars)");
        /* selected precision drives the whole-model size */
        uint64_t l8 = 0, s8 = 0, l4 = 0, s4 = 0;
        idletoken_model_weight_bytes(Q, "Q8_0", &l8, &s8);
        idletoken_model_weight_bytes(Q, "Q4_K_M", &l4, &s4);
        ok(l8 + s8 > l4 + s4, "weight_bytes(Q8_0) > weight_bytes(Q4_K_M)");
        /* GQA overhead behaves like MLA: grows with ctx and with layers */
        ok(idletoken_model_overhead(Q, 32768, 12) > idletoken_model_overhead(Q, 8192, 12),
           "GQA overhead grows with context");
        ok(idletoken_model_overhead(Q, 8192, 24) > idletoken_model_overhead(Q, 8192, 12),
           "GQA overhead grows with layers on node");
    }

    /* ---- needed() sanity -------------------------------------------- */
    ok(idletoken_needed_bytes(M, 8192, 4)  <  idletoken_needed_bytes(M, 32768, 4),
       "needed() grows with context tier");
    ok(idletoken_needed_bytes(M, 8192, 2)  <  idletoken_needed_bytes(M, 8192, 6),
       "needed() grows with node count");
    ok(idletoken_needed_bytes(M, 1048576, 4) > g(110),
       "1M-context 4-node need exceeds 110 GiB");
    ok(idletoken_needed_bytes(idletoken_model_get("glm-5.2"), 8192, 4) >
       2 * idletoken_needed_bytes(M, 8192, 4),
       "GLM-5.2 needs far more than DSv4 (240GB-class weights)");

    /* ---- mode: plenty of VRAM → GPU_ONLY ----------------------------- */
    {
        idletoken_node_mem nodes[] = {
            NM(g(108), g(108), 1),   /* DGX-like unified */
            NM(g(13.2), g(18.3), 0), /* 5060 Ti */
            NM(g(5.3), g(6.4), 0), /* 2070 */
        };
        idletoken_mode m = idletoken_mode_decide(M, nodes, 3, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_GPU_ONLY, "real-cluster tier1 → GPU_ONLY");
    }

    /* ---- unavailable model refuses regardless of resources ----------- */
    {
        idletoken_node_mem huge[] = { NM(g(1000), g(1000), 0) };
        idletoken_mode m = idletoken_mode_decide(idletoken_model_get("glm-5.2"),
                                           huge, 1, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_REFUSE, "unavailable model → refuse even with resources");
        ok(strstr(why, "not runnable") != NULL, "refusal names availability");
    }

    /* ---- unified pool must not be double counted --------------------- */
    {
        /* One unified node with 50+50: if double-counted (100) HYBRID would
         * pass for a 90 GiB need; correctly counted (50) it must refuse. */
        idletoken_node_mem uni[] = { NM(g(50), g(50), 1) };
        idletoken_mode m = idletoken_mode_decide(M, uni, 1, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_REFUSE, "unified 50/50 counts once → refuse");
    }

    /* ---- HYBRID: VRAM short, VRAM+RAM covers, all nodes ≥4G ---------- */
    {
        idletoken_node_mem nodes[] = {
            NM(g(24), g(48), 0),
            NM(g(16), g(32), 0),
            NM(g(8), g(24), 0),
        };  /* vram 48 < need; all 48+104=152 > need */
        idletoken_mode m = idletoken_mode_decide(M, nodes, 3, 32768, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_HYBRID, "VRAM-short RAM-rich trio → HYBRID");
    }

    /* ---- HYBRID blocked by <4G VRAM node ----------------------------- */
    {
        idletoken_node_mem nodes[] = {
            NM(g(24), g(64), 0),
            NM(g(2), g(64), 0),   /* below the 4 GiB floor */
        };
        idletoken_mode m = idletoken_mode_decide(M, nodes, 2, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_REFUSE, "HYBRID with a <4GiB-VRAM node → refuse");
        ok(strstr(why, "4 GiB") != NULL, "refusal reason names the floor");
    }

    /* ---- flat-out insufficient → refuse with actionable text --------- */
    {
        idletoken_node_mem tiny[] = { NM(g(8), g(16), 0) };
        idletoken_mode m = idletoken_mode_decide(M, tiny, 1, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_REFUSE, "single small node → refuse");
        ok(strstr(why, "Add nodes") != NULL, "refusal suggests remedies");
    }

    /* ---- layer split: coverage, floor, proportionality ---------------- */
    {
        idletoken_node_mem nodes[] = {
            NM(g(108), g(108), 1),
            NM(g(13.2), g(18.3), 0),
            NM(g(5.3), g(6.4), 0),
        };
        int c[3];
        ok(idletoken_plan_layers(M, nodes, 3, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0, "split(3, 43) succeeds");
        ok(c[0] + c[1] + c[2] == 43, "split covers exactly 43 layers");
        ok(c[0] >= 1 && c[1] >= 1 && c[2] >= 1, "every node gets >=1 layer");
        ok(c[0] > c[1] && c[1] >= c[2], "stronger nodes get more layers");
        printf("      real-cluster split: %d / %d / %d\n", c[0], c[1], c[2]);
    }
    {
        idletoken_node_mem one[] = { NM(g(100), g(100), 1) };
        int c[1];
        ok(idletoken_plan_layers(M, one, 1, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0 && c[0] == 43,
           "single node takes all 43");
    }
    {
        idletoken_node_mem eq[44];
        int c[44];
        for (int i = 0; i < 44; i++) { eq[i].vram_usable = g(8); eq[i].ram_usable = g(8); eq[i].ram_pinnable = 0; eq[i].unified = 0; }
        ok(idletoken_plan_layers(M, eq, 44, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == -1, "44 nodes on 43 layers rejected");
        ok(idletoken_plan_layers(M, eq, 43, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0, "43 nodes on 43 layers ok");
        int sum = 0, mn = 99, mx = 0;
        for (int i = 0; i < 43; i++) { sum += c[i]; if (c[i] < mn) mn = c[i]; if (c[i] > mx) mx = c[i]; }
        ok(sum == 43 && mn == 1 && mx == 1, "43 equal nodes → 1 layer each");
    }
    {
        /* Zero-probe guard: nodes reporting 0/0 must not crash or divide by 0. */
        idletoken_node_mem z[] = { NM(0, 0, 0), NM(g(10), g(10), 0) };
        int c[2];
        ok(idletoken_plan_layers(M, z, 2, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0 && c[0] + c[1] == 43 && c[0] >= 1,
           "zero-report node still gets a floor share");
    }

    {
        /* HYBRID split sizes by VRAM+RAM: a low-VRAM/high-RAM node should get
         * MORE layers than under GPU_ONLY (VRAM-only) sizing, so it can hold
         * more of the model and offload the overflow to its RAM. */
        idletoken_node_mem nodes[] = {
            NM(g(16), g(16), 0),   /* strong VRAM */
            NM(g(4), g(40), 0),   /* weak VRAM, ample RAM */
        };
        int cg[2], ch[2];
        idletoken_plan_layers(M, nodes, 2, 8192, cg, IDLETOKEN_MODE_GPU_ONLY);
        idletoken_plan_layers(M, nodes, 2, 8192, ch, IDLETOKEN_MODE_HYBRID);
        ok(cg[0]+cg[1]==43 && ch[0]+ch[1]==43, "hybrid split still covers 43");
        ok(ch[1] > cg[1], "HYBRID gives the ample-RAM node more layers than GPU_ONLY");
        printf("      GPU_ONLY %d/%d  vs  HYBRID %d/%d\n", cg[0],cg[1],ch[0],ch[1]);
    }

    {
        /* Capacity-cap repair (per-node capacity ceiling): on a tight 90 GiB cluster the
         * middle node's proportional share (9) exceeds what its 18 GiB can
         * absolutely hold (cap 8 = (18-2.5)/1.88); the excess layer must move
         * to the big node, which still has headroom. */
        idletoken_node_mem tight[] = {
            NM(g(60), g(8), 0),
            NM(g(18), g(8), 0),
            NM(g(12), g(8), 0),
        };
        int c[3];
        ok(idletoken_plan_layers(M, tight, 3, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0,
           "tight-cluster split succeeds");
        ok(c[0] + c[1] + c[2] == 43, "capped split still covers 43");
        ok(c[0] == 30 && c[1] == 8 && c[2] == 5,
           "over-cap node clamped, excess moved to the node with headroom");
        printf("      capped split: %d / %d / %d\n", c[0], c[1], c[2]);
    }
    {
        /* Graceful degradation: when even the caps can't fit 43 layers
         * (mode_decide would refuse such a cluster; synthetic here), the
         * repair must not fail or lose layers — proportional result stands. */
        idletoken_node_mem small[] = { NM(g(10), g(4), 0), NM(g(10), g(4), 0) };
        int c[2];
        ok(idletoken_plan_layers(M, small, 2, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0
               && c[0] + c[1] == 43,
           "infeasible caps degrade gracefully, still 43 layers");
    }

    /* ---- multi-model: MLA overhead formula ---------------------------- */
    {
        idletoken_model_spec mla = {
            .id = "test-mla", .label = "t", .backend = IDLETOKEN_BACKEND_DS4X,
            .available = 1, .n_layers = 48, .n_embd = 4096, .hc_streams = 1,
            .n_vocab = 100000,
            .layer_weight_bytes = g(96), .shared_weight_bytes = g(2),
            .ctx_max = 1048576, .split_boundary_multiple = 0,
            .kv_kind = IDLETOKEN_KV_MLA, .kv_bytes_per_token_layer = 1152,
            .overhead_base_bytes = g(2), .default_gguf = "t.gguf",
        };
        ok(idletoken_model_overhead(&mla, 32768, 12) >
           idletoken_model_overhead(&mla, 8192, 12),
           "MLA overhead grows with context");
        ok(idletoken_model_overhead(&mla, 32768, 24) >
           idletoken_model_overhead(&mla, 32768, 12),
           "MLA overhead grows with layers on node");
        /* 32K × 1152 B × 12 layers ≈ 0.42 GiB KV + 2 GiB base (+10%) */
        uint64_t oh = idletoken_model_overhead(&mla, 32768, 12);
        ok(oh > g(2.2) && oh < g(3.2), "MLA overhead lands in the expected band");
    }

    /* ---- multi-model: boundary-multiple snap (GLM's 4-layer groups) --- */
    {
        idletoken_model_spec grp = {
            .id = "test-grp", .label = "t", .backend = IDLETOKEN_BACKEND_DS4X,
            .available = 1, .n_layers = 16, .n_embd = 4096, .hc_streams = 1,
            .n_vocab = 100000,
            .layer_weight_bytes = g(16), .shared_weight_bytes = g(1),
            .ctx_max = 1048576, .split_boundary_multiple = 4,
            .kv_kind = IDLETOKEN_KV_MLA, .kv_bytes_per_token_layer = 1152,
            .overhead_base_bytes = g(1), .default_gguf = "t.gguf",
        };
        idletoken_node_mem nodes[] = {
            NM(g(40), g(16), 0),
            NM(g(40), g(16), 0),
            NM(g(40), g(16), 0),
        };
        int c[3];
        ok(idletoken_plan_layers(&grp, nodes, 3, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0
               && c[0] + c[1] + c[2] == 16,
           "grouped split covers all 16 layers");
        ok(c[0] % 4 == 0 && (c[0] + c[1]) % 4 == 0,
           "stage boundaries snap to multiples of 4");
        printf("      grouped split: %d / %d / %d\n", c[0], c[1], c[2]);
        /* Snap must never break the ≥1 floor: 5 nodes on 16 layers leaves
         * some stage un-snappable — sum must still hold. */
        idletoken_node_mem five[5];
        for (int i = 0; i < 5; i++) { five[i].vram_usable = g(40); five[i].ram_usable = g(16); five[i].ram_pinnable = 0; five[i].unified = 0; }
        int c5[5], sum5 = 0;
        ok(idletoken_plan_layers(&grp, five, 5, 8192, c5, IDLETOKEN_MODE_GPU_ONLY) == 0,
           "5-node grouped split succeeds");
        for (int i = 0; i < 5; i++) { sum5 += c5[i]; ok(c5[i] >= 1, "grouped split keeps the 1-layer floor"); }
        ok(sum5 == 16, "5-node grouped split still covers 16");
    }

    /* ---- capability advisor (G-ADVISE) ---------------------------------
     * The table users read must be the planner's own verdict. These pin the
     * two behaviours a wrong table would break: adding a machine can only
     * improve an answer, and lowering a machine's cap can only worsen it. */
    {
        idletoken_advice_row rows[IDLETOKEN_ADVISE_MAX_ROWS];

        /* A small laptop-class node: some models yes, the 80GB one no. */
        idletoken_node_mem small[] = { NM(g(5), g(6), 0) };
        int n1 = idletoken_advise(small, 1, rows, IDLETOKEN_ADVISE_MAX_ROWS);
        ok(n1 > 0, "advisor returns rows");
        int found_yes = 0, found_no_with_shortfall = 0;
        for (int i = 0; i < n1; i++) {
            if (!rows[i].unavailable && rows[i].mode != IDLETOKEN_MODE_REFUSE) found_yes = 1;
            if (!rows[i].unavailable && rows[i].mode == IDLETOKEN_MODE_REFUSE &&
                rows[i].shortfall > 0) found_no_with_shortfall = 1;
        }
        ok(found_yes, "a small node can still run something");
        ok(found_no_with_shortfall,
           "an out-of-reach model reports HOW MUCH memory is missing");

        /* Every verdict must agree with the planner asked directly. */
        int agreed = 1;
        for (int i = 0; i < n1 && agreed; i++) {
            if (rows[i].unavailable || rows[i].mode == IDLETOKEN_MODE_REFUSE) continue;
            const idletoken_model_spec *m = idletoken_model_get(rows[i].model_id);
            idletoken_mode direct = idletoken_mode_decide_quant(
                m, rows[i].quant, small, 1, rows[i].max_ctx, NULL, NULL, 0);
            if (direct != rows[i].mode) agreed = 0;
        }
        ok(agreed, "advisor verdicts match idletoken_mode_decide_quant exactly");

        /* More machines never make an answer worse. */
        idletoken_node_mem many[4];
        for (int i = 0; i < 4; i++) { many[i].vram_usable = g(5); many[i].ram_usable = g(6); many[i].ram_pinnable = 0; many[i].unified = 0; }
        idletoken_advice_row rows4[IDLETOKEN_ADVISE_MAX_ROWS];
        int n4 = idletoken_advise(many, 4, rows4, IDLETOKEN_ADVISE_MAX_ROWS);
        ok(n4 == n1, "same row count regardless of cluster size");
        int improved = 0, regressed = 0;
        for (int i = 0; i < n1; i++) {
            if (rows[i].unavailable) continue;
            if (rows4[i].mode != IDLETOKEN_MODE_REFUSE && rows[i].mode == IDLETOKEN_MODE_REFUSE) improved = 1;
            if (rows4[i].mode == IDLETOKEN_MODE_REFUSE && rows[i].mode != IDLETOKEN_MODE_REFUSE) regressed = 1;
        }
        ok(!regressed, "adding machines never turns a yes into a no");
        ok(improved, "adding machines turns at least one no into a yes");

        /* Halving the cap must not increase the context tier anywhere. */
        idletoken_node_mem tight[] = { NM(g(2), g(3), 0) };
        idletoken_advice_row rowsT[IDLETOKEN_ADVISE_MAX_ROWS];
        int nT = idletoken_advise(tight, 1, rowsT, IDLETOKEN_ADVISE_MAX_ROWS);
        int ctx_never_grew = 1;
        for (int i = 0; i < nT && i < n1; i++)
            if (rowsT[i].max_ctx > rows[i].max_ctx) ctx_never_grew = 0;
        ok(ctx_never_grew, "a smaller machine never reports a LARGER context");

        /* The JSON the client consumes must fit and stay well-formed. */
        char buf[16384];
        int len = idletoken_advise_json(rows, n1, 1, buf, sizeof buf);
        ok(len > 0 && buf[0] == '{' && buf[len - 1] == '}',
           "capability JSON is complete");
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("PLAN_TEST_FAIL\n"); return 1; }
    printf("PLAN_TEST_OK\n");
    return 0;
}
