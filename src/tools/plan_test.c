/* plan_test.c — unit tests for the planning core (mode decision + layer
 * split) and the model registry. Pure C, runs anywhere:
 *   cc -Wall -Wextra -std=c99 -Iinclude src/common/plan.c src/common/model.c \
 *      src/tools/plan_test.c -o plan_test && ./plan_test
 * Prints PLAN_TEST_OK on success (project principle: core algorithms must
 * have unit tests). */
#include "idletoken_plan.h"
#include "idletoken_advise.h"
#include "idletoken_modelsize.h"

#include <limits.h>
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

/* A file of exactly `bytes`, without writing them: seek past the end and put
 * one byte, which leaves a hole on every filesystem the tests run on (APFS,
 * ext4, tmpfs — verified 0 blocks used for a 15.6 GiB file). The sizes matter
 * because the resolver's whole job is reading them; the contents never do.
 *
 * Windows is excluded: MinGW's `long` is 32-bit (so fseek cannot reach these
 * offsets) and NTFS would zero-fill rather than leave a hole — a unit test that
 * writes 15 GiB is not a unit test. The callers skip loudly. */
#if !defined(_WIN32) && LONG_MAX > 2147483647L
#define HAVE_SPARSE_FIXTURES 1
static int make_sized_file(const char *path, uint64_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (bytes > 0) {
        if (fseek(f, (long)(bytes - 1), SEEK_SET) != 0) { fclose(f); return -1; }
        if (fputc(0, f) == EOF) { fclose(f); return -1; }
    }
    return fclose(f) == 0 ? 0 : -1;
}
#endif

int main(void) {
    char why[256];
    const idletoken_model_spec *M = idletoken_model_default();  /* DSv4-Flash */

    /* ---- registry --------------------------------------------------- */
    ok(strcmp(M->id, "deepseek-v4-flash") == 0 && M->available,
       "default model is DSv4-Flash and available");
    ok(idletoken_model_get("deepseek-v4-flash") == M, "lookup by id hits default");
    ok(idletoken_model_get("no-such-model") == NULL, "unknown id → NULL");
    /* Availability restored on 2026-08-20: GLM-5.2 / Kimi K2.5 / DeepSeek V4
     * Pro are cluster-tier models whose architectures the pinned llama.cpp
     * reads, so they are registered AND available (the client filters the
     * picker by this flag). Kimi K3 was dropped from the curated list entirely:
     * its 2.8T KDA+AttnRes architecture is not in the pinned engine — upstream
     * support is still an unmerged PR — so the client cannot load it at all. */
    ok(idletoken_model_get("glm-5.2") != NULL &&
       idletoken_model_get("glm-5.2")->available,
       "GLM-5.2 registered and available");
    ok(idletoken_model_get("kimi-k2.5") != NULL &&
       idletoken_model_get("kimi-k2.5")->available &&
       idletoken_model_get("deepseek-v4-pro") != NULL &&
       idletoken_model_get("deepseek-v4-pro")->available,
       "Kimi K2.5 / DSv4-Pro registered and available");
    ok(idletoken_model_get("kimi-k3") == NULL,
       "Kimi K3 dropped from the curated list");

    /* DSv4-Flash tracks the OFFICIAL release repo (unsloth/DeepSeek-V4-Flash-GGUF)
     * since 2026-08-15, with every precision it publishes — the two hand-mixed
     * preview conversions are gone. Neither the file name nor the menu length is
     * asserted: both are upstream's to change, and a test that must be edited
     * whenever upstream publishes a quant is not evidence of anything. What must
     * hold is that the default resolves to a real, measured entry. */
    ok(M->n_variants > 0 &&
       idletoken_model_variant_get(M, NULL)->layer_weight_bytes == M->layer_weight_bytes,
       "DSv4 has a precision menu, scalars mirror the default");
    ok(strstr(M->default_gguf, ".gguf") != NULL && M->layer_weight_bytes > 0,
       "DSv4 default GGUF is named and measured");

    /* ---- small-model precision menu (small-model-design.md §3.2) ------ */
    {
        const idletoken_model_spec *Q = idletoken_model_get("qwen3-8b");
        /* The COUNT is not asserted any more (2026-08-15): the menu is now
         * whatever the upstream repo publishes, measured by
         * scripts/manifest_fill_variants.py, so a number here would have to be
         * edited every time upstream adds a quant — and a test that must be
         * edited to stay green stops being evidence. What must hold is that the
         * model is GQA and its menu is not empty. */
        ok(Q != NULL && Q->kv_kind == IDLETOKEN_KV_GQA && Q->n_variants > 0,
           "qwen3-8b registered as GQA with a precision menu");
        /* scalars mirror the default (Q4_K_M) variant */
        const idletoken_model_variant *def = idletoken_model_variant_get(Q, NULL);
        ok(def != NULL && strcmp(def->quant, "Q4_K_M") == 0 &&
           def->layer_weight_bytes == Q->layer_weight_bytes,
           "NULL quant → default variant, mirrors spec scalars");
        ok(strcmp(idletoken_model_variant_get(Q, "Q8_0")->quant, "Q8_0") == 0,
           "quant name resolves to its variant");
        ok(idletoken_model_variant_get(Q, "does-not-exist") == def,
           "unknown quant falls back to default variant");
        ok(idletoken_model_variant_get(Q, "Q8_0")->layer_weight_bytes >
           idletoken_model_variant_get(Q, "Q4_K_M")->layer_weight_bytes,
           "higher precision costs more weight bytes");
        /* Every model's menu, not just this one: precision lists are ordered
         * cheapest-first and each step must genuinely cost more bytes. The UI
         * shows them in registry order, so an out-of-order or duplicated entry
         * is a menu that lies about which option is smaller. This also catches
         * the failure mode that prompted the 2026-08-11 sweep — invented sizes
         * copied in without measuring, which do not have to come out ordered. */
        int menu_ok = 1;
        for (int mi = 0; mi < idletoken_model_count(); mi++) {
            const idletoken_model_spec *m = idletoken_model_at(mi);
            for (int vi = 1; vi < m->n_variants; vi++) {
                uint64_t prev = m->variants[vi - 1].layer_weight_bytes +
                                m->variants[vi - 1].shared_weight_bytes;
                uint64_t cur  = m->variants[vi].layer_weight_bytes +
                                m->variants[vi].shared_weight_bytes;
                if (cur < prev) {
                    fprintf(stderr, "  %s: %s (%llu B) is SMALLER than %s (%llu B)\n",
                            m->id, m->variants[vi].quant, (unsigned long long)cur,
                            m->variants[vi - 1].quant, (unsigned long long)prev);
                    menu_ok = 0;
                }
            }
        }
        ok(menu_ok, "every precision menu is ordered cheapest-to-largest");

        /* glm-5.2 has no variant table. This used to point at DSv4, which
         * only passed because DSv4 happened to ship a single quant — it now
         * has a precision menu (Q2 default + Q4_K-mixed), so the assertion
         * needs a model that genuinely has no variants, not an incidental one. */
        /* Every shipped model now carries a variant table, so the
         * no-table path is exercised against a local spec rather than by
         * keeping a real model's menu empty. */
        idletoken_model_spec bare = { .id = "bare", .label = "bare",
                                      .variants = NULL, .n_variants = 0 };
        ok(idletoken_model_variant_get(&bare, "Q4_K_M") == NULL,
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
            NM(g(108), g(108), 1),   /* unified memory   */
            NM(g(13.2), g(18.3), 0), /* 5060 Ti */
            NM(g(5.3), g(6.4), 0), /* 2070 */
        };
        idletoken_mode m = idletoken_mode_decide(M, nodes, 3, 8192, why, sizeof(why));
        ok(m == IDLETOKEN_MODE_GPU_ONLY, "real-cluster tier1 → GPU_ONLY");
    }

    /* ---- unavailable model refuses regardless of resources ----------- */
    {
        idletoken_node_mem huge[] = { NM(g(1000), g(1000), 0) };
        /* A spec that is registered but not runnable. Built locally rather
         * than picked from the catalogue: which models carry available=0 is a
         * curation decision that moves, and this rule must keep being tested
         * regardless of what the catalogue happens to hold. */
        idletoken_model_spec unavailable = *idletoken_model_get("glm-5.2");
        unavailable.available = 0;
        idletoken_mode m = idletoken_mode_decide(&unavailable,
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
        /* Attribution (plan D3): the numbers are each machine's own
         * declaration, so a refusal must say WHICH machine — "a node has only
         * 2 GiB" sends the owner of five machines looking through all five.
         * Unnamed rows fall back to an index, which is still actionable. */
        ok(strstr(why, "node #2") != NULL, "unnamed rows are blamed by position");

        idletoken_node_mem named[] = {
            { .vram_usable = g(24), .ram_usable = g(64), .ram_pinnable = 0,
              .unified = 0, .label = "studio-mac" },
            { .vram_usable = g(2),  .ram_usable = g(64), .ram_pinnable = 0,
              .unified = 0, .label = "gaming-pc" },
        };
        idletoken_mode mn = idletoken_mode_decide(M, named, 2, 8192, why, sizeof(why));
        ok(mn == IDLETOKEN_MODE_REFUSE, "same case, named rows → still refuse");
        ok(strstr(why, "gaming-pc") != NULL, "refusal names the machine at fault");
        /* Control: it must name the OFFENDER, not just any machine it has. A
         * message that always printed the first row would pass the check above. */
        ok(strstr(why, "studio-mac") == NULL, "and does not blame the healthy one");
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
        idletoken_node_mem eq[44] = {0};   /* zero-init: see the struct's `label` note */
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
        /* Capacity-cap repair (per-node capacity ceiling): a node whose
         * proportional share exceeds what it can physically hold must be
         * clamped, and the excess must land on a node with headroom.
         *
         * The cluster is sized FROM THE MODEL rather than written as three
         * fixed GiB numbers (2026-08-15). The old literals (60/18/12 GiB,
         * expecting 30/8/5) encoded the preview build's per-layer weight; when
         * DSv4-Flash moved to the official release the layers got lighter, the
         * middle node stopped being over its cap, and the test failed while
         * reporting a perfectly correct plan. A scenario that must be
         * hand-retuned every time a model is remeasured tests the catalogue,
         * not the planner. */
        const uint64_t per_layer =
            (M->layer_weight_bytes + M->shared_weight_bytes) / (uint64_t)M->n_layers;
        /* Middle node: room for ~6 layers, well under its proportional share of
         * a 43-layer model, so the clamp has to fire. The big node is given
         * enough headroom to absorb what the middle one cannot take. */
        const uint64_t mid_bytes = per_layer * 6 + (uint64_t)(2.5 * (double)GiB);
        idletoken_node_mem tight[] = {
            NM(per_layer * 40 + (uint64_t)(3.0 * (double)GiB), g(8), 0),
            NM(mid_bytes, g(8), 0),
            NM(per_layer * 8 + (uint64_t)(2.5 * (double)GiB), g(8), 0),
        };
        int c[3];
        ok(idletoken_plan_layers(M, tight, 3, 8192, c, IDLETOKEN_MODE_GPU_ONLY) == 0,
           "tight-cluster split succeeds");
        ok(c[0] + c[1] + c[2] == M->n_layers, "capped split still covers every layer");
        /* The invariant, not a literal split: the middle node is held at or
         * below what it can hold, and the biggest node took the rest. */
        ok(c[1] <= 7 && c[0] > c[1] && c[0] > c[2],
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
        idletoken_node_mem five[5] = {0};
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
        idletoken_node_mem many[4] = {0};
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
        ok(!improved, "extra machines do NOT unlock a single-node model");

        /* ...but they must still unlock a CLUSTER model. This used to be one
         * assertion ("adding machines turns some no into a yes") over the whole
         * table; since single-node models stopped pooling memory, the only
         * model that can flip is a cluster one, and it takes a roster big
         * enough to hold 81 GiB. Asserting it on the specific model keeps the
         * check honest instead of letting it pass on whatever happens to flip. */
        idletoken_node_mem big[8] = {0};
        for (int i = 0; i < 8; i++) { big[i].vram_usable = g(12); big[i].ram_usable = g(12); big[i].ram_pinnable = 0; big[i].unified = 0; }
        idletoken_advice_row rows8[IDLETOKEN_ADVISE_MAX_ROWS];
        int n8 = idletoken_advise(big, 8, rows8, IDLETOKEN_ADVISE_MAX_ROWS);
        int dsv4_flipped = 0;
        for (int i = 0; i < n8; i++)
            if (!strcmp(rows8[i].model_id, "deepseek-v4-flash") &&
                rows8[i].mode != IDLETOKEN_MODE_REFUSE) dsv4_flipped = 1;
        ok(dsv4_flipped, "a big enough cluster unlocks the cluster model");

        /* The flag the UI reads, and the invariant behind it: a single-node
         * model's verdict must not depend on how many machines are present. */
        int flag_ok = 1, verdict_stable = 1;
        for (int i = 0; i < n1; i++) {
            const idletoken_model_spec *m = idletoken_model_get(rows[i].model_id);
            if (rows[i].single_node != !idletoken_model_may_cluster(m, NULL, 0)) flag_ok = 0;
            if (rows[i].single_node &&
                (rows4[i].mode != rows[i].mode || rows4[i].max_ctx != rows[i].max_ctx))
                verdict_stable = 0;
        }
        ok(flag_ok, "single_node flag matches the registry's deployment field");
        ok(verdict_stable, "a single-node model's verdict ignores the roster size");

        /* Halving the cap must not increase the context tier anywhere. */
        idletoken_node_mem tight[] = { NM(g(2), g(3), 0) };
        idletoken_advice_row rowsT[IDLETOKEN_ADVISE_MAX_ROWS];
        int nT = idletoken_advise(tight, 1, rowsT, IDLETOKEN_ADVISE_MAX_ROWS);
        int ctx_never_grew = 1;
        for (int i = 0; i < nT && i < n1; i++)
            if (rowsT[i].max_ctx > rows[i].max_ctx) ctx_never_grew = 0;
        ok(ctx_never_grew, "a smaller machine never reports a LARGER context");

        /* The JSON the client consumes must fit and stay well-formed, in a
         * buffer sized the way its callers size theirs. */
        static char buf[262144];
        int len = idletoken_advise_json(rows, n1, 1, buf,
                                        idletoken_advise_json_cap(n1));
        ok(len > 0 && buf[0] == '{' && buf[len - 1] == '}',
           "capability JSON is complete");
        ok(idletoken_advise_json_cap(n1) <= sizeof buf,
           "...and the sizing rule stays inside a sane allocation");

        /* The whole curated catalogue really does need more than the 16 KiB
         * the coordinator's /idletoken/v1/capability used to hold, and this is
         * the failure mode: not a truncated table, no table (-1 → HTTP 500).
         * Asserting it here keeps the fix from being "a bigger constant that
         * happens to be enough today" — if the rule ever stops covering the
         * catalogue, this pair disagrees. */
        ok(n1 >= 100 && len > 16384,
           "the full curated catalogue does not fit 16 KiB (the 500 that was)");
        ok(idletoken_advise_json(rows, n1, 1, buf, 16384) < 0,
           "...and a too-small buffer reports failure rather than truncating");
    }

    /* ==== llama.cpp-engine scheduling (v2 WS-B2) =========================
     * Four quadrants (fits/doesn't × single/multi) + layer-0 pinning +
     * refusal message quality, per the WS-B2 acceptance criterion. */
    {
        /* An 8 GiB model with a known KV shape (64 KiB/token — big enough
         * that ctx sizing is visible in the numbers). */
        idletoken_llm_model_size small_m = {
            .total_bytes = g(8), .n_layers = 32, .kv_bytes_per_token = 65536,
        };
        idletoken_llm_model_size big_m = {
            .total_bytes = g(80), .n_layers = 43, .kv_bytes_per_token = 65536,
        };
        idletoken_llama_plan p;

        /* Q1: fits one machine, several present → SINGLE (invariant #5). */
        idletoken_node_mem trio[] = {
            NM(g(12), g(20), 0),   /* coordinator */
            NM(g(24), g(24), 0),
            NM(g(6), g(10), 0),
        };
        ok(idletoken_plan_llamacpp(&small_m, trio, 3, 0, 32768, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_SINGLE,
           "fits-one + multi roster → SINGLE, never cluster");
        ok(p.single_node == 0 && p.layer0_node == 0,
           "SINGLE prefers the coordinator when the coordinator fits");
        ok(strstr(p.why, "SINGLE") != NULL, "single decision explains itself");

        /* Q1b: escape hatch (acceptance vehicle) flips the same input to
         * CLUSTER — without it every cross-machine gate loses its vehicle. */
        ok(idletoken_plan_llamacpp(&small_m, trio, 3, 0, 32768, 1, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_CLUSTER,
           "allow_small_cluster=1 forces CLUSTER on a fitting model");
        ok(p.order[0] == 0 && p.layer0_node == 0,
           "escape-hatch cluster still pins layer 0 to the coordinator");
        ok(strstr(p.why, "FITS") != NULL && strstr(p.why, "ALLOW_SMALL_CLUSTER") != NULL,
           "forced cluster says the model fits and names the override");
        ok(strstr(p.why, "exceeds") == NULL,
           "forced cluster does not claim need exceeds usable (it does not)");

        /* Q2: fits one machine, single-node roster → SINGLE (trivial). */
        idletoken_node_mem one[] = { NM(g(12), g(20), 0) };
        ok(idletoken_plan_llamacpp(&small_m, one, 1, 0, 32768, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_SINGLE && p.single_node == 0,
           "single roster + fitting model → SINGLE");

        /* Q3: doesn't fit any single machine, fits the roster → CLUSTER.
         *
         * The VRAM figures carry the roster since 2026-08-20 (T16): on the
         * cluster path a node is handed layers in the pool its engine can
         * address, and on a discrete card that is video memory alone. The old
         * fixture (12/24/6/4 GiB of VRAM behind 108 GiB of vram+ram) described
         * a cluster with 46 GiB of card for an 80 GiB model — a plan the
         * engine cannot execute, which is the bug this rewrite is about.
         * 96 GiB of VRAM over four nodes, none of them able to hold the model
         * alone, keeps the case the test was written for. */
        idletoken_node_mem four[] = {
            NM(g(20), g(20), 0),   /* coordinator: 40 usable, 20 addressable */
            NM(g(32), g(16), 0),   /* strongest:   48 usable, 32 addressable */
            NM(g(24), g(8),  0),   /* 32 usable, 24 addressable */
            NM(g(20), g(4),  0),   /* 24 usable, 20 addressable */
        };                          /* 96 GiB of VRAM vs 82 GiB + 4 x overhead */
        ok(idletoken_plan_llamacpp(&big_m, four, 4, 0, 32768, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_CLUSTER && p.n_nodes == 4,
           "doesn't-fit-one + big roster → CLUSTER");
        ok(p.order[0] == 0 && p.layer0_node == 0,
           "cluster order starts at the coordinator = layer 0 pinned there");
        ok(p.order[1] == 1,
           "remaining nodes ordered strongest-first after the coordinator");
        {
            double s = 0;
            int positive = 1;
            for (int i = 0; i < p.n_nodes; i++) {
                s += p.tensor_split[i];
                if (p.tensor_split[i] <= 0) positive = 0;
            }
            ok(positive && s > 0.999 && s < 1.001,
               "tensor-split ratios are positive and sum to 1");
            ok(p.tensor_split[0] >= 1.0 / (double)big_m.n_layers - 1e-9,
               "coordinator's slice covers at least one layer (layer 0)");
        }

        /* Q3b: strong workers cannot rescue a memoryless coordinator —
         * layer 0 + embedding may not leave it (privacy invariant #1). */
        idletoken_node_mem headless[] = {
            NM(0, 0, 0),           /* coordinator with no usable memory */
            NM(g(64), g(64), 0),
            NM(g(64), g(64), 0),
        };
        ok(idletoken_plan_llamacpp(&big_m, headless, 3, 0, 32768, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_REFUSE,
           "coordinator without compute memory → refuse, workers don't count");
        ok(strstr(p.why, "coordinator") != NULL &&
               strstr(p.why, "layer 0") != NULL,
           "headless-coordinator refusal names the invariant");

        /* Q4: doesn't fit anywhere → REFUSE with need/have/shortfall in GiB.
         * Since 2026-08-16 this refusal is the OVER-SUBSCRIPTION one, not a
         * capacity one: 80 GiB of weights against ~22 GiB of cache is 3.6x,
         * past the 2x that was measured to still be usable. It would start —
         * and re-read most of the model per token, which is not a product. */
        idletoken_node_mem tiny2[] = { NM(g(6), g(10), 0), NM(g(4), g(8), 0) };
        ok(idletoken_plan_llamacpp(&big_m, tiny2, 2, 0, 32768, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_REFUSE,
           "roster too small in total → REFUSE");
        ok(strstr(p.why, "GiB") != NULL && strstr(p.why, "short") != NULL &&
               strstr(p.why, "Add machines") != NULL,
           "refusal names the numbers and suggests remedies");

        /* ---- hard vs soft need (2026-08-16 measurement) ------------------
         * Weights are mmap'd and evictable, so they are NOT a capacity bound;
         * only KV + per-node engine overhead must be resident. A roster that
         * covers the resident part and can cache the working set runs, and
         * runs FAST — even though the weights alone exceed no single node. */
        {
            /* big_m: 80 GiB weights, 43 layers, 64 KiB/token KV.
             * hard @32K = 2 GiB KV + 2 x (768 MiB + 1.25 GiB) ~ 6 GiB. */
            const uint64_t hard = idletoken_llama_hard_need(&big_m, 32768, 2);
            ok(hard < g(8) && hard > g(4),
               "hard need excludes the weights (KV + per-node overhead only)");
            ok(idletoken_llama_working_set(&big_m) == big_m.total_bytes,
               "dense model: working set is the whole file");

            /* Same 80 GiB model, two roomy nodes: the weights exceed either
             * node, the cluster caches them, so CLUSTER and NOT slow.
             * (VRAM raised from 24 to 44 GiB per node on 2026-08-20 — see the
             * Q3 note: two 24 GiB cards cannot hold 82 GiB of layers, so the
             * old fixture asserted a plan the engine cannot run.) */
            idletoken_node_mem roomy2[] = { NM(g(44), g(30), 0), NM(g(44), g(30), 0) };
            ok(idletoken_plan_llamacpp(&big_m, roomy2, 2, 0, 32768, 0, &p) == 0 &&
                   p.kind == IDLETOKEN_LLPLAN_CLUSTER && p.working_set_fits == 1,
               "weights over one node but cached by the cluster → CLUSTER, full speed");

            /* Between the two: enough to be resident, not enough to cache the
             * working set, but inside the measured 2x — runs, labelled slow.
             *
             * UNIFIED nodes since 2026-08-20, and that is the whole point: the
             * slow tier is the page-cache story (2026-08-16), and page cache is
             * something only a node whose engine addresses the machine's one
             * pool can have. The discrete twin of this fixture is asserted
             * below — it does not get a slow tier, it gets a refusal. */
            idletoken_node_mem mid2[] = { NM(g(24), g(24), 1), NM(g(24), g(24), 1) };
            ok(idletoken_plan_llamacpp(&big_m, mid2, 2, 0, 32768, 0, &p) == 0 &&
                   p.kind == IDLETOKEN_LLPLAN_CLUSTER && p.working_set_fits == 0,
               "within 2x over-subscription → runs, marked slow");
            ok(strstr(p.why, "slow") != NULL && strstr(p.why, "stream") != NULL,
               "the slow plan says so, and says why");

            /* MoE: only the consulted experts are hot, so the same file size
             * needs far less cache. 256 experts, 8 used → the working set must
             * come out well under the file. */
            idletoken_llm_model_size moe = big_m;
            moe.n_expert = 256; moe.n_expert_used = 8;
            ok(idletoken_llama_working_set(&moe) < big_m.total_bytes / 2,
               "MoE working set is a fraction of the file, not the whole file");
        }

        /* Unified pool counted once (Apple Silicon / Grace): 50 GiB vram aliasing
         * 50 GiB ram is 50, not 100.
         *
         * Asserted through the SPEED verdict since 2026-08-16 — the old form
         * asserted a refusal, and weights beyond memory are no longer a
         * refusal. It still catches the bug it was written for: counted twice,
         * the 80 GiB working set would fit in "96 GiB" of cache and the plan
         * would claim full speed. */
        idletoken_node_mem uni1[] = { NM(g(50), g(50), 1) };
        ok(idletoken_plan_llamacpp(&big_m, uni1, 1, 0, 32768, 0, &p) == 0 &&
               p.working_set_fits == 0,
           "unified 50/50 counts once → the 80 GiB model cannot be cached");

        /* ---- ctx sizing (idletoken_llama_fit_ctx) ---------------------- */
        /* Overhead is the calibrated 768 MiB + weights/64 (2026-08-15):
         * for 8 GiB weights that is 896 MiB. */
        /* Inside 12 GiB → ~3.1 GiB KV budget → well past the 32K ask. */
        ok(idletoken_llama_fit_ctx(g(12), &small_m, 32768, 16384) == 32768,
           "roomy machine grants the full 32K ask");
        /* 10 GiB usable → 10 − 8 − 0.875 = 1.125 GiB KV budget
         * = 18432 tokens at 64 KiB/token (already 1024-aligned). */
        ok(idletoken_llama_fit_ctx(g(10), &small_m, 32768, 16384) == 18432,
           "tight machine grants a smaller (floor-respecting) context");
        /* 9.5 GiB usable → 0.5 GiB KV = 8192 tokens < 16K floor → 0 (refuse
         * loudly rather than silently under-serve Claude Code). */
        ok(idletoken_llama_fit_ctx((uint64_t)(9.5 * (double)GiB), &small_m,
                                   32768, 16384) == 0,
           "below the ctx floor → 0 (caller must refuse, not shrink silently)");
        /* Weights alone exceed the machine → 0. */
        ok(idletoken_llama_fit_ctx(g(6), &small_m, 32768, 16384) == 0,
           "weights don't fit → 0");
        /* Unknown KV shape grants the ask unchanged (never invents a cost). */
        idletoken_llm_model_size nokv = { .total_bytes = g(8), .n_layers = 32,
                                          .kv_bytes_per_token = 0 };
        ok(idletoken_llama_fit_ctx(g(12), &nokv, 32768, 16384) == 32768,
           "unknown KV shape passes the ask through");

        /* usable-metric helper: unified counts once, discrete sums. */
        idletoken_node_mem um = NM(g(10), g(12), 1);
        idletoken_node_mem dm = NM(g(10), g(12), 0);
        ok(idletoken_llama_node_usable(&um) == g(12) &&
               idletoken_llama_node_usable(&dm) == g(22),
           "usable metric: unified=max(once), discrete=vram+ram");

        /* ---- sequence slots (idletoken_llama_seq_slots, §4.5b) ----------
         *
         * These cases are UNIFIED-memory nodes, and they are written that way
         * on purpose: the KV budget is the whole usable pool exactly when the
         * machine has one pool. The discrete cases below are where the two
         * pools differ, and where this file used to say nothing. */
        /* small_m at 8K: kv_per_seq = 64 KiB x 8192 = 0.5 GiB. Overhead for
         * 8 GiB of weights is 768 MiB + 128 MiB = 896 MiB. */
        /* 12 GiB − 8 − 0.875 = 3.125 GiB → 6 slots, capped to 4. */
        idletoken_node_mem u12 = NM(g(12), g(12), 1);
        idletoken_node_mem u10 = NM(g(10), g(10), 1);
        idletoken_node_mem u6  = NM(g(6),  g(6),  1);
        idletoken_node_mem u64 = NM(g(64), g(64), 1);
        ok(idletoken_llama_seq_slots(&u12, &small_m, 8192, 1.0, 4) == 4,
           "roomy machine: slots capped, not unbounded");
        /* Same machine, 32K context → kv_per_seq = 2 GiB → 1 slot.
         * The whole point of the per-ctxClass table: one machine is 4-wide at
         * 8K and 1-wide at 32K. */
        ok(idletoken_llama_seq_slots(&u12, &small_m, 32768, 1.0, 4) == 1,
           "same machine, longer context → fewer slots");
        /* 10 GiB − 8.875 = 1.125 GiB → 2 slots at 8K. */
        ok(idletoken_llama_seq_slots(&u10, &small_m, 8192, 1.0, 4) == 2,
           "tight machine: the arithmetic, not the cap, decides");
        /* Weights alone do not fit → still 1, never 0: one slot is what the
         * engine does anyway, and 0 would tell the platform "cannot serve". */
        ok(idletoken_llama_seq_slots(&u6, &small_m, 8192, 1.0, 4) == 1,
           "no room at all → 1 slot, never 0");
        /* Unknown KV cost → 1. Deliberately the OPPOSITE direction from
         * fit_ctx's "grant the ask": not knowing is a reason to open fewer
         * slots, and a reason not to refuse a machine outright. */
        ok(idletoken_llama_seq_slots(&u64, &nokv, 8192, 1.0, 4) == 1,
           "unknown KV shape → 1 slot (not a licence to open four)");
        /* NULL node → 1, same reflex as an unknown KV shape. */
        ok(idletoken_llama_seq_slots(NULL, &small_m, 8192, 1.0, 4) == 1,
           "no node → 1 slot (never guess a budget)");
        /* cap <= 0 means the built-in cap, not "unlimited". */
        ok(idletoken_llama_seq_slots(&u64, &small_m, 8192, 1.0, 0) ==
               IDLETOKEN_LLAMA_SLOT_CAP,
           "cap<=0 falls back to IDLETOKEN_LLAMA_SLOT_CAP");
        /* A cluster node holding half the layers pays half the weights AND
         * half the KV — so half a machine can still be as wide. 12 GiB at
         * share 0.5: 12 − 4 − 0.875 = 7.125 GiB over 0.25 GiB/seq → capped 4. */
        ok(idletoken_llama_seq_slots(&u12, &small_m, 8192, 0.5, 4) == 4,
           "layer_share scales the weights and the KV together");
        /* And a cluster gets the TIGHTEST node, not the sum: node A is 4-wide,
         * node B (same share but a 32K window: 6 − 4 − 0.875 = 1.125 GiB over
         * 1 GiB/seq) is 1-wide, so the cluster carries one sequence. Summing
         * would promise five. */
        {
            const int a = idletoken_llama_seq_slots(&u12, &small_m, 8192, 0.5, 4);
            const int b = idletoken_llama_seq_slots(&u6, &small_m, 32768, 0.5, 4);
            ok(a == 4 && b == 1 && (a < b ? a : b) == 1,
               "cluster slots come from the tightest node, not the sum");
        }

        /* ---- the KV pool: discrete cards budget VRAM, not VRAM+RAM -------
         *
         * Regression fixtures for the 2026-08-18 desktop freeze on a Windows
         * test node. The machine:
         * RTX 5060 Ti, 16 GiB VRAM + 64 GiB system RAM; qwen3-8b Q4_K_M is
         * ~5 GiB of weights at ~144 KiB of KV per token; the client asked for
         * 40960 tokens per slot.
         *
         * The old formula budgeted against idletoken_llama_node_usable() =
         * 80 GiB and returned 4 (verified against the pre-fix code, and
         * against the engine's own `n_slots = 4` in that machine's log). Four
         * slots is ~22 GiB of KV on a 16 GiB card: WDDM paged it out to host
         * memory instead of failing, and the desktop froze. */
        idletoken_llm_model_size q8 = { .total_bytes = g(5), .n_layers = 36,
                                        .kv_bytes_per_token = 144 * 1024 };
        idletoken_node_mem discrete16 = NM(g(16), g(64), 0);
        ok(idletoken_llama_node_usable(&discrete16) == g(80),
           "16+64 fixture: 80 GiB across the machine (what the old budget used)");
        ok(idletoken_llama_kv_pool(&discrete16) == g(16),
           "16+64 fixture: but only 16 GiB of it can hold KV");
        /* 16 − 5 − 0.83 = 10.17 GiB over 5.625 GiB/seq → 1. */
        ok(idletoken_llama_seq_slots(&discrete16, &q8, 40960, 1.0, 4) == 1,
           "discrete 16 GiB card at 40K ctx → 1 slot (was 4: the freeze)");

        /* Same model and context on a unified machine with a genuinely large
         * pool → still wide. The fix must not turn into "everyone gets one
         * slot", which would silently undo the multi-slot work. */
        idletoken_node_mem uni96 = NM(g(96), g(96), 1);
        ok(idletoken_llama_seq_slots(&uni96, &q8, 40960, 1.0, 4) == 4,
           "96 GiB unified at the same ctx → still 4 slots");

        /* Slow tier: the weights do not fit VRAM, so they spill to host RAM
         * and every token crosses PCIe. One slot — parallel sequences buy
         * nothing there and cost another full context of the scarce pool. */
        idletoken_llm_model_size big8 = { .total_bytes = g(16), .n_layers = 60,
                                          .kv_bytes_per_token = 64 * 1024 };
        idletoken_node_mem spill = NM(g(8), g(64), 0);
        ok(idletoken_llama_seq_slots(&spill, &big8, 8192, 1.0, 4) == 1,
           "weights spill out of VRAM (slow tier) → 1 slot");
        /* And the same node is NOT refused by the capacity model — slow is a
         * speed verdict, not a feasibility one. This keeps the two apart. */
        ok(idletoken_llama_node_usable(&spill) > g(16),
           "the slow-tier node still has capacity for the model (slow ≠ refuse)");

        /* Mac bench regression floor: the 0.8B multi-slot measurement
         * (results/llamacpp-multislot-mac-20260818.md) ran 4-wide and must
         * stay 4-wide. ~0.6 GiB of weights, ~24 KiB of KV per token, 32K ctx,
         * on the M4's ~11 GiB of usable unified memory. */
        idletoken_llm_model_size small08 = { .total_bytes = (uint64_t)(0.6 * (double)GiB),
                                             .n_layers = 24,
                                             .kv_bytes_per_token = 24 * 1024 };
        idletoken_node_mem mac = NM(g(11), g(11), 1);
        ok(idletoken_llama_seq_slots(&mac, &small08, 32768, 1.0, 4) == 4,
           "0.8B on the 16 GiB Mac → still 4 slots (bench regression floor)");

        /* Cluster path: a discrete node's share is charged to its VRAM too.
         * Half the layers of the 8B on the same 16 GiB card is comfortable —
         * which is the point of clustering — but the budget is still 16, not
         * 80, so a node that would overflow its card cannot hide behind its
         * host RAM. */
        ok(idletoken_llama_seq_slots(&discrete16, &q8, 40960, 0.25, 4) == 4,
           "cluster share on a discrete node: a quarter of the KV fits VRAM");
        ok(idletoken_llama_seq_slots(&discrete16, &q8, 40960, 0.75, 4) == 2,
           "cluster share on a discrete node: three quarters gets fewer");
        /* In-test positive control for the pair above: hand the SAME numbers a
         * node whose pool is the old whole-machine figure (80 GiB, expressed
         * as unified so the pool is the usable total) and both shares come
         * back at the cap. That is the answer the pre-fix code gave, so these
         * assertions demonstrably fail on it rather than merely restating it. */
        idletoken_node_mem as_before = NM(g(80), g(80), 1);
        ok(idletoken_llama_seq_slots(&as_before, &q8, 40960, 1.0, 4) == 4 &&
               idletoken_llama_seq_slots(&as_before, &q8, 40960, 0.75, 4) == 4,
           "control: budgeted against the whole machine, every case is 4");
    }

    /* ---- the CLUSTER split is charged to the same pool (T16) --------------
     *
     * Replay of the cell that crashed on 2026-08-19
     * (results/t14-engine-bump-phaseb-20260820.md, "Why that run could not
     * answer the question it was asked"): a unified-memory node coordinating with
     * 107.61 GiB, a discrete-GPU node joining with 13.2 GiB of VRAM behind 37.3
     * GiB of system RAM, DeepSeek-V4-Flash IQ2_XXS at 80.76 GiB.
     *
     * The split was the ratio of idletoken_llama_node_usable() and nothing
     * else, so the joiner was handed 50.48/158.09 = 0.3193 = 25.8 GiB onto a card
     * that holds 13.2 — and its rpc-server, started `-d CUDA0`, can reach
     * nothing else. Windows paged VRAM to system memory instead of failing and
     * the rpc-server died mid-decode. Same disease as the slot budget above:
     * memory the allocation cannot touch was counted as capacity. */
    {
        idletoken_llama_plan p;
        idletoken_llm_model_size dsv4 = {
            .total_bytes = (uint64_t)(80.76 * (double)GiB),
            .n_layers = 43, .kv_bytes_per_token = 65536,
        };
        idletoken_node_mem cell[] = {
            NM(g(107.61), g(107.61), 1),                 /* unified memory    */
            NM(g(13.2),   g(37.3),   0),                 /* discrete GPU      */
        };
        /* The layer bytes the split divides: weights + the KV it sizes. */
        const double  slice = 80.76 + 2.0;               /* GiB, ctx 32768 */
        /* allow_small_cluster = 1 throughout this block, because that is how
         * the cell ran: 80.76 GiB fits the 107.61 GiB node on its own, so the
         * matrix harness sets IDLETOKEN_ALLOW_SMALL_CLUSTER=1 to get a cluster
         * at all (hard invariant #5 — fits → don't cluster). Without it these
         * fixtures would all return SINGLE and assert nothing about splitting. */

        /* First, that the fixture really is the incident: the OLD rule's ratio
         * is a pure function of node_usable, so it can be computed here and
         * checked against the number in that report. Without this the
         * assertions below could pass on a fixture that never reproduced it. */
        const double old_share =
            (double)idletoken_llama_node_usable(&cell[1]) /
            (double)(idletoken_llama_node_usable(&cell[0]) +
                     idletoken_llama_node_usable(&cell[1]));
        ok(old_share > 0.318 && old_share < 0.320,
           "fixture replays the incident: the old rule hands the joiner 0.3193");
        ok(old_share * slice > 25.0 && old_share * slice > 13.2,
           "...which is 25.8 GiB onto a 13.2 GiB card (the crash)");

        ok(idletoken_plan_llamacpp(&dsv4, cell, 2, 0, 32768, 1, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_CLUSTER && p.order[1] == 1,
           "unified coordinator + discrete-GPU worker on DSv4 → CLUSTER");
        /* THE regression: whatever the split says, the discrete worker's share
         * must fit the memory its rpc-server can address. */
        ok(p.tensor_split[1] * slice <= 13.2,
           "the discrete worker's share fits its VRAM, not its VRAM+RAM");
        /* And it is not a token slice either — 13.2 GiB of card is real
         * capacity and the cluster is pointless if it goes unused. The run
         * that PASSED on 2026-08-20 used 0.1121 (9.05 GiB), reached by faking
         * the node's usable bytes by hand; this must land in the same place
         * without the hand. */
        ok(p.tensor_split[1] > 0.09 && p.tensor_split[1] < 0.13,
           "...and lands where the passing run's hand-aimed split did (~0.11)");
        /* Overhead is charged per node, so the coordinator's share is the rest
         * and the ratios still sum to one. */
        ok(p.tensor_split[0] + p.tensor_split[1] > 0.999 &&
               p.tensor_split[0] + p.tensor_split[1] < 1.001,
           "split still sums to 1 after the cap");

        /* Control 1 — unified nodes are untouched. Both answers coincide on a
         * machine with one physical pool, which is why this bug was invisible
         * on every unified-memory cell we ran: the split here must still be the
         * plain usable ratio. */
        idletoken_node_mem uni2[] = { NM(g(107.61), g(107.61), 1),
                                      NM(g(50.48),  g(50.48),  1) };
        ok(idletoken_plan_llamacpp(&dsv4, uni2, 2, 0, 32768, 1, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_CLUSTER &&
               p.tensor_split[1] > 0.318 && p.tensor_split[1] < 0.320,
           "control: two unified nodes still split by usable (0.3193)");

        /* Control 2 — the SINGLE path is not touched. One discrete machine
         * whose card cannot hold the model but whose machine can is HYBRID's
         * whole reason to exist: llama-server offloads what fits and keeps the
         * rest in host RAM, so vram+ram remains the right budget there. If
         * this ever flips to REFUSE, the cluster fix has leaked. */
        idletoken_llm_model_size m8 = { .total_bytes = g(8), .n_layers = 32,
                                        .kv_bytes_per_token = 65536 };
        idletoken_node_mem hybrid1[] = { NM(g(4), g(32), 0) };
        ok(idletoken_plan_llamacpp(&m8, hybrid1, 1, 0, 8192, 0, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_SINGLE,
           "control: single-node HYBRID still budgets VRAM+RAM (4 GiB card, 8 GiB model)");

        /* Refusal quality: a cluster whose cards cannot hold the model between
         * them is refused, and the refusal names the machine and which memory
         * it ran out of. The aggregate vram+ram of this pair is 205 GiB, which
         * is why the checks above it wave it through. */
        idletoken_node_mem thin[] = { NM(g(8),  g(96), 0),
                                      NM(g(8),  g(96), 0) };
        ok(idletoken_plan_llamacpp(&dsv4, thin, 2, 0, 32768, 1, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_REFUSE,
           "two thin cards behind big host RAM → REFUSE (they cannot hold it)");
        ok(strstr(p.why, "video memory") != NULL &&
               strstr(p.why, "system RAM") != NULL &&
               strstr(p.why, "short") != NULL,
           "...and the refusal names the node, the memory kind and the shortfall");

        /* The coordinator's own card is subject to the same rule, and there
         * the refusal is the privacy invariant's: layer 0 may not move, so a
         * coordinator that cannot hold one layer is the end of the road. */
        idletoken_node_mem tinycoord[] = { NM(g(1), g(96), 0),
                                           NM(g(96), g(96), 1) };
        ok(idletoken_plan_llamacpp(&dsv4, tinycoord, 2, 0, 32768, 1, &p) == 0 &&
               p.kind == IDLETOKEN_LLPLAN_REFUSE &&
               strstr(p.why, "layer 0") != NULL,
           "coordinator's card too small for one layer → REFUSE, naming layer 0");
    }

    /* ---- where the budget's byte count comes from (T8) ------------------
     *
     * Regression fixtures for the 2026-08-19 discrete-16-GiB measurement
     * (results/llamacpp-multislot-big-win-20260819.md §4): the coordinator
     * budgeted 7.98 GiB (the manifest's DEFAULT quant, IQ2_XXS) while the
     * engine opened a 15.59 GiB Q4_K_M file, and derived 2 sequence slots
     * where 1 is correct. The three sources below are the fix; the last
     * assertion of each group is the outcome that machine cared about. */
    {
        char rwhy[512];
        idletoken_llm_model_size ms;

        const idletoken_model_spec *q27 = idletoken_model_get("qwen3.5-27b");
        ok(q27 != NULL && q27->n_variants > 1, "27B spec has a variant menu");

        /* The machine, as measured: RTX 5060 Ti, 13.17 GiB of usable VRAM next
         * to 64 GiB of system RAM, serving at 4096 tokens per slot. */
        idletoken_node_mem discrete13 = NM((uint64_t)(13.17 * (double)GiB), g(64), 0);
        const uint64_t IQ2_XXS_BYTES = 8573593504ull;   /* variants[default] */
        const uint64_t Q4_K_M_BYTES  = 16740812704ull;  /* the file it served */

        /* --- source 3: model only. Allowed, but it must SAY that it guessed. */
        ok(idletoken_model_size_resolve(q27, NULL, NULL, &ms, rwhy, sizeof rwhy) == 0 &&
               ms.total_bytes == IQ2_XXS_BYTES,
           "no quant, no path → the manifest's default variant");
        ok(strstr(rwhy, "DEFAULT") != NULL && strstr(rwhy, "WARNING") != NULL,
           "...and it warns that a different quant makes this budget wrong");
        ok(ms.n_layers == q27->n_layers && ms.kv_bytes_per_token ==
               (uint64_t)q27->kv_bytes_per_token_layer * q27->n_layers,
           "shape (layers, KV/token) comes from the manifest in every case");
        /* THE BUG, kept as a live control: this is the number the pre-fix code
         * fed the planner on the T3 run, and it really does derive 2 slots. If
         * a future change makes the default source produce 1 anyway, the
         * assertions below would pass for the wrong reason. */
        const int slots_from_default =
            idletoken_llama_seq_slots(&discrete13, &ms, 4096, 1.0, 4);
        ok(slots_from_default == 2,
           "control: the pre-fix source (default quant) really does say 2 slots");

        /* --- source 2: the named precision. */
        ok(idletoken_model_size_resolve(q27, "Q4_K_M", NULL, &ms, rwhy, sizeof rwhy) == 0 &&
               ms.total_bytes == Q4_K_M_BYTES,
           "--quant Q4_K_M → that variant's bytes");
        ok(strstr(rwhy, "Q4_K_M") != NULL && strstr(rwhy, "WARNING") == NULL,
           "...named explicitly, so nothing to warn about");
        ok(idletoken_llama_seq_slots(&discrete13, &ms, 4096, 1.0, 4) == 1,
           "15.59 GiB of weights vs a 13.17 GiB KV pool → 1 slot");

        /* An unknown quant name falls back to the default — as it always has —
         * but the fallback is now audible. */
        ok(idletoken_model_size_resolve(q27, "Q9_NOPE", NULL, &ms, rwhy, sizeof rwhy) == 0 &&
               ms.total_bytes == IQ2_XXS_BYTES && strstr(rwhy, "WARNING") != NULL,
           "unknown quant → default variant, and it says the name was not in the menu");

        /* A model with no variant menu has nothing to pick wrong, so it must
         * not be warned at — a warning on every single-precision model is a
         * warning nobody reads by the time one matters. */
        {
            /* Built here rather than borrowed from the registry: which shipped
             * models carry a menu is upstream's business and has changed twice
             * already, so a test that names one goes red for the wrong reason. */
            idletoken_model_spec single;
            memset(&single, 0, sizeof single);
            single.id = "single-precision-fixture";
            single.n_layers = 32;
            single.kv_bytes_per_token_layer = 2048;
            single.layer_weight_bytes = g(4);
            single.shared_weight_bytes = 0;

            ok(idletoken_model_size_resolve(&single, NULL, NULL, &ms,
                                            rwhy, sizeof rwhy) == 0 &&
                   ms.total_bytes == g(4) &&
                   strstr(rwhy, "single precision") != NULL &&
                   strstr(rwhy, "WARNING") == NULL,
               "a single-precision model is not warned at for having one precision");
            ok(idletoken_model_size_resolve(&single, "Q8_0", NULL, &ms,
                                            rwhy, sizeof rwhy) == 0 &&
                   ms.total_bytes == g(4) && strstr(rwhy, "WARNING") != NULL,
               "...but a --quant it cannot honour is not swallowed either");
        }

#ifdef HAVE_SPARSE_FIXTURES
        /* --- source 1: the file the engine will actually open. This is the T3
         * invocation verbatim: a model id and a GGUF path, NO --quant. */
        {
            const char *p = "/tmp/idletoken-plantest-Qwen3.5-27B-Q4_K_M.gguf";
            if (make_sized_file(p, Q4_K_M_BYTES) != 0) {
                printf("  [skip] cannot create the sized GGUF fixture (%s)\n", p);
            } else {
                ok(idletoken_model_size_resolve(q27, NULL, p, &ms, rwhy, sizeof rwhy) == 0 &&
                       ms.total_bytes == Q4_K_M_BYTES,
                   "T3 replay: --llama-gguf with no --quant → the file's real size");
                ok(strstr(rwhy, "GGUF on disk") != NULL && strstr(rwhy, "Q4_K_M") != NULL,
                   "...and it names the file and the quant it recognised");
                ok(idletoken_llama_seq_slots(&discrete13, &ms, 4096, 1.0, 4) == 1 &&
                       slots_from_default == 2,
                   "T3 replay: 27B Q4_K_M on the 16 GiB card → 1 slot (was 2)");

                /* An explicit --quant that contradicts the file: the file wins,
                 * loudly. Believing the flag is how the budget got here. */
                ok(idletoken_model_size_resolve(q27, "IQ2_XXS", p, &ms, rwhy, sizeof rwhy) == 0 &&
                       ms.total_bytes == Q4_K_M_BYTES &&
                       strstr(rwhy, "WARNING") != NULL,
                   "--quant that disagrees with the file → the file wins, with a warning");
                remove(p);
            }

            /* A path that is not there degrades to the manifest — the engine is
             * about to fail on the same path and will say more than we can —
             * but never silently. */
            ok(idletoken_model_size_resolve(q27, "Q4_K_M", "/tmp/idletoken-no-such.gguf",
                                            &ms, rwhy, sizeof rwhy) == 0 &&
                   ms.total_bytes == Q4_K_M_BYTES &&
                   strstr(rwhy, "WARNING") != NULL && strstr(rwhy, "cannot read") != NULL,
               "an unreadable GGUF path falls back to the manifest, loudly");
        }

        /* --- byte-count matching, on a fixture menu so the sizes stay tiny.
         * Three neighbours 0.5% apart (inside the 1% window) plus one far
         * away: enough to pin "nearest wins", "ambiguity is disclosed" and
         * "no match is a warning, not a silent default". */
        {
            static const idletoken_model_variant FIXV[] = {
                { .quant = "TINY", .layer_weight_bytes = 100000, .shared_weight_bytes = 0,
                  .gguf = "tiny.gguf" },
                { .quant = "NEAR_A", .layer_weight_bytes = 200000, .shared_weight_bytes = 0,
                  .gguf = "a.gguf" },
                { .quant = "NEAR_B", .layer_weight_bytes = 201000, .shared_weight_bytes = 0,
                  .gguf = "b.gguf" },
            };
            idletoken_model_spec fix;
            memset(&fix, 0, sizeof fix);
            fix.id = "fixture-model";
            fix.label = "Fixture";
            fix.available = 1;
            fix.n_layers = 8;
            fix.kv_bytes_per_token_layer = 1024;
            fix.variants = FIXV;
            fix.n_variants = 3;
            fix.default_variant = 0;
            fix.layer_weight_bytes = FIXV[0].layer_weight_bytes;

            const char *fp = "/tmp/idletoken-plantest-fixture.gguf";

            if (make_sized_file(fp, 200100) == 0) {
                ok(idletoken_model_size_resolve(&fix, NULL, fp, &ms, rwhy, sizeof rwhy) == 0 &&
                       ms.total_bytes == 200100 && strstr(rwhy, "NEAR_A") != NULL,
                   "two quants inside the window → the NEAREST one is named");
                ok(strstr(rwhy, "nearest of several") != NULL,
                   "...and the ambiguity is disclosed rather than sounding certain");
            } else {
                printf("  [skip] fixture file\n");
            }
            if (make_sized_file(fp, 150000) == 0) {
                ok(idletoken_model_size_resolve(&fix, NULL, fp, &ms, rwhy, sizeof rwhy) == 0 &&
                       ms.total_bytes == 150000 && strstr(rwhy, "WARNING") != NULL &&
                       strstr(rwhy, "matches no quantization") != NULL,
                   "a size between quants → the real bytes, and a warning it is unrecognised");
            }
            remove(fp);

            /* Split sets: llama.cpp loads every part, so the budget must too.
             * Sizing from part 1 alone is the same understatement bug wearing
             * a different hat. */
            const char *s1 = "/tmp/idletoken-plantest-split-00001-of-00003.gguf";
            const char *s2 = "/tmp/idletoken-plantest-split-00002-of-00003.gguf";
            const char *s3 = "/tmp/idletoken-plantest-split-00003-of-00003.gguf";
            remove(s2); remove(s3);
            if (make_sized_file(s1, 60000) == 0) {
                char ewhy[256] = "";
                ok(idletoken_gguf_bytes_on_disk(s1, ewhy, sizeof ewhy) == 0 &&
                       strstr(ewhy, "incomplete") != NULL,
                   "a split set missing its parts is refused, not sized at part 1");
                if (make_sized_file(s2, 60000) == 0 && make_sized_file(s3, 40000) == 0) {
                    ok(idletoken_gguf_bytes_on_disk(s1, ewhy, sizeof ewhy) == 160000,
                       "a complete split set is summed over all its parts");
                    ok(idletoken_gguf_bytes_on_disk(s2, ewhy, sizeof ewhy) == 0 &&
                           strstr(ewhy, "part 1") != NULL,
                       "pointing at part 2 is named as the user error it is");
                }
            }
            remove(s1); remove(s2); remove(s3);
        }
#else
        printf("  [skip] sized-GGUF fixtures need 64-bit fseek and sparse files\n");
#endif
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("PLAN_TEST_FAIL\n"); return 1; }
    printf("PLAN_TEST_OK\n");
    return 0;
}
