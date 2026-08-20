/* Capability advisor — see include/idletoken_advise.h. */

#include "idletoken_advise.h"
#include "idletoken_modelsize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Context tiers, descending so the first fit is the largest. */
static const uint32_t TIERS[] = { 1048576u, 524288u, 131072u, 32768u, 8192u };
#define N_TIERS ((int)(sizeof(TIERS) / sizeof(TIERS[0])))

static const double GiB = 1024.0 * 1024.0 * 1024.0;

/* Tiers this model can actually be asked for: never above its context_max, and
 * always at least one entry (a model with a small window still gets judged at
 * that window rather than dropping out of the table). */
static int tiers_for(const idletoken_model_spec *m, uint32_t *out) {
    int n = 0;
    for (int i = 0; i < N_TIERS; i++) {
        if (m->ctx_max == 0 || TIERS[i] <= m->ctx_max) out[n++] = TIERS[i];
    }
    if (n == 0) out[n++] = m->ctx_max ? m->ctx_max : 8192u;
    return n;
}

/* Best verdict for one (model, quant): the largest tier that fits GPU_ONLY,
 * else the largest that fits HYBRID, else REFUSE with the shortfall measured at
 * the SMALLEST tier (the friendliest possible answer — "even at 8K you are
 * still N GB short"). */
static void judge(const idletoken_model_spec *m, const char *quant,
                  const idletoken_node_mem *nodes, int n_nodes,
                  idletoken_advice_row *row) {
    uint32_t tiers[N_TIERS];
    const int nt = tiers_for(m, tiers);

    row->mode = IDLETOKEN_MODE_REFUSE;
    row->max_ctx = 0;
    row->shortfall = 0;

    for (int pass = 0; pass < 2; pass++) {
        const idletoken_mode want = pass == 0 ? IDLETOKEN_MODE_GPU_ONLY : IDLETOKEN_MODE_HYBRID;
        for (int i = 0; i < nt; i++) {
            const idletoken_mode got = idletoken_mode_decide_quant(
                m, quant, nodes, n_nodes, tiers[i], NULL, NULL, 0);
            if (got == want) {
                row->mode = want;
                row->max_ctx = tiers[i];
                return;
            }
        }
    }

    /* Nothing fits — report how far off the smallest tier is. */
    uint64_t missing = 0;
    idletoken_mode_decide_quant(m, quant, nodes, n_nodes, tiers[nt - 1], &missing, NULL, 0);
    row->shortfall = missing;
}

int idletoken_advise(const idletoken_node_mem *nodes, int n_nodes,
                  idletoken_advice_row *out, int max_rows) {
    if (!nodes || n_nodes <= 0 || !out || max_rows <= 0) return -1;

    int n = 0;
    for (int mi = 0; mi < idletoken_model_count() && n < max_rows; mi++) {
        const idletoken_model_spec *m = idletoken_model_at(mi);
        if (!m) continue;

        /* One row per shipped precision; models without a variant table get a
         * single row at their default weights. */
        const int n_var = m->n_variants > 0 ? m->n_variants : 1;
        for (int vi = 0; vi < n_var && n < max_rows; vi++) {
            const char *quant = m->n_variants > 0 ? m->variants[vi].quant : "";
            idletoken_advice_row *row = &out[n++];
            memset(row, 0, sizeof(*row));
            row->model_id = m->id;
            row->label    = m->label ? m->label : m->id;
            row->quant    = quant;
            row->unavailable = !m->available;

            /* Same resolver the coordinator budgets with (T8). There is no GGUF
             * on this machine to point at — the advisor answers "what COULD run
             * here" — so this is always the (model, quant) source, and the point
             * of routing it through one function is that it stays that way when
             * a future caller does have a path. An advisor that sizes models
             * its own way is an advisor that promises what the planner refuses. */
            idletoken_llm_model_size msize;
            if (idletoken_model_size_resolve(m, quant, NULL, &msize, NULL, 0) != 0)
                continue;
            row->weight_bytes = msize.total_bytes;

            if (!m->available) {
                /* Honest: the manifest says this build cannot run it, so no
                 * amount of memory changes the answer. */
                row->mode = IDLETOKEN_MODE_REFUSE;
                continue;
            }
            /* A single-node model is judged against the BEST ONE machine, not
             * the roster's sum — that is the only verdict the coordinator will
             * honour (it refuses --num-workers > 1 for these). "Best" is by
             * total usable memory, which is what the planner spends. */
            if (!idletoken_model_may_cluster(m, NULL, 0)) {
                int best = 0;
                for (int i = 1; i < n_nodes; i++)
                    if (nodes[i].vram_usable + nodes[i].ram_usable >
                        nodes[best].vram_usable + nodes[best].ram_usable) best = i;
                row->single_node = 1;
                judge(m, quant, &nodes[best], 1, row);
            } else {
                judge(m, quant, nodes, n_nodes, row);
            }
        }
    }
    return n;
}

/* Sub-GB models exist (Qwen3.5-0.8B Q4_K_M is ~0.5 GB); printing them as
 * "0 GB" reads like a bug. */
static void size_word(uint64_t bytes, char out[16]) {
    const double gb = (double)bytes / GiB;
    if (bytes == 0)   snprintf(out, 16, "?");
    else if (gb < 1)  snprintf(out, 16, "%.0f MB", (double)bytes / (1024.0 * 1024.0));
    else if (gb < 10) snprintf(out, 16, "%.1f GB", gb);
    else              snprintf(out, 16, "%.0f GB", gb);
}

static const char *mode_word(const idletoken_advice_row *r) {
    if (r->unavailable)                  return "not in this build";
    if (r->mode == IDLETOKEN_MODE_GPU_ONLY) return "yes (GPU only)";
    if (r->mode == IDLETOKEN_MODE_HYBRID)   return "yes (GPU + RAM)";
    return "no";
}

static void ctx_word(uint32_t ctx, char out[16]) {
    if (ctx == 0)              snprintf(out, 16, "-");
    else if (ctx >= 1048576u)  snprintf(out, 16, "%uM", ctx / 1048576u);
    else                       snprintf(out, 16, "%uK", ctx / 1024u);
}

void idletoken_advise_print(const idletoken_advice_row *rows, int n,
                         int n_nodes, const char *hw_line) {
    printf("IdleToken capability report — %d machine%s\n",
           n_nodes, n_nodes == 1 ? "" : "s");
    if (hw_line && *hw_line) printf("  %s\n", hw_line);
    printf("\n  %-22s %-8s %8s  %-18s %-8s %s\n",
           "MODEL", "QUANT", "SIZE", "CAN RUN", "MAX CTX", "NOTE");
    for (int i = 0; i < n; i++) {
        const idletoken_advice_row *r = &rows[i];
        char ctx[16]; ctx_word(r->max_ctx, ctx);
        char size[16]; size_word(r->weight_bytes, size);
        char note[80] = "";
        if (r->unavailable)
            snprintf(note, sizeof note, "backend not implemented yet");
        else if (r->mode == IDLETOKEN_MODE_REFUSE && r->shortfall > 0)
            /* For a single-node model the shortfall is measured against ONE
             * machine and more machines will not close it, so name the remedy
             * that actually works instead of letting the footer's "add a
             * machine" advice be read as applying to this row. */
            snprintf(note, sizeof note, "needs %.0f GB more%s",
                     (double)r->shortfall / GiB,
                     r->single_node ? " on one machine — try a lower precision" : "");
        else if (r->mode == IDLETOKEN_MODE_HYBRID)
            snprintf(note, sizeof note, "slower: part of the model sits in RAM");
        if (r->single_node && !r->unavailable && r->mode != IDLETOKEN_MODE_REFUSE) {
            const size_t used = strlen(note);
            snprintf(note + used, sizeof note - used, "%sruns on one machine",
                     used ? "; " : "");
        }
        printf("  %-22s %-8s %8s  %-18s %-8s %s\n",
               r->model_id, r->quant[0] ? r->quant : "-", size,
               mode_word(r), ctx, note);
    }
    printf("\n  \"yes (GPU only)\" is the fast path. \"yes (GPU + RAM)\" works but is\n"
           "  slower. \"no\" tells you how much memory is missing — for a cluster\n"
           "  model, adding another machine adds its memory to the pool.\n"
           "  Models marked \"runs on one machine\" are served by a single node:\n"
           "  they are small enough that splitting them over a LAN would cost more\n"
           "  in round-trips than it saves, so extra machines do not help them.\n");
}

/* JSON writer shared by the printing and buffer forms. `emit` appends to buf. */
int idletoken_advise_json(const idletoken_advice_row *rows, int n, int n_nodes,
                       char *buf, size_t cap) {
    size_t off = 0;
    #define EMIT(...) do { \
        int _w = snprintf(buf + off, cap > off ? cap - off : 0, __VA_ARGS__); \
        if (_w < 0) return -1; \
        off += (size_t)_w; \
        if (off >= cap) return -1; \
    } while (0)

    EMIT("{\"nodes\":%d,\"models\":[", n_nodes);
    for (int i = 0; i < n; i++) {
        const idletoken_advice_row *r = &rows[i];
        EMIT("%s{\"id\":\"%s\",\"label\":\"%s\",\"quant\":\"%s\","
             "\"mode\":\"%s\",\"max_ctx\":%u,\"weight_bytes\":%llu,"
             "\"shortfall_bytes\":%llu,\"available\":%s,\"single_node\":%s}",
             i ? "," : "", r->model_id, r->label, r->quant,
             r->unavailable ? "unavailable"
               : r->mode == IDLETOKEN_MODE_GPU_ONLY ? "gpu_only"
               : r->mode == IDLETOKEN_MODE_HYBRID   ? "hybrid" : "no",
             r->max_ctx,
             (unsigned long long)r->weight_bytes,
             (unsigned long long)r->shortfall,
             r->unavailable ? "false" : "true",
             r->single_node ? "true" : "false");
    }
    EMIT("]}");
    #undef EMIT
    return (int)off;
}

size_t idletoken_advise_json_cap(int n) {
    if (n < 0) n = 0;
    return 256 + (size_t)n * 512;
}

void idletoken_advise_print_json(const idletoken_advice_row *rows, int n, int n_nodes) {
    /* Sized from the row count rather than from a constant somebody has to
     * remember to raise — see idletoken_advise_json_cap for why. */
    const size_t cap = idletoken_advise_json_cap(n);
    char *buf = (char *)malloc(cap);
    if (!buf) {
        fprintf(stderr, "idletoken-advise: out of memory for the capability JSON\n");
        return;
    }
    if (idletoken_advise_json(rows, n, n_nodes, buf, cap) < 0) {
        fprintf(stderr, "idletoken-advise: capability JSON did not fit\n");
        free(buf);
        return;
    }
    puts(buf);
    free(buf);
}
