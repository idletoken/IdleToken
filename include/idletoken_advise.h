/* IdleToken Cluster — capability advisor ("what can this machine run?").
 *
 * The first question a non-expert has after installing is not "which quant
 * should I pick", it is "can my computer run any of this at all". This module
 * answers it: given the memory of one machine (or of a whole cluster), it walks
 * every registered model × every shipped precision and reports the best mode
 * (GPU_ONLY / HYBRID / no) plus the largest context tier that fits — and, when
 * the answer is no, HOW MUCH memory is missing.
 *
 * Every verdict comes from the planner (`idletoken_mode_decide_quant`), never from
 * a second estimate written here. A capability table that disagrees with the
 * planner is worse than none: it promises what the cluster then refuses to do.
 */

#ifndef IDLETOKEN_ADVISE_H
#define IDLETOKEN_ADVISE_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_model.h"
#include "idletoken_plan.h"

/* Context tiers (8K / 32K / 128K / 512K / 1M). The
 * advisor reports the largest tier a given (model, quant) fits in; a model
 * whose own context_max is smaller is capped to it. */
#define IDLETOKEN_ADVISE_MAX_ROWS 128

typedef struct {
    const char *model_id;
    const char *label;      /* human name from the manifest */
    const char *quant;      /* "" when the model ships a single precision */
    idletoken_mode mode;       /* GPU_ONLY / HYBRID / REFUSE */
    uint32_t    max_ctx;    /* largest tier that fits in `mode`; 0 when REFUSE */
    uint64_t    weight_bytes;   /* download size at this precision */
    uint64_t    shortfall;  /* bytes of memory still missing (REFUSE only) */
    int         unavailable;/* model not runnable in this build (manifest flag) */
    /* 1 = this model is single-node-only, so the verdict was computed against
     * the BEST ONE machine in the roster rather than their sum. Without this
     * the advisor would answer "yes" for a 20 GiB model on four 8 GiB machines
     * and the coordinator would then refuse to start it — a capability table
     * that disagrees with the planner is worse than none. */
    int         single_node;
} idletoken_advice_row;

/* Fill `out` with one row per (model, precision). Returns the number of rows
 * written, or -1 on bad arguments. `nodes`/`n_nodes` describe the machines the
 * user actually has: pass 1 node for "this machine", or the whole roster for
 * "this cluster". */
int idletoken_advise(const idletoken_node_mem *nodes, int n_nodes,
                  idletoken_advice_row *out, int max_rows);

/* Human-readable table (CLI). `hw_line` is an already-formatted hardware
 * summary line, or NULL. */
void idletoken_advise_print(const idletoken_advice_row *rows, int n,
                         int n_nodes, const char *hw_line);

/* One JSON object: {"nodes":N,"models":[{...}]}. Consumed by the client and by
 * the coordinator's GET /idletoken/v1/capability. */
void idletoken_advise_print_json(const idletoken_advice_row *rows, int n, int n_nodes);

/* Same JSON, into a caller-provided buffer (the coordinator serves it over
 * HTTP rather than printing). Returns bytes written excluding the NUL, or -1
 * if the buffer is too small. */
int idletoken_advise_json(const idletoken_advice_row *rows, int n, int n_nodes,
                       char *buf, size_t cap);

#endif /* IDLETOKEN_ADVISE_H */
