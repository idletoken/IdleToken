/* ds4x_runner.c — PP stage driver (Phase B.4).
 *
 * Owns this stage's per-layer MLA KV caches (sized to the context window) and
 * runs the model's whole [layer_lo,layer_hi) range over hidden states. This
 * is the surface the worker's inference loop calls: hidden in → hidden out,
 * KV staying local (never crosses the PP boundary — only the hidden tensor
 * does, matching ds4's cur_hc handoff). Correctness-first fp32, no GPU. */
#include "idletoken_ds4x.h"
#ifdef IDLETOKEN_DS4X_CUDA
#include "idletoken_ds4x_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The linear-attention recurrent state goes to VRAM under the same switch that
 * puts the weights there (ds4x_gpu_wanted() + a usable device), because the
 * two only pay off together: with the projections on the GPU the recurrence is
 * the dominant term of a linear layer, and with them on the CPU the copies
 * would dominate. IDLETOKEN_DS4X_GDN_CPU=1 forces the recurrence back to the CPU
 * without giving up the GPU matvecs — that A/B is how the speedup is measured
 * rather than guessed. */
#ifdef IDLETOKEN_DS4X_CUDA
static int gdn_gpu_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *off = getenv("IDLETOKEN_DS4X_GDN_CPU");
        v = ds4x_gpu_wanted() && !(off && *off == '1') && ds4x_cuda_available();
    }
    return v;
}
#endif

struct ds4x_runner {
    const ds4x_model *model;
    const ds4x_config *cfg;
    uint32_t ctx;
    uint16_t lo, hi;
    ds4x_kv_cache *caches;   /* [hi - lo], one per local layer */
};

ds4x_runner *ds4x_runner_create(const ds4x_model *model, uint32_t ctx_size,
                                char *err, size_t errlen) {
    const ds4x_config *cfg = ds4x_model_config(model);
    if (ctx_size == 0) {
        if (err) snprintf(err, errlen, "ctx_size must be > 0");
        return NULL;
    }
    ds4x_runner *r = (ds4x_runner *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->model = model;
    r->cfg = cfg;
    r->ctx = ctx_size;

    /* Recover the shard's layer range from which layers have weights. */
    r->lo = 0;
    while (r->lo < cfg->n_layer && !ds4x_model_layer(model, r->lo)) r->lo++;
    r->hi = (uint16_t)cfg->n_layer;
    while (r->hi > r->lo && !ds4x_model_layer(model, r->hi - 1)) r->hi--;
    if (r->lo >= r->hi) {
        if (err) snprintf(err, errlen, "model holds no layers");
        free(r);
        return NULL;
    }

    const uint32_t nloc = r->hi - r->lo;
    r->caches = (ds4x_kv_cache *)calloc(nloc, sizeof(ds4x_kv_cache));
    if (!r->caches) { free(r); return NULL; }
    const size_t kv_dim = (size_t)cfg->n_head_kv * cfg->head_dim;  /* GQA */
    /* Linear (Gated DeltaNet) state is ctx-INDEPENDENT — that is the point. */
    const size_t lin_state = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    const size_t lin_conv  = (size_t)(cfg->conv_kernel ? cfg->conv_kernel - 1 : 0) *
                             ((size_t)cfg->lin_k_heads * cfg->lin_k_dim * 2 +
                              (size_t)cfg->lin_v_heads * cfg->lin_v_dim);
    for (uint32_t i = 0; i < nloc; i++) {
        const uint32_t il = r->lo + i;
        const uint8_t lt = (il < DS4X_MAX_LAYERS && cfg->layer_types[il])
                           ? cfg->layer_types[il] : cfg->attn_kind;
        const int gqa = (lt == DS4X_ATTN_GQA);
        int ok;
        if (lt == DS4X_ATTN_LINEAR) {
            r->caches[i].state    = (float *)calloc(lin_state, sizeof(float));
            r->caches[i].conv_win = (float *)calloc(lin_conv ? lin_conv : 1, sizeof(float));
            ok = r->caches[i].state && r->caches[i].conv_win;
#ifdef IDLETOKEN_DS4X_CUDA
            if (ok && gdn_gpu_enabled()) {
                r->caches[i].dev_state =
                    ds4x_cuda_gdn_create(cfg->lin_k_heads, cfg->lin_v_heads,
                                         cfg->lin_k_dim, cfg->lin_v_dim);
                /* A failed upload is not fatal — the CPU recurrence is always
                 * valid — but say so, because a silent fall back to CPU here
                 * looks exactly like "the GPU is just slow". */
                if (!r->caches[i].dev_state)
                    fprintf(stderr, "ds4x: layer %u GDN state stays on CPU (%s)\n",
                            il, ds4x_cuda_last_error());
            }
#endif
        } else if (gqa) {
            /* full K/V cache; latent/k_rope stay NULL */
            r->caches[i].k = (float *)calloc((size_t)ctx_size * kv_dim, sizeof(float));
            r->caches[i].v = (float *)calloc((size_t)ctx_size * kv_dim, sizeof(float));
            ok = r->caches[i].k && r->caches[i].v;
        } else {
            r->caches[i].latent = (float *)calloc((size_t)ctx_size * cfg->kv_lora_rank, sizeof(float));
            r->caches[i].k_rope = (float *)calloc((size_t)ctx_size * cfg->qk_rope_head_dim, sizeof(float));
            ok = r->caches[i].latent && r->caches[i].k_rope;
        }
        if (!ok) {
            if (err) snprintf(err, errlen, "oom sizing KV cache for %u-ctx", ctx_size);
            ds4x_runner_free(r);
            return NULL;
        }
    }
    return r;
}

/* Last failure reason, for the worker's log. A bare -1 cost a real debugging
 * round on the cross-machine run (2026-07-28): the worker could only print
 * "ds4x_runner_run failed", which says nothing about WHICH layer or why. */
/* Wide enough to hold the forward's own reason (192) plus this layer/cache
 * context — truncating the reason would defeat the point of having one. */
static char g_run_err[384];

const char *ds4x_runner_last_error(void) {
    return g_run_err[0] ? g_run_err : "(no detail recorded)";
}

/* Zero the recurrent state of every linear-attention layer. Required at the
 * start of each new sequence: a GQA/MLA cache is indexed BY POSITION, so a new
 * sequence starting at pos 0 simply overwrites slots 0..n-1 and attention never
 * reads beyond `pos`. A Gated DeltaNet state has no position index at all — it
 * is a running accumulator, so without this the previous request's state leaks
 * into the next one. Symptom (2026-07-28, 3-node cluster): the SAME prompt at
 * temperature 0 produced different tokens on the 2nd request, and a later
 * request echoed its prompt instead of answering. */
void ds4x_runner_reset(ds4x_runner *r) {
    if (!r) return;
    const ds4x_config *cfg = r->cfg;
    const size_t lin_state = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    const size_t lin_conv  = (size_t)(cfg->conv_kernel ? cfg->conv_kernel - 1 : 0) *
                             ((size_t)cfg->lin_k_heads * cfg->lin_k_dim * 2 +
                              (size_t)cfg->lin_v_heads * cfg->lin_v_dim);
    for (uint32_t i = 0; i < (uint32_t)(r->hi - r->lo); i++) {
        if (r->caches[i].state)    memset(r->caches[i].state, 0, lin_state * sizeof(float));
        if (r->caches[i].conv_win) memset(r->caches[i].conv_win, 0, lin_conv * sizeof(float));
#ifdef IDLETOKEN_DS4X_CUDA
        /* The VRAM copy is the one the recurrence actually reads when it is
         * present — clearing only the host mirror would leak the previous
         * request straight through the GPU path. */
        if (r->caches[i].dev_state)
            ds4x_cuda_gdn_zero((ds4x_cuda_gdn *)r->caches[i].dev_state);
#endif
    }
}

int ds4x_runner_run(ds4x_runner *r, float *hidden, uint32_t n_tokens,
                    uint32_t pos0) {
    g_run_err[0] = '\0';
    /* pos0 == 0 means "a new sequence starts here" (chunked prefill only has
     * pos0 == 0 on its FIRST chunk), so this is the right place to clear the
     * recurrent state — it makes every caller correct without them knowing. */
    if (r && pos0 == 0) ds4x_runner_reset(r);
    if (!r || !hidden || n_tokens == 0) {
        snprintf(g_run_err, sizeof g_run_err, "bad args (runner=%p hidden=%p n=%u)",
                 (void *)r, (void *)hidden, n_tokens);
        return -1;
    }
    if ((size_t)pos0 + n_tokens > r->ctx) {
        snprintf(g_run_err, sizeof g_run_err,
                 "pos0 %u + n %u exceeds ctx %u", pos0, n_tokens, (unsigned)r->ctx);
        return -1;
    }
    for (uint16_t il = r->lo; il < r->hi; il++) {
        const ds4x_layer_weights *w = ds4x_model_layer(r->model, il);
        if (!w) {
            snprintf(g_run_err, sizeof g_run_err, "layer %u has no weights", il);
            return -1;
        }
        const ds4x_kv_cache *c = &r->caches[il - r->lo];
        if (ds4x_layer_forward_cpu(r->cfg, il, w, hidden, n_tokens, pos0,
                                   &r->caches[il - r->lo], hidden) != 0) {
            const uint8_t lt = (il < DS4X_MAX_LAYERS && r->cfg->layer_types[il])
                               ? r->cfg->layer_types[il] : r->cfg->attn_kind;
            /* The forward's own reason comes FIRST: it names the actual branch
             * (which slot, or which allocation and how big). The cache-slot
             * dump stays because it is the context that makes the reason
             * actionable. */
            snprintf(g_run_err, sizeof g_run_err,
                     "layer %u forward failed: %s (attn=%s; cache: state=%s "
                     "conv=%s k=%s v=%s latent=%s)", il,
                     ds4x_forward_last_error(),
                     lt == DS4X_ATTN_LINEAR ? "LINEAR" :
                     lt == DS4X_ATTN_GQA ? "GQA" : "MLA",
                     c->state ? "y" : "NULL", c->conv_win ? "y" : "NULL",
                     c->k ? "y" : "NULL", c->v ? "y" : "NULL",
                     c->latent ? "y" : "NULL");
            return -1;
        }
    }
    return 0;
}

void ds4x_runner_free(ds4x_runner *r) {
    if (!r) return;
    if (r->caches) {
        const uint32_t nloc = r->hi - r->lo;
        for (uint32_t i = 0; i < nloc; i++) {
            free(r->caches[i].latent);
            free(r->caches[i].k_rope);
            free(r->caches[i].k);
            free(r->caches[i].v);
            free(r->caches[i].state);
            free(r->caches[i].conv_win);
#ifdef IDLETOKEN_DS4X_CUDA
            ds4x_cuda_gdn_free((ds4x_cuda_gdn *)r->caches[i].dev_state);
#endif
        }
        free(r->caches);
    }
    free(r);
}
