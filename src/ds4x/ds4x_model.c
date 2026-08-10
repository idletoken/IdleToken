/* ds4x_model.c — GGUF → ds4x weights (Phase B.3).
 *
 * Tensor naming follows llama.cpp's deepseek2-family mapping. Big matrices are
 * kept in their on-disk (possibly quantized) form and dequantized one row at a
 * time by the forward (idletoken_ds4x_quant), so a real model stays at ~its file
 * size in RAM instead of the 4× fp32 blowup. Small 1-D tensors (norms, bias)
 * and the assembled kv_b are materialized as fp32. */
/* _DEFAULT_SOURCE exposes MADV_* under -std=c99 (glibc hides them behind
 * __STRICT_ANSI__). Without the madvise hint, mapping a multi-GB GGUF is
 * SLOWER than fread: the CUDA upload then faults the file in page by page
 * (measured 39 s vs 33 s on a 4.7 GB Q4_K_M). Must precede every include. */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif

#include "idletoken_ds4x.h"
#include "idletoken_ds4x_quant.h"
#include "idletoken_gguf.h"
#ifdef IDLETOKEN_DS4X_CUDA
#include "idletoken_ds4x_cuda.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

struct ds4x_model {
    ds4x_config cfg;
    uint16_t layer_lo, layer_hi;
    ds4x_layer_weights *layers;   /* [layer_hi - layer_lo] */
    ds4x_wt token_embd;           /* [n_vocab][n_embd], stage 0 only */
    const float *output_norm;     /* [n_embd], last stage only */
    ds4x_wt output;               /* [n_vocab][n_embd] lm_head (tied→token_embd) */
    void **blobs;                 /* every allocation, for free() */
    size_t n_blobs, cap_blobs;
    /* Whole-file mapping: big matrices point INTO this instead of being copied
     * into fresh mallocs. Loading a 4.7 GB Q4_K_M by malloc+fread measured
     * ~30 s (≈157 MB/s — an extra copy plus page-zeroing per tensor); mapping
     * makes load essentially free and lets the page cache serve re-runs.
     * NULL = mapping unavailable (Windows / mmap failed) → the malloc+fread
     * fallback below is used and stays correct, just slower. */
    void  *map;
    size_t map_len;
};

static void *track_blob(ds4x_model *m, void *p) {
    if (!p) return NULL;
    if (m->n_blobs == m->cap_blobs) {
        m->cap_blobs = m->cap_blobs ? m->cap_blobs * 2 : 64;
        m->blobs = (void **)realloc(m->blobs, m->cap_blobs * sizeof(void *));
    }
    m->blobs[m->n_blobs++] = p;
    return p;
}

/* 64-bit file seek. `long` is 32-bit on Windows (LLP64), so plain fseek()
 * SILENTLY TRUNCATES any offset past 2 GB — it seeks somewhere else entirely
 * and the caller happily loads garbage. Real symptom (win-a, 2026-07-28): the
 * last PP stage owns the layers at the END of a 5 GB GGUF, every tensor it
 * read was garbage, and the model emitted token 0 ("!") forever while every
 * protocol log looked perfectly healthy. Linux never hit it (64-bit long). */
static int seek64(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* Locate + validate a tensor's element count; also returns its raw byte size
 * and type. Returns 0, -1 missing, -2 wrong shape / unsupported. */
static int find_checked(const idletoken_gguf_meta *meta, const char *name,
                        uint64_t want_elems, idletoken_gguf_tensor *t,
                        uint64_t *nbytes, char *err, size_t errlen) {
    if (idletoken_gguf_tensor_find(meta, name, t) != 0) return -1;
    uint64_t elems = 1;
    for (uint32_t d = 0; d < t->ndim; d++) elems *= t->dims[d];
    if (elems != want_elems) {
        if (err) snprintf(err, errlen, "tensor %s has %llu elems, want %llu",
                          name, (unsigned long long)elems, (unsigned long long)want_elems);
        return -2;
    }
    const uint32_t bc = ds4x_type_block_count(t->type);
    if (!ds4x_type_supported(t->type) || bc == 0 || elems % bc) {
        if (err) snprintf(err, errlen,
            "tensor %s: ggml type %u not readable (supported: F32=0 F16=1 "
            "Q4_0=2 Q8_0=8 Q2_K=10 Q4_K=12 Q5_K=13 Q6_K=14 IQ2_XXS=16 "
            "BF16=30) or not block-aligned", name, t->type);
        return -2;
    }
    *nbytes = (elems / bc) * ds4x_type_block_bytes(t->type);
    return 0;
}

/* Read the raw (possibly quantized) tensor bytes, keep them as-is → ds4x_wt.
 * The forward dequantizes per row. {NULL,0} on error/missing. */
static ds4x_wt read_wt(ds4x_model *m, const idletoken_gguf_meta *meta, FILE *f,
                       const char *name, uint64_t want_elems, int required,
                       char *err, size_t errlen) {
    ds4x_wt none = { NULL, 0, NULL, 0 };
    idletoken_gguf_tensor t; uint64_t nbytes;
    int rc = find_checked(meta, name, want_elems, &t, &nbytes, err, errlen);
    if (rc != 0) { if (rc == -1 && required && err) snprintf(err, errlen, "missing tensor %s", name); return none; }
    const uint64_t fpos = idletoken_gguf_data_offset(meta) + t.offset;
    /* Zero-copy: point straight into the file mapping. */
    if (m->map && fpos + nbytes <= m->map_len) {
        ds4x_wt w = { (const uint8_t *)m->map + fpos, t.type, NULL, 0 };
        return w;
    }
    uint8_t *raw = (uint8_t *)malloc((size_t)nbytes);
    if (!raw) { if (err) snprintf(err, errlen, "oom for %s", name); return none; }
    if (seek64(f, fpos) != 0 || fread(raw, 1, (size_t)nbytes, f) != nbytes) {
        free(raw); if (err) snprintf(err, errlen, "short read on %s", name); return none;
    }
    ds4x_wt w = { track_blob(m, raw), t.type, NULL, 0 };
    return w;
}

/* Read + dequantize a tensor to fp32 (for small 1-D weights / kv_b assembly). */
static const float *read_f32(ds4x_model *m, const idletoken_gguf_meta *meta, FILE *f,
                             const char *name, uint64_t want_elems, int required,
                             char *err, size_t errlen) {
    idletoken_gguf_tensor t; uint64_t nbytes;
    int rc = find_checked(meta, name, want_elems, &t, &nbytes, err, errlen);
    if (rc != 0) { if (rc == -1 && required && err) snprintf(err, errlen, "missing tensor %s", name); return NULL; }
    float *out = (float *)malloc((size_t)want_elems * sizeof(float));
    if (!out) { if (err) snprintf(err, errlen, "oom for %s", name); return NULL; }
    const uint64_t fpos = idletoken_gguf_data_offset(meta) + t.offset;
    /* Dequantize straight out of the mapping when we have one. */
    if (m->map && fpos + nbytes <= m->map_len) {
        if (ds4x_dequant_row(t.type, (const uint8_t *)m->map + fpos, out,
                             want_elems, err, errlen) != 0) { free(out); return NULL; }
        return (const float *)track_blob(m, out);
    }
    uint8_t *raw = (uint8_t *)malloc((size_t)nbytes);
    if (!raw) { free(out); if (err) snprintf(err, errlen, "oom for %s", name); return NULL; }
    if (seek64(f, fpos) != 0 || fread(raw, 1, (size_t)nbytes, f) != nbytes) {
        free(raw); free(out); if (err) snprintf(err, errlen, "short read on %s", name); return NULL;
    }
    if (ds4x_dequant_row(t.type, raw, out, want_elems, err, errlen) != 0) { free(raw); free(out); return NULL; }
    free(raw);
    return (const float *)track_blob(m, out);
}

/* Dequantize row `o` (n elements) of a ds4x_wt into buf (or return direct fp32). */
static const float *wt_row_of(ds4x_wt w, size_t o, uint32_t n, float *buf) {
    if (w.type == 0) return (const float *)w.data + o * (size_t)n;
    const uint32_t bc = ds4x_type_block_count(w.type);
    const uint64_t bb = ds4x_type_block_bytes(w.type);
    const uint8_t *p = (const uint8_t *)w.data + (o * (n / bc)) * bb;
    ds4x_dequant_row(w.type, p, buf, n, NULL, 0);
    return buf;
}

#ifdef IDLETOKEN_DS4X_CUDA
/* Every per-layer weight that can live in VRAM, with its matrix shape.
 *
 * ONE list, used by both the upload loop and ds4x_model_free. They used to be
 * two hand-maintained copies and had already drifted: the free list still only
 * knew the attention + dense-FFN tensors, so every linear-attention projection
 * uploaded on a hybrid model was never released. Two lists that must agree is
 * the same shape of bug as two copies of a build script.
 *
 * Returns how many entries were written (<= cap). */
typedef struct { ds4x_wt *w; uint32_t n_out, n_in; } ds4x_dev_slot;

static size_t ds4x_layer_dev_slots(ds4x_layer_weights *w, const ds4x_config *c,
                                   ds4x_dev_slot *out, size_t cap) {
    const uint64_t hd = (c->attn_kind == DS4X_ATTN_GQA) ? c->head_dim : 0;
    /* Gated attention makes q_proj TWICE as tall ([q_h | gate_h] per head).
     * Uploading it at the ungated height would push only half the matrix to
     * VRAM and leave the gate half of the output buffer unwritten — wrong
     * numbers, silently, only on the GPU path. */
    const uint64_t qrows = hd ? (uint64_t)c->n_head * hd * (c->attn_out_gate ? 2u : 1u) : 0;
    /* Linear (Gated DeltaNet) layers: their projections are the bulk of the
     * per-layer FLOPs (in_proj_qkv alone is n_embd x (2*k_dim + v_dim)), so
     * leaving them on the CPU would waste the GPU on a hybrid model. */
    const uint64_t lin_kd = (uint64_t)c->lin_k_heads * c->lin_k_dim;
    const uint64_t lin_vd = (uint64_t)c->lin_v_heads * c->lin_v_dim;
    const uint64_t conv_ch = lin_kd * 2 + lin_vd;
    const ds4x_dev_slot tab[] = {
        { &w->q_proj,  (uint32_t)qrows, c->n_embd },
        { &w->k_proj,  (uint32_t)(hd ? c->n_head_kv * hd : 0), c->n_embd },
        { &w->v_proj,  (uint32_t)(hd ? c->n_head_kv * hd : 0), c->n_embd },
        { &w->attn_out, c->n_embd, (uint32_t)(hd ? c->n_head * hd : c->n_head * c->v_head_dim) },
        { &w->in_proj_qkv, (uint32_t)conv_ch, c->n_embd },
        { &w->in_proj_z,   (uint32_t)lin_vd,  c->n_embd },
        { &w->in_proj_b,   c->lin_v_heads,    c->n_embd },
        { &w->in_proj_a,   c->lin_v_heads,    c->n_embd },
        { &w->out_proj,    c->n_embd, (uint32_t)lin_vd },
        { &w->gate, c->n_ff_dense, c->n_embd },
        { &w->up,   c->n_ff_dense, c->n_embd },
        { &w->down, c->n_embd, c->n_ff_dense },
        /* MoE. Missing until 2026-08-04, so on a MoE model the GPU held the
         * attention projections while EVERY expert matvec — the bulk of the
         * FLOPs — ran on the CPU (findings R-08). Each is uploaded as ONE
         * stacked tensor; ds4x_forward reaches an individual expert through
         * wt_slice's dev_elem_off, so this is 3 handles per layer instead of
         * 3*n_expert. Over budget -> stays on the CPU, as always. */
        { &w->router, c->n_expert, c->n_embd },
        { &w->e_gate, (uint32_t)((uint64_t)c->n_expert * c->n_ff_exp), c->n_embd },
        { &w->e_up,   (uint32_t)((uint64_t)c->n_expert * c->n_ff_exp), c->n_embd },
        { &w->e_down, (uint32_t)((uint64_t)c->n_expert * c->n_embd),   c->n_ff_exp },
        { &w->s_gate, c->n_ff_exp, c->n_embd },
        { &w->s_up,   c->n_ff_exp, c->n_embd },
        { &w->s_down, c->n_embd,   c->n_ff_exp },
    };
    const size_t n = sizeof(tab) / sizeof(tab[0]);
    const size_t take = n < cap ? n : cap;
    for (size_t i = 0; i < take; i++) out[i] = tab[i];
    return take;
}
#define DS4X_DEV_SLOTS_MAX 32
#endif

ds4x_model *ds4x_model_load(const char *path, uint16_t layer_lo, uint16_t layer_hi,
                            char *err, size_t errlen) {
    ds4x_model *m = (ds4x_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    if (ds4x_config_from_gguf(path, &m->cfg, err, errlen) != 0) { free(m); return NULL; }
    const ds4x_config *c = &m->cfg;
    if (layer_hi == 0) layer_hi = (uint16_t)c->n_layer;
    if (layer_lo >= layer_hi || layer_hi > c->n_layer) {
        if (err) snprintf(err, errlen, "bad layer range [%u,%u) of %u", layer_lo, layer_hi, c->n_layer);
        free(m); return NULL;
    }
    m->layer_lo = layer_lo;
    m->layer_hi = layer_hi;
    m->layers = (ds4x_layer_weights *)calloc((size_t)(layer_hi - layer_lo), sizeof(ds4x_layer_weights));

    idletoken_gguf_meta *meta = idletoken_gguf_meta_open(path, err, errlen);
    FILE *f = fopen(path, "rb");
    if (!meta || !f) { if (f) fclose(f); if (meta) idletoken_gguf_meta_close(meta); ds4x_model_free(m); return NULL; }

#ifndef _WIN32
    /* Map the whole file read-only; weights point into it (see struct note).
     * Failure is non-fatal — read_wt/read_f32 fall back to malloc+fread. */
    {
        long cur = ftell(f);
        if (fseek(f, 0, SEEK_END) == 0) {
            long sz = ftell(f);
            if (sz > 0) {
                /* MAP_POPULATE pre-faults the whole file in one kernel pass.
                 * Without it the pages fault in DURING the CUDA upload, and a
                 * pageable H2D copy that faults runs at ~573 MB/s (measured:
                 * 7.6 s for 4.35 GB). Populating first lets the copy read
                 * already-resident memory. Linux-only flag; harmless if the
                 * kernel ignores it. */
                int mflags = MAP_PRIVATE;
#ifdef MAP_POPULATE
                mflags |= MAP_POPULATE;
#endif
                void *mp = mmap(NULL, (size_t)sz, PROT_READ, mflags, fileno(f), 0);
                if (mp != MAP_FAILED) {
                    m->map = mp;
                    m->map_len = (size_t)sz;
                    /* Weights are streamed once per token, so tell the kernel to
                     * read ahead rather than fault page-by-page. Advisory only:
                     * MADV_WILLNEED is not exposed under strict -std=c99
                     * (_POSIX_C_SOURCE) on glibc, so skip it when absent. */
#ifdef MADV_WILLNEED
                    madvise(mp, (size_t)sz, MADV_WILLNEED);
#endif
                }
            }
        }
        fseek(f, cur, SEEK_SET);
    }
#endif

    const uint64_t qd = (uint64_t)c->qk_nope_head_dim + c->qk_rope_head_dim;
    const uint64_t kvd = (uint64_t)c->qk_nope_head_dim + c->v_head_dim;
    char nm[160];
    int ok = 1;
    /* matrix weight (ds4x_wt, kept quantized) */
#define TW(field, suffix, elems, required) do { \
        if (ok) { snprintf(nm, sizeof(nm), "blk.%u." suffix, il); \
            w->field = read_wt(m, meta, f, nm, (elems), (required), err, errlen); \
            if (!w->field.data && (required)) ok = 0; } } while (0)
    /* small fp32 weight (norm / bias) */
#define TF(field, suffix, elems, required) do { \
        if (ok) { snprintf(nm, sizeof(nm), "blk.%u." suffix, il); \
            w->field = read_f32(m, meta, f, nm, (elems), (required), err, errlen); \
            if (!w->field && (required)) ok = 0; } } while (0)

    for (uint32_t il = layer_lo; il < layer_hi && ok; il++) {
        ds4x_layer_weights *w = &m->layers[il - layer_lo];
        TF(attn_norm, "attn_norm.weight", c->n_embd, 1);
      const uint8_t lt_ = (il < DS4X_MAX_LAYERS && c->layer_types[il])
                          ? c->layer_types[il] : c->attn_kind;
      if (c->arch == DS4X_ARCH_QWEN35) {
        /* Hybrid qwen35 — tensor names verified against a real GGUF
         * (docs/linear-attention-design.md §4b). NOTE there is no conv1d bias
         * tensor; conv1d_b stays NULL and the forward treats it as zero. */
        const uint64_t kd = (uint64_t)c->lin_k_heads * c->lin_k_dim;
        const uint64_t vd = (uint64_t)c->lin_v_heads * c->lin_v_dim;
        const uint64_t conv_ch = kd * 2 + vd;
        const uint64_t hd = c->head_dim;
        if (lt_ == DS4X_ATTN_LINEAR) {
            TW(in_proj_qkv, "attn_qkv.weight",   conv_ch * c->n_embd, 1);
            TW(in_proj_z,   "attn_gate.weight",  vd * c->n_embd, 1);
            TW(in_proj_b,   "ssm_beta.weight",   (uint64_t)c->lin_v_heads * c->n_embd, 1);
            TW(in_proj_a,   "ssm_alpha.weight",  (uint64_t)c->lin_v_heads * c->n_embd, 1);
            TF(conv1d_w,    "ssm_conv1d.weight", conv_ch * c->conv_kernel, 1);
            TF(A_log,       "ssm_a",             c->lin_v_heads, 1);
            TF(dt_bias,     "ssm_dt.bias",       c->lin_v_heads, 1);
            TF(ssm_norm,    "ssm_norm.weight",   c->lin_v_dim, 1);
            TW(out_proj,    "ssm_out.weight",    (uint64_t)c->n_embd * vd, 1);
        } else {
            /* attn_q carries [q | gate], hence the ×2 */
            TW(q_proj,   "attn_q.weight",      (uint64_t)c->n_head * hd * 2 * c->n_embd, 1);
            TW(k_proj,   "attn_k.weight",      (uint64_t)c->n_head_kv * hd * c->n_embd, 1);
            TW(v_proj,   "attn_v.weight",      (uint64_t)c->n_head_kv * hd * c->n_embd, 1);
            TF(q_norm,   "attn_q_norm.weight", hd, 1);
            TF(k_norm,   "attn_k_norm.weight", hd, 1);
            TW(attn_out, "attn_output.weight", (uint64_t)c->n_embd * c->n_head * hd, 1);
        }
        /* qwen35 calls the pre-FFN norm post_attention_norm */
        TF(ffn_norm, "post_attention_norm.weight", c->n_embd, 1);
        if (c->n_expert == 0) {
            TW(gate, "ffn_gate.weight", (uint64_t)c->n_ff_dense * c->n_embd, 1);
            TW(up,   "ffn_up.weight",   (uint64_t)c->n_ff_dense * c->n_embd, 1);
            TW(down, "ffn_down.weight", (uint64_t)c->n_embd * c->n_ff_dense, 1);
        } else {
            /* qwen35moe: same expert tensor names as the other MoE families,
             * plus a gate on the shared expert. */
            TW(router, "ffn_gate_inp.weight", (uint64_t)c->n_expert * c->n_embd, 1);
            TW(e_gate, "ffn_gate_exps.weight", (uint64_t)c->n_expert * c->n_ff_exp * c->n_embd, 1);
            TW(e_up,   "ffn_up_exps.weight",   (uint64_t)c->n_expert * c->n_ff_exp * c->n_embd, 1);
            TW(e_down, "ffn_down_exps.weight", (uint64_t)c->n_expert * c->n_embd * c->n_ff_exp, 1);
            if (c->n_expert_shared > 0) {
                const uint64_t sff = (uint64_t)c->n_ff_exp * c->n_expert_shared;
                TW(s_gate, "ffn_gate_shexp.weight", sff * c->n_embd, 1);
                TW(s_up,   "ffn_up_shexp.weight",   sff * c->n_embd, 1);
                TW(s_down, "ffn_down_shexp.weight", (uint64_t)c->n_embd * sff, 1);
                TW(s_gate_inp, "ffn_gate_inp_shexp.weight", c->n_embd, 1);
            }
        }
      } else if (c->attn_kind == DS4X_ATTN_GQA) {
        /* standard GQA (qwen3/llama): separate Q/K/V + optional per-head norms */
        const uint64_t hd = c->head_dim;
        TW(q_proj, "attn_q.weight", (uint64_t)c->n_head * hd * c->n_embd, 1);
        TW(k_proj, "attn_k.weight", (uint64_t)c->n_head_kv * hd * c->n_embd, 1);
        TW(v_proj, "attn_v.weight", (uint64_t)c->n_head_kv * hd * c->n_embd, 1);
        if (c->qk_norm) {
            TF(q_norm, "attn_q_norm.weight", hd, 1);
            TF(k_norm, "attn_k_norm.weight", hd, 1);
        }
        TW(attn_out, "attn_output.weight", (uint64_t)c->n_embd * c->n_head * hd, 1);
      } else {
        if (c->q_lora_rank > 0) {
            TW(q_a,       "attn_q_a.weight", (uint64_t)c->q_lora_rank * c->n_embd, 1);
            TF(q_a_norm,  "attn_q_a_norm.weight", c->q_lora_rank, 1);
            TW(q_b,       "attn_q_b.weight", (uint64_t)c->n_head * qd * c->q_lora_rank, 1);
        } else {
            TW(q_proj,    "attn_q.weight", (uint64_t)c->n_head * qd * c->n_embd, 1);
        }
        TW(kv_a,      "attn_kv_a_mqa.weight",
           ((uint64_t)c->kv_lora_rank + c->qk_rope_head_dim) * c->n_embd, 1);
        TF(kv_a_norm, "attn_kv_a_norm.weight", c->kv_lora_rank, 1);
        /* kv_b: combined attn_kv_b OR split attn_k_b/attn_v_b (assembled to
         * fp32; small, and the forward indexes per-head sub-blocks). Split
         * shapes from llama.cpp: wk_b {nope,kv_rank,nh} → [kv_rank][nope]
         * (TRANSPOSE), wv_b {kv_rank,v_dim,nh} → [v_dim][kv_rank] (as-is). */
        if (ok) {
            const uint64_t kr = c->kv_lora_rank, nh = c->n_head;
            const uint64_t np = c->qk_nope_head_dim, vd = c->v_head_dim;
            const float *comb_src = read_f32(m, meta, f,
                (snprintf(nm, sizeof(nm), "blk.%u.attn_kv_b.weight", il), nm),
                nh * kvd * kr, 0, NULL, 0);
            if (comb_src) {
                w->kv_b.data = comb_src; w->kv_b.type = 0;
            } else {
                const float *kb = read_f32(m, meta, f, (snprintf(nm, sizeof(nm),
                    "blk.%u.attn_k_b.weight", il), nm), nh * np * kr, 0, NULL, 0);
                const float *vb = read_f32(m, meta, f, (snprintf(nm, sizeof(nm),
                    "blk.%u.attn_v_b.weight", il), nm), nh * vd * kr, 0, NULL, 0);
                if (!kb || !vb) {
                    if (err) snprintf(err, errlen, "layer %u: no attn_kv_b and no "
                        "attn_k_b/attn_v_b pair", il);
                    ok = 0;
                } else {
                    float *comb = (float *)malloc((size_t)nh * (np + vd) * kr * sizeof(float));
                    if (!comb) { if (err) snprintf(err, errlen, "oom kv_b"); ok = 0; }
                    else {
                        for (uint64_t hd = 0; hd < nh; hd++) {
                            float *dst = comb + hd * (np + vd) * kr;
                            const float *ks = kb + hd * kr * np;
                            const float *vs = vb + hd * vd * kr;
                            for (uint64_t j = 0; j < np; j++)
                                for (uint64_t r = 0; r < kr; r++)
                                    dst[j * kr + r] = ks[r * np + j];
                            memcpy(dst + np * kr, vs, (size_t)vd * kr * sizeof(float));
                        }
                        w->kv_b.data = track_blob(m, comb); w->kv_b.type = 0;
                    }
                }
            }
        }
        TW(attn_out,  "attn_output.weight", (uint64_t)c->n_embd * c->n_head * c->v_head_dim, 1);
      }
      if (c->arch != DS4X_ARCH_QWEN35) {
        TF(ffn_norm,  "ffn_norm.weight", c->n_embd, 1);
        if (il < c->n_dense_lead || c->n_expert == 0) {
            TW(gate, "ffn_gate.weight", (uint64_t)c->n_ff_dense * c->n_embd, 1);
            TW(up,   "ffn_up.weight",   (uint64_t)c->n_ff_dense * c->n_embd, 1);
            TW(down, "ffn_down.weight", (uint64_t)c->n_embd * c->n_ff_dense, 1);
        } else {
            TW(router, "ffn_gate_inp.weight", (uint64_t)c->n_expert * c->n_embd, 1);
            TF(e_score_bias, "exp_probs_b.bias", c->n_expert, 0);   /* optional */
            TW(e_gate, "ffn_gate_exps.weight", (uint64_t)c->n_expert * c->n_ff_exp * c->n_embd, 1);
            TW(e_up,   "ffn_up_exps.weight",   (uint64_t)c->n_expert * c->n_ff_exp * c->n_embd, 1);
            TW(e_down, "ffn_down_exps.weight", (uint64_t)c->n_expert * c->n_embd * c->n_ff_exp, 1);
            if (c->n_expert_shared > 0) {
                const uint64_t sff = (uint64_t)c->n_ff_exp * c->n_expert_shared;
                TW(s_gate, "ffn_gate_shexp.weight", sff * c->n_embd, 1);
                TW(s_up,   "ffn_up_shexp.weight",   sff * c->n_embd, 1);
                TW(s_down, "ffn_down_shexp.weight", (uint64_t)c->n_embd * sff, 1);
            }
        }
      }
    }
#undef TW
#undef TF

    if (ok && layer_lo == 0) {
        m->token_embd = read_wt(m, meta, f, "token_embd.weight",
                                (uint64_t)c->n_vocab * c->n_embd, 1, err, errlen);
        ok = m->token_embd.data != NULL;
    }
    if (ok && layer_hi == c->n_layer) {
        m->output_norm = read_f32(m, meta, f, "output_norm.weight", c->n_embd, 1, err, errlen);
        ok = m->output_norm != NULL;
        if (ok) {
            m->output = read_wt(m, meta, f, "output.weight",
                                (uint64_t)c->n_vocab * c->n_embd, 0, NULL, 0);
            /* Tied embeddings: the lm_head IS token_embd. On a single stage we
             * already hold it, but under PP the last stage does NOT own layer 0
             * and so never loaded it — read it here. Qwen3-8B hid this because
             * its GGUF ships an explicit output.weight; Qwen3.5 is tied, and
             * without this the last stage silently fell back to mock
             * ("neither output.weight nor token_embd"). */
            if (!m->output.data && layer_lo != 0)
                m->output = read_wt(m, meta, f, "token_embd.weight",
                                    (uint64_t)c->n_vocab * c->n_embd, 0, NULL, 0);
            if (!m->output.data) m->output = m->token_embd;
            if (!m->output.data) {
                if (err) snprintf(err, errlen,
                    "last stage has no lm_head: neither output.weight nor "
                    "token_embd.weight (tied) found in the GGUF");
                ok = 0;
            }
        }
    }

    idletoken_gguf_meta_close(meta);
    fclose(f);
    if (!ok) { ds4x_model_free(m); return NULL; }

#ifdef IDLETOKEN_DS4X_CUDA
    /* VRAM residency for the big projections. Weights stay QUANTIZED on the
     * device — a Q4_K_M 8B is ~4.7 GB, so it fits a 12-16 GB card whole. Any
     * upload that fails just stays CPU-side (mixed CPU/GPU is correct, only
     * slower) — never a hard failure.
     *
     * The GPU is the DEFAULT, opt out with IDLETOKEN_DS4X_CPU=1. It used to be
     * opt-IN via IDLETOKEN_DS4X_CUDA=1, which meant every small model ran on the
     * CPU for anyone who did not know to set an environment variable — i.e. for
     * every GUI user, on six of the seven runnable models. Correct but many
     * times too slow, and nothing in the UI said why (principle 10:
     * running is not the same as running well). IDLETOKEN_DS4X_CUDA is still
     * accepted so old scripts and docs
     * keep working; it simply no longer has to be set. */
    if (!ds4x_gpu_wanted() && ds4x_cuda_available()) {
        fprintf(stderr, "ds4x: IDLETOKEN_DS4X_CPU=1 — staying on the CPU even "
                        "though a usable CUDA device is present\n");
    }
    if (ds4x_gpu_wanted() && !ds4x_cuda_available()) {
        /* Asked for the GPU and didn't get it — say so. Falling back to CPU
         * silently is how a 3-node cluster ended up 15x slower than expected
         * with every log looking fine (the 2070 box's driver 555.85 caps at
         * CUDA 12.5 while the DLL was built against 13.3). */
        fprintf(stderr, "ds4x: no usable CUDA device — running on CPU "
                        "(much slower). Reason: %s\n",
                ds4x_cuda_last_error());
    }
    if (ds4x_gpu_wanted() && ds4x_cuda_available()) {
        int up = 0, skip = 0;
        for (uint32_t il = layer_lo; il < layer_hi; il++) {
            ds4x_layer_weights *w = &m->layers[il - layer_lo];
            ds4x_dev_slot tab[DS4X_DEV_SLOTS_MAX];
            const size_t nslot = ds4x_layer_dev_slots(w, c, tab, DS4X_DEV_SLOTS_MAX);
            for (size_t t = 0; t < nslot; t++) {
                if (!tab[t].w->data || tab[t].n_out == 0 || tab[t].n_in == 0) continue;
                void *d = ds4x_cuda_upload(tab[t].w->data, tab[t].w->type,
                                           tab[t].n_out, tab[t].n_in);
                if (d) { tab[t].w->dev = d; up++; } else skip++;
            }
        }
        /* The LM head is the single biggest matvec in the whole model
         * (n_vocab x n_embd = 151936 x 4096 for Qwen3 ~ 622 M MACs per token —
         * more than a whole layer). Leaving it on the CPU made it ~92% of
         * decode time even after every layer weight was on the GPU. */
        if (m->output.data && c->n_vocab && c->n_embd) {
            void *d = ds4x_cuda_upload(m->output.data, m->output.type,
                                       c->n_vocab, c->n_embd);
            if (d) { m->output.dev = d; up++; } else skip++;
        }
        double ms_malloc = 0, ms_h2d = 0;
        ds4x_cuda_upload_stats(&ms_malloc, &ms_h2d);
        fprintf(stderr, "ds4x: CUDA on %s — %d weights in VRAM (%.2f GB), %d on CPU"
                        "  [cudaMalloc %.2f s, H2D %.2f s]\n",
                ds4x_cuda_device_name(), up,
                (double)ds4x_cuda_bytes_resident() / 1073741824.0, skip,
                ms_malloc / 1000.0, ms_h2d / 1000.0);
    }
#endif
    return m;
}

int ds4x_embed_tokens(const ds4x_model *m, const int32_t *tokens, uint32_t n, float *out) {
    if (!m->token_embd.data) return -1;
    const uint32_t d = m->cfg.n_embd;
    for (uint32_t t = 0; t < n; t++) {
        if (tokens[t] < 0 || (uint32_t)tokens[t] >= m->cfg.n_vocab) return -1;
        wt_row_of(m->token_embd, (size_t)tokens[t], d, out + (size_t)t * d);
        /* wt_row_of returns a direct pointer for fp32 — copy in that case. */
        if (m->token_embd.type == 0)
            memcpy(out + (size_t)t * d,
                   (const float *)m->token_embd.data + (size_t)tokens[t] * d,
                   d * sizeof(float));
    }
    return 0;
}

int ds4x_output_logits(const ds4x_model *m, const float *hidden, float *logits) {
    if (!m->output_norm || !m->output.data) return -1;
    const uint32_t d = m->cfg.n_embd, v = m->cfg.n_vocab;
    float *normed = (float *)malloc((size_t)d * sizeof(float));
    float *rowbuf = (float *)malloc((size_t)d * sizeof(float));
    if (!normed || !rowbuf) { free(normed); free(rowbuf); return -1; }
    float ss = 0.0f;
    for (uint32_t i = 0; i < d; i++) ss += hidden[i] * hidden[i];
    const float inv = 1.0f / sqrtf(ss / (float)d + 1e-6f);
    for (uint32_t i = 0; i < d; i++) normed[i] = hidden[i] * m->output_norm[i] * inv;
#ifdef IDLETOKEN_DS4X_CUDA
    if (m->output.dev &&
        ds4x_cuda_matvec((const ds4x_cuda_wt *)m->output.dev, normed, logits) == 0) {
        free(normed); free(rowbuf);
        return 0;
    }
#endif
    for (uint32_t o = 0; o < v; o++) {
        const float *row = wt_row_of(m->output, o, d, rowbuf);
        float acc = 0.0f;
        for (uint32_t i = 0; i < d; i++) acc += row[i] * normed[i];
        logits[o] = acc;
    }
    free(normed); free(rowbuf);
    return 0;
}

const ds4x_config *ds4x_model_config(const ds4x_model *m) { return &m->cfg; }

const ds4x_layer_weights *ds4x_model_layer(const ds4x_model *m, uint32_t il) {
    if (il < m->layer_lo || il >= m->layer_hi) return NULL;
    return &m->layers[il - m->layer_lo];
}

void ds4x_model_free(ds4x_model *m) {
    if (!m) return;
#ifdef IDLETOKEN_DS4X_CUDA
    if (m->output.dev) { ds4x_cuda_free((ds4x_cuda_wt *)m->output.dev); m->output.dev = NULL; }
    if (m->layers) {
        for (uint32_t il = m->layer_lo; il < m->layer_hi; il++) {
            ds4x_layer_weights *w = &m->layers[il - m->layer_lo];
            ds4x_dev_slot tab[DS4X_DEV_SLOTS_MAX];
            const size_t nslot = ds4x_layer_dev_slots(w, &m->cfg, tab, DS4X_DEV_SLOTS_MAX);
            for (size_t t = 0; t < nslot; t++)
                if (tab[t].w->dev) {
                    ds4x_cuda_free((ds4x_cuda_wt *)tab[t].w->dev);
                    tab[t].w->dev = NULL;
                }
        }
    }
#endif
    for (size_t i = 0; i < m->n_blobs; i++) free(m->blobs[i]);
    free(m->blobs);
    free(m->layers);
#ifndef _WIN32
    if (m->map) munmap(m->map, m->map_len);
#endif
    free(m);
}
