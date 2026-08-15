/* model_auto.c — open model intake: GGUF header -> runtime model spec (WS-B4).
 * See include/idletoken_model_auto.h for the contract.
 *
 * Everything here reads the header region only (idletoken_gguf_meta_open); the
 * one whole-file fact used is the on-disk size (stat), which the scheduler
 * needs because llama.cpp keeps the entire file's weights resident.
 */
#include "idletoken_model_auto.h"
#include "idletoken_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GiB (1024ull * 1024 * 1024)

/* --- helpers -------------------------------------------------------------- */

static const char *path_basename(const char *path) {
    const char *b = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

/* "-00001-of-00004.gguf" style multi-file naming (llama.cpp split convention).
 * Detected from the NAME so the error can also fire on part 2..N, whose header
 * parses fine but whose tensors are elsewhere. */
static int looks_like_split_name(const char *base) {
    const char *dot = strrchr(base, '.');
    if (!dot || strcmp(dot, ".gguf") != 0) return 0;
    /* expect ...-NNNNN-of-NNNNN.gguf */
    const char *p = dot;                     /* points at ".gguf" */
    int digits = 0;
    const char *q = p;
    while (q > base && q[-1] >= '0' && q[-1] <= '9') { q--; digits++; }
    if (digits != 5) return 0;
    if (q - base < 4 || strncmp(q - 4, "-of-", 4) != 0) return 0;
    q -= 4;
    digits = 0;
    while (q > base && q[-1] >= '0' && q[-1] <= '9') { q--; digits++; }
    if (digits != 5) return 0;
    return q > base && q[-1] == '-';
}

/* Lowercase + keep [a-z0-9._], everything else collapses to a single '-'.
 * Result never empty (falls back to "model") and never starts/ends with '-'. */
static void sanitize_id(const char *name, char *out, size_t cap) {
    size_t o = 0;
    int pending_dash = 0;
    for (const char *p = name; *p && o + 1 < cap; p++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        const int keep = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                         c == '.' || c == '_';
        if (keep) {
            if (pending_dash && o > 0 && o + 2 < cap) out[o++] = '-';
            pending_dash = 0;
            out[o++] = c;
        } else {
            pending_dash = 1;   /* collapse runs; drop leading/trailing */
        }
    }
    out[o] = '\0';
    if (o == 0) snprintf(out, cap, "model");
}

/* <arch>.<suffix> u32 lookup. Returns 0/-1. */
static int arch_u32(const idletoken_gguf_meta *m, const char *arch,
                    const char *suffix, uint32_t *out) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    return idletoken_gguf_meta_u32(m, key, out);
}

/* --- tensor directory: layer vs shared byte split -------------------------
 * Tensor sizes are recovered from offset deltas (offsets are ascending in
 * every llama.cpp-family writer; we sort defensively anyway), which includes
 * each tensor's alignment padding — the conservative direction for a memory
 * planner. Classification: "blk.<n>.*" = per-layer, everything else
 * (token_embd, output, output_norm, rope factors...) = shared. */
static void split_layer_shared(const idletoken_gguf_meta *m, uint64_t data_bytes,
                               uint64_t *layer_out, uint64_t *shared_out) {
    *layer_out = 0;
    *shared_out = 0;
    const uint64_t n = idletoken_gguf_meta_n_tensors(m);
    if (n == 0 || data_bytes == 0) return;

    typedef struct { uint64_t off; uint8_t is_layer; } ent;
    ent *e = (ent *)malloc(sizeof(ent) * (size_t)n);
    if (!e) return;   /* split stays 0/0 — callers treat that as "unknown" */
    for (uint64_t i = 0; i < n; i++) {
        idletoken_gguf_tensor t;
        if (idletoken_gguf_tensor_info(m, i, &t) != 0) { free(e); return; }
        e[i].off = t.offset;
        e[i].is_layer = strncmp(t.name, "blk.", 4) == 0;
    }
    /* insertion sort by offset: directories are already ordered in practice,
     * so this is O(n) there and merely correct elsewhere. */
    for (uint64_t i = 1; i < n; i++) {
        ent k = e[i];
        uint64_t j = i;
        while (j > 0 && e[j - 1].off > k.off) { e[j] = e[j - 1]; j--; }
        e[j] = k;
    }
    for (uint64_t i = 0; i < n; i++) {
        const uint64_t end = (i + 1 < n) ? e[i + 1].off : data_bytes;
        const uint64_t sz = end > e[i].off ? end - e[i].off : 0;
        if (e[i].is_layer) *layer_out += sz; else *shared_out += sz;
    }
    free(e);
}

/* --- KV bytes per context token (whole model) -----------------------------
 * f16 KV cache (llama.cpp default): per attention layer,
 *   n_head_kv × (key_length + value_length) × 2 bytes.
 * head_count_kv may be a PER-LAYER array (hybrid linear-attention models store
 * 0 on the recurrent layers, which correctly contributes no KV); a scalar
 * combined with <arch>.full_attention_interval means only every interval-th
 * layer holds a KV cache (Qwen3.5's encoding). The recurrent layers' fixed
 * state is small and ctx-independent — it is charged to the per-node overhead
 * constant, not here. */
static uint64_t kv_bytes_per_token(const idletoken_gguf_meta *m, const char *arch,
                                   uint32_t n_layers, uint32_t n_embd) {
    char key[128];
    uint32_t head_count = 0;
    (void)arch_u32(m, arch, "attention.head_count", &head_count);

    uint32_t key_len = 0, val_len = 0;
    (void)arch_u32(m, arch, "attention.key_length", &key_len);
    (void)arch_u32(m, arch, "attention.value_length", &val_len);
    if (key_len == 0 && head_count > 0) key_len = n_embd / head_count;
    if (val_len == 0) val_len = key_len;

    /* Per-layer kv-head array first (qwen3next-style hybrids). */
    snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
    uint64_t arr_n = 0;
    if (idletoken_gguf_meta_arr_len(m, key, &arr_n) == 0 && arr_n > 0) {
        uint64_t total = 0;
        for (uint64_t i = 0; i < arr_n; i++) {
            int32_t v = 0;
            if (idletoken_gguf_meta_arr_i32(m, key, i, &v) == 0 && v > 0)
                total += (uint64_t)v * (key_len + val_len) * 2;
        }
        return total;
    }

    uint32_t n_kv = 0;
    if (arch_u32(m, arch, "attention.head_count_kv", &n_kv) != 0 || n_kv == 0) {
        if (head_count == 0) return 0;   /* no attention shape at all */
        n_kv = head_count;               /* plain MHA */
    }
    if (key_len == 0) return 0;

    uint32_t attn_layers = n_layers;
    uint32_t interval = 0;
    if (arch_u32(m, arch, "full_attention_interval", &interval) == 0 && interval > 1)
        attn_layers = (n_layers + interval - 1) / interval;

    return (uint64_t)attn_layers * n_kv * (key_len + val_len) * 2;
}

/* --- public entry ---------------------------------------------------------- */

int idletoken_model_from_gguf(const char *path, idletoken_auto_model *out,
                              char *err, size_t errlen) {
#define FAILF(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); \
                        idletoken_gguf_meta_close(m); return -1; } while (0)
    if (!path || !out) {
        if (err && errlen) snprintf(err, errlen, "no path");
        return -1;
    }
    memset(out, 0, sizeof(*out));

    const char *base = path_basename(path);
    snprintf(out->gguf_name, sizeof(out->gguf_name), "%s", base);

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "cannot stat %s", path);
        return -1;
    }
    out->file_bytes = (uint64_t)st.st_size;

    char merr[256] = "";
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(path, merr, sizeof(merr));
    if (!m) {
        if (err && errlen) snprintf(err, errlen, "%s", merr);
        return -1;
    }

    /* Split (multi-file) GGUFs: refuse with instructions, both by filename
     * convention and by the split.count metadata (either alone can be absent —
     * renamed part files keep the KV, merged-then-renamed files keep neither). */
    uint32_t split_count = 0;
    (void)idletoken_gguf_meta_u32(m, "split.count", &split_count);
    if (split_count > 1 || looks_like_split_name(base))
        FAILF("%s is one part of a multi-file (split) GGUF; loading split models "
              "is not supported yet. Merge it first (llama-gguf-split --merge) "
              "and point at the single merged .gguf file.", base);

    if (idletoken_gguf_meta_str(m, "general.architecture",
                                out->arch, sizeof(out->arch)) != 0)
        FAILF("%s has no general.architecture — not a model GGUF this build can "
              "serve", base);

    /* Label + id: general.name, else the filename stem. */
    if (idletoken_gguf_meta_str(m, "general.name",
                                out->label, sizeof(out->label)) != 0 ||
        out->label[0] == '\0') {
        snprintf(out->label, sizeof(out->label), "%s", base);
        char *dot = strrchr(out->label, '.');
        if (dot && !strcmp(dot, ".gguf")) *dot = '\0';
    }
    sanitize_id(out->label, out->id, sizeof(out->id));

    uint32_t n_layers = 0;
    if (arch_u32(m, out->arch, "block_count", &n_layers) != 0 || n_layers == 0)
        FAILF("%s (arch %s) declares no %s.block_count — cannot plan a model "
              "with an unknown layer count", base, out->arch, out->arch);

    uint32_t n_embd = 0;
    (void)arch_u32(m, out->arch, "embedding_length", &n_embd);

    uint32_t ctx_max = 0;
    (void)arch_u32(m, out->arch, "context_length", &ctx_max);
    if (ctx_max == 0) ctx_max = 32768;   /* header silent: a served-context cap
                                          * still needs SOME value; 32K matches
                                          * the coordinator's default ask */

    /* vocab: explicit key first, else the token table's length. */
    uint32_t n_vocab = 0;
    if (arch_u32(m, out->arch, "vocab_size", &n_vocab) != 0 || n_vocab == 0) {
        uint64_t ntok = 0;
        if (idletoken_gguf_meta_arr_len(m, "tokenizer.ggml.tokens", &ntok) == 0)
            n_vocab = (uint32_t)ntok;
    }

    out->kv_bytes_per_token = kv_bytes_per_token(m, out->arch, n_layers, n_embd);

    uint64_t layer_b = 0, shared_b = 0;
    const uint64_t data_off = idletoken_gguf_data_offset(m);
    const uint64_t data_bytes =
        out->file_bytes > data_off ? out->file_bytes - data_off : 0;
    split_layer_shared(m, data_bytes, &layer_b, &shared_b);
    if (layer_b == 0 && shared_b == 0) {
        /* Metadata-only fixture or unclassifiable directory: put everything in
         * the layer bucket so total sizing still adds up to the file. */
        layer_b = data_bytes;
    }
    /* The header region itself is resident too (mmap'd); charge it to shared. */
    shared_b += data_off < out->file_bytes ? data_off : out->file_bytes;

    idletoken_gguf_meta_close(m);
    m = NULL;

    idletoken_model_spec *s = &out->spec;
    s->id      = out->id;
    s->label   = out->label;
    s->backend = IDLETOKEN_BACKEND_LLAMACPP;
    s->available = 1;
    /* Deployment is CLUSTER here NOT because every open model should span
     * machines, but because the static single/cluster flag is superseded by
     * the dynamic judgement (v2 plan §1.9): idletoken_plan_llamacpp() decides
     * per run from real bytes vs real memory, with "fits -> don't cluster" as
     * a hard rule. A static SINGLE_NODE here would veto legitimately large
     * open models before the scheduler ever saw the numbers. */
    s->deployment = IDLETOKEN_DEPLOY_CLUSTER;
    s->n_layers = (uint16_t)n_layers;
    s->n_embd   = n_embd;
    s->hc_streams = 1;
    s->n_vocab  = n_vocab;
    s->layer_weight_bytes  = layer_b;
    s->shared_weight_bytes = shared_b;
    s->ctx_max  = ctx_max;
    s->split_boundary_multiple = 0;
    s->kv_kind = IDLETOKEN_KV_GQA;
    s->kv_bytes_per_token_layer =
        n_layers ? (uint32_t)(out->kv_bytes_per_token / n_layers) : 0;
    /* Engine compute buffers + margin. ESTIMATE, calibration TODO (WS-B2):
     * llama.cpp's compute buffer scales with n_embd and batch size; 1 GiB
     * covers every model measured so far on the M4/DGX with headroom. */
    s->overhead_base_bytes = 1ull * GiB;
    s->default_gguf = out->gguf_name;
    s->variants = NULL;
    s->n_variants = 0;
    s->default_variant = 0;
    return 0;
#undef FAILF
}
