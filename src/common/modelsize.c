/* modelsize.c — resolve the scheduler's model size from the file that will
 * really be loaded. See include/idletoken_modelsize.h for the contract and the
 * measurement that made it necessary.
 */
#include "idletoken_modelsize.h"
#include "idletoken_gguf.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GiB (1024.0 * 1024.0 * 1024.0)

/* How far a file may sit from a manifest variant's byte count and still be
 * called that quantization. The manifest sums tensor bytes; the file also
 * carries its header/metadata region, so exact equality is not guaranteed even
 * when the two describe the same file.
 *
 * 1% is not arbitrary: the closest neighbours in any shipped menu are the 27B's
 * Q4_0 (15721973664) and Q4_K_S (15769159584), 0.3% apart. So a tolerance this
 * wide CAN cover two candidates — which is why the match takes the NEAREST and
 * says when the runner-up was also inside the window, instead of taking the
 * first hit and sounding certain. The budget itself is unaffected either way:
 * it always uses the real byte count, never the variant's. */
#define QUANT_MATCH_TOLERANCE 0.01

static const char *basename_of(const char *path) {
    const char *b = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

int idletoken_gguf_split_parts(const char *base, unsigned *idx, unsigned *total) {
    if (!base) return 0;
    const char *dot = strrchr(base, '.');
    if (!dot || strcmp(dot, ".gguf") != 0) return 0;
    const char *q = dot;
    int digits = 0;
    while (q > base && q[-1] >= '0' && q[-1] <= '9') { q--; digits++; }
    if (digits != 5) return 0;
    const char *tot_at = q;
    if (q - base < 4 || strncmp(q - 4, "-of-", 4) != 0) return 0;
    q -= 4;
    digits = 0;
    while (q > base && q[-1] >= '0' && q[-1] <= '9') { q--; digits++; }
    if (digits != 5) return 0;
    if (!(q > base && q[-1] == '-')) return 0;
    if (idx)   *idx   = (unsigned)strtoul(q, NULL, 10);
    if (total) *total = (unsigned)strtoul(tot_at, NULL, 10);
    return 1;
}

uint64_t idletoken_gguf_bytes_on_disk(const char *path, char *why, size_t why_cap) {
#define BAIL(...) do { if (why && why_cap) snprintf(why, why_cap, __VA_ARGS__); \
                       return 0; } while (0)
    if (!path || !path[0]) BAIL("no GGUF path given");

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0)
        BAIL("cannot read %s (missing, unreadable, or empty)", path);
    uint64_t total = (uint64_t)st.st_size;

    const char *base = basename_of(path);
    unsigned part_idx = 0, part_total = 0;
    if (!idletoken_gguf_split_parts(base, &part_idx, &part_total) || part_total < 2)
        return total;

    /* Only part 1 carries the header, and llama.cpp finds the rest by name.
     * Pointing at part 3 is a user error worth naming rather than silently
     * sizing the model at one part's worth. */
    if (part_idx != 1)
        BAIL("%s is part %u of %u of a split GGUF; point at part 1 "
             "(-00001-of-%05u.gguf), which carries the model header",
             base, part_idx, part_total, part_total);

    char sib[1024];
    const size_t plen = strlen(path);
    const size_t SUFFIX = strlen("-00001-of-00001.gguf");
    if (plen < SUFFIX || plen >= sizeof(sib))
        BAIL("cannot derive the sibling parts of %s", base);
    for (unsigned i = 2; i <= part_total; i++) {
        memcpy(sib, path, plen + 1);
        /* Every part name has identical length by construction, so the 5-digit
         * index is rewritten in place. */
        char *tail = sib + plen - strlen(".gguf") - strlen("-00001-of-00001") + 1;
        char idxbuf[6];
        snprintf(idxbuf, sizeof(idxbuf), "%05u", i);
        memcpy(tail, idxbuf, 5);
        struct stat sst;
        if (stat(sib, &sst) != 0 || sst.st_size <= 0)
            BAIL("%s is part 1 of %u but part %u is missing (%s) — the download "
                 "is incomplete; fetch the remaining part(s)",
                 base, part_total, i, sib);
        total += (uint64_t)sst.st_size;
    }
    return total;
#undef BAIL
}

static int mtp_arch_supported(const char *arch) {
    /* Actual DECODER_MTP graph implementations in the pinned engine. Keeping
     * unused NextN tensors (glm4/glm4moe) is not inference support. */
    static const char *const arches[] = {
        "deepseek2", "deepseek32", "deepseek4", "glm-dsa", "qwen35",
        "qwen35moe", "qwen3next", "hy-v3", "mimo2", "nemotron-h-moe",
        "step35", "bailingmoe3", "cohere2moe"
    };
    for (size_t i = 0; i < sizeof arches / sizeof arches[0]; i++)
        if (!strcmp(arch, arches[i])) return 1;
    return 0;
}

int idletoken_gguf_mtp_info(const char *path, uint16_t *layers,
                            uint64_t *weight_bytes, uint64_t *kv_bytes_per_token,
                            char *why, size_t why_cap) {
    idletoken_gguf_meta *m = NULL;
    uint8_t *seen = NULL;
    uint64_t weights = 0;
    if (layers) *layers = 0;
    if (weight_bytes) *weight_bytes = 0;
    if (kv_bytes_per_token) *kv_bytes_per_token = 0;
    if (why && why_cap) why[0] = '\0';
#define MTP_FAIL(...) do { if (why && why_cap) snprintf(why, why_cap, __VA_ARGS__); \
    idletoken_gguf_meta_close(m); free(seen); return -1; } while (0)
    if (!path || !path[0]) MTP_FAIL("no GGUF path for MTP inspection");
    m = idletoken_gguf_meta_open(path, why, why_cap);
    if (!m) return -1;
    char arch[64], key[128];
    if (idletoken_gguf_meta_str(m, "general.architecture", arch, sizeof arch))
        MTP_FAIL("GGUF has no architecture for MTP inspection");
    uint32_t blocks = 0, nextn = 0;
    snprintf(key, sizeof key, "%s.nextn_predict_layers", arch);
    (void)idletoken_gguf_meta_u32(m, key, &nextn);
    if (!nextn || !mtp_arch_supported(arch)) {
        idletoken_gguf_meta_close(m);
        return 0;
    }
    snprintf(key, sizeof key, "%s.block_count", arch);
    (void)idletoken_gguf_meta_u32(m, key, &blocks);
    if (nextn >= blocks || nextn > 65535)
        MTP_FAIL("invalid MTP block geometry (%u NextN of %u blocks)", nextn, blocks);
    uint32_t rank = 0, rope = 0, heads = 0, key_len = 0, value_len = 0;
#define MTP_META(suffix, dst) do { snprintf(key, sizeof key, "%s.%s", arch, suffix); \
    (void)idletoken_gguf_meta_u32(m, key, &(dst)); } while (0)
    MTP_META("attention.kv_lora_rank", rank);
    MTP_META("rope.dimension_count", rope);
    MTP_META("attention.head_count_kv", heads);
    MTP_META("attention.key_length", key_len);
    MTP_META("attention.value_length", value_len);
    if (!value_len) value_len = key_len;
    uint64_t kv = (rank ? (uint64_t)rank + rope :
                         (uint64_t)heads * (key_len + value_len)) * 2ull * nextn;
    uint32_t split_count = 0;
    (void)idletoken_gguf_meta_u32(m, "split.count", &split_count);
    unsigned idx = 1, total = 1;
    const int split = idletoken_gguf_split_parts(basename_of(path), &idx, &total);
    if ((split_count > 1 && (!split || total != split_count)) || idx != 1)
        MTP_FAIL("MTP inspection requires part 1 and all original split filenames");
    seen = (uint8_t *)calloc(nextn, 1);
    if (!seen) MTP_FAIL("out of memory inspecting MTP tensors");
    char part_path[1024];
    if (strlen(path) >= sizeof part_path) MTP_FAIL("GGUF path too long");
    for (unsigned part = 1; part <= total; part++) {
        snprintf(part_path, sizeof part_path, "%s", path);
        if (part > 1) {
            char digits[6];
            snprintf(digits, sizeof digits, "%05u", part);
            memcpy(part_path + strlen(part_path) - strlen("00001-of-00001.gguf"), digits, 5);
            m = idletoken_gguf_meta_open(part_path, why, why_cap);
            if (!m) { free(seen); return -1; }
        }
        struct stat st;
        const uint64_t start = idletoken_gguf_data_offset(m);
        if (stat(part_path, &st) || st.st_size <= 0 || (uint64_t)st.st_size < start)
            MTP_FAIL("cannot size MTP GGUF part %s", part_path);
        const uint64_t n = idletoken_gguf_meta_n_tensors(m);
        typedef struct { uint64_t offset; uint8_t mtp; } mtp_ent;
        mtp_ent *entries = (mtp_ent *)calloc(n ? (size_t)n : 1, sizeof(*entries));
        if (!entries) MTP_FAIL("out of memory inspecting MTP tensor offsets");
        for (uint64_t t = 0; t < n; t++) {
            idletoken_gguf_tensor info;
            if (idletoken_gguf_tensor_info(m, t, &info)) {
                free(entries); MTP_FAIL("invalid MTP tensor directory");
            }
            unsigned block = 0;
            int used = 0;
            entries[t].offset = info.offset;
            if (sscanf(info.name, "blk.%u.%n", &block, &used) == 1 && used &&
                block >= blocks - nextn && block < blocks) {
                entries[t].mtp = 1;
                const char *tail = info.name + used;
                uint8_t bit = !strcmp(tail, "nextn.eh_proj.weight") ? 1 :
                              !strcmp(tail, "nextn.enorm.weight") ? 2 :
                              !strcmp(tail, "nextn.hnorm.weight") ? 4 : 0;
                seen[block - (blocks - nextn)] |= bit;
            }
        }
        for (uint64_t t = 1; t < n; t++) {
            mtp_ent entry = entries[t];
            uint64_t j = t;
            while (j && entries[j - 1].offset > entry.offset) {
                entries[j] = entries[j - 1]; j--;
            }
            entries[j] = entry;
        }
        for (uint64_t t = 0; t < n; t++) {
            uint64_t end = t + 1 < n ? entries[t + 1].offset : (uint64_t)st.st_size - start;
            if (end < entries[t].offset) {
                free(entries); MTP_FAIL("invalid MTP tensor offsets");
            }
            if (entries[t].mtp) weights += end - entries[t].offset;
        }
        free(entries);
        idletoken_gguf_meta_close(m); m = NULL;
    }
    for (uint32_t i = 0; i < nextn; i++)
        if (seen[i] != 7) MTP_FAIL("GGUF declares MTP but block %u lacks required NextN tensors", blocks - nextn + i);
    free(seen);
    if (!weights || !kv) {
        if (why && why_cap) snprintf(why, why_cap, "MTP has incomplete weight/cache geometry");
        return -1;
    }
    if (layers) *layers = (uint16_t)nextn;
    if (weight_bytes) *weight_bytes = weights;
    if (kv_bytes_per_token) *kv_bytes_per_token = kv;
    return 0;
#undef MTP_META
#undef MTP_FAIL
}

/* Does this tensor belong to the exact family moved by llama.cpp's
 * `--n-cpu-moe` override? Keep this spelling aligned with the pinned engine's
 * LLM_FFN_EXPS_REGEX in vendor/llama.cpp/common/common.h. Regex-search there
 * also catches the bias/scale tensors after `_exps`, so a prefix test here is
 * deliberately broader than a `.weight` suffix test. */
static int tensor_is_moe_expert(const char *name, unsigned *layer_out) {
    unsigned layer = 0;
    int used = 0;
    if (!name || sscanf(name, "blk.%u.%n", &layer, &used) != 1 || used <= 0)
        return 0;
    const char *tail = name + used;
    if (strncmp(tail, "ffn_", 4) != 0) return 0;
    tail += 4;
    /* Pinned llama.cpp: ffn_(up|down|gate_up|gate)_(ch|)exps. Do not use a
     * loose `_exps` substring test: a future tensor such as ffn_norm_exps
     * would then be charged as CPU savings even though --n-cpu-moe leaves it
     * on the GPU. Longer alternatives first so gate_up is not read as gate. */
    static const char *roots[] = { "gate_up", "down", "gate", "up" };
    int matched = 0;
    for (size_t i = 0; i < sizeof roots / sizeof roots[0]; i++) {
        const size_t n = strlen(roots[i]);
        if (strncmp(tail, roots[i], n) != 0) continue;
        const char *suffix = tail + n;
        if (strncmp(suffix, "_exps", 5) == 0 ||
            strncmp(suffix, "_chexps", 7) == 0) {
            matched = 1;
            break;
        }
    }
    if (!matched) return 0;
    if (layer_out) *layer_out = layer;
    return 1;
}

static uint64_t tensor_name_fnv1a(const char *name) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        h ^= *p;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

/* Subset of tensor_is_moe_expert() that the scheduler can actually cache:
 * a routed-expert weight whose third GGML dimension is the expert axis. This
 * is metadata-derived and architecture-independent; bias/scale tensors remain
 * in the Hybrid capacity accounting above but never enter the pool contract. */
static int tensor_is_moe_pool_weight(const idletoken_gguf_tensor *t,
                                     uint32_t n_expert, unsigned *layer_out) {
    if (!t || t->ndim < 3 || t->dims[2] != n_expert) return 0;
    const size_t n = strlen(t->name);
    if (n < 7 || strcmp(t->name + n - 7, ".weight") != 0) return 0;
    return tensor_is_moe_expert(t->name, layer_out);
}

/* Add one GGUF part's padded tensor spans to the per-layer expert buckets.
 * Offsets, rather than a hand-maintained ggml dtype table, are the source of
 * truth; this is the same byte-accounting rule model_auto.c uses for the
 * layer/shared split. */
static int add_expert_part(const char *path, uint32_t n_layers,
                           uint32_t n_expert,
                           uint64_t per_layer[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint32_t pool_count[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint64_t pool_hash[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint64_t pool_slot[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint64_t pool_max[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint64_t block_bytes[IDLETOKEN_LLPLAN_MAX_LAYERS],
                           uint64_t *shared_bytes,
                           uint64_t *total, char *err, size_t err_cap) {
    char gerr[256] = "";
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(path, gerr, sizeof gerr);
    if (!m) {
        if (err && err_cap) snprintf(err, err_cap, "%s", gerr);
        return -1;
    }
    struct stat st;
    const uint64_t data_start = idletoken_gguf_data_offset(m);
    if (stat(path, &st) != 0 || st.st_size <= 0 || (uint64_t)st.st_size < data_start) {
        if (err && err_cap) snprintf(err, err_cap, "cannot size GGUF part %s", path);
        idletoken_gguf_meta_close(m);
        return -1;
    }
    const uint64_t data_bytes = (uint64_t)st.st_size - data_start;
    const uint64_t nt = idletoken_gguf_meta_n_tensors(m);
    typedef struct { uint64_t off; int layer; int block; int pool; } ent;
    ent *e = (ent *)calloc(nt ? (size_t)nt : 1, sizeof(*e));
    if (!e) {
        if (err && err_cap) snprintf(err, err_cap, "out of memory reading GGUF tensors");
        idletoken_gguf_meta_close(m);
        return -1;
    }
    for (uint64_t i = 0; i < nt; i++) {
        idletoken_gguf_tensor t;
        if (idletoken_gguf_tensor_info(m, i, &t) != 0 || t.offset > data_bytes) {
            if (err && err_cap) snprintf(err, err_cap, "invalid tensor directory in %s", path);
            free(e);
            idletoken_gguf_meta_close(m);
            return -1;
        }
        unsigned layer = 0;
        e[i].off = t.offset;
        unsigned block = 0;
        int consumed = 0;
        e[i].block = sscanf(t.name, "blk.%u.%n", &block, &consumed) == 1 && consumed > 0
            ? (int)block : -1;
        /* A block PAST the transformer stack is not a corrupt directory: GLM
         * and DeepSeek ship a multi-token-prediction block at index n_layer,
         * and llama.cpp still reports n_layer = 78. Treating it as corruption
         * failed the whole scan, so GLM-5.2 reported no expert layout at all
         * and Hybrid refused it on every machine — measured on a real 222 GiB
         * file, 2026-09-13, `blk.78.attn_k_b.weight names an out-of-range
         * block`. Its bytes are real and stay charged (to shared, below); what
         * it must never be is a block the planner hands out or spills, because
         * the engine does not place it as one. */
        const int past_stack = e[i].block >= 0 &&
            (block >= n_layers || block >= IDLETOKEN_LLPLAN_MAX_LAYERS);
        if (past_stack) e[i].block = -1;   /* charged to shared, below */
        e[i].layer = !past_stack && tensor_is_moe_expert(t.name, &layer)
            ? (int)layer : -1;
        if (e[i].layer >= 0 &&
            ((uint32_t)e[i].layer >= n_layers ||
             e[i].layer >= IDLETOKEN_LLPLAN_MAX_LAYERS)) {
            if (err && err_cap)
                snprintf(err, err_cap, "expert tensor %s names out-of-range layer %u",
                         t.name, layer);
            free(e);
            idletoken_gguf_meta_close(m);
            return -1;
        }
        unsigned pool_layer = 0;
        if (!past_stack && tensor_is_moe_pool_weight(&t, n_expert, &pool_layer)) {
            if (pool_layer >= n_layers ||
                pool_layer >= IDLETOKEN_LLPLAN_MAX_LAYERS ||
                pool_count[pool_layer] == UINT32_MAX) {
                if (err && err_cap)
                    snprintf(err, err_cap, "invalid expert-pool weight %s", t.name);
                free(e);
                idletoken_gguf_meta_close(m);
                return -1;
            }
            pool_count[pool_layer]++;
            pool_hash[pool_layer] += tensor_name_fnv1a(t.name);
            e[i].pool = 1;
        }
    }
    for (uint64_t i = 1; i < nt; i++) {
        ent key = e[i];
        uint64_t j = i;
        while (j > 0 && e[j - 1].off > key.off) { e[j] = e[j - 1]; j--; }
        e[j] = key;
    }
    for (uint64_t i = 0; i < nt; i++) {
        const uint64_t end = i + 1 < nt ? e[i + 1].off : data_bytes;
        if (end < e[i].off) {
            if (err && err_cap) snprintf(err, err_cap, "unsorted tensor spans in %s", path);
            free(e);
            idletoken_gguf_meta_close(m);
            return -1;
        }
        const uint64_t bytes = end - e[i].off;
        uint64_t *bucket = e[i].block >= 0 ? &block_bytes[e[i].block] : shared_bytes;
        if (UINT64_MAX - *bucket < bytes) {
            if (err && err_cap) snprintf(err, err_cap, "weight byte count overflow in %s", path);
            free(e);
            idletoken_gguf_meta_close(m);
            return -1;
        }
        *bucket += bytes;
        if (e[i].layer >= 0) {
            if (UINT64_MAX - per_layer[e[i].layer] < bytes ||
                UINT64_MAX - *total < bytes) {
                if (err && err_cap) snprintf(err, err_cap, "expert byte count overflow in %s", path);
                free(e);
                idletoken_gguf_meta_close(m);
                return -1;
            }
            per_layer[e[i].layer] += bytes;
            *total += bytes;
            if (e[i].pool) {
                const uint64_t slot = bytes / n_expert + (bytes % n_expert != 0);
                if (slot == 0 || UINT64_MAX - pool_slot[e[i].layer] < slot) {
                    if (err && err_cap) snprintf(err, err_cap, "invalid expert cache span in %s", path);
                    free(e);
                    idletoken_gguf_meta_close(m);
                    return -1;
                }
                pool_slot[e[i].layer] += slot;
                if (bytes > pool_max[e[i].layer]) pool_max[e[i].layer] = bytes;
            }
        }
    }
    const uint64_t prefix = data_start + (nt ? e[0].off : data_bytes);
    if (UINT64_MAX - *shared_bytes < prefix) {
        if (err && err_cap) snprintf(err, err_cap, "shared byte count overflow in %s", path);
        free(e);
        idletoken_gguf_meta_close(m);
        return -1;
    }
    *shared_bytes += prefix;
    free(e);
    idletoken_gguf_meta_close(m);
    return 0;
}

/* Recover the exact prefix savings available to `--n-cpu-moe N` from every
 * local GGUF part. This sits on the startup sizing path only; the header is a
 * few MiB and no tensor payload is read. */
static int gguf_expert_layout(const char *path, uint32_t n_layers,
                              idletoken_llm_model_size *out,
                              char *err, size_t err_cap) {
    if (!path || !path[0] || !out || n_layers == 0 ||
        n_layers > IDLETOKEN_LLPLAN_MAX_LAYERS) {
        if (err && err_cap) snprintf(err, err_cap, "unsupported MoE layer count");
        return -1;
    }
    const char *base = basename_of(path);
    unsigned part_idx = 0, part_total = 1;
    if (idletoken_gguf_split_parts(base, &part_idx, &part_total) && part_idx != 1) {
        if (err && err_cap) snprintf(err, err_cap, "MoE sizing requires split part 1");
        return -1;
    }
    if (part_total == 0) part_total = 1;
    const size_t plen = strlen(path);
    for (unsigned part = 1; part <= part_total; part++) {
        char current[1024];
        if (plen >= sizeof current) {
            if (err && err_cap) snprintf(err, err_cap, "GGUF path is too long");
            return -1;
        }
        memcpy(current, path, plen + 1);
        if (part_total > 1) {
            char *tail = current + plen - strlen(".gguf") - strlen("-00001-of-00001") + 1;
            char idxbuf[6];
            snprintf(idxbuf, sizeof idxbuf, "%05u", part);
            memcpy(tail, idxbuf, 5);
        }
        if (add_expert_part(current, n_layers, out->n_expert,
                            out->expert_bytes_per_layer,
                            out->expert_pool_weight_count_per_layer,
                            out->expert_pool_weight_hash_per_layer,
                            out->expert_pool_slot_bytes_per_layer,
                            out->expert_pool_max_tensor_per_layer,
                            out->weight_bytes_per_layer,
                            &out->weight_bytes_shared,
                            &out->expert_bytes_total, err, err_cap) != 0)
            return -1;
    }
    if (out->expert_bytes_total == 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "GGUF contains no tensors matched by --n-cpu-moe");
        return -1;
    }
    out->expert_bytes_complete = 1;
    return 0;
}

/* Spread the manifest's MEASURED expert total over the blocks that have
 * experts, for the case where no GGUF exists yet.
 *
 * The user is choosing a model here, and for a MoE model that choice is about
 * two pools. Without this the only honest answer before the download was "we
 * cannot say", and the card instead showed the GPU-only total — a placement
 * that model will never use — with no RAM figure at all.
 *
 * What is measured: the expert total and the largest expert tensor, per quant.
 * What is assumed: that the blocks carry equal shares. Real files sit within
 * about 20% of that on dynamic quants, which moves the chosen boundary by a
 * block or so and leaves the two pool totals close. `expert_bytes_estimated`
 * carries that caveat to everything downstream; the exact scan always wins
 * when the file is on disk. */
static void estimate_expert_layout(const idletoken_model_spec *spec,
                                   uint64_t layer_bytes, uint64_t shared_bytes,
                                   uint64_t expert_bytes, uint64_t max_tensor,
                                   idletoken_llm_model_size *out) {
    const uint32_t layers = spec->n_layers;
    const uint32_t first = spec->moe_first_layer < layers ? spec->moe_first_layer : 0;
    const uint32_t moe_layers = layers - first;
    if (layers == 0 || layers > IDLETOKEN_LLPLAN_MAX_LAYERS || moe_layers == 0 ||
        expert_bytes == 0 || expert_bytes >= layer_bytes || spec->n_expert == 0)
        return;

    /* Blocks first: the Hybrid budget prices each owner's own range, and
     * expert_buckets_consistent() checks that these buckets plus the shared
     * bytes are the whole model. Integer remainder goes to the leading blocks
     * so the sum is exact rather than "exact to within n_layers bytes". */
    const uint64_t block = layer_bytes / layers;
    const uint64_t block_rem = layer_bytes % layers;
    for (uint32_t i = 0; i < layers; i++)
        out->weight_bytes_per_layer[i] = block + (i < block_rem ? 1 : 0);
    out->weight_bytes_shared = shared_bytes;

    const uint64_t per = expert_bytes / moe_layers;
    const uint64_t rem = expert_bytes % moe_layers;
    for (uint32_t i = first; i < layers; i++) {
        uint64_t e = per + (i - first < rem ? 1 : 0);
        /* A block cannot be more expert than it is block. Equal shares of a
         * near-uniform model never hit this; clamping keeps a mis-measured
         * manifest from producing buckets the planner would reject outright. */
        if (e > out->weight_bytes_per_layer[i]) e = out->weight_bytes_per_layer[i];
        out->expert_bytes_per_layer[i] = e;
        out->expert_bytes_total += e;
        /* Pool geometry. The slot is one expert's share of the block, which is
         * the definition the exact path computes per tensor. The staging
         * buffer is the measured largest tensor, clamped to this block so the
         * cache geometry stays self-consistent. The weight COUNT is the usual
         * gate/up/down triple; it only buys a 512-byte allocation tail each,
         * so being wrong by one on a fused-projection model costs ~0.5 KiB. */
        out->expert_pool_slot_bytes_per_layer[i] = (e + spec->n_expert - 1) / spec->n_expert;
        out->expert_pool_max_tensor_per_layer[i] = max_tensor && max_tensor < e ? max_tensor : e;
        out->expert_pool_weight_count_per_layer[i] = 3;
    }
    out->expert_bytes_estimated = 1;
}

/* Nearest variant to `bytes`, or NULL when the spec ships no variant menu or
 * nothing is within tolerance. `*ambiguous` is set when a second variant also
 * fell inside the window. */
static const idletoken_model_variant *
nearest_variant(const idletoken_model_spec *m, uint64_t bytes, int *ambiguous) {
    if (ambiguous) *ambiguous = 0;
    if (!m || m->n_variants == 0 || !m->variants || bytes == 0) return NULL;

    const idletoken_model_variant *best = NULL;
    double best_rel = 0.0;
    int inside = 0;
    for (int i = 0; i < (int)m->n_variants; i++) {
        const idletoken_model_variant *v = &m->variants[i];
        const uint64_t vb = v->layer_weight_bytes + v->shared_weight_bytes;
        if (vb == 0) continue;
        const double rel = (double)(bytes > vb ? bytes - vb : vb - bytes) / (double)vb;
        if (rel <= QUANT_MATCH_TOLERANCE) inside++;
        if (!best || rel < best_rel) { best = v; best_rel = rel; }
    }
    if (!best || best_rel > QUANT_MATCH_TOLERANCE) return NULL;
    if (ambiguous) *ambiguous = inside > 1;
    return best;
}

/* Append to a bounded string without a second snprintf dance at each call. */
#if defined(__GNUC__)
__attribute__((format(printf, 3, 4)))
#endif
static void app(char *buf, size_t cap, const char *fmt, ...) {
    if (!buf || cap == 0) return;
    const size_t off = strlen(buf);
    if (off + 1 >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf + off, cap - off, fmt, ap);
    va_end(ap);
}

/* Re-point the workspace at a DIFFERENT KV tier than the weight quantization
 * implies. The only caller is the coordinator, when IDLETOKEN_KV_CACHE_TYPE
 * overrides the automatic rule: without this the budget would price the cache
 * the override selected against the workspace of the cache it replaced, and on
 * CUDA that gap reaches 4 GiB (qwen3.8-27b at 1M: 1146 MiB f16 vs 5200 MiB
 * quantized). `tier` < 0 means the forced dtype has no measurement of its own
 * (q5_1, iq4_nl, ...): charge the LARGEST measured tier, because the escape
 * hatch is a measurement tool and the one thing it must not do is quietly make
 * a configuration look cheaper than any we have ever measured. */
void idletoken_model_size_set_kv_tier(const idletoken_model_spec *spec,
                                      idletoken_llm_model_size *out, int tier) {
    if (!spec || !out) return;
    struct { uint64_t *dst; const uint64_t *src; } f[] = {
        { &out->compute_bytes_128k_cuda,  spec->compute_bytes_128k_cuda  },
        { &out->compute_bytes_256k_cuda,  spec->compute_bytes_256k_cuda  },
        { &out->compute_bytes_1m_cuda,    spec->compute_bytes_1m_cuda    },
        { &out->compute_bytes_128k_metal, spec->compute_bytes_128k_metal },
        { &out->compute_bytes_256k_metal, spec->compute_bytes_256k_metal },
        { &out->compute_bytes_1m_metal,   spec->compute_bytes_1m_metal   },
        { &out->mtp_compute_bytes_128k_cuda,  spec->mtp_compute_bytes_128k_cuda  },
        { &out->mtp_compute_bytes_256k_cuda,  spec->mtp_compute_bytes_256k_cuda  },
        { &out->mtp_compute_bytes_1m_cuda,    spec->mtp_compute_bytes_1m_cuda    },
        { &out->mtp_compute_bytes_128k_metal, spec->mtp_compute_bytes_128k_metal },
        { &out->mtp_compute_bytes_256k_metal, spec->mtp_compute_bytes_256k_metal },
        { &out->mtp_compute_bytes_1m_metal,   spec->mtp_compute_bytes_1m_metal   },
        { &out->mtp_host_compute_bytes_128k_cuda, spec->mtp_host_compute_bytes_128k_cuda },
        { &out->mtp_host_compute_bytes_256k_cuda, spec->mtp_host_compute_bytes_256k_cuda },
        { &out->mtp_host_compute_bytes_1m_cuda, spec->mtp_host_compute_bytes_1m_cuda },
        { &out->mtp_host_compute_bytes_128k_metal, spec->mtp_host_compute_bytes_128k_metal },
        { &out->mtp_host_compute_bytes_256k_metal, spec->mtp_host_compute_bytes_256k_metal },
        { &out->mtp_host_compute_bytes_1m_metal, spec->mtp_host_compute_bytes_1m_metal },
    };
    for (size_t i = 0; i < sizeof(f) / sizeof(f[0]); i++) {
        if (tier >= 0 && tier < IDLETOKEN_KV_TIER_COUNT) {
            *f[i].dst = f[i].src[tier];
            continue;
        }
        uint64_t max = 0;
        for (int t = 0; t < IDLETOKEN_KV_TIER_COUNT; t++)
            if (f[i].src[t] > max) max = f[i].src[t];
        *f[i].dst = max;
    }
    out->kv_tier = (uint8_t)(tier >= 0 && tier < IDLETOKEN_KV_TIER_COUNT
                                 ? tier : IDLETOKEN_KV_TIER_COUNT);
}

static const idletoken_model_spec *mtp_catalog_spec(const idletoken_model_spec *spec) {
    if (spec->mtp_draft) return spec;
    const idletoken_model_spec *known = idletoken_model_get(spec->id);
    return known ? known : spec;
}

static void apply_external_mtp_size(const idletoken_model_spec *spec,
                                    const char *quant, const char *path,
                                    idletoken_llm_model_size *out) {
    if (out->mtp_enabled) return; /* embedded head takes precedence */
    const idletoken_model_spec *known = mtp_catalog_spec(spec);
    const idletoken_model_variant *v = idletoken_model_variant_get(known, quant);
    if (quant && quant[0] && (!v || strcmp(v->quant, quant))) return;
    if (path && path[0]) {
        v = NULL;
        for (uint8_t i = 0; i < known->n_variants; ++i)
            if (!strcmp(basename_of(path), basename_of(known->variants[i].gguf))) {
                v = &known->variants[i]; break;
            }
    }
    const idletoken_model_mtp_draft *d = known->mtp_draft;
    if (!d || !v || !v->mtp_draft_compatible) return;
    out->mtp_enabled = 1;
    out->mtp_external_weight_bytes = d->weight_bytes;
    out->mtp_kv_bytes_per_token = d->kv_bytes_per_token;
    if (spec->kv_kind == IDLETOKEN_KV_HYBRID)
        out->kv_fixed_bytes_per_seq *= 4;
    idletoken_model_size_set_kv_tier(known, out, out->kv_tier);
}

int idletoken_mtp_draft_shard_path(const char *first_path, const char *remote_part,
                          char *out, size_t cap) {
    unsigned first_idx = 0, first_count = 0, part_idx = 0, part_count = 0;
    if (!first_path || !remote_part || !out || !cap || strlen(first_path) >= cap ||
        !idletoken_gguf_split_parts(basename_of(first_path), &first_idx, &first_count) ||
        !idletoken_gguf_split_parts(basename_of(remote_part), &part_idx, &part_count) ||
        first_idx != 1 || part_idx <= 1 || part_count != first_count || part_idx > part_count)
        return -1;
    snprintf(out, cap, "%s", first_path);
    char digits[6];
    snprintf(digits, sizeof digits, "%05u", part_idx);
    memcpy(out + strlen(out) - strlen("00001-of-00001.gguf"), digits, 5);
    return 0;
}

int idletoken_model_size_validate_mtp_draft(const idletoken_model_spec *spec,
        const char *quant, const char *target_path, const char *draft_path,
        const idletoken_llm_model_size *size, char *why, size_t why_cap) {
    (void)quant;
    idletoken_gguf_meta *target = NULL, *draft = NULL;
#define DRAFT_FAIL(...) do { if (why && why_cap) snprintf(why, why_cap, __VA_ARGS__); \
    idletoken_gguf_meta_close(target); idletoken_gguf_meta_close(draft); return -1; } while (0)
    if (!spec || !size) DRAFT_FAIL("invalid standalone MTP admission");
    const idletoken_model_spec *known = mtp_catalog_spec(spec);
    const idletoken_model_mtp_draft *d = known->mtp_draft;
    if (!size->mtp_external_weight_bytes) {
        if (draft_path && draft_path[0]) DRAFT_FAIL("standalone MTP asset was not admitted for this target GGUF");
        return 0;
    }
    if (!d || !draft_path || !draft_path[0])
        DRAFT_FAIL("[MODEL_ASSET_MISSING] required MTP weights are missing; repair this model's download");
    if (!d->sha256 || strlen(d->sha256) != 64 || !d->layers || !d->weight_bytes ||
        (d->n_parts && !d->parts))
        DRAFT_FAIL("standalone MTP manifest has incomplete integrity or geometry data");
    unsigned first_index = 1, shard_count = 1;
    const int is_split = idletoken_gguf_split_parts(basename_of(draft_path), &first_index, &shard_count);
    if (first_index != 1 || shard_count != (unsigned)d->n_parts + 1 || (d->n_parts && !is_split))
        DRAFT_FAIL("standalone MTP manifest must declare every split shard");
    for (uint16_t i = 0; i < d->n_parts; ++i) {
        unsigned index = 0, count = 0;
        if (!d->parts[i].gguf || !idletoken_gguf_split_parts(basename_of(d->parts[i].gguf), &index, &count) ||
            index != (unsigned)i + 2 || count != shard_count)
            DRAFT_FAIL("standalone MTP shard manifest is incomplete or unordered");
    }
    uint64_t first_bytes = d->bytes;
    for (uint16_t i = 0; i < d->n_parts; ++i) {
        if (d->parts[i].bytes >= first_bytes) DRAFT_FAIL("invalid MTP shard sizes");
        first_bytes -= d->parts[i].bytes;
    }
    for (uint16_t i = 0; i <= d->n_parts; ++i) {
        char part[1024];
        const uint64_t bytes = i ? d->parts[i - 1].bytes : first_bytes;
        const char *sha = i ? d->parts[i - 1].sha256 : d->sha256;
        if (!sha || strlen(sha) != 64) DRAFT_FAIL("MTP shard has no SHA-256");
        if (!i) snprintf(part, sizeof part, "%s", draft_path);
        else if (idletoken_mtp_draft_shard_path(draft_path, d->parts[i - 1].gguf, part, sizeof part))
            DRAFT_FAIL("invalid standalone MTP shard filename");
        struct stat st;
        if (stat(part, &st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size != bytes)
            DRAFT_FAIL("[MODEL_ASSET_MISSING] required MTP weights are missing or incomplete: %s", part);
        uint8_t digest[32]; char hex[65];
        if (idletoken_gguf_file_sha256(part, digest, 4096, why, why_cap)) return -1;
        for (unsigned j = 0; j < 32; ++j) snprintf(hex + 2 * j, 3, "%02x", digest[j]);
        if (strcmp(hex, sha)) DRAFT_FAIL("[MODEL_ASSET_INVALID] MTP weights failed SHA-256 verification: %s", part);
    }
    uint16_t layers = 0; uint64_t weights = 0, kv = 0;
    if (idletoken_gguf_mtp_info(draft_path, &layers, &weights, &kv, why, why_cap)) return -1;
    if (layers != d->layers || kv != d->kv_bytes_per_token)
        DRAFT_FAIL("standalone MTP head/cache geometry does not match its manifest");
    target = idletoken_gguf_meta_open(target_path, why, why_cap);
    draft = idletoken_gguf_meta_open(draft_path, why, why_cap);
    if (!target || !draft) DRAFT_FAIL("cannot inspect the target/MTP metadata pair");
    char ta[64], da[64], key[128];
    if (idletoken_gguf_meta_str(target, "general.architecture", ta, sizeof ta) ||
        idletoken_gguf_meta_str(draft, "general.architecture", da, sizeof da) || strcmp(ta, da))
        DRAFT_FAIL("target and standalone MTP architectures differ");
    static const char *fields[] = {"embedding_length", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length", "expert_count", "expert_used_count"};
    for (unsigned i = 0; i < sizeof fields / sizeof fields[0]; ++i) {
        uint32_t a = 0, b = 0;
        snprintf(key, sizeof key, "%s.%s", ta, fields[i]);
        const int ar = idletoken_gguf_meta_u32(target, key, &a);
        const int br = idletoken_gguf_meta_u32(draft, key, &b);
        if (ar != br || a != b) DRAFT_FAIL("target/MTP geometry differs: %s", fields[i]);
    }
    uint32_t a = 0, b = 0;
    snprintf(key, sizeof key, "%s.block_count", ta);
    if (idletoken_gguf_meta_u32(target, key, &a) || idletoken_gguf_meta_u32(draft, key, &b) || b != a + layers)
        DRAFT_FAIL("standalone MTP block indices do not match the target");
    idletoken_gguf_tensor ti;
    int has_embd = idletoken_gguf_tensor_find(draft, "token_embd.weight", &ti) == 0;
    int has_norm = idletoken_gguf_tensor_find(draft, "output_norm.weight", &ti) == 0;
    for (uint16_t i = 0; i < d->n_parts; ++i) {
        char part[1024];
        if (idletoken_mtp_draft_shard_path(draft_path, d->parts[i].gguf, part, sizeof part))
            DRAFT_FAIL("invalid standalone MTP shard filename");
        idletoken_gguf_meta *m = idletoken_gguf_meta_open(part, why, why_cap);
        if (!m) DRAFT_FAIL("cannot inspect standalone MTP shard metadata");
        has_embd |= idletoken_gguf_tensor_find(m, "token_embd.weight", &ti) == 0;
        has_norm |= idletoken_gguf_tensor_find(m, "output_norm.weight", &ti) == 0;
        idletoken_gguf_meta_close(m);
    }
    if (!has_embd || !has_norm)
        DRAFT_FAIL("standalone MTP is missing its required embedding/output tensors");
    uint64_t na = 0, nb = 0;
    idletoken_gguf_str_iter ia, ib;
    if (idletoken_gguf_meta_arr_str_begin(target, "tokenizer.ggml.tokens", &ia, &na) ||
        idletoken_gguf_meta_arr_str_begin(draft, "tokenizer.ggml.tokens", &ib, &nb) || na != nb)
        DRAFT_FAIL("target and standalone MTP token vocabularies differ");
    for (uint64_t i = 0; i < na; ++i) {
        char at[16384], bt[16384];
        const int64_t al = idletoken_gguf_meta_arr_str_next(&ia, at, sizeof at);
        const int64_t bl = idletoken_gguf_meta_arr_str_next(&ib, bt, sizeof bt);
        if (al < 0 || bl < 0 || al >= (int64_t)sizeof at || bl >= (int64_t)sizeof bt || al != bl || memcmp(at, bt, (size_t)al))
            DRAFT_FAIL("target and standalone MTP token IDs differ");
    }
    idletoken_gguf_meta_close(target); idletoken_gguf_meta_close(draft);
    return 0;
#undef DRAFT_FAIL
}

static void apply_mtp_size(const idletoken_model_spec *spec,
                           idletoken_llm_model_size *out, uint16_t layers,
                           uint64_t weights, uint64_t kv) {
    out->mtp_enabled = layers > 0;
    out->mtp_layers = layers;
    out->mtp_weight_bytes = weights;
    out->mtp_kv_bytes_per_token = kv;
    if (layers && spec->kv_kind == IDLETOKEN_KV_HYBRID)
        out->kv_fixed_bytes_per_seq *= 4; /* target rollback: 1 + --draft-max 3 */
    idletoken_model_size_set_kv_tier(spec, out, out->kv_tier);
}

int idletoken_model_size_resolve(const idletoken_model_spec *spec,
                                 const char *quant,
                                 const char *gguf_path,
                                 idletoken_llm_model_size *out,
                                 char *why, size_t why_cap) {
    if (!spec || !out || spec->n_layers == 0) return -1;
    if (why && why_cap) why[0] = '\0';

    memset(out, 0, sizeof(*out));
    /* Shape, not size: these come from measured manifest/GGUF geometry and do
     * not change with weight quantization. Hybrid attention has growing KV on
     * only every Nth full-attention layer plus fixed f32 recurrent state on the
     * remaining layers. DeepSeek4 uses the pinned engine's padded raw/CSA/HCA
     * cache formula, carried separately rather than flattened into a guessed
     * bytes/token slope. */
    out->n_layers = spec->n_layers;
    out->n_expert = spec->n_expert;
    out->n_expert_used = spec->n_expert_used;
    /* Measured graph workspace, carried through from the manifest for the KV
     * cache dtype this precision will actually be launched with.
     *
     * The workspace does not vary with the WEIGHT quantization — verified
     * 2026-09-01: Q4_K_M through BF16 all report 489.00 MiB on Qwen3.5-0.8B at
     * 256K. It DOES vary with the KV cache dtype on some architectures (the
     * same model measures 745.28 MiB at 256K once the cache is quantized), and
     * the coordinator picks that dtype FROM the weight quantization. So the
     * quant does select a workspace here — indirectly, through the KV tier, not
     * because the weights themselves cost graph memory.
     *
     * The tier is resolved once, here, so every downstream caller keeps its
     * signature and cannot disagree with the engine about which cache is
     * running. Zero means "not measured for this model+tier", which the planner
     * treats as a refusal rather than as free.
     * See results/memory-need-measured-20260901.md and the 09-02 addendum. */
    {
        /* Name first, path second — the same order the coordinator resolves
         * weight bits in, so a `--llama-gguf`-only launch (the client's real
         * shape) reads the tier off the filename exactly as coord does. */
        int wbits = (quant && quant[0]) ? idletoken_quant_weight_bits(quant) : 0;
        if (!wbits && gguf_path && gguf_path[0])
            wbits = idletoken_quant_bits_from_path(gguf_path);
        const int tier = idletoken_llama_kv_tier_for_weight(wbits);
        out->kv_tier = (uint8_t)tier;
        out->compute_bytes_128k_cuda  = spec->compute_bytes_128k_cuda[tier];
        out->compute_bytes_256k_cuda  = spec->compute_bytes_256k_cuda[tier];
        out->compute_bytes_1m_cuda    = spec->compute_bytes_1m_cuda[tier];
        out->compute_bytes_128k_metal = spec->compute_bytes_128k_metal[tier];
        out->compute_bytes_256k_metal = spec->compute_bytes_256k_metal[tier];
        out->compute_bytes_1m_metal   = spec->compute_bytes_1m_metal[tier];
    }
    /* The vision tower, straight from the registry: its size is a property of
     * the model, not of the weight precision or the KV tier (one tower serves
     * every quant in the menu). 0 here for a text-only model, which is what
     * makes the planner's single extra term a no-op for them.
     *
     * ⚠ This charges the tower whenever the MODEL has one, even if this launch
     * will not load it. That is the safe direction: a plan sized without it
     * cannot be run with it, whereas a plan sized with it merely leaves a few
     * hundred MiB unused on a text-only run of a vision model. */
    out->mmproj_bytes = spec->mmproj ? spec->mmproj->bytes : 0;
    if (spec->kv_kind == IDLETOKEN_KV_HYBRID) {
        const uint32_t iv = spec->full_attn_interval ? spec->full_attn_interval : 1;
        const uint64_t n_full = ((uint64_t)spec->n_layers + iv - 1) / iv;
        const uint64_t n_linear = (uint64_t)spec->n_layers - n_full;
        out->kv_bytes_per_token =
            (uint64_t)spec->kv_bytes_per_token_layer * n_full;
        out->kv_fixed_bytes_per_seq =
            (uint64_t)spec->state_bytes_per_layer * n_linear;
    } else if (spec->dsv4_raw_bytes_per_cell != 0) {
        out->dsv4_raw_bytes_per_cell = spec->dsv4_raw_bytes_per_cell;
        out->dsv4_csa_bytes_per_cell = spec->dsv4_csa_bytes_per_cell;
        out->dsv4_hca_bytes_per_cell = spec->dsv4_hca_bytes_per_cell;
        out->kv_fixed_bytes_per_seq = spec->dsv4_fixed_bytes_per_seq;
        /* Asymptotic slope for diagnostics only. Capacity calls the exact
         * padded helper in plan.c. Round upward so the log never understates. */
        out->kv_bytes_per_token = spec->dsv4_raw_bytes_per_cell +
            (spec->dsv4_csa_bytes_per_cell + 3) / 4 +
            (spec->dsv4_hca_bytes_per_cell + 127) / 128;
    } else {
        out->kv_bytes_per_token =
            (uint64_t)spec->kv_bytes_per_token_layer * (uint64_t)spec->n_layers;
    }

    const int has_quant = quant && quant[0];
    const int has_path  = gguf_path && gguf_path[0];

    /* --- 1. the file that will really be opened --------------------------- */
    char ferr[512] = "";
    if (has_path) {
        const uint64_t bytes = idletoken_gguf_bytes_on_disk(gguf_path, ferr, sizeof(ferr));
        if (bytes > 0) {
            out->total_bytes = bytes;
            char xerr[256] = "";
            if (spec->n_expert > 0 &&
                gguf_expert_layout(gguf_path, spec->n_layers, out,
                                   xerr, sizeof xerr) != 0) {
                /* The full-GPU budget remains valid. Only Hybrid depends on
                 * this layout, and its planner checks `expert_bytes_complete`
                 * before promising anything. Keep serving GPU-only while
                 * making the missing evidence visible in this one source log. */
                out->expert_bytes_total = 0;
                memset(out->expert_bytes_per_layer, 0,
                       sizeof out->expert_bytes_per_layer);
                memset(out->expert_pool_weight_count_per_layer, 0,
                       sizeof out->expert_pool_weight_count_per_layer);
                memset(out->expert_pool_weight_hash_per_layer, 0,
                       sizeof out->expert_pool_weight_hash_per_layer);
                memset(out->expert_pool_slot_bytes_per_layer, 0,
                       sizeof out->expert_pool_slot_bytes_per_layer);
                memset(out->expert_pool_max_tensor_per_layer, 0,
                       sizeof out->expert_pool_max_tensor_per_layer);
                memset(out->weight_bytes_per_layer, 0, sizeof out->weight_bytes_per_layer);
                out->weight_bytes_shared = 0;
            }
            int ambiguous = 0;
            const idletoken_model_variant *v = nearest_variant(spec, bytes, &ambiguous);
            if (why && why_cap) {
                snprintf(why, why_cap, "the GGUF on disk %s (%.2f GiB)",
                         basename_of(gguf_path), (double)bytes / GiB);
                if (v) {
                    app(why, why_cap, ", manifest quant %s", v->quant);
                    if (ambiguous)
                        app(why, why_cap, " (nearest of several similarly sized "
                                          "quants in this menu)");
                    if (has_quant && strcmp(v->quant, quant) != 0)
                        app(why, why_cap,
                            " — WARNING: --quant %s was requested but the file on "
                            "disk is %s; the budget follows the file", quant, v->quant);
                } else {
                    app(why, why_cap,
                        " — WARNING: this size matches no quantization in the %s "
                        "manifest, so the budget uses the file's real size and the "
                        "per-quant layer data is unavailable; a layer split derived "
                                          "from it is less precise than usual", spec->id);
                }
                if (spec->n_expert > 0 && !out->expert_bytes_complete)
                    app(why, why_cap,
                        " — WARNING: exact MoE expert placement is unavailable (%s); "
                        "GPU-only remains usable but Hybrid will refuse", xerr);
            }
            uint16_t mtp_layers = 0;
            uint64_t mtp_weights = 0, mtp_kv = 0;
            char merr[256] = "";
            if (idletoken_gguf_mtp_info(gguf_path, &mtp_layers, &mtp_weights,
                                       &mtp_kv, merr, sizeof merr) != 0) {
                if (why && why_cap) snprintf(why, why_cap,
                    "cannot verify GGUF MTP capability: %s", merr);
                return -1;
            }
            apply_mtp_size(spec, out, mtp_layers, mtp_weights, mtp_kv);
            apply_external_mtp_size(spec, quant, gguf_path, out);
            if (mtp_layers) app(why, why_cap,
                "; MTP enabled (%u head, %.2f MiB draft weights)",
                (unsigned)mtp_layers, (double)mtp_weights / 1048576.0);
            return 0;
        }
        /* Falling back is allowed; falling back quietly is not. The engine is
         * about to open the same path and will say what is wrong with it far
         * more precisely, so refusing here would only replace one clear error
         * with an earlier vaguer one. */
    }

    /* --- 2. the named precision ------------------------------------------- */
    uint64_t layer_b = 0, shared_b = 0;
    const idletoken_model_variant *v = idletoken_model_variant_get(spec, quant);
    idletoken_model_weight_bytes(spec, quant, &layer_b, &shared_b);
    out->total_bytes = layer_b + shared_b;
    if (spec->n_expert > 0)
        estimate_expert_layout(spec, layer_b, shared_b,
                               v ? v->expert_weight_bytes : 0,
                               v ? v->expert_max_tensor_bytes : 0, out);
    apply_mtp_size(spec, out, v ? v->mtp_layers : 0,
                   v ? v->mtp_weight_bytes : 0, spec->mtp_kv_bytes_per_token);
    apply_external_mtp_size(spec, quant, NULL, out);
    const char *resolved = v ? v->quant : "";
    const int quant_honoured = has_quant && v && strcmp(v->quant, quant) == 0;

    if (why && why_cap) {
        if (!v) {
            /* No variant menu: one implicit precision, so there is nothing to
             * pick wrong and nothing to warn about — unless a --quant was named,
             * in which case it was ignored and that should not be a secret. */
            snprintf(why, why_cap, "the %s manifest (single precision, %.2f GiB)",
                     spec->id, (double)out->total_bytes / GiB);
            if (has_quant)
                app(why, why_cap,
                    " — WARNING: --quant %s was ignored; this model ships one "
                    "precision", quant);
        } else if (quant_honoured) {
            snprintf(why, why_cap, "the %s manifest at quant %s (%.2f GiB)",
                     spec->id, resolved, (double)out->total_bytes / GiB);
        } else if (has_quant) {
            snprintf(why, why_cap,
                     "the %s manifest at its DEFAULT quant %s (%.2f GiB) "
                     "— WARNING: quant '%s' is not in this model's menu",
                     spec->id, resolved, (double)out->total_bytes / GiB, quant);
        } else {
            snprintf(why, why_cap,
                     "the %s manifest at its DEFAULT quant %s (%.2f GiB) "
                     "— WARNING: no quantization was named, so this budget is "
                     "only right if the engine loads that exact file; a different "
                     "quant makes the slot count and the tensor split wrong",
                     spec->id, resolved, (double)out->total_bytes / GiB);
        }
        /* A path was given and could not be sized: that is the source we were
         * supposed to trust, so it is a warning wherever we landed instead. */
        if (has_path)
            app(why, why_cap,
                "%s the GGUF path was not usable (%s), so this is the manifest's "
                "number and not the file's",
                strstr(why, "WARNING") ? " Also," : " — WARNING:", ferr);
    }
    return 0;
}
