/* modelsize.c — resolve the scheduler's model size from the file that will
 * really be loaded. See include/idletoken_modelsize.h for the contract and the
 * measurement that made it necessary.
 */
#include "idletoken_modelsize.h"

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
            }
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
