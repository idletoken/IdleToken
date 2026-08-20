/* idletoken_model_auto.h — open model intake (v2 rebuild WS-B4).
 *
 * Builds a runtime idletoken_model_spec from ANY local GGUF's header, so a
 * model nobody ever registered can be selected, sized by the scheduler, and
 * reported truthfully on /v1/models + /idletoken/v1/stats. The static registry
 * (src/common/model.c) stays what it is — a curated recommendation list; the
 * GGUF header is the runtime truth for whatever the user actually points at.
 *
 * Pure C, no engine dependencies: reads the header region only (via
 * src/common/gguf.c), never tensor bytes, so it costs milliseconds even on an
 * 80 GiB file.
 */
#ifndef IDLETOKEN_MODEL_AUTO_H
#define IDLETOKEN_MODEL_AUTO_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_model.h"

/* A model spec derived from a GGUF header plus the storage its string fields
 * point into. The embedded spec is valid for the lifetime of this struct. */
typedef struct {
    idletoken_model_spec spec;   /* id/label/default_gguf point at the buffers below */

    char id[96];         /* sanitized from general.name (fallback: filename stem) */
    char label[128];     /* general.name verbatim, or the filename stem */
    char arch[64];       /* general.architecture */
    char gguf_name[512]; /* basename of the file (spec.default_gguf) */

    uint64_t file_bytes;          /* total on-disk size of the GGUF */
    /* Whole-model KV-cache bytes per context token (all layers), f16 cache.
     * Derived from <arch>.attention.head_count_kv (scalar or per-layer array;
     * array entries of 0 = linear/recurrent layers with no KV cache, as
     * hybrid models like Qwen3.5 encode them) and key/value lengths.
     * 0 = the header carries no attention shape at all (caller must treat the
     * KV cost as unknown, not as free). */
    uint64_t kv_bytes_per_token;
    /* MoE shape from <arch>.expert_count / <arch>.expert_used_count; 0/0 on a
     * dense model. The scheduler needs them to size the WORKING SET — the
     * bytes a token actually touches, which on an MoE is a fraction of the
     * file and is what decides speed (idletoken_llama_working_set). */
    uint32_t n_expert;
    uint32_t n_expert_used;
} idletoken_auto_model;

/* Parse `path`'s GGUF header into `out`. Returns 0 on success, -1 with a
 * human-readable reason in `err` otherwise. Failures include: not a GGUF,
 * no general.architecture / <arch>.block_count (not a model we can plan), and
 * split GGUFs (-00001-of-000NN naming or split.count > 1) — multi-file models
 * are not supported yet and are refused with instructions, never half-loaded. */
int idletoken_model_from_gguf(const char *path, idletoken_auto_model *out,
                              char *err, size_t errlen);

#endif /* IDLETOKEN_MODEL_AUTO_H */
