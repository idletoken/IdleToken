/* Public-build refusal for the retired pre-llama.cpp inference backend.
 *
 * This file contains no model implementation. It exists because the public
 * mirror intentionally excludes the frozen historical headers and sources,
 * while the old unreachable call sites remain for source-history continuity.
 * Every entry point is loud and fail-closed. The private build does not use
 * this file: it keeps compiling src/common/ds4_stub.c against the real frozen
 * headers as its signature check.
 */

#include <stdio.h>
#include <stdlib.h>

#include "idletoken_legacy_backend_refusal.h"

static void legacy_backend_absent(const char *fn) {
    static int said = 0;
    if (!said) {
        said = 1;
        fprintf(stderr,
                "idletoken: the retired legacy inference backend is not built "
                "into this public binary; llama.cpp is the only engine. "
                "Something called %s() -- this is a caller bug and there is "
                "no fallback.\n",
                fn);
    }
}

#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define REFUSE_VOID(name, sig)   void name sig { legacy_backend_absent(#name); }
#define REFUSE_INT(name, sig)    int name sig { legacy_backend_absent(#name); return -1; }
#define REFUSE_BOOL(name, sig)   bool name sig { legacy_backend_absent(#name); return false; }
#define REFUSE_PTR(name, sig, t) t name sig { legacy_backend_absent(#name); return NULL; }

REFUSE_INT(ds4_engine_open,
           (ds4_engine **out, const ds4_engine_options *opt))
REFUSE_VOID(ds4_engine_close, (ds4_engine *e))
REFUSE_VOID(ds4_tokens_push, (ds4_tokens *tv, int token))
REFUSE_VOID(ds4_tokens_free, (ds4_tokens *tv))
REFUSE_VOID(ds4_tokenize_text,
            (ds4_engine *e, const char *text, ds4_tokens *out))
REFUSE_VOID(ds4_chat_begin, (ds4_engine *e, ds4_tokens *tokens))
REFUSE_VOID(ds4_chat_append_message,
            (ds4_engine *e, ds4_tokens *tokens, const char *role,
             const char *content))
REFUSE_VOID(ds4_chat_append_assistant_prefix,
            (ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode))
REFUSE_VOID(ds4_encode_chat_prompt,
            (ds4_engine *e, const char *system, const char *prompt,
             ds4_think_mode think_mode, ds4_tokens *out))
REFUSE_INT(ds4_token_eos, (ds4_engine *e))
REFUSE_PTR(ds4_token_text,
           (ds4_engine *e, int token, size_t *len), char *)

uint32_t ds4_prefill_chunk_cap_for_ctx(int ctx_size) {
    legacy_backend_absent("ds4_prefill_chunk_cap_for_ctx");
    return 0;
}

REFUSE_INT(ds4_session_create,
           (ds4_session **out, ds4_engine *e, int ctx_size))
REFUSE_VOID(ds4_session_free, (ds4_session *s))
REFUSE_VOID(ds4_session_rewind, (ds4_session *s, int pos))
REFUSE_BOOL(ds4_session_encode_layer_range,
            (ds4_session *s, int token, uint32_t pos, uint32_t lo,
             uint32_t hi))
REFUSE_BOOL(ds4_session_prefill_layer_range,
            (ds4_session *s, const int *tokens, uint32_t n_tokens,
             uint32_t pos0, uint32_t lo, uint32_t hi))
REFUSE_BOOL(ds4_session_hc_tensor_read,
            (ds4_session *s, void *out, uint64_t bytes))
REFUSE_BOOL(ds4_session_hc_tensor_write,
            (ds4_session *s, const void *in, uint64_t bytes))
REFUSE_BOOL(ds4_session_batch_hc_read,
            (ds4_session *s, void *out, uint32_t n_tokens))
REFUSE_BOOL(ds4_session_batch_hc_write,
            (ds4_session *s, const void *in, uint32_t n_tokens))
REFUSE_BOOL(ds4_session_logits_read, (ds4_session *s, float *out))

uint64_t ds4_gpu_probe_pinned_max(void) {
    legacy_backend_absent("ds4_gpu_probe_pinned_max");
    return 0;
}
REFUSE_VOID(ds4_gpu_set_hybrid_vram_budget, (uint64_t bytes))
REFUSE_VOID(ds4_gpu_set_moe_cache, (uint64_t bytes, uint32_t n_layers))
REFUSE_INT(ds4_gpu_synchronize, (void))

ds4x_model *ds4x_model_load(const char *path, uint16_t layer_lo,
                            uint16_t layer_hi, char *err, size_t errlen) {
    legacy_backend_absent("ds4x_model_load");
    if (err && errlen)
        snprintf(err, errlen,
                 "the retired backend is not built; llama.cpp is the engine");
    return NULL;
}
REFUSE_VOID(ds4x_model_free, (ds4x_model *m))
REFUSE_INT(ds4x_embed_tokens,
           (const ds4x_model *m, const int32_t *tokens, uint32_t n,
            float *out))
REFUSE_INT(ds4x_output_logits,
           (const ds4x_model *m, const float *hidden, float *logits))

ds4x_runner *ds4x_runner_create(const ds4x_model *model, uint32_t ctx_size,
                                char *err, size_t errlen) {
    legacy_backend_absent("ds4x_runner_create");
    if (err && errlen)
        snprintf(err, errlen,
                 "the retired backend is not built; llama.cpp is the engine");
    return NULL;
}
REFUSE_VOID(ds4x_runner_free, (ds4x_runner *r))
REFUSE_INT(ds4x_runner_run,
           (ds4x_runner *r, float *hidden, uint32_t n_tokens, uint32_t pos0))

const char *ds4x_runner_last_error(void) {
    legacy_backend_absent("ds4x_runner_last_error");
    return "the retired backend is not built; llama.cpp is the engine";
}

REFUSE_VOID(ds4x_cuda_set_budget, (uint64_t bytes))
void ds4x_cuda_ffn_stats(double *ms_kernel, double *ms_total,
                         uint64_t *calls, uint64_t *rows) {
    legacy_backend_absent("ds4x_cuda_ffn_stats");
    if (ms_kernel) *ms_kernel = 0.0;
    if (ms_total) *ms_total = 0.0;
    if (calls) *calls = 0;
    if (rows) *rows = 0;
}
void ds4x_cuda_proj_stats(double *ms_kernel, double *ms_total,
                          uint64_t *calls, uint64_t *rows) {
    legacy_backend_absent("ds4x_cuda_proj_stats");
    if (ms_kernel) *ms_kernel = 0.0;
    if (ms_total) *ms_total = 0.0;
    if (calls) *calls = 0;
    if (rows) *rows = 0;
}

ds4x_tokenizer *ds4x_tok_load(const char *gguf_path, char *err,
                              size_t errlen) {
    legacy_backend_absent("ds4x_tok_load");
    if (err && errlen)
        snprintf(err, errlen,
                 "the retired tokenizer is not built; llama.cpp tokenizes");
    return NULL;
}
REFUSE_VOID(ds4x_tok_free, (ds4x_tokenizer *t))

int64_t ds4x_tok_encode(const ds4x_tokenizer *t, const char *text,
                        int32_t *out, uint32_t cap) {
    legacy_backend_absent("ds4x_tok_encode");
    return -1;
}
int64_t ds4x_tok_chat_apply(const ds4x_tokenizer *t,
                            const char *const *roles,
                            const char *const *contents, uint32_t n_msgs,
                            int add_generation_prompt, int32_t *out,
                            uint32_t cap) {
    legacy_backend_absent("ds4x_tok_chat_apply");
    return -1;
}
REFUSE_PTR(ds4x_tok_decode,
           (const ds4x_tokenizer *t, const int32_t *ids, uint32_t n,
            int keep_special), char *)

int32_t ds4x_tok_eos(const ds4x_tokenizer *t) {
    legacy_backend_absent("ds4x_tok_eos");
    return -1;
}
