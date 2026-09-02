#ifndef IDLETOKEN_LEGACY_BACKEND_REFUSAL_H
#define IDLETOKEN_LEGACY_BACKEND_REFUSAL_H

/* Public-build boundary for the retired pre-llama.cpp inference backend.
 *
 * The private tree deliberately compiles its refusal implementation against
 * the historical headers so signature drift is a compile error. Those frozen
 * headers and their implementation are not published. A public build still
 * needs the old call sites to compile, so this file exposes only the opaque
 * types, the one options structure constructed by those call sites, and the
 * refusing functions they reference. It contains no inference implementation.
 *
 * Public builds link src/common/legacy_backend_refusal.c. Every entry point
 * reports a loud error and returns failure (or NULL); none can act as a silent
 * fallback for llama.cpp.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dspark_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    bool warm_weights;
    bool quality;
    int load_layer_lo;
    int load_layer_hi;
} ds4_engine_options;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens,
                             const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens,
                                      ds4_think_mode think_mode);
void ds4_encode_chat_prompt(ds4_engine *e, const char *system,
                            const char *prompt, ds4_think_mode think_mode,
                            ds4_tokens *out);
int ds4_token_eos(ds4_engine *e);
char *ds4_token_text(ds4_engine *e, int token, size_t *len);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
bool ds4_session_encode_layer_range(ds4_session *s, int token, uint32_t pos,
                                    uint32_t lo, uint32_t hi);
bool ds4_session_prefill_layer_range(ds4_session *s, const int *tokens,
                                     uint32_t n_tokens, uint32_t pos0,
                                     uint32_t lo, uint32_t hi);
bool ds4_session_hc_tensor_read(ds4_session *s, void *out, uint64_t bytes);
bool ds4_session_hc_tensor_write(ds4_session *s, const void *in, uint64_t bytes);
bool ds4_session_batch_hc_read(ds4_session *s, void *out, uint32_t n_tokens);
bool ds4_session_batch_hc_write(ds4_session *s, const void *in,
                                uint32_t n_tokens);
bool ds4_session_logits_read(ds4_session *s, float *out);
uint32_t ds4_prefill_chunk_cap_for_ctx(int ctx_size);

uint64_t ds4_gpu_probe_pinned_max(void);
void ds4_gpu_set_hybrid_vram_budget(uint64_t bytes);
void ds4_gpu_set_moe_cache(uint64_t bytes, uint32_t n_layers);
int ds4_gpu_synchronize(void);

typedef struct ds4x_model ds4x_model;
typedef struct ds4x_runner ds4x_runner;

ds4x_model *ds4x_model_load(const char *path, uint16_t layer_lo,
                            uint16_t layer_hi, char *err, size_t errlen);
void ds4x_model_free(ds4x_model *m);
int ds4x_embed_tokens(const ds4x_model *m, const int32_t *tokens, uint32_t n,
                      float *out);
int ds4x_output_logits(const ds4x_model *m, const float *hidden, float *logits);
ds4x_runner *ds4x_runner_create(const ds4x_model *model, uint32_t ctx_size,
                                char *err, size_t errlen);
void ds4x_runner_free(ds4x_runner *r);
int ds4x_runner_run(ds4x_runner *r, float *hidden, uint32_t n_tokens,
                    uint32_t pos0);
const char *ds4x_runner_last_error(void);

void ds4x_cuda_set_budget(uint64_t bytes);
void ds4x_cuda_ffn_stats(double *ms_kernel, double *ms_total,
                         uint64_t *calls, uint64_t *rows);
void ds4x_cuda_proj_stats(double *ms_kernel, double *ms_total,
                          uint64_t *calls, uint64_t *rows);

typedef struct ds4x_tokenizer ds4x_tokenizer;

ds4x_tokenizer *ds4x_tok_load(const char *gguf_path, char *err, size_t errlen);
void ds4x_tok_free(ds4x_tokenizer *t);
int64_t ds4x_tok_encode(const ds4x_tokenizer *t, const char *text,
                        int32_t *out, uint32_t cap);
int64_t ds4x_tok_chat_apply(const ds4x_tokenizer *t,
                            const char *const *roles,
                            const char *const *contents, uint32_t n_msgs,
                            int add_generation_prompt, int32_t *out,
                            uint32_t cap);
char *ds4x_tok_decode(const ds4x_tokenizer *t, const int32_t *ids, uint32_t n,
                      int keep_special);
int32_t ds4x_tok_eos(const ds4x_tokenizer *t);

#endif
