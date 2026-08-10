#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

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

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef struct {
    const char *model_path;
    const char *mtp_path;
    /* IdleToken: optional DSpark speculative-decoding draft module
     * (DeepSeek-V4-Flash-DSpark-support.gguf). NULL/empty = not loaded. */
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
    /* IdleToken PP extension: load only this layer range into the accelerator
     * cache.  When `load_layer_hi <= load_layer_lo` (e.g. both 0, the default),
     * the full model is loaded — the original single-machine behavior.
     * Non-layer tensors (token_embd, output_*, mtp.0.*) are always loaded,
     * regardless of range. */
    int load_layer_lo;
    int load_layer_hi;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
/* DSpark speculative decode: commit first_token, draft a block, verify it in
 * one pass. Returns the number of tokens written to `accepted` (>=1), or -1 on
 * a hard eval error. Falls back to a single token whenever the module is
 * absent or the block is rejected. */
int ds4_session_eval_dspark_argmax(ds4_session *s, int first_token,
                                   int max_tokens, int eos_token,
                                   int *accepted, int accepted_cap,
                                   char *err, size_t errlen);

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);

/* IdleToken PP extension: encode decode-layers [lo, hi) on this worker shard.
 * - lo == 0          → embeds `token` into cur_hc first (stage-0 entry).
 * - 0 < lo < N_LAYER → caller must stage cur_hc from previous PP stage.
 * - hi == DS4_N_LAYER → runs the output head, logits become readable.
 * - Otherwise        → caller harvests cur_hc for the next PP stage.
 *
 * Only compiled when DS4_NO_GPU is not defined (it needs ds4_gpu_graph).
 * Coord builds with -DDS4_NO_GPU and must not call this. */
bool ds4_session_encode_layer_range(ds4_session *s, int token,
                                    uint32_t pos, uint32_t lo, uint32_t hi);

/* IdleToken PP extension: HC residual tensor accessors at the PP boundary.
 * - `_bytes` returns the live byte size of cur_hc (depends on n_tokens
 *   currently configured in the graph; v0.1 decode is one token at a time
 *   so this is N_HC * N_EMBD * sizeof(float) = 64 KiB).
 * - `_read` harvests cur_hc into host memory (sending worker side).
 * - `_write` stages incoming bytes back into cur_hc (receiving worker side).
 * Only callable when DS4_NO_GPU is not defined. */
uint64_t ds4_session_hc_tensor_bytes(ds4_session *s);
bool     ds4_session_hc_tensor_read (ds4_session *s, void *out, uint64_t bytes);
bool     ds4_session_hc_tensor_write(ds4_session *s, const void *in, uint64_t bytes);

/* Read full logits vector from the last PP stage. Valid after
 * ds4_session_encode_layer_range(..., hi=DS4_N_LAYER). `out` must hold at
 * least DS4_N_VOCAB floats. */
bool ds4_session_logits_read(ds4_session *s, float *out);

/* IdleToken PP extension: batched chunk prefill over decode-layers [lo, hi).
 * Mirrors official chunked-prefill numerics exactly (per-token prefill via
 * encode_layer_range diverges: batch vs small-batch FP8 rounding).
 * `tokens` is required on EVERY stage (the MoE router is token-aware).
 * Chunks must not exceed ds4_session_prefill_chunk_cap(); the coordinator
 * gets the same value without a GPU via ds4_prefill_chunk_cap_for_ctx(ctx).
 * Batch HC crosses the PP boundary with the _batch_hc accessors (n_tokens
 * rows of N_HC*N_EMBD floats, always in/out of batch_cur_hc). GPU builds
 * only, except ds4_prefill_chunk_cap_for_ctx which is plain CPU math. */
uint32_t ds4_session_prefill_chunk_cap(ds4_session *s);
bool ds4_session_prefill_layer_range(ds4_session *s, const int *tokens,
                                     uint32_t n_tokens, uint32_t pos0,
                                     uint32_t lo, uint32_t hi);
uint64_t ds4_session_batch_hc_bytes(ds4_session *s, uint32_t n_tokens);
bool     ds4_session_batch_hc_read (ds4_session *s, void *out, uint32_t n_tokens);
bool     ds4_session_batch_hc_write(ds4_session *s, const void *in, uint32_t n_tokens);
uint32_t ds4_prefill_chunk_cap_for_ctx(int ctx_size);

void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
/* IdleToken: DSpark draft module present? block size = draft tokens per pass. */
bool ds4_engine_has_dspark(ds4_engine *e);
/* Build the DSpark drafter's context vector from the captured aux hidden
 * states: main_norm(main_proj(concat(aux))). `n` must be n_embd. Run after a
 * decode step; on a fresh session the captures are still zero. */
int ds4_engine_dspark_main_x(ds4_engine *e, ds4_session *s, float *out, uint32_t n);

/* Test hook: run one of the two exact verifiers (0 = hardcoded N=2,
 * 1 = generalised N). With n_tokens == 2 they must agree bit for bit.
 * `logits_out` must hold n_tokens * vocab floats. */
int ds4_session_verify_probe(ds4_session *s, int which, const int *tokens,
                             uint32_t n_tokens, uint32_t start,
                             int *tops, float *logits_out);

/* Produce one DSpark draft block: `n` candidate tokens starting at `pos0`,
 * given the anchor token. Call after push_context() has filled the drafter's
 * context window. Returns 0 when no module is loaded or n > block_size. */
int ds4_engine_dspark_draft(ds4_engine *e, ds4_session *s, int anchor_token,
                            uint32_t pos0, int *out_tokens, uint32_t n);

/* Append this position's DSpark context KV to every drafter stage cache.
 * Call after ds4_engine_dspark_main_x() for the same position. */
int ds4_engine_dspark_push_context(ds4_engine *e, ds4_session *s, uint32_t pos);
/* Read one stage's context KV (post-norm, pre-RoPE); `n` must be head_dim. */
int ds4_engine_dspark_context_kv(ds4_engine *e, ds4_session *s,
                                 uint32_t stage, float *out, uint32_t n);

/* DSpark markov bias for one draft step: bias = markov_w2(markov_w1[prev]).
 * `n` must be the model's vocab size. Returns 0 when no module is loaded. */
int ds4_engine_dspark_markov_bias(ds4_engine *e, int prev_token, float *out, uint32_t n);

/* Read one captured DSpark aux hidden state (n must be the model's n_embd).
 * Returns 0 when no module is loaded or the index is out of range. */
int ds4_session_dspark_aux_read(ds4_session *s, uint32_t idx, float *out, uint32_t n);
int ds4_engine_dspark_block_size(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Disk KV cache payload helpers.  The server owns the outer file header and
 * policy; the engine owns the DS4-specific serialized graph state. */
uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

#endif
