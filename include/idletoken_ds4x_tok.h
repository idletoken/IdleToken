/* idletoken_ds4x_tok.h — GGUF-metadata-driven byte-level BPE tokenizer.
 *
 * Loads tokenizer.ggml.* from a GGUF (tokens/token_type/merges/special ids)
 * and provides encode/decode over the GPT-2 byte↔unicode alphabet that the
 * DeepSeek / GLM / Kimi vocabularies use.
 *
 * Verification scope (honesty): DECODE + vocab + special-token loading are
 * exactly reproducible and unit-tested (round-trip on a synthetic vocab).
 * ENCODE implements the byte-level BPE merge core and is round-trip tested,
 * BUT exact parity with a real model additionally needs that model's
 * PRETOKENIZER regex (word splitting before BPE), which is model-specific and
 * must be validated against the real vocab on the DGX. Until then, encode
 * treats the input as a single BPE segment — correct for its merge machinery,
 * not a drop-in for a real tokenizer on punctuation/whitespace-heavy text.
 *
 * C only. No C++. No GPU. Unit-tests anywhere.
 */
#ifndef IDLETOKEN_DS4X_TOK_H
#define IDLETOKEN_DS4X_TOK_H

#include <stddef.h>
#include <stdint.h>

typedef struct ds4x_tokenizer ds4x_tokenizer;

/* Load the tokenizer from a GGUF. NULL on error (err gets a reason). */
ds4x_tokenizer *ds4x_tok_load(const char *gguf_path, char *err, size_t errlen);
void ds4x_tok_free(ds4x_tokenizer *t);

uint32_t ds4x_tok_vocab_size(const ds4x_tokenizer *t);
int32_t  ds4x_tok_bos(const ds4x_tokenizer *t);   /* -1 if unset */
int32_t  ds4x_tok_eos(const ds4x_tokenizer *t);

/* Decode ids → UTF-8 text (malloc'd, caller frees; NUL-terminated). Control /
 * special tokens render as empty unless `keep_special`. NULL on error. */
char *ds4x_tok_decode(const ds4x_tokenizer *t, const int32_t *ids, uint32_t n,
                      int keep_special);

/* Encode UTF-8 text → ids. Writes up to `cap` ids, returns the count actually
 * produced (may exceed cap — then only cap were written), or -1 on error.
 * Special/added tokens present in the text (e.g. "<|im_start|>") are matched
 * as whole units and emitted as their single id; the runs between them go
 * through byte-BPE (single-segment — full pretokenizer regex is the DGX item,
 * see the scope note above). */
int64_t ds4x_tok_encode(const ds4x_tokenizer *t, const char *text,
                        int32_t *out, uint32_t cap);

/* Render a chat turn list into token ids using the ChatML template that
 * Qwen3 (and most current open chat models) use:
 *   <|im_start|>{role}\n{content}<|im_end|>\n   per message, then
 *   <|im_start|>assistant\n                     when add_generation_prompt.
 * Requires the vocab to carry <|im_start|>/<|im_end|> as special tokens
 * (returns -1 otherwise — the caller should fall back). `roles`/`contents`
 * are parallel arrays of length `n_msgs`. Returns the id count (as encode). */
int64_t ds4x_tok_chat_apply(const ds4x_tokenizer *t,
                            const char *const *roles,
                            const char *const *contents, uint32_t n_msgs,
                            int add_generation_prompt,
                            int32_t *out, uint32_t cap);

/* Is `s` present in the vocab as a special/added token? (helper for callers
 * deciding whether the ChatML path is available.) id or -1. */
int32_t ds4x_tok_special_id(const ds4x_tokenizer *t, const char *s);

#endif /* IDLETOKEN_DS4X_TOK_H */
