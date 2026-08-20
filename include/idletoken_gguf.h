/* idletoken_gguf.h — read-only GGUF metadata access (header + KV section only).
 *
 * The ds4x runtime config (multi-model design §3.2) is driven entirely by
 * GGUF metadata, dispatched on `general.architecture`. This module loads the
 * header region of a GGUF (never the tensor bytes — works on the coordinator,
 * on a sparse layer shard, or on a metadata-only fixture) and answers typed
 * key lookups. The cursor logic matches weights.c's embedded parser; this is
 * the shared, reusable form.
 *
 * C only. No C++. No mmap — plain buffered read with grow-and-retry, so it
 * works the same on Windows.
 */
#ifndef IDLETOKEN_GGUF_H
#define IDLETOKEN_GGUF_H

#include <stddef.h>
#include <stdint.h>

typedef struct idletoken_gguf_meta idletoken_gguf_meta;

/* Parse the GGUF header + metadata KVs of `path`. NULL on error (err gets a
 * short reason). Tensor directory is parsed for the count only. */
idletoken_gguf_meta *idletoken_gguf_meta_open(const char *path, char *err, size_t errlen);
void idletoken_gguf_meta_close(idletoken_gguf_meta *m);

uint64_t idletoken_gguf_meta_n_tensors(const idletoken_gguf_meta *m);

/* Typed lookups. Return 0 on success, -1 when the key is missing or the
 * stored type cannot be represented (e.g. asking u32 of a string).
 * Integer getters coerce between GGUF int widths/signs as long as the value
 * fits; f32 also accepts integer-typed values. */
int idletoken_gguf_meta_str(const idletoken_gguf_meta *m, const char *key,
                         char *out, size_t cap);
int idletoken_gguf_meta_u32(const idletoken_gguf_meta *m, const char *key, uint32_t *out);
int idletoken_gguf_meta_f32(const idletoken_gguf_meta *m, const char *key, float *out);
/* Element count of an array-typed key (elements themselves are not exposed —
 * vocab size = len(tokenizer.ggml.tokens) without materializing the vocab). */
int idletoken_gguf_meta_arr_len(const idletoken_gguf_meta *m, const char *key, uint64_t *out);

/* idx-th element of a string array (e.g. tokenizer.ggml.tokens[42]). Copies up
 * to cap-1 bytes + NUL; returns the element's true byte length, or -1 on error
 * (missing key / not a string array / idx out of range). Sequential scan —
 * fine for one-shot vocab load, callers that need every element should iterate
 * in order (the parser is O(idx) per call). */
int64_t idletoken_gguf_meta_arr_str(const idletoken_gguf_meta *m, const char *key,
                                 uint64_t idx, char *out, size_t cap);

/* In-order reader for a string array — O(1) per element instead of O(idx).
 * Reading a whole vocab by index is quadratic: Qwen3's 151,936-token array
 * cost ~30 s that way (it dominated model load entirely). Always use this to
 * walk a full array; keep the indexed accessor for one-off lookups.
 *   idletoken_gguf_str_iter it; uint64_t n;
 *   if (idletoken_gguf_meta_arr_str_begin(m, "tokenizer.ggml.tokens", &it, &n)) ...
 *   for (uint64_t i = 0; i < n; i++) idletoken_gguf_meta_arr_str_next(&it, buf, cap);
 * Fields are public only so callers can stack-allocate; treat them as opaque. */
typedef struct {
    const unsigned char *b;
    size_t p, n;
    int err;
    unsigned long long left;
} idletoken_gguf_str_iter;

/* Returns 0 (and *count) on success, -1 if the key is missing / not a string
 * array. The iterator borrows the meta object — do not use it after close. */
int idletoken_gguf_meta_arr_str_begin(const idletoken_gguf_meta *m, const char *key,
                                   idletoken_gguf_str_iter *it, uint64_t *count);
/* Next element: copies up to cap-1 bytes + NUL, returns its true byte length,
 * or -1 when exhausted / malformed. */
int64_t idletoken_gguf_meta_arr_str_next(idletoken_gguf_str_iter *it, char *out, size_t cap);
/* idx-th element of an int array (token_type, merges-as-ints, ...). Coerces
 * any GGUF int width/sign. Returns 0 / -1. */
int idletoken_gguf_meta_arr_i32(const idletoken_gguf_meta *m, const char *key,
                             uint64_t idx, int32_t *out);

/* ---- tensor directory --------------------------------------------------- */
/* GGUF stores dims innermost-first: for a row-major [out][in] matrix,
 * dims[0] = in, dims[1] = out. `offset` is relative to the data section;
 * absolute file position = idletoken_gguf_data_offset() + offset. */
typedef struct {
    char     name[128];
    uint32_t ndim;
    uint64_t dims[4];
    uint32_t type;      /* GGML type id: 0=F32, 1=F16, quants above */
    uint64_t offset;
} idletoken_gguf_tensor;

/* Tensor info by index [0, n_tensors). Returns 0 / -1. */
int idletoken_gguf_tensor_info(const idletoken_gguf_meta *m, uint64_t idx,
                            idletoken_gguf_tensor *out);
/* By name; -1 when absent. */
int idletoken_gguf_tensor_find(const idletoken_gguf_meta *m, const char *name,
                            idletoken_gguf_tensor *out);
/* Absolute file offset where tensor data begins (header padded to
 * general.alignment, default 32). */
uint64_t idletoken_gguf_data_offset(const idletoken_gguf_meta *m);

/* ---- model identity ------------------------------------------------------
 * SHA-256 over bytes [0, data_offset) — magic, version, every metadata KV, and
 * the whole tensor directory (names, shapes, ggml types, offsets). Writes 32
 * bytes to `out`; 0 on success, -1 with `err` filled otherwise.
 *
 * Why not the whole file: it is 80 GiB (minutes per start), and — decisive —
 * a worker that fetched only its own layers over HTTP Range does not HAVE the
 * whole file, so a whole-file digest could never be checked where it matters.
 * The metadata region is a few MB, is present in every shard, and pins model,
 * quantisation, tensor layout and count.
 *
 * What this DOES catch: wrong model, wrong quant variant, a different tensor
 * layout, a truncated/rebuilt file. What it does NOT catch: bit corruption
 * INSIDE tensor data. Per-shard content digests would be needed for that and
 * are deliberately not attempted here — see docs/review/findings.md R-06. */
int idletoken_gguf_identity(const char *path, uint8_t out[32],
                         char *err, size_t errlen);

/* SHA-256 of the WHOLE file — the digest curated manifests pin (the `sha256`
 * field in models/<id>.json, which is also the value Hugging Face publishes as
 * the LFS oid), so this is the one that can be reconciled with them by eye or
 * by script.
 *
 * Distinct from idletoken_gguf_identity above on purpose, and NOT a substitute
 * for it: this reads every byte (minutes on an 80 GiB model) and needs a
 * complete file, so it must never sit on a startup path. It exists for the
 * development channel — record what a machine actually loaded — while the
 * cheap metadata identity keeps doing the per-start cluster agreement check.
 *
 * `progress_mib`, when > 0, prints a line to stderr every that many MiB, so a
 * multi-minute hash of a large model does not look like a hang.
 * Returns 0 and fills `out`; -1 with `err` on failure. */
int idletoken_gguf_file_sha256(const char *path, uint8_t out[32],
                               unsigned progress_mib,
                               char *err, size_t errlen);

#endif /* IDLETOKEN_GGUF_H */
