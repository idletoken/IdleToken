/* IdleToken layer-shard weight fetch.
 *
 * A pipeline worker only runs layers [lo, hi), so it should not need the whole
 * 80GB GGUF. ds4's loader requires the full tensor directory but never reads
 * the bytes of skipped layers, so a worker can load a *sparse* partial GGUF
 * (original apparent size, holes for skipped layers). This module materializes
 * that partial locally by fetching only the needed byte ranges (header +
 * shared tensors + [lo,hi) layers) from a weight repo over HTTP byte-range.
 *
 * The repo hosts the original master GGUF plus a manifest.json (produced by
 * scripts/gguf_shard.py) mapping every tensor to {layer, offset, bytes}. */
#ifndef IDLETOKEN_WEIGHTS_H
#define IDLETOKEN_WEIGHTS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fetch/materialize the partial GGUF for layers [layer_lo, layer_hi) from the
 * http(s) repo `base_url` (which hosts master.gguf + master.gguf.manifest.json).
 * The sparse partial is written under `cache_dir` and its path returned in
 * `out_path` (size out_cap). Returns 0 on success, non-zero on failure.
 * Idempotent: an already-complete cached partial is reused. */
int idletoken_shard_fetch(const char *base_url, unsigned layer_lo, unsigned layer_hi,
                       const char *cache_dir, char *out_path, size_t out_cap);

/* Serve `dir` over HTTP with byte-range support on `bind_addr` (blocking; the
 * coordinator runs this as an isolated `idletoken-worker --serve-weights` sidecar
 * so big weight transfers never touch the inference HTTP path). Returns only on
 * fatal listen error. POSIX forks per connection (parallel); Windows serves
 * serially. */
int idletoken_serve_weights(const char *dir, const char *bind_addr);

/* Write the `.idx` manifest for `gguf_path` to `idx_path` (C port of
 * scripts/gguf_shard.py idx). Lets the coordinator generate its repo index
 * without Python. Returns 0 on success. */
int idletoken_write_idx(const char *gguf_path, const char *idx_path);

/* 1 when `idx_path` is missing, unreadable, or records a file_size that does
 * not match `gguf_path` — i.e. it indexes a DIFFERENT file.
 *
 * Why this exists: an index built from a still-downloading GGUF looks perfectly
 * valid to stat(). Its tensor offsets are right (they come from the header,
 * which lands first), so models still load — but its file_size is the partial
 * size, and that is the field the shard cache compares against. Result: every
 * worker silently refetches its whole shard on every run, forever, with no
 * error anywhere. Existence is not validity. */
int idletoken_idx_stale(const char *gguf_path, const char *idx_path);

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_WEIGHTS_H */
