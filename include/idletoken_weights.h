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
 * scripts/gguf_shard.py) mapping every tensor to {layer, offset, bytes, name}. */
#ifndef IDLETOKEN_WEIGHTS_H
#define IDLETOKEN_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

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

/* Prepare the coordinator's local sparse GGUF from the scheduler's exact
 * prefix assignment [0, layer_hi).  The file always contains the GGUF header
 * and shared tensors, plus only the repeating layers which may execute on the
 * coordinator.  Its cache is keyed by the repository index identity rather
 * than by a machine name or a fixed two-node split.
 *
 * The view keeps a high-water layer prefix: a changed heterogeneous topology
 * which needs fewer local layers reuses it unchanged; one which needs more
 * fetches only the new suffix.  llama.cpp placement still loads only the
 * current assignment.  Returns 0 and the prepared path on success. */
int idletoken_local_model_prepare(const char *base_url,
                                  unsigned layer_hi,
                                  const char *cache_dir,
                                  char *out_path, size_t out_cap);

/* Same cache contract, but copy coordinator-owned ranges directly from the
 * local GGUF set instead of looping them through its HTTP repository. Remote
 * workers still use `base_url`; this fast path is only for the machine which
 * already owns `local_primary_gguf`. */
int idletoken_local_model_prepare_from_file(const char *base_url,
                                            const char *local_primary_gguf,
                                            unsigned layer_hi,
                                            const char *cache_dir,
                                            char *out_path, size_t out_cap);

/* Read the exact local CPU prefix ranges from an already-prepared sparse GGUF
 * view into the OS file cache.  Cluster mmap deliberately skips whole-model
 * prefetch so remote tensors are never read on the coordinator; without this
 * targeted warm-up, however, the first inference demand-pages the local CPU
 * layers from disk.  `layer_hi` is the CPU prefix, not the coordinator's GPU
 * suffix.  Returns 0 only after every selected byte has been read. */
int idletoken_local_model_prefetch(const char *base_url,
                                   unsigned layer_hi,
                                   const char *local_view_primary_gguf);

/* Seed llama.cpp's ggml-RPC tensor cache directly from the layer repository.
 *
 * The stock RPC cold-start path reads every tensor on the coordinator and
 * sends remote tensors over the LAN.  With the RPC cache enabled, the client
 * first sends a content hash; a matching file named by that hash lets the
 * worker load the tensor locally. This function copies the selected
 * [layer_lo, layer_hi) tensors (plus GGUF shared/output tensors) straight into
 * that cache. When `local_primary_gguf` is non-empty, bytes come from this
 * node's complete verified GGUF set; NULL retains repository range fetching
 * for compatibility and diagnostics. It never imports the full GGUF into the
 * worker and its peak extra disk use is the assigned shard plus one
 * tensor-sized temporary file.
 *
 * `cache_dir` is the exact directory passed to ggml-RPC (normally
 * <LLAMA_CACHE>/rpc).  `progress`, when non-NULL, is called after durable
 * pieces and once on a warm-cache reuse.  Returns 0 on success. */
typedef void (*idletoken_weights_progress_fn)(uint64_t done_bytes,
                                               uint64_t total_bytes,
                                               void *opaque);
int idletoken_rpc_cache_fetch(const char *base_url,
                              unsigned layer_lo, unsigned layer_hi,
                              const char *local_primary_gguf,
                              const char *cache_dir,
                              idletoken_weights_progress_fn progress,
                              void *progress_opaque,
                              uint64_t *bytes_out,
                              unsigned *tensors_out);

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
