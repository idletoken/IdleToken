#ifndef IDLETOKEN_GGUF_GEOM_H
#define IDLETOKEN_GGUF_GEOM_H

/* (block_elems, block_bytes) per ggml type id, indexed by GGML_TYPE_*.
 *
 * These are `ggml_blck_size()` and `ggml_type_size()` for the pinned engine.
 * They are NOT independent knowledge: a wrong row makes our GGUF index describe
 * a file that does not exist, and nothing at runtime can notice, because the
 * index stays internally consistent. The failure surfaces on another machine as
 * a tensor the cache "does not have".
 *
 * This lives in a header so there is exactly ONE copy.
 * `src/tools/gguf_geom_test.c` includes it next to the engine's own
 * ggml-common.h and re-derives every row from `sizeof(block_*)`, so drift from
 * the engine fails a test instead of a cluster. A second hand-written copy —
 * in the test, or in a script — would only prove that two copies of the same
 * mistake agree.
 *
 * ⚠ 2026-09-02: three rows were wrong and one cost a real cluster. IQ1_S (19)
 * carried IQ3_S's 110 bytes per block and IQ4_NL (20) carried IQ1_S's 50 with
 * the wrong block size — an insertion that shifted two neighbours. Unsloth's UD
 * quants mix types per tensor, so qwen3.8-27b UD-IQ2_XXS contains IQ1_S
 * tensors: the index claimed blk.33.ffn_gate.weight was 38297600 bytes when it
 * is 17408000, the worker cached that many bytes under that name, and the
 * coordinator's engine aborted with "required RPC model cache miss" — a message
 * about the symptom naming neither the size nor the type. Q8_1 (9) was 40
 * instead of 36; harmless only because GGUF files never contain it.
 *
 * A {0,0} row means "no such type": ids removed upstream, and gaps in the enum.
 * The fetcher refuses a file that names one, which is the right answer — a
 * model whose tensors we cannot size is a model we cannot shard.
 */
static const struct { unsigned be, bb; } IDLETOKEN_GGUF_GEOM[] = {
    {1,4},{1,2},{32,18},{32,20},{0,0},{0,0},{32,22},{32,24},{32,34},{32,36},
    {256,84},{256,110},{256,144},{256,176},{256,210},{256,292},{256,66},{256,74},
    {256,98},{256,50},{32,18},{256,110},{256,82},{256,136},{1,1},{1,2},{1,4},
    {1,8},{1,8},{256,56},{1,2},{0,0},{0,0},{0,0},{256,54},{256,66},{0,0},
    {0,0},{0,0},{32,17},{64,36},{128,18},{64,18},
};

#define IDLETOKEN_GGUF_GEOM_N \
    (sizeof(IDLETOKEN_GGUF_GEOM)/sizeof(IDLETOKEN_GGUF_GEOM[0]))

#endif /* IDLETOKEN_GGUF_GEOM_H */
