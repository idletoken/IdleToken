/* model.c — the engine-side model registry (see include/idletoken_model.h).
 * Keep in sync with models/<id>.json (client manifests) — same ids, same numbers.
 */
#include "idletoken_model.h"

#include <stdio.h>
#include <string.h>

#define GiB (1024ull * 1024 * 1024)

/* Qwen3-8B precision menu (small-model-design.md §3). ESTIMATES from typical
 * community GGUF sizes; embd+output head ≈ 15% of params (vocab 151936 ×
 * n_embd 4096, untied). Corrected against real GGUFs on the DGX. Index 0 =
 * default (Q4_K_M) — mirrored into the spec scalars below. */
/* Qwen3.5-0.8B — hybrid Gated DeltaNet (3 linear : 1 full). Sizes from the
 * real unsloth GGUFs — ALL FIVE MEASURED (2026-07-28), not scaled from bpw:
 * each file's tensor directory was summed, blk.N.* to layer and the rest to
 * shared. These numbers drive mode selection and the layer split, so an
 * estimate that is 20% high can turn a workable GPU_ONLY plan into HYBRID. */
/* Every entry below is MEASURED, never estimated: scripts/measure_gguf.py reads
 * the file's tensor directory over HTTP Range and sums blk.N.* into layer and
 * the rest into shared, asserting layer + shared + header == file size. An
 * invented number here becomes a "fact" inside the resource planner, silently.
 *
 * These lists used to hold one quant each precisely because nobody had measured
 * the others -- the right call, but it left the precision menu present on some
 * models and absent on others. Meanwhile QWEN3_8B_VARIANTS had quietly broken
 * the rule: four of its five entries were round numbers somebody estimated,
 * 0.4-0.6 GB high, and its BF16 entry named a file that does not exist in
 * Qwen/Qwen3-8B-GGUF at all -- picking it could only ever 404. Measuring is now
 * one command, so "we have not measured it" is no longer a reason to ship a
 * short menu, and estimating is not a shortcut anyone needs to take.
 *
 * A quant appears here only if the repo really publishes it AND ds4x can
 * dequantize it (src/ds4x/ds4x_quant.c: F32/F16/BF16/Q8_0/Q4_0/Q2_K/Q4_K/Q5_K/
 * Q6_K/IQ2_XXS) -- which is why the IQ4_XS and Q3_K files upstream ships are
 * absent, and why three repos have no BF16 row. */
static const idletoken_model_variant QWEN35_4B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 2208487424ull, .shared_weight_bytes =  521482240ull, .gguf = "Qwen3.5-4B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 2611206144ull, .shared_weight_bytes =  521482240ull, .gguf = "Qwen3.5-4B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes = 2993506304ull, .shared_weight_bytes =  521482240ull, .gguf = "Qwen3.5-4B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes = 3795994624ull, .shared_weight_bytes =  675440640ull, .gguf = "Qwen3.5-4B-Q8_0.gguf" },
    { .quant = "BF16",   .layer_weight_bytes = 7142017024ull, .shared_weight_bytes = 1271408640ull, .gguf = "Qwen3.5-4B-BF16.gguf" },
};

static const idletoken_model_variant QWEN35_9B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes =  4263053312ull, .shared_weight_bytes = 1406500864ull, .gguf = "Qwen3.5-9B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes =  5033232384ull, .shared_weight_bytes = 1533640704ull, .gguf = "Qwen3.5-9B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes =  5778606080ull, .shared_weight_bytes = 1668726784ull, .gguf = "Qwen3.5-9B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes =  7355140096ull, .shared_weight_bytes = 2161393664ull, .gguf = "Qwen3.5-9B-Q8_0.gguf" },
    { .quant = "BF16",   .layer_weight_bytes = 13841238016ull, .shared_weight_bytes = 4068491264ull, .gguf = "Qwen3.5-9B-BF16.gguf" },
};

static const idletoken_model_variant QWEN35_27B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 14971693056ull, .shared_weight_bytes = 1758126080ull, .gguf = "Qwen3.5-27B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 17680951296ull, .shared_weight_bytes = 1917050880ull, .gguf = "Qwen3.5-27B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes = 20357031936ull, .shared_weight_bytes = 2085908480ull, .gguf = "Qwen3.5-27B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes = 25883027456ull, .shared_weight_bytes = 2701742080ull, .gguf = "Qwen3.5-27B-Q8_0.gguf" },
};

static const idletoken_model_variant QWEN35_35B_A3B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 21047503360ull, .shared_weight_bytes =  957530112ull, .gguf = "Qwen3.5-35B-A3B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 25157921280ull, .shared_weight_bytes = 1080696832ull, .gguf = "Qwen3.5-35B-A3B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes = 27761175040ull, .shared_weight_bytes = 1080696832ull, .gguf = "Qwen3.5-35B-A3B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes = 35811453440ull, .shared_weight_bytes = 1080696832ull, .gguf = "Qwen3.5-35B-A3B-Q8_0.gguf" },
};

static const idletoken_model_variant QWEN35_08B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 312962304ull, .shared_weight_bytes = 208592896ull, .gguf = "Qwen3.5-0.8B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 370502912ull, .shared_weight_bytes = 208592896ull, .gguf = "Qwen3.5-0.8B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes = 419474688ull, .shared_weight_bytes = 208592896ull, .gguf = "Qwen3.5-0.8B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes = 530705664ull, .shared_weight_bytes = 270176256ull, .gguf = "Qwen3.5-0.8B-Q8_0.gguf" },
    { .quant = "BF16",   .layer_weight_bytes = 997219584ull, .shared_weight_bytes = 508563456ull, .gguf = "Qwen3.5-0.8B-BF16.gguf" },
};

static const idletoken_model_variant QWEN3_8B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 4161245184ull, .shared_weight_bytes =  860581888ull, .gguf = "Qwen3-8B-Q4_K_M.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 4906782720ull, .shared_weight_bytes =  938373120ull, .gguf = "Qwen3-8B-Q5_K_M.gguf" },
    { .quant = "Q6_K",   .layer_weight_bytes = 5698916352ull, .shared_weight_bytes = 1021026304ull, .gguf = "Qwen3-8B-Q6_K.gguf" },
    { .quant = "Q8_0",   .layer_weight_bytes = 7381094400ull, .shared_weight_bytes = 1322467328ull, .gguf = "Qwen3-8B-Q8_0.gguf" },
};

/* DSv4-Flash precision menu. Both entries are the OFFICIAL 0731 release
 * (deepseek-ai/DeepSeek-V4-Flash-0731); the preview conversions are gone.
 * MEASURED, not estimated: each file's GGUF tensor directory was summed,
 * blk.N.* to layer and the rest to shared, and layer+shared+data_start was
 * checked against the published file size to the byte. The old 81 GiB / 1 GiB
 * pair was an estimate that split layer-vs-shared wrong by 1.75 GiB — shared
 * is token_embd (first stage) + output head (last stage), so getting that
 * split wrong biases the PP plan at both ends of the pipeline. */
static const idletoken_model_variant DSV4_FLASH_VARIANTS[] = {
    { .quant = "IQ2_XXS+Q2_K",        .layer_weight_bytes = 85092941824ull, .shared_weight_bytes = 1621835840ull,
      .gguf = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" },
    { .quant = "IQ2_XXS+Q4_K-L37-42", .layer_weight_bytes = 95964577792ull, .shared_weight_bytes = 1621835840ull,
      .gguf = "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf" },
};

/* Registry order matters only for [0] == default (DSv4-Flash). */
static const idletoken_model_spec MODELS[] = {
    {
        .id      = "deepseek-v4-flash",
        .label   = "DeepSeek V4 Flash",
        .backend = IDLETOKEN_BACKEND_DS4,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,  /* 304B at Q2 = 80.76 GiB; no single home machine holds it */
        .n_layers = 43,
        .n_embd   = 4096,
        .hc_streams = 4,           /* mHC 4-stream hyper-connections */
        .n_vocab  = 129280,
        .layer_weight_bytes  = 85092941824ull, /* == IQ2_XXS+Q2_K variant (measured) */
        .shared_weight_bytes = 1621835840ull,  /* token_embd + output + norms */
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_DSV4,          /* per-tier table in overhead() */
        /* The filename antirez actually publishes. "ds4flash.gguf" was a
         * local nickname: every gate passed --model-path explicitly, so the
         * drift stayed invisible until something (the topology matrix, the
         * weight fetcher) tried to RESOLVE the default name. The -0731 suffix
         * is the official release; the unsuffixed files are the superseded
         * preview and must not be resolved by default any more. */
        .default_gguf = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        .variants = DSV4_FLASH_VARIANTS,
        .n_variants = sizeof(DSV4_FLASH_VARIANTS) / sizeof(DSV4_FLASH_VARIANTS[0]),
        .default_variant = 0,      /* IQ2_XXS+Q2_K */
    },
    {
        /* First small dense model (small-model-design.md). GQA, not MLA —
         * runs the ds4x GQA path (Phase S-B). Scalars mirror the default
         * variant (Q4_K_M) so quant-unaware callers see the default size.
         * KV: 2·n_head_kv(8)·head_dim(128)·2 (fp16) = 4096 B/token/layer. */
        .id      = "qwen3.5-0.8b",
        .label   = "Qwen3.5 0.8B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-0.8B Q4_K_M GGUF (DGX): output
                                    * matches llama.cpp token-for-token on
                                    * greedy prompts (design doc §4d) */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 24,
        .n_embd   = 1024,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 312962304ull,   /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 208592896ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        /* full layers: 2 kv heads × 256 head_dim × 2 (K+V) × 4 B = 4096 B/token */
        .kv_bytes_per_token_layer = 4096,
        /* linear layers: 16 heads × 128 × 128 state + conv window (3×6144),
         * fp32 → 1 MiB + 72 KiB, ctx-INDEPENDENT */
        .state_bytes_per_layer = 1122304,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(0.8 * (double)GiB),
        .default_gguf = "Qwen3.5-0.8B-Q4_K_M.gguf",
        .variants = QWEN35_08B_VARIANTS,
        .n_variants = sizeof(QWEN35_08B_VARIANTS) / sizeof(QWEN35_08B_VARIANTS[0]),
        .default_variant = 0,      /* Q4_K_M */
    },
    {
        /* Same qwen35 hybrid architecture as the 0.8B, but the linear path
         * SHARES k-heads: k_heads(16) < v_heads(32), so each key head serves
         * two value heads. The 0.8B is 16/16 and can never exercise that
         * branch — which is exactly why this entry is worth having. */
        .id      = "qwen3.5-4b",
        .label   = "Qwen3.5 4B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-4B Q4_K_M GGUF (DGX): CPU and
                                    * CUDA agree token-for-token, and the
                                    * cpu/gpu recurrence check stays at 6e-08
                                    * over 168 chunks (design doc §4j) */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 32,
        .n_embd   = 2560,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 2208487424ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 521482240ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        /* full layers: 4 kv heads × 256 head_dim × 2 (K+V) × 4 B = 8192 B/token */
        .kv_bytes_per_token_layer = 8192,
        /* linear layers: 32 heads × 128 × 128 state + conv window (3×8192),
         * fp32 → 2 MiB + 96 KiB, ctx-INDEPENDENT */
        .state_bytes_per_layer = 2195456,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.2 * (double)GiB),
        .default_gguf = "Qwen3.5-4B-Q4_K_M.gguf",
        .variants = QWEN35_4B_VARIANTS,
        .n_variants = sizeof(QWEN35_4B_VARIANTS) / sizeof(QWEN35_4B_VARIANTS[0]),
        .default_variant = 0,      /* Q4_K_M */
    },
    {
        /* Same shape as the 4B in every field that the forward branches on
         * (32 layers, k_heads 16 / v_heads 32, full layers 16/4 heads × 256,
         * interval 4) — only n_embd and the FFN width differ. That is why it
         * needed no code at all once the 4B's k-head sharing was fixed. */
        .id      = "qwen3.5-9b",
        .label   = "Qwen3.5 9B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-9B Q4_K_M GGUF (DGX): greedy
                                    * output matches llama.cpp word for word */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 32,
        .n_embd   = 4096,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 4263053312ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 1406500864ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 8192,
        .state_bytes_per_layer = 2195456,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .default_gguf = "Qwen3.5-9B-Q4_K_M.gguf",
        .variants = QWEN35_9B_VARIANTS,
        .n_variants = sizeof(QWEN35_9B_VARIANTS) / sizeof(QWEN35_9B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* grp=3 on the linear path (k_heads 16 / v_heads 48) — the 0.8B is 1,
         * the 4B/9B are 2. Three different sharing ratios have now run on real
         * weights, which is what makes the strided k-head mapping trustworthy
         * rather than a rule that happened to fit one model. */
        .id      = "qwen3.5-27b",
        .label   = "Qwen3.5 27B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-27B Q4_K_M GGUF (DGX): the
                                    * counting prompt matches llama.cpp word
                                    * for word; layer-0 attn_output/final_output
                                    * agree to <0.1% (design doc §4j) */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 64,
        .n_embd   = 5120,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 14971693056ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 1758126080ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 8192,   /* 4 kv heads × 256 × 2 × 4 B */
        .state_bytes_per_layer = 3268608,   /* 48×128×128×4 + conv 3×10240×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(2.0 * (double)GiB),
        .default_gguf = "Qwen3.5-27B-Q4_K_M.gguf",
        .variants = QWEN35_27B_VARIANTS,
        .n_variants = sizeof(QWEN35_27B_VARIANTS) / sizeof(QWEN35_27B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* qwen35moe: hybrid linear attention AND a 256-expert MoE. This is the
         * FIRST MoE model the ds4x path has ever run on real weights — DSv4
         * (80 GB) and GLM-5.2 (240 GB) do not fit anything we have. It also
         * carries the one piece of new math: the shared expert is scaled by a
         * scalar sigmoid gate. */
        .id      = "qwen3.5-35b-a3b",
        .label   = "Qwen3.5 35B-A3B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-35B-A3B Q4_K_M GGUF (DGX): the
                                    * counting prompt matches llama.cpp word
                                    * for word and layer-0 agrees (§4n) */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 40,
        .n_embd   = 2048,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 21047503360ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 957530112ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 4096,   /* 2 kv heads × 256 × 2 × 4 B */
        .state_bytes_per_layer = 2195456,   /* 32×128×128×4 + conv 3×8192×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .default_gguf = "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        .variants = QWEN35_35B_A3B_VARIANTS,
        .n_variants = sizeof(QWEN35_35B_A3B_VARIANTS) / sizeof(QWEN35_35B_A3B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        .id      = "qwen3-8b",
        .label   = "Qwen3 8B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-27 on real Qwen3-8B
                                    * Q4_K_M GGUF (DGX): coherent output, and
                                    * the CUDA path matches CPU token-for-token */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 36,
        .n_embd   = 4096,
        .hc_streams = 1,           /* plain residual */
        .n_vocab  = 151936,
        .layer_weight_bytes  = 4161245184ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 860581888ull,
        .ctx_max  = 40960,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_GQA,
        .kv_bytes_per_token_layer = 4096,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB), /* n_embd 4096 + CUDA */
        .default_gguf = "Qwen3-8B-Q4_K_M.gguf",
        .variants = QWEN3_8B_VARIANTS,
        .n_variants = sizeof(QWEN3_8B_VARIANTS) / sizeof(QWEN3_8B_VARIANTS[0]),
        .default_variant = 0,      /* Q4_K_M */
    },
    {
        /* ESTIMATES from unsloth/GLM-5.2-GGUF UD-IQ2_XXS (~238 GB) — correct
         * when the ds4x backend lands (Phase B). MLA KV: (kv_lora_rank 512 +
         * rope 64) × 2 bytes ≈ 1152 B/token/layer. */
        .id      = "glm-5.2",
        .label   = "GLM-5.2",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 0,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,  /* ~240 GiB at Q2 */
        .n_layers = 78,            /* 3 dense + 75 MoE */
        .n_embd   = 6144,
        .hc_streams = 1,           /* plain residual */
        .n_vocab  = 154880,
        .layer_weight_bytes  = 236ull * GiB,
        .shared_weight_bytes = 2ull * GiB,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 4,  /* DSA indexer shared per 4-layer group */
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,  /* activations at n_embd 6144 + CUDA */
        .default_gguf = "glm-5.2-q2.gguf",
    },
    {
        /* ESTIMATES (1T total / 32B active, DeepSeek2-shaped; ~240 GB at
         * 1.8-bit community GGUF). Runs on ds4x once Phase B lands. */
        .id      = "kimi-k2.5",
        .label   = "Kimi K2.5",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 0,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 61,
        .n_embd   = 7168,
        .hc_streams = 1,
        .n_vocab  = 163840,
        .layer_weight_bytes  = 236ull * GiB,
        .shared_weight_bytes = 2ull * GiB,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,
        .default_gguf = "kimi-k2.5-q2.gguf",
    },
    {
        /* Placeholder — weights unreleased until 2026-07-27; KDA unsupported;
         * ~1.4 TB native MXFP4 likely exceeds home clusters permanently.
         * Registered so UIs can show it greyed out; numbers unknown. */
        .id      = "kimi-k3",
        .label   = "Kimi K3",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 0,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 0,
        .n_embd   = 0,
        .hc_streams = 1,
        .n_vocab  = 0,
        .layer_weight_bytes  = 0,
        .shared_weight_bytes = 0,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 0,
        .overhead_base_bytes = 0,
        .default_gguf = "",
    },
};

const idletoken_model_spec *idletoken_model_get(const char *id) {
    if (!id || !id[0]) return NULL;
    for (size_t i = 0; i < sizeof(MODELS) / sizeof(MODELS[0]); i++)
        if (strcmp(MODELS[i].id, id) == 0) return &MODELS[i];
    return NULL;
}

const idletoken_model_spec *idletoken_model_default(void) {
    return &MODELS[0];
}

int idletoken_model_count(void) {
    return (int)(sizeof(MODELS) / sizeof(MODELS[0]));
}

const idletoken_model_spec *idletoken_model_at(int index) {
    if (index < 0 || index >= idletoken_model_count()) return NULL;
    return &MODELS[index];
}

int idletoken_model_may_cluster(const idletoken_model_spec *m, char *why, size_t why_cap) {
    if (!m) {
        if (why && why_cap) snprintf(why, why_cap, "unknown model");
        return 0;
    }
    if (m->deployment == IDLETOKEN_DEPLOY_CLUSTER) return 1;
    if (why && why_cap) {
        if (m->deployment == IDLETOKEN_DEPLOY_SINGLE_NODE)
            snprintf(why, why_cap,
                     "%s runs on ONE machine only. Small models fit a single node, "
                     "and splitting one across a LAN spends more time on pipeline "
                     "round-trips than on compute. Serve it standalone, or pick a "
                     "model marked for clusters.", m->label);
        else
            /* Only reachable if a model was added without declaring
             * "deployment" — a build-time mistake, so name it as one instead
             * of blaming the user's setup. */
            snprintf(why, why_cap,
                     "%s does not declare how it may be deployed (models/%s.json "
                     "\"deployment\"), so this build refuses to cluster it.",
                     m->label, m->id);
    }
    return 0;
}

const idletoken_model_variant *idletoken_model_variant_get(const idletoken_model_spec *m,
                                                     const char *quant) {
    if (!m || m->n_variants == 0) return NULL;  /* caller uses scalar fields */
    if (quant && quant[0])
        for (uint8_t i = 0; i < m->n_variants; i++)
            if (strcmp(m->variants[i].quant, quant) == 0) return &m->variants[i];
    return &m->variants[m->default_variant];    /* NULL/unknown → default */
}

void idletoken_model_weight_bytes(const idletoken_model_spec *m, const char *quant,
                               uint64_t *layer_out, uint64_t *shared_out) {
    const idletoken_model_variant *v = idletoken_model_variant_get(m, quant);
    if (layer_out)  *layer_out  = v ? v->layer_weight_bytes  : m->layer_weight_bytes;
    if (shared_out) *shared_out = v ? v->shared_weight_bytes : m->shared_weight_bytes;
}

uint64_t idletoken_model_overhead(const idletoken_model_spec *m, uint32_t ctx_size,
                               int layers_on_node) {
    if (m->kv_kind == IDLETOKEN_KV_HYBRID) {
        if (layers_on_node < 1) layers_on_node = 1;
        /* Split the node's layers into full-attention and linear by period.
         * Worst case for the caller is assuming the full layers land here, so
         * round the full count UP — under-estimating KV is what OOMs a node. */
        const uint32_t iv = m->full_attn_interval ? m->full_attn_interval : 1;
        const uint64_t n_full = ((uint64_t)layers_on_node + iv - 1) / iv;
        const uint64_t n_lin  = (uint64_t)layers_on_node - n_full;
        const uint64_t kv = (uint64_t)m->kv_bytes_per_token_layer * ctx_size * n_full
                          + (uint64_t)m->state_bytes_per_layer * n_lin;
        return m->overhead_base_bytes + kv + (m->overhead_base_bytes + kv) / 10;
    }
    if (m->kv_kind == IDLETOKEN_KV_MLA || m->kv_kind == IDLETOKEN_KV_GQA) {
        if (layers_on_node < 1) layers_on_node = 1;
        uint64_t kv = (uint64_t)m->kv_bytes_per_token_layer * ctx_size *
                      (uint64_t)layers_on_node;
        /* +10% margin, same policy as the DSv4 table's derivation. */
        return m->overhead_base_bytes + kv + (m->overhead_base_bytes + kv) / 10;
    }
    /* IDLETOKEN_KV_DSV4: calibrated per-tier table (docs/architecture.md §5). */
    (void)layers_on_node;
    if (ctx_size <= 8192)    return (uint64_t)(1.5 * (double)GiB);
    if (ctx_size <= 32768)   return 2ull * GiB;
    if (ctx_size <= 131072)  return 3ull * GiB;
    if (ctx_size <= 524288)  return 6ull * GiB;
    return 9ull * GiB;
}
