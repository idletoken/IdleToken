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
    { .quant = "IQ2_XXS", .layer_weight_bytes = 1520217248ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 1759997088ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-IQ2_M.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 1940825248ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q2_K_XL.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 1949047968ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-IQ3_XXS.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 2105791648ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 2293388448ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 2436420768ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 2477053088ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 2579944608ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-IQ4_NL.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 2583221408ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q4_0.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 2590430368ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q4_K_S.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 2740937888ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 2784416928ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 2912109728ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 3024934048ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 3143656608ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 3250869408ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 3525956768ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 4145548448ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 4482403488ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 5952048288ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 8424393632ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-4B-BF16.gguf" },
};

static const idletoken_model_variant QWEN35_9B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 3190613216ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 3649365216ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-IQ2_M.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 4016235744ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-IQ3_XXS.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 4121781472ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q2_K_XL.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 4316865760ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 4673643744ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 5053834464ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 5168653536ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 5371028704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-IQ4_NL.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 5379417312ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q4_0.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 5394097376ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q4_K_S.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 5680522464ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 5837251808ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 5966095584ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 6361146592ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 6577841376ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 6743680224ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 7458301152ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 8756929760ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 9527502048ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 12974040288ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 17920697312ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-9B-BF16.gguf" },
};

static const idletoken_model_variant QWEN38_27B_VARIANTS[] = {
    { .quant = "IQ1_S", .layer_weight_bytes = 6192222208ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ1_S.gguf" },
    { .quant = "IQ1_M", .layer_weight_bytes = 6729166848ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ1_M.gguf" },
    { .quant = "IQ2_XXS", .layer_weight_bytes = 7266070528ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_S", .layer_weight_bytes = 8371970048ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ2_S.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 9828981664ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q2_K_XL.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 10934860704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ3_XXS.gguf" },
    { .quant = "IQ3_S", .layer_weight_bytes = 12040883104ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ3_S.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 13146393504ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 14252845984ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-IQ4_XS.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 15358213024ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q4_K_S.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 16056478688ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-Q4_0.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 16464440224ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 17540705248ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 17559178144ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 18665753504ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 19771509664ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 20876938144ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 21983677344ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q6_K.gguf" },
    { .quant = "Q6_K_M", .layer_weight_bytes = 23088409504ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q6_K_M.gguf" },
    { .quant = "Q6_K_L", .layer_weight_bytes = 24193919904ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q6_K_L.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 25299061664ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_K_L", .layer_weight_bytes = 28045695904ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q8_K_L.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 29047086048ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 31457991680ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.8-27B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 54657735616ull, .shared_weight_bytes = 0ull, .gguf = "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf" },
};

static const idletoken_model_variant QWEN35_27B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 8573593504ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 10188072864ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-IQ2_M.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 11213752224ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q2_K_XL.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 11506493344ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-IQ3_XXS.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 12289423264ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 13505116064ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 14438533024ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 14977484704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 15687894944ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-IQ4_NL.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 15721973664ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q4_0.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 15769159584ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q4_K_S.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 16740812704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 17182934944ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 17621125024ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 18889000864ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 19608995744ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 20171253664ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 22453933984ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 25675642784ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 28595763104ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 35528652704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-27B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 53808281472ull, .shared_weight_bytes = 0ull, .gguf = "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf" },
};

static const idletoken_model_variant QWEN35_35B_A3B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 10656955008ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 11391613568ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ2_M.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 12156754560ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 13080066688ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf" },
    { .quant = "IQ3_S", .layer_weight_bytes = 13583383168ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ3_S.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 15265856128ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 16356375168ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 16601176704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 17486174848ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 17821719168ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-IQ4_NL.gguf" },
    { .quant = "Q4_K_L", .layer_weight_bytes = 20205632160ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 20673845888ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q4_K_S.gguf" },
    { .quant = "Qwen3.5-35B-A3B-MXFP4_MOE", .layer_weight_bytes = 21587638912ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-MXFP4_MOE.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 22016023168ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q4_K_M.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 22241950336ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 24823544448ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 26249607808ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 26385922688ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K_S", .layer_weight_bytes = 28515105440ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q6_K_S.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 28852861568ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 32071842432ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 36903139968ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 48688560768ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 69376637824ull, .shared_weight_bytes = 0ull, .gguf = "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf" },
};

static const idletoken_model_variant QWEN35_08B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 338227456ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 371933440ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-IQ2_M.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 398237952ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-IQ3_XXS.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 417718528ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q2_K_XL.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 440750336ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 470167808ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 492216576ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 492605696ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 506859776ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-IQ4_NL.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 507154688ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q4_0.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 508104960ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q4_K_S.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 532517120ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 535171328ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 558772480ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 568889600ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 590057728ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 606585088ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 639029504ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 771092736ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 811843840ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 1186443520ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 1516744736ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-0.8B-BF16.gguf" },
};

static const idletoken_model_variant QWEN3_8B_VARIANTS[] = {
    { .quant = "Q4_K_M", .layer_weight_bytes = 5027783488ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3-8B-Q4_K_M.gguf" },
    { .quant = "Q5_0", .layer_weight_bytes = 5720761152ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3-8B-Q5_0.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 5851112224ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3-8B-Q5_K_M.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 6725899040ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3-8B-Q6_K.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 8709518112ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3-8B-Q8_0.gguf" },
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
    { .quant = "IQ1_S", .layer_weight_bytes = 82539237024ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_S/DeepSeek-V4-Flash-UD-IQ1_S-00001-of-00003.gguf" },
    { .quant = "IQ1_M", .layer_weight_bytes = 86901313152ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_M/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf" },
    { .quant = "IQ2_XXS", .layer_weight_bytes = 90860736128ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_XXS/DeepSeek-V4-Flash-UD-IQ2_XXS-00001-of-00003.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 90926927488ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_M/DeepSeek-V4-Flash-UD-IQ2_M-00001-of-00003.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 96832507552ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q2_K_XL/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 102999887616ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf" },
    { .quant = "IQ3_S", .layer_weight_bytes = 117310852864ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_S/DeepSeek-V4-Flash-UD-IQ3_S-00001-of-00004.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 129319997216ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_M/DeepSeek-V4-Flash-UD-Q3_K_M-00001-of-00004.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 129448242976ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_XL/DeepSeek-V4-Flash-UD-Q3_K_XL-00001-of-00004.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 137903959808ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_NL/DeepSeek-V4-Flash-UD-IQ4_NL-00001-of-00004.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 137903959808ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_XS/DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 155095240320ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 161869614720ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf" },
};

/* Registry order matters only for [0] == default (DSv4-Flash). */
/* GLM-5.2: measured from the repo named in models/glm-5.2.json (HF API,
 * 2026-08-15). Every quant here is a SPLIT GGUF — .gguf names part one,
 * which is what the engine is handed; llama.cpp opens the rest by name.
 * llama.cpp reads all of these, including the sub-2-bit ones (the frozen
 * ds4x dequantizer does not, but these models do not run on it). */
static const idletoken_model_variant GLM52_VARIANTS[] = {
    { .quant = "IQ1_S", .layer_weight_bytes = 216715360960ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf" },  /* 6 parts, 202 GiB */
    { .quant = "IQ1_M", .layer_weight_bytes = 228492966624ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_M/GLM-5.2-UD-IQ1_M-00001-of-00006.gguf" },  /* 6 parts, 213 GiB */
    { .quant = "IQ2_XXS", .layer_weight_bytes = 238458632928ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf" },  /* 6 parts, 222 GiB */
    { .quant = "IQ2_M", .layer_weight_bytes = 238577580768ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf" },  /* 6 parts, 222 GiB */
    { .quant = "Q2_K_XL", .layer_weight_bytes = 253878401856ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf" },  /* 7 parts, 236 GiB */
    { .quant = "IQ3_XXS", .layer_weight_bytes = 281688431424ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_XXS/GLM-5.2-UD-IQ3_XXS-00001-of-00007.gguf" },  /* 7 parts, 262 GiB */
    { .quant = "IQ3_S", .layer_weight_bytes = 308641029024ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_S/GLM-5.2-UD-IQ3_S-00001-of-00008.gguf" },  /* 8 parts, 287 GiB */
    { .quant = "Q3_K_M", .layer_weight_bytes = 342735510656ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_M/GLM-5.2-UD-Q3_K_M-00001-of-00009.gguf" },  /* 9 parts, 319 GiB */
    { .quant = "Q3_K_XL", .layer_weight_bytes = 342965972096ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_XL/GLM-5.2-UD-Q3_K_XL-00001-of-00009.gguf" },  /* 9 parts, 319 GiB */
    { .quant = "IQ4_XS", .layer_weight_bytes = 365313223776ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_XS/GLM-5.2-UD-IQ4_XS-00001-of-00009.gguf" },  /* 9 parts, 340 GiB */
    { .quant = "IQ4_NL", .layer_weight_bytes = 372661644384ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_NL/GLM-5.2-UD-IQ4_NL-00001-of-00009.gguf" },  /* 9 parts, 347 GiB */
    { .quant = "Q4_K_S", .layer_weight_bytes = 436431842432ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_S/GLM-5.2-UD-Q4_K_S-00001-of-00010.gguf" },  /* 10 parts, 406 GiB */
    { .quant = "Q4_K_M", .layer_weight_bytes = 465825525088ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_M/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf" },  /* 11 parts, 434 GiB */
    { .quant = "Q4_K_XL", .layer_weight_bytes = 467289111904ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf" },  /* 11 parts, 435 GiB */
    { .quant = "Q5_K_S", .layer_weight_bytes = 527259270528ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q5_K_S/GLM-5.2-UD-Q5_K_S-00001-of-00012.gguf" },  /* 12 parts, 491 GiB */
    { .quant = "Q5_K_M", .layer_weight_bytes = 560830479904ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q5_K_M/GLM-5.2-UD-Q5_K_M-00001-of-00013.gguf" },  /* 13 parts, 522 GiB */
    { .quant = "Q5_K_XL", .layer_weight_bytes = 562466258496ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q5_K_XL/GLM-5.2-UD-Q5_K_XL-00001-of-00013.gguf" },  /* 13 parts, 524 GiB */
    { .quant = "Q6_K", .layer_weight_bytes = 625858969376ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q6_K/GLM-5.2-UD-Q6_K-00001-of-00014.gguf" },  /* 14 parts, 583 GiB */
    { .quant = "Q6_K_XL", .layer_weight_bytes = 684369510240ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q6_K_XL/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf" },  /* 16 parts, 637 GiB */
    { .quant = "Q8_0", .layer_weight_bytes = 801357672256ull, .shared_weight_bytes = 0ull, .gguf = "Q8_0/GLM-5.2-Q8_0-00001-of-00017.gguf" },  /* 17 parts, 746 GiB */
    { .quant = "Q8_K_XL", .layer_weight_bytes = 819700080736ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/GLM-5.2-UD-Q8_K_XL-00001-of-00018.gguf" },  /* 18 parts, 763 GiB */
    { .quant = "BF16", .layer_weight_bytes = 1507988023008ull, .shared_weight_bytes = 0ull, .gguf = "BF16/GLM-5.2-BF16-00001-of-00033.gguf" },  /* 33 parts, 1404 GiB */
};

/* Kimi K2.5: measured from the repo named in models/kimi-k2.5.json (HF API,
 * 2026-08-15). Every quant here is a SPLIT GGUF — .gguf names part one,
 * which is what the engine is handed; llama.cpp opens the rest by name.
 * llama.cpp reads all of these, including the sub-2-bit ones (the frozen
 * ds4x dequantizer does not, but these models do not run on it). */
static const idletoken_model_variant KIMI_K25_VARIANTS[] = {
    { .quant = "IQ1_S", .layer_weight_bytes = 210299214240ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ1_S/moonshotai_Kimi-K2.5-IQ1_S-00001-of-00006.gguf" },  /* 6 parts, 196 GiB */
    { .quant = "IQ1_M", .layer_weight_bytes = 219584616864ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ1_M/moonshotai_Kimi-K2.5-IQ1_M-00001-of-00006.gguf" },  /* 6 parts, 205 GiB */
    { .quant = "IQ2_XXS", .layer_weight_bytes = 245277612576ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ2_XXS/moonshotai_Kimi-K2.5-IQ2_XXS-00001-of-00007.gguf" },  /* 7 parts, 228 GiB */
    { .quant = "IQ2_XS", .layer_weight_bytes = 283113380512ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ2_XS/moonshotai_Kimi-K2.5-IQ2_XS-00001-of-00008.gguf" },  /* 8 parts, 264 GiB */
    { .quant = "IQ2_S", .layer_weight_bytes = 283797330528ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ2_S/moonshotai_Kimi-K2.5-IQ2_S-00001-of-00008.gguf" },  /* 8 parts, 264 GiB */
    { .quant = "IQ2_M", .layer_weight_bytes = 322954304224ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ2_M/moonshotai_Kimi-K2.5-IQ2_M-00001-of-00009.gguf" },  /* 9 parts, 301 GiB */
    { .quant = "Q2_K", .layer_weight_bytes = 358637722528ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q2_K/moonshotai_Kimi-K2.5-Q2_K-00001-of-00010.gguf" },  /* 10 parts, 334 GiB */
    { .quant = "Q2_K_L", .layer_weight_bytes = 359784602528ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q2_K_L/moonshotai_Kimi-K2.5-Q2_K_L-00001-of-00010.gguf" },  /* 10 parts, 335 GiB */
    { .quant = "IQ3_XXS", .layer_weight_bytes = 404581325856ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ3_XXS/moonshotai_Kimi-K2.5-IQ3_XXS-00001-of-00011.gguf" },  /* 11 parts, 377 GiB */
    { .quant = "IQ3_XS", .layer_weight_bytes = 420081867776ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ3_XS/moonshotai_Kimi-K2.5-IQ3_XS-00001-of-00011.gguf" },  /* 11 parts, 391 GiB */
    { .quant = "Q3_K_S", .layer_weight_bytes = 444827360352ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q3_K_S/moonshotai_Kimi-K2.5-Q3_K_S-00001-of-00012.gguf" },  /* 12 parts, 414 GiB */
    { .quant = "IQ3_M", .layer_weight_bytes = 466870099264ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ3_M/moonshotai_Kimi-K2.5-IQ3_M-00001-of-00013.gguf" },  /* 13 parts, 435 GiB */
    { .quant = "Q3_K_M", .layer_weight_bytes = 467321920864ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q3_K_M/moonshotai_Kimi-K2.5-Q3_K_M-00001-of-00013.gguf" },  /* 13 parts, 435 GiB */
    { .quant = "Q3_K_L", .layer_weight_bytes = 487537745216ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q3_K_L/moonshotai_Kimi-K2.5-Q3_K_L-00001-of-00013.gguf" },  /* 13 parts, 454 GiB */
    { .quant = "Q3_K_XL", .layer_weight_bytes = 488565349696ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q3_K_XL/moonshotai_Kimi-K2.5-Q3_K_XL-00001-of-00013.gguf" },  /* 13 parts, 455 GiB */
    { .quant = "IQ4_XS", .layer_weight_bytes = 547605634464ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ4_XS/moonshotai_Kimi-K2.5-IQ4_XS-00001-of-00014.gguf" },  /* 14 parts, 510 GiB */
    { .quant = "IQ4_NL", .layer_weight_bytes = 579469909568ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-IQ4_NL/moonshotai_Kimi-K2.5-IQ4_NL-00001-of-00016.gguf" },  /* 16 parts, 540 GiB */
    { .quant = "Q4_K_S", .layer_weight_bytes = 579815480896ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q4_K_S/moonshotai_Kimi-K2.5-Q4_K_S-00001-of-00016.gguf" },  /* 16 parts, 540 GiB */
    { .quant = "Q4_K_M", .layer_weight_bytes = 580355550784ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q4_K_M/moonshotai_Kimi-K2.5-Q4_K_M-00001-of-00016.gguf" },  /* 16 parts, 540 GiB */
    { .quant = "Q5_K_S", .layer_weight_bytes = 580517834304ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q5_K_S/moonshotai_Kimi-K2.5-Q5_K_S-00001-of-00016.gguf" },  /* 16 parts, 541 GiB */
    { .quant = "Q4_K_L", .layer_weight_bytes = 581227179584ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q4_K_L/moonshotai_Kimi-K2.5-Q4_K_L-00001-of-00016.gguf" },  /* 16 parts, 541 GiB */
    { .quant = "Q5_K_M", .layer_weight_bytes = 581232893504ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q5_K_M/moonshotai_Kimi-K2.5-Q5_K_M-00001-of-00016.gguf" },  /* 16 parts, 541 GiB */
    { .quant = "Q6_K", .layer_weight_bytes = 582178528832ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q6_K/moonshotai_Kimi-K2.5-Q6_K-00001-of-00016.gguf" },  /* 16 parts, 542 GiB */
    { .quant = "Q8_0", .layer_weight_bytes = 583710875200ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q8_0/moonshotai_Kimi-K2.5-Q8_0-00001-of-00016.gguf" },  /* 16 parts, 544 GiB */
    { .quant = "Q4_0", .layer_weight_bytes = 589765221952ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q4_0/moonshotai_Kimi-K2.5-Q4_0-00001-of-00016.gguf" },  /* 16 parts, 549 GiB */
    { .quant = "Q4_1", .layer_weight_bytes = 643155730176ull, .shared_weight_bytes = 0ull, .gguf = "moonshotai_Kimi-K2.5-Q4_1/moonshotai_Kimi-K2.5-Q4_1-00001-of-00017.gguf" },  /* 17 parts, 599 GiB */
};

/* Kimi K3: measured from the repo named in models/kimi-k3.json (HF API,
 * 2026-08-15). Every quant here is a SPLIT GGUF — .gguf names part one,
 * which is what the engine is handed; llama.cpp opens the rest by name.
 * llama.cpp reads all of these, including the sub-2-bit ones (the frozen
 * ds4x dequantizer does not, but these models do not run on it). */
static const idletoken_model_variant KIMI_K3_VARIANTS[] = {
    { .quant = "Q1_0", .layer_weight_bytes = 466369456320ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q1_0/Kimi-K3-UD-Q1_0-00001-of-00011.gguf" },  /* 11 parts, 434 GiB */
    { .quant = "TQ1_0", .layer_weight_bytes = 508912281952ull, .shared_weight_bytes = 0ull, .gguf = "UD-TQ1_0/Kimi-K3-UD-TQ1_0-00001-of-00012.gguf" },  /* 12 parts, 474 GiB */
    { .quant = "TQ2_0", .layer_weight_bytes = 551455107488ull, .shared_weight_bytes = 0ull, .gguf = "UD-TQ2_0/Kimi-K3-UD-TQ2_0-00001-of-00013.gguf" },  /* 13 parts, 514 GiB */
    { .quant = "IQ1_S", .layer_weight_bytes = 594040923616ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" },  /* 14 parts, 553 GiB */
    { .quant = "IQ1_M", .layer_weight_bytes = 648872012448ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf" },  /* 15 parts, 604 GiB */
    { .quant = "IQ2_XXS", .layer_weight_bytes = 711067773664ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf" },  /* 16 parts, 662 GiB */
    { .quant = "Q2_K_XL", .layer_weight_bytes = 861277858912ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf" },  /* 19 parts, 802 GiB */
    { .quant = "Q4_K_XL", .layer_weight_bytes = 1508668683104ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/Kimi-K3-UD-Q4_K_XL-00001-of-00032.gguf" },  /* 32 parts, 1405 GiB */
    { .quant = "Q8_K_XL", .layer_weight_bytes = 1561157884384ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/Kimi-K3-UD-Q8_K_XL-00001-of-00034.gguf" },  /* 34 parts, 1454 GiB */
};

/* DeepSeek V4 Pro: measured from unsloth/DeepSeek-V4-Pro-0813-GGUF
 * (HF API, 2026-08-15). 671B/37B-active, 61 layers — a different model
 * from Flash (43 layers), not one of its precisions. */
static const idletoken_model_variant DSV4_PRO_VARIANTS[] = {
    { .quant = "Q4_K_XL", .layer_weight_bytes = 849683927055ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf" },  /* 20 parts, 791 GiB */
    { .quant = "Q8_K_XL", .layer_weight_bytes = 873445601295ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/DeepSeek-V4-Pro-0813-UD-Q8_K_XL-00001-of-00020.gguf" },  /* 20 parts, 813 GiB */
};

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
        .layer_weight_bytes  = 82539237024ull, /* == IQ2_XXS+Q2_K variant (measured) */
        .shared_weight_bytes = 0ull,  /* token_embd + output + norms */
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_DSV4,          /* per-tier table in overhead() */
        /* The filename antirez actually publishes. "ds4flash.gguf" was a
         * local nickname: every gate passed --model-path explicitly, so the
         * drift stayed invisible until something (the topology matrix, the
         * weight fetcher) tried to RESOLVE the default name. The -0731 suffix
         * is the official release; the unsuffixed files are the superseded
         * preview and must not be resolved by default any more. */
        .default_gguf = "UD-IQ1_S/DeepSeek-V4-Flash-UD-IQ1_S-00001-of-00003.gguf",
        .variants = DSV4_FLASH_VARIANTS,
        .n_variants = sizeof(DSV4_FLASH_VARIANTS) / sizeof(DSV4_FLASH_VARIANTS[0]),
        .default_variant = 0,      /* IQ2_XXS+Q2_K */
    },
    {
        /* DeepSeek V4 Pro (2026-08-13 release). Enabled for users with the
         * memory to hold it; ~791 GiB at the smallest published quant, which
         * is far past this project's testbed. */
        .id      = "deepseek-v4-pro",
        .label   = "DeepSeek V4 Pro",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 61,
        .n_embd   = 7168,
        .hc_streams = 1,
        .n_vocab  = 129280,
        .layer_weight_bytes  = 849683927055ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 0,
        .overhead_base_bytes = 3221225472ull,
        .default_gguf = "UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf",
        .variants = DSV4_PRO_VARIANTS,
        .n_variants = sizeof(DSV4_PRO_VARIANTS) / sizeof(DSV4_PRO_VARIANTS[0]),
        .default_variant = 0,
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
        .layer_weight_bytes  = 338227456ull,   /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
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
        .default_gguf = "Qwen3.5-0.8B-UD-IQ2_XXS.gguf",
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
        .layer_weight_bytes  = 1520217248ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
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
        .default_gguf = "Qwen3.5-4B-UD-IQ2_XXS.gguf",
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
        .layer_weight_bytes  = 3190613216ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 8192,
        .state_bytes_per_layer = 2195456,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .default_gguf = "Qwen3.5-9B-UD-IQ2_XXS.gguf",
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
        .layer_weight_bytes  = 8573593504ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 8192,   /* 4 kv heads × 256 × 2 × 4 B */
        .state_bytes_per_layer = 3268608,   /* 48×128×128×4 + conv 3×10240×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(2.0 * (double)GiB),
        .default_gguf = "Qwen3.5-27B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_27B_VARIANTS,
        .n_variants = sizeof(QWEN35_27B_VARIANTS) / sizeof(QWEN35_27B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* Qwen3.8-27B. The GGUF declares general.architecture = 'qwen35', so
         * this is a Qwen3.5-27B-SHAPED model, not a new architecture -- every
         * geometry field below was read from this model's own GGUF header and
         * is identical to qwen3.5-27b above. It rides the llamacpp backend,
         * not ds4x, which is the frozen line.
         *
         * n_layers is 64, NOT qwen35.block_count (65): block 64 is the MTP
         * (NextN) draft head. The engine logs it as an unused tensor and
         * ignores it without --spec-type draft-mtp, and counting it as a
         * decode layer would mis-size every node in the cluster. */
        .id      = "qwen3.8-27b",
        .label   = "Qwen3.8 27B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* onboarded 2026-08-20 (T14 phase B) on the
                                    * b10502 pin: ppl gate on the real weights
                                    * (DGX, CUDA) came back PPL 1.9780 +/-
                                    * 0.05305, identical across 3 runs, and the
                                    * testbed smoke served it through coord.
                                    * Evidence: results/t14-engine-bump-phaseb-20260820.md */
        /* single-node, not cluster: the draft manifest guessed cluster because
         * the other llamacpp-backed models are clustered, but those are
         * clustered for being 200-850 GB. This one's default is 6.2 GB. Rule
         * #8 -- if it fits, do not go over the network. */
        .deployment = IDLETOKEN_DEPLOY_SINGLE_NODE,
        .n_layers = 64,
        .n_embd   = 5120,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 6192222208ull,  /* == variants[default_variant] = IQ1_S */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 8192,   /* 4 kv heads x 256 x 2 x 4 B */
        .state_bytes_per_layer = 3268608,   /* 48x128x128x4 + conv 3x10240x4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(2.0 * (double)GiB),
        .default_gguf = "Qwen3.8-27B-UD-IQ1_S.gguf",
        .variants = QWEN38_27B_VARIANTS,
        .n_variants = sizeof(QWEN38_27B_VARIANTS) / sizeof(QWEN38_27B_VARIANTS[0]),
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
        .layer_weight_bytes  = 10656955008ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 4096,   /* 2 kv heads × 256 × 2 × 4 B */
        .state_bytes_per_layer = 2195456,   /* 32×128×128×4 + conv 3×8192×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .default_gguf = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf",
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
        .layer_weight_bytes  = 5027783488ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 0ull,
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
        /* MEASURED from unsloth/GLM-5.2-GGUF UD-IQ1_S (2026-08-15, HF API):
         * 6 parts, 201.8 GiB. Architecture from zai-org/GLM-5.2 config.json.
         * MLA KV: (kv_lora_rank 512 + rope 64) × 2 bytes ≈ 1152 B/token/layer.
         * Enabled for users whose hardware can hold it; the project's own
         * testbed cannot (~277 GiB pooled, and that is the whole cluster). */
        .id      = "glm-5.2",
        .label   = "GLM-5.2",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,  /* ~240 GiB at Q2 */
        .n_layers = 78,            /* 3 dense + 75 MoE */
        .n_embd   = 6144,
        .hc_streams = 1,           /* plain residual */
        .n_vocab  = 154880,
        .layer_weight_bytes  = 216715360960ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 4,  /* DSA indexer shared per 4-layer group */
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,  /* activations at n_embd 6144 + CUDA */
        .default_gguf = "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf",
        .variants = GLM52_VARIANTS,
        .n_variants = sizeof(GLM52_VARIANTS) / sizeof(GLM52_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* MEASURED from bartowski/moonshotai_Kimi-K2.5-GGUF IQ1_M
         * (2026-08-15, HF API): 6 parts, 204.5 GiB. 1T total / 32B active,
         * DeepSeek2-shaped. Enabled for users with the hardware. */
        .id      = "kimi-k2.5",
        .label   = "Kimi K2.5",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 61,
        .n_embd   = 7168,
        .hc_streams = 1,
        .n_vocab  = 163840,
        .layer_weight_bytes  = 210299214240ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,
        .default_gguf = "moonshotai_Kimi-K2.5-IQ1_S/moonshotai_Kimi-K2.5-IQ1_S-00001-of-00006.gguf",
        .variants = KIMI_K25_VARIANTS,
        .n_variants = sizeof(KIMI_K25_VARIANTS) / sizeof(KIMI_K25_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* MEASURED from unsloth/Kimi-K3-GGUF UD-Q1_0 (2026-08-15, HF API):
         * 11 parts, 434.3 GiB — the smallest published quant. Architecture
         * from moonshotai/Kimi-K3 config.json (2.8T total, 93 layers).
         * Enabled for users with server-class memory; far beyond this
         * project's testbed. */
        .id      = "kimi-k3",
        .label   = "Kimi K3",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 93,
        .n_embd   = 7168,
        .hc_streams = 1,
        .n_vocab  = 163840,
        .layer_weight_bytes  = 466369456320ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,
        .default_gguf = "UD-Q1_0/Kimi-K3-UD-Q1_0-00001-of-00011.gguf",
        .variants = KIMI_K3_VARIANTS,
        .n_variants = sizeof(KIMI_K3_VARIANTS) / sizeof(KIMI_K3_VARIANTS[0]),
        .default_variant = 0,
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
