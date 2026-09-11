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
 * A quant appears here only if the repository really publishes it. Qwen3.5 is
 * served by the pinned llama.cpp backend; the retired ds4x quant allow-list no
 * longer limits the public precision menu. */
static const idletoken_model_variant QWEN35_2B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 768270592ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 859857152ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-IQ2_M.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 931823872ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-IQ3_XXS.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 966533376ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q2_K_XL.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 1030947072ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q3_K_S.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 1107149056ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q3_K_M.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 1159274752ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q3_K_XL.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 1172996352ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-IQ4_XS.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 1213300992ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-IQ4_NL.gguf" },
    { .quant = "Q4_0", .layer_weight_bytes = 1214873856ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q4_0.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 1217757440ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q4_K_S.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 1280835840ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q4_K_M.gguf" },
    { .quant = "Q4_1", .layer_weight_bytes = 1293517056ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q4_1.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 1339752704ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q4_K_XL.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 1384546560ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q5_K_S.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 1435238656ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q5_K_M.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 1466687744ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q5_K_XL.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 1574961408ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q6_K.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 1864483072ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q6_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 2012012800ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-Q8_0.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 2834940160ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-UD-Q8_K_XL.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 3775709216ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-2B-BF16.gguf" },
};

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

static const idletoken_model_variant QWEN35_122B_A10B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 36637668544ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 39148942528ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-122B-A10B-UD-IQ2_M.gguf" },
    { .quant = "Q2_K_XL", .layer_weight_bytes = 41845347520ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 44745020608ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf" },
    { .quant = "IQ3_S", .layer_weight_bytes = 46556959936ull, .shared_weight_bytes = 0ull, .gguf = "Qwen3.5-122B-A10B-UD-IQ3_S.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 52495013376ull, .shared_weight_bytes = 0ull, .gguf = "Q3_K_S/Qwen3.5-122B-A10B-Q3_K_S-00001-of-00003.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 56420881952ull, .shared_weight_bytes = 0ull, .gguf = "Q3_K_M/Qwen3.5-122B-A10B-Q3_K_M-00001-of-00003.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 56970536480ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 60229510656ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 61437470208ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_NL/Qwen3.5-122B-A10B-UD-IQ4_NL-00001-of-00003.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 71705126400ull, .shared_weight_bytes = 0ull, .gguf = "Q4_K_S/Qwen3.5-122B-A10B-Q4_K_S-00001-of-00003.gguf" },
    { .quant = "MXFP4_MOE", .layer_weight_bytes = 74664408608ull, .shared_weight_bytes = 0ull, .gguf = "MXFP4_MOE/Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 76536964608ull, .shared_weight_bytes = 0ull, .gguf = "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 77029996032ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 86385391136ull, .shared_weight_bytes = 0ull, .gguf = "Q5_K_S/Qwen3.5-122B-A10B-Q5_K_S-00001-of-00003.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 91519219200ull, .shared_weight_bytes = 0ull, .gguf = "Q5_K_M/Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 91928163840ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q5_K_XL/Qwen3.5-122B-A10B-UD-Q5_K_XL-00001-of-00003.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 101009782432ull, .shared_weight_bytes = 0ull, .gguf = "Q6_K/Qwen3.5-122B-A10B-Q6_K-00001-of-00004.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 112401249920ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q6_K_XL/Qwen3.5-122B-A10B-UD-Q6_K_XL-00001-of-00004.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 129871935104ull, .shared_weight_bytes = 0ull, .gguf = "Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 170838161152ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/Qwen3.5-122B-A10B-UD-Q8_K_XL-00001-of-00005.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 244314011488ull, .shared_weight_bytes = 0ull, .gguf = "BF16/Qwen3.5-122B-A10B-BF16-00001-of-00005.gguf" },
};

static const idletoken_model_variant QWEN35_397B_A17B_VARIANTS[] = {
    { .quant = "IQ2_XXS", .layer_weight_bytes = 114872940736ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" },
    { .quant = "IQ2_M", .layer_weight_bytes = 123053144288ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf" },
    { .quant = "IQ3_XXS", .layer_weight_bytes = 140333676736ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_XXS/Qwen3.5-397B-A17B-UD-IQ3_XXS-00001-of-00004.gguf" },
    { .quant = "IQ3_S", .layer_weight_bytes = 146373474496ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ3_S/Qwen3.5-397B-A17B-UD-IQ3_S-00001-of-00004.gguf" },
    { .quant = "Q3_K_S", .layer_weight_bytes = 164323293504ull, .shared_weight_bytes = 0ull, .gguf = "Q3_K_S/Qwen3.5-397B-A17B-Q3_K_S-00001-of-00005.gguf" },
    { .quant = "Q3_K_M", .layer_weight_bytes = 177409522016ull, .shared_weight_bytes = 0ull, .gguf = "Q3_K_M/Qwen3.5-397B-A17B-Q3_K_M-00001-of-00005.gguf" },
    { .quant = "Q3_K_XL", .layer_weight_bytes = 178739034432ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf" },
    { .quant = "IQ4_XS", .layer_weight_bytes = 189735450944ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_XS/Qwen3.5-397B-A17B-UD-IQ4_XS-00001-of-00005.gguf" },
    { .quant = "IQ4_NL", .layer_weight_bytes = 193761982784ull, .shared_weight_bytes = 0ull, .gguf = "UD-IQ4_NL/Qwen3.5-397B-A17B-UD-IQ4_NL-00001-of-00005.gguf" },
    { .quant = "Q4_K_S", .layer_weight_bytes = 227987503520ull, .shared_weight_bytes = 0ull, .gguf = "Q4_K_S/Qwen3.5-397B-A17B-Q4_K_S-00001-of-00006.gguf" },
    { .quant = "MXFP4_MOE", .layer_weight_bytes = 237353302496ull, .shared_weight_bytes = 0ull, .gguf = "MXFP4_MOE/Qwen3.5-397B-A17B-MXFP4_MOE-00001-of-00006.gguf" },
    { .quant = "Q4_K_M", .layer_weight_bytes = 244093630912ull, .shared_weight_bytes = 0ull, .gguf = "Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 245272148448ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf" },
    { .quant = "Q5_K_S", .layer_weight_bytes = 276552219168ull, .shared_weight_bytes = 0ull, .gguf = "Q5_K_S/Qwen3.5-397B-A17B-Q5_K_S-00001-of-00007.gguf" },
    { .quant = "Q5_K_M", .layer_weight_bytes = 293664979584ull, .shared_weight_bytes = 0ull, .gguf = "Q5_K_M/Qwen3.5-397B-A17B-Q5_K_M-00001-of-00008.gguf" },
    { .quant = "Q5_K_XL", .layer_weight_bytes = 294865599104ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q5_K_XL/Qwen3.5-397B-A17B-UD-Q5_K_XL-00001-of-00008.gguf" },
    { .quant = "Q6_K", .layer_weight_bytes = 326595345056ull, .shared_weight_bytes = 0ull, .gguf = "Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf" },
    { .quant = "Q6_K_XL", .layer_weight_bytes = 362328980288ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00001-of-00009.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 421507365824ull, .shared_weight_bytes = 0ull, .gguf = "Q8_0/Qwen3.5-397B-A17B-Q8_0-00001-of-00010.gguf" },
    { .quant = "Q8_K_XL", .layer_weight_bytes = 427726568384ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/Qwen3.5-397B-A17B-UD-Q8_K_XL-00001-of-00010.gguf" },
    { .quant = "BF16", .layer_weight_bytes = 792961318816ull, .shared_weight_bytes = 0ull, .gguf = "BF16/Qwen3.5-397B-A17B-BF16-00001-of-00017.gguf" },
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

/* DeepSeek V4 Pro: measured from unsloth/DeepSeek-V4-Pro-0813-GGUF
 * (HF API, 2026-08-15). 671B/37B-active, 61 layers — a different model
 * from Flash (43 layers), not one of its precisions. */
/* Two repos, smallest first (2026-08-21). unsloth publishes exactly two
 * precisions of V4 Pro and neither is below Q4, which left the largest model in
 * the catalogue with the least room to make it fit; DevQuasar's plain-K quants
 * add a 530 GiB floor under it.
 * ⚠ These are NOT one quality ladder. unsloth's UD-* files are its dynamic
 * quantization (selected tensors kept wide) and DevQuasar's are ordinary
 * llama-quantize output, so "Q4_K_M is bigger than Q4_K_XL" here is a fact
 * about two different quantizers, not a typo — and the ppl gate has to be run
 * per repo. The apparent 22 GiB gap between UD-Q4_K_XL and UD-Q8_K_XL is
 * likewise unsloth's own doing; both totals were re-measured against the HF
 * blob sizes on 2026-08-21 and match to the byte. */
static const idletoken_model_variant DSV4_PRO_VARIANTS[] = {
    { .quant = "Q2_K",    .layer_weight_bytes = 569417385408ull, .shared_weight_bytes = 0ull, .gguf = "Q2_K/deepseek-ai.DeepSeek-V4-Pro-0813.Q2_K-00001-of-00037.gguf" },      /* 37 parts, 530 GiB, DevQuasar */
    { .quant = "Q3_K_M",  .layer_weight_bytes = 748403475872ull, .shared_weight_bytes = 0ull, .gguf = "Q3_K_M/deepseek-ai.DeepSeek-V4-Pro-0813.Q3_K_M-00001-of-00061.gguf" },  /* 61 parts, 697 GiB, DevQuasar */
    { .quant = "Q4_K_XL", .layer_weight_bytes = 849683927055ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf" },      /* 20 parts, 791 GiB, unsloth */
    { .quant = "Q8_K_XL", .layer_weight_bytes = 873445601295ull, .shared_weight_bytes = 0ull, .gguf = "UD-Q8_K_XL/DeepSeek-V4-Pro-0813-UD-Q8_K_XL-00001-of-00020.gguf" },      /* 20 parts, 813 GiB, unsloth */
    { .quant = "Q4_K_M",  .layer_weight_bytes = 950879580768ull, .shared_weight_bytes = 0ull, .gguf = "Q4_K_M/deepseek-ai.DeepSeek-V4-Pro-0813.Q4_K_M-00001-of-00076.gguf" },  /* 76 parts, 886 GiB, DevQuasar */
};

/* Curated Unsloth gpt-oss quantizations, smallest first. Tensor bytes were
 * summed from every GGUF shard at fixed HF revisions; graph workspaces were
 * measured with the pinned llama-fit-params on CUDA and Metal at the model's
 * 128K ceiling. Unsloth keeps the expert tensors MXFP4-derived, so the storage
 * gap between these rows is intentionally much smaller than a dense model's. */
static const idletoken_model_variant GPT_OSS_20B_VARIANTS[] = {
    { .quant = "Q3_K_S", .layer_weight_bytes = 10509782016ull,
      .shared_weight_bytes = 941103360ull,
      .gguf = "gpt-oss-20b-Q3_K_S.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 10628668416ull,
      .shared_weight_bytes = 1230670080ull,
      .gguf = "gpt-oss-20b-UD-Q4_K_XL.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 10865888256ull,
      .shared_weight_bytes = 1230670080ull,
      .gguf = "gpt-oss-20b-Q8_0.gguf" },
    { .quant = "F16", .layer_weight_bytes = 11463085056ull,
      .shared_weight_bytes = 2316545280ull,
      .gguf = "gpt-oss-20b-F16.gguf" },
};

static const idletoken_model_variant GPT_OSS_120B_VARIANTS[] = {
    { .quant = "Q3_K_S", .layer_weight_bytes = 61609494528ull,
      .shared_weight_bytes = 941103360ull,
      .gguf = "Q3_K_S/gpt-oss-120b-Q3_K_S-00001-of-00002.gguf" },
    { .quant = "Q4_K_XL", .layer_weight_bytes = 61772617728ull,
      .shared_weight_bytes = 1230670080ull,
      .gguf = "UD-Q4_K_XL/gpt-oss-120b-UD-Q4_K_XL-00001-of-00002.gguf" },
    { .quant = "Q8_0", .layer_weight_bytes = 62143653888ull,
      .shared_weight_bytes = 1230670080ull,
      .gguf = "Q8_0/gpt-oss-120b-Q8_0-00001-of-00002.gguf" },
    { .quant = "F16", .layer_weight_bytes = 63039449088ull,
      .shared_weight_bytes = 2316545280ull,
      .gguf = "gpt-oss-120b-F16.gguf" },
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
        .n_expert = 256,
        .n_expert_used = 6,
        .layer_weight_bytes  = 82539237024ull, /* == IQ2_XXS+Q2_K variant (measured) */
        .shared_weight_bytes = 0ull,  /* token_embd + output + norms */
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_DSV4,          /* per-tier table in overhead() */
        .kv_bytes_per_token_layer = 0, /* exact padded DSV4 fields below */
        /* GGUF compress_ratios: 21 CSA + 20 HCA over 43 raw-K layers.
         * Whole-model f16 bytes per cache cell, including the CSA lightning
         * indexer; fixed is the pinned engine's f32 compressor state. */
        .dsv4_raw_bytes_per_cell = 44032,
        .dsv4_csa_bytes_per_cell = 26880,
        .dsv4_hca_bytes_per_cell = 20480,
        .dsv4_fixed_bytes_per_seq = 12206080,
        /* The filename antirez actually publishes. "ds4flash.gguf" was a
         * local nickname: every gate passed --model-path explicitly, so the
         * drift stayed invisible until something (the topology matrix, the
         * weight fetcher) tried to RESOLVE the default name. The -0731 suffix
         * is the official release; the unsuffixed files are the superseded
         * preview and must not be resolved by default any more. */
        .compute_bytes_128k_cuda = { 391999652ull, 453865636ull, 428699812ull },
        .compute_bytes_256k_cuda = { 585986212ull, 698183844ull, 656240804ull },
        .compute_bytes_1m_cuda = { 1800237220ull, 1744295690ull, 1676987597ull },
        .compute_bytes_128k_metal = { 568128963ull, 569114624ull, 521289073ull },
        .compute_bytes_256k_metal = { 804090020ull, 772989256ull, 736886784ull },
        .compute_bytes_1m_metal = { 2688643236ull, 2699170939ull, 2564565238ull },
        .default_gguf = "UD-IQ1_S/DeepSeek-V4-Flash-UD-IQ1_S-00001-of-00003.gguf",
        .variants = DSV4_FLASH_VARIANTS,
        .n_variants = sizeof(DSV4_FLASH_VARIANTS) / sizeof(DSV4_FLASH_VARIANTS[0]),
        .default_variant = 0,      /* IQ2_XXS+Q2_K */
    },
    {
        /* DeepSeek V4 Pro (2026-08-13 release). 671B/37B-active, 61 layers —
         * a different model from Flash, not one of its precisions. Cluster-tier:
         * the smallest published quant is Q2_K at 530 GiB (791 GiB is the
         * smallest unsloth dynamic quant, not the catalogue minimum).
         * Architecture deepseek4 is read by the pinned llama.cpp; the numbers
         * below are measured from the GGUF headers (HF API). */
        .id      = "deepseek-v4-pro",
        .label   = "DeepSeek V4 Pro",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 61,
        .n_embd   = 7168,
        .hc_streams = 4,
        .n_vocab  = 129280,
        .n_expert = 384,
        .n_expert_used = 6,
        .layer_weight_bytes  = 569417385408ull, /* == variants[default_variant] = Q2_K */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_DSV4,
        .kv_bytes_per_token_layer = 0,
        /* GGUF compress_ratios: 30 CSA + 31 HCA over 61 raw-K layers. */
        .dsv4_raw_bytes_per_cell = 62464,
        .dsv4_csa_bytes_per_cell = 38400,
        .dsv4_hca_bytes_per_cell = 31744,
        .dsv4_fixed_bytes_per_seq = 18710528,
        .overhead_base_bytes = 3221225472ull,
        .compute_bytes_128k_cuda = { 617181348ull, 713650340ull, 713650340ull },
        .compute_bytes_256k_cuda = { 719679652ull, 871723172ull, 871723172ull },
        .compute_bytes_1m_cuda = { 1948610724ull, 1959778058ull, 1816752292ull },
        .compute_bytes_128k_metal = { 995161539ull, 911359345ull, 911327887ull },
        .compute_bytes_256k_metal = { 1131245732ull, 1113535283ull, 1054783570ull },
        .compute_bytes_1m_metal = { 2948165796ull, 2888260649ull, 2673050911ull },
        .default_gguf = "Q2_K/deepseek-ai.DeepSeek-V4-Pro-0813.Q2_K-00001-of-00037.gguf",
        .variants = DSV4_PRO_VARIANTS,
        .n_variants = sizeof(DSV4_PRO_VARIANTS) / sizeof(DSV4_PRO_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* First small dense model (small-model-design.md). GQA, not MLA.
         * The pinned llama.cpp backend serves it. Scalars mirror the default
         * variant (Q4_K_M) so quant-unaware callers see the default size.
         * KV: 2·n_head_kv(8)·head_dim(128)·2 (fp16) = 4096 B/token/layer. */
        .id      = "qwen3.5-0.8b",
        .label   = "Qwen3.5 0.8B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-0.8B Q4_K_M GGUF (DGX): output
                                    * matches llama.cpp token-for-token on
                                    * greedy prompts (design doc §4d) */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 24,
        .n_embd   = 1024,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 338227456ull,   /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        /* full layers: 2 KV heads × (256 K + 256 V) × 2 B/f16 */
        .kv_bytes_per_token_layer = 2048,
        /* linear layers: 16 heads × 128 × 128 state + conv window (3×6144),
         * fp32 → 1 MiB + 72 KiB, ctx-INDEPENDENT */
        .state_bytes_per_layer = 1122304,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(0.8 * (double)GiB),
        /* MEASURED 2026-09-01 with llama.cpp's no_alloc dry-run on the pinned
         * engine (scripts/measure_model_memory.sh, -ngl 99 -np 1).
         * 256K: 512753664 B (489.00 MiB) — byte-identical on Mac/Metal,
         *       CUDA-unified and CUDA-discrete alike, and identical across
         *       Q4_K_M..BF16.
         * 1M:   Metal 1115915551 B vs CUDA 1115852636 B; the larger is taken,
         *       because a manifest that understates is the failure mode this
         *       whole change exists to remove.
         * results/memory-need-measured-20260901.md */
        .compute_bytes_128k_cuda = { 512753664ull, 512753664ull, 512753664ull },
        .compute_bytes_256k_cuda = { 512753664ull, 830765793ull, 830765793ull },
        .compute_bytes_1m_cuda = { 1115852636ull, 3246684897ull, 3246684897ull },
        .compute_bytes_128k_metal = { 512753664ull, 647264993ull, 647264993ull },
        .compute_bytes_256k_metal = { 512753664ull, 781482721ull, 781482721ull },
        .compute_bytes_1m_metal = { 1115915551ull, 1116198666ull, 1116198666ull },
        .default_gguf = "Qwen3.5-0.8B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_08B_VARIANTS,
        .n_variants = sizeof(QWEN35_08B_VARIANTS) / sizeof(QWEN35_08B_VARIANTS[0]),
        .default_variant = 0,      /* IQ2_XXS */
    },
    {
        /* Official Qwen3.5 dense 2B. Geometry comes from the official config
         * and the published IQ2_XXS GGUF header; sub-2-bit files are excluded
         * from the curated precision menu. */
        .id      = "qwen3.5-2b",
        .label   = "Qwen3.5 2B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* metadata-onboarded 2026-09-05; qwen35 is
                                    * already supported by the pinned engine */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 24,
        .n_embd   = 2048,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 768270592ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,
        .state_bytes_per_layer = 1122304,
        .full_attn_interval = 4,
        .overhead_base_bytes = 1ull * GiB,
        .compute_bytes_128k_cuda = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_256k_cuda = { 516947968ull, 834960097ull, 834960097ull },
        .compute_bytes_1m_cuda = { 1120046940ull, 3250879201ull, 3250879201ull },
        .compute_bytes_128k_metal = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_256k_metal = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_1m_metal = { 1124167844ull, 1124440474ull, 1124440474ull },
        .default_gguf = "Qwen3.5-2B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_2B_VARIANTS,
        .n_variants = sizeof(QWEN35_2B_VARIANTS) / sizeof(QWEN35_2B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* Same qwen35 hybrid architecture as the 0.8B, but the linear path
         * SHARES k-heads: k_heads(16) < v_heads(32), so each key head serves
         * two value heads. The 0.8B is 16/16 and can never exercise that
         * branch — which is exactly why this entry is worth having. */
        .id      = "qwen3.5-4b",
        .label   = "Qwen3.5 4B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-4B Q4_K_M GGUF (DGX): CPU and
                                    * CUDA agree token-for-token, and the
                                    * cpu/gpu recurrence check stays at 6e-08
                                    * over 168 chunks (design doc §4j) */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 32,
        .n_embd   = 2560,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 1520217248ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        /* full layers: 4 KV heads × (256 K + 256 V) × 2 B/f16 */
        .kv_bytes_per_token_layer = 4096,
        /* linear layers: 32 heads × 128 × 128 state + conv window (3×8192),
         * fp32 → 2 MiB + 96 KiB, ctx-INDEPENDENT */
        .state_bytes_per_layer = 2195456,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.2 * (double)GiB),
        .compute_bytes_128k_cuda = { 519045120ull, 723811041ull, 723811041ull },
        .compute_bytes_256k_cuda = { 519045120ull, 1394899681ull, 1394899681ull },
        .compute_bytes_1m_cuda = { 1140871660ull, 5421431521ull, 5421431521ull },
        .compute_bytes_128k_metal = { 519045120ull, 653556449ull, 653556449ull },
        .compute_bytes_256k_metal = { 519045120ull, 787774177ull, 787774177ull },
        .compute_bytes_1m_metal = { 1157743247ull, 1149239296ull, 1148977152ull },
        .default_gguf = "Qwen3.5-4B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_4B_VARIANTS,
        .n_variants = sizeof(QWEN35_4B_VARIANTS) / sizeof(QWEN35_4B_VARIANTS[0]),
        .default_variant = 0,      /* IQ2_XXS */
    },
    {
        /* Same shape as the 4B in every field that the forward branches on
         * (32 layers, k_heads 16 / v_heads 32, full layers 16/4 heads × 256,
         * interval 4) — only n_embd and the FFN width differ. That is why it
         * needed no code at all once the 4B's k-head sharing was fixed. */
        .id      = "qwen3.5-9b",
        .label   = "Qwen3.5 9B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-9B Q4_K_M GGUF (DGX): greedy
                                    * output matches llama.cpp word for word */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 32,
        .n_embd   = 4096,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 3190613216ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 4096,
        .state_bytes_per_layer = 2195456,
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .compute_bytes_128k_cuda = { 525336576ull, 730102497ull, 730102497ull },
        .compute_bytes_256k_cuda = { 525336576ull, 1401191137ull, 1401191137ull },
        .compute_bytes_1m_cuda = { 1166037484ull, 5427722977ull, 5427722977ull },
        .compute_bytes_128k_metal = { 525336576ull, 525336576ull, 525336576ull },
        .compute_bytes_256k_metal = { 525336576ull, 525336576ull, 525336576ull },
        .compute_bytes_1m_metal = { 1174520463ull, 1174803579ull, 1174803579ull },
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
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-27B Q4_K_M GGUF (DGX): the
                                    * counting prompt matches llama.cpp word
                                    * for word; layer-0 attn_output/final_output
                                    * agree to <0.1% (design doc §4j) */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 64,
        .n_embd   = 5120,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 8573593504ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 4096,   /* 4 × (256 K + 256 V) × 2 B/f16 */
        .state_bytes_per_layer = 3268608,   /* 48×128×128×4 + conv 3×10240×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(2.0 * (double)GiB),
        .compute_bytes_128k_cuda = { 646971392ull, 755268321ull, 755268321ull },
        .compute_bytes_256k_cuda = { 529530880ull, 1426356961ull, 1426356961ull },
        .compute_bytes_1m_cuda = { 1201689068ull, 5452888801ull, 5452888801ull },
        .compute_bytes_128k_metal = { 663769580ull, 664042209ull, 664042209ull },
        .compute_bytes_256k_metal = { 797987308ull, 798259937ull, 798259937ull },
        .compute_bytes_1m_metal = { 1224873083ull, 1212447457ull, 1212447457ull },
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
        /* Cluster-capable even though the default precision is only 6.2 GB:
         * higher precisions can need several machines, and deployment is the
         * user's choice after model + precision, not a model-wide verdict. */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 64,
        .n_embd   = 5120,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .layer_weight_bytes  = 6192222208ull,  /* == variants[default_variant] = IQ1_S */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 4096,   /* 4 x (256 K + 256 V) x 2 B/f16 */
        .state_bytes_per_layer = 3268608,   /* 48x128x128x4 + conv 3x10240x4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(2.0 * (double)GiB),
        .compute_bytes_128k_cuda = { 646971392ull, 755268321ull, 755268321ull },
        .compute_bytes_256k_cuda = { 529530880ull, 1426356961ull, 1426356961ull },
        .compute_bytes_1m_cuda = { 1201689068ull, 5452888801ull, 5452888801ull },
        .compute_bytes_128k_metal = { 663769580ull, 664042209ull, 664042209ull },
        .compute_bytes_256k_metal = { 797987308ull, 798259937ull, 798259937ull },
        .compute_bytes_1m_metal = { 1224873083ull, 1212447457ull, 1212447457ull },
        .default_gguf = "Qwen3.8-27B-UD-IQ1_S.gguf",
        .variants = QWEN38_27B_VARIANTS,
        .n_variants = sizeof(QWEN38_27B_VARIANTS) / sizeof(QWEN38_27B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* qwen35moe: hybrid linear attention and a 256-expert MoE. The shared
         * expert is scaled by a scalar sigmoid gate. */
        .id      = "qwen3.5-35b-a3b",
        .label   = "Qwen3.5 35B-A3B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* validated 2026-07-28 on the real
                                    * Qwen3.5-35B-A3B Q4_K_M GGUF (DGX): the
                                    * counting prompt matches llama.cpp word
                                    * for word and layer-0 agrees (§4n) */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 40,
        .n_embd   = 2048,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .n_expert = 256,
        .n_expert_used = 8,
        .layer_weight_bytes  = 10656955008ull,  /* == variants[default_variant] = IQ2_XXS */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,   /* 2 × (256 K + 256 V) × 2 B/f16 */
        .state_bytes_per_layer = 2195456,   /* 32×128×128×4 + conv 3×8192×4 */
        .full_attn_interval = 4,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .compute_bytes_128k_cuda = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_256k_cuda = { 516947968ull, 860125921ull, 860125921ull },
        .compute_bytes_1m_cuda = { 1145065964ull, 3276045025ull, 3276045025ull },
        .compute_bytes_128k_metal = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_256k_metal = { 516947968ull, 516947968ull, 516947968ull },
        .compute_bytes_1m_metal = { 1146135511ull, 1146408141ull, 1146408141ull },
        .default_gguf = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_35B_A3B_VARIANTS,
        .n_variants = sizeof(QWEN35_35B_A3B_VARIANTS) / sizeof(QWEN35_35B_A3B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* Official Qwen3.5 122B-A10B MoE. Geometry comes from the official
         * config and published IQ2_XXS GGUF header. */
        .id      = "qwen3.5-122b-a10b",
        .label   = "Qwen3.5 122B-A10B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* metadata-onboarded 2026-09-05; qwen35moe
                                    * is already supported by the pinned engine */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 48,
        .n_embd   = 3072,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .n_expert = 256,
        .n_expert_used = 8,
        .layer_weight_bytes  = 36637668544ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,
        .state_bytes_per_layer = 4341760,
        .full_attn_interval = 4,
        .overhead_base_bytes = 3ull * GiB,
        .compute_bytes_128k_cuda = { 629145600ull, 644874240ull, 644874240ull },
        .compute_bytes_256k_cuda = { 521142272ull, 908360417ull, 908360417ull },
        .compute_bytes_1m_cuda = { 1193300460ull, 3324279521ull, 3324279521ull },
        .compute_bytes_128k_metal = { 521142272ull, 521142272ull, 521142272ull },
        .compute_bytes_256k_metal = { 521142272ull, 521142272ull, 521142272ull },
        .compute_bytes_1m_metal = { 1194370007ull, 1207676436ull, 1207613522ull },
        .default_gguf = "Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf",
        .variants = QWEN35_122B_A10B_VARIANTS,
        .n_variants = sizeof(QWEN35_122B_A10B_VARIANTS) / sizeof(QWEN35_122B_A10B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* Official Qwen3.5 397B-A17B MoE. The default is a four-part GGUF;
         * all part hashes and the immutable repository revision live in the
         * client manifest. */
        .id      = "qwen3.5-397b-a17b",
        .label   = "Qwen3.5 397B-A17B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,            /* metadata-onboarded 2026-09-05; qwen35moe
                                    * is already supported by the pinned engine */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 60,
        .n_embd   = 4096,
        .hc_streams = 1,
        .n_vocab  = 248320,
        .n_expert = 512,
        .n_expert_used = 10,
        .layer_weight_bytes  = 114872940736ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,
        .state_bytes_per_layer = 4341760,
        .full_attn_interval = 4,
        .overhead_base_bytes = 3ull * GiB,
        .compute_bytes_128k_cuda = { 525336576ull, 525336576ull, 525336576ull },
        .compute_bytes_256k_cuda = { 525336576ull, 914651873ull, 914651873ull },
        .compute_bytes_1m_cuda = { 1245729260ull, 3330570977ull, 3330570977ull },
        .compute_bytes_128k_metal = { 659575276ull, 659847905ull, 659847905ull },
        .compute_bytes_256k_metal = { 793793004ull, 794065633ull, 794065633ull },
        .compute_bytes_1m_metal = { 1247847383ull, 1248130499ull, 1248130499ull },
        .default_gguf = "UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf",
        .variants = QWEN35_397B_A17B_VARIANTS,
        .n_variants = sizeof(QWEN35_397B_A17B_VARIANTS) / sizeof(QWEN35_397B_A17B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        .id      = "qwen3-8b",
        .label   = "Qwen3 8B",
        .backend = IDLETOKEN_BACKEND_DS4X,
        .available = 1,            /* validated 2026-07-27 on real Qwen3-8B
                                    * Q4_K_M GGUF (DGX): coherent output, and
                                    * the CUDA path matches CPU token-for-token */
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 36,
        .n_embd   = 4096,
        .hc_streams = 1,           /* plain residual */
        .n_vocab  = 151936,
        .layer_weight_bytes  = 5027783488ull,  /* == Q4_K_M variant (measured) */
        .shared_weight_bytes = 0ull,
        .ctx_max  = 40960,
        .ctx_yarn_max = 163840,   /* Qwen ships YaRN configs for 4x */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_GQA,
        .kv_bytes_per_token_layer = 4096,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB), /* n_embd 4096 + CUDA */
        .compute_bytes_128k_cuda = { 319553536ull, 704737444ull, 704737444ull },
        .compute_bytes_256k_cuda = { 319553536ull, 872509604ull, 872509604ull },
        .compute_bytes_1m_cuda = { 0ull, 0ull, 0ull },
        .compute_bytes_128k_metal = { 327942144ull, 319553536ull, 319553536ull },
        .compute_bytes_256k_metal = { 327942144ull, 319553536ull, 319553536ull },
        .compute_bytes_1m_metal = { 0ull, 0ull, 0ull },
        .default_gguf = "Qwen3-8B-Q4_K_M.gguf",
        .variants = QWEN3_8B_VARIANTS,
        .n_variants = sizeof(QWEN3_8B_VARIANTS) / sizeof(QWEN3_8B_VARIANTS[0]),
        .default_variant = 0,      /* Q4_K_M */
    },
    {
        /* MEASURED from unsloth/GLM-5.2-GGUF UD-IQ1_S (2026-08-15, HF API):
         * 6 parts, 201.8 GiB. Architecture GLM_DSA from zai-org/GLM-5.2
         * config.json, read by the pinned llama.cpp. MLA KV: (kv_lora_rank 512
         * + rope 64) × 2 bytes ≈ 1152 B/token/layer. Cluster-tier: the smallest
         * published quant is ~202 GiB, so it needs a multi-machine cluster. */
        .id      = "glm-5.2",
        .label   = "GLM-5.2",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,  /* ~240 GiB at Q2 */
        .n_layers = 78,            /* 3 dense + 75 MoE */
        .n_embd   = 6144,
        .hc_streams = 1,           /* plain residual */
        .n_vocab  = 154880,
        .n_expert = 256,
        .n_expert_used = 8,
        .layer_weight_bytes  = 216715360960ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 1048576,
        .split_boundary_multiple = 4,  /* DSA indexer shared per 4-layer group */
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,  /* activations at n_embd 6144 + CUDA */
        .compute_bytes_128k_cuda = { 1078009528ull, 1296113336ull, 1296113336ull },
        .compute_bytes_256k_cuda = { 1614880440ull, 1983979192ull, 1983979192ull },
        .compute_bytes_1m_cuda = { 4836105912ull, 6111174328ull, 6111174328ull },
        .compute_bytes_128k_metal = { 18325511864ull, 18325511864ull, 18325511864ull },
        .compute_bytes_256k_metal = { 35706707640ull, 35706707640ull, 35706707640ull },
        .compute_bytes_1m_metal = { 141201841848ull, 141201841848ull, 141201841848ull },
        .default_gguf = "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf",
        .variants = GLM52_VARIANTS,
        .n_variants = sizeof(GLM52_VARIANTS) / sizeof(GLM52_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        /* MEASURED from bartowski/moonshotai_Kimi-K2.5-GGUF IQ1_M
         * (2026-08-15, HF API): 6 parts, 204.5 GiB. 1T total / 32B active,
         * DeepSeek2-shaped, read by the pinned llama.cpp. Cluster-tier: the
         * smallest published quant is ~196 GiB, so it needs a cluster. */
        .id      = "kimi-k2.5",
        .label   = "Kimi K2.5",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 61,
        .n_embd   = 7168,
        .hc_streams = 1,
        .n_vocab  = 163840,
        .n_expert = 384,
        .n_expert_used = 8,
        .layer_weight_bytes  = 210299214240ull,
        .shared_weight_bytes = 0ull,
        .ctx_max  = 262144,
        .ctx_yarn_max = 1048576,   /* x4 — owner-approved extension */
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_MLA,
        .kv_bytes_per_token_layer = 1152,
        .overhead_base_bytes = 3ull * GiB,
        .compute_bytes_128k_cuda = { 398469366ull, 564144374ull, 564144374ull },
        .compute_bytes_256k_cuda = { 532687094ull, 849357046ull, 849357046ull },
        .compute_bytes_1m_cuda = { 1337993462ull, 2560633078ull, 2560633078ull },
        .compute_bytes_128k_metal = { 545091748ull, 545018348ull, 544976404ull },
        .compute_bytes_256k_metal = { 679435305ull, 679372390ull, 679330447ull },
        .compute_bytes_1m_metal = { 1485528105ull, 1485465190ull, 1485423247ull },
        .default_gguf = "moonshotai_Kimi-K2.5-IQ1_S/moonshotai_Kimi-K2.5-IQ1_S-00001-of-00006.gguf",
        .variants = KIMI_K25_VARIANTS,
        .n_variants = sizeof(KIMI_K25_VARIANTS) / sizeof(KIMI_K25_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        .id      = "gpt-oss-20b",
        .label   = "GPT-OSS 20B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 24,
        .n_embd   = 2880,
        .hc_streams = 1,
        .n_vocab  = 201088,
        .n_expert = 32,
        .n_expert_used = 4,
        .layer_weight_bytes  = 10509782016ull,
        .shared_weight_bytes = 941103360ull,
        .ctx_max  = 131072,
        .split_boundary_multiple = 0,
        /* Even blocks use a 128-token sliding window; odd blocks use full
         * attention. The bounded half is represented by its fixed KV bytes. */
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,
        .state_bytes_per_layer = 262144,
        .full_attn_interval = 2,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .compute_bytes_128k_cuda = { 423624704ull, 432097198ull, 432097198ull },
        .compute_bytes_256k_cuda = { 0ull, 0ull, 0ull },
        .compute_bytes_1m_cuda = { 0ull, 0ull, 0ull },
        .compute_bytes_128k_metal = { 423624704ull, 423624704ull, 423624704ull },
        .compute_bytes_256k_metal = { 0ull, 0ull, 0ull },
        .compute_bytes_1m_metal = { 0ull, 0ull, 0ull },
        .default_gguf = "gpt-oss-20b-Q3_K_S.gguf",
        .variants = GPT_OSS_20B_VARIANTS,
        .n_variants = sizeof(GPT_OSS_20B_VARIANTS) / sizeof(GPT_OSS_20B_VARIANTS[0]),
        .default_variant = 0,
    },
    {
        .id      = "gpt-oss-120b",
        .label   = "GPT-OSS 120B",
        .backend = IDLETOKEN_BACKEND_LLAMACPP,
        .available = 1,
        .deployment = IDLETOKEN_DEPLOY_CLUSTER,
        .n_layers = 36,
        .n_embd   = 2880,
        .hc_streams = 1,
        .n_vocab  = 201088,
        .n_expert = 128,
        .n_expert_used = 4,
        .layer_weight_bytes  = 61609494528ull,
        .shared_weight_bytes = 941103360ull,
        .ctx_max  = 131072,
        .split_boundary_multiple = 0,
        .kv_kind  = IDLETOKEN_KV_HYBRID,
        .kv_bytes_per_token_layer = 2048,
        .state_bytes_per_layer = 262144,
        .full_attn_interval = 2,
        .overhead_base_bytes = (uint64_t)(1.5 * (double)GiB),
        .compute_bytes_128k_cuda = { 423624704ull, 432097198ull, 432097198ull },
        .compute_bytes_256k_cuda = { 0ull, 0ull, 0ull },
        .compute_bytes_1m_cuda = { 0ull, 0ull, 0ull },
        .compute_bytes_128k_metal = { 423624704ull, 423624704ull, 423624704ull },
        .compute_bytes_256k_metal = { 0ull, 0ull, 0ull },
        .compute_bytes_1m_metal = { 0ull, 0ull, 0ull },
        .default_gguf = "Q3_K_S/gpt-oss-120b-Q3_K_S-00001-of-00002.gguf",
        .variants = GPT_OSS_120B_VARIANTS,
        .n_variants = sizeof(GPT_OSS_120B_VARIANTS) / sizeof(GPT_OSS_120B_VARIANTS[0]),
        .default_variant = 0,
    },
};

const idletoken_model_spec *idletoken_model_get(const char *id) {
    if (!id || !id[0]) return NULL;
    for (size_t i = 0; i < sizeof(MODELS) / sizeof(MODELS[0]); i++)
        if (strcmp(MODELS[i].id, id) == 0) return &MODELS[i];
    /* Compatibility for settings, scripts and API clients saved while long
     * context was exposed as a duplicate `<model>-1m` SKU. The alias resolves
     * to the base spec, so every outward-facing id is the one real model. */
    const size_t len = strlen(id);
    if (len > 3 && strcmp(id + len - 3, "-1m") == 0 && len - 3 < 128) {
        char base[128];
        memcpy(base, id, len - 3);
        base[len - 3] = '\0';
        for (size_t i = 0; i < sizeof(MODELS) / sizeof(MODELS[0]); i++)
            if (strcmp(MODELS[i].id, base) == 0) return &MODELS[i];
    }
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

/* Leaf name of a path, tolerating either separator (Windows nodes pass
 * backslashes; the variant table is written with forward ones). */
static const char *gguf_leaf(const char *path) {
    const char *base = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    return base;
}

const char *idletoken_model_quant_from_gguf(const idletoken_model_spec *m,
                                            const char *gguf) {
    if (!m || !gguf || !gguf[0]) return "";
    /* An AUTO-GENERATED manifest (model_auto.c, the `--llama-gguf` path the
     * client actually launches) carries no variant table at all: it describes
     * the one file it read a header from, not the family. Borrow the table
     * from the registry row of the same id — the merge in model_auto.c already
     * treats that row as describing the same model, and without this the
     * precision of every client-launched coordinator resolves to "" and gets
     * published as "any precision", which is the most permissive declaration
     * there is rather than the honest one (measured 2026-09-03, both Windows
     * nodes: results/agent-stale-registration-20260903.md). */
    const idletoken_model_spec *table = m;
    if (table->n_variants == 0 && m->id && m->id[0]) {
        const idletoken_model_spec *reg = idletoken_model_get(m->id);
        if (reg && reg->n_variants) table = reg;
    }
    if (table->n_variants == 0) return "";
    const char *base = gguf_leaf(gguf);
    for (uint8_t i = 0; i < table->n_variants; i++)
        if (!strcmp(gguf_leaf(table->variants[i].gguf), base))
            return table->variants[i].quant;
    return "";
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
