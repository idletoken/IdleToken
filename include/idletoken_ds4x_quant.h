/* idletoken_ds4x_quant.h — CPU dequantization for GGUF tensor types.
 *
 * The ds4x reference loader materializes every weight as fp32; this module
 * turns a raw quantized tensor row into fp32. v0.1 covers the spec-simple,
 * directly-verifiable formats (F32/F16/Q8_0/Q4_0) with formula unit tests.
 * The K-quants and i-quants GLM/Kimi Q2 actually ship (Q2_K/IQ2_XXS/…) are
 * complex codebook formats whose byte layout must be checked against
 * llama.cpp reference blocks on the DGX — ds4x_dequant_row returns -1 with a
 * clear "unsupported" reason for those until then (never silent garbage).
 *
 * ggml type ids (subset): F32=0 F16=1 Q4_0=2 Q8_0=8 Q2_K=10 Q4_K=12 Q6_K=14
 * IQ2_XXS=16.  C only. No GPU. Unit-tests anywhere.
 */
#ifndef IDLETOKEN_DS4X_QUANT_H
#define IDLETOKEN_DS4X_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* Elements per block / bytes per block for `ggml_type`. 0 = unknown type.
 * (F32/F16 are treated as block_count=1.) */
uint32_t ds4x_type_block_count(uint32_t ggml_type);
uint64_t ds4x_type_block_bytes(uint32_t ggml_type);

/* Supported by ds4x_dequant_row right now? (F32/F16/Q8_0/Q4_0) */
int ds4x_type_supported(uint32_t ggml_type);

/* Dequantize `n` elements from `src` (which must hold n/block_count blocks,
 * n a multiple of block_count) into dst[n] fp32. Returns 0, or -1 with a
 * reason (unsupported type / bad length). */
int ds4x_dequant_row(uint32_t ggml_type, const void *src, float *dst,
                     uint64_t n, char *err, size_t errlen);

/* Standalone half→float (also used by the loader). */
float ds4x_f16_to_f32(uint16_t h);

#endif /* IDLETOKEN_DS4X_QUANT_H */
