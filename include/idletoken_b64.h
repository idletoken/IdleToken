/* IdleToken — base64, standard alphabet with '=' padding.
 *
 * That is libsodium's ORIGINAL variant, which is what both ends of the sealed
 * envelope speak: the platform gateway calls `_sodium.to_base64(...,
 * base64_variants.ORIGINAL)`, and the agent and the coordinator have to match
 * it byte for byte.
 *
 * It lives here rather than inside one binary because THREE places now need the
 * same codec: the platform agent (provider side, opens envelopes), the
 * coordinator's overflow path (consumer side, seals them), and the tests that
 * build fixtures for either. Two copies of a codec that must agree is how a
 * "works on the agent, 400s from the coordinator" bug gets written.
 *
 * C99, no dependencies. */

#ifndef IDLETOKEN_B64_H
#define IDLETOKEN_B64_H

#include <stddef.h>
#include <stdint.h>

/* Encode `n` bytes. Returns a malloc'd NUL-terminated string (caller frees),
 * or NULL on allocation failure. */
char *idletoken_b64_encode(const uint8_t *in, size_t n);

/* Decode a padded base64 string of length `slen`. Returns malloc'd bytes and
 * sets *out_len, or NULL on any invalid character, bad length or misplaced
 * padding — callers treat NULL as "malformed input", never as "empty".
 * `out_len` may be NULL. */
uint8_t *idletoken_b64_decode(const char *s, size_t slen, size_t *out_len);

#endif /* IDLETOKEN_B64_H */
