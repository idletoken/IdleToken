/* BLAKE2b (RFC 7693) — unkeyed one-shot subset. See UPSTREAM in this
 * directory for provenance and why it is vendored.
 *
 * This is the exact primitive libsodium's crypto_generichash(out, outlen,
 * in, inlen, NULL, 0) computes; IdleToken uses it only to derive the sealed-box
 * nonce (24-byte digest of eph_pk || recip_pk, a 64-byte input).
 *
 * C99, no dependencies. */

#ifndef BLAKE2_H
#define BLAKE2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE2B_OUTBYTES 64   /* maximum digest length */

/* One-shot unkeyed BLAKE2b. `outlen` in [1, 64]. Returns 0 on success,
 * -1 on bad arguments. Matches libsodium crypto_generichash with no key. */
int blake2b(void *out, size_t outlen, const void *in, size_t inlen);

#ifdef __cplusplus
}
#endif

#endif /* BLAKE2_H */
