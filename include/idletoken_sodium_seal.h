/* IdleToken — libsodium-compatible X25519 sealed box (crypto_box_seal shape).
 *
 * Why this exists next to idletoken_seal/idletoken_seal_open (idletoken_privacy.h):
 * the in-cluster sealed box derives its nonce with TweetNaCl's SHA-512
 * (documented deviation, self-consistent inside the cluster). The marketplace
 * platform (platform/packages/gateway) speaks REAL libsodium crypto_box_seal,
 * whose nonce is BLAKE2b-192(eph_pk || recip_pk). The two are therefore NOT
 * wire-compatible. The platform Agent talks to the platform, so it needs this
 * libsodium-exact construction:
 *
 *   seal:  eph = fresh keypair
 *          nonce = BLAKE2b(24, eph_pk || recip_pk)        (unkeyed)
 *          out   = eph_pk(32) || crypto_box(plain, nonce, recip_pk, eph_sk)
 *   open:  recompute the nonce from out[0..32) and my pk, crypto_box_open.
 *
 * Overhead is the same 48 bytes (eph pk 32 + Poly1305 MAC 16) as the
 * in-cluster variant, so IDLETOKEN_SEAL_OVERHEAD applies unchanged.
 *
 * Crypto primitives: vendored TweetNaCl (X25519 + XSalsa20-Poly1305) +
 * vendored BLAKE2b (vendor/blake2). Verified against real libsodium by the
 * cross-language e2e (test/platform-agent.e2e.spec.ts).
 *
 * C only. No C++. */

#ifndef IDLETOKEN_SODIUM_SEAL_H
#define IDLETOKEN_SODIUM_SEAL_H

#include "idletoken_privacy.h"   /* idletoken_keypair, idletoken_priv_rc, sizes */

#ifdef __cplusplus
extern "C" {
#endif

/* Seal `plain[plen]` to `recip_pk`, libsodium crypto_box_seal wire format.
 * Writes plen + IDLETOKEN_SEAL_OVERHEAD bytes to `out`; sets *out_len. */
idletoken_priv_rc idletoken_sodium_seal(const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                                  const uint8_t *plain, size_t plen,
                                  uint8_t *out, size_t out_cap, size_t *out_len);

/* Open a libsodium crypto_box_seal envelope with the recipient's keypair.
 * IDLETOKEN_PRIV_EAUTH if the box was not sealed to this key (forged/corrupt). */
idletoken_priv_rc idletoken_sodium_seal_open(const idletoken_keypair *recip,
                                       const uint8_t *sealed, size_t slen,
                                       uint8_t *out, size_t out_cap, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_SODIUM_SEAL_H */
