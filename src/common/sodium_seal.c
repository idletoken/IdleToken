/* IdleToken — libsodium-compatible X25519 sealed box. See
 * include/idletoken_sodium_seal.h for why this exists (the in-cluster sealed box
 * in src/common/privacy.c derives its nonce with SHA-512 and is NOT wire-
 * compatible with libsodium; the platform speaks real libsodium).
 *
 * Structure mirrors idletoken_seal/idletoken_seal_open in src/common/privacy.c —
 * same TweetNaCl zero-padding handling, same secure-wipe discipline — with
 * exactly one difference: the nonce is BLAKE2b-192(eph_pk || recip_pk).
 *
 * C only. No C++. */

#include "idletoken_sodium_seal.h"
#include "tweetnacl.h"
#include "blake2.h"

#include <stdlib.h>
#include <string.h>

/* crypto_box uses the NaCl zero-padding convention: ZEROBYTES(32) leading
 * zeros on the plaintext, BOXZEROBYTES(16) on the ciphertext (see
 * src/common/privacy.c for the long version of this comment). */
#define ZB   crypto_box_ZEROBYTES      /* 32 */
#define BZB  crypto_box_BOXZEROBYTES   /* 16 */

/* nonce = BLAKE2b(24, eph_pk || recip_pk) — libsodium crypto_box_seal_nonce. */
static void sodium_seal_nonce(const uint8_t eph_pk[IDLETOKEN_PK_BYTES],
                              const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                              uint8_t nonce[IDLETOKEN_NONCE_BYTES]) {
    unsigned char cat[2 * IDLETOKEN_PK_BYTES];
    memcpy(cat, eph_pk, IDLETOKEN_PK_BYTES);
    memcpy(cat + IDLETOKEN_PK_BYTES, recip_pk, IDLETOKEN_PK_BYTES);
    blake2b(nonce, IDLETOKEN_NONCE_BYTES, cat, sizeof(cat));
}

idletoken_priv_rc idletoken_sodium_seal(const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                                  const uint8_t *plain, size_t plen,
                                  uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!recip_pk || (!plain && plen) || !out) return IDLETOKEN_PRIV_EINVAL;
    if (out_cap < plen + IDLETOKEN_SEAL_OVERHEAD) return IDLETOKEN_PRIV_EBUF;

    idletoken_keypair eph;
    crypto_box_keypair(eph.pk, eph.sk);

    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    sodium_seal_nonce(eph.pk, recip_pk, nonce);

    size_t mlen = ZB + plen;
    unsigned char *m = (unsigned char *)malloc(mlen);
    unsigned char *c = (unsigned char *)malloc(mlen);
    if (!m || !c) { free(m); free(c); idletoken_secure_zero(eph.sk, sizeof(eph.sk)); return IDLETOKEN_PRIV_EINVAL; }
    memset(m, 0, ZB);
    if (plen) memcpy(m + ZB, plain, plen);

    int rc = crypto_box(c, m, mlen, nonce, recip_pk, eph.sk);
    idletoken_secure_zero(eph.sk, sizeof(eph.sk));   /* ephemeral sk dies now */
    if (rc != 0) { idletoken_secure_zero(m, mlen); free(m); free(c); return IDLETOKEN_PRIV_EINVAL; }

    /* out = eph.pk || compact_ciphertext (drop the 16 leading zero bytes) */
    memcpy(out, eph.pk, IDLETOKEN_PK_BYTES);
    memcpy(out + IDLETOKEN_PK_BYTES, c + BZB, plen + IDLETOKEN_MAC_BYTES);
    if (out_len) *out_len = plen + IDLETOKEN_SEAL_OVERHEAD;

    idletoken_secure_zero(m, mlen);
    free(m); free(c);
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_sodium_seal_open(const idletoken_keypair *recip,
                                       const uint8_t *sealed, size_t slen,
                                       uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!recip || !sealed || !out) return IDLETOKEN_PRIV_EINVAL;
    if (slen < IDLETOKEN_SEAL_OVERHEAD) return IDLETOKEN_PRIV_EAUTH;

    const uint8_t *eph_pk = sealed;
    const uint8_t *ct     = sealed + IDLETOKEN_PK_BYTES;
    size_t ctlen = slen - IDLETOKEN_PK_BYTES;          /* = plen + MAC */
    size_t plen  = ctlen - IDLETOKEN_MAC_BYTES;
    if (out_cap < plen) return IDLETOKEN_PRIV_EBUF;

    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    sodium_seal_nonce(eph_pk, recip->pk, nonce);

    size_t buflen = BZB + ctlen;   /* = ZB + plen */
    unsigned char *c = (unsigned char *)malloc(buflen);
    unsigned char *m = (unsigned char *)malloc(buflen);
    if (!c || !m) { free(c); free(m); return IDLETOKEN_PRIV_EINVAL; }
    memset(c, 0, BZB);
    memcpy(c + BZB, ct, ctlen);

    int rc = crypto_box_open(m, c, buflen, nonce, eph_pk, recip->sk);
    if (rc != 0) { free(c); idletoken_secure_zero(m, buflen); free(m); return IDLETOKEN_PRIV_EAUTH; }

    if (plen) memcpy(out, m + ZB, plen);
    if (out_len) *out_len = plen;

    idletoken_secure_zero(m, buflen);
    free(c); free(m);
    return IDLETOKEN_PRIV_OK;
}
