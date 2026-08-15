/* idletoken_nodecrypt.h — counter-nonce framing for coord <-> worker payloads.
 *
 * The confidentiality primitive is NOT here: `idletoken_cipher_encrypt/decrypt`
 * (src/common/privacy.c, XSalsa20-Poly1305 over vendored TweetNaCl) already
 * does that, and has since the privacy proxy. What was missing is the part that
 * makes it safe to use on a long-lived connection: nonce discipline and replay
 * rejection. That is all this file adds.
 *
 * Why counters rather than random nonces (docs/inter-node-encryption.md §4):
 * a 24-byte random nonce is fine in isolation, but it gives the receiver no way
 * to notice a replayed frame -- it decrypts perfectly, because it IS a frame we
 * sent. A strictly increasing counter turns replay into an arithmetic check
 * that happens *before* any crypto.
 *
 * Nonce layout (24 bytes), unique per (key, sender, receiver, counter):
 *     "ITNC" | sender_id(1) | receiver_id(1) | 10 zero bytes | counter, BE (8)
 *
 * Both endpoint ids are in there because ONE key covers MANY links: the cluster
 * key is derived independently by every node (docs/inter-node-encryption.md §3),
 * so coord->stage0 counter 1 and stage0->stage1 counter 1 would otherwise land
 * on the same nonce. Encrypting two different plaintexts under one key and one
 * nonce cancels the XSalsa20 keystream outright -- the worst failure available
 * here, and it would look like nothing at all from the outside.
 *
 * With both ids present each link counts from 1 independently, and a frame
 * reflected back at its sender fails too: swapping the roles changes the nonce.
 *
 * ids: coordinator = IDLETOKEN_NC_ID_COORD, stage k = k. Both ends already know
 * who they are talking to, so nothing has to be negotiated.
 *
 * Wire format of a wrapped field:  counter (8, big-endian) | ciphertext+MAC
 * The counter travels in the clear because the receiver needs it to rebuild the
 * nonce; it is authenticated implicitly -- a tampered counter yields a nonce
 * that fails the MAC.
 *
 * C only. No C++.
 */
#ifndef IDLETOKEN_NODECRYPT_H
#define IDLETOKEN_NODECRYPT_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_proto.h"    /* IDLETOKEN_SESSION_KEY_BYTES */

/* counter(8) + Poly1305 MAC(16) */
#define IDLETOKEN_NODECRYPT_OVERHEAD 24

/* Endpoint id of the coordinator. Stages use their own stage_id (0..N-1), which
 * cannot collide with this because a cluster never has 255 stages. */
#define IDLETOKEN_NC_ID_COORD 0xFF

/* Bytes of per-formation salt the coordinator mints and ships in ASSIGN_PLAN. */
#define IDLETOKEN_NC_SALT_BYTES 16

typedef enum {
    IDLETOKEN_NC_OK      = 0,
    IDLETOKEN_NC_EINVAL  = -1,  /* bad argument / buffer too small        */
    IDLETOKEN_NC_EAUTH   = -2,  /* MAC failed: tampered, or wrong key     */
    IDLETOKEN_NC_EREPLAY = -3   /* counter not strictly increasing        */
} idletoken_nc_rc;

typedef struct {
    uint8_t  key[IDLETOKEN_SESSION_KEY_BYTES];
    uint64_t send_ctr;    /* next counter to use when wrapping            */
    uint64_t recv_hwm;    /* highest counter accepted so far (0 = none)   */
    uint8_t  my_id;       /* our endpoint id; goes in nonces we send      */
    uint8_t  peer_id;     /* their endpoint id; goes in nonces we accept  */
    uint8_t  ready;       /* 0 = no session key; caller must not wrap     */
} idletoken_nodecrypt;

/* Bind a key to one link. `my_id`/`peer_id` identify the two endpoints and go
 * into every nonce. Safe to call with key == NULL: leaves `ready` 0, which the
 * caller must treat as "this link cannot be encrypted" (a --coordinator cluster
 * has no shared secret at all). */
void idletoken_nodecrypt_init(idletoken_nodecrypt *st,
                              const uint8_t key[IDLETOKEN_SESSION_KEY_BYTES],
                              uint8_t my_id, uint8_t peer_id);

/* cluster_key = HMAC-SHA256(psk, "idletoken-cluster-token-key-v1" || salt).
 *
 * Every node derives this independently from the pairing psk it already holds,
 * so the key needs no distribution and covers worker<->worker links that the
 * pairwise session keys never reached.
 *
 * `salt` is fresh per cluster formation and is NOT optional: without it the same
 * join code yields the same key every time while counters restart at 1, which is
 * nonce reuse. */
void idletoken_nodecrypt_cluster_key(const uint8_t *psk, size_t psk_len,
                                     const uint8_t *salt, size_t salt_len,
                                     uint8_t out[IDLETOKEN_SESSION_KEY_BYTES]);

/* Wipe the key and counters. */
void idletoken_nodecrypt_clear(idletoken_nodecrypt *st);

/* plain -> counter || ciphertext. Needs plen + IDLETOKEN_NODECRYPT_OVERHEAD. */
idletoken_nc_rc idletoken_nodecrypt_wrap(idletoken_nodecrypt *st,
                                         const uint8_t *plain, size_t plen,
                                         uint8_t *out, size_t out_cap, size_t *out_len);

/* counter || ciphertext -> plain. Rejects a counter that is not strictly
 * greater than every counter accepted before, BEFORE attempting to decrypt.
 * On any failure the receive high-water mark is left untouched, so a forged
 * frame cannot advance it and lock out the genuine ones behind it. */
idletoken_nc_rc idletoken_nodecrypt_unwrap(idletoken_nodecrypt *st,
                                           const uint8_t *in, size_t in_len,
                                           uint8_t *out, size_t out_cap, size_t *out_len);

/* Self-test: returns the number of FAILED assertions (0 = all good) and prints
 * one line per assertion to stderr, matching the coordinator's --selftest. */
int idletoken_nodecrypt_selftest(void);

#endif /* IDLETOKEN_NODECRYPT_H */
