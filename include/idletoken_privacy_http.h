/* IdleToken — sealed envelope framing for the HTTP API (G-PRIV, model X).
 *
 * Wraps a normal request/response body (plaintext JSON) in a sealed box so an
 * external consumer can talk to the trusted decrypting node over an untrusted
 * network with no plaintext on the wire. This is the transport framing on top
 * of include/idletoken_privacy.h's SealedTransport — pure portable C, no sockets,
 * so it is unit-testable without a cluster.
 *
 * Design (docs/privacy-design.md model X): the trusted node (Coordinator, or a
 * front proxy co-located with it) is the one plaintext window. A consumer:
 *   1. fetches the node's X25519 public key,
 *   2. generates a per-session keypair,
 *   3. seals the request body to the node's pubkey and sends its session
 *      pubkey alongside (so the node can seal the reply back),
 *   4. opens the sealed reply with its session secret key.
 * The node never ships keys/vocab/text to workers; only hidden states cross the
 * PP boundary. Determined insiders / metadata side channels remain out of scope
 * (design §7) — the goal is to raise cost, not full defence.
 *
 * Wire framing (little-endian, matches the project's length-prefixed style):
 *
 *   Request envelope  (consumer -> node):
 *     0   u8[4]   magic = "HPQ1"
 *     4   u8      version = 1
 *     5   u8      reserved[3]
 *     8   u8[32]  session_pk        consumer session pubkey (reply sealed to it)
 *     40  u32     sealed_len
 *     44  u8[]    sealed_request    = seal(node_pk, inner_request_json)
 *
 *   Response envelope (node -> consumer):
 *     0   u8[4]   magic = "HPS1"
 *     4   u8      version = 1
 *     5   u8      reserved[3]
 *     8   u32     sealed_len
 *     12  u8[]    sealed_response   = seal(session_pk, inner_response_json)
 *
 * C only. No C++.
 */

#ifndef IDLETOKEN_PRIVACY_HTTP_H
#define IDLETOKEN_PRIVACY_HTTP_H

#include "idletoken_privacy.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IDLETOKEN_PHTTP_REQ_MAGIC  "HPQ1"
#define IDLETOKEN_PHTTP_RESP_MAGIC  "HPS1"
#define IDLETOKEN_PHTTP_VERSION    1

/* Fixed header sizes (before the variable sealed blob). */
#define IDLETOKEN_PHTTP_REQ_HDR   44   /* magic4 + ver1 + rsv3 + pk32 + len4 */
#define IDLETOKEN_PHTTP_RESP_HDR   12   /* magic4 + ver1 + rsv3 + len4 */

/* Total overhead added to an inner body when enveloped (header + seal). */
#define IDLETOKEN_PHTTP_REQ_OVERHEAD  (IDLETOKEN_PHTTP_REQ_HDR  + IDLETOKEN_SEAL_OVERHEAD)
#define IDLETOKEN_PHTTP_RESP_OVERHEAD (IDLETOKEN_PHTTP_RESP_HDR + IDLETOKEN_SEAL_OVERHEAD)

/* -------- consumer side -------------------------------------------------- */

/* Seal `inner` (plaintext request JSON) into a request envelope addressed to
 * `node_pk`, tagged with `session->pk` so the node seals its reply back.
 * Writes inner_len + IDLETOKEN_PHTTP_REQ_OVERHEAD bytes to `out`. */
idletoken_priv_rc idletoken_phttp_seal_request(const uint8_t node_pk[IDLETOKEN_PK_BYTES],
                                         const idletoken_keypair *session,
                                         const uint8_t *inner, size_t inner_len,
                                         uint8_t *out, size_t out_cap, size_t *out_len);

/* Open a response envelope with the session secret key. IDLETOKEN_PRIV_EAUTH if
 * the reply was not sealed to this session key or was tampered. */
idletoken_priv_rc idletoken_phttp_open_response(const idletoken_keypair *session,
                                          const uint8_t *env, size_t env_len,
                                          uint8_t *out, size_t out_cap, size_t *out_len);

/* -------- node (Coordinator / proxy) side ------------------------------- */

/* Open a request envelope with the node's keypair. On success writes the inner
 * plaintext request to `out` and copies the consumer's session pubkey to
 * `out_session_pk` (used to seal the reply). IDLETOKEN_PRIV_EAUTH on bad magic/
 * version, wrong node key, or tamper — i.e. an unauthenticated request is
 * rejected (G-PRIV oracle item 4). The caller MUST treat `out` as the sole
 * plaintext window: mlock it, use it, zeroize it. */
idletoken_priv_rc idletoken_phttp_open_request(const idletoken_keypair *node,
                                         const uint8_t *env, size_t env_len,
                                         uint8_t *out, size_t out_cap, size_t *out_len,
                                         uint8_t out_session_pk[IDLETOKEN_PK_BYTES]);

/* Seal `inner` (plaintext response JSON) back to the consumer's session key. */
idletoken_priv_rc idletoken_phttp_seal_response(const uint8_t session_pk[IDLETOKEN_PK_BYTES],
                                          const uint8_t *inner, size_t inner_len,
                                          uint8_t *out, size_t out_cap, size_t *out_len);

/* -------- key encoding helpers (for GET /pubkey, key files) -------------- */

/* Lowercase hex of a 32-byte key. `out` must hold >= 65 bytes (64 + NUL). */
void idletoken_phttp_pk_to_hex(const uint8_t pk[IDLETOKEN_PK_BYTES], char out[65]);
/* Parse 64 hex chars into 32 bytes. Returns IDLETOKEN_PRIV_OK / _EINVAL. */
idletoken_priv_rc idletoken_phttp_pk_from_hex(const char *hex, uint8_t pk[IDLETOKEN_PK_BYTES]);

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_PRIVACY_HTTP_H */
