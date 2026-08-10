/* IdleToken — sealed-envelope-over-HTTP codec self-test.
 *
 * Proves the transport framing in src/common/privacy_http.c: a consumer seals
 * a request to the node's pubkey, the node opens it (sole plaintext window),
 * produces a response, seals it back to the consumer's session key, and the
 * consumer opens it — with no plaintext in either envelope and unauthenticated
 * requests rejected. This is the piece that lets an external client talk to
 * the trusted decrypting node over an untrusted network (G-PRIV model X).
 *
 * Prints `G_PRIV_HTTP_SELFTEST_OK` on success (exit 0), else FAIL (exit 1).
 * C only.
 */

#include "idletoken_privacy_http.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static void ok(const char *w)  { printf("  [ok]   %s\n", w); }
static void bad(const char *w) { printf("  [FAIL] %s\n", w); g_fail = 1; }
#define CHECK(c, w) do { if (c) ok(w); else bad(w); } while (0)

static int contains(const uint8_t *hay, size_t hlen, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || nlen > hlen) return 0;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return 1;
    return 0;
}

int main(void) {
    printf("IdleToken G-PRIV sealed-HTTP codec self-test\n");
    printf("-------------------------------------------------------------\n");

    const char *REQ  = "{\"content\":\"SECRET-PROMPT weather in Paris?\",\"max_tokens\":16}";
    const char *RESP = "{\"content\":[{\"type\":\"text\",\"text\":\"SECRET-ANSWER sunny\"}]}";

    idletoken_keypair node, session, attacker;
    idletoken_keypair_generate(&node);
    idletoken_keypair_generate(&session);
    idletoken_keypair_generate(&attacker);

    /* --- consumer seals request to node pubkey --- */
    uint8_t req_env[1024]; size_t req_env_len = 0;
    CHECK(idletoken_phttp_seal_request(node.pk, &session,
                                    (const uint8_t *)REQ, strlen(REQ),
                                    req_env, sizeof(req_env), &req_env_len) == IDLETOKEN_PRIV_OK,
          "consumer seals request envelope");
    CHECK(req_env_len == strlen(REQ) + IDLETOKEN_PHTTP_REQ_OVERHEAD,
          "request envelope length = inner + fixed overhead");
    CHECK(!contains(req_env, req_env_len, "SECRET-PROMPT"),
          "no plaintext in request envelope (wire is ciphertext)");
    CHECK(memcmp(req_env, IDLETOKEN_PHTTP_REQ_MAGIC, 4) == 0,
          "request envelope carries the request magic");

    /* --- node opens request (sole plaintext window) --- */
    uint8_t inner[1024]; size_t inner_len = 0;
    uint8_t reply_to[IDLETOKEN_PK_BYTES];
    CHECK(idletoken_phttp_open_request(&node, req_env, req_env_len,
                                    inner, sizeof(inner), &inner_len, reply_to) == IDLETOKEN_PRIV_OK,
          "node opens request with its secret key");
    CHECK(inner_len == strlen(REQ) && memcmp(inner, REQ, inner_len) == 0,
          "recovered inner request matches original exactly");
    CHECK(idletoken_ct_eq(reply_to, session.pk, IDLETOKEN_PK_BYTES),
          "node learns the consumer session pubkey to reply to");

    /* --- rejections (auth works) --- */
    uint8_t junk[1024]; size_t junk_len = 0; uint8_t junk_pk[IDLETOKEN_PK_BYTES];
    CHECK(idletoken_phttp_open_request(&attacker, req_env, req_env_len,
                                    junk, sizeof(junk), &junk_len, junk_pk) == IDLETOKEN_PRIV_EAUTH,
          "wrong node key is rejected (EAUTH)");
    uint8_t tampered[1024];
    memcpy(tampered, req_env, req_env_len);
    tampered[req_env_len - 1] ^= 0x01;
    CHECK(idletoken_phttp_open_request(&node, tampered, req_env_len,
                                    junk, sizeof(junk), &junk_len, junk_pk) == IDLETOKEN_PRIV_EAUTH,
          "tampered request is rejected");
    uint8_t badmagic[1024];
    memcpy(badmagic, req_env, req_env_len);
    badmagic[0] = 'X';
    CHECK(idletoken_phttp_open_request(&node, badmagic, req_env_len,
                                    junk, sizeof(junk), &junk_len, junk_pk) == IDLETOKEN_PRIV_EAUTH,
          "bad-magic request is rejected");

    /* --- node seals response back to the session key --- */
    uint8_t resp_env[1024]; size_t resp_env_len = 0;
    CHECK(idletoken_phttp_seal_response(reply_to, (const uint8_t *)RESP, strlen(RESP),
                                     resp_env, sizeof(resp_env), &resp_env_len) == IDLETOKEN_PRIV_OK,
          "node seals response to consumer session key");
    CHECK(!contains(resp_env, resp_env_len, "SECRET-ANSWER"),
          "no plaintext in response envelope (wire is ciphertext)");

    /* --- consumer opens response --- */
    uint8_t got[1024]; size_t got_len = 0;
    CHECK(idletoken_phttp_open_response(&session, resp_env, resp_env_len,
                                     got, sizeof(got), &got_len) == IDLETOKEN_PRIV_OK &&
          got_len == strlen(RESP) && memcmp(got, RESP, got_len) == 0,
          "consumer opens response end-to-end (matches original)");
    /* the attacker's session key can't open it */
    CHECK(idletoken_phttp_open_response(&attacker, resp_env, resp_env_len,
                                     got, sizeof(got), &got_len) == IDLETOKEN_PRIV_EAUTH,
          "response is not openable by a different session key");

    /* --- pubkey hex round-trip (GET /pubkey wire format) --- */
    char hex[65]; uint8_t back[IDLETOKEN_PK_BYTES];
    idletoken_phttp_pk_to_hex(node.pk, hex);
    CHECK(strlen(hex) == 64 &&
          idletoken_phttp_pk_from_hex(hex, back) == IDLETOKEN_PRIV_OK &&
          idletoken_ct_eq(back, node.pk, IDLETOKEN_PK_BYTES),
          "node pubkey hex encode/decode round-trips");
    CHECK(idletoken_phttp_pk_from_hex("nothex", back) != IDLETOKEN_PRIV_OK,
          "malformed pubkey hex is rejected");

    idletoken_secure_zero(inner, sizeof(inner));

    printf("-------------------------------------------------------------\n");
    if (g_fail) { printf("G_PRIV_HTTP_SELFTEST_FAIL\n"); return 1; }
    printf("G_PRIV_HTTP_SELFTEST_OK\n");
    return 0;
}
