/* IdleToken — sealed envelope framing for the HTTP API. Implementation of
 * include/idletoken_privacy_http.h. Pure framing over idletoken_seal/seal_open;
 * no sockets, unit-testable without a cluster. C only. */

#include "idletoken_privacy_http.h"

#include <string.h>

/* little-endian u32 load/store (self-contained; avoids pulling in net.h) */
static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* -------- consumer side -------------------------------------------------- */

idletoken_priv_rc idletoken_phttp_seal_request(const uint8_t node_pk[IDLETOKEN_PK_BYTES],
                                         const idletoken_keypair *session,
                                         const uint8_t *inner, size_t inner_len,
                                         uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!node_pk || !session || (!inner && inner_len) || !out)
        return IDLETOKEN_PRIV_EINVAL;
    size_t need = inner_len + IDLETOKEN_PHTTP_REQ_OVERHEAD;
    if (out_cap < need) return IDLETOKEN_PRIV_EBUF;

    memcpy(out, IDLETOKEN_PHTTP_REQ_MAGIC, 4);
    out[4] = IDLETOKEN_PHTTP_VERSION;
    out[5] = out[6] = out[7] = 0;                       /* reserved */
    memcpy(out + 8, session->pk, IDLETOKEN_PK_BYTES);      /* reply address */

    size_t sealed_len = 0;
    idletoken_priv_rc rc = idletoken_seal(node_pk, inner, inner_len,
                                    out + IDLETOKEN_PHTTP_REQ_HDR,
                                    out_cap - IDLETOKEN_PHTTP_REQ_HDR, &sealed_len);
    if (rc != IDLETOKEN_PRIV_OK) return rc;
    put_u32(out + 40, (uint32_t)sealed_len);
    if (out_len) *out_len = IDLETOKEN_PHTTP_REQ_HDR + sealed_len;
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_phttp_open_response(const idletoken_keypair *session,
                                          const uint8_t *env, size_t env_len,
                                          uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!session || !env || !out) return IDLETOKEN_PRIV_EINVAL;
    if (env_len < IDLETOKEN_PHTTP_RESP_OVERHEAD) return IDLETOKEN_PRIV_EAUTH;
    if (memcmp(env, IDLETOKEN_PHTTP_RESP_MAGIC, 4) != 0) return IDLETOKEN_PRIV_EAUTH;
    if (env[4] != IDLETOKEN_PHTTP_VERSION) return IDLETOKEN_PRIV_EAUTH;

    uint32_t sealed_len = get_u32(env + 8);
    if ((size_t)IDLETOKEN_PHTTP_RESP_HDR + sealed_len != env_len)
        return IDLETOKEN_PRIV_EAUTH;

    return idletoken_seal_open(session, env + IDLETOKEN_PHTTP_RESP_HDR, sealed_len,
                            out, out_cap, out_len);
}

/* -------- node side ----------------------------------------------------- */

idletoken_priv_rc idletoken_phttp_open_request(const idletoken_keypair *node,
                                         const uint8_t *env, size_t env_len,
                                         uint8_t *out, size_t out_cap, size_t *out_len,
                                         uint8_t out_session_pk[IDLETOKEN_PK_BYTES]) {
    if (!node || !env || !out || !out_session_pk) return IDLETOKEN_PRIV_EINVAL;
    if (env_len < IDLETOKEN_PHTTP_REQ_OVERHEAD) return IDLETOKEN_PRIV_EAUTH;
    if (memcmp(env, IDLETOKEN_PHTTP_REQ_MAGIC, 4) != 0) return IDLETOKEN_PRIV_EAUTH;
    if (env[4] != IDLETOKEN_PHTTP_VERSION) return IDLETOKEN_PRIV_EAUTH;

    uint32_t sealed_len = get_u32(env + 40);
    if ((size_t)IDLETOKEN_PHTTP_REQ_HDR + sealed_len != env_len)
        return IDLETOKEN_PRIV_EAUTH;

    /* copy reply address first (cheap, before the fallible decrypt) */
    memcpy(out_session_pk, env + 8, IDLETOKEN_PK_BYTES);

    return idletoken_seal_open(node, env + IDLETOKEN_PHTTP_REQ_HDR, sealed_len,
                            out, out_cap, out_len);
}

idletoken_priv_rc idletoken_phttp_seal_response(const uint8_t session_pk[IDLETOKEN_PK_BYTES],
                                          const uint8_t *inner, size_t inner_len,
                                          uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!session_pk || (!inner && inner_len) || !out) return IDLETOKEN_PRIV_EINVAL;
    size_t need = inner_len + IDLETOKEN_PHTTP_RESP_OVERHEAD;
    if (out_cap < need) return IDLETOKEN_PRIV_EBUF;

    memcpy(out, IDLETOKEN_PHTTP_RESP_MAGIC, 4);
    out[4] = IDLETOKEN_PHTTP_VERSION;
    out[5] = out[6] = out[7] = 0;

    size_t sealed_len = 0;
    idletoken_priv_rc rc = idletoken_seal(session_pk, inner, inner_len,
                                    out + IDLETOKEN_PHTTP_RESP_HDR,
                                    out_cap - IDLETOKEN_PHTTP_RESP_HDR, &sealed_len);
    if (rc != IDLETOKEN_PRIV_OK) return rc;
    put_u32(out + 8, (uint32_t)sealed_len);
    if (out_len) *out_len = IDLETOKEN_PHTTP_RESP_HDR + sealed_len;
    return IDLETOKEN_PRIV_OK;
}

/* -------- key encoding helpers ------------------------------------------ */

void idletoken_phttp_pk_to_hex(const uint8_t pk[IDLETOKEN_PK_BYTES], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < IDLETOKEN_PK_BYTES; i++) {
        out[2 * i]     = hex[(pk[i] >> 4) & 0xF];
        out[2 * i + 1] = hex[pk[i] & 0xF];
    }
    out[64] = 0;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

idletoken_priv_rc idletoken_phttp_pk_from_hex(const char *hex, uint8_t pk[IDLETOKEN_PK_BYTES]) {
    if (!hex || !pk) return IDLETOKEN_PRIV_EINVAL;
    for (int i = 0; i < IDLETOKEN_PK_BYTES; i++) {
        int hi = hexval(hex[2 * i]);
        int lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return IDLETOKEN_PRIV_EINVAL;
        pk[i] = (uint8_t)((hi << 4) | lo);
    }
    if (hex[2 * IDLETOKEN_PK_BYTES] != 0) return IDLETOKEN_PRIV_EINVAL;  /* extra junk */
    return IDLETOKEN_PRIV_OK;
}
