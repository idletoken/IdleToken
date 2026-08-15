/* nodecrypt.c — counter-nonce framing + replay rejection for coord <-> worker.
 * Contract and rationale: include/idletoken_nodecrypt.h.
 * C only. No C++. */

#include "idletoken_nodecrypt.h"
#include "idletoken_privacy.h"
#include "idletoken_discovery.h"   /* idletoken_hmac_sha256 */

#include <stdio.h>
#include <string.h>

#define NC_CTR_BYTES 8

static void nc_nonce(uint8_t nonce[IDLETOKEN_NONCE_BYTES],
                     uint8_t sender, uint8_t receiver, uint64_t ctr) {
    memset(nonce, 0, IDLETOKEN_NONCE_BYTES);
    nonce[0] = 'I'; nonce[1] = 'T'; nonce[2] = 'N'; nonce[3] = 'C';
    /* BOTH endpoints: one cluster key covers many links, so the sender alone
     * would let coord->stage0 #1 and stage0->stage1 #1 share a nonce. */
    nonce[4] = sender;
    nonce[5] = receiver;
    for (int i = 0; i < 8; i++)
        nonce[IDLETOKEN_NONCE_BYTES - 1 - i] = (uint8_t)((ctr >> (8 * i)) & 0xff);
}

static void nc_put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)((v >> (8 * (7 - i))) & 0xff);
}

static uint64_t nc_get_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}

void idletoken_nodecrypt_cluster_key(const uint8_t *psk, size_t psk_len,
                                     const uint8_t *salt, size_t salt_len,
                                     uint8_t out[IDLETOKEN_SESSION_KEY_BYTES]) {
    static const char label[] = "idletoken-cluster-token-key-v1";
    uint8_t msg[sizeof(label) - 1 + IDLETOKEN_NC_SALT_BYTES];
    size_t n = sizeof(label) - 1;
    memcpy(msg, label, n);
    if (salt && salt_len) {
        if (salt_len > IDLETOKEN_NC_SALT_BYTES) salt_len = IDLETOKEN_NC_SALT_BYTES;
        memcpy(msg + n, salt, salt_len);
        n += salt_len;
    }
    idletoken_hmac_sha256(psk, psk_len, msg, n, out);
}

void idletoken_nodecrypt_init(idletoken_nodecrypt *st,
                              const uint8_t key[IDLETOKEN_SESSION_KEY_BYTES],
                              uint8_t my_id, uint8_t peer_id) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->my_id   = my_id;
    st->peer_id = peer_id;
    if (!key) return;                 /* ready stays 0: cannot be encrypted */
    memcpy(st->key, key, IDLETOKEN_SESSION_KEY_BYTES);
    /* Counters start at 1, not 0: 0 doubles as "nothing accepted yet" in
     * recv_hwm, and a frame numbered 0 would be indistinguishable from that. */
    st->send_ctr = 1;
    st->ready    = 1;
}

void idletoken_nodecrypt_clear(idletoken_nodecrypt *st) {
    if (!st) return;
    idletoken_secure_zero(st->key, sizeof(st->key));
    memset(st, 0, sizeof(*st));
}

idletoken_nc_rc idletoken_nodecrypt_wrap(idletoken_nodecrypt *st,
                                         const uint8_t *plain, size_t plen,
                                         uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!st || !st->ready || (!plain && plen) || !out) return IDLETOKEN_NC_EINVAL;
    if (out_cap < plen + IDLETOKEN_NODECRYPT_OVERHEAD) return IDLETOKEN_NC_EINVAL;

    uint64_t ctr = st->send_ctr;
    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    nc_nonce(nonce, st->my_id, st->peer_id, ctr);

    size_t clen = 0;
    if (idletoken_cipher_encrypt(st->key, nonce, plain, plen,
                                 out + NC_CTR_BYTES, out_cap - NC_CTR_BYTES, &clen) != IDLETOKEN_PRIV_OK)
        return IDLETOKEN_NC_EINVAL;

    nc_put_u64(out, ctr);
    st->send_ctr++;
    if (out_len) *out_len = NC_CTR_BYTES + clen;
    return IDLETOKEN_NC_OK;
}

idletoken_nc_rc idletoken_nodecrypt_unwrap(idletoken_nodecrypt *st,
                                           const uint8_t *in, size_t in_len,
                                           uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!st || !st->ready || !in || !out) return IDLETOKEN_NC_EINVAL;
    if (in_len < IDLETOKEN_NODECRYPT_OVERHEAD) return IDLETOKEN_NC_EINVAL;

    uint64_t ctr = nc_get_u64(in);
    /* Replay/reorder check FIRST — cheaper than crypto, and it is the whole
     * reason the counter exists: a replayed frame is a frame we really did
     * send, so the MAC on it is perfectly valid and would let it through. */
    if (ctr <= st->recv_hwm) return IDLETOKEN_NC_EREPLAY;

    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    nc_nonce(nonce, st->peer_id, st->my_id, ctr);

    size_t plen = 0;
    if (idletoken_cipher_decrypt(st->key, nonce, in + NC_CTR_BYTES, in_len - NC_CTR_BYTES,
                                 out, out_cap, &plen) != IDLETOKEN_PRIV_OK)
        return IDLETOKEN_NC_EAUTH;   /* hwm deliberately NOT advanced */

    st->recv_hwm = ctr;
    if (out_len) *out_len = plen;
    return IDLETOKEN_NC_OK;
}

/* ---------------------------------------------------------------------- */

/* Substring search over raw bytes. Not memmem(): that is a GNU/BSD extension
 * and these binaries also build under MinGW, where it does not exist. */
static int nc_contains(const uint8_t *hay, size_t hn, const uint8_t *needle, size_t nn) {
    if (nn == 0 || hn < nn) return 0;
    for (size_t i = 0; i + nn <= hn; i++)
        if (memcmp(hay + i, needle, nn) == 0) return 1;
    return 0;
}

int idletoken_nodecrypt_selftest(void) {
    int fails = 0;
    #define NCT(cond, what) do { \
        if (cond) { fprintf(stderr, "selftest PASS nodecrypt: %s\n", (what)); } \
        else { fprintf(stderr, "selftest FAIL nodecrypt: %s\n", (what)); fails++; } \
    } while (0)

    uint8_t key[IDLETOKEN_SESSION_KEY_BYTES];
    for (size_t i = 0; i < sizeof(key); i++) key[i] = (uint8_t)(i * 7 + 1);

    idletoken_nodecrypt c, w;
    idletoken_nodecrypt_init(&c, key, IDLETOKEN_NC_ID_COORD, 0);  /* coord -> stage 0 */
    idletoken_nodecrypt_init(&w, key, 0, IDLETOKEN_NC_ID_COORD);  /* stage 0 <- coord */

    const uint8_t toks[] = { 0x11,0x22,0x33,0x44, 0x55,0x66,0x77,0x88 };
    uint8_t box[64], plain[64];
    size_t blen = 0, plen = 0;

    NCT(idletoken_nodecrypt_wrap(&c, toks, sizeof(toks), box, sizeof(box), &blen) == IDLETOKEN_NC_OK,
        "wrap succeeds");
    NCT(blen == sizeof(toks) + IDLETOKEN_NODECRYPT_OVERHEAD, "wrapped size = plain + 24");
    /* The point of the exercise: the token bytes must not survive in the clear. */
    NCT(!nc_contains(box, blen, toks, sizeof(toks)), "token bytes are not in the wrapped frame");

    NCT(idletoken_nodecrypt_unwrap(&w, box, blen, plain, sizeof(plain), &plen) == IDLETOKEN_NC_OK,
        "peer unwraps it");
    NCT(plen == sizeof(toks) && memcmp(plain, toks, sizeof(toks)) == 0, "round trip is byte-exact");

    /* Replay: the very frame we just accepted, offered again. */
    NCT(idletoken_nodecrypt_unwrap(&w, box, blen, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EREPLAY,
        "replaying an accepted frame is rejected");

    /* A second, genuine frame still goes through after the rejection — the
     * failed replay must not have disturbed the receive state. */
    uint8_t box2[64]; size_t blen2 = 0;
    NCT(idletoken_nodecrypt_wrap(&c, toks, sizeof(toks), box2, sizeof(box2), &blen2) == IDLETOKEN_NC_OK,
        "second wrap succeeds");
    NCT(memcmp(box, box2, blen < blen2 ? blen : blen2) != 0,
        "same plaintext twice produces different bytes (counter really moves)");
    NCT(idletoken_nodecrypt_unwrap(&w, box2, blen2, plain, sizeof(plain), &plen) == IDLETOKEN_NC_OK,
        "the next genuine frame is still accepted after a rejected replay");

    /* Tamper: flip a ciphertext bit. */
    uint8_t bad[64]; memcpy(bad, box2, blen2); bad[blen2 - 1] ^= 0x01;
    idletoken_nodecrypt w2; idletoken_nodecrypt_init(&w2, key, 0, IDLETOKEN_NC_ID_COORD);
    NCT(idletoken_nodecrypt_unwrap(&w2, bad, blen2, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EAUTH,
        "a flipped ciphertext bit fails the MAC");

    /* Tamper: rewrite the counter. It is in the clear, but it feeds the nonce,
     * so changing it breaks the MAC rather than shifting the replay window. */
    uint8_t bad2[64]; memcpy(bad2, box2, blen2); bad2[7] ^= 0x40;
    idletoken_nodecrypt w3; idletoken_nodecrypt_init(&w3, key, 0, IDLETOKEN_NC_ID_COORD);
    NCT(idletoken_nodecrypt_unwrap(&w3, bad2, blen2, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EAUTH,
        "rewriting the cleartext counter fails the MAC");

    /* Wrong key. */
    uint8_t other[IDLETOKEN_SESSION_KEY_BYTES];
    for (size_t i = 0; i < sizeof(other); i++) other[i] = (uint8_t)(i * 3 + 9);
    idletoken_nodecrypt w4; idletoken_nodecrypt_init(&w4, other, 0, IDLETOKEN_NC_ID_COORD);
    NCT(idletoken_nodecrypt_unwrap(&w4, box2, blen2, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EAUTH,
        "a different session key cannot open it");

    /* Reflection: bounce a coord->worker frame back at a coordinator-side
     * state. Same key, wrong direction tag => the nonce differs => MAC fails. */
    idletoken_nodecrypt c2; idletoken_nodecrypt_init(&c2, key, IDLETOKEN_NC_ID_COORD, 0);
    NCT(idletoken_nodecrypt_unwrap(&c2, box2, blen2, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EAUTH,
        "a frame reflected back at its own direction is rejected");

    /* No session key (an unpaired --coordinator cluster) must refuse to wrap
     * rather than quietly emit cleartext. */
    idletoken_nodecrypt none; idletoken_nodecrypt_init(&none, NULL, IDLETOKEN_NC_ID_COORD, 0);
    NCT(idletoken_nodecrypt_wrap(&none, toks, sizeof(toks), box, sizeof(box), &blen) == IDLETOKEN_NC_EINVAL,
        "without a session key, wrap refuses (never falls back to cleartext)");

    /* Too small an output buffer is an error, not a truncated frame. */
    idletoken_nodecrypt c3; idletoken_nodecrypt_init(&c3, key, IDLETOKEN_NC_ID_COORD, 0);
    NCT(idletoken_nodecrypt_wrap(&c3, toks, sizeof(toks), box, sizeof(toks) + 4, &blen) == IDLETOKEN_NC_EINVAL,
        "an undersized output buffer is refused");

    /* --- one cluster key, many links -------------------------------------
     *
     * This is the pair of assertions the endpoint ids exist for. With only a
     * direction byte, coord->stage0 #1 and stage0->stage1 #1 shared a nonce;
     * under one key that cancels the XSalsa20 keystream, and NOTHING about the
     * output looks wrong. So: same key, same counter, two different links must
     * produce different bytes, and neither link may open the other's frame. */
    {
        /* SAME sender, two different receivers. Getting this wrong was the first
         * version of this test: it used two different senders, where a
         * sender-only nonce already differs, so removing the receiver byte did
         * not turn it red and the assertion proved nothing.
         *
         * Today's chain topology happens to give each node exactly one outgoing
         * token link, so a sender-only nonce would survive -- which is precisely
         * why this is worth pinning now. The moment SP or concurrent
         * micro-batching gives a node a second outgoing link, sender-only
         * collides, and a keystream collision is invisible from the outside. */
        idletoken_nodecrypt a, b;
        idletoken_nodecrypt_init(&a, key, 0, 1);                      /* stage0 -> stage1 */
        idletoken_nodecrypt_init(&b, key, 0, IDLETOKEN_NC_ID_COORD);  /* stage0 -> coord  */
        uint8_t fa[64], fb[64]; size_t la = 0, lb = 0;
        idletoken_nodecrypt_wrap(&a, toks, sizeof(toks), fa, sizeof(fa), &la);
        idletoken_nodecrypt_wrap(&b, toks, sizeof(toks), fb, sizeof(fb), &lb);
        NCT(la == lb && memcmp(fa, fb, la) != 0,
            "two links, same key and counter, produce different ciphertext (no nonce collision)");

        idletoken_nodecrypt rx;
        idletoken_nodecrypt_init(&rx, key, IDLETOKEN_NC_ID_COORD, 0); /* coord expects stage0 */
        NCT(idletoken_nodecrypt_unwrap(&rx, fa, la, plain, sizeof(plain), &plen) == IDLETOKEN_NC_EAUTH,
            "a frame meant for another link cannot be opened here");
        idletoken_nodecrypt_clear(&a); idletoken_nodecrypt_clear(&b); idletoken_nodecrypt_clear(&rx);
    }

    /* --- cluster key derivation ------------------------------------------ */
    {
        const uint8_t psk[32] = { 1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,
                                  17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32 };
        uint8_t s1[IDLETOKEN_NC_SALT_BYTES], s2[IDLETOKEN_NC_SALT_BYTES];
        memset(s1, 0xA1, sizeof(s1)); memset(s2, 0xB2, sizeof(s2));
        uint8_t k1[32], k1b[32], k2[32];
        idletoken_nodecrypt_cluster_key(psk, sizeof(psk), s1, sizeof(s1), k1);
        idletoken_nodecrypt_cluster_key(psk, sizeof(psk), s1, sizeof(s1), k1b);
        idletoken_nodecrypt_cluster_key(psk, sizeof(psk), s2, sizeof(s2), k2);
        NCT(memcmp(k1, k1b, 32) == 0,
            "cluster key is deterministic (every node derives the same one, no distribution)");
        /* Without this the same join code reuses one key while counters restart
         * at 1 on every formation -- nonce reuse, the failure that voids it all. */
        NCT(memcmp(k1, k2, 32) != 0, "a different salt gives a different cluster key");
        uint8_t zero[32]; memset(zero, 0, sizeof(zero));
        NCT(memcmp(k1, zero, 32) != 0 && memcmp(k1, psk, 32) != 0,
            "the cluster key is neither zero nor the psk itself");
    }

    idletoken_nodecrypt_clear(&c);
    idletoken_nodecrypt_clear(&w);
    #undef NCT
    return fails;
}
