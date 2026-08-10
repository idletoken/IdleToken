/* BLAKE2b (RFC 7693) — unkeyed one-shot subset. Public domain / CC0; see
 * UPSTREAM in this directory. Same algorithm as the reference blake2b-ref.c,
 * with the API cut down to the single call IdleToken needs. */

#include "blake2.h"

#include <string.h>

/* Initialization vector (same as SHA-512's IV, RFC 7693 §2.6). */
static const uint64_t blake2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

/* Message word schedule permutations (RFC 7693 §2.7). Rounds 10 and 11 of
 * the 12-round BLAKE2b reuse rows 0 and 1. */
static const uint8_t blake2b_sigma[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};

static uint64_t load64_le(const uint8_t *p) {
    return  (uint64_t)p[0]        | ((uint64_t)p[1] <<  8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static void store64_le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t rotr64(uint64_t x, unsigned n) {
    return (x >> n) | (x << (64 - n));
}

/* Compress one 128-byte block. `t` is the total byte counter INCLUDING this
 * block; `final` is nonzero for the last block (sets the f0 flag). */
static void blake2b_compress(uint64_t h[8], const uint8_t block[128],
                             uint64_t t, int final) {
    uint64_t m[16], v[16];

    for (int i = 0; i < 16; i++) m[i] = load64_le(block + 8 * i);
    for (int i = 0; i <  8; i++) v[i] = h[i];
    for (int i = 0; i <  8; i++) v[8 + i] = blake2b_iv[i];

    v[12] ^= t;                     /* low word of the offset counter  */
    /* v[13] ^= t_hi — inputs here are 64 bytes, the high word stays 0 */
    if (final) v[14] = ~v[14];      /* f0: final-block flag            */

    /* The G mixing function (RFC 7693 §3.1) applied per the round layout:
     * columns first, then diagonals. */
#define G(r, i, a, b, c, d)                                     \
    do {                                                        \
        a = a + b + m[blake2b_sigma[r][2 * (i)]];               \
        d = rotr64(d ^ a, 32);                                  \
        c = c + d;                                              \
        b = rotr64(b ^ c, 24);                                  \
        a = a + b + m[blake2b_sigma[r][2 * (i) + 1]];           \
        d = rotr64(d ^ a, 16);                                  \
        c = c + d;                                              \
        b = rotr64(b ^ c, 63);                                  \
    } while (0)

    for (int round = 0; round < 12; round++) {
        int r = round % 10;         /* rounds 10,11 reuse sigma rows 0,1 */
        G(r, 0, v[0], v[4], v[ 8], v[12]);
        G(r, 1, v[1], v[5], v[ 9], v[13]);
        G(r, 2, v[2], v[6], v[10], v[14]);
        G(r, 3, v[3], v[7], v[11], v[15]);
        G(r, 4, v[0], v[5], v[10], v[15]);
        G(r, 5, v[1], v[6], v[11], v[12]);
        G(r, 6, v[2], v[7], v[ 8], v[13]);
        G(r, 7, v[3], v[4], v[ 9], v[14]);
    }
#undef G

    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[8 + i];
}

int blake2b(void *out, size_t outlen, const void *in, size_t inlen) {
    if (!out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) return -1;
    if (!in && inlen) return -1;

    /* Parameter block folded into h[0] (RFC 7693 §2.5): digest_length,
     * key_length=0, fanout=1, depth=1, everything else zero. NOTE: outlen is
     * part of the parameters, so BLAKE2b-192 is NOT a truncated BLAKE2b-512
     * — this is why we cannot fake it with a 64-byte hash. */
    uint64_t h[8];
    for (int i = 0; i < 8; i++) h[i] = blake2b_iv[i];
    h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;

    const uint8_t *p = (const uint8_t *)in;
    uint64_t t = 0;

    /* All full blocks except the last one (the final block must go through
     * the finalizing compress, even when it is exactly 128 bytes). */
    while (inlen > 128) {
        t += 128;
        blake2b_compress(h, p, t, 0);
        p += 128;
        inlen -= 128;
    }

    /* Final block, zero-padded (an empty message hashes one zero block). */
    uint8_t block[128];
    memset(block, 0, sizeof(block));
    if (inlen) memcpy(block, p, inlen);
    t += (uint64_t)inlen;
    blake2b_compress(h, block, t, 1);

    uint8_t full[BLAKE2B_OUTBYTES];
    for (int i = 0; i < 8; i++) store64_le(full + 8 * i, h[i]);
    memcpy(out, full, outlen);
    return 0;
}
