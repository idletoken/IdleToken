/* IdleToken platform Agent — the cluster-side end of the marketplace platform
 * (docs/platform-design.md; wire protocol spec = platform/packages/gateway/
 * tools/mock-provider.ts, platform side = src/crypto/sealed-transport.ts).
 *
 * Runs ON THE COORDINATOR NODE, next to idletoken-coord. It is the provider-side
 * plaintext window of the platform↔provider envelope-encryption leg: it holds
 * the provider's X25519 secret key, opens sealed InferenceRequests from the
 * platform, forwards the plaintext to the local coord HTTP API over LOOPBACK
 * (127.0.0.1 — plaintext never leaves this host), then seals the reply back
 * to the platform's public key. Architectural twin of privacy_proxy.c, aimed
 * at the platform instead of at end consumers.
 *
 * Wire protocol (libsodium crypto_box_seal, base64 ORIGINAL variant):
 *   POST /infer  { sealed_request: b64(box_seal(json, my_pk)),
 *                  reply_to:       b64(platform_pk) }
 *   -> 200       { sealed_response: b64(box_seal(json({text}), reply_to)) }
 *   GET  /healthz -> 200 {"ok":true,"pubkey":"<b64>"}
 *   (any other)   -> 404
 * Malformed / unopenable envelope -> 400. Coord unreachable/bad -> 502.
 *
 * The sealed payload is the platform's normalized InferenceRequest
 * {model, messages:[{role,content}], maxTokens?}; we translate it to the
 * coord's OpenAI-shape /v1/chat/completions request, and wrap
 * choices[0].message.content as {"text":"..."} for the sealed reply.
 *
 * Registration/heartbeat (optional, when --platform is given): registers via
 * POST /providers {name, pubkey, endpoint} with a JWT bearer token, then
 * heartbeats POST /providers/:id/heartbeat every --heartbeat-secs. Usage
 * reporting is intentionally absent: metering/billing is platform-side
 * (server-side token counting, design §6) — nothing for the agent to send.
 *
 * Key management: same contract as privacy_proxy.c (--key-file loads or
 * generates a 32-byte raw secret key, 0600, public key printed).
 *
 * C only. No C++.
 */

#include "idletoken_privacy.h"
#include "idletoken_sodium_seal.h"
#include "idletoken_net.h"
#include "idletoken_http.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <pthread.h>
/* Winsock carries select()/fd_set itself; the POSIX socket headers do not
 * exist on Windows (same split as src/common/net.c). <io.h> is where MinGW
 * keeps open/read/write/close. */
#ifdef _WIN32
  #include <winsock2.h>
  #include <io.h>
#else
  #include <sys/select.h>
  #include <sys/socket.h>
  #include <unistd.h>
#endif

/* ======================================================================
 * sha256 -- compact self-contained implementation (FIPS 180-4). Used by the
 * KV prefix block-hash / Bloom contract (docs/kv-cache-design.md §6; it must
 * agree byte for byte with the platform's prefix-cache.ts). Our vendored
 * crypto only offers SHA-512 and BLAKE2b, and the contract specifies sha256,
 * hence this local copy.
 * ====================================================================== */

typedef struct { uint32_t h[8]; uint64_t nbits; uint8_t buf[64]; size_t off; } sha256_ctx;

static const uint32_t sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static void sha256_init(sha256_ctx *c) {
    static const uint32_t iv[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
    memcpy(c->h, iv, sizeof(iv));
    c->nbits = 0; c->off = 0;
}

static void sha256_block(sha256_ctx *c, const uint8_t *p) {
    uint32_t w[64], a, b, d, e, f, g, hh, t1, t2, s0, s1, ch, maj, hcc;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
               ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
#define ROR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
    for (int i = 16; i < 64; i++) {
        s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
        s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = c->h[0]; b = c->h[1]; hcc = c->h[2]; d = c->h[3];
    e = c->h[4]; f = c->h[5]; g = c->h[6]; hh = c->h[7];
    for (int i = 0; i < 64; i++) {
        s1 = ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25);
        ch = (e & f) ^ ((~e) & g);
        t1 = hh + s1 + ch + sha256_k[i] + w[i];
        s0 = ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22);
        maj = (a & b) ^ (a & hcc) ^ (b & hcc);
        t2 = s0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = hcc; hcc = b; b = a; a = t1 + t2;
    }
#undef ROR
    c->h[0] += a; c->h[1] += b; c->h[2] += hcc; c->h[3] += d;
    c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += hh;
}

static void sha256_update(sha256_ctx *c, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    c->nbits += (uint64_t)n * 8;
    while (n > 0) {
        size_t take = 64 - c->off;
        if (take > n) take = n;
        memcpy(c->buf + c->off, p, take);
        c->off += take; p += take; n -= take;
        if (c->off == 64) { sha256_block(c, c->buf); c->off = 0; }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    uint64_t nbits = c->nbits;
    uint8_t pad = 0x80;
    sha256_update(c, &pad, 1);
    uint8_t z = 0;
    while (c->off != 56) sha256_update(c, &z, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(nbits >> (56 - i * 8));
    c->nbits = nbits;  /* update modified nbits; the length field needs the original */
    sha256_update(c, lenb, 8);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(c->h[i] >> 24);
        out[i*4+1] = (uint8_t)(c->h[i] >> 16);
        out[i*4+2] = (uint8_t)(c->h[i] >> 8);
        out[i*4+3] = (uint8_t)(c->h[i]);
    }
}

/* ======================================================================
 * KV prefix cache state (engine half of P2). After unsealing, the agent sees
 * plaintext, so it follows the platform contract (prefix-cache.ts): render the
 * messages canonically -> split into 1 KB blocks -> salted chained sha256 ->
 * block hashes as hex. It periodically POSTs a Bloom filter of "prefixes this
 * machine's coordinator probably still holds" to the platform (cache-state),
 * which routes by affinity on that basis.
 *
 * Where the truth ends, stated plainly: the coordinator keeps exactly one live
 * session history, and the agent approximates it with "the last session served
 * successfully". If the coordinator restarts or another session displaces it,
 * the Bloom filter reports a false positive -- harmless: routing there simply
 * misses, the work is done in full and no discount applies (a discount needs
 * the coordinator's own cache_hit).
 * Note: \uXXXX escapes are not decoded yet (an extractor limitation), so a
 * message containing rare control characters mismatches conservatively -- a
 * little less saving, no effect on correctness.
 * ====================================================================== */

#define PFX_BLOCK_BYTES 1024
#define PFX_MAX_BLOCKS  256
#define PFX_BLOOM_BYTES 4096   /* 32768 bits, k=4: negligible FPR at a few dozen blocks */

/* Helpers defined further down, used early by this section. */
static char *b64_encode(const uint8_t *data, size_t n);
static uint8_t *http_post_json(const char *addr, const char *path, const char *jwt,
                               const uint8_t *body, size_t body_len,
                               int *out_status, size_t *out_len, int timeout_secs);
static uint8_t *http_get_json(const char *addr, const char *path,
                              int *out_status, size_t *out_len, int timeout_secs);

static struct {
    char hashes[PFX_MAX_BLOCKS][65];   /* hex block-hash chain of the live session */
    int  n;
    int  dirty;                        /* new state waiting to be POSTed */
} g_prefix;
/* Now that the agent is concurrent, g_prefix is its only cross-thread mutable
 * state: inference threads write it (once per request) and the heartbeat and
 * reporting paths read it. The lock is deliberately scoped to guard this struct
 * alone -- inference itself (sealing, the coordinator round trip) lives on each
 * thread's own stack and must never enter the critical section, or concurrency
 * would buy us nothing. */
static pthread_mutex_t g_prefix_mu = PTHREAD_MUTEX_INITIALIZER;

/* Render the messages canonically -- identical to renderMessages in
 * prefix-cache.ts: role + "\n" + content, with "\n\x1e" between messages --
 * then chain-hash the blocks. msgs_tok is the raw JSON array token; role and
 * content are extracted and unescaped object by object. Results are written to
 * out (at most PFX_MAX_BLOCKS hex strings); returns the number of blocks. */
static int prefix_hash_messages(const char *msgs_tok, size_t msgs_len,
                                char out[][65], int max_blocks) {
    /* 1. Render into a heap buffer. */
    size_t cap = msgs_len + 64, len = 0;
    char *r = malloc(cap);
    if (!r) return 0;
    size_t p = 0;
    int first = 1;
    while (p < msgs_len) {
        while (p < msgs_len && msgs_tok[p] != '{' && msgs_tok[p] != ']') p++;
        if (p >= msgs_len || msgs_tok[p] == ']') break;
        size_t o0 = p;
        int depth = 0, in_str = 0, esc = 0;
        while (p < msgs_len) {
            char ch = msgs_tok[p];
            if (esc)             { esc = 0; }
            else if (ch == '\\') { esc = 1; }
            else if (in_str)     { if (ch == '"') in_str = 0; }
            else if (ch == '"')  { in_str = 1; }
            else if (ch == '{')  { depth++; }
            else if (ch == '}')  { if (--depth == 0) { p++; break; } }
            p++;
        }
        size_t olen = p - o0;
        char role[32] = "";
        char *content = malloc(olen + 1);
        if (!content) { free(r); return 0; }
        content[0] = '\0';
        idletoken_http_json_extract_str(msgs_tok + o0, olen, "role", role, sizeof(role));
        if (idletoken_http_json_extract_str(msgs_tok + o0, olen, "content",
                                         content, olen + 1) != 0)
            idletoken_http_json_extract_str(msgs_tok + o0, olen, "text", content, olen + 1);
        if (role[0] && content[0]) {
            size_t need = strlen(role) + 1 + strlen(content) + 2;
            if (len + need + 1 > cap) {
                cap = (len + need + 1) * 2;
                char *nr = realloc(r, cap);
                if (!nr) { free(content); free(r); return 0; }
                r = nr;
            }
            if (!first) { r[len++] = '\n'; r[len++] = '\x1e'; }
            first = 0;
            len += (size_t)snprintf(r + len, cap - len, "%s\n%s", role, content);
        }
        free(content);
    }

    /* 2. Chained block hashes: h_0 = sha256(salt), h_i = sha256(h_{i-1} || block_i) */
    const char *salt = getenv("IDLETOKEN_PREFIX_SALT");
    if (!salt || !salt[0]) salt = "idletoken-prefix-v1";
    int n_blocks = (int)(len / PFX_BLOCK_BYTES);
    if (n_blocks > max_blocks) n_blocks = max_blocks;
    uint8_t prev[32];
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, salt, strlen(salt));
    sha256_final(&c, prev);
    for (int i = 0; i < n_blocks; i++) {
        sha256_init(&c);
        sha256_update(&c, prev, 32);
        sha256_update(&c, r + (size_t)i * PFX_BLOCK_BYTES, PFX_BLOCK_BYTES);
        sha256_final(&c, prev);
        for (int j = 0; j < 32; j++)
            snprintf(out[i] + j * 2, 3, "%02x", prev[j]);
    }
    free(r);
    return n_blocks;
}

/* Bloom bit positions, k=4: the first 4 bytes of sha256(hex || ASCII j), big
 * endian, mod mBits -- exactly as the contract specifies. */
static void bloom_add_hex(uint8_t *bloom, size_t bloom_bytes, const char *hex) {
    uint32_t mbits = (uint32_t)bloom_bytes * 8;
    for (int j = 0; j < 4; j++) {
        char dj = (char)('0' + j);
        uint8_t d[32];
        sha256_ctx c;
        sha256_init(&c);
        sha256_update(&c, hex, strlen(hex));
        sha256_update(&c, &dj, 1);
        sha256_final(&c, d);
        uint32_t posn = (((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
                         ((uint32_t)d[2] << 8) | d[3]) % mbits;
        bloom[posn >> 3] |= (uint8_t)(1u << (posn & 7));
    }
}

/* --selftest: standard sha256 vectors plus internal consistency of the prefix
 * hashes and the Bloom filter (no network, no keys). The authoritative check
 * that the two languages agree byte for byte is the gateway's platform-agent
 * e2e, where the TypeScript side hashes the same messages with prefix-cache.ts
 * and matches the Bloom filter this agent reports. */
static int agent_selftest(void) {
    int fails = 0;
#define ST(cond, name) do { \
        if (cond) fprintf(stderr, "selftest PASS %s\n", name); \
        else      { fprintf(stderr, "selftest FAIL %s\n", name); fails++; } \
    } while (0)
    {   /* FIPS 180-4 vector: sha256("abc") */
        uint8_t d[32]; char hex[65];
        sha256_ctx c; sha256_init(&c); sha256_update(&c, "abc", 3); sha256_final(&c, d);
        for (int i = 0; i < 32; i++) snprintf(hex + i * 2, 3, "%02x", d[i]);
        ST(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
           "sha256 vector abc");
    }
    {   /* Empty-string vector. */
        uint8_t d[32]; char hex[65];
        sha256_ctx c; sha256_init(&c); sha256_final(&c, d);
        for (int i = 0; i < 32; i++) snprintf(hex + i * 2, 3, "%02x", d[i]);
        ST(!strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
           "sha256 vector empty");
    }
    {   /* Prefix hashes: same input, same output; a shared prefix keeps the
         * head of the chain identical and only diverges after the fork. The
         * fork must fall inside the range covered by whole blocks (a trailing
         * partial block is discarded and cannot be compared):
         * render = "user\n" (5 B) + 1200 x + fork 4 B + 1000 y = 2209 B ->
         * 2 blocks, with the fork inside block 2 (byte 1205). */
        char pre[1201], tail[1001], msgs1[4096], msgs2[4096];
        memset(pre, 'x', sizeof(pre) - 1); pre[sizeof(pre) - 1] = 0;
        memset(tail, 'y', sizeof(tail) - 1); tail[sizeof(tail) - 1] = 0;
        snprintf(msgs1, sizeof(msgs1), "[{\"role\":\"user\",\"content\":\"%sAAAA%s\"}]", pre, tail);
        snprintf(msgs2, sizeof(msgs2), "[{\"role\":\"user\",\"content\":\"%sBBBB%s\"}]", pre, tail);
        char h1[PFX_MAX_BLOCKS][65], h1b[PFX_MAX_BLOCKS][65], h2[PFX_MAX_BLOCKS][65];
        int n1  = prefix_hash_messages(msgs1, strlen(msgs1), h1,  PFX_MAX_BLOCKS);
        int n1b = prefix_hash_messages(msgs1, strlen(msgs1), h1b, PFX_MAX_BLOCKS);
        int n2  = prefix_hash_messages(msgs2, strlen(msgs2), h2,  PFX_MAX_BLOCKS);
        ST(n1 == 2 && n1b == 2 && n2 == 2, "prefix: 2209B render -> 2 blocks");
        ST(!strcmp(h1[0], h1b[0]) && !strcmp(h1[1], h1b[1]), "prefix: deterministic");
        ST(!strcmp(h1[0], h2[0]), "prefix: common first block matches");
        ST(strcmp(h1[1], h2[1]) != 0, "prefix: divergent tail differs");
        /* Bloom consistency: what was added is found again, via the same bit positions. */
        uint8_t bloom[PFX_BLOOM_BYTES];
        memset(bloom, 0, sizeof(bloom));
        bloom_add_hex(bloom, sizeof(bloom), h1[0]);
        int all_set = 1;
        for (int j = 0; j < 4 && all_set; j++) {
            char dj = (char)('0' + j); uint8_t d[32];
            sha256_ctx c; sha256_init(&c);
            sha256_update(&c, h1[0], strlen(h1[0]));
            sha256_update(&c, &dj, 1); sha256_final(&c, d);
            uint32_t posn = (((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
                             ((uint32_t)d[2] << 8) | d[3]) % (PFX_BLOOM_BYTES * 8);
            all_set = (bloom[posn >> 3] >> (posn & 7)) & 1;
        }
        ST(all_set, "bloom: added hash queryable");
    }
#undef ST
    fprintf(stderr, "selftest: %s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}

/* POST /providers/:id/cache-state {bloom}. Skipped when there is nothing to
 * report. Returns 0 on success. */
static int platform_post_cache_state(const char *platform_addr, const char *jwt,
                                     const char *provider_id) {
    /* Fold g_prefix into the Bloom filter under the lock, then let go: the HTTP
     * POST that follows can take seconds, and holding the lock across network
     * I/O would stall the inference threads with it. */
    uint8_t bloom[PFX_BLOOM_BYTES] = {0};
    pthread_mutex_lock(&g_prefix_mu);
    if (g_prefix.n <= 0) { g_prefix.dirty = 0; pthread_mutex_unlock(&g_prefix_mu); return 0; }
    for (int i = 0; i < g_prefix.n; i++)
        bloom_add_hex(bloom, sizeof(bloom), g_prefix.hashes[i]);
    pthread_mutex_unlock(&g_prefix_mu);
    char *b64 = b64_encode(bloom, sizeof(bloom));
    if (!b64) return -1;
    size_t body_cap = strlen(b64) + 96;
    char *body = malloc(body_cap);
    if (!body) { free(b64); return -1; }
    int bl = snprintf(body, body_cap, "{\"bloom\":\"%s\",\"blockTokens\":0}", b64);
    free(b64);
    char path[256];
    snprintf(path, sizeof(path), "/providers/%s/cache-state", provider_id);
    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_post_json(platform_addr, path, jwt,
                                   (const uint8_t *)body, (size_t)bl, &status, &rlen, 15);
    free(body);
    if (!resp) return -1;
    free(resp);
    if (status >= 200 && status < 300) {
        pthread_mutex_lock(&g_prefix_mu);
        g_prefix.dirty = 0;
        pthread_mutex_unlock(&g_prefix_mu);
        return 0;
    }
    return -1;
}

/* ======================================================================
 * base64 — standard alphabet with '=' padding (libsodium ORIGINAL variant).
 * ====================================================================== */

static const char b64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Encode `n` bytes; returns malloc'd NUL-terminated string (caller frees). */
static char *b64_encode(const uint8_t *in, size_t n) {
    size_t olen = 4 * ((n + 2) / 3);
    char *out = malloc(olen + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < n) v |= (uint32_t)in[i + 1] << 8;
        if (i + 2 < n) v |= (uint32_t)in[i + 2];
        out[o++] = b64_alphabet[(v >> 18) & 63];
        out[o++] = b64_alphabet[(v >> 12) & 63];
        out[o++] = (i + 1 < n) ? b64_alphabet[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? b64_alphabet[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

/* Decode a padded base64 string. Returns malloc'd bytes and sets *out_len;
 * NULL on any invalid character / bad length (treated as malformed → 400). */
static uint8_t *b64_decode(const char *s, size_t slen, size_t *out_len) {
    if (slen == 0 || slen % 4 != 0) return NULL;
    int8_t rev[256];
    memset(rev, -1, sizeof(rev));
    for (int i = 0; i < 64; i++) rev[(uint8_t)b64_alphabet[i]] = (int8_t)i;

    size_t pad = 0;
    if (s[slen - 1] == '=') pad++;
    if (s[slen - 2] == '=') pad++;
    size_t olen = slen / 4 * 3 - pad;
    uint8_t *out = malloc(olen ? olen : 1);
    if (!out) return NULL;

    size_t o = 0;
    for (size_t i = 0; i < slen; i += 4) {
        int last = (i + 4 == slen);
        uint32_t v = 0;
        int npad = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i + k];
            if (c == '=') {
                /* '=' only allowed as the final group's tail: "xx==" / "xxx=" */
                if (!last || k < 2 || (k == 2 && s[i + 3] != '=')) { free(out); return NULL; }
                npad = 4 - k;
                v <<= 6 * npad;
                break;
            }
            int8_t d = rev[(uint8_t)c];
            if (d < 0) { free(out); return NULL; }
            v = (v << 6) | (uint32_t)d;
        }
        int nout = 3 - npad;
        for (int k = 0; k < nout; k++) {
            if (o >= olen) { free(out); return NULL; }
            out[o++] = (uint8_t)(v >> (16 - 8 * k));
        }
    }
    if (o != olen) { free(out); return NULL; }
    if (out_len) *out_len = olen;
    return out;
}

/* ======================================================================
 * tiny JSON pickers — same spirit as coord_main.c / idletoken_http.h: fixed
 * shapes, no general parser. Raw "tokens" keep JSON escapes intact so they
 * can be re-embedded verbatim between quotes.
 * ====================================================================== */

/* Find `"key"` used as an object key (followed by ':') from `from`;
 * returns the index just past the ':' or -1. */
static long json_key_colon(const char *json, size_t len, const char *key) {
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen + 2 <= len; i++) {
        if (json[i] != '"') continue;
        if (memcmp(json + i + 1, key, klen) != 0) continue;
        if (json[i + 1 + klen] != '"') continue;
        size_t p = i + klen + 2;
        while (p < len && (json[p] == ' ' || json[p] == '\t' ||
                           json[p] == '\n' || json[p] == '\r')) p++;
        if (p < len && json[p] == ':') return (long)(p + 1);
    }
    return -1;
}

/* Extract the raw string token of `"key": "...."` — a pointer INTO json
 * plus a length, escapes untouched. Returns 0 / -1. */
static int json_str_token(const char *json, size_t len, const char *key,
                          const char **tok, size_t *tok_len) {
    long p = json_key_colon(json, len, key);
    if (p < 0) return -1;
    size_t i = (size_t)p;
    while (i < len && (json[i] == ' ' || json[i] == '\t' ||
                       json[i] == '\n' || json[i] == '\r')) i++;
    if (i >= len || json[i] != '"') return -1;
    i++;
    size_t start = i;
    while (i < len) {
        if (json[i] == '\\') { i += 2; continue; }   /* skip escape pair */
        if (json[i] == '"') {
            *tok = json + start;
            *tok_len = i - start;
            return 0;
        }
        i++;
    }
    return -1;
}

/* Like json_str_token but returns a malloc'd NUL-terminated copy. */
static char *json_str_dup(const char *json, size_t len, const char *key) {
    const char *tok; size_t tlen;
    if (json_str_token(json, len, key, &tok, &tlen) != 0) return NULL;
    char *s = malloc(tlen + 1);
    if (!s) return NULL;
    memcpy(s, tok, tlen);
    s[tlen] = '\0';
    return s;
}

/* Extract the raw `[...]` token of an array value (balanced brackets,
 * string-literal aware). Returns 0 / -1. */
static int json_array_token(const char *json, size_t len, const char *key,
                            const char **tok, size_t *tok_len) {
    long p = json_key_colon(json, len, key);
    if (p < 0) return -1;
    size_t i = (size_t)p;
    while (i < len && (json[i] == ' ' || json[i] == '\t' ||
                       json[i] == '\n' || json[i] == '\r')) i++;
    if (i >= len || json[i] != '[') return -1;
    size_t start = i;
    int depth = 0, in_str = 0;
    for (; i < len; i++) {
        char c = json[i];
        if (in_str) {
            if (c == '\\') { i++; continue; }
            if (c == '"') in_str = 0;
            continue;
        }
        if (c == '"') { in_str = 1; continue; }
        if (c == '[' || c == '{') depth++;
        else if (c == ']' || c == '}') {
            depth--;
            if (depth == 0) {
                *tok = json + start;
                *tok_len = i - start + 1;
                return 0;
            }
        }
    }
    return -1;
}

/* Extract `"key": N` as an int; `dflt` if absent (coord_main.c pattern). */
static int json_int_field(const char *json, size_t len, const char *key, int dflt) {
    long p = json_key_colon(json, len, key);
    if (p < 0) return dflt;
    size_t i = (size_t)p;
    while (i < len && (json[i] == ' ' || json[i] == '\t' ||
                       json[i] == '\n' || json[i] == '\r')) i++;
    int sign = 1;
    if (i < len && json[i] == '-') { sign = -1; i++; }
    int v = 0, digits = 0;
    while (i < len && json[i] >= '0' && json[i] <= '9') {
        v = v * 10 + (json[i] - '0');
        i++; digits++;
    }
    return digits ? sign * v : dflt;
}

/* ======================================================================
 * HTTP client — privacy_proxy's http_post_json pattern, extended with the
 * status code (we must tell coord 200 from coord 4xx/5xx) and an optional
 * Authorization: Bearer header (for the platform control plane).
 * ====================================================================== */

/* POST `body` as JSON to addr("host:port")+path. Returns malloc'd response
 * body (caller frees), sets *out_len and *out_status; NULL on I/O error.
 * timeout_secs > 0 arms SO_RCVTIMEO/SO_SNDTIMEO so a dead platform link can't
 * hang a relay long-poll forever (0 = block indefinitely, e.g. slow coord
 * inference where the caller owns pacing). */
static uint8_t *http_request_json(const char *method,
                                  const char *addr, const char *path,
                                  const char *bearer,
                                  const uint8_t *body, size_t body_len,
                                  int *out_status, size_t *out_len,
                                  int timeout_secs) {
    int fd = idletoken_connect_tcp(addr);
    if (fd < 0) return NULL;
    if (timeout_secs > 0) {
        /* Winsock's SO_RCVTIMEO/SO_SNDTIMEO take a DWORD of milliseconds, not
         * a struct timeval — passing the timeval silently sets a nonsense
         * timeout (its first 4 bytes read as milliseconds). */
#ifdef _WIN32
        DWORD to = (DWORD)timeout_secs * 1000u;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof(to));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&to, sizeof(to));
#else
        struct timeval to = { timeout_secs, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof(to));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &to, sizeof(to));
#endif
    }

    char auth[1024] = "";
    if (bearer) {
        int an = snprintf(auth, sizeof(auth), "Authorization: Bearer %s\r\n", bearer);
        if (an < 0 || (size_t)an >= sizeof(auth)) { close(fd); return NULL; }
    }
    char head[1536];
    int hn = snprintf(head, sizeof(head),
                      "%s %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "%s"
                      "Connection: close\r\n\r\n",
                      method, path, addr, body_len, auth);
    if (hn < 0 || (size_t)hn >= sizeof(head)) { close(fd); return NULL; }
    if (idletoken_sendall(fd, head, (size_t)hn) < 0 ||
        (body_len && idletoken_sendall(fd, body, body_len) < 0)) {
        close(fd); return NULL;
    }

    /* Read the whole response until EOF (Connection: close). */
    size_t cap = 8192, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) { close(fd); return NULL; }
    for (;;) {
        if (len + 4096 > cap) {
            size_t ncap = cap * 2;
            uint8_t *nb = realloc(buf, ncap);
            if (!nb) { free(buf); close(fd); return NULL; }
            buf = nb; cap = ncap;
        }
        ssize_t r = read(fd, buf + len, 4096);
        if (r > 0) { len += (size_t)r; continue; }
        if (r < 0 && errno == EINTR) continue;
        break;
    }
    close(fd);

    /* Status line: "HTTP/1.x NNN ..." */
    int status = 0;
    if (len > 12 && memcmp(buf, "HTTP/1.", 7) == 0)
        status = (buf[9]-'0')*100 + (buf[10]-'0')*10 + (buf[11]-'0');
    if (status < 100) { free(buf); return NULL; }

    /* Split headers/body at CRLFCRLF; return the body only. */
    uint8_t *sep = NULL;
    for (size_t i = 0; i + 3 < len; i++) {
        if (buf[i]=='\r' && buf[i+1]=='\n' && buf[i+2]=='\r' && buf[i+3]=='\n') {
            sep = buf + i + 4; break;
        }
    }
    if (!sep) { free(buf); return NULL; }
    size_t blen = len - (size_t)(sep - buf);
    uint8_t *body_out = malloc(blen ? blen : 1);
    if (!body_out) { free(buf); return NULL; }
    memcpy(body_out, sep, blen);
    free(buf);
    if (out_status) *out_status = status;
    if (out_len) *out_len = blen;
    return body_out;
}

/* "http://host:port[/...]" or "host:port" → "host:port" into `out`.
 * Any path suffix is dropped (both coord and platform APIs hang off /). */
static int url_to_addr(const char *url, char *out, size_t cap) {
    const char *p = url;
    if (!strncmp(p, "http://", 7)) p += 7;
    else if (!strncmp(p, "https://", 8)) {
        fprintf(stderr, "platform-agent: https:// upstreams are not supported (no TLS client); "
                        "terminate TLS in front or use http\n");
        return -1;
    }
    size_t n = strcspn(p, "/");
    if (n == 0 || n >= cap) return -1;
    memcpy(out, p, n);
    out[n] = '\0';
    if (!strchr(out, ':')) {
        if (n + 3 >= cap) return -1;
        memcpy(out + n, ":80", 4);   /* default http port */
    }
    return 0;
}

/* ======================================================================
 * platform registration + heartbeat (control plane, JWT bearer)
 * ====================================================================== */

/* Minimal JSON string escaper for values WE emit (name, model). */
static void json_escape_into(char *dst, size_t cap, const char *src) {
    size_t o = 0;
    for (const char *p = src; *p && o + 6 < cap; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { dst[o++] = '\\'; dst[o++] = (char)c; }
        else if (c < 0x20) o += (size_t)snprintf(dst + o, cap - o, "\\u%04x", c);
        else dst[o++] = (char)c;
    }
    dst[o] = '\0';
}

/* POST /providers → provider id (malloc'd) or NULL. relay != 0 registers the
 * outbound-only transport: no endpoint at all (the platform never dials us —
 * zero home-side configuration, integration-plan §4). */
static char *platform_register(const char *platform_addr, const char *jwt,
                               const char *name, const char *pubkey_b64,
                               const char *endpoint, int relay,
                               const char *model, const char *quant) {
    char name_esc[256], ep_esc[512], model_esc[128], quant_esc[64];
    json_escape_into(name_esc, sizeof(name_esc), name);
    json_escape_into(model_esc, sizeof(model_esc), model);
    json_escape_into(quant_esc, sizeof(quant_esc), quant ? quant : "");
    /* Precision-aware capability (small-model-design §6.3): when a quant is
     * loaded, advertise capacity.models:[{model,quant}] so the platform only
     * routes matching-precision requests here; else keep the single-model form
     * the router treats as "any precision". */
    char cap[256];
    if (quant && quant[0])
        snprintf(cap, sizeof(cap),
                 "\"capacity\":{\"models\":[{\"model\":\"%s\",\"quant\":\"%s\"}],\"tiers\":[1]}",
                 model_esc, quant_esc);
    else
        snprintf(cap, sizeof(cap), "\"capacity\":{\"model\":\"%s\",\"tiers\":[1]}", model_esc);
    char body[1024];
    int bl;
    if (relay) {
        bl = snprintf(body, sizeof(body),
                      "{\"name\":\"%s\",\"pubkey\":\"%s\",\"transport\":\"relay\",%s}",
                      name_esc, pubkey_b64, cap);
    } else {
        json_escape_into(ep_esc, sizeof(ep_esc), endpoint);
        bl = snprintf(body, sizeof(body),
                      "{\"name\":\"%s\",\"pubkey\":\"%s\",\"endpoint\":\"%s\",%s}",
                      name_esc, pubkey_b64, ep_esc, cap);
    }
    if (bl < 0 || (size_t)bl >= sizeof(body)) return NULL;

    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_post_json(platform_addr, "/providers", jwt,
                                   (const uint8_t *)body, (size_t)bl, &status, &rlen, 30);
    if (!resp) {
        fprintf(stderr, "platform-agent: register: platform %s unreachable\n", platform_addr);
        return NULL;
    }
    if (status != 200 && status != 201) {
        fprintf(stderr, "platform-agent: register: platform said %d: %.*s\n",
                status, (int)(rlen > 512 ? 512 : rlen), (const char *)resp);
        free(resp);
        return NULL;
    }
    char *id = json_str_dup((const char *)resp, rlen, "id");
    free(resp);
    if (!id) fprintf(stderr, "platform-agent: register: no id in response\n");
    return id;
}

/* Two thin per-method wrappers; body and timeout semantics are shared with the
 * implementation above. */
static uint8_t *http_post_json(const char *addr, const char *path, const char *bearer,
                               const uint8_t *body, size_t body_len,
                               int *out_status, size_t *out_len, int timeout_secs) {
    return http_request_json("POST", addr, path, bearer, body, body_len,
                             out_status, out_len, timeout_secs);
}

static uint8_t *http_get_json(const char *addr, const char *path,
                              int *out_status, size_t *out_len, int timeout_secs) {
    return http_request_json("GET", addr, path, NULL, NULL, 0,
                             out_status, out_len, timeout_secs);
}

/**
 * Carry the coordinator's live load along with the heartbeat (scheduler-design
 * §7, contract 1).
 *
 * The platform's dispatch cost function needs "how long would a request wait on
 * this machine" = queue depth x service time. Until now it could only infer
 * that from the **end-to-end round trip** it observed itself, which mixes in
 * network, sealing and queueing, producing a service time that is both too
 * large and jittery. The coordinator already knows the exact numbers
 * (`avg_service_ms` / `queue_depth` / `seq_slots` from `/v1/stats`), so we
 * simply bring them along.
 *
 * They are keyed per context size (`{"<ctx_size>": v}`) because the platform
 * buckets by context tier, and both slot count and service time vary with
 * context. The coordinator reports the numbers for **its current tier**, so it
 * fills only that one entry; on the platform side `numFromCtxMap` is allowed to
 * borrow from a larger tier, so smaller contexts pick this value up naturally.
 *
 * If the coordinator cannot be probed (no --http, not warmed up yet) the
 * heartbeat still goes out with an empty body, degrading to the behaviour that
 * preceded this change.
 */
static int coord_stats_json(const char *coord_addr, char *out, size_t out_cap) {
    out[0] = '\0';
    if (!coord_addr || !coord_addr[0]) return -1;
    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_get_json(coord_addr, "/v1/stats", &status, &rlen, 5);
    if (!resp) return -1;
    if (status < 200 || status >= 300) { free(resp); return -1; }
    const char *j = (const char *)resp;
    int ctx     = json_int_field(j, rlen, "ctx_size", 0);
    /* WARNING: what we report must be the **effective concurrency** (the
     * coordinator's `concurrency`), not the KV slot count (`seq_slots`). They
     * are not the same thing: slots are "how many independent KV caches fit",
     * concurrency is "how many will actually be in flight at once". A
     * serially-executing coordinator may have 4 slots yet run one at a time --
     * reporting slots would let the platform believe it can run 4 in parallel,
     * underestimate the wait by 4x, and over-dispatch.
     * An older coordinator has no `concurrency` field, so fall back to
     * seq_slots (the behaviour that preceded this change). */
    int conc    = json_int_field(j, rlen, "concurrency", 0);
    int slots   = conc > 0 ? conc : json_int_field(j, rlen, "seq_slots", 0);
    int qdepth  = json_int_field(j, rlen, "queue_depth", -1);
    int svc_ms  = json_int_field(j, rlen, "avg_service_ms", 0);
    /* TTFT (time to first token). The platform's **admission decision** is
     * bounded by a TTFT budget, whereas avg_service_ms is the duration of a
     * whole request -- an order of magnitude apart. An older coordinator does
     * not have this field, so we omit it and the platform degrades to bounding
     * queueing alone. That is more honest than substituting service time, which
     * would reject every real machine. */
    int ttft_ms = json_int_field(j, rlen, "avg_ttft_ms", 0);
    free(resp);
    if (ctx <= 0 || qdepth < 0) return -1;   /* older coordinator lacks these fields: report nothing rather than guess */
    char ttft_field[64] = "";
    if (ttft_ms > 0)
        snprintf(ttft_field, sizeof(ttft_field), "\"avg_ttft_ms\":{\"%d\":%d},", ctx, ttft_ms);
    int n = snprintf(out, out_cap,
        "{\"seq_slots_by_ctx\":{\"%d\":%d},\"avg_service_ms\":{\"%d\":%d},%s"
        "\"queue_depth\":%d,\"max_ctx_tokens\":%d}",
        ctx, slots > 0 ? slots : 1, ctx, svc_ms > 0 ? svc_ms : 0, ttft_field, qdepth, ctx);
    if (n < 0 || (size_t)n >= out_cap) { out[0] = '\0'; return -1; }
    return 0;
}

/* POST /providers/:id/heartbeat. Returns 0 on 2xx. */
static int platform_heartbeat(const char *platform_addr, const char *jwt,
                              const char *provider_id, const char *coord_addr) {
    char path[256];
    int pn = snprintf(path, sizeof(path), "/providers/%s/heartbeat", provider_id);
    if (pn < 0 || (size_t)pn >= sizeof(path)) return -1;
    char body[512] = "{}";
    char stats[400];
    if (coord_stats_json(coord_addr, stats, sizeof(stats)) == 0) {
        int bn = snprintf(body, sizeof(body), "{\"capacity\":%s}", stats);
        if (bn < 0 || (size_t)bn >= sizeof(body)) snprintf(body, sizeof(body), "{}");
    }
    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_post_json(platform_addr, path, jwt,
                                   (const uint8_t *)body, strlen(body), &status, &rlen, 15);
    if (!resp) return -1;
    free(resp);
    return (status >= 200 && status < 300) ? 0 : -1;
}

/* ======================================================================
 * relay mode — the "reverse long-lived connection" (integration-plan §4).
 *
 * Instead of listening for the platform, the agent dials OUT:
 *   POST /providers/:id/relay/poll {wait_ms}   (long poll; also heartbeats)
 *     -> {job_id, sealed_request, reply_to}  or  {job_id:null} on idle timeout
 *   POST /providers/:id/relay/result {job_id, sealed_response | error}
 * The sealed envelope is byte-identical to the direct POST /infer body, so
 * jobs go through the exact same process_sealed() path. Home side needs no
 * open ports, no router config; robustness = reconnect with backoff.
 * ====================================================================== */

#define RELAY_WAIT_MS 25000            /* server default; must stay < proxies' idle cuts */
#define RELAY_BACKOFF_MAX_SECS 30

/* Shared sealed-envelope data path (defined below with the /infer handler). */
static int process_sealed(const idletoken_keypair *node, const char *coord_addr,
                          const char *json, size_t json_len,
                          char **out_b64, int *err_status, const char **err_msg);

/* POST the job result (success or error). Best effort: on failure the job
 * simply expires platform-side and routing fails over. Returns 0 on 2xx. */
static int relay_post_result(const char *platform_addr, const char *jwt,
                             const char *provider_id, const char *job_id,
                             const char *sealed_b64 /* NULL on error */,
                             const char *err_msg    /* used when sealed_b64 == NULL */) {
    char path[256];
    int pn = snprintf(path, sizeof(path), "/providers/%s/relay/result", provider_id);
    if (pn < 0 || (size_t)pn >= sizeof(path)) return -1;

    size_t cap = (sealed_b64 ? strlen(sealed_b64) : strlen(err_msg)) + strlen(job_id) + 64;
    char *body = malloc(cap);
    if (!body) return -1;
    int bl;
    if (sealed_b64) {
        bl = snprintf(body, cap, "{\"job_id\":\"%s\",\"sealed_response\":\"%s\"}",
                      job_id, sealed_b64);
    } else {
        char msg_esc[256];
        json_escape_into(msg_esc, sizeof(msg_esc), err_msg);
        bl = snprintf(body, cap, "{\"job_id\":\"%s\",\"error\":\"%s\"}", job_id, msg_esc);
    }
    if (bl < 0 || (size_t)bl >= cap) { free(body); return -1; }

    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_post_json(platform_addr, path, jwt,
                                   (const uint8_t *)body, (size_t)bl, &status, &rlen, 30);
    free(body);
    if (!resp) return -1;
    free(resp);
    return (status >= 200 && status < 300) ? 0 : -1;
}

/* The relay main loop: long-poll → process job → post result → repeat.
 * Never returns. Network/HTTP failures back off 1/2/4/8/30s and reconnect —
 * that retry ladder IS the robustness of the reverse connection. */
static void relay_loop(const idletoken_keypair *node, const char *coord_addr,
                       const char *platform_addr, const char *jwt,
                       const char *provider_id) {
    char path[256];
    snprintf(path, sizeof(path), "/providers/%s/relay/poll", provider_id);
    char poll_body[64];
    int pbl = snprintf(poll_body, sizeof(poll_body), "{\"wait_ms\":%d}", RELAY_WAIT_MS);
    int backoff = 1;

    for (;;) {
        int status = 0; size_t rlen = 0;
        /* Socket timeout comfortably above the server's hold time. */
        uint8_t *resp = http_post_json(platform_addr, path, jwt,
                                       (const uint8_t *)poll_body, (size_t)pbl,
                                       &status, &rlen, RELAY_WAIT_MS / 1000 + 15);
        if (!resp || status < 200 || status >= 300) {
            fprintf(stderr, "platform-agent: relay poll %s (http %d); retry in %ds\n",
                    resp ? "rejected" : "unreachable", status, backoff);
            free(resp);
            sleep((unsigned)backoff);
            backoff = backoff >= 8 ? RELAY_BACKOFF_MAX_SECS
                                   : backoff * 2;      /* 1,2,4,8,30,30,... */
            continue;
        }
        backoff = 1;   /* healthy round trip → reset the ladder */

        /* {"job_id":null} (idle timeout) → just poll again. */
        char *job_id = json_str_dup((const char *)resp, rlen, "job_id");
        if (!job_id) { free(resp); continue; }

        /* The poll response body IS the sealed envelope (same keys as the
         * direct /infer body) — feed it straight to the shared path. */
        char *sealed_b64 = NULL;
        int err_status = 500;
        const char *err_msg = "internal error";
        int rc = process_sealed(node, coord_addr, (const char *)resp, rlen,
                                &sealed_b64, &err_status, &err_msg);
        free(resp);
        fprintf(stderr, "platform-agent: relay infer job=%s -> %s\n",
                job_id, rc == 0 ? "sealed ok" : err_msg);

        if (relay_post_result(platform_addr, jwt, provider_id, job_id,
                              rc == 0 ? sealed_b64 : NULL, err_msg) != 0)
            fprintf(stderr, "platform-agent: relay result post failed for job=%s "
                            "(job will expire platform-side)\n", job_id);
        free(sealed_b64);
        free(job_id);

        /* KV cache state: report it in the gap between finishing a job and the
         * next poll (the poll is the heartbeat; cache-state is its own short
         * request, so its cadence is quantized by the poll period, <=25s --
         * good enough). */
        pthread_mutex_lock(&g_prefix_mu);
        const int pfx_dirty = g_prefix.dirty, pfx_n = g_prefix.n;
        pthread_mutex_unlock(&g_prefix_mu);
        if (pfx_dirty && platform_post_cache_state(platform_addr, jwt, provider_id) == 0)
            fprintf(stderr, "platform-agent: cache-state posted (%d blocks)\n", pfx_n);
    }
}

/* ======================================================================
 * the /infer handler — open envelope → coord → seal reply
 * ====================================================================== */

/* Free + wipe helper for plaintext buffers (wipe BEFORE free, always). */
static void wipe_free(void *p, size_t n) {
    if (!p) return;
    idletoken_secure_zero(p, n);
    free(p);
}

/* Process one sealed envelope: open → coord → seal reply. The envelope JSON
 * {sealed_request, reply_to} is EXACTLY the same shape in both transports —
 * the direct POST /infer body and the relay poll job — so this is the single
 * shared data path (handle_infer and the relay loop are thin wrappers).
 * `coord_addr` is "host:port" on loopback.
 * On success returns 0 and sets *out_b64 (malloc'd base64 sealed_response,
 * caller frees). On failure returns -1 and sets *err_status (HTTP-ish code:
 * 400 bad envelope / 500 internal / 502 coord) + *err_msg (static string). */
static int process_sealed(const idletoken_keypair *node, const char *coord_addr,
                          const char *json, size_t json_len,
                          char **out_b64, int *err_status, const char **err_msg) {
    *out_b64 = NULL;
#define FAIL(code, msg) do { *err_status = (code); *err_msg = (msg); return -1; } while (0)

    /* -- parse the envelope wrapper {sealed_request, reply_to} (b64) ------ */
    const char *sr_tok, *rt_tok; size_t sr_len, rt_len;
    if (json_str_token(json, json_len, "sealed_request", &sr_tok, &sr_len) != 0 ||
        json_str_token(json, json_len, "reply_to", &rt_tok, &rt_len) != 0 ||
        memchr(sr_tok, '\\', sr_len) || memchr(rt_tok, '\\', rt_len)) {
        FAIL(400, "sealed_request and reply_to (base64) required");
    }
    size_t sealed_len = 0, rtpk_len = 0;
    uint8_t *sealed = b64_decode(sr_tok, sr_len, &sealed_len);
    uint8_t *reply_to = b64_decode(rt_tok, rt_len, &rtpk_len);
    if (!sealed || !reply_to || rtpk_len != IDLETOKEN_PK_BYTES ||
        sealed_len < IDLETOKEN_SEAL_OVERHEAD) {
        free(sealed); free(reply_to);
        FAIL(400, "malformed base64 envelope");
    }

    /* -- open the envelope: the ONLY plaintext window on this box --------- */
    size_t plain_cap = sealed_len;   /* plaintext < ciphertext length */
    uint8_t *plain = malloc(plain_cap);
    if (!plain) { free(sealed); free(reply_to); FAIL(500, "oom"); }
    idletoken_mlock(plain, plain_cap);  /* keep the plaintext off swap */

    size_t plain_len = 0;
    idletoken_priv_rc rc = idletoken_sodium_seal_open(node, sealed, sealed_len,
                                                plain, plain_cap, &plain_len);
    free(sealed);
    if (rc != IDLETOKEN_PRIV_OK) {
        idletoken_secure_zero(plain, plain_cap);
        idletoken_munlock(plain, plain_cap); free(plain);
        free(reply_to);
        FAIL(400, "cannot open sealed request (wrong key or corrupt)");
    }

    /* -- translate InferenceRequest → OpenAI chat request ----------------- *
     * {model, messages, maxTokens?} → {model, messages, max_tokens?}.
     * The messages array is re-embedded verbatim (raw token), so nested
     * content survives untouched. */
    const char *model_tok = "dsv4-flash"; size_t model_len = 10;
    json_str_token((const char *)plain, plain_len, "model", &model_tok, &model_len);
    const char *msgs_tok; size_t msgs_len;
    if (json_array_token((const char *)plain, plain_len, "messages", &msgs_tok, &msgs_len) != 0) {
        idletoken_secure_zero(plain, plain_cap);
        idletoken_munlock(plain, plain_cap); free(plain);
        free(reply_to);
        FAIL(400, "opened request has no messages array");
    }
    int max_tokens = json_int_field((const char *)plain, plain_len, "maxTokens", -1);

    /* Contract hashes for the KV prefix: they must be computed into a staging
     * buffer while `plain` still exists (it is wiped moments from now), and are
     * committed as the "live session" state only after the whole round trip
     * succeeds (see the end of this function). */
    static char staged[PFX_MAX_BLOCKS][65];
    int staged_n = prefix_hash_messages(msgs_tok, msgs_len, staged, PFX_MAX_BLOCKS);

    size_t creq_cap = msgs_len + model_len + 96;
    char *creq = malloc(creq_cap);
    if (!creq) {
        idletoken_secure_zero(plain, plain_cap);
        idletoken_munlock(plain, plain_cap); free(plain);
        free(reply_to);
        FAIL(500, "oom");
    }
    idletoken_mlock(creq, creq_cap);    /* also plaintext */
    int cl;
    if (max_tokens > 0)
        cl = snprintf(creq, creq_cap, "{\"model\":\"%.*s\",\"messages\":%.*s,\"max_tokens\":%d}",
                      (int)model_len, model_tok, (int)msgs_len, msgs_tok, max_tokens);
    else
        cl = snprintf(creq, creq_cap, "{\"model\":\"%.*s\",\"messages\":%.*s}",
                      (int)model_len, model_tok, (int)msgs_len, msgs_tok);

    /* -- forward plaintext to coord over loopback ------------------------- *
     * Deliberately NO "stream":true here: the sealed envelope is a one-shot
     * roundtrip (seal → open → reply → seal), so the agent takes the coord's
     * complete response and seals it whole. The coord's own SSE streaming
     * (integration-plan 3.4) serves DIRECT LAN clients; cross-envelope
     * streaming needs a framed sealing protocol (per-frame seq + MAC) and is
     * deferred — the seam for it is the platform's SealedChannel interface
     * (gateway/src/crypto/sealed-transport.ts) + this single data path. */
    int cstatus = 0; size_t cresp_len = 0;
    uint8_t *cresp = NULL;
    if (cl > 0 && (size_t)cl < creq_cap)
        cresp = http_post_json(coord_addr, "/v1/chat/completions", NULL,
                               (const uint8_t *)creq, (size_t)cl, &cstatus, &cresp_len,
                               0 /* no timeout: real-model inference is slow by design */);
    /* plaintext request buffers are done — wipe immediately */
    idletoken_secure_zero(creq, creq_cap);
    idletoken_munlock(creq, creq_cap); free(creq);
    idletoken_secure_zero(plain, plain_cap);
    idletoken_munlock(plain, plain_cap); free(plain);

    if (!cresp || cstatus != 200) {
        if (cresp) wipe_free(cresp, cresp_len);
        free(reply_to);
        FAIL(502, "upstream coord unreachable or errored");
    }

    /* -- pick choices[0].message.content and wrap as {"text": "..."} ------ *
     * The raw string token (escapes intact) is re-embedded verbatim, so no
     * decode/re-encode round trip is needed. */
    const char *content_tok; size_t content_len;
    if (json_str_token((const char *)cresp, cresp_len, "content",
                       &content_tok, &content_len) != 0) {
        wipe_free(cresp, cresp_len);
        free(reply_to);
        FAIL(502, "coord response has no content");
    }
    /* The real KV-prefix cache-hit signal, which the platform discounts against
     * (docs/kv-cache-design.md §6). A coordinator that does not report it (an
     * older build) yields a constant false/0, so no discount applies -- the
     * contract stays backward compatible. cache_hit is a bare boolean literal,
     * so json_int_field does not apply; we probe for the fixed shape
     * "cache_hit":true instead. This agent only ever talks to our own
     * coordinator's fixed output, consistent with the rest of the file. */
    int cached_tokens = json_int_field((const char *)cresp, cresp_len, "cached_tokens", 0);
    int cache_hit = 0;
    {
        long ch = json_key_colon((const char *)cresp, cresp_len, "cache_hit");
        if (ch >= 0) {
            size_t ci = (size_t)ch;
            while (ci < cresp_len && (cresp[ci] == ' ' || cresp[ci] == '\t')) ci++;
            cache_hit = (ci + 4 <= cresp_len && !memcmp(cresp + ci, "true", 4));
        }
    }
    if (cached_tokens < 0) cached_tokens = 0;
    size_t reply_cap = content_len + 96;
    char *reply = malloc(reply_cap);
    if (!reply) { wipe_free(cresp, cresp_len); free(reply_to); FAIL(500, "oom"); }
    idletoken_mlock(reply, reply_cap);
    int rl = snprintf(reply, reply_cap,
                      "{\"text\":\"%.*s\",\"cache_hit\":%s,\"cached_tokens\":%d}",
                      (int)content_len, content_tok,
                      cache_hit ? "true" : "false", cached_tokens);
    wipe_free(cresp, cresp_len);
    if (rl < 0 || (size_t)rl >= reply_cap) {
        idletoken_secure_zero(reply, reply_cap);
        idletoken_munlock(reply, reply_cap); free(reply);
        free(reply_to);
        FAIL(500, "reply build failed");
    }

    /* -- seal the reply back to the platform's key ------------------------ */
    size_t sealed_out_cap = (size_t)rl + IDLETOKEN_SEAL_OVERHEAD;
    uint8_t *sealed_out = malloc(sealed_out_cap);
    size_t sealed_out_len = 0;
    idletoken_priv_rc src_rc = IDLETOKEN_PRIV_EINVAL;
    if (sealed_out)
        src_rc = idletoken_sodium_seal(reply_to, (const uint8_t *)reply, (size_t)rl,
                                    sealed_out, sealed_out_cap, &sealed_out_len);
    idletoken_secure_zero(reply, reply_cap);   /* wipe plaintext reply */
    idletoken_munlock(reply, reply_cap); free(reply);
    free(reply_to);

    if (src_rc != IDLETOKEN_PRIV_OK) {
        free(sealed_out);
        FAIL(500, "seal failed");
    }

    char *resp_b64 = b64_encode(sealed_out, sealed_out_len);
    free(sealed_out);
    if (!resp_b64) FAIL(500, "oom");

    /* The round trip succeeded, so this session becomes the coordinator's live
     * history: commit the prefix state for the next cache-state report. */
    if (staged_n > 0) {
        pthread_mutex_lock(&g_prefix_mu);
        memcpy(g_prefix.hashes, staged, (size_t)staged_n * sizeof(staged[0]));
        g_prefix.n = staged_n;
        g_prefix.dirty = 1;
        pthread_mutex_unlock(&g_prefix_mu);
    }

    *out_b64 = resp_b64;
    return 0;
#undef FAIL
}

/* Handle POST /infer (direct transport): process_sealed + HTTP framing. */
static void handle_infer(int conn_fd, const idletoken_keypair *node,
                         const char *coord_addr,
                         const uint8_t *body, size_t body_len) {
    char *resp_b64 = NULL;
    int err_status = 500;
    const char *err_msg = "internal error";
    if (process_sealed(node, coord_addr, (const char *)body, body_len,
                       &resp_b64, &err_status, &err_msg) != 0) {
        idletoken_http_send_error(conn_fd, err_status, err_msg);
        return;
    }
    size_t out_cap = strlen(resp_b64) + 32;
    char *out = malloc(out_cap);
    if (out) {
        int on = snprintf(out, out_cap, "{\"sealed_response\":\"%s\"}", resp_b64);
        if (on > 0 && (size_t)on < out_cap)
            idletoken_http_send_json(conn_fd, 200, out, (size_t)on);
        else
            idletoken_http_send_error(conn_fd, 500, "response build failed");
        free(out);
    } else {
        idletoken_http_send_error(conn_fd, 500, "oom");
    }
    free(resp_b64);
}

/* --- Concurrent handling ---------------------------------------------------
 * The agent used to be strictly serial: accept -> handle_conn (blocking, the
 * entire coordinator round trip included) -> close. That **strangled the gain
 * from interleaved execution on the engine side**: the platform dispatches N
 * jobs concurrently based on seq_slots, the agent let exactly one into the
 * coordinator at a time, the coordinator never held more than one request, and
 * the pipeline bubbles stayed unfilled.
 *
 * One thread per connection, capped at AGENT_MAX_INFLIGHT. Past the cap we do
 * not reject; we handle the connection inline, serially -- degrading to the
 * behaviour that preceded this change rather than dropping jobs. Real rate
 * limiting lives on the platform side (dispatch follows the reported
 * concurrency) and in the coordinator (429 when the intake queue is full); the
 * agent must not stack a third rejection semantics on top.
 *
 * Thread-safety boundary: everything below handle_conn touches only its own
 * stack plus the read-only node / coord_addr / pubkey_b64. The single piece of
 * cross-thread mutable state is g_prefix, guarded by g_prefix_mu. */
#define AGENT_MAX_INFLIGHT 16

static pthread_mutex_t g_inflight_mu = PTHREAD_MUTEX_INITIALIZER;
static int             g_inflight    = 0;

static void handle_conn(int conn_fd, const idletoken_keypair *node,
                        const char *coord_addr, const char *pubkey_b64) {
    idletoken_http_req req;
    if (idletoken_http_read_request(conn_fd, &req) != 0) {
        idletoken_http_send_error(conn_fd, 400, "bad request");
        return;
    }
    /* Log method/path/sizes only — NEVER bodies (they hold the envelope; and
     * downstream of seal_open there is plaintext this log must never see). */
    fprintf(stderr, "platform-agent: %s %s body=%zuB\n", req.method, req.path, req.body_len);

    if (!strcmp(req.method, "GET") && !strcmp(req.path, "/healthz")) {
        char body[128];
        int bl = snprintf(body, sizeof(body), "{\"ok\":true,\"pubkey\":\"%s\"}", pubkey_b64);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(req.body);
        return;
    }
    if (!strcmp(req.method, "POST") && !strcmp(req.path, "/infer")) {
        if (!req.body || req.body_len == 0)
            idletoken_http_send_error(conn_fd, 400, "empty body");
        else
            handle_infer(conn_fd, node, coord_addr, req.body, req.body_len);
        free(req.body);
        return;
    }
    idletoken_http_send_error(conn_fd, 404, "no such endpoint");
    free(req.body);
}

typedef struct {
    int                     fd;
    const idletoken_keypair   *node;         /* read-only: fixed after init */
    const char             *coord_addr;   /* read-only */
    const char             *pubkey_b64;   /* read-only */
} agent_conn;

static void *agent_conn_thread(void *ud) {
    agent_conn *a = ud;
    handle_conn(a->fd, a->node, a->coord_addr, a->pubkey_b64);
    close(a->fd);
    free(a);
    pthread_mutex_lock(&g_inflight_mu);
    g_inflight--;
    pthread_mutex_unlock(&g_inflight_mu);
    return NULL;
}

/* ======================================================================
 * key file + main loop
 * ====================================================================== */

/* Load the provider secret key from `path`, or generate+persist it (0600).
 * Same contract as privacy_proxy.c. */
static int load_or_make_key(const char *path, idletoken_keypair *node) {
    if (path) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            uint8_t sk[IDLETOKEN_SK_BYTES];
            ssize_t r = read(fd, sk, sizeof(sk));
            close(fd);
            if (r == (ssize_t)sizeof(sk)) {
                idletoken_keypair_from_sk(sk, node);
                idletoken_secure_zero(sk, sizeof(sk));
                return 0;
            }
            fprintf(stderr, "platform-agent: key file %s is not %d bytes; refusing\n",
                    path, IDLETOKEN_SK_BYTES);
            return -1;
        }
    }
    /* generate fresh */
    if (idletoken_keypair_generate(node) != IDLETOKEN_PRIV_OK) return -1;
    if (path) {
        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) {
            fprintf(stderr, "platform-agent: cannot create key file %s: %s\n",
                    path, strerror(errno));
            return -1;
        }
        size_t off = 0;
        while (off < IDLETOKEN_SK_BYTES) {
            ssize_t w = write(fd, node->sk + off, IDLETOKEN_SK_BYTES - off);
            if (w > 0) { off += (size_t)w; continue; }
            if (w < 0 && errno == EINTR) continue;
            fprintf(stderr, "platform-agent: write key file failed: %s\n", strerror(errno));
            close(fd); return -1;
        }
        close(fd);
        fprintf(stderr, "platform-agent: generated new provider key, saved to %s (0600)\n", path);
    }
    return 0;
}

static void ignore_sigpipe(void) {
    /* No SIGPIPE on Windows — a send() to a closed socket returns
     * WSAECONNRESET, which the send paths already check. */
#ifndef _WIN32
    struct sigaction sa = {0};
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, NULL);
#endif
}

static void usage(FILE *o) {
    fprintf(o,
"idletoken-platform-agent  cluster-side marketplace agent in front of idletoken-coord\n"
"Usage: idletoken-platform-agent [--port N] [--coord URL] [--key-file PATH]\n"
"                             [--platform URL --jwt TOKEN --name NAME]\n"
"                             [--provider-id ID] [--heartbeat-secs N]\n"
"                             [--endpoint URL] [--relay]\n"
"\n"
"  --port N            platform-facing listen port (default 9700; unused in --relay)\n"
"  --coord URL         local coord OpenAI API (default http://127.0.0.1:8000)\n"
"  --key-file PATH     persist/reuse the 32-byte provider secret key (0600).\n"
"                      Omit for an ephemeral per-run key.\n"
"  --platform URL      platform gateway; enables registration + heartbeat\n"
"  --jwt TOKEN         platform JWT (the provider account's bearer token)\n"
"  --name NAME         provider display name for registration\n"
"  --provider-id ID    already registered: skip POST /providers, just beat\n"
"  --heartbeat-secs N  heartbeat interval (default 30; unused in --relay)\n"
"  --endpoint URL      advertised /infer endpoint (default\n"
"                      http://127.0.0.1:<port>/infer; set to this machine's\n"
"                      public URL when registering with a real platform)\n"
"  --model ID          model this cluster serves, reported in the capacity\n"
"                      registration so the platform routes matching requests\n"
"                      (default dsv4-flash — the platform billing id of\n"
"                      DeepSeek V4 Flash)\n"
"  --quant Q           precision this cluster loaded (small models, e.g.\n"
"                      Q4_K_M/Q8_0/BF16); reported so the platform only routes\n"
"                      matching-precision requests here. Omit = any precision.\n"
"  --relay             reverse connection mode: no listen port at all; the\n"
"                      agent dials OUT to the platform (long poll) and jobs\n"
"                      are pushed back over that connection. Zero home-side\n"
"                      network config (no port forward, no router setup).\n"
"                      Requires --platform and --jwt. Poll doubles as the\n"
"                      heartbeat. Same sealed envelope as direct mode.\n"
"  -h, --help          this help\n");
}

int main(int argc, char **argv) {
    int         port           = 9700;
    const char *coord_url      = "http://127.0.0.1:8000"; /* idletoken-coord --api-bind default */
    const char *key_file       = NULL;
    const char *platform_url   = NULL;
    const char *jwt            = NULL;
    const char *name           = "idletoken-cluster";
    const char *provider_id_in = NULL;
    const char *endpoint_arg   = NULL;
    /* Platform billing id of the model this cluster serves (multi-model §6).
     * NOTE: the platform keeps 'dsv4-flash' as DSv4's billing id; the engine
     * registry id 'deepseek-v4-flash' is normalized gateway-side. */
    const char *model          = "dsv4-flash";
    const char *quant          = "";      /* precision (small models); "" = any */
    int         beat_secs      = 30;
    int         relay          = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--port")           && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--coord")          && i + 1 < argc) coord_url = argv[++i];
        else if (!strcmp(a, "--key-file")       && i + 1 < argc) key_file = argv[++i];
        else if (!strcmp(a, "--platform")       && i + 1 < argc) platform_url = argv[++i];
        else if (!strcmp(a, "--jwt")            && i + 1 < argc) jwt = argv[++i];
        else if (!strcmp(a, "--name")           && i + 1 < argc) name = argv[++i];
        else if (!strcmp(a, "--provider-id")    && i + 1 < argc) provider_id_in = argv[++i];
        else if (!strcmp(a, "--heartbeat-secs") && i + 1 < argc) beat_secs = atoi(argv[++i]);
        else if (!strcmp(a, "--endpoint")       && i + 1 < argc) endpoint_arg = argv[++i];
        else if (!strcmp(a, "--model")          && i + 1 < argc) model = argv[++i];
        else if (!strcmp(a, "--quant")          && i + 1 < argc) quant = argv[++i];
        else if (!strcmp(a, "--relay"))                          relay = 1;
        else if (!strcmp(a, "--selftest"))                       return agent_selftest();
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "platform-agent: unknown arg: %s\n\n", a); usage(stderr); return 2; }
    }
    if (port <= 0 || port > 65535) { fprintf(stderr, "platform-agent: bad --port\n"); return 2; }
    if (beat_secs <= 0) beat_secs = 30;
    if (platform_url && !jwt && !provider_id_in) {
        fprintf(stderr, "platform-agent: --platform needs --jwt (and --name), or --provider-id\n");
        return 2;
    }
    if (relay && (!platform_url || !jwt)) {
        /* jwt is mandatory even with --provider-id: every poll authenticates. */
        fprintf(stderr, "platform-agent: --relay needs --platform and --jwt\n");
        return 2;
    }

    /* Tier-1 hardening: this process is the provider-side plaintext window. */
    idletoken_harden_process();

    idletoken_keypair node;
    if (load_or_make_key(key_file, &node) != 0) return 1;
    idletoken_mlock(node.sk, sizeof(node.sk));

    char *pubkey_b64 = b64_encode(node.pk, IDLETOKEN_PK_BYTES);
    if (!pubkey_b64) return 1;

    char coord_addr[256];
    if (url_to_addr(coord_url, coord_addr, sizeof(coord_addr)) != 0) {
        fprintf(stderr, "platform-agent: bad --coord URL: %s\n", coord_url);
        return 2;
    }
    char platform_addr[256] = "";
    if (platform_url && url_to_addr(platform_url, platform_addr, sizeof(platform_addr)) != 0) {
        fprintf(stderr, "platform-agent: bad --platform URL: %s\n", platform_url);
        return 2;
    }

    char endpoint[512];
    if (endpoint_arg)
        snprintf(endpoint, sizeof(endpoint), "%s", endpoint_arg);
    else
        snprintf(endpoint, sizeof(endpoint), "http://127.0.0.1:%d/infer", port);

    printf("idletoken-platform-agent  (backend: %s + blake2b sealed box)\n",
           idletoken_privacy_backend());
    if (relay)
        printf("  transport        : relay (outbound long poll; no listen port)\n");
    else
        printf("  listen (platform): 0.0.0.0:%d\n", port);
    printf("  coord (loopback) : %s\n", coord_addr);
    if (!relay) printf("  endpoint         : %s\n", endpoint);
    printf("  pubkey (b64)     : %s\n", pubkey_b64);
    printf("  key file         : %s\n", key_file ? key_file : "(ephemeral)");
    fflush(stdout);

    /* Register with the platform (unless an existing provider id was given). */
    char *provider_id = NULL;
    if (platform_url) {
        if (provider_id_in) {
            provider_id = strdup(provider_id_in);
        } else {
            provider_id = platform_register(platform_addr, jwt, name, pubkey_b64, endpoint, relay, model, quant);
            if (!provider_id) {
                fprintf(stderr, "platform-agent: registration failed; refusing to start\n");
                return 1;
            }
        }
        if (relay)
            printf("  provider id      : %s  (poll = heartbeat)\n", provider_id);
        else
            printf("  provider id      : %s  (heartbeat every %ds)\n", provider_id, beat_secs);
        fflush(stdout);
    } else {
        printf("  provider id      : (not registered; pass --platform to register)\n");
        printf("Register manually: POST /providers { name:\"%s\", endpoint:\"%s\", pubkey:\"<above>\" }\n",
               name, endpoint);
        fflush(stdout);
    }

    ignore_sigpipe();

    if (relay) {
        /* Reverse connection mode: never listens. relay_loop never returns —
         * it reconnects with backoff forever (the process is the connection). */
        fprintf(stderr, "platform-agent: relay mode; polling %s. Ctrl-C to stop.\n",
                platform_addr);
        relay_loop(&node, coord_addr, platform_addr, jwt, provider_id);
        /* unreachable */
        free(provider_id);
        free(pubkey_b64);
        return 0;
    }
    char bind_addr[32];
    snprintf(bind_addr, sizeof(bind_addr), "0.0.0.0:%d", port);
    int lfd = idletoken_listen_tcp(bind_addr);
    if (lfd < 0) {
        fprintf(stderr, "platform-agent: listen(%s): %s\n", bind_addr, strerror(errno));
        return 1;
    }
    fprintf(stderr, "platform-agent: listening. Ctrl-C to stop.\n");

    /* Accept loop with a 1s select() tick so heartbeats piggyback on the same
     * single thread (no threads needed — beats are cheap and infrequent). */
    time_t last_beat = 0;   /* 0 → beat immediately (puts us ONLINE at once) */
    for (;;) {
        if (provider_id && jwt) {
            time_t now = time(NULL);
            if (now - last_beat >= (time_t)beat_secs) {
                if (platform_heartbeat(platform_addr, jwt, provider_id, coord_addr) != 0)
                    fprintf(stderr, "platform-agent: heartbeat failed (will retry)\n");
                last_beat = now;
            }
            /* KV cache state: report whenever there is a new session (the
             * platform expires it after >10 min, so frequent re-reporting buys
             * nothing).
             * Reading `dirty` also takes the lock: under concurrency this is a
             * cross-thread visibility question, not "just reading an int". */
            pthread_mutex_lock(&g_prefix_mu);
            const int pfx_dirty = g_prefix.dirty, pfx_n = g_prefix.n;
            pthread_mutex_unlock(&g_prefix_mu);
            if (pfx_dirty && platform_post_cache_state(platform_addr, jwt, provider_id) == 0)
                fprintf(stderr, "platform-agent: cache-state posted (%d blocks)\n", pfx_n);
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(lfd, &rfds);
        struct timeval tv = { 1, 0 };
        int sr = select(lfd + 1, &rfds, NULL, NULL, &tv);
        if (sr < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "platform-agent: select: %s\n", strerror(errno));
            break;
        }
        if (sr == 0 || !FD_ISSET(lfd, &rfds)) continue;

        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "platform-agent: accept: %s\n", strerror(errno));
            break;
        }
        pthread_mutex_lock(&g_inflight_mu);
        const int have_room = g_inflight < AGENT_MAX_INFLIGHT;
        if (have_room) g_inflight++;
        pthread_mutex_unlock(&g_inflight_mu);

        agent_conn *ac = have_room ? malloc(sizeof(*ac)) : NULL;
        if (ac) {
            ac->fd = cfd; ac->node = &node;
            ac->coord_addr = coord_addr; ac->pubkey_b64 = pubkey_b64;
            pthread_t th;
            if (pthread_create(&th, NULL, agent_conn_thread, ac) == 0) {
                pthread_detach(th);
                continue;
            }
            free(ac);                       /* no thread available: handle it inline, serially */
        }
        if (have_room) {
            pthread_mutex_lock(&g_inflight_mu);
            g_inflight--;
            pthread_mutex_unlock(&g_inflight_mu);
        }
        handle_conn(cfd, &node, coord_addr, pubkey_b64);
        close(cfd);
    }

    close(lfd);
    free(provider_id);
    free(pubkey_b64);
    idletoken_secure_zero(node.sk, sizeof(node.sk));
    idletoken_munlock(node.sk, sizeof(node.sk));
    return 0;
}
