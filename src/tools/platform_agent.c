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
 * choices[0].message.content as {"text":"..."} for the sealed reply, with the
 * model's thinking beside it as "reasoning" when there was any.
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
#include "idletoken_b64.h"
#include "idletoken_apiconv.h"
#include "idletoken_admission.h"   /* prove to the coordinator that this job is
                                    * platform work (threat register PROV-28) */

/* Release builds inject this from client/package.json.  There is deliberately
 * no fallback: a hand-maintained default would let one of the three platform
 * agent build paths ship a stale compatibility claim forever. */
#ifndef IDLETOKEN_CLIENT_VERSION
#error "IDLETOKEN_CLIENT_VERSION must be generated from client/package.json"
#endif

#define IDLETOKEN_VERSION_HTTP_HEADER \
    "X-IdleToken-Version: " IDLETOKEN_CLIENT_VERSION "\r\n"

#include <errno.h>
#include <signal.h>
#ifdef __linux__
  #include <sys/prctl.h>
#endif
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

/* Thread-local storage across our three toolchains: MSVC spells it
 * __declspec(thread); MinGW and the Unix compilers take C11 _Thread_local. */
#if defined(_MSC_VER)
  #define IDLETOKEN_TLS __declspec(thread)
#else
  #define IDLETOKEN_TLS _Thread_local
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
/* Is the KV-affinity report switched off for this machine? (PRIV-06/PRIV-11.)
 *
 * What this report contains is worth naming precisely, because "a Bloom filter
 * of hashes" sounds like it contains nothing. The hashes are chained SHA-256
 * over the rendered message text under a salt that is a PUBLIC CONSTANT — it
 * has to be, or the platform could not compute the same values for a consumer's
 * prompt and affinity routing would not work at all. So anyone holding the
 * filter can TEST A GUESS: take a prompt prefix you suspect, hash it the same
 * way, and see whether the bits are set. That is not recovery of unknown text,
 * but it is confirmation of guessed text, about prompts that belong to this
 * provider's CUSTOMERS rather than to its owner.
 *
 * It is left ON by default because it is what makes prefix-cache affinity work,
 * and a provider that reports nothing simply loses that routing preference. But
 * "the mechanism needs it" is not the same as "the operator agreed to it", so
 * there is a switch, and it says what it costs. Making the salt per-provider
 * instead would remove the confirmation oracle AND the feature; that trade is a
 * platform-side decision, recorded as a cross-owner request in
 * results/security-hardening-overflow-privacy-20260830.md. */
static int cache_report_off(void) {
    static int resolved = 0, off = 0;
    if (!resolved) {
        const char *e = getenv("IDLETOKEN_NO_CACHE_REPORT");
        resolved = 1;
        off = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
        if (off)
            fprintf(stderr, "platform-agent: IDLETOKEN_NO_CACHE_REPORT is set — "
                            "not reporting cached prompt prefixes. This machine "
                            "loses prefix-affinity routing; nothing derived from "
                            "a customer's prompt text leaves it (PRIV-06).\n");
    }
    return off;
}

static int platform_post_cache_state(const char *platform_addr, const char *jwt,
                                     const char *provider_id) {
    /* Fold g_prefix into the Bloom filter under the lock, then let go: the HTTP
     * POST that follows can take seconds, and holding the lock across network
     * I/O would stall the inference threads with it. */
    uint8_t bloom[PFX_BLOOM_BYTES] = {0};
    if (cache_report_off()) {
        /* Clear the dirty flag as well, or every finished job re-enters this
         * function and the "off" line becomes a per-request decision. */
        pthread_mutex_lock(&g_prefix_mu);
        g_prefix.dirty = 0;
        pthread_mutex_unlock(&g_prefix_mu);
        return -1;   /* -1 = nothing was posted; the caller only logs on 0 */
    }
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
 *
 * The codec moved to src/common/b64.c on 2026-08-19: the coordinator's overflow
 * path seals envelopes with the same encoding this file opens them with, and a
 * second copy of a codec that must agree byte for byte is how "the agent
 * accepts it, the coordinator 400s it" gets written. These two thin wrappers
 * keep the ~20 call sites below untouched.
 * ====================================================================== */

static char *b64_encode(const uint8_t *in, size_t n) {
    return idletoken_b64_encode(in, n);
}

static uint8_t *b64_decode(const char *s, size_t slen, size_t *out_len) {
    return idletoken_b64_decode(s, slen, out_len);
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
/* Does this (possibly non-terminated) JSON buffer contain `needle`?
 *
 * The stats body is read as bytes + length, so plain strstr() would run past
 * the end on a response that is not NUL-terminated. Only used for exact-shape
 * literals like "\"shared_mode\":true", which the coordinator emits without
 * spaces — a pretty-printer on the other side would make this miss, and the
 * platform would read a hardened node as unhardened. That direction of error
 * is the safe one, and the two sides are the same repo.  */
static int json_has_literal(const char *json, size_t len, const char *needle) {
    size_t nl = strlen(needle);
    if (!json || nl == 0 || len < nl) return 0;
    for (size_t i = 0; i + nl <= len; i++)
        if (memcmp(json + i, needle, nl) == 0) return 1;
    return 0;
}

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
                                  const char *bearer, const char *extra_hdr,
                                  const uint8_t *body, size_t body_len,
                                  int *out_status, size_t *out_len,
                                  int timeout_secs) {
    /* "unix:<path>" is the shared-mode transport to the coordinator. The agent
     * opens the platform's envelope, so everything it sends onward is the
     * buyer's plaintext; on loopback TCP that is readable with one tcpdump by
     * the owner of this machine. Same dispatch shape as the sidecar's engine
     * link, so there is one idiom to learn, not two. */
    int is_unix = strncmp(addr, "unix:", 5) == 0;
    int fd = is_unix ? idletoken_connect_unix(addr + 5)
                     : idletoken_connect_tcp(addr);
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
        if (an < 0 || (size_t)an >= sizeof(auth)) { idletoken_close_fd(fd); return NULL; }
    }
    char head[1536];
    int hn = snprintf(head, sizeof(head),
                      "%s %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "%s%s"
                      "Connection: close\r\n\r\n",
                      /* A socket path is not a host name; send a name the
                       * server will accept (cpp-httplib and our own parser
                       * both ignore which). */
                      method, path, is_unix ? "localhost" : addr, body_len, auth,
                      extra_hdr ? extra_hdr : "");
    if (hn < 0 || (size_t)hn >= sizeof(head)) { idletoken_close_fd(fd); return NULL; }
    if (idletoken_sendall(fd, head, (size_t)hn) < 0 ||
        (body_len && idletoken_sendall(fd, body, body_len) < 0)) {
        idletoken_close_fd(fd); return NULL;
    }

    /* Read the whole response until EOF (Connection: close). */
    size_t cap = 8192, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) { idletoken_close_fd(fd); return NULL; }
    for (;;) {
        if (len + 4096 > cap) {
            size_t ncap = cap * 2;
            uint8_t *nb = realloc(buf, ncap);
            if (!nb) { free(buf); idletoken_close_fd(fd); return NULL; }
            buf = nb; cap = ncap;
        }
        /* recv, not read: on Windows the CRT's read() only understands CRT
         * file descriptors, and a SOCKET is not one -- read() returns EBADF
         * without touching the wire. The request had already been sent, so
         * every reply (register, relay poll, coord answer) was thrown away
         * and the agent called a working platform "unreachable". recv() is
         * identical to read() for sockets on POSIX, so one spelling serves
         * both platforms. */
        ssize_t r = recv(fd, (char *)buf + len, 4096, 0);
        if (r > 0) { len += (size_t)r; continue; }
        if (r < 0 && errno == EINTR) continue;
        break;
    }
    idletoken_close_fd(fd);

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

/**
 * The SERVICE IDENTITY this machine publishes: base model + precision +
 * per-slot context (decision 13b, docs/ctx-tiers-2026-09.md). The platform
 * lists one service per triple, so this struct is exactly what a buyer sees
 * and exactly what the matcher filters on.
 *
 * It is read from the running coordinator, never from this process's own
 * arguments -- see coord_identity() for why that distinction is the whole
 * point.
 */
typedef struct {
    char model[128];
    char quant[64];
    /* Stable coordinator pseudonym used only to prevent overflow from being
     * routed straight back to its busy origin.  It is routing metadata, not
     * part of the public service identity below. */
    char origin_id[24];
    int  ctx;          /* per-slot context; 0 = the coordinator did not say */
    /* Is an engine actually loaded and warm right now (`engine_state:"ready"`)?
     * NOT part of the published identity -- identity_same() ignores it on
     * purpose, because "the same service, currently down" is still the same
     * service. It gates whether we advertise AT ALL, not what we advertise. */
    int  engine_ready;
} agent_identity;

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
                               const agent_identity *want) {
    /* A marketplace service is a concrete model + precision + context triple.
     * Refuse defensively here as well as at the coordinator-reading call sites:
     * an omitted precision must never turn into a wildcard listing. */
    if (!want->model[0] || !want->quant[0]) {
        fprintf(stderr, "platform-agent: refuse: marketplace registration needs "
                        "a concrete model and precision from the running coordinator\n");
        return NULL;
    }
    char name_esc[256], ep_esc[512], model_esc[128], quant_esc[64];
    json_escape_into(name_esc, sizeof(name_esc), name);
    json_escape_into(model_esc, sizeof(model_esc), want->model);
    json_escape_into(quant_esc, sizeof(quant_esc), want->quant);
    /* Precision-aware capability: the loaded quant is always advertised so
     * the platform only routes matching-precision requests here. There is no
     * wildcard/unspecified precision service.
     *
     * `ctx` is the per-slot context and the third element of the SERVICE
     * identity (model + quant + per-slot context): the platform lists one
     * service per declared triple, and a relaunch with a different triple is a
     * new listing while the same triple resumes the old one. Omitted when the
     * coordinator's ctx_size could not be read -- the platform must record
     * "undeclared", not a guess. */
    char ctx_part[48];
    ctx_part[0] = '\0';
    if (want->ctx > 0)
        snprintf(ctx_part, sizeof(ctx_part), ",\"ctx\":%d", want->ctx);
    char origin_part[64];
    origin_part[0] = '\0';
    if (want->origin_id[0])
        snprintf(origin_part, sizeof(origin_part),
                 ",\"origin_id\":\"%s\"", want->origin_id);
    char cap[384];
    snprintf(cap, sizeof(cap),
             "\"capacity\":{\"models\":[{\"model\":\"%s\",\"quant\":\"%s\"%s}],\"tiers\":[1]%s}",
             model_esc, quant_esc, ctx_part, origin_part);
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
    /* This wrapper is used only for authenticated platform control-plane
     * requests (registration, cache state, heartbeat, relay poll/result).
     * Local coordinator/OpenAI traffic uses its own wrapper below, so it does
     * not inherit a platform-only compatibility header. */
    return http_request_json("POST", addr, path, bearer,
                             IDLETOKEN_VERSION_HTTP_HEADER, body, body_len,
                             out_status, out_len, timeout_secs);
}

/* The forward into the local coordinator, marked as platform-dispatched.
 *
 * The coordinator must be able to tell this job from one a LAN client sent
 * directly, because a platform job has to be finished here or refused -- it is
 * never forwarded back out (docs/overflow-routing-design.md §2). Both arrive on
 * the same endpoint over loopback, so the marker is the only thing that
 * distinguishes them; if this header is ever dropped, that rule silently stops
 * holding. G_OVERFLOW_ORIGIN exists to catch exactly that. */
/* The coordinator's --api-token, when it has one. Empty = the coordinator does
 * not ask for one, which is the LAN default.
 *
 * Needed since overflow routing landed (2026-08-19). The coordinator refuses to
 * enable overflow without an --api-token, and the client's sharing switch turns
 * lending and borrowing on together -- so from that switch onwards every
 * coordinator that shares also demands a token, and an agent that could not
 * present one would have every dispatched job answered 401. The failure would
 * read as "the platform stopped sending me work". */
static char g_coord_token[256] = "";

/* --- the admission capability (threat register PROV-28) --------------------
 *
 * The header above is a claim the sender chooses to make. It was, until
 * 2026-08-30, the ONLY thing telling the coordinator that a job came from the
 * platform — and an agent that deleted that one line had every dispatched job
 * treated as local work: eligible to be forwarded to a third machine, charged
 * a second time, and shown to one more stranger, with nothing anywhere
 * reporting an error.
 *
 * So the agent now also mints a single-use capability under the channel key the
 * coordinator beside it published to a 0600 file. What that proves is narrow
 * and worth stating exactly: it proves the sender can read this coordinator's
 * channel key, not that the sender is honest. A modified agent on the same
 * machine reads the same file (the accepted HOST-02/HOST-16 boundary). What it
 * closes is the PROV-28 attack itself — omitting the marker no longer reads as
 * "local", because on a sharing coordinator absence is refused rather than
 * promoted.
 *
 * Attachment is LAZY and RE-CHECKED, not done once at startup, because the
 * client spawns the coordinator and the agent together and the file may not
 * exist yet when this process starts. A failure is loud but never fatal: the
 * legacy header still goes out on every request, so an agent that cannot mint
 * gets its jobs served and not forwarded, which is the safe direction.
 *
 * ⚠ RE-CHECKED, not cached-forever. This WAS a latch until 2026-09-01, and the
 * latch was the bug. The coordinator rolls its channel key on every start and
 * the client deliberately keeps this process alive across a coordinator restart
 * (engine.rs::stop_engine) — so after a model switch, a manual stop/start or a
 * crash respawn, the cached key was one nobody recognised. Minting is a local
 * HMAC, so it kept SUCCEEDING; the refusal appeared only at the coordinator, as
 * a 403 on every dispatched job, which the platform read as a provider fault.
 * Reproduced end to end on a Windows test node: one restart took it from
 * serving to `providers_isolated` with `/v1/models` empty while its owner could
 * see the engine running. So: an attachment is a CACHE, and it is invalidated
 * whenever the file says something else. */
static int  g_adm_attached  = 0;
static int  g_adm_complained = 0;

static void agent_admission_refresh(void) {
    char path[400] = "", err[240] = "";
    const char *env = getenv("IDLETOKEN_ADMISSION_KEY");

    /* An explicit key wins: the client that spawned both processes may pass it
     * directly rather than let this one go looking on disk. It cannot go stale
     * behind our back the way the file can — the only way it changes is a
     * restart of this process — so it is attached once and left alone. */
    if (env && env[0]) {
        if (g_adm_attached) return;
        if (idletoken_admission_attach(env, err, sizeof err) == 0) {
            g_adm_attached = 1;
            fprintf(stderr, "platform-agent: admission channel attached from "
                            "IDLETOKEN_ADMISSION_KEY\n");
            return;
        }
    } else if (idletoken_admission_default_paths(path, sizeof path, NULL, 0) == 0) {
        int rc = idletoken_admission_attach_file_if_changed(path, err, sizeof err);
        if (rc >= 0) {
            /* Name the re-attach. A coordinator restart under a live agent used
             * to be invisible from here, and "invisible" is what let it run for
             * days as an unexplained provider fault. */
            if (rc == 1)
                fprintf(stderr, "platform-agent: admission channel %s from %s\n",
                        g_adm_attached ? "re-attached (the coordinator restarted)"
                                       : "attached", path);
            g_adm_attached = 1;
            g_adm_complained = 0;   /* a later loss deserves to be said again */
            return;
        }
    }
    /* Complain once, then stop: this is retried on every job, and a coordinator
     * that never publishes a key would otherwise fill the log. */
    g_adm_attached = 0;
    if (!g_adm_complained) {
        g_adm_complained = 1;
        fprintf(stderr, "platform-agent: cannot attach the coordinator's admission "
                        "channel (%s). Dispatched jobs will still be served, and "
                        "will still not be forwarded, because the legacy origin "
                        "header is sent as well — but they arrive unproven "
                        "(threat register PROV-28).\n",
                err[0] ? err : "no channel key available");
    }
}

/* A job id the capability can be bound to. The platform's own id when it is
 * usable; otherwise one derived from the body hash, which is unique per request
 * without inventing state. The id is a label inside the MAC — the single-use
 * property comes from the nonce and the body binding, not from this — so a
 * fallback here weakens nothing. */
static void agent_job_label(const char *job_id, const uint8_t body_hash[32],
                            char *out, size_t cap) {
    size_t i;
    int ok = job_id && job_id[0];
    for (i = 0; ok && job_id[i]; i++) {
        char c = job_id[i];
        if (i >= 63) { ok = 0; break; }
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-' || c == ':'))
            ok = 0;
    }
    if (ok) { snprintf(out, cap, "%s", job_id); return; }
    {
        char hex[65];
        idletoken_admission_hex(body_hash, 32, hex, sizeof hex);
        snprintf(out, cap, "body-%.24s", hex);
    }
}

/* The headers this agent puts on a plaintext forward into the coordinator.
 *
 * The legacy origin header is ALWAYS present, even when a capability was
 * minted. Two mechanisms, one meaning: a coordinator older than this change
 * understands only the header, and a coordinator that fails to mint for any
 * reason must still be told this is platform work. Dropping the header once the
 * capability existed would have made the upgrade a downgrade for every mixed
 * pair of versions. */
static void agent_forward_headers(const char *job_id, int hops,
                                  const uint8_t *body, size_t body_len,
                                  char *out, size_t cap) {
    uint8_t bh[32];
    char label[IDLETOKEN_ADM_JOB_CAP];
    char ticket[IDLETOKEN_ADM_TICKET_CAP] = "";
    char err[200] = "";
    size_t off;

    off = (size_t)snprintf(out, cap,
                           IDLETOKEN_HDR_ORIGIN ": " IDLETOKEN_ORIGIN_PLATFORM "\r\n"
                           IDLETOKEN_HDR_HOPS ": %d\r\n",
                           hops < 0 ? 0 : hops);

    agent_admission_refresh();
    if (!g_adm_attached) return;

    idletoken_admission_body_hash(body, body_len, bh);
    agent_job_label(job_id, bh, label, sizeof label);
    if (idletoken_admission_mint(label, bh, (long long)time(NULL),
                                 ticket, sizeof ticket, err, sizeof err) != 0) {
        /* Say which job could not be proven. Silence here would present as the
         * coordinator refusing to forward the OWNER's traffic with no visible
         * cause on a --shared machine. */
        fprintf(stderr, "platform-agent: could not mint an admission capability "
                        "for job %s (%s) — forwarding with the legacy header "
                        "only\n", label, err[0] ? err : "unknown error");
        g_adm_attached = 0;          /* re-read the key on the next job */
        return;
    }
    snprintf(out + off, cap - off, IDLETOKEN_HDR_ADMISSION ": %s\r\n", ticket);
}

/* Defined with the other buffer helpers further down; needed here for the one
 * response this function throws away. */
static void wipe_free(void *p, size_t n);

/* Did the coordinator refuse this because the capability did not verify?
 *
 * Matched on the coordinator's own wording (coord_main.c: "admission capability
 * rejected: <reason>") and only ever used to decide whether to re-mint — never
 * to decide anything about the answer, so a wording drift costs one retry, not
 * a wrong result. Paired with the 403 status so an unrelated 403 body that
 * happens to quote the phrase cannot trigger it.
 *
 * Scanned over an explicit length rather than with strstr: the body is
 * attacker-adjacent bytes from a socket and is not promised to be NUL
 * terminated. */
static int coord_refused_admission(int status, const uint8_t *body, size_t len) {
    static const char needle[] = "admission capability rejected";
    const size_t n = sizeof needle - 1;
    size_t i;
    if (status != 403 || !body || len < n) return 0;
    for (i = 0; i + n <= len; i++)
        if (memcmp(body + i, needle, n) == 0) return 1;
    return 0;
}

/* One retry, and only for a refused capability.
 *
 * The per-job key re-read in agent_admission_refresh() closes the steady state,
 * but not the race: the coordinator can restart between the read and the
 * request landing, and a single refusal is not free — the platform charges the
 * provider a strike and a cooldown for it (reliability.ts), so the machine goes
 * out of routing for something that was a version skew of a few milliseconds.
 * Re-minting under the key that is on disk NOW costs one file read.
 *
 * Bounded at one attempt on purpose: if the second ticket is refused too, the
 * cause is not a restart, and looping would turn a real refusal into a stall. */
static uint8_t *http_post_json_platform(const char *addr, const char *path,
                                        const char *job_id, int hops,
                                        const uint8_t *body, size_t body_len,
                                        int *out_status, size_t *out_len,
                                        int timeout_secs) {
    char hdrs[IDLETOKEN_ADM_TICKET_CAP + 160];
    uint8_t *resp;
    int attempt;

    for (attempt = 0; ; attempt++) {
        agent_forward_headers(job_id, hops, body, body_len, hdrs, sizeof hdrs);
        resp = http_request_json("POST", addr, path,
                                 g_coord_token[0] ? g_coord_token : NULL,
                                 hdrs,
                                 body, body_len, out_status, out_len, timeout_secs);
        if (attempt > 0 || !coord_refused_admission(*out_status, resp, *out_len))
            return resp;
        fprintf(stderr, "platform-agent: the coordinator refused this job's "
                        "admission capability — re-reading its channel key and "
                        "minting once more (it restarted under us)\n");
        if (resp) wipe_free(resp, *out_len);
        *out_status = 0;
        *out_len = 0;
        /* No flag to clear: the next agent_forward_headers() re-reads the file
         * and compares, so the fresh ticket is minted under whatever the
         * coordinator published when it came back up. */
    }
}

static uint8_t *http_get_json(const char *addr, const char *path,
                              int *out_status, size_t *out_len, int timeout_secs) {
    return http_request_json("GET", addr, path, NULL, NULL, NULL, 0,
                             out_status, out_len, timeout_secs);
}

/* ----------------------------------------------------------------------
 * Listing floor: a machine with less than this much context per slot is
 * refused at the door (cleanup-t9 §A3).
 *
 * Why a floor and not a smaller platform tier. The platform buckets capacity
 * declarations by context class, and the smallest class is 8192 (`CTX_CLASSES`
 * in scheduler/tiers.ts); `numFromCtxMap` only ever borrows UPWARD. So a
 * machine declaring, say, 4096 matches no class at all, every declared number
 * is silently dropped, and the platform falls back to "one slot, no TTFT
 * estimate" -- the machine looks serial and blind, and the buyer whose request
 * lands there gets the worst experience on the market with nothing in the logs
 * to explain it.
 *
 * The floor is the honest end of that. Claude Code's system prompt alone is
 * ~13K tokens; a slot smaller than 8192 cannot hold a real conversation, so
 * listing it produces a bad trade rather than a cheap one. Refusing at startup
 * costs this machine's owner nothing they were going to earn, and it is a
 * sentence they can act on ("give the coordinator a bigger -c") instead of a
 * silent mismatch three layers away.
 *
 * This is the SHARING side only. The platform's borrow rule is unchanged:
 * adding a 4096 class there would move the problem, not remove it -- somebody
 * would still be selling a context too small to serve anyone. */
#define AGENT_MIN_LIST_CTX 8192

/* How long registration waits for the engine to finish loading, and how often
 * it re-asks. The bound is generous on purpose: the largest curated models take
 * minutes to load, and a machine that is merely slow to start is not a machine
 * that should be refused. */
#define AGENT_READY_POLL_SECS      2
#define AGENT_READY_WAIT_MAX_SECS  1800

/**
 * Read the service identity from the coordinator's /idletoken/v1/stats.
 * Returns 0 on success, -1 when the coordinator could not be reached or did
 * not report a usable `ctx_size`.
 *
 * The COORDINATOR is the only honest source for these three fields. The
 * agent's `--model` / `--quant` arguments are a snapshot of what the launcher
 * believed at the moment the agent started, and the client deliberately lets
 * the agent outlive a coordinator restart
 * (client/src-tauri/src/engine.rs::stop_engine) -- so changing model,
 * precision or context and restarting only the coordinator used to leave the
 * marketplace advertising the old triple forever. Measured 2026-09-03 on both
 * Windows nodes: one ran Q2_K_XL@128K while selling IQ2_XXS@256K, the other
 * ran IQ2_XXS@256K while selling @128K -- each was advertising the other
 * machine's configuration (results/agent-stale-registration-20260903.md).
 *
 * -1 means "we could not check", NOT "the machine is fine": every caller has
 * to say which one it is out loud, and none of them may treat it as a reason
 * to change what is already published.
 *
 * `ctx_size` is the context of ONE slot (llama_sidecar.c) -- the number a
 * single conversation actually gets, which is both the number the listing
 * floor is about and the third element of the service identity. `-np N`
 * divides `-c`, so a machine with a large `-c` and many slots can still be
 * under the floor per slot.
 */
static int coord_identity(const char *coord_addr, agent_identity *out) {
    memset(out, 0, sizeof(*out));
    if (!coord_addr || !coord_addr[0]) return -1;
    int status = 0; size_t rlen = 0;
    uint8_t *resp = http_get_json(coord_addr, IDLETOKEN_PATH_STATS, &status, &rlen, 5);
    if (!resp) return -1;
    if (status < 200 || status >= 300) { free(resp); return -1; }
    const char *j = (const char *)resp;
    int ctx = json_int_field(j, rlen, "ctx_size", 0);
    char *model = json_str_dup(j, rlen, "model");
    char *quant = json_str_dup(j, rlen, "quant");
    char *origin_id = json_str_dup(j, rlen, "overflow_origin_id");
    /* The coordinator answers /stats while the engine is still loading (it owns
     * the API port from the moment it starts), so "the coordinator replied" is
     * NOT "there is an engine to sell". `engine_state` is the coordinator's own
     * word for that difference (coord_llama_state_name(): ready/starting/
     * failed/...). An older coordinator does not emit the field at all; absent
     * means ready, which is exactly the behaviour that preceded this check --
     * we do not take a machine off the market over a field it never sent. */
    char *estate = json_str_dup(j, rlen, "engine_state");
    free(resp);
    if (model) snprintf(out->model, sizeof(out->model), "%s", model);
    if (quant) snprintf(out->quant, sizeof(out->quant), "%s", quant);
    if (origin_id) snprintf(out->origin_id, sizeof(out->origin_id), "%s", origin_id);
    out->engine_ready = !estate || !strcmp(estate, "ready");
    free(model); free(quant); free(origin_id); free(estate);
    out->ctx = ctx > 0 ? ctx : 0;
    return out->ctx > 0 ? 0 : -1;
}

/**
 * May this machine be advertised right now?
 *
 * One probe, one answer, used by both transports before every beat. `*live` is
 * filled only when the answer is yes, so a caller can never accidentally
 * re-declare from a reading that says "nothing is running here".
 *
 * WHY THIS GATE EXISTS (2026-09-12). The product rule is one sentence: no
 * engine means not sharing, and not sharing means not on the discovery page.
 * The client already enforced it at the switch ("start a model before turning
 * on sharing"), but nothing enforced it afterwards -- and the switch is a
 * standing setting, so every later launch resumed an agent that outlived the
 * coordinator it was started next to. Found on two real test machines: both
 * had a client and this agent running with NO coordinator and NO llama-server
 * at all, and both were on sale on the public marketplace -- one of them
 * advertising qwen3.5-35b-a3b@256K that nothing could have served.
 *
 * The beat is the right place for it because the beat is what "online" means:
 * relay's poll and direct mode's heartbeat both refresh `lastBeat`, and the
 * platform's freshness TTL (60s) then does the rest by itself -- no delisting,
 * no provider churn, and the listing comes straight back when the engine does.
 */
static int engine_sellable(const char *coord_addr, agent_identity *live) {
    agent_identity probe;
    if (coord_identity(coord_addr, &probe) != 0) return 0;
    if (!probe.engine_ready) return 0;
    *live = probe;
    return 1;
}

/* Do two declarations name the same service? All three fields, because all
 * three are the identity -- a changed context is as much a different product
 * as a changed precision (a buyer may be here precisely for the long window). */
static int identity_same(const agent_identity *a, const agent_identity *b) {
    return a->ctx == b->ctx &&
           !strcmp(a->model, b->model) &&
           !strcmp(a->quant, b->quant);
}

static void identity_print(const agent_identity *id, char *out, size_t cap) {
    snprintf(out, cap, "%s%s%s @ %d tokens/slot",
             id->model[0] ? id->model : "(no model)",
             id->quant[0] ? ":" : "", id->quant[0] ? id->quant : "",
             id->ctx);
}

/**
 * Block until this machine has an engine worth listing, or give up.
 * Returns 0 with `*live` filled, or an exit code.
 *
 * This is where "no engine means not sharing" is enforced at startup, and it
 * WAITS rather than refuses because the two states that look alike here are
 * "the coordinator is still loading a 200 GB model" and "there is no
 * coordinator" -- and the first one resolves by itself, often minutes later.
 * The client already promises exactly this ("the agent waits for the
 * coordinator on port N"), so waiting is also the behaviour users were told.
 *
 * What it must never do is the thing it used to do: give up checking and
 * register anyway. That warning-and-continue path is how a machine with no
 * engine at all ended up on sale (see engine_sellable()).
 *
 * Bounded, because a wait with no end is indistinguishable from a hang -- the
 * same reason cluster formation grew one (hard invariant 12).
 */
static int await_sellable_engine(const char *coord_addr, agent_identity *live) {
    memset(live, 0, sizeof(*live));
    int ever_answered = 0;
    for (time_t waited = 0; ; waited += AGENT_READY_POLL_SECS) {
        agent_identity probe;
        if (coord_identity(coord_addr, &probe) == 0) {
            ever_answered = 1;
            *live = probe;
            if (probe.engine_ready) return 0;
        }
        if (waited >= AGENT_READY_WAIT_MAX_SECS) {
            fprintf(stderr,
                    "platform-agent: refuse: %s within %ds.\n"
                    "  Not registering: listing a machine with no engine behind it "
                    "sells work nothing can do. Start a model first, then turn "
                    "sharing on.%s\n",
                    ever_answered
                        ? "the coordinator never reported a ready engine"
                        : "the coordinator never answered /idletoken/v1/stats",
                    AGENT_READY_WAIT_MAX_SECS,
                    ever_answered ? " Check the engine log for a load failure." : "");
            return 4;
        }
        if (waited % 30 == 0)
            fprintf(stderr, "platform-agent: no engine to list yet at %s (%s); "
                            "waiting before registering (%ds so far)\n",
                    coord_addr, ever_answered ? "still loading" : "not answering",
                    (int)waited);
        sleep(AGENT_READY_POLL_SECS);
    }
}

/**
 * Refuse to list a coordinator whose slots are too small. Returns 0 to go on,
 * non-zero to exit with that code. The reading is already established by
 * await_sellable_engine() -- there is no "could not check" case left.
 */
static int assert_listable_ctx(const agent_identity *id_out) {
    const agent_identity live = *id_out;
    const int ctx = live.ctx;
    if (ctx < AGENT_MIN_LIST_CTX) {
        fprintf(stderr,
                "platform-agent: refuse: this coordinator serves %d tokens per slot, "
                "and listing on the marketplace needs at least %d.\n"
                "  Why: the platform's smallest context class is %d tokens and it "
                "never borrows downward, so everything this machine declares about "
                "its own speed would be dropped and it would be dispatched to as if "
                "it were serial and untimed. Claude Code's system prompt alone is "
                "~13K tokens, so buyers landing here would get the worst sessions on "
                "the market.\n"
                "  Fix: restart the coordinator with a larger context "
                "(-c per-slot >= %d; remember -np divides -c), then start sharing "
                "again. Using the cluster yourself is unaffected -- this only stops "
                "it from being listed.\n",
                ctx, AGENT_MIN_LIST_CTX, AGENT_MIN_LIST_CTX, AGENT_MIN_LIST_CTX);
        return 4;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Keeping the published declaration equal to what is actually running.
 *
 * Registration is a one-shot at startup, but a coordinator is not: the client
 * restarts it whenever the user changes model, precision or context window,
 * and deliberately leaves this agent alive across that restart. So the triple
 * has to be re-checked for as long as the agent runs, and re-published the
 * moment it moves.
 *
 * Re-publishing means POST /providers again -- the SAME entrance a first
 * registration uses, deliberately, because that is the entrance that mints
 * service identities. The heartbeat cannot do this job: it whitelists load
 * fields only, on purpose, so that a cheap high-frequency request can never
 * silently promote a machine to selling a different model (providers.controller.ts,
 * mergeLiveCapacity). Going through registration also means decision 13b is
 * honoured for free: a changed triple ends the old Provider lifetime and
 * registers a NEW Provider with new services, STANDARD pricing and clean local
 * health. Account reputation remains on the platform User.
 * ---------------------------------------------------------------------- */

/* Everything a re-declaration needs, so the two transports' loops can each
 * call one function instead of threading nine arguments through. */
typedef struct {
    const char *platform_addr, *jwt, *name, *pubkey_b64, *endpoint, *coord_addr;
    int         relay;
    char       *provider_id;    /* owned here; re-registration may return a new one */
    agent_identity declared;    /* what the platform currently has from this machine */
    time_t      retry_after;    /* backoff after a refused re-registration */
    int         backoff_s;
} agent_registration;

#define AGENT_REDECLARE_BACKOFF_MIN_S  60
#define AGENT_REDECLARE_BACKOFF_MAX_S  900

/**
 * Publish the coordinator's triple if it moved.
 *
 * `live` is a reading the caller actually got (engine_sellable()); this
 * function is never called on a failed probe, because "the coordinator is
 * restarting" must not be mistaken for "this machine now serves something
 * else". The declaration only ever changes on a reading we actually got.
 */
static void reconcile_identity(agent_registration *reg, const agent_identity *live_in) {
    if (!reg->provider_id || !reg->jwt) return;
    const agent_identity live = *live_in;
    if (!live.model[0] || !live.quant[0]) {
        char what[224];
        identity_print(&live, what, sizeof(what));
        fprintf(stderr,
                "platform-agent: refuse: the coordinator now reports an incomplete "
                "service identity ('%s'). A marketplace service requires an explicit "
                "model and precision.\n"
                "  Stopping, because staying up would keep selling a precision this "
                "machine no longer proves it runs. Restart the coordinator so "
                "/idletoken/v1/stats reports both model and quant, then turn sharing "
                "back on.\n",
                what);
        exit(4);
    }
    if (identity_same(&live, &reg->declared)) return;          /* the common case */

    char was[224], now[224];
    identity_print(&reg->declared, was, sizeof(was));
    identity_print(&live, now, sizeof(now));

    /* The listing floor applies to a re-declaration exactly as it does to the
     * first one, and the only way to stop selling is to stop being online: a
     * process that keeps polling keeps the old declaration sellable, which is
     * the very fault this function exists to fix. */
    if (live.ctx < AGENT_MIN_LIST_CTX) {
        fprintf(stderr,
                "platform-agent: refuse: the coordinator restarted as '%s', which is "
                "below the %d-token listing floor (was '%s').\n"
                "  Stopping, because staying up would keep selling '%s' -- a service "
                "this machine no longer runs.\n"
                "  Fix: restart the coordinator with -c per-slot >= %d (remember -np "
                "divides -c), then turn sharing back on. Using the cluster yourself is "
                "unaffected.\n",
                now, AGENT_MIN_LIST_CTX, was, was, AGENT_MIN_LIST_CTX);
        exit(4);
    }

    const time_t t = time(NULL);
    if (reg->retry_after && t < reg->retry_after) return;

    fprintf(stderr, "platform-agent: the coordinator now serves '%s' (was '%s'); "
                    "re-registering so the marketplace stops advertising the old one\n",
            now, was);
    char *id = platform_register(reg->platform_addr, reg->jwt, reg->name,
                                 reg->pubkey_b64, reg->endpoint, reg->relay, &live);
    if (!id) {
        reg->backoff_s = reg->backoff_s ? reg->backoff_s * 2 : AGENT_REDECLARE_BACKOFF_MIN_S;
        if (reg->backoff_s > AGENT_REDECLARE_BACKOFF_MAX_S)
            reg->backoff_s = AGENT_REDECLARE_BACKOFF_MAX_S;
        reg->retry_after = t + reg->backoff_s;
        /* Loud, and it keeps saying it: until this succeeds the machine is
         * selling something it is not running, and the operator is the only
         * one who can act on the reason (a churn limit, an expired JWT). */
        fprintf(stderr, "platform-agent: WARNING: re-registration failed; the "
                        "marketplace still advertises '%s' while this machine runs "
                        "'%s'. Retrying in %ds.\n", was, now, reg->backoff_s);
        return;
    }
    if (strcmp(id, reg->provider_id) != 0)
        fprintf(stderr, "platform-agent: provider id changed %s -> %s\n",
                reg->provider_id, id);
    free(reg->provider_id);
    reg->provider_id = id;
    reg->declared    = live;
    reg->retry_after = 0;
    reg->backoff_s   = 0;
    fprintf(stderr, "platform-agent: now advertising '%s' (provider %s)\n",
            now, reg->provider_id);
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
 * (`avg_service_ms` / `queue_depth` / `seq_slots` from `/idletoken/v1/stats`), so we
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
    uint8_t *resp = http_get_json(coord_addr, IDLETOKEN_PATH_STATS, &status, &rlen, 5);
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
    char *origin_id = json_str_dup(j, rlen, "overflow_origin_id");
    /* Shared-mode posture (P1-6). Relayed as the coordinator states it —
     * three facts, not one "trusted" flag. This is NOT proof: the node runs
     * this code and could report whatever it likes. Its value is that a node
     * which lies has to lie ON THE RECORD, which is what makes cheating
     * accountable rather than merely discouraged.
     *
     * Absent fields (older coordinator, or shared mode off) simply do not
     * appear in the heartbeat — the platform must not read "silent" as
     * "hardened". */
    char shared_field[128] = "";
    {
        int sm = json_has_literal(j, rlen, "\"shared_mode\":true");
        if (sm) {
            int ev = json_has_literal(j, rlen, "\"engine_verified\":true");
            int lu = json_has_literal(j, rlen, "\"engine_link\":\"unix\"");
            snprintf(shared_field, sizeof(shared_field),
                     "\"shared_mode\":true,\"engine_verified\":%s,"
                     "\"engine_link\":\"%s\",",
                     ev ? "true" : "false", lu ? "unix" : "tcp");
        }
    }
    free(resp);
    if (ctx <= 0 || qdepth < 0) { free(origin_id); return -1; }   /* older coordinator lacks these fields: report nothing rather than guess */
    char ttft_field[64] = "";
    if (ttft_ms > 0)
        snprintf(ttft_field, sizeof(ttft_field), "\"avg_ttft_ms\":{\"%d\":%d},", ctx, ttft_ms);
    char origin_field[64] = "";
    if (origin_id && origin_id[0])
        snprintf(origin_field, sizeof(origin_field),
                 "\"origin_id\":\"%s\",", origin_id);
    free(origin_id);
    int n = snprintf(out, out_cap,
        "{\"seq_slots_by_ctx\":{\"%d\":%d},\"avg_service_ms\":{\"%d\":%d},%s%s"
        "%s\"queue_depth\":%d,\"max_ctx_tokens\":%d}",
        ctx, slots > 0 ? slots : 1, ctx, svc_ms > 0 ? svc_ms : 0, ttft_field,
        shared_field, origin_field, qdepth, ctx);
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
/* How often to re-probe while there is no engine. Short enough that coming
 * back from a model switch costs seconds, not a poll period. */
#define RELAY_IDLE_RECHECK_SECS 5

/* Shared sealed-envelope data path (defined below with the /infer handler).
 * `job_id` may be NULL on the direct transport, which has no platform-assigned
 * id; it names the capability the forward into the coordinator carries. */
static int process_sealed(const idletoken_keypair *node, const char *coord_addr,
                          const char *json, size_t json_len, const char *job_id,
                          char **out_b64, int *err_status, const char **err_msg);

/* POST the job result (success or error). Best effort: on failure the job
 * simply expires platform-side and routing fails over. Returns 0 on 2xx. */
static int relay_post_result(const char *platform_addr, const char *jwt,
                             const char *provider_id, const char *job_id,
                             const char *sealed_b64 /* NULL on error */,
                             const char *err_msg,   /* used when sealed_b64 == NULL */
                             int err_status         /* HTTP-ish code for that error */) {
    char path[256];
    int pn = snprintf(path, sizeof(path), "/providers/%s/relay/result", provider_id);
    if (pn < 0 || (size_t)pn >= sizeof(path)) return -1;

    size_t cap = (sealed_b64 ? strlen(sealed_b64) : strlen(err_msg)) + strlen(job_id) + 96;
    char *body = malloc(cap);
    if (!body) return -1;
    int bl;
    if (sealed_b64) {
        bl = snprintf(body, cap, "{\"job_id\":\"%s\",\"sealed_response\":\"%s\"}",
                      job_id, sealed_b64);
    } else {
        char msg_esc[256];
        json_escape_into(msg_esc, sizeof(msg_esc), err_msg);
        /* `status` carries WHY it failed, not just that it did. Without it the
         * relay transport is strictly worse than the direct one at the same
         * job: direct returns an HTTP status the platform classifies (429 =
         * busy, not a fault), while a relay failure arrived as a bare string
         * and defaulted to CONNECT — a real fault, with a strike and a
         * cooldown. Relay is the DEFAULT for home providers, so the transport
         * most people run was the one that punished them for being busy.
         * An older gateway simply ignores the extra field. */
        bl = snprintf(body, cap, "{\"job_id\":\"%s\",\"error\":\"%s\",\"status\":%d}",
                      job_id, msg_esc, err_status > 0 ? err_status : 502);
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
static void relay_loop(const idletoken_keypair *node, agent_registration *reg) {
    const char *coord_addr    = reg->coord_addr;
    const char *platform_addr = reg->platform_addr;
    const char *jwt           = reg->jwt;
    int backoff = 1;
    int was_sellable = 1;   /* registration proved an engine was there */

    for (;;) {
        /* The poll is this transport's heartbeat, so it is also where the
         * declaration gets checked against what the coordinator is actually
         * running. Before the poll, not after: a poll accepts work for the
         * triple the platform believes is here.
         *
         * And when there is no engine, there is no poll: see engine_sellable().
         * Skipping the poll is both halves of the honest answer at once --
         * `lastBeat` goes stale (the platform stops listing us) and we stop
         * claiming jobs we could not run. */
        agent_identity live;
        if (!engine_sellable(coord_addr, &live)) {
            if (was_sellable) {
                fprintf(stderr, "platform-agent: no engine is running behind this agent "
                                "(coordinator at %s unreachable or still loading); "
                                "pausing the relay poll, so this machine drops off the "
                                "marketplace until it is back.\n", coord_addr);
                was_sellable = 0;
            }
            sleep(RELAY_IDLE_RECHECK_SECS);
            continue;
        }
        if (!was_sellable) {
            fprintf(stderr, "platform-agent: the engine is back; resuming the relay poll\n");
            was_sellable = 1;
        }
        reconcile_identity(reg, &live);
        /* Rebuilt from the (possibly re-registered) id every iteration —
         * a provider id that moved and a path that did not is a poll that
         * silently stops receiving work. */
        char path[256];
        snprintf(path, sizeof(path), "/providers/%s/relay/poll", reg->provider_id);

        /* The poll IS the heartbeat on this transport, so the live load has to
         * ride along with it — rebuilt every iteration because that is the
         * point of it (queue depth and service time change between polls).
         *
         * It used to be a fixed `{"wait_ms":N}` with no capacity at all, and
         * the consequence was invisible from here: relay providers reported
         * their load NOWHERE, so the platform's scheduler fell back to "one
         * slot, no TTFT budget" for every one of them (measured 2026-08-19,
         * results/platform-seam-20260819.md V1). Relay is the DEFAULT for
         * home machines — the transport most people run was the one the
         * scheduler knew least about.
         *
         * A coordinator that cannot be probed yields an empty stats string and
         * the poll goes out exactly as it did before: the wait, and nothing
         * claimed. */
        char poll_body[512];
        char stats[400];
        int pbl;
        if (coord_stats_json(coord_addr, stats, sizeof(stats)) == 0)
            pbl = snprintf(poll_body, sizeof(poll_body),
                           "{\"wait_ms\":%d,\"capacity\":%s}", RELAY_WAIT_MS, stats);
        else
            pbl = snprintf(poll_body, sizeof(poll_body), "{\"wait_ms\":%d}", RELAY_WAIT_MS);
        if (pbl < 0 || (size_t)pbl >= sizeof(poll_body))
            pbl = snprintf(poll_body, sizeof(poll_body), "{\"wait_ms\":%d}", RELAY_WAIT_MS);

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
        int rc = process_sealed(node, coord_addr, (const char *)resp, rlen, job_id,
                                &sealed_b64, &err_status, &err_msg);
        free(resp);
        fprintf(stderr, "platform-agent: relay infer job=%s -> %s\n",
                job_id, rc == 0 ? "sealed ok" : err_msg);

        if (relay_post_result(platform_addr, jwt, reg->provider_id, job_id,
                              rc == 0 ? sealed_b64 : NULL, err_msg, err_status) != 0)
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
        if (pfx_dirty && platform_post_cache_state(platform_addr, jwt, reg->provider_id) == 0)
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
                          const char *json, size_t json_len, const char *job_id,
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
     * {model, messages, maxTokens?, tools?} → {model, messages, max_tokens?,
     * tools?}. The messages and tools arrays are re-embedded verbatim (raw
     * tokens), so nested content — including tool_calls / tool_call_id on
     * individual messages — survives untouched. tool_choice also survives:
     * agent clients use "required" to force a structured call, and silently
     * turning that into "auto" changes the request. */
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
    /* How many machines this prompt has already been handed through, as the
     * platform declared it inside the envelope. Absent today, which reads as 0
     * and is exactly the pre-existing behaviour; read now so that when the
     * platform starts declaring it (the cross-owner request in
     * results/security-hardening-overflow-privacy-20260830.md) every already
     * deployed agent honours it instead of being exempt from the rule. */
    int hops_in = json_int_field((const char *)plain, plain_len, "hops", 0);
    const char *tools_tok = NULL; size_t tools_len = 0;
    int have_req_tools = json_array_token((const char *)plain, plain_len, "tools",
                                          &tools_tok, &tools_len) == 0 && tools_len > 2;
    const char *choice_tok = idletoken_json_obj_get((const char *)plain, plain_len,
                                                    "tool_choice");
    long choice_len_raw = choice_tok
        ? idletoken_json_value_len(choice_tok, (const char *)plain + plain_len)
        : -1;
    size_t choice_len = choice_len_raw > 0 ? (size_t)choice_len_raw : 0;
    /* The consumer's thinking switch, forwarded VERBATIM.
     *
     * The platform seals it under the key the engine itself reads
     * (`chat_template_kwargs`), so this agent neither parses nor re-spells it:
     * it copies the object through, coord passes the OpenAI face through
     * wholesale, and the engine reads it. Absent = the consumer did not say,
     * and the model template's own default decides — which is why nothing is
     * synthesised here when the key is missing. */
    const char *ctk_tok = idletoken_json_obj_get((const char *)plain, plain_len,
                                                 "chat_template_kwargs");
    long ctk_len_raw = (ctk_tok && *ctk_tok == '{')
        ? idletoken_json_value_len(ctk_tok, (const char *)plain + plain_len)
        : -1;
    size_t ctk_len = ctk_len_raw > 0 ? (size_t)ctk_len_raw : 0;

    /* Contract hashes for the KV prefix: they must be computed into a staging
     * buffer while `plain` still exists (it is wiped moments from now), and are
     * committed as the "live session" state only after the whole round trip
     * succeeds (see the end of this function). */
    static char staged[PFX_MAX_BLOCKS][65];
    int staged_n = prefix_hash_messages(msgs_tok, msgs_len, staged, PFX_MAX_BLOCKS);

    size_t creq_cap = msgs_len + model_len + tools_len + choice_len + ctk_len + 192;
    char *creq = malloc(creq_cap);
    if (!creq) {
        idletoken_secure_zero(plain, plain_cap);
        idletoken_munlock(plain, plain_cap); free(plain);
        free(reply_to);
        FAIL(500, "oom");
    }
    idletoken_mlock(creq, creq_cap);    /* also plaintext */
    char mt_frag[40];
    mt_frag[0] = '\0';
    if (max_tokens > 0)
        snprintf(mt_frag, sizeof mt_frag, ",\"max_tokens\":%d", max_tokens);
    int cl = snprintf(creq, creq_cap,
                      "{\"model\":\"%.*s\",\"messages\":%.*s%s%s%.*s%s%.*s%s%.*s}",
                      (int)model_len, model_tok, (int)msgs_len, msgs_tok, mt_frag,
                      have_req_tools ? ",\"tools\":" : "",
                      (int)tools_len, have_req_tools ? tools_tok : "",
                      choice_len ? ",\"tool_choice\":" : "",
                      (int)choice_len, choice_len ? choice_tok : "",
                      ctk_len ? ",\"chat_template_kwargs\":" : "",
                      (int)ctk_len, ctk_len ? ctk_tok : "");

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
        cresp = http_post_json_platform(coord_addr, "/v1/chat/completions",
                               job_id, hops_in,
                               (const uint8_t *)creq, (size_t)cl, &cstatus, &cresp_len,
                               0 /* no timeout: real-model inference is slow by design */);
    /* plaintext request buffers are done — wipe immediately */
    idletoken_secure_zero(creq, creq_cap);
    idletoken_munlock(creq, creq_cap); free(creq);
    idletoken_secure_zero(plain, plain_cap);
    idletoken_munlock(plain, plain_cap); free(plain);

    if (!cresp || cstatus != 200) {
        int coord_answered = cresp != NULL;
        /* Salvage the coord's own words before the body is wiped: an engine
         * 400 carries a one-line reason ("request (48549 tokens) exceeds the
         * available context size (32768 tokens)") that is the whole diagnosis
         * — the status code alone cost an afternoon of packet captures
         * (2026-08-24). Engine/coord error messages are generated text plus
         * numbers, never an echo of the prompt, so relaying them breaks no
         * privacy invariant; bounded and control-stripped all the same. */
        static IDLETOKEN_TLS char coord_msg[192];
        coord_msg[0] = '\0';
        if (cresp && cstatus != 429) {
            const char *needle = "\"message\":\"";
            const char *at = strstr((const char *)cresp, needle);
            if (at) {
                at += strlen(needle);
                size_t o = 0;
                while (*at && *at != '"' && o + 1 < sizeof coord_msg) {
                    char c = *at++;
                    if (c == '\\' && *at) { at++; c = ' '; } /* collapse escapes */
                    coord_msg[o++] = ((unsigned char)c < 0x20) ? ' ' : c;
                }
                coord_msg[o] = '\0';
            }
        }
        if (cresp) wipe_free(cresp, cresp_len);
        free(reply_to);
        /* "Busy" is not "broken", and the difference must survive this hop.
         *
         * The coordinator answers 429 when every sequence slot is in flight and
         * its queue is full (coord_main.c coord_send_busy_429). The platform
         * already knows what to do with that: reliability.ts classifies 429 as
         * BUSY and pointedly does NOT count it against the provider's failure
         * EWMA, because a popular machine must not be penalised for being
         * popular. Collapsing it into 502 here threw that away — measured on a
         * live chain (results/platform-seam-20260819.md, V6): one refusal took
         * a healthy provider from failCount 0 to 1, strikes 0 to 1, and put it
         * in cooldown, i.e. the machine was removed from routing for being in
         * use. The consumer meanwhile saw "provider error" for something that
         * was nobody's error.
         *
         * Every other non-200 stays 502: those really are "this machine cannot
         * serve you", and pretending otherwise would suppress real faults. */
        if (cstatus == 429) FAIL(429, "coordinator busy: no free sequence slot");
        /* Everything else used to collapse into one flattened string
         * ("upstream coord unreachable or errored"), which made an engine
         * still loading (503), a request the coord rejected (400/500) and a
         * dead loopback socket indistinguishable in the platform log — the
         * only place a relay provider's fault can be diagnosed from. The
         * coord's actual status must survive this hop. The buffer is
         * thread-local because process_sealed also runs on per-connection
         * threads, and *err_msg is read after this function returns. */
        if (!coord_answered)
            FAIL(502, "upstream coord unreachable (no response on loopback)");
        if (cstatus >= 400 && cstatus < 500) {
            /* The coord REJECTED this request (malformed body, template
             * refusal, context overflow). Relay the 4xx untouched: the
             * platform classifies it as "this one request, not this machine"
             * (reliability.ts CAPACITY, no strike) — flattening it into 502
             * once let a single malformed consumer request cool down every
             * provider of the model (2026-08-24). */
            static IDLETOKEN_TLS char coord_rej[256];
            snprintf(coord_rej, sizeof coord_rej,
                     "upstream coord rejected the request (HTTP %d)%s%s",
                     cstatus, coord_msg[0] ? ": " : "", coord_msg);
            *err_status = cstatus;
            *err_msg = coord_rej;
            return -1;
        }
        if (cstatus == 503) {
            /* The coord's own words matter here: 503 covers the engine's real
             * not-serving states (loading, restarting, failed), and asserting
             * "loading" without evidence once sent a whole debugging session
             * the wrong way (2026-08-24). */
            static IDLETOKEN_TLS char coord_unavail[256];
            snprintf(coord_unavail, sizeof coord_unavail,
                     "upstream coord answered 503%s%s",
                     coord_msg[0] ? ": " : " (engine not serving; no detail)",
                     coord_msg);
            FAIL(503, coord_unavail);
        }
        {
            static IDLETOKEN_TLS char coord_err[256];
            if (coord_msg[0])
                snprintf(coord_err, sizeof coord_err,
                         "upstream coord returned HTTP %d: %s", cstatus, coord_msg);
            else
                snprintf(coord_err, sizeof coord_err,
                         "upstream coord returned HTTP %d", cstatus);
            *err_status = 502; *err_msg = coord_err;
            return -1;
        }
    }

    /* -- pick choices[0].message.{content,tool_calls} for the sealed reply -- *
     * The raw tokens (escapes intact) are re-embedded verbatim, so no
     * decode/re-encode round trip is needed.
     *
     * A tool-calling turn has `content: null` and the payload in `tool_calls`
     * (OpenAI shape, produced by the engine). Requiring `content` here used to
     * turn every such turn into "coord response has no content" — a 502 for a
     * response that was perfectly fine. Text and tool_calls are each optional;
     * a response carrying NEITHER is still a real fault. */
    const char *content_tok = NULL; size_t content_len = 0;
    int have_content = json_str_token((const char *)cresp, cresp_len, "content",
                                      &content_tok, &content_len) == 0;
    /* The model's thinking, when it did any. Travels beside the answer and
     * never inside it: the platform renders it into `reasoning_content` /
     * a thinking block, and merging the two here would make the scratchpad
     * indistinguishable from the reply for every consumer downstream.
     *
     * ⚠ `json_key_colon` matches a whole QUOTED key, so the search for
     * "content" above cannot land on "reasoning_content" — which the engine
     * emits FIRST. A substring match here would ship the scratchpad as the
     * answer on every platform request. */
    const char *reason_tok = NULL; size_t reason_len = 0;
    int have_reason = json_str_token((const char *)cresp, cresp_len, "reasoning_content",
                                     &reason_tok, &reason_len) == 0 && reason_len > 0;
    const char *tc_tok = NULL; size_t tc_len = 0;
    int have_tools = json_array_token((const char *)cresp, cresp_len, "tool_calls",
                                      &tc_tok, &tc_len) == 0;
    if (!have_content && !have_tools) {
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

    /* The engine's own token counts, forwarded verbatim (pricing decision D3,
     * platform/pricing/anchor-proposal-v1.md). The platform used to bill input
     * on its own count of the message BODIES, while the engine prefills the
     * whole chat template around them: measured 7 against 15 on one request
     * (results/platform-seam-20260819.md §3), i.e. the provider did the work
     * for fifteen tokens and was paid for seven. Only the coordinator knows the
     * post-template length, so it has to travel this hop.
     *
     * Absent is not zero. A coordinator that reports no usage block (an older
     * build) must leave the object OUT entirely, so the platform can fall back
     * to its own count instead of billing a confident 0 -- and the platform
     * clamps whatever does arrive, because this number is produced by the party
     * it charges for. */
    char usage_frag[64];
    usage_frag[0] = '\0';
    if (json_key_colon((const char *)cresp, cresp_len, "prompt_tokens") >= 0) {
        int in_tok  = json_int_field((const char *)cresp, cresp_len, "prompt_tokens", 0);
        int out_tok = json_int_field((const char *)cresp, cresp_len, "completion_tokens", 0);
        if (in_tok  < 0) in_tok  = 0;
        if (out_tok < 0) out_tok = 0;
        snprintf(usage_frag, sizeof usage_frag,
                 ",\"usage\":{\"in\":%d,\"out\":%d}", in_tok, out_tok);
    }

    /* finish_reason travels so the platform can hand the consumer the real
     * one ("tool_calls" vs "stop" vs "length"); absent on older coords, and
     * the platform then falls back to its historical hard-coded "stop". */
    const char *fr_tok = NULL; size_t fr_len = 0;
    int have_fr = json_str_token((const char *)cresp, cresp_len, "finish_reason",
                                 &fr_tok, &fr_len) == 0 && fr_len < 32;

    size_t reply_cap = content_len + reason_len + tc_len + 224 + sizeof usage_frag;
    char *reply = malloc(reply_cap);
    if (!reply) { wipe_free(cresp, cresp_len); free(reply_to); FAIL(500, "oom"); }
    idletoken_mlock(reply, reply_cap);
    char tc_frag_head[24];
    snprintf(tc_frag_head, sizeof tc_frag_head, "%s", have_tools ? ",\"tool_calls\":" : "");
    int rl = snprintf(reply, reply_cap,
                      "{\"text\":\"%.*s\"%s%.*s%s,\"cache_hit\":%s,\"cached_tokens\":%d%s%s%.*s%s%.*s%s}",
                      (int)content_len, have_content ? content_tok : "",
                      have_reason ? ",\"reasoning\":\"" : "",
                      (int)reason_len, have_reason ? reason_tok : "",
                      have_reason ? "\"" : "",
                      cache_hit ? "true" : "false", cached_tokens, usage_frag,
                      tc_frag_head, (int)tc_len, have_tools ? tc_tok : "",
                      have_fr ? ",\"finish_reason\":\"" : "",
                      (int)fr_len, have_fr ? fr_tok : "",
                      have_fr ? "\"" : "");
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
    if (process_sealed(node, coord_addr, (const char *)body, body_len, NULL,
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
    idletoken_close_fd(a->fd);
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
"                             [--heartbeat-secs N]\n"
"                             [--endpoint URL] [--relay]\n"
"\n"
"  --port N            platform-facing listen port (default 9700; unused in --relay)\n"
"  --coord URL         local coord OpenAI API (default http://127.0.0.1:8000)\n"
"  --coord-token TOK   the coordinator's --api-token, when it requires one.\n"
"                      A coordinator with overflow routing enabled always does\n"
"                      (it refuses to start otherwise), so a sharing machine\n"
"                      that also borrows needs this or every dispatched job\n"
"                      comes back 401. Also via env IDLETOKEN_API_TOKEN.\n"
"  --key-file PATH     persist/reuse the 32-byte provider secret key (0600).\n"
"                      Omit for an ephemeral per-run key.\n"
"  --platform URL      platform gateway; enables registration + heartbeat\n"
"  --jwt TOKEN         platform JWT (the provider account's bearer token)\n"
"  --name NAME         provider display name for registration\n"
"  --heartbeat-secs N  heartbeat interval (default 30; unused in --relay)\n"
"  --endpoint URL      advertised /infer endpoint (default\n"
"                      http://127.0.0.1:<port>/infer; set to this machine's\n"
"                      public URL when registering with a real platform)\n"
"  --model ID          model this cluster serves, reported in the capacity\n"
"                      registration. When supplied, it is cross-checked against\n"
"                      the running coordinator; the coordinator remains authoritative.\n"
"  --quant Q           precision assertion (e.g. Q4_K_M/Q8_0/BF16), cross-checked\n"
"                      against the running coordinator. The coordinator must always\n"
"                      report a concrete precision; wildcard listings are rejected.\n"
"  --relay             reverse connection mode: no listen port at all; the\n"
"                      agent dials OUT to the platform (long poll) and jobs\n"
"                      are pushed back over that connection. Zero home-side\n"
"                      network config (no port forward, no router setup).\n"
"                      Requires --platform and --jwt. Poll doubles as the\n"
"                      heartbeat. Same sealed envelope as direct mode.\n"
"  -h, --help          this help\n");
}

int main(int argc, char **argv) {
#ifdef __linux__
    /* Opt-in (set by the client supervisor): die with the launching client.
     * Same block as coord_main.c/worker_main.c — the agent was the ONE
     * sidecar that never had it, so on Windows a closed client left
     * idletoken-platform-agent.exe running and the installer could not
     * overwrite it until the user killed it by hand (measured 2026-08-24,
     * every upgrade on every machine). */
    if (getenv("IDLETOKEN_DIE_WITH_PARENT")) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() == 1) _exit(0);
    }
#elif defined(_WIN32)
    if (idletoken_win_require_utf8_paths() != 0) return 2;
    idletoken_die_with_parent();
#endif
    int         port           = 9700;
    const char *coord_url      = "http://127.0.0.1:8000"; /* idletoken-coord --api-bind default */
    const char *coord_unix     = NULL;   /* --coord-unix: socket path, shared mode */
    const char *key_file       = NULL;
    const char *platform_url   = NULL;
    const char *jwt            = NULL;
    const char *name           = "idletoken-cluster";
    const char *endpoint_arg   = NULL;
    /* Optional startup assertions. The running coordinator is the only source
     * of the marketplace service identity. */
    const char *model          = "";
    const char *quant          = "";
    int         beat_secs      = 30;
    int         relay          = 0;
    /* Env fallback first, so an explicit flag still wins. Same pattern as the
     * coordinator's own --api-token: the Tauri supervisor passes secrets
     * without putting them through shell argument quoting. */
    {
        const char *t = getenv("IDLETOKEN_API_TOKEN");
        if (t && t[0]) snprintf(g_coord_token, sizeof g_coord_token, "%s", t);
    }
    /* Same for the platform JWT (A-P0-4): the more valuable of the two — it
     * authenticates every poll and spends the account's credits. Env keeps it
     * out of argv, where /proc/<pid>/cmdline exposes it to any local user. An
     * explicit --jwt still wins (set below in the loop). */
    {
        const char *j = getenv("IDLETOKEN_PLATFORM_JWT");
        if (j && j[0]) jwt = j;
    }

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--port")           && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--coord")          && i + 1 < argc) coord_url = argv[++i];
        /* Shared mode: hand the buyer's plaintext to the coordinator over a
         * socket file instead of loopback TCP. Explicit on purpose — if this
         * is given and the socket is not there, the agent fails loudly rather
         * than falling back to a transport this machine's owner can sniff. */
        else if (!strcmp(a, "--coord-unix")     && i + 1 < argc) coord_unix = argv[++i];
        else if (!strcmp(a, "--coord-token")    && i + 1 < argc)
            snprintf(g_coord_token, sizeof g_coord_token, "%s", argv[++i]);
        else if (!strcmp(a, "--key-file")       && i + 1 < argc) key_file = argv[++i];
        else if (!strcmp(a, "--platform")       && i + 1 < argc) platform_url = argv[++i];
        else if (!strcmp(a, "--jwt")            && i + 1 < argc) jwt = argv[++i];
        else if (!strcmp(a, "--name")           && i + 1 < argc) name = argv[++i];
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
    if (platform_url && !jwt) {
        fprintf(stderr, "platform-agent: --platform needs --jwt and --name\n");
        return 2;
    }
    if (relay && (!platform_url || !jwt)) {
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
    /* Socket path wins when given: it is the shared-mode transport, and the
     * caller asked for it explicitly. A missing socket must NOT quietly become
     * a TCP connection — that is the leg this exists to close. */
    if (coord_unix && coord_unix[0]) {
        /* WAIT for the socket rather than refuse (changed 2026-08-21). The
         * client turns sharing on independently of starting the cluster, and
         * its panel promises the agent will wait for the coordinator — but
         * this used to be a one-shot probe whose refusal the supervisor
         * treats as terminal, so "sharing on, then start the cluster" left a
         * dead agent and a switch that said on. Waiting is safe for the same
         * reason refusing was: at no point does a missing socket become a TCP
         * connection — that is still the leg this check exists to close. */
        int probe, waited_s = 0;
        while ((probe = idletoken_connect_unix(coord_unix)) < 0) {
            if (waited_s % 30 == 0)
                fprintf(stderr, "platform-agent: waiting for the coordinator's "
                                "socket %s (%s). Not falling back to loopback "
                                "TCP: the coordinator would then receive "
                                "buyers' prompts over a link this machine's "
                                "owner can capture.\n",
                        coord_unix, strerror(errno));
            sleep(2);
            waited_s += 2;
        }
        idletoken_close_fd(probe);
        snprintf(coord_addr, sizeof(coord_addr), "unix:%s", coord_unix);
        fprintf(stderr, "platform-agent: coordinator link: unix socket %s\n", coord_unix);
    } else if (url_to_addr(coord_url, coord_addr, sizeof(coord_addr)) != 0) {
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

    /* Register a fresh Provider lifetime with the platform. */
    agent_registration reg;
    memset(&reg, 0, sizeof(reg));
    reg.platform_addr = platform_addr; reg.jwt = jwt; reg.name = name;
    reg.pubkey_b64 = pubkey_b64; reg.endpoint = endpoint; reg.coord_addr = coord_addr;
    reg.relay = relay;
    if (platform_url) {
        /* Two gates before anything is registered or beaten, in this order:
         *
         *   1. Is there an engine at all? A machine with nothing loaded is not
         *      a provider -- that is the whole product rule, and waiting for
         *      one is what the client already tells users happens.
         *   2. Is its context worth listing? A machine that cannot serve 8192
         *      tokens per slot should never appear on the market at all, not
         *      appear and then disappoint. Escape hatch on purpose absent --
         *      see §A3 in the cleanup plan.
         *
         * Gated on platform_url because without it this agent lists nothing.
         * The probe also hands back the coordinator's model and precision:
         * this is where the whole declaration comes from. */
        agent_identity live;
        int wait_rc = await_sellable_engine(coord_addr, &live);
        if (wait_rc != 0) { free(pubkey_b64); return wait_rc; }
        int floor_rc = assert_listable_ctx(&live);
        if (floor_rc != 0) { free(pubkey_b64); return floor_rc; }
        /* The COORDINATOR decides; --model/--quant are assertions only. Falling
         * back to startup flags would re-create the stale-listing bug one
         * restart later. A disagreement is worth a line of its own:
         * on the machine that produced this fix, `--quant IQ2_XXS` had been
         * true when the agent started and had been wrong ever since. */
        if (!live.model[0] || !live.quant[0]) {
            char what[224];
            identity_print(&live, what, sizeof(what));
            fprintf(stderr,
                    "platform-agent: refuse: the running coordinator reports an "
                    "incomplete service identity ('%s'). Marketplace registration "
                    "requires /idletoken/v1/stats to contain both a concrete model "
                    "and quant; --model/--quant cannot substitute for live state.\n",
                    what);
            free(pubkey_b64);
            return 4;
        }
        if (model[0] && strcmp(live.model, model) != 0)
            fprintf(stderr, "platform-agent: the coordinator serves model '%s', not the "
                            "'%s' this agent was started with; declaring the coordinator's\n",
                    live.model, model);
        if (quant[0] && strcmp(live.quant, quant) != 0) {
            fprintf(stderr, "platform-agent: the coordinator serves precision '%s', not "
                            "the '%s' this agent was started with; declaring the "
                            "coordinator's\n", live.quant, quant);
        }
        reg.provider_id = platform_register(platform_addr, jwt, name, pubkey_b64,
                                            endpoint, relay, &live);
        if (!reg.provider_id) {
            fprintf(stderr, "platform-agent: registration failed; refusing to start\n");
            return 1;
        }
        reg.declared = live;
        char what[224];
        identity_print(&live, what, sizeof(what));
        printf("  serving          : %s\n", what);
        if (relay)
            printf("  provider id      : %s  (poll = heartbeat)\n", reg.provider_id);
        else
            printf("  provider id      : %s  (heartbeat every %ds)\n", reg.provider_id, beat_secs);
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
        relay_loop(&node, &reg);
        /* unreachable */
        free(reg.provider_id);
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
    int was_sellable = 1;   /* registration proved an engine was there */
    for (;;) {
        if (reg.provider_id && jwt) {
            time_t now = time(NULL);
            if (now - last_beat >= (time_t)beat_secs) {
                /* Before the beat, not after: a beat republishes the load of a
                 * machine the platform still believes is serving the old
                 * triple, and the whole point is that the two agree.
                 *
                 * No engine → no beat at all (engine_sellable()), so the
                 * platform's freshness TTL takes this machine off the shelf
                 * instead of us selling something nothing can serve. */
                agent_identity live;
                if (!engine_sellable(coord_addr, &live)) {
                    if (was_sellable) {
                        fprintf(stderr, "platform-agent: no engine is running behind this "
                                        "agent (coordinator at %s unreachable or still "
                                        "loading); withholding the heartbeat, so this "
                                        "machine drops off the marketplace until it is "
                                        "back.\n", coord_addr);
                        was_sellable = 0;
                    }
                    last_beat = now;
                } else {
                    if (!was_sellable) {
                        fprintf(stderr, "platform-agent: the engine is back; resuming heartbeats\n");
                        was_sellable = 1;
                    }
                    reconcile_identity(&reg, &live);
                    if (platform_heartbeat(platform_addr, jwt, reg.provider_id, coord_addr) != 0)
                        fprintf(stderr, "platform-agent: heartbeat failed (will retry)\n");
                    last_beat = now;
                }
            }
            /* KV cache state: report whenever there is a new session (the
             * platform expires it after >10 min, so frequent re-reporting buys
             * nothing).
             * Reading `dirty` also takes the lock: under concurrency this is a
             * cross-thread visibility question, not "just reading an int". */
            pthread_mutex_lock(&g_prefix_mu);
            const int pfx_dirty = g_prefix.dirty, pfx_n = g_prefix.n;
            pthread_mutex_unlock(&g_prefix_mu);
            if (pfx_dirty && platform_post_cache_state(platform_addr, jwt, reg.provider_id) == 0)
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
        idletoken_close_fd(cfd);
    }

    idletoken_close_fd(lfd);
    free(reg.provider_id);
    free(pubkey_b64);
    idletoken_secure_zero(node.sk, sizeof(node.sk));
    idletoken_munlock(node.sk, sizeof(node.sk));
    return 0;
}
