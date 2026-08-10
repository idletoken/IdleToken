/* IdleToken Cluster — LAN discovery + verification-code / account pairing.
 *
 * See include/idletoken_discovery.h for the contract and docs/wire-protocol.md
 * §Discovery for the on-wire beacon/query/auth layouts. Pure C99, no external
 * crypto deps: a self-contained SHA-256 + HMAC-SHA256 backs code derivation,
 * beacon authentication, and the session-key handshake.
 *
 * C only. No C++. */

#include "idletoken_discovery.h"
#include "idletoken_net.h"
#include "idletoken_sha256.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
  #include <winsock2.h>      /* before windows.h */
  #include <ws2tcpip.h>
  #include <windows.h>
  #include <bcrypt.h>        /* BCryptGenRandom — worker links -lbcrypt */
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
#endif

/* SHA-256 now lives in include/idletoken_sha256.h (header-only). It used to be a
 * private static copy right here; model-identity verification in
 * src/common/gguf.c needs the SAME implementation, and two copies of a hash
 * that must agree byte for byte is how a verification mismatch ends up looking
 * like a corrupt model file. Local aliases keep the code below unchanged. */
#define sha256_ctx    idletoken_sha256_ctx
#define sha256_init   idletoken_sha256_init
#define sha256_update idletoken_sha256_update
#define sha256_final  idletoken_sha256_final
#define sha256        idletoken_sha256

void idletoken_hmac_sha256(const uint8_t *key, size_t key_len,
                        const uint8_t *msg, size_t msg_len,
                        uint8_t out[32]) {
    uint8_t k[64]; memset(k, 0, sizeof(k));
    if (key_len > 64) sha256(key, key_len, k);
    else if (key_len > 0) memcpy(k, key, key_len);

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; i++) { ipad[i] = k[i]^0x36; opad[i] = k[i]^0x5c; }

    sha256_ctx c; uint8_t inner[32];
    sha256_init(&c); sha256_update(&c, ipad, 64); sha256_update(&c, msg, msg_len); sha256_final(&c, inner);
    sha256_init(&c); sha256_update(&c, opad, 64); sha256_update(&c, inner, 32); sha256_final(&c, out);
}

/* Constant-time-ish tag compare (short fixed-length tags). */
static int ct_equal(const uint8_t *a, const uint8_t *b, size_t n) {
    uint8_t diff = 0;
    for (size_t i = 0; i < n; i++) diff |= (uint8_t)(a[i] ^ b[i]);
    return diff == 0;
}

/* ============================================================================
 * Randomness.
 * ========================================================================== */

int idletoken_random_bytes(void *buf, size_t n) {
#if defined(_WIN32)
    return BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)n,
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0 ? 0 : -1;
#else
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t got = fread(buf, 1, n, f);
        fclose(f);
        if (got == n) return 0;
    }
    /* Last-resort weak seed — never for production keys, only avoids a hard
     * failure on an exotic host with no /dev/urandom. */
    unsigned char *p = (unsigned char *)buf;
    unsigned long s = (unsigned long)time(NULL) ^ ((unsigned long)(size_t)buf << 3);
    for (size_t i = 0; i < n; i++) { s = s * 1103515245u + 12345u; p[i] = (unsigned char)(s >> 16); }
    return 0;
#endif
}

/* ============================================================================
 * Pairing identity — code + account derivation.
 * ========================================================================== */

const char IDLETOKEN_CODE_ALPHABET[] = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";  /* 32 chars, no O/0/I/1 */

int idletoken_pair_code_mint(char *out, size_t cap) {
    if (!out || cap < IDLETOKEN_CODE_LEN + 1) { errno = EINVAL; return -1; }
    uint8_t r[IDLETOKEN_CODE_LEN];
    if (idletoken_random_bytes(r, sizeof(r)) != 0) return -1;
    for (int i = 0; i < IDLETOKEN_CODE_LEN; i++)
        out[i] = IDLETOKEN_CODE_ALPHABET[r[i] % 32];
    out[IDLETOKEN_CODE_LEN] = '\0';
    return 0;
}

/* Uppercase + strip surrounding whitespace into `dst` (>= IDLETOKEN_CODE_LEN+1). */
static int canon_code(const char *code, char dst[IDLETOKEN_CODE_LEN + 1]) {
    if (!code) return -1;
    while (*code == ' ' || *code == '\t') code++;
    int n = 0;
    for (; code[n] && n < IDLETOKEN_CODE_LEN; n++) {
        char ch = code[n];
        if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 'a' + 'A');
        dst[n] = ch;
    }
    dst[n] = '\0';
    return n;
}

int idletoken_pair_code_valid(const char *code) {
    char c[IDLETOKEN_CODE_LEN + 1];
    int n = canon_code(code, c);
    if (n != IDLETOKEN_CODE_LEN) return 0;
    /* trailing content beyond 6 chars (ignoring whitespace) is invalid */
    const char *p = code;
    while (*p == ' ' || *p == '\t') p++;
    p += IDLETOKEN_CODE_LEN;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '\0') return 0;
    for (int i = 0; i < IDLETOKEN_CODE_LEN; i++)
        if (!strchr(IDLETOKEN_CODE_ALPHABET, c[i])) return 0;
    return 1;
}

/* group_id = HMAC(code, "idletoken-group-v1"); psk = HMAC(code, "idletoken-psk-v1").
 * Using the code as the HMAC key (not the message) means recovering the code
 * from a leaked group_id is a full brute force over 32^6 (~1e9) even knowing
 * the label -- cheap to raise later, adequate for raising the cost of an attack. */
static void derive_from_secret(const char *secret, size_t slen, idletoken_pair_id *id) {
    idletoken_hmac_sha256((const uint8_t *)secret, slen,
                       (const uint8_t *)"idletoken-group-v1", 15, id->group_id);
    idletoken_hmac_sha256((const uint8_t *)secret, slen,
                       (const uint8_t *)"idletoken-psk-v1", 13, id->psk);
}

int idletoken_pair_id_from_code(idletoken_pair_id *id, const char *code) {
    if (!id) { errno = EINVAL; return -1; }
    char c[IDLETOKEN_CODE_LEN + 1];
    if (canon_code(code, c) != IDLETOKEN_CODE_LEN) { errno = EINVAL; return -1; }
    memset(id, 0, sizeof(*id));
    id->mode = IDLETOKEN_PAIR_MODE_CODE;
    derive_from_secret(c, IDLETOKEN_CODE_LEN, id);
    return 0;
}

int idletoken_pair_id_from_account(idletoken_pair_id *id,
                                const char *account_cluster,
                                const char *token,
                                const char *rendezvous_url) {
    if (!id || !account_cluster || !token) { errno = EINVAL; return -1; }
    memset(id, 0, sizeof(*id));
    id->mode = IDLETOKEN_PAIR_MODE_ACCOUNT;
    /* group_id from the cloud cluster label (public-ish, matches same-account
     * machines); psk from the token (proves the account). */
    idletoken_hmac_sha256((const uint8_t *)account_cluster, strlen(account_cluster),
                       (const uint8_t *)"idletoken-group-v1", 15, id->group_id);
    idletoken_hmac_sha256((const uint8_t *)token, strlen(token),
                       (const uint8_t *)"idletoken-psk-v1", 13, id->psk);
    snprintf(id->account_token, sizeof(id->account_token), "%s", token);
    snprintf(id->account_cluster, sizeof(id->account_cluster), "%s", account_cluster);
    if (rendezvous_url) snprintf(id->rendezvous_url, sizeof(id->rendezvous_url), "%s", rendezvous_url);
    return 0;
}

/* ============================================================================
 * Beacon / query datagram framing (48-byte idletoken header + payload over UDP).
 * ========================================================================== */

/* A beacon carries: group_id(32) coord_port(u32) coord_addr(str) mode(u8)
 * reserved(3) nonce(16) tag(16). tag = HMAC(psk,"beacon"|group|addr|nonce). */
static int pack_beacon(uint8_t *out, size_t cap, size_t *out_len,
                       const idletoken_pair_id *id, const char *coord_addr,
                       uint16_t coord_port) {
    uint8_t pay[512];
    idletoken_buf b; idletoken_buf_init(&b, pay, sizeof(pay));
    idletoken_buf_put_bytes(&b, id->group_id, IDLETOKEN_GROUP_ID_BYTES);
    idletoken_buf_put_u32(&b, coord_port);
    idletoken_buf_put_str(&b, coord_addr);
    idletoken_buf_put_u8(&b, (uint8_t)id->mode);
    uint8_t z3[3] = {0}; idletoken_buf_put_bytes(&b, z3, 3);
    uint8_t nonce[IDLETOKEN_PAIR_NONCE_BYTES];
    idletoken_random_bytes(nonce, sizeof(nonce));
    idletoken_buf_put_bytes(&b, nonce, sizeof(nonce));
    /* tag over group_id||coord_addr||nonce, keyed by psk */
    uint8_t macbuf[IDLETOKEN_GROUP_ID_BYTES + 128 + IDLETOKEN_PAIR_NONCE_BYTES], full[32];
    size_t ml = 0;
    memcpy(macbuf + ml, id->group_id, IDLETOKEN_GROUP_ID_BYTES); ml += IDLETOKEN_GROUP_ID_BYTES;
    size_t al = strlen(coord_addr); if (al > 128) al = 128;
    memcpy(macbuf + ml, coord_addr, al); ml += al;
    memcpy(macbuf + ml, nonce, sizeof(nonce)); ml += sizeof(nonce);
    idletoken_hmac_sha256(id->psk, IDLETOKEN_SESSION_KEY_BYTES, macbuf, ml, full);
    idletoken_buf_put_bytes(&b, full, IDLETOKEN_PAIR_TAG_BYTES);
    if (b.err) { errno = EOVERFLOW; return -1; }

    idletoken_msg_header h;
    memset(&h, 0, sizeof(h));
    h.magic = IDLETOKEN_PROTO_MAGIC; h.version = IDLETOKEN_PROTO_VERSION;
    h.msg_type = IDLETOKEN_MSG_DISCOVER_BEACON; h.payload_bytes = b.pos;
    h.stage_id = IDLETOKEN_STAGE_COORD;
    if (cap < 48 + b.pos) { errno = EOVERFLOW; return -1; }
    idletoken_header_pack(&h, out);
    memcpy(out + 48, pay, b.pos);
    *out_len = 48 + b.pos;
    return 0;
}

/* Parse a beacon datagram; verify group_id match + tag. Returns 0 on an
 * authentic matching beacon and writes coord_addr; -1 otherwise. */
static int parse_beacon(const uint8_t *dg, size_t n, const idletoken_pair_id *id,
                        const char *src_ip, char *coord_addr, size_t addr_cap) {
    if (n < 48) return -1;
    idletoken_msg_header h; idletoken_header_unpack(dg, &h);
    if (h.magic != IDLETOKEN_PROTO_MAGIC || h.version != IDLETOKEN_PROTO_VERSION) return -1;
    if (h.msg_type != IDLETOKEN_MSG_DISCOVER_BEACON) return -1;
    if (h.payload_bytes + 48 > n) return -1;

    idletoken_buf b; idletoken_buf_init(&b, (uint8_t *)dg + 48, (size_t)h.payload_bytes);
    uint8_t gid[IDLETOKEN_GROUP_ID_BYTES]; uint32_t cport; char addr[128];
    uint8_t mode, z3[3], nonce[IDLETOKEN_PAIR_NONCE_BYTES], tag[IDLETOKEN_PAIR_TAG_BYTES];
    idletoken_buf_get_bytes(&b, gid, sizeof(gid));
    idletoken_buf_get_u32(&b, &cport);
    idletoken_buf_get_str(&b, addr, sizeof(addr));
    idletoken_buf_get_u8(&b, &mode);
    idletoken_buf_get_bytes(&b, z3, 3);
    idletoken_buf_get_bytes(&b, nonce, sizeof(nonce));
    idletoken_buf_get_bytes(&b, tag, sizeof(tag));
    if (b.err) return -1;
    if (!ct_equal(gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES)) return -1;

    /* Coordinator may advertise an empty/loopback addr and let the receiver use
     * the datagram source ip. Resolve the address to hand back. */
    char resolved[160];
    if (addr[0] == '\0' || !strncmp(addr, "0.0.0.0:", 8)) {
        const char *colon = strrchr(addr, ':');
        snprintf(resolved, sizeof(resolved), "%s:%u",
                 (src_ip && src_ip[0]) ? src_ip : "127.0.0.1",
                 colon ? (unsigned)atoi(colon + 1) : (unsigned)cport);
    } else {
        snprintf(resolved, sizeof(resolved), "%s", addr);
    }

    /* verify tag over group_id||addr(as advertised)||nonce */
    uint8_t macbuf[IDLETOKEN_GROUP_ID_BYTES + 128 + IDLETOKEN_PAIR_NONCE_BYTES], full[32];
    size_t ml = 0;
    memcpy(macbuf + ml, gid, IDLETOKEN_GROUP_ID_BYTES); ml += IDLETOKEN_GROUP_ID_BYTES;
    size_t al = strlen(addr); if (al > 128) al = 128;
    memcpy(macbuf + ml, addr, al); ml += al;
    memcpy(macbuf + ml, nonce, sizeof(nonce)); ml += sizeof(nonce);
    idletoken_hmac_sha256(id->psk, IDLETOKEN_SESSION_KEY_BYTES, macbuf, ml, full);
    if (!ct_equal(full, tag, IDLETOKEN_PAIR_TAG_BYTES)) return -1;

    snprintf(coord_addr, addr_cap, "%s", resolved);
    return 0;
}

/* A query carries: group_id(32) nonce(16) reserved(8). */
static int pack_query(uint8_t *out, size_t cap, size_t *out_len, const idletoken_pair_id *id) {
    uint8_t pay[64];
    idletoken_buf b; idletoken_buf_init(&b, pay, sizeof(pay));
    idletoken_buf_put_bytes(&b, id->group_id, IDLETOKEN_GROUP_ID_BYTES);
    uint8_t nonce[IDLETOKEN_PAIR_NONCE_BYTES]; idletoken_random_bytes(nonce, sizeof(nonce));
    idletoken_buf_put_bytes(&b, nonce, sizeof(nonce));
    uint8_t z8[8] = {0}; idletoken_buf_put_bytes(&b, z8, 8);
    if (b.err) { errno = EOVERFLOW; return -1; }
    idletoken_msg_header h; memset(&h, 0, sizeof(h));
    h.magic = IDLETOKEN_PROTO_MAGIC; h.version = IDLETOKEN_PROTO_VERSION;
    h.msg_type = IDLETOKEN_MSG_DISCOVER_QUERY; h.payload_bytes = b.pos;
    if (cap < 48 + b.pos) { errno = EOVERFLOW; return -1; }
    idletoken_header_pack(&h, out); memcpy(out + 48, pay, b.pos); *out_len = 48 + b.pos;
    return 0;
}

/* Parse a query datagram; return 0 + fill gid if it is a well-formed query
 * for our group. */
static int parse_query(const uint8_t *dg, size_t n, const idletoken_pair_id *id) {
    if (n < 48) return -1;
    idletoken_msg_header h; idletoken_header_unpack(dg, &h);
    if (h.magic != IDLETOKEN_PROTO_MAGIC || h.msg_type != IDLETOKEN_MSG_DISCOVER_QUERY) return -1;
    if (h.payload_bytes + 48 > n) return -1;
    idletoken_buf b; idletoken_buf_init(&b, (uint8_t *)dg + 48, (size_t)h.payload_bytes);
    uint8_t gid[IDLETOKEN_GROUP_ID_BYTES];
    idletoken_buf_get_bytes(&b, gid, sizeof(gid));
    if (b.err) return -1;
    return ct_equal(gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES) ? 0 : -1;
}

/* ============================================================================
 * Portable beacon thread (coord side).
 * ========================================================================== */

typedef struct {
    uint16_t       port;
    idletoken_pair_id id;
    char           self_addr[160];
    int            fd;            /* UDP socket, broadcast-enabled */
    volatile int   stop;
#if defined(_WIN32)
    HANDLE         th;
#else
    pthread_t      th;
    int            th_valid;
#endif
} beacon_ctx;

/* Beacon loop: broadcast a beacon every second AND answer unicast queries. */
static void beacon_loop(beacon_ctx *bc) {
    char bcast_addr[32];
    snprintf(bcast_addr, sizeof(bcast_addr), "255.255.255.255:%u", (unsigned)bc->port);
    char loop_addr[32];
    snprintf(loop_addr, sizeof(loop_addr), "127.0.0.1:%u", (unsigned)bc->port);
    int tick = 0;
    while (!bc->stop) {
        /* (1) periodic broadcast (once/sec) */
        if (tick == 0) {
            uint8_t dg[512]; size_t dl = 0;
            if (pack_beacon(dg, sizeof(dg), &dl, &bc->id, bc->self_addr, bc->port) == 0) {
                idletoken_udp_sendto(bc->fd, bcast_addr, dg, dl);
                idletoken_udp_sendto(bc->fd, loop_addr, dg, dl);  /* same-host joiners */
            }
        }
        tick = (tick + 1) % 5;   /* 5 * 200ms = 1s between broadcasts */

        /* (2) answer queries — recvfrom with a short timeout so we stay
         *     responsive to both the stop flag and the broadcast cadence.
         *     Reply unicast to the querier's ACTUAL source ip:port (its
         *     ephemeral socket), so same-host resolvers get the reply too. */
        uint8_t rq[512]; char src[64] = ""; int sport = 0;
        ssize_t rn = idletoken_udp_recvfrom(bc->fd, rq, sizeof(rq), src, sizeof(src), &sport, 200);
        if (rn > 0 && parse_query(rq, (size_t)rn, &bc->id) == 0 && src[0] && sport > 0) {
            char reply_to[80];
            snprintf(reply_to, sizeof(reply_to), "%s:%d", src, sport);
            uint8_t dg[512]; size_t dl = 0;
            if (pack_beacon(dg, sizeof(dg), &dl, &bc->id, bc->self_addr, bc->port) == 0)
                idletoken_udp_sendto(bc->fd, reply_to, dg, dl);
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI beacon_tramp(LPVOID p) { beacon_loop((beacon_ctx *)p); return 0; }
static int beacon_thread_start(beacon_ctx *bc) {
    bc->th = CreateThread(NULL, 0, beacon_tramp, bc, 0, NULL);
    return bc->th ? 0 : -1;
}
static void beacon_thread_stop(beacon_ctx *bc) {
    bc->stop = 1;
    if (bc->th) { WaitForSingleObject(bc->th, 2000); CloseHandle(bc->th); bc->th = NULL; }
}
#else
static void *beacon_tramp(void *p) { beacon_loop((beacon_ctx *)p); return NULL; }
static int beacon_thread_start(beacon_ctx *bc) {
    if (pthread_create(&bc->th, NULL, beacon_tramp, bc) != 0) return -1;
    bc->th_valid = 1;
    return 0;
}
static void beacon_thread_stop(beacon_ctx *bc) {
    bc->stop = 1;
    if (bc->th_valid) { pthread_join(bc->th, NULL); bc->th_valid = 0; }
}
#endif

/* ============================================================================
 * Provider: BROADCAST (and shared resolve core, reused by SUBNET).
 * ========================================================================== */

typedef struct {
    uint16_t   port;
    beacon_ctx *beacon;   /* non-NULL once advertising */
    int        subnet;    /* 1 = also fan out unicast /24 queries on resolve */
} bcast_state;

static int bcast_advertise(idletoken_discovery *d, const idletoken_pair_id *id, const char *self_addr) {
    bcast_state *s = (bcast_state *)d->state;
    beacon_ctx *bc = (beacon_ctx *)calloc(1, sizeof(*bc));
    if (!bc) return -1;
    bc->port = s->port;
    bc->id = *id;
    snprintf(bc->self_addr, sizeof(bc->self_addr), "%s", self_addr ? self_addr : "");
    char bind[32]; snprintf(bind, sizeof(bind), "0.0.0.0:%u", (unsigned)s->port);
    bc->fd = idletoken_udp_open(bind);
    if (bc->fd < 0) { free(bc); return -1; }
    idletoken_udp_set_broadcast(bc->fd);
    if (beacon_thread_start(bc) != 0) { idletoken_close_fd(bc->fd); free(bc); return -1; }
    s->beacon = bc;
    return 0;
}

/* Send /24 unicast queries from `fd` for group `id`. */
static void subnet_fanout(int fd, const idletoken_pair_id *id, uint16_t port) {
    char ip[64];
    if (idletoken_local_ipv4(ip, sizeof(ip)) != 0) return;
    char *dot = strrchr(ip, '.');
    if (!dot) return;
    *dot = '\0';   /* ip now holds the /24 prefix "a.b.c" */
    uint8_t dg[128]; size_t dl = 0;
    if (pack_query(dg, sizeof(dg), &dl, id) != 0) return;
    for (int host = 1; host <= 254; host++) {
        char dst[80];
        snprintf(dst, sizeof(dst), "%s.%d:%u", ip, host, (unsigned)port);
        idletoken_udp_sendto(fd, dst, dg, dl);
    }
}

static int bcast_resolve(idletoken_discovery *d, const idletoken_pair_id *id,
                         char *out_addr, size_t out_cap, int timeout_ms) {
    bcast_state *s = (bcast_state *)d->state;
    /* Bind an EPHEMERAL port to send queries and receive the coordinator's
     * unicast reply. This avoids colliding with a co-located coordinator (or
     * another resolver) already bound to the discovery port on this host, and
     * still works cross-host because the coordinator replies to our real
     * source ip:port. */
    int fd = idletoken_udp_open("0.0.0.0:0");
    if (fd < 0) return -1;
    idletoken_udp_set_broadcast(fd);

    char bcast_addr[32];
    snprintf(bcast_addr, sizeof(bcast_addr), "255.255.255.255:%u", (unsigned)s->port);

    /* Active probe: broadcast a query (and, in subnet mode, unicast the /24) so
     * a coordinator answers immediately instead of us waiting for its beacon. */
    uint8_t q[128]; size_t ql = 0;
    if (pack_query(q, sizeof(q), &ql, id) == 0) {
        idletoken_udp_sendto(fd, bcast_addr, q, ql);
        idletoken_udp_sendto(fd, "127.0.0.1", q, ql);   /* harmless if unbound */
    }
    if (s->subnet) subnet_fanout(fd, id, s->port);

    int waited = 0;
    const int slice = 300;
    while (timeout_ms <= 0 || waited < timeout_ms) {
        uint8_t dg[512]; char src[64] = ""; int sport = 0;
        ssize_t n = idletoken_udp_recvfrom(fd, dg, sizeof(dg), src, sizeof(src), &sport, slice);
        if (n > 0 && parse_beacon(dg, (size_t)n, id, src, out_addr, out_cap) == 0) {
            idletoken_close_fd(fd);
            return 0;
        }
        waited += slice;
        /* re-probe every ~1s in case the first query was dropped */
        if ((waited % 900) < slice) {
            if (ql) { idletoken_udp_sendto(fd, bcast_addr, q, ql); }
            if (s->subnet) subnet_fanout(fd, id, s->port);
        }
    }
    idletoken_close_fd(fd);
    errno = ETIMEDOUT;
    return -1;
}

static void bcast_close(idletoken_discovery *d) {
    if (!d) return;
    bcast_state *s = (bcast_state *)d->state;
    if (s) {
        if (s->beacon) { beacon_thread_stop(s->beacon); idletoken_close_fd(s->beacon->fd); free(s->beacon); }
        free(s);
    }
    free(d);
}

static idletoken_discovery *bcast_new(uint16_t port, int subnet, idletoken_discovery_kind kind) {
    idletoken_discovery *d = (idletoken_discovery *)calloc(1, sizeof(*d));
    bcast_state *s = (bcast_state *)calloc(1, sizeof(*s));
    if (!d || !s) { free(d); free(s); return NULL; }
    s->port = port ? port : (uint16_t)IDLETOKEN_DISCOVERY_PORT;
    s->subnet = subnet;
    d->kind = kind; d->state = s;
    d->advertise = bcast_advertise;
    d->resolve = bcast_resolve;
    d->destroy = bcast_close;
    return d;
}

idletoken_discovery *idletoken_discovery_broadcast(uint16_t udp_port) {
    return bcast_new(udp_port, 0, IDLETOKEN_DISCOVERY_BROADCAST);
}
idletoken_discovery *idletoken_discovery_subnet(uint16_t udp_port) {
    return bcast_new(udp_port, 1, IDLETOKEN_DISCOVERY_SUBNET);
}

/* ============================================================================
 * Provider: MANUAL.
 * ========================================================================== */

static int manual_advertise(idletoken_discovery *d, const idletoken_pair_id *id, const char *self_addr) {
    (void)d; (void)id; (void)self_addr; return 0;   /* nothing to announce */
}
static int manual_resolve(idletoken_discovery *d, const idletoken_pair_id *id,
                          char *out_addr, size_t out_cap, int timeout_ms) {
    (void)id; (void)timeout_ms;
    const char *addr = (const char *)d->state;
    if (!addr || !addr[0]) { errno = EINVAL; return -1; }
    snprintf(out_addr, out_cap, "%s", addr);
    return 0;
}
static void manual_close(idletoken_discovery *d) { if (d) { free(d->state); free(d); } }

idletoken_discovery *idletoken_discovery_manual(const char *coord_addr) {
    if (!coord_addr) return NULL;
    idletoken_discovery *d = (idletoken_discovery *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->kind = IDLETOKEN_DISCOVERY_MANUAL;
    d->state = strdup(coord_addr);
    d->advertise = manual_advertise;
    d->resolve = manual_resolve;
    d->destroy = manual_close;
    return d;
}

/* ============================================================================
 * Provider: MOCK — deterministic, in-process, no sockets.
 * ========================================================================== */

/* A tiny process-global registry keyed by group_id. Enough for unit tests and
 * the mock/real parity check; not thread-heavy by design. */
#define MOCK_SLOTS 8
static struct { int used; uint8_t gid[IDLETOKEN_GROUP_ID_BYTES]; char addr[160]; } g_mock[MOCK_SLOTS];

static int mock_advertise(idletoken_discovery *d, const idletoken_pair_id *id, const char *self_addr) {
    (void)d;
    for (int i = 0; i < MOCK_SLOTS; i++) {
        if (g_mock[i].used && ct_equal(g_mock[i].gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES)) {
            snprintf(g_mock[i].addr, sizeof(g_mock[i].addr), "%s", self_addr ? self_addr : "");
            return 0;
        }
    }
    for (int i = 0; i < MOCK_SLOTS; i++) {
        if (!g_mock[i].used) {
            g_mock[i].used = 1;
            memcpy(g_mock[i].gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES);
            snprintf(g_mock[i].addr, sizeof(g_mock[i].addr), "%s", self_addr ? self_addr : "");
            return 0;
        }
    }
    errno = ENOSPC; return -1;
}
static int mock_resolve(idletoken_discovery *d, const idletoken_pair_id *id,
                        char *out_addr, size_t out_cap, int timeout_ms) {
    (void)d; (void)timeout_ms;
    for (int i = 0; i < MOCK_SLOTS; i++) {
        if (g_mock[i].used && ct_equal(g_mock[i].gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES)) {
            snprintf(out_addr, out_cap, "%s", g_mock[i].addr);
            return 0;
        }
    }
    errno = ETIMEDOUT; return -1;
}
static void mock_close(idletoken_discovery *d) { free(d); }

idletoken_discovery *idletoken_discovery_mock(void) {
    idletoken_discovery *d = (idletoken_discovery *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->kind = IDLETOKEN_DISCOVERY_MOCK;
    d->advertise = mock_advertise;
    d->resolve = mock_resolve;
    d->destroy = mock_close;
    return d;
}

/* ============================================================================
 * Provider: RENDEZVOUS (cloud, account mode) — minimal HTTP/1.1 client.
 * ========================================================================== */

/* Hex-encode group_id for the JSON body. */
static void hex32(const uint8_t *in, char out[65]) {
    static const char *h = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[i*2] = h[in[i] >> 4]; out[i*2+1] = h[in[i] & 15]; }
    out[64] = '\0';
}

/* POST `body` (JSON) to rendezvous_url ("host:port") path /v1/rendezvous with a
 * bearer token; return the response body (malloc'd, NUL-terminated) or NULL. */
static char *rdv_post(const char *addr, const char *token, const char *body) {
    int fd = idletoken_connect_tcp(addr);
    if (fd < 0) return NULL;
    char host[256] = ""; { const char *c = strrchr(addr, ':'); size_t hl = c ? (size_t)(c-addr) : strlen(addr); if (hl > 255) hl = 255; memcpy(host, addr, hl); host[hl] = '\0'; }
    char req[1536];
    int rl = snprintf(req, sizeof(req),
        "POST /v1/rendezvous HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\n"
        "Authorization: Bearer %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",
        host, token ? token : "", strlen(body), body);
    if (rl <= 0 || idletoken_sendall(fd, req, (size_t)rl) < 0) { idletoken_close_fd(fd); return NULL; }

    char *resp = (char *)malloc(8192); size_t cap = 8192, len = 0;
    if (!resp) { idletoken_close_fd(fd); return NULL; }
    for (;;) {
        if (len + 1 >= cap) { cap *= 2; char *n = realloc(resp, cap); if (!n) { free(resp); idletoken_close_fd(fd); return NULL; } resp = n; }
        ssize_t r = recv(fd, resp + len, (int)(cap - len - 1), 0);
        if (r <= 0) break;
        len += (size_t)r;
    }
    idletoken_close_fd(fd);
    resp[len] = '\0';
    char *body_p = strstr(resp, "\r\n\r\n");
    if (!body_p) { free(resp); return NULL; }
    char *out = strdup(body_p + 4);
    free(resp);
    return out;
}

/* Extract a JSON string field `"key":"value"` into out (best-effort, flat). */
static int json_str_field(const char *json, const char *key, char *out, size_t cap) {
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return -1;
    p = strchr(p + strlen(pat), ':'); if (!p) return -1;
    p++; while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return -1;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) { out[i++] = *p++; }
    out[i] = '\0';
    return (*p == '"') ? 0 : -1;
}

typedef struct { idletoken_pair_id id; char self_addr[160]; volatile int stop;
#if defined(_WIN32)
    HANDLE th;
#else
    pthread_t th; int th_valid;
#endif
} rdv_state;

static void rdv_register_once(rdv_state *s, const char *role) {
    if (!s->id.rendezvous_url[0]) return;
    char gid[65]; hex32(s->id.group_id, gid);
    char body[512];
    snprintf(body, sizeof(body),
             "{\"group\":\"%s\",\"role\":\"%s\",\"addr\":\"%s\"}", gid, role, s->self_addr);
    char *resp = rdv_post(s->id.rendezvous_url, s->id.account_token, body);
    free(resp);
}

static void rdv_loop(rdv_state *s) {
    while (!s->stop) {
        rdv_register_once(s, "coord");
        for (int i = 0; i < 25 && !s->stop; i++) {
#if defined(_WIN32)
            Sleep(200);
#else
            usleep(200000);
#endif
        }
    }
}
#if defined(_WIN32)
static DWORD WINAPI rdv_tramp(LPVOID p) { rdv_loop((rdv_state *)p); return 0; }
#else
static void *rdv_tramp(void *p) { rdv_loop((rdv_state *)p); return NULL; }
#endif

static int rdv_advertise(idletoken_discovery *d, const idletoken_pair_id *id, const char *self_addr) {
    rdv_state *s = (rdv_state *)d->state;
    s->id = *id;
    snprintf(s->self_addr, sizeof(s->self_addr), "%s", self_addr ? self_addr : "");
    if (!s->id.rendezvous_url[0]) { errno = EINVAL; return -1; }
    s->stop = 0;
#if defined(_WIN32)
    s->th = CreateThread(NULL, 0, rdv_tramp, s, 0, NULL);
    return s->th ? 0 : -1;
#else
    if (pthread_create(&s->th, NULL, rdv_tramp, s) != 0) return -1;
    s->th_valid = 1; return 0;
#endif
}

static int rdv_resolve(idletoken_discovery *d, const idletoken_pair_id *id,
                       char *out_addr, size_t out_cap, int timeout_ms) {
    (void)d;
    if (!id->rendezvous_url[0]) { errno = EINVAL; return -1; }
    char lan[64]; idletoken_local_ipv4(lan, sizeof(lan));
    char gid[65]; hex32(id->group_id, gid);
    char body[512];
    snprintf(body, sizeof(body),
             "{\"group\":\"%s\",\"role\":\"worker\",\"addr\":\"%s\"}", gid, lan);
    int waited = 0;
    while (timeout_ms <= 0 || waited < timeout_ms) {
        char *resp = rdv_post(id->rendezvous_url, id->account_token, body);
        if (resp) {
            char coord[160] = "";
            if (json_str_field(resp, "coord", coord, sizeof(coord)) == 0 && coord[0]) {
                snprintf(out_addr, out_cap, "%s", coord);
                free(resp);
                return 0;
            }
            free(resp);
        }
#if defined(_WIN32)
        Sleep(700);
#else
        usleep(700000);
#endif
        waited += 700;
    }
    errno = ETIMEDOUT; return -1;
}

static void rdv_close(idletoken_discovery *d) {
    if (!d) return;
    rdv_state *s = (rdv_state *)d->state;
    if (s) {
        s->stop = 1;
#if defined(_WIN32)
        if (s->th) { WaitForSingleObject(s->th, 2000); CloseHandle(s->th); }
#else
        if (s->th_valid) pthread_join(s->th, NULL);
#endif
        free(s);
    }
    free(d);
}

idletoken_discovery *idletoken_discovery_rendezvous(void) {
    idletoken_discovery *d = (idletoken_discovery *)calloc(1, sizeof(*d));
    rdv_state *s = (rdv_state *)calloc(1, sizeof(*s));
    if (!d || !s) { free(d); free(s); return NULL; }
    d->kind = IDLETOKEN_DISCOVERY_RENDEZVOUS; d->state = s;
    d->advertise = rdv_advertise; d->resolve = rdv_resolve; d->destroy = rdv_close;
    return d;
}

/* ============================================================================
 * Provider: MULTI — degrade order broadcast -> manual -> subnet -> rendezvous.
 * ========================================================================== */

typedef struct {
    idletoken_discovery *broadcast;
    idletoken_discovery *manual;      /* may be NULL */
    idletoken_discovery *subnet;
    idletoken_discovery *rendezvous;  /* used only in account mode */
} multi_state;

static int multi_advertise(idletoken_discovery *d, const idletoken_pair_id *id, const char *self_addr) {
    multi_state *s = (multi_state *)d->state;
    int ok = -1;
    if (s->broadcast && s->broadcast->advertise(s->broadcast, id, self_addr) == 0) ok = 0;
    if (id->mode == IDLETOKEN_PAIR_MODE_ACCOUNT && id->rendezvous_url[0] && s->rendezvous)
        if (s->rendezvous->advertise(s->rendezvous, id, self_addr) == 0) ok = 0;
    return ok;
}

/* Try each sub-provider for a slice of the total budget, in degrade order. */
static int multi_resolve(idletoken_discovery *d, const idletoken_pair_id *id,
                         char *out_addr, size_t out_cap, int timeout_ms) {
    multi_state *s = (multi_state *)d->state;
    /* Manual, if present, is instant and authoritative — try it first with 0
     * budget so an explicit address always wins without waiting. */
    if (s->manual && s->manual->resolve(s->manual, id, out_addr, out_cap, 0) == 0)
        return 0;
    /* Broadcast gets the bulk of the budget (home main path). */
    int budget = timeout_ms > 0 ? timeout_ms : 15000;
    int t_bcast = budget * 6 / 10;
    if (s->broadcast && s->broadcast->resolve(s->broadcast, id, out_addr, out_cap, t_bcast) == 0)
        return 0;
    int t_subnet = budget * 3 / 10;
    if (s->subnet && s->subnet->resolve(s->subnet, id, out_addr, out_cap, t_subnet) == 0)
        return 0;
    if (id->mode == IDLETOKEN_PAIR_MODE_ACCOUNT && id->rendezvous_url[0] && s->rendezvous) {
        int t_rdv = budget - t_bcast - t_subnet;
        if (s->rendezvous->resolve(s->rendezvous, id, out_addr, out_cap, t_rdv) == 0)
            return 0;
    }
    errno = ETIMEDOUT; return -1;
}

static void multi_close(idletoken_discovery *d) {
    if (!d) return;
    multi_state *s = (multi_state *)d->state;
    if (s) {
        if (s->broadcast)  s->broadcast->destroy(s->broadcast);
        if (s->manual)     s->manual->destroy(s->manual);
        if (s->subnet)     s->subnet->destroy(s->subnet);
        if (s->rendezvous) s->rendezvous->destroy(s->rendezvous);
        free(s);
    }
    free(d);
}

idletoken_discovery *idletoken_discovery_multi(uint16_t udp_port, const char *manual_addr) {
    idletoken_discovery *d = (idletoken_discovery *)calloc(1, sizeof(*d));
    multi_state *s = (multi_state *)calloc(1, sizeof(*s));
    if (!d || !s) { free(d); free(s); return NULL; }
    s->broadcast  = idletoken_discovery_broadcast(udp_port);
    s->subnet     = idletoken_discovery_subnet(udp_port);
    s->rendezvous = idletoken_discovery_rendezvous();
    if (manual_addr && manual_addr[0]) s->manual = idletoken_discovery_manual(manual_addr);
    d->kind = IDLETOKEN_DISCOVERY_MULTI; d->state = s;
    d->advertise = multi_advertise; d->resolve = multi_resolve; d->destroy = multi_close;
    return d;
}

/* ============================================================================
 * TCP pairing auth preamble (PAIR_HELLO / PAIR_ACCEPT).
 * ========================================================================== */

static void derive_session(const idletoken_pair_id *id,
                           const uint8_t wnonce[IDLETOKEN_PAIR_NONCE_BYTES],
                           const uint8_t cnonce[IDLETOKEN_PAIR_NONCE_BYTES],
                           uint8_t out[IDLETOKEN_SESSION_KEY_BYTES]) {
    uint8_t m[16 + IDLETOKEN_PAIR_NONCE_BYTES * 2];
    memcpy(m, "pair-session-v1", 15); m[15] = 0;
    memcpy(m + 16, wnonce, IDLETOKEN_PAIR_NONCE_BYTES);
    memcpy(m + 16 + IDLETOKEN_PAIR_NONCE_BYTES, cnonce, IDLETOKEN_PAIR_NONCE_BYTES);
    idletoken_hmac_sha256(id->psk, IDLETOKEN_SESSION_KEY_BYTES, m, sizeof(m), out);
}

/* tag = HMAC(psk, label || group_id || nonces...) truncated to 16. */
static void auth_tag(const idletoken_pair_id *id, const char *label,
                     const uint8_t *n1, const uint8_t *n2, uint8_t out[IDLETOKEN_PAIR_TAG_BYTES]) {
    uint8_t m[8 + IDLETOKEN_GROUP_ID_BYTES + IDLETOKEN_PAIR_NONCE_BYTES * 2]; size_t l = 0;
    size_t ll = strlen(label); if (ll > 8) ll = 8;
    memcpy(m + l, label, ll); l += ll;
    memcpy(m + l, id->group_id, IDLETOKEN_GROUP_ID_BYTES); l += IDLETOKEN_GROUP_ID_BYTES;
    memcpy(m + l, n1, IDLETOKEN_PAIR_NONCE_BYTES); l += IDLETOKEN_PAIR_NONCE_BYTES;
    if (n2) { memcpy(m + l, n2, IDLETOKEN_PAIR_NONCE_BYTES); l += IDLETOKEN_PAIR_NONCE_BYTES; }
    uint8_t full[32];
    idletoken_hmac_sha256(id->psk, IDLETOKEN_SESSION_KEY_BYTES, m, l, full);
    memcpy(out, full, IDLETOKEN_PAIR_TAG_BYTES);
}

int idletoken_pair_client_auth(int fd, const idletoken_pair_id *id,
                            uint8_t session_key[IDLETOKEN_SESSION_KEY_BYTES]) {
    uint8_t wnonce[IDLETOKEN_PAIR_NONCE_BYTES];
    idletoken_random_bytes(wnonce, sizeof(wnonce));
    uint8_t wtag[IDLETOKEN_PAIR_TAG_BYTES];
    auth_tag(id, "pair-w", wnonce, NULL, wtag);

    uint8_t pay[IDLETOKEN_GROUP_ID_BYTES + IDLETOKEN_PAIR_NONCE_BYTES + IDLETOKEN_PAIR_TAG_BYTES];
    idletoken_buf b; idletoken_buf_init(&b, pay, sizeof(pay));
    idletoken_buf_put_bytes(&b, id->group_id, IDLETOKEN_GROUP_ID_BYTES);
    idletoken_buf_put_bytes(&b, wnonce, sizeof(wnonce));
    idletoken_buf_put_bytes(&b, wtag, sizeof(wtag));

    idletoken_msg_header h; memset(&h, 0, sizeof(h));
    h.magic = IDLETOKEN_PROTO_MAGIC; h.version = IDLETOKEN_PROTO_VERSION;
    h.msg_type = IDLETOKEN_MSG_PAIR_HELLO; h.payload_bytes = b.pos;
    if (idletoken_send_msg(fd, &h, pay, b.pos) != 0) return -1;

    uint8_t rp[128]; idletoken_msg_header rh;
    if (idletoken_recv_msg(fd, &rh, rp, sizeof(rp)) != 0) return -1;
    if (rh.msg_type != IDLETOKEN_MSG_PAIR_ACCEPT) { errno = EACCES; return -1; }
    idletoken_buf rb; idletoken_buf_init(&rb, rp, rh.payload_bytes);
    uint8_t accepted, z3[3], cnonce[IDLETOKEN_PAIR_NONCE_BYTES], ctag[IDLETOKEN_PAIR_TAG_BYTES];
    idletoken_buf_get_u8(&rb, &accepted);
    idletoken_buf_get_bytes(&rb, z3, 3);
    idletoken_buf_get_bytes(&rb, cnonce, sizeof(cnonce));
    idletoken_buf_get_bytes(&rb, ctag, sizeof(ctag));
    if (rb.err || !accepted) { errno = EACCES; return -1; }

    uint8_t expect[IDLETOKEN_PAIR_TAG_BYTES];
    auth_tag(id, "pair-c", wnonce, cnonce, expect);
    if (!ct_equal(expect, ctag, IDLETOKEN_PAIR_TAG_BYTES)) { errno = EACCES; return -1; }

    if (session_key) derive_session(id, wnonce, cnonce, session_key);
    return 0;
}

int idletoken_pair_server_auth(int fd, const idletoken_pair_id *id,
                            uint8_t session_key[IDLETOKEN_SESSION_KEY_BYTES]) {
    uint8_t rp[128]; idletoken_msg_header rh;
    if (idletoken_recv_msg(fd, &rh, rp, sizeof(rp)) != 0) return -1;
    if (rh.msg_type != IDLETOKEN_MSG_PAIR_HELLO) { errno = EPROTO; return -1; }
    idletoken_buf rb; idletoken_buf_init(&rb, rp, rh.payload_bytes);
    uint8_t gid[IDLETOKEN_GROUP_ID_BYTES], wnonce[IDLETOKEN_PAIR_NONCE_BYTES], wtag[IDLETOKEN_PAIR_TAG_BYTES];
    idletoken_buf_get_bytes(&rb, gid, sizeof(gid));
    idletoken_buf_get_bytes(&rb, wnonce, sizeof(wnonce));
    idletoken_buf_get_bytes(&rb, wtag, sizeof(wtag));

    int ok = !rb.err
             && ct_equal(gid, id->group_id, IDLETOKEN_GROUP_ID_BYTES);
    if (ok) {
        uint8_t expect[IDLETOKEN_PAIR_TAG_BYTES];
        auth_tag(id, "pair-w", wnonce, NULL, expect);
        ok = ct_equal(expect, wtag, IDLETOKEN_PAIR_TAG_BYTES);
    }

    uint8_t cnonce[IDLETOKEN_PAIR_NONCE_BYTES];
    idletoken_random_bytes(cnonce, sizeof(cnonce));
    uint8_t ctag[IDLETOKEN_PAIR_TAG_BYTES] = {0};
    if (ok) auth_tag(id, "pair-c", wnonce, cnonce, ctag);

    uint8_t pay[1 + 3 + IDLETOKEN_PAIR_NONCE_BYTES + IDLETOKEN_PAIR_TAG_BYTES];
    idletoken_buf b; idletoken_buf_init(&b, pay, sizeof(pay));
    idletoken_buf_put_u8(&b, ok ? 1 : 0);
    uint8_t z3[3] = {0}; idletoken_buf_put_bytes(&b, z3, 3);
    idletoken_buf_put_bytes(&b, cnonce, sizeof(cnonce));
    idletoken_buf_put_bytes(&b, ctag, sizeof(ctag));

    idletoken_msg_header h; memset(&h, 0, sizeof(h));
    h.magic = IDLETOKEN_PROTO_MAGIC; h.version = IDLETOKEN_PROTO_VERSION;
    h.msg_type = IDLETOKEN_MSG_PAIR_ACCEPT; h.payload_bytes = b.pos;
    idletoken_send_msg(fd, &h, pay, b.pos);

    if (!ok) { errno = EACCES; return -1; }
    if (session_key) derive_session(id, wnonce, cnonce, session_key);
    return 0;
}
