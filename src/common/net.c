/* IdleToken Cluster — TCP framing helpers. C99. _GNU_SOURCE comes via Makefile. */

#include "idletoken_net.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #ifndef MSG_NOSIGNAL
  #define MSG_NOSIGNAL 0        /* Windows has no SIGPIPE */
  #endif
  #define closesock(fd) closesocket((SOCKET)(fd))
  /* WSAStartup must run once before any socket call. */
  static void ensure_wsa(void) {
      static int inited = 0;
      if (!inited) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); inited = 1; }
  }
#else
  #include <arpa/inet.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <unistd.h>
  #define closesock(fd) close(fd)
  static void ensure_wsa(void) {}
#endif

/* The wire is little-endian. Two architectures we ship to (x86_64, aarch64)
 * are both little-endian, but the pack/unpack helpers serialize byte-by-byte
 * to be safe on any host. */

/* Compile-time check: header struct must serialize to 48 bytes with natural
 * alignment so memcpy round-trips work where endianness already matches. The
 * wire format does NOT rely on struct layout — pack/unpack handle bytes — but
 * keeping the struct at 48 bytes lets us pass it as a single block in tests. */
typedef char idletoken_proto_header_is_48[
    sizeof(idletoken_msg_header) == 48 ? 1 : -1
];

ssize_t idletoken_sendall(int fd, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = send(fd, (const char *)(p + sent), (int)(n - sent), MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) { errno = EPIPE; return -1; }
        sent += (size_t)r;
    }
    return (ssize_t)sent;
}

/* Has the peer closed its end? Non-blocking, never consumes data.
 *
 * Exists so a long generation stops when nobody is listening any more. The
 * streaming path notices a hang-up because its writes fail, but a non-stream
 * request writes nothing until it is done -- so it used to decode to the very
 * end for a client that left, which with an unbounded --max-decode means
 * occupying the whole cluster for hours over an answer no one will read.
 *
 * Only a genuine EOF counts. Readable-with-data is NOT treated as closed: an
 * HTTP/1.1 client may pipeline the next request onto the same connection, and
 * killing a live generation for that would be far worse than the waste.
 * Returns 1 = closed, 0 = still open or unknown (always fail open). */
int idletoken_peer_closed(int fd) {
    if (fd < 0) return 0;
    fd_set rd;
    struct timeval tv = { 0, 0 };
    FD_ZERO(&rd);
#ifdef _WIN32
    FD_SET((SOCKET)fd, &rd);
#else
    if (fd >= FD_SETSIZE) return 0;   /* select() cannot express it; assume open */
    FD_SET(fd, &rd);
#endif
    int n = select(fd + 1, &rd, NULL, NULL, &tv);
    if (n <= 0) return 0;             /* not readable => no EOF pending */
    char b;
    ssize_t r = recv(fd, &b, 1, MSG_PEEK);
    if (r == 0) return 1;             /* orderly shutdown */
    if (r < 0) {
#ifdef _WIN32
        return WSAGetLastError() != WSAEWOULDBLOCK;
#else
        return !(errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR);
#endif
    }
    return 0;                         /* real data waiting (pipelined request) */
}

ssize_t idletoken_recvall(int fd, void *buf, size_t n) {
    unsigned char *p = (unsigned char *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, (char *)(p + got), (int)(n - got), 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) {
            /* EOF mid-message — caller can decide if partial bytes are useful */
            errno = ECONNRESET;
            return (ssize_t)got;
        }
        got += (size_t)r;
    }
    return (ssize_t)got;
}

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
    return  (uint32_t)p[0]
          | ((uint32_t)p[1] << 8)
          | ((uint32_t)p[2] << 16)
          | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

void idletoken_header_pack(const idletoken_msg_header *h, uint8_t out[48]) {
    put_u32(out +  0, h->magic);
    put_u16(out +  4, h->version);
    put_u16(out +  6, h->msg_type);
    put_u64(out +  8, h->payload_bytes);
    put_u64(out + 16, h->request_id);
    put_u32(out + 24, h->stage_id);
    put_u32(out + 28, h->segment_id);
    memcpy(out + 32, h->reserved, 16);
}

void idletoken_header_unpack(const uint8_t in[48], idletoken_msg_header *h) {
    h->magic         = get_u32(in +  0);
    h->version       = get_u16(in +  4);
    h->msg_type      = get_u16(in +  6);
    h->payload_bytes = get_u64(in +  8);
    h->request_id    = get_u64(in + 16);
    h->stage_id      = get_u32(in + 24);
    h->segment_id    = get_u32(in + 28);
    memcpy(h->reserved, in + 32, 16);
}

/* ============================================================================
 * Higher-level message helpers and TCP wiring.
 * ========================================================================== */

int idletoken_send_msg(int fd, const idletoken_msg_header *h,
                    const void *payload, size_t payload_len) {
    if ((size_t)h->payload_bytes != payload_len) {
        errno = EINVAL;
        return -1;
    }
    uint8_t hdr[48];
    idletoken_header_pack(h, hdr);
    if (idletoken_sendall(fd, hdr, 48) != 48) return -1;
    if (payload_len > 0 && payload != NULL) {
        if (idletoken_sendall(fd, payload, payload_len) < 0) return -1;
    }
    return 0;
}

int idletoken_recv_msg(int fd, idletoken_msg_header *out_h,
                    void *out_payload, size_t max_payload) {
    uint8_t hdr[48];
    ssize_t got = idletoken_recvall(fd, hdr, 48);
    if (got != 48) return -1;

    idletoken_header_unpack(hdr, out_h);

    if (out_h->magic != IDLETOKEN_PROTO_MAGIC) { errno = EPROTO; return -1; }
    if (out_h->version > IDLETOKEN_PROTO_VERSION) { errno = EPROTO; return -1; }
    if (out_h->payload_bytes > IDLETOKEN_MAX_PAYLOAD_BYTES) { errno = EMSGSIZE; return -1; }
    if (out_h->segment_id != 0) { errno = EPROTO; return -1; }  /* v0.1 */

    if (out_h->payload_bytes == 0) return 0;
    if (out_h->payload_bytes > max_payload) { errno = EMSGSIZE; return -1; }

    if (idletoken_recvall(fd, out_payload, (size_t)out_h->payload_bytes)
        != (ssize_t)out_h->payload_bytes) return -1;
    return 0;
}

/* Parse "host:port" → host (in 256-byte buffer) + port. Returns 0 / -1. */
static int parse_addr(const char *s, char host_out[256], int *port_out) {
    if (!s || !host_out || !port_out) { errno = EINVAL; return -1; }
    const char *colon = strrchr(s, ':');
    if (!colon) { errno = EINVAL; return -1; }
    size_t hlen = (size_t)(colon - s);
    if (hlen >= 256) { errno = ENAMETOOLONG; return -1; }
    memcpy(host_out, s, hlen);
    host_out[hlen] = '\0';
    char *endp = NULL;
    long p = strtol(colon + 1, &endp, 10);
    /* Allow port 0: an ephemeral bind (the OS picks a free port), used by the
     * discovery resolver's UDP socket and by tests. connect() to 0 simply
     * fails downstream, so this stays safe for the connect path. */
    if (!endp || *endp != '\0' || p < 0 || p > 65535) { errno = EINVAL; return -1; }
    *port_out = (int)p;
    return 0;
}

int idletoken_listen_tcp(const char *bind_addr) {
    ensure_wsa();
    char host[256] = "";
    int port = 0;
    if (parse_addr(bind_addr, host, &port) != 0) return -1;

    /* If host is "0.0.0.0" or "*" or empty, listen on INADDR_ANY. Else
     * resolve to a specific interface. */
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_PASSIVE | AI_NUMERICSERV;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    const char *node = (host[0] == '\0' || !strcmp(host, "0.0.0.0") || !strcmp(host, "*"))
                       ? NULL : host;
    int gai = getaddrinfo(node, port_str, &hints, &res);
    if (gai != 0) { fprintf(stderr, "idletoken-net: getaddrinfo(%s): %s\n", bind_addr, gai_strerror(gai)); errno = EINVAL; return -1; }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { int e = errno; freeaddrinfo(res); errno = e; return -1; }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));

    if (bind(fd, res->ai_addr, res->ai_addrlen) != 0) {
        int e = errno; closesock(fd); freeaddrinfo(res); errno = e; return -1;
    }
    freeaddrinfo(res);

    if (listen(fd, 16) != 0) { int e = errno; closesock(fd); errno = e; return -1; }
    return fd;
}

int idletoken_accept_tcp(int listener) {
    struct sockaddr_in peer;
    socklen_t plen = sizeof(peer);
    int fd = accept(listener, (struct sockaddr *)&peer, &plen);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
    return fd;
}

int idletoken_peer_ip(int fd, char *out, size_t cap) {
    if (!out || cap == 0) { errno = EINVAL; return -1; }
    struct sockaddr_in pa;
    socklen_t plen = sizeof(pa);
    memset(&pa, 0, sizeof(pa));
    if (getpeername(fd, (struct sockaddr *)&pa, &plen) != 0) return -1;
    if (!inet_ntop(AF_INET, &pa.sin_addr, out, (socklen_t)cap)) return -1;
    return 0;
}

int idletoken_connect_tcp(const char *peer_addr) {
    ensure_wsa();
    char host[256] = "";
    int port = 0;
    if (parse_addr(peer_addr, host, &port) != 0) return -1;

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_NUMERICSERV;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    const char *node = (host[0] == '\0') ? "127.0.0.1" : host;
    int gai = getaddrinfo(node, port_str, &hints, &res);
    if (gai != 0) { fprintf(stderr, "idletoken-net: getaddrinfo(%s): %s\n", peer_addr, gai_strerror(gai)); errno = EINVAL; return -1; }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { int e = errno; freeaddrinfo(res); errno = e; return -1; }

    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        int e = errno; closesock(fd); freeaddrinfo(res); errno = e; return -1;
    }
    freeaddrinfo(res);

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
    return fd;
}

void idletoken_close_fd(int fd) {
    if (fd >= 0) closesock(fd);
}

/* ============================================================================
 * UDP datagram helpers (LAN discovery).
 * ========================================================================== */

int idletoken_udp_open(const char *bind_addr) {
    ensure_wsa();
    char host[256] = "";
    int port = 0;
    if (parse_addr(bind_addr, host, &port) != 0) return -1;

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags    = AI_PASSIVE | AI_NUMERICSERV;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    const char *node = (host[0] == '\0' || !strcmp(host, "0.0.0.0") || !strcmp(host, "*"))
                       ? NULL : host;
    int gai = getaddrinfo(node, port_str, &hints, &res);
    if (gai != 0) { fprintf(stderr, "idletoken-net: getaddrinfo(%s): %s\n", bind_addr, gai_strerror(gai)); errno = EINVAL; return -1; }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { int e = errno; freeaddrinfo(res); errno = e; return -1; }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
#ifdef SO_REUSEPORT
    /* Lets several processes on one host (test instances, co-located
     * coord+worker) all bind the discovery port and each get a copy. */
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, (const char *)&one, sizeof(one));
#endif

    if (bind(fd, res->ai_addr, res->ai_addrlen) != 0) {
        int e = errno; closesock(fd); freeaddrinfo(res); errno = e; return -1;
    }
    freeaddrinfo(res);
    return fd;
}

int idletoken_udp_set_broadcast(int fd) {
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, (const char *)&one, sizeof(one)) != 0)
        return -1;
    return 0;
}

int idletoken_udp_sendto(int fd, const char *dst_addr, const void *buf, size_t n) {
    char host[256] = "";
    int port = 0;
    if (parse_addr(dst_addr, host, &port) != 0) return -1;

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags    = AI_NUMERICSERV;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    int gai = getaddrinfo(host, port_str, &hints, &res);
    if (gai != 0) { errno = EINVAL; return -1; }

    ssize_t r = sendto(fd, (const char *)buf, (int)n, 0, res->ai_addr, res->ai_addrlen);
    int e = errno;
    freeaddrinfo(res);
    if (r < 0) { errno = e; return -1; }
    return 0;
}

ssize_t idletoken_udp_recvfrom(int fd, void *buf, size_t cap,
                            char *src_ip, size_t src_ip_cap,
                            int *src_port, int timeout_ms) {
    if (timeout_ms > 0) {
#ifdef _WIN32
        DWORD tv = (DWORD)timeout_ms;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
#else
        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
#endif
    }
    struct sockaddr_in src;
    socklen_t slen = sizeof(src);
    memset(&src, 0, sizeof(src));
    ssize_t r = recvfrom(fd, (char *)buf, (int)cap, 0, (struct sockaddr *)&src, &slen);
    if (r < 0) return -1;
    if (src_ip && src_ip_cap > 0) {
        const char *p = inet_ntop(AF_INET, &src.sin_addr, src_ip, (socklen_t)src_ip_cap);
        if (!p) { if (src_ip_cap) src_ip[0] = '\0'; }
    }
    if (src_port) *src_port = (int)ntohs(src.sin_port);
    return r;
}

int idletoken_local_ipv4(char *out, size_t cap) {
    if (!out || cap == 0) { errno = EINVAL; return -1; }
    snprintf(out, cap, "127.0.0.1");
    ensure_wsa();
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in dst;
    memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(53);
    inet_pton(AF_INET, "8.8.8.8", &dst.sin_addr);
    int rc = -1;
    if (connect(fd, (struct sockaddr *)&dst, sizeof(dst)) == 0) {
        struct sockaddr_in local;
        socklen_t llen = sizeof(local);
        if (getsockname(fd, (struct sockaddr *)&local, &llen) == 0) {
            if (inet_ntop(AF_INET, &local.sin_addr, out, (socklen_t)cap)) rc = 0;
        }
    }
    closesock(fd);
    return rc;
}

/* ============================================================================
 * Payload (de)serialization — idletoken_buf cursor.
 * ========================================================================== */

void idletoken_buf_init(idletoken_buf *b, void *backing, size_t cap) {
    b->buf = (uint8_t *)backing;
    b->cap = cap;
    b->pos = 0;
    b->err = 0;
}

static int buf_need(idletoken_buf *b, size_t n) {
    if (b->err) return -1;
    if (b->pos + n > b->cap) { b->err = 1; return -1; }
    return 0;
}

int idletoken_buf_put_u8 (idletoken_buf *b, uint8_t  v) {
    if (buf_need(b, 1) != 0) return -1;
    b->buf[b->pos++] = v;
    return 0;
}

int idletoken_buf_put_u16(idletoken_buf *b, uint16_t v) {
    if (buf_need(b, 2) != 0) return -1;
    b->buf[b->pos++] = (uint8_t)v;
    b->buf[b->pos++] = (uint8_t)(v >> 8);
    return 0;
}

int idletoken_buf_put_u32(idletoken_buf *b, uint32_t v) {
    if (buf_need(b, 4) != 0) return -1;
    for (int i = 0; i < 4; i++) b->buf[b->pos++] = (uint8_t)(v >> (8 * i));
    return 0;
}

int idletoken_buf_put_u64(idletoken_buf *b, uint64_t v) {
    if (buf_need(b, 8) != 0) return -1;
    for (int i = 0; i < 8; i++) b->buf[b->pos++] = (uint8_t)(v >> (8 * i));
    return 0;
}

int idletoken_buf_put_bytes(idletoken_buf *b, const void *src, size_t n) {
    if (buf_need(b, n) != 0) return -1;
    memcpy(b->buf + b->pos, src, n);
    b->pos += n;
    return 0;
}

int idletoken_buf_put_str(idletoken_buf *b, const char *s) {
    size_t n = s ? strlen(s) : 0;
    if (n > 0xFFFFFFFFu) { b->err = 1; return -1; }
    if (idletoken_buf_put_u32(b, (uint32_t)n) != 0) return -1;
    if (n == 0) return 0;
    return idletoken_buf_put_bytes(b, s, n);
}

int idletoken_buf_get_u8 (idletoken_buf *b, uint8_t  *v) {
    if (buf_need(b, 1) != 0) return -1;
    *v = b->buf[b->pos++];
    return 0;
}

int idletoken_buf_get_u16(idletoken_buf *b, uint16_t *v) {
    if (buf_need(b, 2) != 0) return -1;
    *v = (uint16_t)b->buf[b->pos] | ((uint16_t)b->buf[b->pos+1] << 8);
    b->pos += 2;
    return 0;
}

int idletoken_buf_get_u32(idletoken_buf *b, uint32_t *v) {
    if (buf_need(b, 4) != 0) return -1;
    uint32_t x = 0;
    for (int i = 0; i < 4; i++) x |= (uint32_t)b->buf[b->pos + i] << (8 * i);
    b->pos += 4;
    *v = x;
    return 0;
}

int idletoken_buf_get_u64(idletoken_buf *b, uint64_t *v) {
    if (buf_need(b, 8) != 0) return -1;
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x |= (uint64_t)b->buf[b->pos + i] << (8 * i);
    b->pos += 8;
    *v = x;
    return 0;
}

int idletoken_buf_get_bytes(idletoken_buf *b, void *dst, size_t n) {
    if (buf_need(b, n) != 0) return -1;
    memcpy(dst, b->buf + b->pos, n);
    b->pos += n;
    return 0;
}

int idletoken_buf_get_str(idletoken_buf *b, char *out, size_t max) {
    uint32_t n = 0;
    if (idletoken_buf_get_u32(b, &n) != 0) return -1;
    if (buf_need(b, n) != 0) return -1;
    if (max == 0) { b->pos += n; return 0; }
    size_t copy = n < (max - 1) ? n : (max - 1);
    if (copy > 0) memcpy(out, b->buf + b->pos, copy);
    out[copy] = '\0';
    b->pos += n;
    return 0;
}
