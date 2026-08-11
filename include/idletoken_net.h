/* IdleToken Cluster — TCP framing helpers for the idletoken wire protocol.
 *
 * Pair with include/idletoken_proto.h. C99, no C++. */

#ifndef IDLETOKEN_NET_H
#define IDLETOKEN_NET_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "idletoken_proto.h"

/* Send `n` bytes from `buf` over `fd`. Retries on EINTR; treats peer close
 * as error (EPIPE). Returns n on success, -1 on error with errno set. */
ssize_t idletoken_sendall(int fd, const void *buf, size_t n);

/* Receive exactly `n` bytes into `buf` from `fd`. Retries on EINTR. Returns
 * n on full read. On EOF before n bytes, returns the partial count and sets
 * errno=ECONNRESET. On error returns -1 with errno set. */
ssize_t idletoken_recvall(int fd, void *buf, size_t n);

/* 1 if the peer has closed its end (never consumes data, never blocks).
 * Fails open: 0 also means "cannot tell". */
int idletoken_peer_closed(int fd);

/* Serialize/deserialize the 48-byte fixed header in canonical little-endian
 * wire form. Safe to call on big-endian hosts (not that we target any). */
void idletoken_header_pack(const idletoken_msg_header *h, uint8_t out[48]);
void idletoken_header_unpack(const uint8_t in[48], idletoken_msg_header *h);

/* Higher-level helpers built on the primitives above. */

/* Send a complete message (header + optional payload) atomically (well,
 * as atomically as TCP allows). Returns 0 on success, -1 on error. */
int idletoken_send_msg(int fd, const idletoken_msg_header *h,
                    const void *payload, size_t payload_len);

/* Receive one complete message. Reads the 48-byte header first, validates
 * magic/version, then reads payload_bytes into `out_payload` (caller
 * allocates at least `max_payload` bytes). On success returns 0 and writes
 * the parsed header to `*out_h`. On bad header returns -1 with errno set
 * to EPROTO. Payload truncation (`payload_bytes > max_payload`) returns
 * -1 with errno=EMSGSIZE. */
int idletoken_recv_msg(int fd, idletoken_msg_header *out_h,
                    void *out_payload, size_t max_payload);

/* TCP wiring. Address strings are "host:port". host="" means INADDR_ANY for
 * listen, "127.0.0.1" for connect. */

int idletoken_listen_tcp(const char *bind_addr);   /* returns listener fd, -1 on error */
int idletoken_accept_tcp(int listener);            /* returns conn fd, -1 on error */

/* Numeric IPv4 of the remote end of a connected socket (getpeername +
 * inet_ntop), written to `out`. Lets the coordinator learn a worker's real LAN
 * address so it can rewrite a wildcard ("0.0.0.0") bind into a dialable
 * next/prev-stage address — workers then need zero addressing config. 0 / -1. */
int idletoken_peer_ip(int fd, char *out, size_t cap);
int idletoken_connect_tcp(const char *peer_addr);  /* returns conn fd, -1 on error */
void idletoken_close_fd(int fd);                    /* platform-correct socket close */

/* ---- UDP datagram helpers (LAN discovery) ------------------------------
 *
 * Discovery frames reuse the 48-byte idletoken header + payload, but travel as
 * single UDP datagrams instead of a TCP byte stream. These helpers are the
 * datagram counterparts of the TCP wiring above. */

/* Open a UDP socket bound to `bind_addr` ("host:port"; host "" / "0.0.0.0"
 * = INADDR_ANY). SO_REUSEADDR is set so multiple listeners can share the
 * discovery port on one host (test instances / co-located coord+worker).
 * Returns fd, -1 on error. */
int idletoken_udp_open(const char *bind_addr);

/* Enable SO_BROADCAST on a UDP socket (needed to send to 255.255.255.255). */
int idletoken_udp_set_broadcast(int fd);

/* Send `n` bytes to `dst_addr` ("host:port"). Returns 0 / -1. */
int idletoken_udp_sendto(int fd, const char *dst_addr, const void *buf, size_t n);

/* Receive one datagram (up to `cap` bytes) into `buf`, waiting at most
 * `timeout_ms` (<=0 = block forever). On success returns the byte count and,
 * if non-NULL, writes the sender's numeric IPv4 address to `src_ip` and its
 * source port to `*src_port` (so a reply can reach the sender's actual
 * ephemeral socket, not a fixed port). Returns -1 on error; on timeout returns
 * -1 with errno set to EAGAIN/EWOULDBLOCK. */
ssize_t idletoken_udp_recvfrom(int fd, void *buf, size_t cap,
                            char *src_ip, size_t src_ip_cap,
                            int *src_port, int timeout_ms);

/* Best-effort primary LAN IPv4 of this host, written to `out` (numeric, e.g.
 * "192.168.1.50"). Uses the "connect a UDP socket toward a public address and
 * read the local address the OS picked" trick (no packet is sent). Falls back
 * to "127.0.0.1". Returns 0 / -1. */
int idletoken_local_ipv4(char *out, size_t cap);

/* ---- Payload (de)serialization helpers --------------------------------
 *
 * `idletoken_buf` is a fixed-capacity bump cursor over a caller-owned buffer.
 * On any overflow or get past end, the buffer's `err` flag is set and all
 * subsequent operations become no-ops returning -1. Check `b->err` at end
 * (or just the return value of the final call). Strings are length-prefixed
 * UTF-8 (u32 len + bytes, no NUL on the wire). */
typedef struct {
    uint8_t *buf;
    size_t   cap;
    size_t   pos;
    int      err;
} idletoken_buf;

void idletoken_buf_init(idletoken_buf *b, void *backing, size_t cap);

int idletoken_buf_put_u8 (idletoken_buf *b, uint8_t  v);
int idletoken_buf_put_u16(idletoken_buf *b, uint16_t v);
int idletoken_buf_put_u32(idletoken_buf *b, uint32_t v);
int idletoken_buf_put_u64(idletoken_buf *b, uint64_t v);
int idletoken_buf_put_bytes(idletoken_buf *b, const void *src, size_t n);
int idletoken_buf_put_str(idletoken_buf *b, const char *s);   /* s may be NULL → empty */

int idletoken_buf_get_u8 (idletoken_buf *b, uint8_t  *v);
int idletoken_buf_get_u16(idletoken_buf *b, uint16_t *v);
int idletoken_buf_get_u32(idletoken_buf *b, uint32_t *v);
int idletoken_buf_get_u64(idletoken_buf *b, uint64_t *v);
int idletoken_buf_get_bytes(idletoken_buf *b, void *dst, size_t n);
/* Reads u32 len + bytes; copies up to max-1 bytes to `out`, NUL-terminates.
 * If the wire string exceeds max-1, the extra bytes are still consumed but
 * `out` is truncated and `b->err` is left clear. Returns 0 / -1. */
int idletoken_buf_get_str(idletoken_buf *b, char *out, size_t max);

#endif /* IDLETOKEN_NET_H */
