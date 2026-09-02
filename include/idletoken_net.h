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

/* Bound how long a blocking recv() on `fd` may wait (SO_RCVTIMEO).
 * `timeout_ms <= 0` removes the bound (blocking forever again).
 *
 * Why this is a security primitive and not a convenience (CLUS-05): the
 * coordinator's join loop is serial — accept, pair-auth, HELLO, one worker at a
 * time. Every one of those steps used to be an unbounded blocking read, so a
 * single TCP connection from ANY device on the LAN that connects and then says
 * nothing pinned the coordinator in recv() forever. No join code, no
 * credential, no packet after the SYN: cluster formation simply never
 * completed, and the bounded IDLETOKEN_JOIN_WAIT_S deadline could not fire
 * because the thread was inside recv(), not inside accept().
 *
 * A timeout makes that connection cost the attacker a re-dial per window and
 * turns the failure into an attributable, logged refusal. Returns 0 / -1. */
int idletoken_set_recv_timeout(int fd, int timeout_ms);

/* Read back the current SO_RCVTIMEO in milliseconds (0 = unbounded). Exists so
 * a helper can impose a deadline for the duration of one handshake and put the
 * caller's own setting back afterwards — a shared function that silently leaves
 * its deadline behind would make the NEXT read on that socket fail for a reason
 * its author never wrote down. Returns 0 / -1. */
int idletoken_get_recv_timeout(int fd, int *out_ms);

/* accept() bounded by a deadline. Returns the conn fd, -2 on timeout (a state
 * the caller must be able to tell from an error), -1 on a real failure.
 * Exists because a plain accept() waits forever: a cluster whose second
 * machine never arrives — or quietly went away — left the coordinator blocked
 * with nothing on screen but "starting" (2026-08-15). */
int idletoken_accept_tcp_timeout(int listener, int timeout_ms);

/* Numeric IPv4 of the remote end of a connected socket (getpeername +
 * inet_ntop), written to `out`. Lets the coordinator learn a worker's real LAN
 * address so it can rewrite a wildcard ("0.0.0.0") bind into a dialable
 * next/prev-stage address — workers then need zero addressing config. 0 / -1. */
int idletoken_peer_ip(int fd, char *out, size_t cap);
int idletoken_connect_tcp(const char *peer_addr);  /* returns conn fd, -1 on error */

/* Connect to an AF_UNIX stream socket at `path`. Returns conn fd, -1 on error
 * (errno ENAMETOOLONG when the path does not fit sun_path — 104 bytes on
 * macOS, 108 on Linux, which a deep home directory can genuinely exceed).
 *
 * Exists for the coordinator↔engine link in shared mode: a Unix socket carries
 * no packets a loopback capture can see, and the directory's 0700 keeps other
 * local accounts out (docs/shared-mode-plan-2026-08.md P0-4). Windows has
 * AF_UNIX in winsock since Windows 10 1803; the path is an ordinary filesystem
 * path there too. */
int idletoken_connect_unix(const char *path);

/* Listen on a unix socket, owner-only (0600), replacing any stale file.
 *
 * Why this exists: a TCP listener on loopback is readable by anyone on the
 * machine who can run tcpdump — which on a home node is the owner. A socket
 * file carries the same bytes without ever entering the network stack, so
 * there is no packet to capture, and the filesystem decides who may connect.
 * Accept with idletoken_accept_tcp(); the call is address-family agnostic. */
int idletoken_listen_unix(const char *path);

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

/* True when `ip` (numeric IPv4 optionally with a ":port" suffix, or numeric
 * IPv6) belongs to an overlay/VPN network that tensor traffic must NEVER
 * cross (v2 plan hard invariant #3 — Tailscale's 100.64.0.0/10 CGNAT range:
 * MTU 1280 plus packet reorder deadlocked a measured transfer after 353 MB).
 * Also matches Tailscale's IPv6 range fd7a:115c:a1e0::/48; other ULA space
 * (fc00::/7) is NOT matched — legitimate LANs use it. Compute traffic goes
 * over the real LAN; discovery/pairing may go anywhere. */
int idletoken_ip_is_overlay(const char *ip);

/* True when `h` is exactly 64 hex chars followed by NUL (a 32-byte key in hex
 * — the cluster RPC PSK format both the coord and the worker persist). */
int idletoken_hex64_valid(const char *h);

/* ---- peer-supplied string gates (CLUS-14, CHAIN-08 LAN side) -------------
 *
 * TLS and the pairing handshake decide WHO may speak. They say nothing about
 * WHAT the peer says. Every string in a HELLO — hostname, engine version, rpc
 * endpoint, GPU name — is chosen by the machine on the other end, and the
 * coordinator then splices it into log lines and into the
 * `/idletoken/v1/cluster/status` JSON body with a plain `%s`. An authorized but
 * modified worker could therefore emit `a","role":"coordinator","x":"` as its
 * hostname and rewrite the document the client parses, or embed a newline and
 * forge a log line. Neither needs a parser bug: it is straight string splicing.
 *
 * These are ingress gates. Egress still escapes (belt and braces) — a gate that
 * is the ONLY defence becomes the single point of failure the next refactor
 * removes.
 *
 * `idletoken_peer_label_ok`: a human-readable label (GPU name, version banner).
 * Accepts printable ASCII 0x20..0x7E except `"` and `\`; rejects empty, longer
 * than `max_len`, control bytes, and anything non-ASCII. Non-ASCII is refused
 * on purpose — a 64-byte field truncated mid-UTF-8 produces invalid JSON, and
 * homoglyph hostnames are a display-spoofing surface we get nothing back for.
 *
 * `idletoken_peer_host_ok`: an identity-bearing address/name (hostname, rpc
 * endpoint). Stricter still: `[A-Za-z0-9]`, `.`, `-`, `_`, `:` only.
 *
 * Both return 1 = acceptable, 0 = refuse. NULL is refused. */
int idletoken_peer_label_ok(const char *s, size_t max_len);
int idletoken_peer_host_ok(const char *s, size_t max_len);

/* JSON-escape `src` (`n` bytes) into `dst` (capacity `cap`, always
 * NUL-terminated). Returns the number of bytes written, excluding the
 * terminator; output is truncated rather than overflowed, and never truncated
 * inside an escape sequence. Escapes `"`, `\` and every byte < 0x20; passes the
 * rest through. Bytes >= 0x80 pass through unchanged, so the caller is
 * responsible for UTF-8 validity — which is why the ingress gates above refuse
 * non-ASCII on the peer path. */
size_t idletoken_json_escape(char *dst, size_t cap, const char *src, size_t n);

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

/* Same wire format, but an over-long string is an ERROR (`b->err` set, `out`
 * emptied) instead of a silent truncation.
 *
 * Use this for every field whose VALUE decides something (CLUS-06, CLUS-14):
 *   - an identity — two peers whose 200-char hostnames differ only after byte
 *     63 truncate to the same 64-byte name, and a name that collides is a name
 *     that can be impersonated wherever the code compares names;
 *   - a URL or path the receiver then dials or opens — a truncated URL is a
 *     DIFFERENT URL, and the caller has no way to notice.
 * `idletoken_buf_get_str` remains correct for genuinely cosmetic fields. */
int idletoken_buf_get_str_strict(idletoken_buf *b, char *out, size_t max);

#endif /* IDLETOKEN_NET_H */
