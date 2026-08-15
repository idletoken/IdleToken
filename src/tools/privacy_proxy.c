/* IdleToken privacy proxy — the sealed-envelope terminator (G-PRIV model X).
 *
 * Runs ON THE TRUSTED decrypting node, in FRONT of idletoken-coord. It is the one
 * plaintext window: it holds the node's X25519 secret key, opens sealed
 * requests from external consumers, forwards the plaintext to the local coord
 * HTTP API over LOOPBACK (127.0.0.1 — plaintext never leaves this host), then
 * seals the reply back to the consumer's session key.
 *
 * Why a separate process instead of editing coord: zero changes to
 * coord_main.c — normal (non-private) operation is byte-identical, and the
 * engine stays independently testable (principle 17: keep front end and back
 * end decoupled). Turn privacy on by
 * putting this proxy in front; turn it off by not running it.
 *
 * Endpoints (consumer-facing):
 *   GET  /idletoken/v1/privacy/pubkey            -> {"pubkey":"<64-hex>","alg":"x25519"}
 *   POST /idletoken/v1/privacy/messages          -> sealed; forwards to coord /v1/messages
 *   POST /idletoken/v1/privacy/chat/completions  -> sealed; forwards to /v1/chat/completions
 *   (any other)                        -> 404
 *
 * The request body of the POSTs is a sealed request envelope (see
 * include/idletoken_privacy_http.h); the 200 response body is a sealed response
 * envelope. No plaintext ever appears on the consumer-facing socket.
 *
 * Key management:
 *   --key-file PATH   load/persist the 32-byte node secret key (raw). If the
 *                     file is missing it is generated (0600) and its PUBLIC key
 *                     printed. Reuse the same file to keep a stable identity.
 *   (no --key-file)   generate an ephemeral key for this run; print the pubkey.
 *
 * C only. No C++.
 */

#include "idletoken_privacy.h"
#include "idletoken_privacy_http.h"
#include "idletoken_net.h"
#include "idletoken_http.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- tiny HTTP client: POST a JSON body to the local coord, read reply ---- */
/* Returns malloc'd response body (caller frees) and sets *out_len; NULL on
 * error. Only what the proxy needs: fixed Content-Length, loopback, blocking. */
static uint8_t *http_post_json(const char *upstream_addr, const char *path,
                               const uint8_t *body, size_t body_len,
                               size_t *out_len) {
    int fd = idletoken_connect_tcp(upstream_addr);
    if (fd < 0) return NULL;

    char head[512];
    int hn = snprintf(head, sizeof(head),
                      "POST %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n",
                      path, upstream_addr, body_len);
    if (hn < 0 || (size_t)hn >= sizeof(head)) { close(fd); return NULL; }
    if (idletoken_sendall(fd, head, (size_t)hn) < 0 ||
        (body_len && idletoken_sendall(fd, body, body_len) < 0)) {
        close(fd); return NULL;
    }

    /* Read the whole response (headers + body) until EOF (Connection: close). */
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
    if (out_len) *out_len = blen;
    return body_out;
}

static void ignore_sigpipe(void) {
    struct sigaction sa = {0};
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, NULL);
}

/* Send an already-built HTTP 200 with a binary (sealed-envelope) body. */
static void send_sealed_200(int conn_fd, const uint8_t *body, size_t len) {
    char head[256];
    int hn = snprintf(head, sizeof(head),
                      "HTTP/1.1 200 OK\r\n"
                      "Content-Type: application/octet-stream\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n", len);
    if (hn < 0 || (size_t)hn >= sizeof(head)) return;
    if (idletoken_sendall(conn_fd, head, (size_t)hn) < 0) return;
    if (len) idletoken_sendall(conn_fd, body, len);
}

static void handle_conn(int conn_fd, const idletoken_keypair *node,
                        const char *upstream_addr) {
    idletoken_http_req req;
    if (idletoken_http_read_request(conn_fd, &req) != 0) {
        idletoken_http_send_error(conn_fd, 400, "bad request");
        return;
    }

    /* GET /idletoken/v1/privacy/pubkey — hand the consumer our public key. */
    if (!strcmp(req.method, "GET") && !strcmp(req.path, IDLETOKEN_PATH_PRIV_PUBKEY)) {
        char hex[65]; idletoken_phttp_pk_to_hex(node->pk, hex);
        char body[128];
        int bl = snprintf(body, sizeof(body),
                          "{\"pubkey\":\"%s\",\"alg\":\"x25519-sealedbox\"}", hex);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(req.body);
        return;
    }

    /* Map the privacy path to the upstream coord path. */
    const char *upstream_path = NULL;
    if (!strcmp(req.method, "POST") && !strcmp(req.path, IDLETOKEN_PATH_PRIV_MSG))
        upstream_path = IDLETOKEN_PATH_ANTHROPIC;
    else if (!strcmp(req.method, "POST") && !strcmp(req.path, IDLETOKEN_PATH_PRIV_CHAT))
        upstream_path = IDLETOKEN_PATH_OPENAI;

    if (!upstream_path) {
        idletoken_http_send_error(conn_fd, 404, "no such endpoint");
        free(req.body);
        return;
    }
    if (!req.body || req.body_len < IDLETOKEN_PHTTP_REQ_OVERHEAD) {
        idletoken_http_send_error(conn_fd, 400, "missing or too-short sealed request");
        free(req.body);
        return;
    }

    /* --- open the envelope: this is the sole plaintext window --- */
    size_t inner_cap = req.body_len;   /* plaintext <= ciphertext length */
    uint8_t *inner = malloc(inner_cap);
    if (!inner) { idletoken_http_send_error(conn_fd, 500, "oom"); free(req.body); return; }
    idletoken_mlock(inner, inner_cap);    /* keep the plaintext off swap */

    size_t inner_len = 0;
    uint8_t reply_to[IDLETOKEN_PK_BYTES];
    idletoken_priv_rc rc = idletoken_phttp_open_request(node, req.body, req.body_len,
                                                  inner, inner_cap, &inner_len, reply_to);
    if (rc != IDLETOKEN_PRIV_OK) {
        /* Wrong key / tamper / bad frame → reject (auth). */
        idletoken_secure_zero(inner, inner_cap);
        idletoken_munlock(inner, inner_cap); free(inner);
        idletoken_http_send_error(conn_fd, 401, "unauthenticated or corrupt sealed request");
        free(req.body);
        return;
    }

    /* --- forward plaintext to coord over loopback (never leaves this host) --- */
    size_t up_len = 0;
    uint8_t *up_resp = http_post_json(upstream_addr, upstream_path,
                                      inner, inner_len, &up_len);
    /* plaintext request buffer is done — wipe immediately */
    idletoken_secure_zero(inner, inner_cap);
    idletoken_munlock(inner, inner_cap); free(inner);

    if (!up_resp) {
        idletoken_http_send_error(conn_fd, 502, "upstream coord unreachable");
        free(req.body);
        return;
    }

    /* --- seal the reply back to the consumer session key --- */
    size_t sealed_cap = up_len + IDLETOKEN_PHTTP_RESP_OVERHEAD;
    uint8_t *sealed = malloc(sealed_cap);
    if (!sealed) {
        idletoken_secure_zero(up_resp, up_len); free(up_resp);
        idletoken_http_send_error(conn_fd, 500, "oom");
        free(req.body);
        return;
    }
    size_t sealed_len = 0;
    rc = idletoken_phttp_seal_response(reply_to, up_resp, up_len,
                                    sealed, sealed_cap, &sealed_len);
    idletoken_secure_zero(up_resp, up_len); free(up_resp);   /* wipe plaintext reply */

    if (rc != IDLETOKEN_PRIV_OK) {
        free(sealed);
        idletoken_http_send_error(conn_fd, 500, "seal failed");
        free(req.body);
        return;
    }
    send_sealed_200(conn_fd, sealed, sealed_len);
    free(sealed);
    free(req.body);
}

/* Load the node secret key from `path`, or generate+persist it (0600). */
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
            fprintf(stderr, "privacy-proxy: key file %s is not %d bytes; refusing\n",
                    path, IDLETOKEN_SK_BYTES);
            return -1;
        }
    }
    /* generate fresh */
    if (idletoken_keypair_generate(node) != IDLETOKEN_PRIV_OK) return -1;
    if (path) {
        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) {
            fprintf(stderr, "privacy-proxy: cannot create key file %s: %s\n",
                    path, strerror(errno));
            return -1;
        }
        /* write() (not idletoken_sendall, which is send()-based/socket-only) */
        size_t off = 0;
        while (off < IDLETOKEN_SK_BYTES) {
            ssize_t w = write(fd, node->sk + off, IDLETOKEN_SK_BYTES - off);
            if (w > 0) { off += (size_t)w; continue; }
            if (w < 0 && errno == EINTR) continue;
            fprintf(stderr, "privacy-proxy: write key file failed: %s\n", strerror(errno));
            close(fd); return -1;
        }
        close(fd);
        fprintf(stderr, "privacy-proxy: generated new node key, saved to %s (0600)\n", path);
    }
    return 0;
}

static void usage(FILE *o) {
    fprintf(o,
"idletoken-privacy-proxy  sealed-envelope terminator in front of idletoken-coord\n"
"Usage: idletoken-privacy-proxy [--bind H:P] [--upstream H:P] [--key-file PATH]\n"
"\n"
"  --bind H:P       consumer-facing bind (default 0.0.0.0:8443)\n"
"  --upstream H:P   local coord HTTP API (default 127.0.0.1:8000)\n"
"  --key-file PATH  persist/reuse the 32-byte node secret key (0600).\n"
"                   Omit for an ephemeral per-run key.\n"
"  -h, --help       this help\n");
}

int main(int argc, char **argv) {
    const char *bind     = "0.0.0.0:8443";
    const char *upstream = "127.0.0.1:8000";
    const char *key_file = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--bind")     && i + 1 < argc) bind     = argv[++i];
        else if (!strcmp(a, "--upstream") && i + 1 < argc) upstream = argv[++i];
        else if (!strcmp(a, "--key-file") && i + 1 < argc) key_file = argv[++i];
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "privacy-proxy: unknown arg: %s\n\n", a); usage(stderr); return 2; }
    }

    /* Tier-1 hardening: this process is the plaintext window. */
    idletoken_harden_process();

    idletoken_keypair node;
    if (load_or_make_key(key_file, &node) != 0) return 1;
    idletoken_mlock(node.sk, sizeof(node.sk));

    char hex[65]; idletoken_phttp_pk_to_hex(node.pk, hex);
    printf("idletoken-privacy-proxy  (backend: %s)\n", idletoken_privacy_backend());
    printf("  bind (consumer) : %s\n", bind);
    printf("  upstream (coord): %s\n", upstream);
    printf("  node pubkey     : %s\n", hex);
    printf("  key file        : %s\n", key_file ? key_file : "(ephemeral)");
    fflush(stdout);

    ignore_sigpipe();
    int lfd = idletoken_listen_tcp(bind);
    if (lfd < 0) {
        fprintf(stderr, "privacy-proxy: listen(%s): %s\n", bind, strerror(errno));
        return 1;
    }
    fprintf(stderr, "privacy-proxy: listening. Ctrl-C to stop.\n");

    for (;;) {
        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "privacy-proxy: accept: %s\n", strerror(errno));
            break;
        }
        handle_conn(cfd, &node, upstream);
        close(cfd);
    }
    close(lfd);
    idletoken_secure_zero(node.sk, sizeof(node.sk));
    idletoken_munlock(node.sk, sizeof(node.sk));
    return 0;
}
