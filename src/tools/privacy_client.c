/* IdleToken privacy client — reference consumer for the sealed HTTP API.
 *
 * Demonstrates the consumer side end-to-end: fetch the node pubkey, generate a
 * per-session keypair, seal a request, POST it to the privacy proxy, and open
 * the sealed reply. Doubles as the end-to-end test driver for idletoken-privacy-
 * proxy. Prints the decrypted response body to stdout; on success the last
 * stderr line is `PRIVACY_CLIENT_OK`.
 *
 *   idletoken-privacy-client --proxy H:P [--endpoint messages|chat] --inner JSON
 *
 * C only.
 */

#include "idletoken_privacy.h"
#include "idletoken_privacy_http.h"
#include "idletoken_net.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Minimal HTTP client: send `method path` with optional body, return malloc'd
 * response BODY (caller frees), set *out_len, *out_status. NULL on error. */
static uint8_t *http_do(const char *addr, const char *method, const char *path,
                        const uint8_t *body, size_t body_len,
                        size_t *out_len, int *out_status) {
    int fd = idletoken_connect_tcp(addr);
    if (fd < 0) { fprintf(stderr, "client: connect %s: %s\n", addr, strerror(errno)); return NULL; }

    char head[512];
    int hn = snprintf(head, sizeof(head),
                      "%s %s HTTP/1.1\r\nHost: %s\r\n"
                      "Content-Type: application/octet-stream\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      method, path, addr, body_len);
    if (hn < 0 || (size_t)hn >= sizeof(head)) { close(fd); return NULL; }
    if (idletoken_sendall(fd, head, (size_t)hn) < 0 ||
        (body_len && idletoken_sendall(fd, body, body_len) < 0)) { close(fd); return NULL; }

    size_t cap = 8192, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) { close(fd); return NULL; }
    for (;;) {
        if (len + 4096 > cap) { uint8_t *nb = realloc(buf, cap * 2); if (!nb) { free(buf); close(fd); return NULL; } buf = nb; cap *= 2; }
        ssize_t r = read(fd, buf + len, 4096);
        if (r > 0) { len += (size_t)r; continue; }
        if (r < 0 && errno == EINTR) continue;
        break;
    }
    close(fd);

    int status = 0;
    if (len >= 12 && memcmp(buf, "HTTP/1.", 7) == 0) status = atoi((char *)buf + 9);
    if (out_status) *out_status = status;

    uint8_t *sep = NULL;
    for (size_t i = 0; i + 3 < len; i++)
        if (buf[i]=='\r'&&buf[i+1]=='\n'&&buf[i+2]=='\r'&&buf[i+3]=='\n') { sep = buf + i + 4; break; }
    if (!sep) { free(buf); return NULL; }
    size_t blen = len - (size_t)(sep - buf);
    uint8_t *out = malloc(blen ? blen : 1);
    if (!out) { free(buf); return NULL; }
    memcpy(out, sep, blen);
    free(buf);
    if (out_len) *out_len = blen;
    return out;
}

int main(int argc, char **argv) {
    const char *proxy = "127.0.0.1:8443";
    const char *endpoint = "messages";
    const char *inner = "{\"content\":\"Reply with the single word: pong\",\"max_tokens\":16}";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--proxy")    && i + 1 < argc) proxy = argv[++i];
        else if (!strcmp(a, "--endpoint") && i + 1 < argc) endpoint = argv[++i];
        else if (!strcmp(a, "--inner")    && i + 1 < argc) inner = argv[++i];
        else { fprintf(stderr, "client: unknown arg %s\n", a); return 2; }
    }
    const char *path = !strcmp(endpoint, "chat")
                     ? "/v1/privacy/chat/completions" : "/v1/privacy/messages";

    /* 1. fetch node pubkey */
    size_t plen = 0; int st = 0;
    uint8_t *pk_json = http_do(proxy, "GET", "/v1/privacy/pubkey", NULL, 0, &plen, &st);
    if (!pk_json || st != 200) { fprintf(stderr, "client: pubkey fetch failed (status %d)\n", st); return 1; }
    char hex[65] = {0};
    /* find "pubkey":"..." */
    const char *p = strstr((char *)pk_json, "\"pubkey\"");
    if (!p || !(p = strchr(p, ':')) || !(p = strchr(p, '"'))) { fprintf(stderr, "client: bad pubkey json\n"); free(pk_json); return 1; }
    p++;
    for (int i = 0; i < 64 && p[i] && p[i] != '"'; i++) hex[i] = p[i];
    free(pk_json);
    uint8_t node_pk[IDLETOKEN_PK_BYTES];
    if (idletoken_phttp_pk_from_hex(hex, node_pk) != IDLETOKEN_PRIV_OK) { fprintf(stderr, "client: bad pubkey hex\n"); return 1; }

    /* 2. session keypair + seal request */
    idletoken_keypair session; idletoken_keypair_generate(&session);
    size_t env_cap = strlen(inner) + IDLETOKEN_PHTTP_REQ_OVERHEAD;
    uint8_t *env = malloc(env_cap); size_t env_len = 0;
    if (idletoken_phttp_seal_request(node_pk, &session, (const uint8_t *)inner, strlen(inner),
                                  env, env_cap, &env_len) != IDLETOKEN_PRIV_OK) {
        fprintf(stderr, "client: seal failed\n"); free(env); return 1;
    }

    /* 3. POST sealed request */
    size_t rlen = 0; st = 0;
    uint8_t *resp_env = http_do(proxy, "POST", path, env, env_len, &rlen, &st);
    free(env);
    if (!resp_env || st != 200) { fprintf(stderr, "client: POST failed (status %d)\n", st); free(resp_env); return 1; }

    /* 4. open sealed reply */
    uint8_t plain[16384]; size_t plain_len = 0;
    idletoken_priv_rc rc = idletoken_phttp_open_response(&session, resp_env, rlen,
                                                   plain, sizeof(plain), &plain_len);
    free(resp_env);
    if (rc != IDLETOKEN_PRIV_OK) { fprintf(stderr, "client: open response failed (rc=%d)\n", rc); return 1; }

    fwrite(plain, 1, plain_len, stdout);
    fputc('\n', stdout);
    fprintf(stderr, "PRIVACY_CLIENT_OK\n");
    return 0;
}
