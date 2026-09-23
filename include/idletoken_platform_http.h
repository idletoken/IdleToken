/* Public-network HTTP transport. Local inference keeps its own queue contract. */
#ifndef IDLETOKEN_PLATFORM_HTTP_H
#define IDLETOKEN_PLATFORM_HTTP_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *url;
    const char *method;
    const char *bearer;
    const char *headers;                 /* optional CRLF-separated headers */
    const char *unix_socket;             /* local IPC; always bypass proxies */
    const void *body;
    size_t body_len;
    int connect_ms;                      /* DNS, TCP, proxy and TLS together */
    int headers_ms;                      /* from the end of the request upload, plus body_idle_ms for
                                            bodies of 64 KiB or more still draining; 0 = caller cancellation */
    int total_ms;                        /* from request start; 0 = no fixed total */
    int body_idle_ms;                    /* no progress, either direction */
    int body_total_ms;                   /* whole upload, and separately the whole response body */
    size_t max_response;
    int receipt;                         /* read the relay deadline prefix */
    int (*cancelled)(void *);
    void *cancel_context;
} idletoken_platform_http_request;

typedef struct {
    int status;
    unsigned char *body;                 /* always NUL-terminated; caller frees */
    size_t len;
    int truncated;                       /* 1 = interrupted, 2 = deadline */
    int cancelled;
    int retryable;                      /* transient transport failure, not an HTTP status */
    int64_t delivery_deadline_ms;        /* local monotonic clock, never wall time */
    char error[256];                     /* no URL credentials or response body */
} idletoken_platform_http_response;

int64_t idletoken_platform_now_ms(void);
/* Isolated system proxy helper. Return -1 for ordinary argv. */
int idletoken_platform_proxy_helper(int argc, char **argv);
/* Accept HTTP(S), bracketed IPv6, paths and explicit ports without rewriting.
 * Credentials, query/fragment base URLs and all other schemes are rejected. */
int idletoken_platform_url(const char *base, const char *path, char *out, size_t cap);
int idletoken_platform_http(const idletoken_platform_http_request *request,
                           idletoken_platform_http_response *response);

#endif
