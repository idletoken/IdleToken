/* IdleToken Cluster — minimal HTTP/1.1 helpers for the coord API server.
 *
 * v0.1 scope: synchronous, one-connection-at-a-time, request/response only.
 * Supports just enough for OpenAI /v1/chat/completions and Anthropic
 * /v1/messages with JSON bodies. No keep-alive, no chunked, no TLS.
 *
 * SSE ("stream":true) is supported via connection-close-delimited streaming:
 * idletoken_http_send_sse_head() sends a 200 + `Content-Type: text/event-stream`
 * head WITHOUT Content-Length, then idletoken_http_sse_event() writes one
 * `event:`/`data:` frame per call; closing the socket ends the stream
 * (legal HTTP/1.1: no Content-Length + `Connection: close`). No chunked
 * transfer-encoding needed.
 *
 * C99, no C++. Pair with src/common/http.c. */

#ifndef IDLETOKEN_HTTP_H
#define IDLETOKEN_HTTP_H

#include <stddef.h>
#include <stdint.h>

#define IDLETOKEN_HTTP_METHOD_MAX 8
#define IDLETOKEN_HTTP_PATH_MAX   256
#define IDLETOKEN_HTTP_BODY_CAP   (4u * 1024u * 1024u) /* 4 MiB cap on request body */
#define IDLETOKEN_HTTP_HEADERS_CAP 4096              /* retained raw header block */

typedef struct {
    char    method[IDLETOKEN_HTTP_METHOD_MAX];   /* "GET", "POST", ... */
    char    path[IDLETOKEN_HTTP_PATH_MAX];       /* "/v1/messages" */
    uint8_t *body;                            /* malloc'd; caller frees */
    size_t  body_len;
    int     keepalive_unused;                 /* reserved */
    /* Raw header block ("Name: value\r\n..."), NUL-terminated, truncated at
     * IDLETOKEN_HTTP_HEADERS_CAP-1. Fixed-size on purpose: existing callers only
     * free `body`, and typical headers are well under 2 KB. Query it with
     * idletoken_http_header_get(). */
    char    headers[IDLETOKEN_HTTP_HEADERS_CAP];
} idletoken_http_req;

/* Read one HTTP request from `conn_fd`. Parses request line + headers +
 * fixed-length body (Content-Length). Returns 0 on success, -1 on parse
 * or I/O error (errno may be EPROTO / EMSGSIZE / ECONNRESET / etc.).
 *
 * On success, populates `*out` and `out->body` is malloc'd (may be NULL if
 * Content-Length is 0 or absent). Caller MUST free `out->body`.
 *
 * Caps body at IDLETOKEN_HTTP_BODY_CAP; oversize → -1, EMSGSIZE. */
int idletoken_http_read_request(int conn_fd, idletoken_http_req *out);

/* Send a status line + Content-Type + Content-Length header + body in one
 * shot. status is the numeric HTTP status (200/400/404/500/501).
 * content_type may be e.g. "application/json" or "text/plain". body/body_len
 * may be NULL/0 for empty responses. Returns 0/-1. */
int idletoken_http_send_response(int conn_fd, int status,
                              const char *content_type,
                              const void *body, size_t body_len);

/* Convenience: send a JSON response (sets Content-Type automatically). */
int idletoken_http_send_json(int conn_fd, int status,
                          const char *json, size_t json_len);

/* Convenience: send a plain-text error message. */
int idletoken_http_send_error(int conn_fd, int status, const char *message);

/* --- SSE (Server-Sent Events) helpers ------------------------------------
 *
 * Streaming responses for `"stream":true` chat requests. The stream is
 * delimited by connection close (no Content-Length, `Connection: close`),
 * which every SSE client accepts — the OpenAI/Anthropic SDKs and Claude Code
 * terminate on `[DONE]` / `message_stop` events anyway. */

/* Send the SSE response head: `200 OK` + `Content-Type: text/event-stream` +
 * `Cache-Control: no-cache` + `Connection: close` + blank line. After this,
 * only idletoken_http_sse_event() writes may follow on this connection.
 * Returns 0/-1. */
int idletoken_http_send_sse_head(int conn_fd);

/* Write one SSE frame:  [`event: <event>\n`] `data: <data>\n\n`.
 * `event` may be NULL/"" for a data-only frame (OpenAI style); Anthropic
 * frames pass the event name. `data` must be a single line (JSON — our
 * emitters never embed raw newlines). Returns 0/-1 (client hangup surfaces
 * as -1/EPIPE; SIGPIPE is ignored by the coord). */
int idletoken_http_sse_event(int conn_fd, const char *event,
                          const char *data, size_t data_len);

/* Where a request came from, so the coordinator can tell a job the platform
 * dispatched from one a client on the LAN sent directly. Both arrive on the same
 * endpoint: the platform agent opens the sealed envelope and forwards the
 * plaintext to the coordinator over loopback, which until now was
 * indistinguishable from a local request.
 *
 * This matters for overflow routing (docs/overflow-routing-design.md): a job the
 * platform dispatched must be finished locally or refused, and must never be
 * forwarded back out. Defining the name in one place keeps the agent that sets
 * it and the coordinator that reads it from drifting apart. */
#define IDLETOKEN_HDR_ORIGIN       "X-IdleToken-Origin"
#define IDLETOKEN_ORIGIN_PLATFORM  "platform"

/* Extract one header value by name (case-insensitive) from `req->headers`.
 * Copies the value (leading/trailing whitespace trimmed) into `out`.
 * Returns 0 on success, -1 if the header is absent (or was truncated away).
 * Used by the coord API server for `Authorization` / `x-api-key`. */
int idletoken_http_header_get(const idletoken_http_req *req, const char *name,
                           char *out, size_t out_cap);

/* --- tiny JSON helpers (just enough for v0.1 stubs) ----------------------
 *
 * We write JSON manually rather than pull a parser/emitter. v0.1 endpoints
 * have fixed-shape responses; switch to a real JSON library when scope
 * grows. */

/* Extract a UTF-8 string field from a flat JSON object. Returns 0/-1.
 * Naive — doesn't handle nested objects, escape sequences beyond \" and \\,
 * or non-string values. Good enough for `"model":"deepseek-v4-flash"` and
 * `"messages":[{"role":"user","content":"hi"}]` (we just grab the first
 * "content" string from a flat search). For real parsing, use a library. */
int idletoken_http_json_extract_str(const char *json, size_t json_len,
                                 const char *key,
                                 char *out, size_t out_cap);

#endif /* IDLETOKEN_HTTP_H */
