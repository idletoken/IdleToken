/* IdleToken Cluster — minimal HTTP/1.1 helpers. See include/idletoken_http.h. */

#include "idletoken_http.h"
#include "idletoken_net.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
  #include <winsock2.h>
#else
  #include <unistd.h>
#endif

/* Winsock sockets are NOT CRT file descriptors, so read() on one fails with
 * EBADF. Sends already go through idletoken_sendall (net.c, which knows this);
 * the two reads below did not, which made the coordinator answer 400 to every
 * request when it ran on Windows — the whole HTTP API was dead there and no
 * test had ever driven a Windows coordinator. */
static ssize_t sock_read(int fd, void *buf, size_t n) {
#ifdef _WIN32
    int r = recv((SOCKET)fd, (char *)buf, (int)n, 0);
    if (r < 0) errno = ECONNRESET;   /* WSAGetLastError() is not in errno */
    return r;
#else
    return read(fd, buf, n);
#endif
}

void idletoken_http_path_strip_query(char *path) {
    /* Strip the query string so route matching sees the bare path. Real
     * clients do send one — Claude Code POSTs /v1/messages?beta=true, which
     * exact-match routing turned into a 404. No current route consumes query
     * parameters, so they are dropped rather than stored. */
    if (!path) return;
    char *qmark = strchr(path, '?');
    if (qmark) *qmark = 0;
}

int idletoken_http_auth_value_matches(const char *hval, const char *token) {
    static const char scheme[] = "bearer ";
    const char *v = hval;
    size_t i = 0;
    while (scheme[i]) {
        char c = v[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != scheme[i]) break;
        i++;
    }
    if (!scheme[i]) v += sizeof(scheme) - 1;
    while (*v == ' ') v++;
    return strcmp(v, token) == 0;
}

/* Read bytes from fd into buf until we see CRLF CRLF (end of headers) or
 * the buffer fills. Returns the number of bytes read on success (always
 * includes the terminating CRLF CRLF), -1 with errno on error. Leaves the
 * caller to parse the head; any body bytes that arrived in the same recv
 * are written into `*out_head_len` for the caller to know how much of buf
 * is head vs leftover body. */
static ssize_t read_until_headers_end(int fd, uint8_t *buf, size_t cap,
                                      size_t *out_head_len) {
    size_t pos = 0;
    while (pos < cap) {
        ssize_t n = sock_read(fd, buf + pos, cap - pos);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = ECONNRESET;
            return -1;
        }
        pos += (size_t)n;
        /* Headers are tiny (typ. <2 KB); rescan from the start each read. */
        if (pos >= 4) {
            for (size_t i = 0; i + 4 <= pos; i++) {
                if (buf[i] == '\r' && buf[i+1] == '\n' &&
                    buf[i+2] == '\r' && buf[i+3] == '\n') {
                    *out_head_len = i + 4;
                    return (ssize_t)pos;
                }
            }
        }
    }
    errno = EMSGSIZE;
    return -1;
}

/* Parse a Content-Length header value. Returns the length, or -1 if absent
 * or invalid. Case-insensitive header name match. */
static long find_content_length(const char *head, size_t head_len) {
    static const char want[] = "content-length:";
    const size_t wlen = sizeof(want) - 1;
    for (size_t i = 0; i + wlen <= head_len; i++) {
        size_t j;
        for (j = 0; j < wlen; j++) {
            char c = head[i + j];
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            if (c != want[j]) break;
        }
        if (j == wlen) {
            /* Skip whitespace, parse digits. */
            i += wlen;
            while (i < head_len && (head[i] == ' ' || head[i] == '\t')) i++;
            long v = 0;
            int saw = 0;
            while (i < head_len && head[i] >= '0' && head[i] <= '9') {
                v = v * 10 + (head[i] - '0');
                if (v > (long)IDLETOKEN_HTTP_BODY_CAP) return (long)IDLETOKEN_HTTP_BODY_CAP + 1;
                i++; saw = 1;
            }
            return saw ? v : -1;
        }
    }
    return -1;
}

int idletoken_http_read_request(int conn_fd, idletoken_http_req *out) {
    if (!out) { errno = EINVAL; return -1; }
    memset(out, 0, sizeof(*out));

    /* 16 KB is plenty for request line + headers. */
    const size_t head_cap = 16 * 1024;
    uint8_t *buf = malloc(head_cap);
    if (!buf) { errno = ENOMEM; return -1; }

    size_t head_len = 0;
    ssize_t got = read_until_headers_end(conn_fd, buf, head_cap, &head_len);
    if (got < 0) { free(buf); return -1; }

    /* Parse request line: METHOD SP PATH SP HTTP/1.x CRLF */
    size_t m_end = 0;
    while (m_end < head_len && buf[m_end] != ' ') m_end++;
    if (m_end >= head_len || m_end >= sizeof(out->method)) {
        free(buf); errno = EPROTO; return -1;
    }
    memcpy(out->method, buf, m_end);
    out->method[m_end] = 0;

    size_t p_start = m_end + 1;
    size_t p_end = p_start;
    while (p_end < head_len && buf[p_end] != ' ') p_end++;
    size_t plen = p_end - p_start;
    if (p_end >= head_len || plen == 0 || plen >= sizeof(out->path)) {
        free(buf); errno = EPROTO; return -1;
    }
    memcpy(out->path, buf + p_start, plen);
    out->path[plen] = 0;

    idletoken_http_path_strip_query(out->path);

    /* Find end of request line to start scanning headers. */
    size_t line_end = p_end;
    while (line_end + 1 < head_len && !(buf[line_end] == '\r' && buf[line_end+1] == '\n')) line_end++;
    if (line_end + 1 >= head_len) { free(buf); errno = EPROTO; return -1; }
    size_t hdr_start = line_end + 2;
    size_t hdr_end   = head_len - 2;  /* exclude the final CRLF before body */
    if (hdr_end < hdr_start) hdr_end = hdr_start;

    /* Retain the raw header block so callers can query individual headers
     * (e.g. Authorization for the coord's --api-token check). Truncation past
     * the cap only loses headers nobody sends in practice. */
    {
        size_t hlen = hdr_end - hdr_start;
        if (hlen >= sizeof(out->headers)) hlen = sizeof(out->headers) - 1;
        memcpy(out->headers, buf + hdr_start, hlen);
        out->headers[hlen] = 0;
    }

    long cl = find_content_length((const char *)(buf + hdr_start),
                                  hdr_end - hdr_start);
    if (cl > (long)IDLETOKEN_HTTP_BODY_CAP) {
        free(buf); errno = EMSGSIZE; return -1;
    }

    /* Body: some of it may already be in buf after head_len. */
    size_t body_avail = (size_t)got - head_len;
    size_t body_total = (cl < 0) ? body_avail : (size_t)cl;

    if (body_total > 0) {
        out->body = malloc(body_total);
        if (!out->body) { free(buf); errno = ENOMEM; return -1; }
        size_t copy = body_avail < body_total ? body_avail : body_total;
        if (copy > 0) memcpy(out->body, buf + head_len, copy);
        size_t pos = copy;
        while (pos < body_total) {
            ssize_t r = sock_read(conn_fd, out->body + pos, body_total - pos);
            if (r < 0) {
                if (errno == EINTR) continue;
                free(buf); free(out->body); out->body = NULL;
                return -1;
            }
            if (r == 0) {
                free(buf); free(out->body); out->body = NULL;
                errno = ECONNRESET; return -1;
            }
            pos += (size_t)r;
        }
        out->body_len = body_total;
    }
    /* If Content-Length absent, body_total = body_avail (whatever arrived
     * with the header burst). Every real client of these routes sends
     * Content-Length; this branch only serves hand-typed probes. */

    free(buf);
    return 0;
}

static const char *http_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default:  return "OK";
    }
}

int idletoken_http_send_response(int conn_fd, int status,
                              const char *content_type,
                              const void *body, size_t body_len) {
    char head[512];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     status, http_reason(status),
                     content_type ? content_type : "application/octet-stream",
                     body_len);
    if (n < 0 || (size_t)n >= sizeof(head)) { errno = EOVERFLOW; return -1; }
    if (idletoken_sendall(conn_fd, head, (size_t)n) < 0) return -1;
    if (body_len > 0 && idletoken_sendall(conn_fd, body, body_len) < 0) return -1;
    return 0;
}

int idletoken_http_send_json(int conn_fd, int status,
                          const char *json, size_t json_len) {
    return idletoken_http_send_response(conn_fd, status,
                                     "application/json",
                                     json, json_len);
}

int idletoken_http_send_error(int conn_fd, int status, const char *message) {
    if (!message) message = http_reason(status);
    return idletoken_http_send_response(conn_fd, status,
                                     "text/plain; charset=utf-8",
                                     message, strlen(message));
}

int idletoken_http_send_sse_head(int conn_fd) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";
    return idletoken_sendall(conn_fd, head, sizeof(head) - 1) < 0 ? -1 : 0;
}

int idletoken_http_sse_event(int conn_fd, const char *event,
                          const char *data, size_t data_len) {
    /* "event: <name>\ndata: " prefix (or just "data: "), then the payload,
     * then the blank-line frame terminator. Three writes keep this
     * allocation-free; the extra syscalls are noise next to a decode step. */
    char pre[128];
    int n;
    if (event && event[0])
        n = snprintf(pre, sizeof(pre), "event: %s\ndata: ", event);
    else
        n = snprintf(pre, sizeof(pre), "data: ");
    if (n < 0 || (size_t)n >= sizeof(pre)) { errno = EOVERFLOW; return -1; }
    if (idletoken_sendall(conn_fd, pre, (size_t)n) < 0) return -1;
    if (data_len > 0 && idletoken_sendall(conn_fd, data, data_len) < 0) return -1;
    return idletoken_sendall(conn_fd, "\n\n", 2) < 0 ? -1 : 0;
}

int idletoken_http_header_get(const idletoken_http_req *req, const char *name,
                           char *out, size_t out_cap) {
    if (!req || !name || !out || out_cap == 0) { errno = EINVAL; return -1; }
    out[0] = 0;
    size_t nlen = strlen(name);
    const char *p = req->headers;
    while (*p) {
        /* Case-insensitive match of `name` followed by ':' at line start. */
        size_t j = 0;
        while (j < nlen) {
            char a = p[j], b = name[j];
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
            if (a != b) break;
            j++;
        }
        if (j == nlen && p[nlen] == ':') {
            const char *v = p + nlen + 1;
            while (*v == ' ' || *v == '\t') v++;
            size_t op = 0;
            while (v[op] && v[op] != '\r' && v[op] != '\n' && op + 1 < out_cap) {
                out[op] = v[op];
                op++;
            }
            /* Trim trailing whitespace. */
            while (op > 0 && (out[op-1] == ' ' || out[op-1] == '\t')) op--;
            out[op] = 0;
            return 0;
        }
        /* Advance to the next line. */
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
    }
    return -1;
}

/* Naive flat-JSON string extractor. Finds `"key"` then `:` then a `"..."`
 * value with simple \\ and \" escapes. Returns 0 on success, -1 if not
 * found or malformed. */
/* Parse exactly four hex digits into *out. Returns 0 on success. */
static int json_hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        unsigned d;
        if (c >= '0' && c <= '9')      d = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A' + 10);
        else return -1;
        v = (v << 4) | d;
    }
    *out = v;
    return 0;
}

/* Encode one code point as UTF-8; returns the number of bytes written (<= 4).
 * Caller guarantees room. Unpaired surrogates become U+FFFD rather than
 * invalid UTF-8 -- the tokenizer downstream must never see a broken sequence. */
static size_t json_utf8_put(char *out, unsigned cp) {
    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));  out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

int idletoken_http_json_extract_str(const char *json, size_t json_len,
                                 const char *key,
                                 char *out, size_t out_cap) {
    if (!json || !key || !out || out_cap == 0) { errno = EINVAL; return -1; }
    size_t klen = strlen(key);
    if (klen == 0 || klen + 2 > json_len) return -1;

    /* Find "key" sequence. */
    for (size_t i = 0; i + klen + 2 <= json_len; i++) {
        if (json[i] != '"') continue;
        if (i + 1 + klen + 1 > json_len) break;
        if (memcmp(json + i + 1, key, klen) != 0) continue;
        if (json[i + 1 + klen] != '"') continue;
        /* Now skip whitespace then `:` then whitespace then `"`. */
        size_t p = i + 1 + klen + 1;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json_len || json[p] != ':') continue;
        p++;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json_len || json[p] != '"') continue;
        p++;
        /* Copy until the closing `"`, decoding JSON string escapes.
         *
         * The table used to stop at \" \\ \n \t and copy anything else with the
         * backslash dropped, which quietly corrupted two things that arrive
         * every day:
         *   \uXXXX -- Python's json.dumps escapes ALL non-ASCII by default
         *             (ensure_ascii=True), so an OpenAI-compatible client
         *             sending "你好" puts \u4f60\u597d on the wire and the
         *             engine fed the model the literal text "u4f60u597d".
         *             Every CJK prompt from such a client was garbage.
         *   \r     -- survives a round trip through any Windows text a user
         *             pastes, and became a bare "r" mid-sentence.
         * Both are silent: the request succeeds and the reply is merely wrong,
         * which is the hardest kind of failure to attribute. */
        size_t op = 0;
        while (p < json_len && op + 1 < out_cap) {
            char c = json[p];
            if (c == '\\' && p + 1 < json_len) {
                char e = json[p + 1];
                if (e == '"' || e == '\\' || e == '/') { out[op++] = e; p += 2; continue; }
                if (e == 'n') { out[op++] = '\n'; p += 2; continue; }
                if (e == 't') { out[op++] = '\t'; p += 2; continue; }
                if (e == 'r') { out[op++] = '\r'; p += 2; continue; }
                if (e == 'b') { out[op++] = '\b'; p += 2; continue; }
                if (e == 'f') { out[op++] = '\f'; p += 2; continue; }
                if (e == 'u' && p + 5 < json_len) {
                    unsigned cp = 0;
                    if (json_hex4(json + p + 2, &cp) == 0) {
                        p += 6;
                        /* Surrogate pair: the escape encodes UTF-16, and a high
                         * surrogate alone is not a character. */
                        if (cp >= 0xD800 && cp <= 0xDBFF && p + 5 < json_len &&
                            json[p] == '\\' && json[p + 1] == 'u') {
                            unsigned lo = 0;
                            if (json_hex4(json + p + 2, &lo) == 0 &&
                                lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                p += 6;
                            }
                        }
                        if (op + 4 >= out_cap) break;   /* no room: stop cleanly */
                        op += json_utf8_put(out + op, cp);
                        continue;
                    }
                }
                /* Unrecognized escape: copy as-is. */
                out[op++] = e; p += 2; continue;
            }
            if (c == '"') {
                out[op] = 0;
                return 0;
            }
            out[op++] = c;
            p++;
        }
        out[op] = 0;
        return -1;  /* unterminated string */
    }
    return -1;
}
