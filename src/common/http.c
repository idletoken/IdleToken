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
  #include <windows.h>
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <time.h>
#endif

/* Winsock sockets are NOT CRT file descriptors, so read() on one fails with
 * EBADF. Sends already go through idletoken_sendall (net.c, which knows this);
 * the two reads below did not, which made the coordinator answer 400 to every
 * request when it ran on Windows — the whole HTTP API was dead there and no
 * test had ever driven a Windows coordinator. */
static ssize_t sock_read(int fd, void *buf, size_t n) {
#ifdef _WIN32
    int r = recv((SOCKET)fd, (char *)buf, (int)n, 0);
    if (r < 0) errno = WSAGetLastError() == WSAETIMEDOUT ? ETIMEDOUT : ECONNRESET;
    return r;
#else
    ssize_t r = read(fd, buf, n);
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) errno = ETIMEDOUT;
    return r;
#endif
}

static uint64_t http_monotonic_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
#endif
}

void idletoken_http_finish_rejection(int fd, int wait_ms) {
#ifdef _WIN32
    shutdown((SOCKET)fd, SD_SEND);
    if (wait_ms <= 0) {
        u_long nonblocking = 1;
        if (ioctlsocket((SOCKET)fd, FIONBIO, &nonblocking)) return;
    }
#else
    shutdown(fd, SHUT_WR);
#endif
    uint64_t end = http_monotonic_ms() + (wait_ms > 0 ? (unsigned)wait_ms : 0);
    size_t total = 0;
    uint8_t discarded[8192];
    while (total < (1u << 20)) {
        if (wait_ms > 0) {
            uint64_t now = http_monotonic_ms();
            if (now >= end) break;
            if (idletoken_set_recv_timeout(fd, (int)(end - now))) break;
        }
#ifdef _WIN32
        ssize_t n = sock_read(fd, discarded, sizeof discarded);
#else
        ssize_t n = wait_ms > 0 ? sock_read(fd, discarded, sizeof discarded) :
                    recv(fd, discarded, sizeof discarded, MSG_DONTWAIT);
#endif
        if (n <= 0) break;
        total += (size_t)n;
    }
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

static int http_equal(const char *text, size_t len, const char *expected) {
    if (len != strlen(expected)) return 0;
    for (size_t i = 0; i < len; i++)
        if (tolower((unsigned char)text[i]) != (unsigned char)expected[i]) return 0;
    return 1;
}

static int http_field_name(const char *text, size_t len) {
    if (!len) return 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || strchr("!#$%&'*+-.^_`|~", c))) return 0;
        if (!c) return 0;
    }
    return 1;
}

/* RFC 9112 framing is a header-line property, never a substring search.
 * Ambiguous lengths/encodings fail closed instead of parsing a partial prompt. */
static int request_framing(const char *head, size_t len, long *length,
                           int *chunked, int *expect_continue) {
    *length = -1; *chunked = 0; *expect_continue = 0;
    size_t pos = 0;
    while (pos < len) {
        size_t end = pos;
        while (end + 1 < len && !(head[end] == '\r' && head[end+1] == '\n')) end++;
        if (end + 1 >= len) goto invalid;
        size_t colon = pos;
        while (colon < end && head[colon] != ':') colon++;
        if (colon == end || !http_field_name(head + pos, colon - pos)) goto invalid;
        size_t start = colon + 1, finish = end;
        while (start < finish && (head[start] == ' ' || head[start] == '\t')) start++;
        while (finish > start && (head[finish-1] == ' ' || head[finish-1] == '\t')) finish--;
        for (size_t i = start; i < finish; i++)
            if (((unsigned char)head[i] < 32 && head[i] != '\t') || head[i] == 127) goto invalid;
        if (http_equal(head + pos, colon - pos, "content-length")) {
            if (*length >= 0 || start == finish) goto invalid;
            long value = 0;
            for (size_t i = start; i < finish; i++) {
                if (head[i] < '0' || head[i] > '9') goto invalid;
                value = value * 10 + (head[i] - '0');
                if (value > (long)IDLETOKEN_HTTP_BODY_CAP) { errno = EMSGSIZE; return -1; }
            }
            *length = value;
        } else if (http_equal(head + pos, colon - pos, "transfer-encoding")) {
            if (*chunked || !http_equal(head + start, finish - start, "chunked")) goto invalid;
            *chunked = 1;
        } else if (http_equal(head + pos, colon - pos, "expect")) {
            if (!http_equal(head + start, finish - start, "100-continue")) goto invalid;
            *expect_continue = 1;
        }
        pos = end + 2;
    }
    if (*chunked && *length >= 0) goto invalid;
    return 0;
invalid:
    errno = EPROTO;
    return -1;
}

int idletoken_http_bodyless_get_path(const uint8_t *head, size_t length,
                                    char *path, size_t path_capacity) {
    if (!head || length < 4 || memcmp(head, "GET ", 4) || !path || !path_capacity)
        return 0;
    size_t end = 4;
    while (end < length && head[end] != ' ') end++;
    if (end == length || end == 4 || end - 4 >= path_capacity) return 0;
    size_t line = end;
    while (line + 1 < length && !(head[line] == '\r' && head[line + 1] == '\n')) line++;
    if (line + 3 >= length || head[length - 2] != '\r' || head[length - 1] != '\n')
        return 0;
    long content_length;
    int chunked, expect_continue;
    if (request_framing((const char *)head + line + 2, length - line - 4,
                        &content_length, &chunked, &expect_continue) || chunked || content_length > 0)
        return 0;
    memcpy(path, head + 4, end - 4);
    path[end - 4] = 0;
    idletoken_http_path_strip_query(path);
    return 1;
}

typedef struct {
    int fd;
    uint8_t *buffer;
    size_t pos, end, capacity;
} http_reader;

static int http_read_exact(http_reader *reader, void *target, size_t count) {
    uint8_t *out = target;
    while (count) {
        if (reader->pos == reader->end) {
            ssize_t n = sock_read(reader->fd, reader->buffer, reader->capacity);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { if (!n) errno = ECONNRESET; return -1; }
            reader->pos = 0; reader->end = (size_t)n;
        }
        size_t take = reader->end - reader->pos;
        if (take > count) take = count;
        memcpy(out, reader->buffer + reader->pos, take);
        reader->pos += take; out += take; count -= take;
    }
    return 0;
}

static int http_read_line(http_reader *reader, char *line, size_t capacity, size_t *length) {
    size_t pos = 0;
    while (pos + 1 < capacity) {
        char c;
        if (http_read_exact(reader, &c, 1)) return -1;
        if (c == '\r') {
            if (http_read_exact(reader, &c, 1)) return -1;
            if (c != '\n') { errno = EPROTO; return -1; }
            line[pos] = 0; *length = pos; return 0;
        }
        if (((unsigned char)c < 32 && c != '\t') || c == 127) { errno = EPROTO; return -1; }
        line[pos++] = c;
    }
    errno = EMSGSIZE;
    return -1;
}

static int http_read_chunked(http_reader *reader, idletoken_http_req *request) {
    size_t allocated = 0, framing = 0;
    for (;;) {
        char line[4096]; size_t length;
        if (http_read_line(reader, line, sizeof line, &length)) return -1;
        framing += length + 4;
        if (framing > IDLETOKEN_HTTP_BODY_CAP) { errno = EMSGSIZE; return -1; }
        size_t size = 0, digits = 0;
        while (digits < length && isxdigit((unsigned char)line[digits])) {
            unsigned char c = (unsigned char)tolower((unsigned char)line[digits]);
            unsigned value = c <= '9' ? c - '0' : c - 'a' + 10;
            if (size > (IDLETOKEN_HTTP_BODY_CAP - value) / 16) { errno = EMSGSIZE; return -1; }
            size = size * 16 + value; digits++;
        }
        if (!digits) { errno = EPROTO; return -1; }
        while (digits < length && (line[digits] == ' ' || line[digits] == '\t')) digits++;
        if (digits < length && line[digits] != ';') { errno = EPROTO; return -1; }
        if (!size) {
            size_t trailers = 0;
            do {
                if (http_read_line(reader, line, sizeof line, &length)) return -1;
                trailers += length + 2;
                if (trailers > IDLETOKEN_HTTP_HEADERS_CAP) { errno = EMSGSIZE; return -1; }
                if (length) {
                    char *colon = strchr(line, ':');
                    if (!colon || !http_field_name(line, (size_t)(colon-line))) { errno = EPROTO; return -1; }
                    /* Trailers cannot smuggle policy/framing headers past the
                     * coordinator's admission and Origin checks. */
                    const char *forbidden[] = { "origin", "authorization", "proxy-authorization", "host",
                        "cookie", "content-length", "transfer-encoding", "expect" };
                    for (size_t i = 0; i < sizeof forbidden / sizeof forbidden[0]; i++)
                        if (http_equal(line, (size_t)(colon-line), forbidden[i])) { errno = EPROTO; return -1; }
                    if ((size_t)(colon-line) >= 12 && http_equal(line, 12, "x-idletoken-")) { errno = EPROTO; return -1; }
                }
            } while (length);
            return 0;
        }
        if (size > IDLETOKEN_HTTP_BODY_CAP - request->body_len) { errno = EMSGSIZE; return -1; }
        size_t needed = request->body_len + size;
        if (needed > allocated) {
            size_t next = allocated ? allocated : 4096;
            while (next < needed) next *= 2;
            uint8_t *body = realloc(request->body, next);
            if (!body) { errno = ENOMEM; return -1; }
            request->body = body; allocated = next;
        }
        if (http_read_exact(reader, request->body + request->body_len, size)) return -1;
        request->body_len += size;
        char ending[2];
        if (http_read_exact(reader, ending, 2)) return -1;
        if (ending[0] != '\r' || ending[1] != '\n') { errno = EPROTO; return -1; }
    }
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

    /* Policy headers must survive the same header budget as framing headers. */
    {
        size_t hlen = hdr_end - hdr_start;
        if (hlen >= sizeof(out->headers)) { free(buf); errno = EMSGSIZE; return -1; }
        memcpy(out->headers, buf + hdr_start, hlen);
        out->headers[hlen] = 0;
    }

    long cl; int chunked, expect_continue;
    if (request_framing((const char *)buf + hdr_start, hdr_end - hdr_start,
                        &cl, &chunked, &expect_continue)) { free(buf); return -1; }
    if (expect_continue && (chunked || cl > 0)) {
        static const char interim[] = "HTTP/1.1 100 Continue\r\n\r\n";
        if (idletoken_sendall(conn_fd, interim, sizeof interim - 1) < 0) { free(buf); return -1; }
    }
    http_reader reader = { conn_fd, buf, head_len, (size_t)got, head_cap };
    int result = 0;
    if (chunked) result = http_read_chunked(&reader, out);
    else if (cl > 0) {
        out->body = malloc((size_t)cl);
        if (!out->body) { free(buf); errno = ENOMEM; return -1; }
        result = http_read_exact(&reader, out->body, (size_t)cl);
        if (!result) out->body_len = (size_t)cl;
    }
    free(buf);
    if (result) { free(out->body); out->body = NULL; out->body_len = 0; }
    return result;
}

static const char *http_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 402: return "Payment Required";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
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
