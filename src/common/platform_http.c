#define _POSIX_C_SOURCE 200809L
#include "idletoken_platform_http.h"
#include "idletoken_apiconv.h"
#include "platform_proxy.h"
#include <curl/curl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static pthread_once_t http_once = PTHREAD_ONCE_INIT;
static int http_ready;
#ifdef __APPLE__
static mach_timebase_info_data_t timebase;
static pthread_once_t time_once = PTHREAD_ONCE_INIT;
static void time_init(void) { mach_timebase_info(&timebase); }
#endif

int64_t idletoken_platform_now_ms(void) {
#ifdef _WIN32
    return (int64_t)GetTickCount64();
#elif defined(__APPLE__)
    pthread_once(&time_once, time_init);
    uint64_t ticks = mach_continuous_time();
    return (int64_t)((ticks / 1000000) * timebase.numer / timebase.denom
        + (ticks % 1000000) * timebase.numer / timebase.denom / 1000000);
#else
    struct timespec ts;
#ifdef CLOCK_BOOTTIME
    clock_gettime(CLOCK_BOOTTIME, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

static void http_init(void) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) return;
    const curl_version_info_data *v = curl_version_info(CURLVERSION_NOW);
    /* Synchronous DNS cannot honor a deadline in a multithreaded program. */
    http_ready = (v->features & CURL_VERSION_ASYNCHDNS) && (v->features & CURL_VERSION_SSL);
}

int idletoken_platform_url(const char *base, const char *path, char *out, size_t cap) {
    pthread_once(&http_once, http_init);
    if (!base || !*base || !out || !cap || strpbrk(base, "\r\n\t ")) return -1;
    CURLU *u = curl_url();
    char *scheme = NULL, *host = NULL, *part = NULL;
    char candidate[4096];
    int result = -1;
    if (!u) return -1;
    int n = snprintf(candidate, sizeof(candidate), "%s%s", strstr(base, "://") ? "" : "http://", base);
    if (n < 0 || (size_t)n >= sizeof(candidate) || curl_url_set(u, CURLUPART_URL, candidate, 0)) goto done;
    if (curl_url_get(u, CURLUPART_SCHEME, &scheme, 0) ||
        (strcmp(scheme, "http") && strcmp(scheme, "https")) ||
        curl_url_get(u, CURLUPART_HOST, &host, 0) || !*host) goto done;
    const CURLUPart forbidden[] = { CURLUPART_USER, CURLUPART_PASSWORD, CURLUPART_QUERY, CURLUPART_FRAGMENT };
    for (size_t i = 0; i < sizeof(forbidden)/sizeof(forbidden[0]); i++) {
        if (curl_url_get(u, forbidden[i], &part, 0) == CURLUE_OK) { curl_free(part); part = NULL; goto done; }
    }
    if (path && *path) {
        if (curl_url_get(u, CURLUPART_PATH, &part, 0)) goto done;
        size_t len = strlen(part);
        while (len && part[len - 1] == '/') len--;
        n = snprintf(candidate, sizeof(candidate), "%.*s%s%s", (int)len, part, *path == '/' ? "" : "/", path);
        curl_free(part); part = NULL;
        if (n < 0 || (size_t)n >= sizeof(candidate) || curl_url_set(u, CURLUPART_PATH, candidate, 0)) goto done;
    }
    if (curl_url_get(u, CURLUPART_URL, &part, 0)) goto done;
    n = snprintf(out, cap, "%s", part);
    result = n >= 0 && (size_t)n < cap ? 0 : -1;
done:
    curl_free(part); curl_free(scheme); curl_free(host); curl_url_cleanup(u);
    return result;
}

typedef struct {
    const idletoken_platform_http_request *request;
    idletoken_platform_http_response *response;
    int64_t start, response_start, body_start, last_byte, server_time_ms;
    /* Upload phase. `headers_ms` ("how long may the server take to start
     * answering") only starts counting once the request body has left this
     * process. A slow uplink carrying a multi-megabyte sealed reply is bounded
     * by the body budgets (idle/total), exactly as the socket-timeout transport
     * it replaced; it is never cut by a 10-30 s "start answering" timeout. */
    int64_t upload_done, last_upload_byte;
    curl_off_t upload_now;
    int receipt_checked;
    size_t capacity;
    char curl_error[CURL_ERROR_SIZE];
} transfer;

static int64_t json_number(const char *json, size_t len, const char *key) {
    const char *p = idletoken_json_obj_get(json, len, key), *end = json + len;
    if (!p || p >= end || *p < '0' || *p > '9') return -1;
    int64_t value = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        if (value > (INT64_MAX - 9) / 10) return -1;
        value = value * 10 + *p++ - '0';
    }
    /* Any JSON whitespace may follow the number. A gateway that started
     * pretty-printing would otherwise make every receipt read as expired. */
    if (p == end || (*p != ',' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t')) return -1;
    return value;
}

static void receipt_deadline(transfer *t) {
    if (!t->request->receipt || t->receipt_checked) return;
    idletoken_platform_http_response *r = t->response;
    size_t len = r->len < 2048 ? r->len : 2048;
    int64_t ttl = json_number((char *)r->body, len, "remaining_ms");
    if (ttl < 0 && t->server_time_ms) {
        int64_t expires = json_number((char *)r->body, len, "expires_at_ms");
        if (expires >= 0) ttl = expires > t->server_time_ms ? expires - t->server_time_ms : 0;
    }
    if (ttl >= 0) {
        /* remaining_ms is computed when the long poll produces a job, not
         * when the poll starts. Anchor once at the final response's first
         * header; header/body transfer still consumes this same budget. */
        if (ttl > 86400000) ttl = 86400000;
        r->delivery_deadline_ms = t->response_start + ttl;
        t->receipt_checked = 1;
    } else if (r->len >= 2048) t->receipt_checked = 1;
}

static size_t receive_body(char *data, size_t size, size_t count, void *context) {
    transfer *t = context;
    idletoken_platform_http_response *r = t->response;
    if (size && count > SIZE_MAX / size) return 0;
    size_t n = size * count;
    size_t max = t->request->max_response ? t->request->max_response : 64u * 1024u * 1024u;
    if (n > max || r->len > max - n) { snprintf(r->error, sizeof(r->error), "response exceeds byte limit"); return 0; }
    if (r->len + n + 1 > t->capacity) {
        size_t cap = t->capacity ? t->capacity : 8192;
        while (cap < r->len + n + 1 && cap <= max / 2) cap *= 2;
        if (cap < r->len + n + 1) cap = max + 1;
        unsigned char *next = realloc(r->body, cap);
        if (!next) return 0;
        r->body = next; t->capacity = cap;
    }
    memcpy(r->body + r->len, data, n); r->len += n; r->body[r->len] = 0;
    t->last_byte = idletoken_platform_now_ms();
    receipt_deadline(t);
    return n;
}

static size_t receive_header(char *data, size_t size, size_t count, void *context) {
    transfer *t = context;
    size_t n = size * count;
    if (n > 6 && (data[0] == 'D' || data[0] == 'd') && (data[1] == 'A' || data[1] == 'a') &&
        (data[2] == 'T' || data[2] == 't') && (data[3] == 'E' || data[3] == 'e') && data[4] == ':') {
        char date[128];
        if (n < sizeof(date)) {
            memcpy(date, data + 5, n - 5); date[n - 5] = 0;
            time_t stamp = curl_getdate(date, NULL);
            if (stamp >= 0) t->server_time_ms = (int64_t)stamp * 1000 + 999;
        }
    }
    if (n >= 12 && !memcmp(data, "HTTP/", 5)) {
        int status = 0;
        if (sscanf(data, "%*s %d", &status) == 1) t->response->status = status;
        t->response_start = idletoken_platform_now_ms();
        /* A status line means the peer consumed or refused the upload. */
        if (!t->upload_done) t->upload_done = t->response_start;
        t->server_time_ms = 0;
        t->body_start = 0;
    }
    if (n == 2 && !memcmp(data, "\r\n", 2) && t->response->status >= 200) {
        t->body_start = t->last_byte = idletoken_platform_now_ms();
    }
    return n;
}

/* libcurl reports upload progress here; that is the only signal that a slow
 * uplink is still moving. `ultotal` is not trusted: it is 0 until libcurl
 * learns the size, so completion is judged against the caller's body length. */
static int transfer_progress(void *context, curl_off_t dltotal, curl_off_t dlnow,
                             curl_off_t ultotal, curl_off_t ulnow) {
    transfer *t = context;
    (void)dltotal; (void)dlnow; (void)ultotal;
    if (ulnow > t->upload_now) { t->upload_now = ulnow; t->last_upload_byte = idletoken_platform_now_ms(); }
    if (!t->upload_done && ulnow >= (curl_off_t)t->request->body_len) t->upload_done = idletoken_platform_now_ms();
    return 0;
}

static int check_deadlines(transfer *t) {
    int64_t now = idletoken_platform_now_ms();
    const idletoken_platform_http_request *q = t->request;
    if (q->cancelled && q->cancelled(q->cancel_context)) { t->response->cancelled = 1; return 1; }
    int uploading = !t->upload_done;
    /* "Upload done" means the last byte was handed to the kernel, not that
     * the peer has it: a large body can still be draining out of the socket
     * buffers at link speed (a few megabytes at 40-80 KB/s is a minute or
     * more), and the peer cannot answer before it has read everything. For
     * large bodies the answer clock therefore also gets the idle budget --
     * the time we already allow anything to go unobserved before calling the
     * link dead. Small control bodies keep the caller's timeout unchanged. */
    int64_t answer_ms = q->headers_ms;
    if (answer_ms > 0 && q->body_idle_ms > 0 && q->body_len >= 65536) answer_ms += q->body_idle_ms;
    /* Three phases, three budgets: the upload is bounded by the body budgets
     * (a crawling link is caught by body_total, a dead one by body_idle); the
     * wait for the first response header by the answer clock, counted from
     * the end of the upload; the response body by the body budgets again.
     * total_ms and the delivery deadline cap all of them from the start. */
    if ((q->total_ms > 0 && now - t->start >= q->total_ms) ||
        (!uploading && !t->body_start && answer_ms > 0 && now - t->upload_done >= answer_ms) ||
        (uploading && q->body_total_ms > 0 && now - t->start >= q->body_total_ms) ||
        (t->body_start && q->body_total_ms > 0 && now - t->body_start >= q->body_total_ms) ||
        (t->response->delivery_deadline_ms && now >= t->response->delivery_deadline_ms)) {
        t->response->truncated = 2;
        snprintf(t->response->error, sizeof(t->response->error), "request deadline exceeded");
        return 1;
    }
    if (uploading && q->body_idle_ms > 0 && now - t->last_upload_byte >= q->body_idle_ms) {
        t->response->truncated = 1;
        snprintf(t->response->error, sizeof(t->response->error), "request body upload stalled");
        return 1;
    }
    if (t->body_start && q->body_idle_ms > 0 && now - t->last_byte >= q->body_idle_ms) {
        t->response->truncated = 1;
        snprintf(t->response->error, sizeof(t->response->error), "response body stalled");
        return 1;
    }
    return 0;
}

/* "host:port" of the route a failure happened on, credentials stripped. The
 * error string is the whole diagnostic surface a user gets (the balance pill
 * tooltip shows it verbatim), and "a proxy" is not actionable where
 * "127.0.0.1:7897" is: that address is what tells someone their proxy
 * variable points at a program that is not running. */
static void route_label(const char *proxy, char *out, size_t cap) {
    if (!proxy || !*proxy) { snprintf(out, cap, "direct"); return; }
    const char *p = strstr(proxy, "://");
    p = p ? p + 3 : proxy;
    const char *slash = strchr(p, '/'), *at = strchr(p, '@');
    if (at && (!slash || at < slash)) p = at + 1;
    size_t n = slash ? (size_t)(slash - p) : strlen(p);
    if (n >= cap) n = cap - 1;
    snprintf(out, cap, "%.*s", (int)n, p);
}

int idletoken_platform_http(const idletoken_platform_http_request *q,
                           idletoken_platform_http_response *r) {
    memset(r, 0, sizeof(*r));
    pthread_once(&http_once, http_init);
    if (!http_ready) { snprintf(r->error, sizeof(r->error), "libcurl requires TLS and asynchronous DNS"); return -1; }
    if (q->cancelled && q->cancelled(q->cancel_context)) {
        r->cancelled = 1; snprintf(r->error, sizeof(r->error), "request cancelled"); return -1;
    }
    transfer t = { .request = q, .response = r, .start = idletoken_platform_now_ms() };
    t.last_upload_byte = t.start;
    if (!q->body_len) t.upload_done = t.start; /* nothing to upload: answer clock runs now */
    int connect_ms = q->connect_ms > 0 ? q->connect_ms : 10000;
    if (q->total_ms > 0 && connect_ms > q->total_ms) connect_ms = q->total_ms;
    idletoken_proxy_routes routes;
    if (q->unix_socket) { memset(&routes, 0, sizeof(routes)); routes.count = 1; }
    else if (idletoken_platform_proxies_cancel(q->url, 5000, &routes, q->cancelled, q->cancel_context) != 0) {
        if (q->cancelled && q->cancelled(q->cancel_context)) r->cancelled = 1;
        snprintf(r->error, sizeof(r->error), "system proxy resolution failed or timed out; direct fallback blocked"); return -1;
    }
    /* Policy discovery has its own bounded budget above (a Windows WPAD
     * detection alone waits 3.5 s) and is re-done for every request. The
     * request's clocks start with the network transfer: charging discovery
     * to the per-second delivery ACK's 2 s budget failed every ACK on such a
     * machine and, after the 60 s lease, aborted a live inference for a
     * network that was fine. The delivery deadline is absolute; unaffected. */
    t.start = idletoken_platform_now_ms();
    t.last_upload_byte = t.start;
    if (!q->body_len) t.upload_done = t.start;
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Expect:");
    char auth[2048];
    if (q->bearer) {
        int n = snprintf(auth, sizeof(auth), "Authorization: Bearer %s", q->bearer);
        if (n < 0 || (size_t)n >= sizeof(auth) || strpbrk(q->bearer, "\r\n")) goto fail;
        headers = curl_slist_append(headers, auth);
    }
    if (q->headers) {
        char *copy = strdup(q->headers), *save = NULL;
        if (!copy) goto fail;
        for (char *line = strtok_r(copy, "\r\n", &save); line; line = strtok_r(NULL, "\r\n", &save)) headers = curl_slist_append(headers, line);
        free(copy);
    }
    for (unsigned route = 0; route < routes.count; route++) {
        /* Each route is a fresh transfer: the upload restarts from byte 0. */
        t.upload_now = 0; t.last_upload_byte = idletoken_platform_now_ms();
        t.upload_done = q->body_len ? 0 : t.last_upload_byte;
        t.curl_error[0] = 0;
        CURL *easy = curl_easy_init();
        CURLM *multi = curl_multi_init();
        if (!easy || !multi) { if (easy) curl_easy_cleanup(easy); if (multi) curl_multi_cleanup(multi); goto fail; }
        int64_t left = connect_ms - (idletoken_platform_now_ms() - t.start);
        if (left <= 0) { curl_easy_cleanup(easy); curl_multi_cleanup(multi); r->truncated = 2; goto fail; }
        curl_easy_setopt(easy, CURLOPT_URL, q->url);
        if (q->unix_socket && curl_easy_setopt(easy, CURLOPT_UNIX_SOCKET_PATH, q->unix_socket) != CURLE_OK) {
            curl_easy_cleanup(easy); curl_multi_cleanup(multi); goto fail;
        }
        curl_easy_setopt(easy, CURLOPT_CUSTOMREQUEST, q->method ? q->method : "POST");
        if (q->body || q->body_len) {
            curl_easy_setopt(easy, CURLOPT_POSTFIELDS, q->body ? q->body : "");
            curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)q->body_len);
        }
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, (long)left);
        curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(easy, CURLOPT_PROXY_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(easy, CURLOPT_PROXY_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(easy, CURLOPT_PROXY, routes.urls[route]);
        curl_easy_setopt(easy, CURLOPT_PROXYAUTH, (long)CURLAUTH_ANY);
        curl_easy_setopt(easy, CURLOPT_NOPROXY, ""); /* policy already resolved */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        /* The installed macOS libcurl may predate the SDK's headers. This
         * option is available on every supported OS; *_STR is much newer. */
        curl_easy_setopt(easy, CURLOPT_PROTOCOLS, (long)(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        curl_easy_setopt(easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(easy, CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, receive_body);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, &t);
        curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, receive_header);
        curl_easy_setopt(easy, CURLOPT_HEADERDATA, &t);
        curl_easy_setopt(easy, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(easy, CURLOPT_XFERINFOFUNCTION, transfer_progress);
        curl_easy_setopt(easy, CURLOPT_XFERINFODATA, &t);
        curl_easy_setopt(easy, CURLOPT_ERRORBUFFER, t.curl_error);
        const char *ca = getenv("CURL_CA_BUNDLE");
        if (ca && *ca) curl_easy_setopt(easy, CURLOPT_CAINFO, ca);
        curl_multi_add_handle(multi, easy);
        int running = 0, stopped = 0;
        CURLMcode mc;
        do {
            if (check_deadlines(&t)) { stopped = 1; break; }
            mc = curl_multi_perform(multi, &running);
            if (mc != CURLM_OK) { stopped = 1; break; }
            if (running) {
                int descriptors = 0;
                curl_multi_wait(multi, NULL, 0, 100, &descriptors);
                if (!descriptors) {
#ifdef _WIN32
                    Sleep(10);
#else
                    struct timespec pause = {0, 10000000}; nanosleep(&pause, NULL);
#endif
                }
            }
        } while (running);
        CURLcode code = CURLE_ABORTED_BY_CALLBACK;
        int messages;
        CURLMsg *message;
        while ((message = curl_multi_info_read(multi, &messages))) if (message->msg == CURLMSG_DONE) code = message->data.result;
        long status = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
        r->status = (int)status;
        double lookup_s = 0, connect_s = 0, tls_s = 0, first_s = 0;
        curl_easy_getinfo(easy, CURLINFO_NAMELOOKUP_TIME, &lookup_s);
        curl_easy_getinfo(easy, CURLINFO_CONNECT_TIME, &connect_s);
        curl_easy_getinfo(easy, CURLINFO_APPCONNECT_TIME, &tls_s);
        curl_easy_getinfo(easy, CURLINFO_STARTTRANSFER_TIME, &first_s);
        curl_multi_remove_handle(multi, easy);
        curl_easy_cleanup(easy); curl_multi_cleanup(multi);
        if (!stopped && check_deadlines(&t)) stopped = 1;
        if (!stopped && code == CURLE_OK) {
            if (!r->body) r->body = calloc(1, 1);
            curl_slist_free_all(headers);
            return r->body ? 0 : -1;
        }
        const char *phase = q->unix_socket ? "local IPC" :
            code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_COULDNT_RESOLVE_PROXY ? "DNS" :
            connect_s <= 0 ? (*routes.urls[route] ? "proxy connection" : "DNS/TCP connection") :
            !strncmp(q->url, "https://", 8) && tls_s <= 0 ? "TLS/proxy handshake" :
            !t.upload_done && q->body_len ? "request body upload" :
            t.body_start ? "response body" : "response headers";
        /* Prefer libcurl's specific message (it names the proxy address on a
         * connection failure) over the generic strerror text. */
        char cause[256], via[128];
        if (q->unix_socket) snprintf(via, sizeof(via), "unix socket");
        else route_label(routes.urls[route], via, sizeof(via));
        snprintf(cause, sizeof(cause), "%s", r->error[0] ? r->error :
                 t.curl_error[0] ? t.curl_error : curl_easy_strerror(code));
        snprintf(r->error, sizeof(r->error), "%.110s [%s via %.60s; dns=%.0fms tcp=%.0fms tls=%.0fms first=%.0fms]",
                 cause, phase, via, lookup_s * 1000, connect_s * 1000, tls_s * 1000, first_s * 1000);
        r->retryable = !r->cancelled && (code == CURLE_OPERATION_TIMEDOUT ||
            code == CURLE_COULDNT_CONNECT || code == CURLE_COULDNT_RESOLVE_PROXY ||
            code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_RECV_ERROR ||
            code == CURLE_SEND_ERROR || code == CURLE_GOT_NOTHING);
        /* Only try another policy-listed route when no HTTP request could
         * have reached the peer. Retrying an ambiguous POST belongs to the
         * receipt protocol, never to a generic transport. */
        if (!stopped && (code == CURLE_COULDNT_CONNECT || code == CURLE_COULDNT_RESOLVE_PROXY || code == CURLE_COULDNT_RESOLVE_HOST) && route + 1 < routes.count) {
            r->error[0] = 0; continue;
        }
        break;
    }
fail:
    if (!r->truncated) r->truncated = 1;
    if (!r->error[0]) snprintf(r->error, sizeof(r->error), "HTTP transport failed");
    curl_slist_free_all(headers);
    return -1;
}
