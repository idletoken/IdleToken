/* llama_sidecar.c — spawn + supervise a local idletoken-server (WS-B1).
 *
 * See include/idletoken_llama_sidecar.h for the contract. Design notes:
 *
 *  - One monitor thread per sidecar drives the whole state machine; the HTTP
 *    handlers only take mutex-protected snapshots. The thread polls every
 *    LLAMA_POLL_MS: reaps the child (waitpid WNOHANG), probes /health while
 *    STARTING, and respawns after the backoff while RESTARTING.
 *
 *  - Backoff mirrors the client supervisor (client/src-tauri/src/engine.rs):
 *    2s, 4s, 8s, 16s, 30s; the 6th consecutive quick crash latches FAILED.
 *    "Quick" means the child never reached STABLE_UPTIME_MS of READY uptime.
 *
 *  - The child's stdout/stderr go to a log file of their own, opened in the
 *    child after fork. idletoken-server's default log level prints request
 *    metadata but no prompt text; we deliberately pass NO extra verbosity
 *    flags (no -v, never --log-prompts-dir) so that stays true. The
 *    coordinator's own stderr never carries engine output at all.
 */
#include "idletoken_llama_sidecar.h"
#include "idletoken_net.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>   /* engine-log size at spawn (see llama_scan_log) */

#ifdef _WIN32
  #include <winsock2.h>
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <signal.h>
  #include <sys/socket.h>
  #include <sys/wait.h>
  #include <unistd.h>
  #ifdef __linux__
    #include <sys/prctl.h>
  #endif
#endif

#define LLAMA_POLL_MS          300     /* monitor loop cadence */
#define LLAMA_MAX_QUICK_RESTARTS 5     /* = engine.rs MAX_QUICK_RESTARTS */
#define LLAMA_STABLE_UPTIME_MS 60000   /* = engine.rs STABLE_UPTIME_MS */
#define LLAMA_HEALTH_TIMEOUT_MS 2000
/* Bounded waiting on the engine while relaying inference (see
 * idletoken_llama_http_watch). The slice only bounds how often we get to look
 * around; it is NOT a deadline for the engine. */
#define LLAMA_READ_SLICE_MS     5000
/* Silence shorter than this is ordinary prefill on a large model — do not go
 * knocking. Past it, ask the engine whether it is still there, no more often
 * than LLAMA_PROBE_EVERY_MS. */
#define LLAMA_SILENCE_GRACE_MS 15000
#define LLAMA_PROBE_EVERY_MS   10000
#define LLAMA_ARGV_MAX 64

struct idletoken_llama {
    /* config (immutable after start) */
    char bin[512];
    char gguf[1024];
    char log_path[512];
    char cluster_args[1024];  /* WS-C cluster flags (--rpc/--device/--tensor-split) */
    char extra_args[1024];    /* IDLETOKEN_LLAMA_ARGS copy, split at spawn */
    int  shared;              /* serving OTHER people's requests — see below */
    char sock_path[256];      /* AF_UNIX path, or "" for TCP loopback */
    char endpoint[300];       /* "127.0.0.1:<port>" | "unix:<sock_path>" */
    int  port;
    uint32_t ctx_size;        /* context of ONE slot */
    int  n_parallel;          /* -np: independent sequences the engine serves */

    /* state (under mu) */
    pthread_mutex_t mu;
    pthread_t tid;
    int  thread_up;
    int  stop;
    idletoken_llama_state state;
    long long pid;            /* 0 = no child */
    int  restarts_total;      /* respawns performed (for /idletoken/v1/stats) */
    int  quick_restarts;      /* consecutive crashes without stable uptime */
    long long spawned_ms;     /* when the current child was exec'd */
    long long became_ready_ms;/* when it last turned READY (0 = never) */
    long long restart_due_ms; /* RESTARTING: when the backoff expires */
    char fail[700];           /* FAILED: the reason, for the API surface */
    /* Engine-log watch (see idletoken_llama_fatal_reason). The log is APPENDED
     * to across runs, so the offset is stamped at each spawn: scanning from 0
     * would re-read a previous run's warning and refuse a start that is fine. */
    long long log_off;
    char fatal[700];          /* startup condition the coordinator must refuse over */
#ifdef _WIN32
    HANDLE child_handle;      /* kept until exit is observed / shutdown */
    HANDLE job;               /* kill-on-close: no orphan on hard coord exit */
#endif
};

static long long llama_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void llama_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
#endif
}

const char *idletoken_llama_state_name(idletoken_llama_state st) {
    switch (st) {
        case IDLETOKEN_LLAMA_OFF:        return "off";
        case IDLETOKEN_LLAMA_STARTING:   return "starting";
        case IDLETOKEN_LLAMA_READY:      return "ready";
        case IDLETOKEN_LLAMA_RESTARTING: return "restarting";
        case IDLETOKEN_LLAMA_FAILED:     return "failed";
    }
    return "unknown";
}

/* --- HTTP client ---------------------------------------------------------- */

static ssize_t conn_recv(int fd, void *buf, size_t cap) {
#ifdef _WIN32
    int r = recv((SOCKET)fd, (char *)buf, cap > INT_MAX ? INT_MAX : (int)cap, 0);
    return (ssize_t)r;
#else
    ssize_t r;
    do { r = recv(fd, buf, cap, 0); } while (r < 0 && errno == EINTR);
    return r;
#endif
}

static void conn_set_timeout(int fd, int timeout_ms) {
    if (timeout_ms <= 0) return;
#ifdef _WIN32
    DWORD tv = (DWORD)timeout_ms;
    setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
    setsockopt((SOCKET)fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));
#else
    struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

/* Case-insensitive substring search inside the raw header block. */
static const char *hdr_find(const char *hay, size_t hay_len, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || nlen > hay_len) return NULL;
    for (size_t i = 0; i + nlen <= hay_len; i++) {
        size_t j = 0;
        while (j < nlen) {
            char a = hay[i + j], b = needle[j];
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
            if (a != b) break;
            j++;
        }
        if (j == nlen) return hay + i;
    }
    return NULL;
}

int idletoken_llama_http_open(const char *endpoint, const char *method,
                              const char *path,
                              const char *body, size_t body_len,
                              int timeout_ms, idletoken_llama_conn *c) {
    memset(c, 0, sizeof(*c));
    c->fd = -1;
    c->content_left = -1;

    if (!endpoint || !endpoint[0]) return -1;
    /* "unix:<path>" is the shared-mode transport. The Host header still has to
     * be a name (cpp-httplib does not care which), so a socket endpoint sends
     * "localhost" rather than a filesystem path. */
    int is_unix = strncmp(endpoint, "unix:", 5) == 0;
    const char *host_hdr = is_unix ? "localhost" : endpoint;
    int fd = is_unix ? idletoken_connect_unix(endpoint + 5)
                     : idletoken_connect_tcp(endpoint);
    if (fd < 0) return -1;
    conn_set_timeout(fd, timeout_ms);

    char head[512];
    int hn = snprintf(head, sizeof(head),
                      "%s %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n",
                      method, path, host_hdr, body_len);
    if (hn < 0 || (size_t)hn >= sizeof(head)) { idletoken_close_fd(fd); return -1; }
    if (idletoken_sendall(fd, head, (size_t)hn) < 0 ||
        (body_len && idletoken_sendall(fd, body, body_len) < 0)) {
        idletoken_close_fd(fd);
        return -1;
    }

    /* Read until the header terminator; whatever follows it stays buffered. */
    size_t hdr_end = 0;
    for (;;) {
        if (c->blen + 1 >= sizeof(c->buf)) { idletoken_close_fd(fd); return -1; }
        ssize_t r = conn_recv(fd, c->buf + c->blen, sizeof(c->buf) - 1 - c->blen);
        if (r <= 0) { idletoken_close_fd(fd); return -1; }
        c->blen += (size_t)r;
        c->buf[c->blen] = '\0';
        char *sep = strstr(c->buf, "\r\n\r\n");
        if (sep) { hdr_end = (size_t)(sep - c->buf) + 4; break; }
    }

    /* Status line: "HTTP/1.1 200 OK". */
    if (c->blen < 12 || strncmp(c->buf, "HTTP/1.", 7) != 0) {
        idletoken_close_fd(fd);
        return -1;
    }
    c->status = atoi(c->buf + 9);

    const char *te = hdr_find(c->buf, hdr_end, "transfer-encoding:");
    if (te && hdr_find(te, hdr_end - (size_t)(te - c->buf), "chunked"))
        c->chunked = 1;
    const char *cl = hdr_find(c->buf, hdr_end, "content-length:");
    if (cl && !c->chunked) {
        cl += strlen("content-length:");
        while (*cl == ' ' || *cl == '\t') cl++;
        c->content_left = atoll(cl);
    }

    c->fd = fd;
    c->boff = hdr_end;   /* leftover payload bytes start here */
    return 0;
}

/* Did that recv fail only because the slice expired, or for a real reason? */
static int conn_recv_timed_out(void) {
#ifdef _WIN32
    int e = WSAGetLastError();
    return e == WSAETIMEDOUT || e == WSAEWOULDBLOCK;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

static int llama_health_ok(const char *endpoint);   /* defined below */

/* recv() with the bounded-wait policy applied (see idletoken_llama_http_watch).
 *
 * EVERY read goes through here. Routing only one of the two readers through it
 * is worse than routing neither: idletoken_llama_http_watch sets SO_RCVTIMEO on
 * the socket, so a reader that does not know about slices sees an ordinary
 * timeout and calls the stream dead — which would kill any prefill silent for
 * longer than one slice, i.e. every large model. Measured: with only
 * conn_raw_read converted, a frozen engine was reported as `decode_failed`
 * after 5s and the liveness check never ran at all. */
static ssize_t conn_recv_watched(idletoken_llama_conn *c, void *dst, size_t cap) {
    for (;;) {
        ssize_t r = conn_recv(c->fd, dst, cap);
        if (r >= 0) {                      /* any byte means the engine is there */
            c->silent_ms = 0;
            c->probed_at_ms = 0;
            return r;
        }
        /* Not watching, or a genuine socket error: unchanged behaviour. */
        if (c->slice_ms <= 0 || !conn_recv_timed_out()) return r;
        c->silent_ms += c->slice_ms;
        if (!c->watch || c->silent_ms < LLAMA_SILENCE_GRACE_MS ||
            c->silent_ms - c->probed_at_ms < LLAMA_PROBE_EVERY_MS)
            continue;                      /* still plausibly just a long prefill */
        c->probed_at_ms = c->silent_ms;
        if (llama_health_ok(c->watch)) continue;   /* alive and working: keep waiting */
        /* It is not answering any more, and waiting longer cannot help. The
         * check is PER CONNECTION, which is what keeps this to one lost
         * request: the other pool threads go on serving their own slots.
         * (Until the pool landed on 2026-08-18 this thread was the
         * coordinator's only executor, so parking here stopped everything —
         * that is the failure the check was written for, and the reason it
         * still matters is that a dead relay must not become a dead machine.) */
        fprintf(stderr,
                "coord: llama-sidecar: the engine stopped answering (%llds of silence, "
                "and GET /health no longer succeeds) — failing this request rather than "
                "waiting forever; other in-flight requests are unaffected\n",
                (long long)(c->silent_ms / 1000));
        c->eof = 1;
        return -1;
    }
}

/* One raw payload byte (post-header stream): buffered leftover first, then the
 * socket. Returns 0..255, or -1 on EOF/error. */
static int conn_byte(idletoken_llama_conn *c) {
    if (c->boff >= c->blen) {
        ssize_t r = conn_recv_watched(c, c->buf, sizeof(c->buf));
        if (r <= 0) return -1;
        c->boff = 0;
        c->blen = (size_t)r;
    }
    return (unsigned char)c->buf[c->boff++];
}

static ssize_t conn_raw_read(idletoken_llama_conn *c, void *dst, size_t cap) {
    if (cap == 0) return 0;
    if (c->boff < c->blen) {
        size_t have = c->blen - c->boff;
        if (have > cap) have = cap;
        memcpy(dst, c->buf + c->boff, have);
        c->boff += have;
        return (ssize_t)have;
    }
    return conn_recv_watched(c, dst, cap);
}

void idletoken_llama_http_watch(idletoken_llama_conn *c, const char *endpoint) {
    if (!c || c->fd < 0) return;
    c->slice_ms     = LLAMA_READ_SLICE_MS;
    c->watch        = endpoint;
    c->silent_ms    = 0;
    c->probed_at_ms = 0;
    conn_set_timeout(c->fd, LLAMA_READ_SLICE_MS);
}

ssize_t idletoken_llama_http_read(idletoken_llama_conn *c, void *dst, size_t cap) {
    if (c->eof || c->fd < 0 || cap == 0) return 0;

    if (!c->chunked) {
        size_t want = cap;
        if (c->content_left >= 0) {
            if (c->content_left == 0) { c->eof = 1; return 0; }
            if ((long long)want > c->content_left) want = (size_t)c->content_left;
        }
        ssize_t r = conn_raw_read(c, dst, want);
        if (r <= 0) {
            c->eof = 1;
            if (r < 0) return -1;
            /* EOF before Content-Length ran out is a transport error; with no
             * Content-Length (close-delimited body) it is the normal end. */
            return c->content_left > 0 ? -1 : 0;
        }
        if (c->content_left > 0) c->content_left -= r;
        return r;
    }

    /* chunked */
    while (c->chunk_left == 0) {
        /* chunk-size line: HEX[;ext]CRLF (a stray CRLF may precede it — the
         * terminator of the previous chunk when we consumed it lazily). */
        char line[64];
        size_t ll = 0;
        for (;;) {
            int b = conn_byte(c);
            if (b < 0) { c->eof = 1; return -1; }
            if (b == '\n') break;
            if (b != '\r' && ll + 1 < sizeof(line)) line[ll++] = (char)b;
        }
        line[ll] = '\0';
        if (ll == 0) continue;   /* the CRLF terminating the previous chunk */
        long long sz = strtoll(line, NULL, 16);
        if (sz < 0) { c->eof = 1; return -1; }
        if (sz == 0) {
            /* last chunk: drain optional trailers up to the blank line */
            int prev_nl = 0;
            for (;;) {
                int b = conn_byte(c);
                if (b < 0) break;
                if (b == '\n') { if (prev_nl) break; prev_nl = 1; }
                else if (b != '\r') prev_nl = 0;
            }
            c->eof = 1;
            return 0;
        }
        c->chunk_left = sz;
    }

    size_t want = cap;
    if ((long long)want > c->chunk_left) want = (size_t)c->chunk_left;
    ssize_t r = conn_raw_read(c, dst, want);
    if (r <= 0) { c->eof = 1; return -1; }
    c->chunk_left -= r;
    return r;
}

char *idletoken_llama_http_read_all(idletoken_llama_conn *c, size_t *out_len,
                                    size_t max_bytes) {
    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (len + 4096 + 1 > cap) {
            if (cap >= max_bytes) { free(buf); return NULL; }
            size_t ncap = cap * 2;
            char *nb = realloc(buf, ncap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
            cap = ncap;
        }
        ssize_t r = idletoken_llama_http_read(c, buf + len, 4096);
        if (r < 0) { free(buf); return NULL; }
        if (r == 0) break;
        len += (size_t)r;
    }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

void idletoken_llama_http_close(idletoken_llama_conn *c) {
    if (c->fd >= 0) idletoken_close_fd(c->fd);
    c->fd = -1;
}

/* --- readiness probe ------------------------------------------------------ */

/* READY means GET /health answered 200 with EXACTLY {"status":"ok"}.
 * idletoken-server binds its port while the model is still loading and answers
 * 503 {"error":{"message":"Loading model",...}} there — a status-only or
 * connect-only check reports ready long before the model exists (this exact
 * mistake invalidated three measurement runs; see the header comment). */
static int llama_health_ok(const char *endpoint) {
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(endpoint, "GET", "/health", NULL, 0,
                                  LLAMA_HEALTH_TIMEOUT_MS, &c) != 0)
        return 0;
    size_t blen = 0;
    char *body = idletoken_llama_http_read_all(&c, &blen, 4096);
    int st = c.status;
    idletoken_llama_http_close(&c);
    if (!body) return 0;
    /* trim surrounding whitespace before the exact compare */
    char *b = body;
    while (*b == ' ' || *b == '\t' || *b == '\r' || *b == '\n') b++;
    char *e = b + strlen(b);
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) *--e = '\0';
    int ok = (st == 200) && (strcmp(b, "{\"status\":\"ok\"}") == 0);
    free(body);
    return ok;
}

/* --- spawn ---------------------------------------------------------------- */

/* Locking argv is only half of it. llama.cpp applies environment variables
 * BEFORE the command line (common/arg.cpp common_params_parse_ex: "handle
 * environment variables" runs first), and 138 of its flags carry one. Argv
 * wins for anything we pass — but for anything we DON'T, the environment is in
 * sole control, and the child inherits ours, which the machine's owner sets.
 * LLAMA_ARG_LOG_PROMPTS_DIR does not exist, but LLAMA_ARG_RPC + LLAMA_ARG_DEVICE
 * would move layer 0 onto another machine (breaking the invariant that keeps
 * the prompt recoverable only where the embedding table is), and the next
 * upstream release may add one that matters more.
 *
 * So shared mode does to the environment what it already does to argv: an
 * allow-list whose allowed set is (almost) empty. Everything in the engine's
 * own namespaces goes, and exactly one variable comes back — GGML_RPC_PSK,
 * which the coordinator sets itself and the cluster TLS link cannot work
 * without. Note this also drops GGML_RPC_ALLOW_PLAINTEXT, which is right:
 * "testing only" is not a thing to honour while holding someone else's prompt.
 *
 * Deliberately NOT applied in local mode — LLAMA_ARG_* is how you experiment
 * with your own engine. */
static int llama_env_is_engine_namespace(const char *name) {
    return strncmp(name, "LLAMA_", 6) == 0 ||
           strncmp(name, "GGML_", 5) == 0 ||
           strncmp(name, "HF_", 3) == 0;
}

#ifndef _WIN32
extern char **environ;

/* Called in the CHILD between fork and exec. The names are collected first:
 * unsetenv rewrites `environ` underneath a walk of it. */
static void llama_scrub_env(void) {
    char keep_psk[160] = "";
    const char *psk = getenv("GGML_RPC_PSK");
    if (psk) snprintf(keep_psk, sizeof(keep_psk), "%s", psk);

    char names[128][64];
    int n = 0;
    for (char **e = environ; *e && n < 128; e++) {
        const char *eq = strchr(*e, '=');
        size_t len = eq ? (size_t)(eq - *e) : strlen(*e);
        if (len >= sizeof(names[0])) continue;
        char nm[64];
        memcpy(nm, *e, len);
        nm[len] = '\0';
        if (llama_env_is_engine_namespace(nm)) snprintf(names[n++], 64, "%s", nm);
    }
    for (int i = 0; i < n; i++) unsetenv(names[i]);
    if (keep_psk[0]) setenv("GGML_RPC_PSK", keep_psk, 1);
}
#endif

/* Spawn the child. Called with the mutex held (fork+exec does not block).
 * Returns 0 and fills lc->pid / lc->spawned_ms, or -1 with lc->fail set. */
static int llama_spawn(idletoken_llama *lc) {
    {
        const char *plain = getenv("GGML_RPC_ALLOW_PLAINTEXT");
        if (plain && !strcmp(plain, "1")) {
            /* The engine-side patch prints its own banner, but that lands in
             * the engine log — repeat it HERE so nobody reading only the
             * coordinator log mistakes this cluster for a TLS one. */
            fprintf(stderr,
                    "coord: ==========================================================\n"
                    "coord: == GGML_RPC_ALLOW_PLAINTEXT=1 is set: idletoken-server may  ==\n"
                    "coord: == move tensor traffic WITHOUT TLS. Testing only —      ==\n"
                    "coord: == never run a real cluster this way.                   ==\n"
                    "coord: ==========================================================\n");
        }
    }
    /* A socket file left behind by a killed run makes the engine's bind fail
     * with "address already in use" — AF_UNIX binds a path, and SO_REUSEADDR
     * has nothing to say about that. Clearing it here covers respawns too. */
    if (lc->sock_path[0]) {
#ifdef _WIN32
        DeleteFileA(lc->sock_path);
#else
        unlink(lc->sock_path);
#endif
    }

    /* Start reading the engine log where THIS child will start writing it. The
     * file is shared by every run of this port (append mode), so the previous
     * run's lines must not be attributed to the child we are about to spawn. */
    {
        struct stat lst;
        lc->log_off = (lc->log_path[0] && stat(lc->log_path, &lst) == 0)
                          ? (long long)lst.st_size : 0;
    }

    /* `-c` is the engine's TOTAL KV budget and `-np` divides it (see the header:
     * n_ctx_seq = n_ctx / n_seq_max). Multiply here, in ONE place, so neither
     * platform branch can be the one that forgets and quietly hands every
     * conversation a quarter of its window. */
    const int  npar    = lc->n_parallel > 0 ? lc->n_parallel : 1;
    const uint64_t ctx_total = (uint64_t)lc->ctx_size * (uint64_t)npar;

#ifdef _WIN32
    char portstr[16], ctxstr[24], nparstr[16], cmd[4096], listen_args[320];
    snprintf(portstr, sizeof(portstr), "%d", lc->port);
    snprintf(ctxstr, sizeof(ctxstr), "%llu", (unsigned long long)ctx_total);
    snprintf(nparstr, sizeof(nparstr), "%d", npar);
    /* The engine picks AF_UNIX off the host name alone: a `--host` ending in
     * .sock switches address families (llama.cpp server-http.cpp), and --port
     * is then meaningless, so it is left out rather than logged misleadingly. */
    if (lc->sock_path[0])
        snprintf(listen_args, sizeof(listen_args), "--host \"%s\"", lc->sock_path);
    else
        snprintf(listen_args, sizeof(listen_args), "--host 127.0.0.1 --port %s", portstr);
    int n = snprintf(cmd, sizeof(cmd),
                     "\"%s\" -m \"%s\" %s%s "
                     "-ngl 99 --reasoning off%s%s -np %s%s%s",
                     lc->bin, lc->gguf, listen_args,
                     lc->shared ? " --no-slots" : "",
                     lc->ctx_size > 0 ? " -c " : "",
                     lc->ctx_size > 0 ? ctxstr : "",
                     nparstr,
                     lc->cluster_args[0] ? " " : "", lc->cluster_args);
    if (n < 0 || (size_t)n >= sizeof(cmd)) {
        snprintf(lc->fail, sizeof(lc->fail), "idletoken-server command line is too long");
        return -1;
    }
    if (lc->extra_args[0]) {
        size_t used = strlen(cmd), need = strlen(lc->extra_args) + 2;
        if (used + need >= sizeof(cmd)) {
            snprintf(lc->fail, sizeof(lc->fail), "IDLETOKEN_LLAMA_ARGS is too long");
            return -1;
        }
        cmd[used++] = ' ';
        snprintf(cmd + used, sizeof(cmd) - used, "%s", lc->extra_args);
    }

    /* The environment the child gets. NULL = inherit ours; in shared mode we
     * hand over a filtered block instead (see llama_env_is_engine_namespace).
     * The block is "NAME=VAL\0NAME=VAL\0\0" and must stay alive across
     * CreateProcess, hence the malloc rather than a scratch local. */
    char *child_env = NULL;
    if (lc->shared) {
        LPCH all = GetEnvironmentStringsA();
        if (!all) {
            snprintf(lc->fail, sizeof(lc->fail),
                     "cannot read this process's environment to filter it "
                     "(winerr %lu)", (unsigned long)GetLastError());
            return -1;
        }
        size_t total = 0;
        for (LPCH p = all; *p; p += strlen(p) + 1) total += strlen(p) + 1;
        child_env = (char *)malloc(total + 2);
        if (!child_env) {
            FreeEnvironmentStringsA(all);
            snprintf(lc->fail, sizeof(lc->fail), "out of memory");
            return -1;
        }
        size_t off = 0;
        for (LPCH p = all; *p; p += strlen(p) + 1) {
            /* An entry may begin with '=' (the per-drive cwd variables Windows
             * keeps, e.g. "=C:=C:\path"); those are not ours to judge. */
            if (p[0] != '=' && llama_env_is_engine_namespace(p) &&
                strncmp(p, "GGML_RPC_PSK=", 13) != 0)
                continue;
            size_t n2 = strlen(p) + 1;
            memcpy(child_env + off, p, n2);
            off += n2;
        }
        child_env[off++] = '\0';
        child_env[off] = '\0';
        FreeEnvironmentStringsA(all);
    }

    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE log = CreateFileA(lc->log_path, FILE_APPEND_DATA,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (log == INVALID_HANDLE_VALUE) {
        snprintf(lc->fail, sizeof(lc->fail), "cannot open engine log %s (winerr %lu)",
                 lc->log_path, (unsigned long)GetLastError());
        free(child_env);
        return -1;
    }
    SetFilePointer(log, 0, NULL, FILE_END);
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = log;
    si.hStdError = log;
    BOOL ok = CreateProcessA(lc->bin, cmd, NULL, NULL, TRUE,
                             CREATE_NO_WINDOW, child_env, NULL, &si, &pi);
    DWORD werr = ok ? 0 : GetLastError();
    CloseHandle(log);
    free(child_env);
    if (!ok) {
        snprintf(lc->fail, sizeof(lc->fail), "CreateProcess %s failed (winerr %lu)",
                 lc->bin, (unsigned long)werr);
        return -1;
    }
    CloseHandle(pi.hThread);
    if (lc->job && !AssignProcessToJobObject(lc->job, pi.hProcess)) {
        fprintf(stderr, "coord: llama-sidecar: warning: could not attach child "
                        "to kill-on-close job (winerr %lu)\n",
                (unsigned long)GetLastError());
    }
    lc->child_handle = pi.hProcess;
    lc->pid = (long long)pi.dwProcessId;
    lc->spawned_ms = llama_now_ms();
    return 0;
#else
    char portstr[16], ctxstr[24], nparstr[16];
    snprintf(portstr, sizeof(portstr), "%d", lc->port);
    snprintf(ctxstr, sizeof(ctxstr), "%llu", (unsigned long long)ctx_total);
    snprintf(nparstr, sizeof(nparstr), "%d", npar);

    char *argv[LLAMA_ARGV_MAX];
    int argc = 0;
    argv[argc++] = lc->bin;
    argv[argc++] = "-m";        argv[argc++] = lc->gguf;
    /* Never reachable from the LAN: the coordinator's api_token gate must
     * remain the only gate. Shared mode goes one further and leaves the IP
     * stack entirely — the engine picks AF_UNIX off a `--host` ending in .sock
     * (llama.cpp server-http.cpp), which is why --port is omitted there: with
     * a socket it is meaningless, and printing one would misdescribe the link. */
    if (lc->sock_path[0]) {
        argv[argc++] = "--host";    argv[argc++] = lc->sock_path;
    } else {
        argv[argc++] = "--host";    argv[argc++] = "127.0.0.1";
        argv[argc++] = "--port";    argv[argc++] = portstr;
    }
    /* GET /slots hands back every live slot's prompt AND its generated text
     * verbatim (llama.cpp server-context.cpp: res["prompt"] / res["generated"]),
     * and upstream has it ON by default (common.h endpoint_slots = true). That
     * is a prompt reader we were shipping without noticing — one curl, no
     * tools. Nothing in the coordinator calls /slots, so turning it off in
     * shared mode costs nothing. Local use keeps it: it is a genuinely useful
     * window into your own machine.
     * (Prompts on DISK were already impossible: --slot-save-path is never
     * passed, and without it the engine refuses the save/restore action.) */
    if (lc->shared) argv[argc++] = "--no-slots";
    argv[argc++] = "-ngl";      argv[argc++] = "99";
    /* Reasoning off by default: thinking models (Qwen3.5 etc.) otherwise burn
     * the whole token budget inside <think> and the visible answer comes back
     * EMPTY with finish_reason "length" — measured with Qwen3.5-0.8B at
     * max_tokens 200. An empty reply is a broken product; thinking support is
     * a future client toggle. Overridable via IDLETOKEN_LLAMA_ARGS (appended
     * last, so a user-supplied --reasoning wins). */
    argv[argc++] = "--reasoning"; argv[argc++] = "off";
    if (lc->ctx_size > 0) { argv[argc++] = "-c"; argv[argc++] = ctxstr; }
    /* Continuous batching is already the upstream default, so `-np` alone is
     * enough to get overlapping prefill/decode across slots. */
    argv[argc++] = "-np";       argv[argc++] = nparstr;
    /* No extra log flags on purpose: the default level logs request metadata
     * but no prompt text. Never add -v or --log-prompts-dir here. */

    /* Cluster args first (their internal order is load-bearing: --rpc must
     * precede --device so RPC device names resolve), then the user's
     * IDLETOKEN_LLAMA_ARGS LAST so on duplicate flags idletoken-server takes the
     * user's value. Split into scratch copies: the config strings must
     * survive for the next respawn. */
    static char cluster_scratch[sizeof(lc->cluster_args)];
    snprintf(cluster_scratch, sizeof(cluster_scratch), "%s", lc->cluster_args);
    static char extra_scratch[sizeof(lc->extra_args)];
    snprintf(extra_scratch, sizeof(extra_scratch), "%s", lc->extra_args);
    char *lists[2] = { cluster_scratch, extra_scratch };
    for (int li = 0; li < 2; li++) {
        char *p = lists[li];
        while (*p && argc + 2 < LLAMA_ARGV_MAX) {
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            argv[argc++] = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            if (*p) *p++ = '\0';
        }
    }
    argv[argc] = NULL;

    pid_t pid = fork();
    if (pid < 0) {
        snprintf(lc->fail, sizeof(lc->fail), "fork: %s", strerror(errno));
        return -1;
    }
    if (pid == 0) {
#ifdef __linux__
        /* die with the coordinator even on SIGKILL (no such hook on macOS;
         * there the coordinator's own signal path does the cleanup) */
        prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
        if (lc->shared) llama_scrub_env();
        /* child: engine output goes to its own log file, not our stderr */
        int lg = open(lc->log_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (lg >= 0) {
            dup2(lg, 1);
            dup2(lg, 2);
            if (lg > 2) close(lg);
        }
        execv(lc->bin, argv);
        /* exec failed: say so in the engine log (fd 2 is the log file now) */
        dprintf(2, "idletoken-coord: execv %s: %s\n", lc->bin, strerror(errno));
        _exit(127);
    }
    lc->pid = (long long)pid;
    lc->spawned_ms = llama_now_ms();
    return 0;
#endif
}

/* --- placement-flag guard --------------------------------------------------
 *
 * Contract and the reasoning: the header. Pure function of the string so
 * coord --selftest can drive both directions. */
const char *idletoken_llama_placement_flag(const char *args) {
    /* The closed set of flags that decide where tensors live. `-dev`/`-ts` are
     * upstream's short aliases for `--device`/`--tensor-split`; refusing the
     * long form alone would be a doorway, not a guard. */
    static const char *placement[] = { "--device", "-dev", "--rpc",
                                       "--tensor-split", "-ts" };
    if (!args || !args[0]) return NULL;

    /* Walk token by token rather than substring-searching the whole string:
     * "--devices-note" contains "--device" and must stay allowed, and a
     * placement flag appearing inside somebody's VALUE ("--chat-template-file
     * /opt/--device") is not a flag either. */
    const char *p = args;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t') p++;

        /* The token, cut at '=' so `--device=RPC0` compares as `--device`. */
        size_t len = (size_t)(p - start);
        const char *eq = memchr(start, '=', len);
        if (eq) len = (size_t)(eq - start);

        char tok[64];
        if (len < sizeof(tok)) {
            memcpy(tok, start, len);
            tok[len] = '\0';
            /* Upstream normalises '_' to '-' for every '--' argument
             * (common/arg.cpp), so `--tensor_split` IS `--tensor-split` to the
             * engine. Do the same before comparing, or the guard has a bypass
             * one keystroke wide — measured 2026-08-20. Only for '--' tokens,
             * matching upstream exactly: a short flag keeps its spelling. */
            if (tok[0] == '-' && tok[1] == '-')
                for (char *c = tok; *c; c++) if (*c == '_') *c = '-';
            for (size_t i = 0; i < sizeof(placement) / sizeof(*placement); i++)
                if (!strcmp(tok, placement[i])) return placement[i];
        }
    }
    return NULL;
}

/* --- monitor thread ------------------------------------------------------- */

/* --- engine-log watch ------------------------------------------------------
 *
 * Contract and the reasoning about upstream's wording: the header. Kept as a
 * pure function of the text so coord --selftest can drive it with a real log
 * line AND with lines that must NOT match. */
int idletoken_llama_log_fit_failed(const char *text) {
    if (!text) return 0;
    /* The sentence, not the function name: __func__ could be renamed upstream
     * without the message changing meaning, and matching both would make this
     * more brittle for no gain. Deliberately NOT matching the milder
     * "encountered an error while trying to fit params" — that one means the
     * fit itself broke (no model at the path etc.), and the engine's own exit
     * already reports it. */
    return strstr(text, "failed to fit params to free device memory") != NULL;
}

/* Read whatever the child appended since the last poll and look for conditions
 * the coordinator must refuse over. Called with the mutex held. Line-oriented:
 * a trailing partial line is left for the next poll rather than half-matched. */
static void llama_scan_log(idletoken_llama *lc) {
    if (!lc->log_path[0] || lc->fatal[0]) return;
    FILE *f = fopen(lc->log_path, "rb");
    if (!f) return;                       /* not written yet — nothing to read */
    if (fseek(f, (long)lc->log_off, SEEK_SET) != 0) { fclose(f); return; }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        const size_t n = strlen(line);
        if (n == 0) break;
        if (line[n - 1] != '\n' && n < sizeof(line) - 1)
            break;   /* the child is mid-line; re-read it whole next time */
        lc->log_off += (long long)n;
        if (!idletoken_llama_log_fit_failed(line)) continue;

        const char *allow = getenv("IDLETOKEN_ALLOW_VRAM_OVERCOMMIT");
        if (allow && !strcmp(allow, "1")) {
            fprintf(stderr,
                    "coord: ==========================================================\n"
                    "coord: == IDLETOKEN_ALLOW_VRAM_OVERCOMMIT=1: the engine says it ==\n"
                    "coord: == could not fit this model into free device memory, and ==\n"
                    "coord: == we are starting anyway. On Windows the driver pages   ==\n"
                    "coord: == VRAM to system memory instead of failing, which can   ==\n"
                    "coord: == freeze the whole machine. Measurement only.           ==\n"
                    "coord: ==========================================================\n");
            lc->log_off = (long long)ftell(f);   /* consume the rest, stay quiet */
            break;
        }
        snprintf(lc->fatal, sizeof(lc->fatal),
                 "the inference engine could not fit this model into free "
                 "device memory and started anyway (its log: \"%s\"). Serving "
                 "in that state makes the GPU driver page video memory out to "
                 "system memory — on Windows that can freeze the whole machine "
                 "rather than fail. What to do: ask for a smaller --ctx-size, "
                 "pick a smaller quantization of this model, close other GPU "
                 "users, or add a machine to the cluster. Set "
                 "IDLETOKEN_ALLOW_VRAM_OVERCOMMIT=1 to start anyway (for "
                 "measurement — it is not a supported way to run)",
                 line);
        /* Trim the copied engine line to one line's worth of noise. */
        for (char *p = lc->fatal; *p; p++) if (*p == '\n' || *p == '\r') *p = ' ';
        fprintf(stderr, "coord: llama-sidecar: %s\n", lc->fatal);
        break;
    }
    fclose(f);
}

/* Terminate the current child from INSIDE the monitor thread (shutdown() does
 * the same from outside, after joining this thread — the two must not both
 * run, which is why this one is static and leaves `stop` alone). */
static void llama_kill_child(idletoken_llama *lc) {
    if (lc->pid <= 0) return;
#ifdef _WIN32
    if (lc->child_handle) {
        TerminateProcess(lc->child_handle, 1);
        WaitForSingleObject(lc->child_handle, 2000);
        CloseHandle(lc->child_handle);
        lc->child_handle = NULL;
    }
#else
    kill((pid_t)lc->pid, SIGTERM);
    for (int i = 0; i < 20; i++) {
        if (waitpid((pid_t)lc->pid, NULL, WNOHANG) == (pid_t)lc->pid) {
            lc->pid = 0;
            break;
        }
        llama_sleep_ms(100);
    }
    if (lc->pid > 0) {
        kill((pid_t)lc->pid, SIGKILL);
        waitpid((pid_t)lc->pid, NULL, 0);
    }
#endif
    lc->pid = 0;
}

static void *llama_monitor(void *arg) {
    idletoken_llama *lc = (idletoken_llama *)arg;
    for (;;) {
        pthread_mutex_lock(&lc->mu);
        if (lc->stop) { pthread_mutex_unlock(&lc->mu); return NULL; }

        long long now = llama_now_ms();

        /* 0. What did the engine say about itself? A child that started but
         * reported a condition we must not serve gets stopped here, and the
         * state latches FAILED so nothing restarts it into the same wall.
         * Runs before the reap so the kill below is not read as a crash. */
        if (lc->pid > 0 && !lc->fatal[0]) llama_scan_log(lc);
        if (lc->fatal[0] && lc->state != IDLETOKEN_LLAMA_FAILED) {
            llama_kill_child(lc);
            lc->state = IDLETOKEN_LLAMA_FAILED;
            snprintf(lc->fail, sizeof(lc->fail), "%s", lc->fatal);
            pthread_mutex_unlock(&lc->mu);
            llama_sleep_ms(LLAMA_POLL_MS);
            continue;
        }

        /* 1. Reap. A dead child moves STARTING/READY -> RESTARTING or FAILED. */
        if (lc->pid > 0) {
#ifdef _WIN32
            DWORD code = STILL_ACTIVE;
            int dead = lc->child_handle &&
                       WaitForSingleObject(lc->child_handle, 0) == WAIT_OBJECT_0;
            char how[48];
            if (dead) {
                GetExitCodeProcess(lc->child_handle, &code);
                snprintf(how, sizeof(how), "exit code %lu", (unsigned long)code);
                CloseHandle(lc->child_handle);
                lc->child_handle = NULL;
            }
#else
            int st = 0;
            pid_t r = waitpid((pid_t)lc->pid, &st, WNOHANG);
            int dead = r == (pid_t)lc->pid;
            char how[48];
            if (dead) {
                /* Decode the wait status for the humans reading the log:
                 * "status 256" and "status 9" are the same fact spelled two
                 * unhelpful ways. */
                if (WIFSIGNALED(st))
                    snprintf(how, sizeof(how), "killed by signal %d", WTERMSIG(st));
                else
                    snprintf(how, sizeof(how), "exit code %d", WEXITSTATUS(st));
            }
#endif
            if (dead) {
                long long ready_uptime =
                    (lc->state == IDLETOKEN_LLAMA_READY && lc->became_ready_ms > 0)
                        ? now - lc->became_ready_ms : 0;
                lc->pid = 0;
                /* A long stable run forgives earlier quick crashes
                 * (engine.rs: uptime >= STABLE_UPTIME_MS resets the counter). */
                if (ready_uptime >= LLAMA_STABLE_UPTIME_MS) lc->quick_restarts = 0;
                if (lc->quick_restarts >= LLAMA_MAX_QUICK_RESTARTS) {
                    lc->state = IDLETOKEN_LLAMA_FAILED;
                    snprintf(lc->fail, sizeof(lc->fail),
                             "idletoken-server kept crashing (%d quick restarts, last: %s); "
                             "a restart will not help — check the engine log",
                             lc->quick_restarts, how);
                    fprintf(stderr, "coord: llama-sidecar: %s (%s)\n",
                            lc->fail, lc->log_path);
                } else {
                    lc->quick_restarts++;
                    long long delay_s = 1LL << (lc->quick_restarts < 5 ? lc->quick_restarts : 5);
                    if (delay_s > 30) delay_s = 30;   /* 2, 4, 8, 16, 30 — as engine.rs */
                    lc->restart_due_ms = now + delay_s * 1000;
                    lc->state = IDLETOKEN_LLAMA_RESTARTING;
                    fprintf(stderr,
                            "coord: llama-sidecar: idletoken-server exited (%s); "
                            "restarting in %llds (attempt %d/%d)\n",
                            how, delay_s, lc->quick_restarts, LLAMA_MAX_QUICK_RESTARTS);
                }
            }
        }

        /* 2. Respawn once the backoff expires. */
        if (lc->state == IDLETOKEN_LLAMA_RESTARTING && now >= lc->restart_due_ms) {
            if (llama_spawn(lc) == 0) {
                lc->restarts_total++;
                lc->state = IDLETOKEN_LLAMA_STARTING;
                fprintf(stderr, "coord: llama-sidecar: respawned idletoken-server "
                                "(pid %lld), waiting for /health\n", lc->pid);
            } else {
                /* The binary itself is gone/broken now — retrying cannot fix
                 * that, and it is a different failure than a crashing model. */
                lc->state = IDLETOKEN_LLAMA_FAILED;
                fprintf(stderr, "coord: llama-sidecar: respawn failed: %s\n", lc->fail);
            }
        }

        /* 3. Readiness probe while STARTING. Probing holds the mutex, which is
         * fine: /health on loopback answers in microseconds even mid-load. */
        if (lc->state == IDLETOKEN_LLAMA_STARTING && lc->pid > 0) {
            if (llama_health_ok(lc->endpoint)) {
                lc->state = IDLETOKEN_LLAMA_READY;
                lc->became_ready_ms = llama_now_ms();
                fprintf(stderr, "coord: llama-sidecar: idletoken-server ready on "
                                "%s (%.1fs to load)\n",
                        lc->endpoint,
                        (double)(lc->became_ready_ms - lc->spawned_ms) / 1000.0);
            }
        }

        pthread_mutex_unlock(&lc->mu);
        llama_sleep_ms(LLAMA_POLL_MS);
    }
}

/* --- public lifecycle ----------------------------------------------------- */

idletoken_llama *idletoken_llama_start(const char *bin, const char *gguf,
                                       int port, const char *engine_sock,
                                       uint32_t ctx_size, int n_parallel,
                                       const char *cluster_args,
                                       const char *log_path, int shared,
                                       char *err, size_t err_cap) {
    idletoken_llama *lc = calloc(1, sizeof(*lc));
    if (!lc) {
        if (err_cap) snprintf(err, err_cap, "out of memory");
        return NULL;
    }
    snprintf(lc->bin, sizeof(lc->bin), "%s", bin);
    snprintf(lc->gguf, sizeof(lc->gguf), "%s", gguf);
    if (cluster_args)
        snprintf(lc->cluster_args, sizeof(lc->cluster_args), "%s", cluster_args);
    snprintf(lc->log_path, sizeof(lc->log_path), "%s", log_path);
    lc->port = port;
    lc->ctx_size = ctx_size;
    lc->n_parallel = n_parallel > 0 ? n_parallel : 1;
    lc->shared = shared;
    /* The endpoint is decided ONCE, here, and every later connect reads it —
     * so there is no code path that can quietly reach the engine over TCP
     * after we told the user the link is a socket. A .sock suffix is what the
     * engine keys on, so a caller-supplied path without one would silently get
     * a TCP listener on a nonsense host: caught here instead. */
    if (engine_sock && engine_sock[0]) {
        size_t n = strlen(engine_sock);
        if (n < 6 || strcmp(engine_sock + n - 5, ".sock") != 0) {
            if (err_cap)
                snprintf(err, err_cap,
                         "engine socket path must end in .sock (the engine "
                         "switches address family on that suffix): %s",
                         engine_sock);
            free(lc);
            return NULL;
        }
        if (n >= sizeof(lc->sock_path)) {
            if (err_cap)
                snprintf(err, err_cap, "engine socket path is too long: %s",
                         engine_sock);
            free(lc);
            return NULL;
        }
        snprintf(lc->sock_path, sizeof(lc->sock_path), "%s", engine_sock);
        snprintf(lc->endpoint, sizeof(lc->endpoint), "unix:%s", lc->sock_path);
    } else {
        snprintf(lc->endpoint, sizeof(lc->endpoint), "127.0.0.1:%d", port);
    }
    /* IDLETOKEN_LLAMA_ARGS is appended LAST at spawn, so whatever it contains
     * wins over our own flags — including -v and --log-prompts-dir, which turn
     * the engine into a prompt logger. That is fine for your own machine and
     * unacceptable when the prompts belong to someone else, so shared mode
     * drops the variable entirely (allow-list of extra args = empty set: a
     * deny-list would leak the next time upstream adds a logging flag).
     * The refusal is PRINTED: "no output" cannot be told apart from "never
     * read the variable" — see G-SHARED-2 in docs/shared-mode-plan-2026-08.md. */
    const char *extra = getenv("IDLETOKEN_LLAMA_ARGS");
    if (extra && extra[0] && lc->shared) {
        fprintf(stderr, "coord: shared mode: IDLETOKEN_LLAMA_ARGS ignored "
                        "(engine args are locked while serving others)\n");
    } else if (extra) {
        /* ...but NOT the flags that decide where tensors live. Because the
         * variable is appended last and llama.cpp's --device handler ASSIGNS
         * rather than appends (common/arg.cpp: `params.devices =
         * parse_device_list(value)`, and repeated args are documented
         * last-wins), `IDLETOKEN_LLAMA_ARGS="--device RPC0"` replaced the
         * coordinator's coordinator-first device list wholesale and moved
         * layer 0 — with the embedding lookup — onto a remote node. That is
         * hard invariant #10, the one measured at 17/17 prompt recovery when
         * violated, and it failed SILENTLY: the "G-PRIV-7 precondition holds"
         * line is printed before this string is appended, so the log asserted
         * a safety the argv then removed.
         *
         * This is a deny-list, and the shared-mode comment above is right that
         * deny-lists leak when upstream adds a flag. The reason it is
         * acceptable here and not there: that set is open-ended (any future
         * logging flag), this one is closed and small (the flags that place
         * tensors), and the engine version is PINNED — a new placement flag can
         * only arrive with a deliberate version bump, which has its own
         * acceptance. Recheck this list when the pin moves.
         *
         * Refuse rather than silently drop: a user who set the variable is
         * trying to do something, and a dropped flag would look like the engine
         * ignoring them for no reason. */
        const char *hit = idletoken_llama_placement_flag(extra);
        if (hit) {
            /* The caller's buffer is small (256 B at the only call site) and
             * snprintf truncates silently, so the WHY goes to stderr, which has
             * no cap, and `err` carries only the short actionable sentence.
             * The first version of this message was ~380 B and reached the user
             * cut off mid-word. */
            fprintf(stderr,
                    "coord: refusing IDLETOKEN_LLAMA_ARGS: it sets %s, which "
                    "decides where tensors live.\n"
                    "coord: layer 0 and the token embedding must stay on this "
                    "machine — with layer 0 remote, prompts were recovered from "
                    "the public GGUF 17 times out of 17 (privacy invariant "
                    "#10).\n"
                    "coord: --rpc/--device/--tensor-split are computed from the "
                    "scheduler plan; IDLETOKEN_LLAMA_ARGS is appended after "
                    "them, so setting %s here would silently win.\n", hit, hit);
            if (err && err_cap)
                snprintf(err, err_cap,
                         "IDLETOKEN_LLAMA_ARGS sets %s, which would move layer 0 "
                         "off this machine (privacy invariant #10). Remove it: "
                         "placement is computed from the scheduler plan.", hit);
            free(lc);
            return NULL;
        }
        snprintf(lc->extra_args, sizeof(lc->extra_args), "%s", extra);
    }
    if (lc->shared) {
        fprintf(stderr, "coord: shared mode: engine args locked, "
                        "prompt logging off\n");
        /* Say which transport, in the log the acceptance script greps. Shared
         * mode without a socket is a real (if unlikely) configuration — say so
         * plainly rather than let the reader assume the stronger one. */
        if (lc->sock_path[0])
            fprintf(stderr, "coord: shared mode: engine link is a unix socket "
                            "(%s), not loopback TCP\n", lc->sock_path);
        else
            fprintf(stderr, "coord: shared mode: WARNING engine link is loopback "
                            "TCP — prompts are readable to anyone who can "
                            "capture loopback on this machine\n");
    }
    pthread_mutex_init(&lc->mu, NULL);

#ifdef _WIN32
    lc->job = CreateJobObjectA(NULL, NULL);
    if (lc->job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION ji;
        memset(&ji, 0, sizeof(ji));
        ji.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(lc->job, JobObjectExtendedLimitInformation,
                                     &ji, sizeof(ji))) {
            CloseHandle(lc->job);
            lc->job = NULL;
        }
    }
#endif
    if (llama_spawn(lc) != 0) {
        if (err_cap) snprintf(err, err_cap, "%s", lc->fail);
#ifdef _WIN32
        if (lc->job) CloseHandle(lc->job);
#endif
        pthread_mutex_destroy(&lc->mu);
        free(lc);
        return NULL;
    }
    lc->state = IDLETOKEN_LLAMA_STARTING;
    fprintf(stderr, "coord: llama-sidecar: spawned %s (pid %lld) on %s, "
                    "%d slot(s) of %u tokens each (-c %llu -np %d), "
                    "engine log: %s\n", lc->bin, lc->pid, lc->endpoint,
            lc->n_parallel, lc->ctx_size,
            (unsigned long long)lc->ctx_size * (unsigned long long)lc->n_parallel,
            lc->n_parallel, lc->log_path);
    if (pthread_create(&lc->tid, NULL, llama_monitor, lc) != 0) {
        if (err_cap) snprintf(err, err_cap, "pthread_create: %s", strerror(errno));
#ifdef _WIN32
        TerminateProcess(lc->child_handle, 1);
        WaitForSingleObject(lc->child_handle, INFINITE);
        CloseHandle(lc->child_handle);
        if (lc->job) CloseHandle(lc->job);
#else
        kill((pid_t)lc->pid, SIGKILL);
        waitpid((pid_t)lc->pid, NULL, 0);
#endif
        pthread_mutex_destroy(&lc->mu);
        free(lc);
        return NULL;
    }
    lc->thread_up = 1;
    return lc;
}

void idletoken_llama_shutdown(idletoken_llama *lc) {
    if (!lc) return;
    pthread_mutex_lock(&lc->mu);
    lc->stop = 1;
    long long pid = lc->pid;
#ifdef _WIN32
    HANDLE child = lc->child_handle;
#endif
    pthread_mutex_unlock(&lc->mu);
    if (lc->thread_up) pthread_join(lc->tid, NULL);
    if (pid > 0) {
#ifdef _WIN32
        if (child) {
            TerminateProcess(child, 0);
            if (WaitForSingleObject(child, 2000) == WAIT_TIMEOUT)
                TerminateProcess(child, 1);
            WaitForSingleObject(child, INFINITE);
            CloseHandle(child);
            lc->child_handle = NULL;
        }
#else
        kill((pid_t)pid, SIGTERM);
        /* give it a moment to exit cleanly, then make sure */
        for (int i = 0; i < 20; i++) {
            if (waitpid((pid_t)pid, NULL, WNOHANG) == (pid_t)pid) { pid = 0; break; }
            llama_sleep_ms(100);
        }
        if (pid > 0) {
            kill((pid_t)pid, SIGKILL);
            waitpid((pid_t)pid, NULL, 0);
        }
#endif
    }
#ifdef _WIN32
    if (lc->job) CloseHandle(lc->job);
#endif
    /* Take the socket file with us. Leaving it behind is not a security hole
     * (nothing listens on it once the engine is gone) but it does leave a
     * dead entry in the user's state directory, and the next run would have to
     * decide whether it was stale or live. */
    if (lc->sock_path[0]) {
#ifdef _WIN32
        DeleteFileA(lc->sock_path);
#else
        unlink(lc->sock_path);
#endif
    }
    pthread_mutex_destroy(&lc->mu);
    free(lc);
}

idletoken_llama_state idletoken_llama_get_state(idletoken_llama *lc) {
    if (!lc) return IDLETOKEN_LLAMA_OFF;
    pthread_mutex_lock(&lc->mu);
    idletoken_llama_state st = lc->state;
    pthread_mutex_unlock(&lc->mu);
    return st;
}

int idletoken_llama_restart_count(idletoken_llama *lc) {
    if (!lc) return 0;
    pthread_mutex_lock(&lc->mu);
    int n = lc->restarts_total;
    pthread_mutex_unlock(&lc->mu);
    return n;
}

const char *idletoken_llama_endpoint_of(idletoken_llama *lc) {
    /* No lock: set once before the monitor thread exists, never written after. */
    return lc ? lc->endpoint : "";
}

void idletoken_llama_fail_reason(idletoken_llama *lc, char *out, size_t cap) {
    if (!cap) return;
    out[0] = '\0';
    if (!lc) return;
    pthread_mutex_lock(&lc->mu);
    snprintf(out, cap, "%s", lc->fail);
    pthread_mutex_unlock(&lc->mu);
}

void idletoken_llama_fatal_reason(idletoken_llama *lc, char *out, size_t cap) {
    if (!cap) return;
    out[0] = '\0';
    if (!lc) return;
    pthread_mutex_lock(&lc->mu);
    snprintf(out, cap, "%s", lc->fatal);
    pthread_mutex_unlock(&lc->mu);
}
