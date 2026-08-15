/* llama_sidecar.c — spawn + supervise a local llama-server (WS-B1).
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
 *    child after fork. llama-server's default log level prints request
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
#define LLAMA_ARGV_MAX 64

struct idletoken_llama {
    /* config (immutable after start) */
    char bin[512];
    char gguf[1024];
    char log_path[512];
    char cluster_args[1024];  /* WS-C cluster flags (--rpc/--device/--tensor-split) */
    char extra_args[1024];    /* IDLETOKEN_LLAMA_ARGS copy, split at spawn */
    int  port;
    uint32_t ctx_size;

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
    char fail[256];           /* FAILED: the reason, for the API surface */
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

int idletoken_llama_http_open(int port, const char *method, const char *path,
                              const char *body, size_t body_len,
                              int timeout_ms, idletoken_llama_conn *c) {
    memset(c, 0, sizeof(*c));
    c->fd = -1;
    c->content_left = -1;

    char addr[32];
    snprintf(addr, sizeof(addr), "127.0.0.1:%d", port);
    int fd = idletoken_connect_tcp(addr);
    if (fd < 0) return -1;
    conn_set_timeout(fd, timeout_ms);

    char head[512];
    int hn = snprintf(head, sizeof(head),
                      "%s %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n",
                      method, path, addr, body_len);
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

/* One raw payload byte (post-header stream): buffered leftover first, then the
 * socket. Returns 0..255, or -1 on EOF/error. */
static int conn_byte(idletoken_llama_conn *c) {
    if (c->boff >= c->blen) {
        ssize_t r = conn_recv(c->fd, c->buf, sizeof(c->buf));
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
    return conn_recv(c->fd, dst, cap);
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
 * llama-server binds its port while the model is still loading and answers
 * 503 {"error":{"message":"Loading model",...}} there — a status-only or
 * connect-only check reports ready long before the model exists (this exact
 * mistake invalidated three measurement runs; see the header comment). */
static int llama_health_ok(int port) {
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(port, "GET", "/health", NULL, 0,
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
                    "coord: == GGML_RPC_ALLOW_PLAINTEXT=1 is set: llama-server may  ==\n"
                    "coord: == move tensor traffic WITHOUT TLS. Testing only —      ==\n"
                    "coord: == never run a real cluster this way.                   ==\n"
                    "coord: ==========================================================\n");
        }
    }
#ifdef _WIN32
    char portstr[16], ctxstr[16], cmd[4096];
    snprintf(portstr, sizeof(portstr), "%d", lc->port);
    snprintf(ctxstr, sizeof(ctxstr), "%u", lc->ctx_size);
    int n = snprintf(cmd, sizeof(cmd),
                     "\"%s\" -m \"%s\" --host 127.0.0.1 --port %s "
                     "-ngl 99 --reasoning off%s%s%s%s",
                     lc->bin, lc->gguf, portstr,
                     lc->ctx_size > 0 ? " -c " : "",
                     lc->ctx_size > 0 ? ctxstr : "",
                     lc->cluster_args[0] ? " " : "", lc->cluster_args);
    if (n < 0 || (size_t)n >= sizeof(cmd)) {
        snprintf(lc->fail, sizeof(lc->fail), "llama-server command line is too long");
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
                             CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    DWORD werr = ok ? 0 : GetLastError();
    CloseHandle(log);
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
    char portstr[16], ctxstr[16];
    snprintf(portstr, sizeof(portstr), "%d", lc->port);
    snprintf(ctxstr, sizeof(ctxstr), "%u", lc->ctx_size);

    char *argv[LLAMA_ARGV_MAX];
    int argc = 0;
    argv[argc++] = lc->bin;
    argv[argc++] = "-m";        argv[argc++] = lc->gguf;
    /* Loopback ALWAYS: the coordinator's api_token gate must remain the only
     * gate, so llama-server must never be reachable from the LAN. */
    argv[argc++] = "--host";    argv[argc++] = "127.0.0.1";
    argv[argc++] = "--port";    argv[argc++] = portstr;
    argv[argc++] = "-ngl";      argv[argc++] = "99";
    /* Reasoning off by default: thinking models (Qwen3.5 etc.) otherwise burn
     * the whole token budget inside <think> and the visible answer comes back
     * EMPTY with finish_reason "length" — measured with Qwen3.5-0.8B at
     * max_tokens 200. An empty reply is a broken product; thinking support is
     * a future client toggle. Overridable via IDLETOKEN_LLAMA_ARGS (appended
     * last, so a user-supplied --reasoning wins). */
    argv[argc++] = "--reasoning"; argv[argc++] = "off";
    if (lc->ctx_size > 0) { argv[argc++] = "-c"; argv[argc++] = ctxstr; }
    /* No extra log flags on purpose: the default level logs request metadata
     * but no prompt text. Never add -v or --log-prompts-dir here. */

    /* Cluster args first (their internal order is load-bearing: --rpc must
     * precede --device so RPC device names resolve), then the user's
     * IDLETOKEN_LLAMA_ARGS LAST so on duplicate flags llama-server takes the
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

/* --- monitor thread ------------------------------------------------------- */

static void *llama_monitor(void *arg) {
    idletoken_llama *lc = (idletoken_llama *)arg;
    for (;;) {
        pthread_mutex_lock(&lc->mu);
        if (lc->stop) { pthread_mutex_unlock(&lc->mu); return NULL; }

        long long now = llama_now_ms();

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
                             "llama-server kept crashing (%d quick restarts, last: %s); "
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
                            "coord: llama-sidecar: llama-server exited (%s); "
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
                fprintf(stderr, "coord: llama-sidecar: respawned llama-server "
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
            int port = lc->port;
            if (llama_health_ok(port)) {
                lc->state = IDLETOKEN_LLAMA_READY;
                lc->became_ready_ms = llama_now_ms();
                fprintf(stderr, "coord: llama-sidecar: llama-server ready on "
                                "127.0.0.1:%d (%.1fs to load)\n",
                        port, (double)(lc->became_ready_ms - lc->spawned_ms) / 1000.0);
            }
        }

        pthread_mutex_unlock(&lc->mu);
        llama_sleep_ms(LLAMA_POLL_MS);
    }
}

/* --- public lifecycle ----------------------------------------------------- */

idletoken_llama *idletoken_llama_start(const char *bin, const char *gguf,
                                       int port, uint32_t ctx_size,
                                       const char *cluster_args,
                                       const char *log_path,
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
    const char *extra = getenv("IDLETOKEN_LLAMA_ARGS");
    if (extra) snprintf(lc->extra_args, sizeof(lc->extra_args), "%s", extra);
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
    fprintf(stderr, "coord: llama-sidecar: spawned %s (pid %lld) on 127.0.0.1:%d, "
                    "engine log: %s\n", lc->bin, lc->pid, lc->port, lc->log_path);
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

int idletoken_llama_port_of(idletoken_llama *lc) {
    return lc ? lc->port : 0;
}

void idletoken_llama_fail_reason(idletoken_llama *lc, char *out, size_t cap) {
    if (!cap) return;
    out[0] = '\0';
    if (!lc) return;
    pthread_mutex_lock(&lc->mu);
    snprintf(out, cap, "%s", lc->fail);
    pthread_mutex_unlock(&lc->mu);
}
