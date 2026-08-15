/* idletoken_llama_sidecar.h — supervise a local llama-server process
 * (v2 rebuild WS-B1, docs/v2-rebuild-plan-2026-08.md §4).
 *
 * The coordinator's "llamacpp single-machine mode" spawns one llama-server
 * on loopback and relays inference to it. This module owns the child's
 * lifecycle only: spawn, poll liveness + /health, restart with exponential
 * backoff, latch FAILED after repeated quick crashes. It knows nothing about
 * chat protocols — the relay lives in coord_main.c.
 *
 * The state machine is a C port of the client supervisor
 * (client/src-tauri/src/engine.rs): MAX_QUICK_RESTARTS = 5 consecutive quick
 * crashes latch FAILED ("restart won't help" — a bad GGUF path crashes
 * identically every time, and retrying forever buries the one line that says
 * why); a child that stayed up STABLE_UPTIME_MS resets the counter so a rare
 * crash after hours of serving does not inch toward giving up.
 *
 * Readiness is NOT "the port answers". llama-server binds its port while the
 * model is still loading and answers 503 there — `curl -s` treats that as
 * success, which invalidated three measurement runs before it was understood.
 * READY therefore requires GET /health to return 200 with the exact body
 * {"status":"ok"}.
 *
 * The child is always started with `--host 127.0.0.1`: the coordinator's own
 * api_token gate must remain the only gate, so llama-server must never be
 * reachable from the LAN.
 *
 * Windows uses CreateProcess plus a kill-on-close Job object; POSIX uses
 * fork/exec and parent-death handling where the platform provides it.
 *
 * C only. No C++. */
#ifndef IDLETOKEN_LLAMA_SIDECAR_H
#define IDLETOKEN_LLAMA_SIDECAR_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>   /* ssize_t */

typedef enum {
    IDLETOKEN_LLAMA_OFF = 0,
    IDLETOKEN_LLAMA_STARTING,    /* child alive, /health not {"status":"ok"} yet */
    IDLETOKEN_LLAMA_READY,       /* serving */
    IDLETOKEN_LLAMA_RESTARTING,  /* child exited; respawn scheduled (backoff) */
    IDLETOKEN_LLAMA_FAILED,      /* latched: kept crashing, restart won't help */
} idletoken_llama_state;

typedef struct idletoken_llama idletoken_llama;   /* opaque; one per sidecar */

/* Spawn llama-server (`bin -m gguf --host 127.0.0.1 --port port -ngl 99 ...`)
 * and start the monitor thread. `cluster_args` (NULL/"" for single-machine)
 * is a whitespace-split argument string appended by the coordinator's cluster
 * path (v2 WS-C: `--rpc ... --device ... --tensor-split ...`); its internal
 * order is preserved, which matters — llama.cpp arg handlers run in argv
 * order and `--device RPC0` only resolves AFTER `--rpc` registered the
 * server. Extra user arguments come from the environment variable
 * IDLETOKEN_LLAMA_ARGS (whitespace-split, appended last so they win over ours
 * on duplicate flags). ctx_size > 0 adds `-c ctx_size` so the engine's
 * context matches what the coordinator reports. The child's stdout+stderr are
 * redirected to `log_path` (its own file, NOT the coordinator's stderr:
 * engine logs are not ours, and keeping them separate is part of the
 * no-prompt-in-coord-logs invariant).
 *
 * Returns the handle, or NULL with a message in `err` when the FIRST spawn
 * fails (bad binary path etc.) — a coordinator that cannot start its engine
 * once must fail loudly at startup, not serve 503 forever. */
idletoken_llama *idletoken_llama_start(const char *bin, const char *gguf,
                                       int port, uint32_t ctx_size,
                                       const char *cluster_args,
                                       const char *log_path,
                                       char *err, size_t err_cap);

/* Stop the monitor thread and terminate the child (SIGTERM, then SIGKILL).
 * Frees the handle. */
void idletoken_llama_shutdown(idletoken_llama *lc);

/* Thread-safe snapshots for the HTTP surface (/idletoken/v1/stats, chat 503s). */
idletoken_llama_state idletoken_llama_get_state(idletoken_llama *lc);
int  idletoken_llama_restart_count(idletoken_llama *lc);   /* respawns performed */
int  idletoken_llama_port_of(idletoken_llama *lc);
/* Why the sidecar is FAILED (empty string otherwise). */
void idletoken_llama_fail_reason(idletoken_llama *lc, char *out, size_t cap);
const char *idletoken_llama_state_name(idletoken_llama_state st);

/* --- Minimal blocking HTTP/1.1 client for the sidecar (loopback only) ------
 *
 * The repo's other hand-rolled clients (privacy_proxy.c, platform_agent.c)
 * read to EOF and split at the header terminator — enough for our own server,
 * which never chunks. llama-server (cpp-httplib) streams SSE with
 * `Transfer-Encoding: chunked`, so this client de-chunks and hands back
 * payload bytes incrementally; the SSE relay in coord_main.c consumes them
 * line by line while the model is still generating. */
typedef struct {
    int       fd;
    int       status;         /* HTTP status of the response */
    int       chunked;        /* Transfer-Encoding: chunked */
    long long content_left;   /* Content-Length remaining; -1 = read to EOF */
    long long chunk_left;     /* bytes left in the current chunk */
    int       eof;
    /* raw bytes read past the header terminator, drained before recv() */
    char      buf[8192];
    size_t    blen, boff;
} idletoken_llama_conn;

/* Connect to 127.0.0.1:port, send one request, read the response head.
 * `timeout_ms` > 0 sets a socket send/recv timeout (used by the health probe;
 * pass 0 for inference, where minutes of silence are legitimate).
 * Returns 0 (head parsed, c->status set) or -1. */
int idletoken_llama_http_open(int port, const char *method, const char *path,
                              const char *body, size_t body_len,
                              int timeout_ms, idletoken_llama_conn *c);

/* Next de-chunked payload bytes. >0 = bytes written to dst, 0 = end of body,
 * -1 = transport error. */
ssize_t idletoken_llama_http_read(idletoken_llama_conn *c, void *dst, size_t cap);

/* Read the whole body (caps at max_bytes). Returns a malloc'd NUL-terminated
 * buffer (caller frees) and sets *out_len, or NULL on error. */
char *idletoken_llama_http_read_all(idletoken_llama_conn *c, size_t *out_len,
                                    size_t max_bytes);

void idletoken_llama_http_close(idletoken_llama_conn *c);

#endif /* IDLETOKEN_LLAMA_SIDECAR_H */
