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
 * The child never listens anywhere the LAN can reach: the coordinator's own
 * api_token gate must remain the only gate. Local use binds `--host 127.0.0.1`;
 * shared mode binds a Unix socket instead (see `engine_sock` below).
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
/* `shared` = this coordinator may serve OTHER PEOPLE's requests (platform
 * work). It hardens the engine against the machine's own owner reading what
 * buyers sent: IDLETOKEN_LLAMA_ARGS is ignored (it can inject -v /
 * --log-prompts-dir), and nothing that could carry prompt text is enabled.
 * See docs/threat-model-shared-compute-2026-08.md — the bar is "not readable
 * by ordinary means", not "safe against a debugger".
 * Local use passes 0 and is unaffected: reading your OWN prompts is normal.
 *
 * `engine_sock` (non-empty only with `shared`) is the AF_UNIX path the engine
 * binds INSTEAD of the TCP port, so the prompts crossing this link never
 * become loopback packets a capture can read. The caller owns the path and
 * must place it in a directory only this user can traverse — a Unix socket's
 * reachability is its directory's, and this module does not create it. A stale
 * file from a crashed run is removed before each spawn (AF_UNIX bind fails on
 * an existing path; SO_REUSEADDR does not help). Empty = plain TCP loopback.
 * There is deliberately no fall back from one to the other: if the socket
 * cannot be used, the engine must fail loudly rather than quietly serve the
 * same bytes over a readable transport. */
/* `n_parallel` (>=1) is how many independent sequences the engine should serve
 * at once — `-np N`, derived from memory by idletoken_llama_seq_slots().
 *
 * ⚠ `ctx_size` here is the context of ONE slot, and the engine's `-c` is the
 * TOTAL KV budget SHARED by all slots (llama-context.cpp: n_ctx_seq = n_ctx /
 * n_seq_max). So this module passes `-c ctx_size * n_parallel`. Passing `-np 4`
 * without touching `-c` would quarter every conversation's window instead of
 * adding capacity — a Claude Code session's system prompt alone is ~13K tokens,
 * so that single-line mistake turns a working machine into one that refuses
 * every real request. Slots are bought with MEMORY, never with context. */
idletoken_llama *idletoken_llama_start(const char *bin, const char *gguf,
                                       int port, const char *engine_sock,
                                       uint32_t ctx_size, int n_parallel,
                                       const char *cluster_args,
                                       const char *log_path, int shared,
                                       char *err, size_t err_cap);

/* Stop the monitor thread and terminate the child (SIGTERM, then SIGKILL).
 * Frees the handle. */
void idletoken_llama_shutdown(idletoken_llama *lc);

/* Thread-safe snapshots for the HTTP surface (/idletoken/v1/stats, chat 503s). */
idletoken_llama_state idletoken_llama_get_state(idletoken_llama *lc);
int  idletoken_llama_restart_count(idletoken_llama *lc);   /* respawns performed */
/* Where to reach the engine, in the form idletoken_llama_http_open takes:
 * "127.0.0.1:<port>" or "unix:<path>". Immutable after start, so the returned
 * pointer stays valid for the sidecar's lifetime. */
const char *idletoken_llama_endpoint_of(idletoken_llama *lc);
/* Why the sidecar is FAILED (empty string otherwise). */
void idletoken_llama_fail_reason(idletoken_llama *lc, char *out, size_t cap);
const char *idletoken_llama_state_name(idletoken_llama_state st);

/* Non-empty once the engine's own log reported a startup condition that no
 * restart and no waiting can fix, and that the coordinator must refuse over
 * instead of serving around. Today there is exactly one: the engine could not
 * fit the model into device memory (see idletoken_llama_log_fit_failed).
 *
 * Distinct from fail_reason on purpose. FAILED means "it kept crashing" — the
 * coordinator stays up and answers 503, because the user may fix the model
 * path and the next respawn may work. This means "it started, and what it
 * started is not something we may serve": the machine is about to page VRAM to
 * host memory, and on Windows that freezes the desktop rather than erroring. */
void idletoken_llama_fatal_reason(idletoken_llama *lc, char *out, size_t cap);

/* Does this slice of engine log say the engine could not fit the model into
 * free device memory? Exposed so it can be unit-tested against real log text
 * (coord --selftest) — a detector nothing can prove red is not a detector.
 *
 * The matched sentence is llama.cpp's, verified against the pinned engine
 * (scripts/llamacpp-patches/UPSTREAM 0a50d99, common/fit.cpp: LOG_WRN("%s:
 * failed to fit params to free device memory: %s\n", ...)). It is reached ONLY
 * when the projected allocation does not fit the free device memory AND the
 * engine cannot shrink its way out — with our `-ngl 99` and explicit `-c`, both
 * knobs it would have turned are pinned by us, so it warns and starts anyway.
 * A fit that succeeds returns from step 1 without logging anything, which is
 * why this is not simply true on every start.
 *
 * Honest about its own limits: it is a STRING match on an upstream log line.
 * If upstream rewords it, this goes quiet — silently. That is why it is the
 * second line of defence and not the first; the budget fix in
 * idletoken_llama_seq_slots is what actually keeps us out of this state, and
 * this catches the cases the budget's estimates get wrong. */
int idletoken_llama_log_fit_failed(const char *text);

/* Does this IDLETOKEN_LLAMA_ARGS string set a flag that decides WHERE tensors
 * live? Returns the offending flag (a static string) or NULL.
 *
 * The coordinator refuses to start when this hits: those flags are computed
 * from the scheduler plan and pin layer 0 + the token embedding to the local
 * machine (privacy invariant #10), IDLETOKEN_LLAMA_ARGS is appended AFTER them,
 * and llama.cpp takes the LAST value — so a user's value would silently win.
 *
 * Pure function of the string, like idletoken_llama_log_fit_failed above and
 * for the same reason: coord --selftest drives it with strings that must hit
 * AND strings that must not. A deny-list nobody has watched over-match is how
 * you find out later that it broke a legitimate flag.
 *
 * Matching is whole-token, accepts the `=` form, and NORMALISES UNDERSCORES for
 * `--` flags — upstream does `std::replace(arg.begin(), arg.end(), '_', '-')`
 * on every `--` argument (common/arg.cpp), so `--tensor_split` reaches the
 * engine as `--tensor-split`. A literal matcher misses it, which is exactly the
 * bypass measured on 2026-08-20. */
const char *idletoken_llama_placement_flag(const char *args);

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
    /* --- bounded waiting on the engine (2026-08-17) ---------------------
     * Set by idletoken_llama_http_watch(). `slice_ms` 0 keeps the old
     * behaviour (one unbounded recv). See that function for why this exists. */
    int          slice_ms;      /* per-recv ceiling, ms */
    const char  *watch;         /* endpoint to health-probe while silent */
    long long    silent_ms;     /* consecutive silence; any byte resets it */
    long long    probed_at_ms;  /* silence level at the last probe */
} idletoken_llama_conn;

/* Connect to `endpoint` ("host:port", or "unix:<path>" for an AF_UNIX socket —
 * idletoken_llama_endpoint_of hands you the right one), send one request, read
 * the response head. `timeout_ms` > 0 sets a socket send/recv timeout (used by
 * the health probe; pass 0 for inference, where minutes of silence are
 * legitimate). Returns 0 (head parsed, c->status set) or -1. */
int idletoken_llama_http_open(const char *endpoint, const char *method,
                              const char *path,
                              const char *body, size_t body_len,
                              int timeout_ms, idletoken_llama_conn *c);

/* Make reads on `c` wait BOUNDEDLY: recv in slices, and while the engine is
 * silent, check on a SEPARATE connection that it still answers GET /health.
 * `endpoint` is idletoken_llama_endpoint_of(); it must outlive `c`.
 *
 * WHY (2026-08-17, measured on a wedged coordinator — results/coord-wedge-20260817.md).
 * Inference opens with timeout_ms = 0 because minutes of silence during prefill
 * are legitimate. But "no ceiling" is not "a generous ceiling": when the engine
 * stopped serving (process still alive, listener gone), the executor thread
 * stayed parked in recv() on a connection that no longer existed. Nothing could
 * wake it — on Windows, closing a socket from another thread does not reliably
 * return a blocked recv. And because the coordinator executed ONE request at a
 * time back then, that one parked thread was not one lost request but the whole
 * coordinator, permanently, until restart: even GET /health went unanswered.
 * llamacpp mode has served requests from a thread pool since 2026-08-18, so a
 * parked reader now costs one slot rather than the machine — which is why this
 * check is per CONNECTION and must stay that way.
 *
 * The liveness check is the point, not the ceiling. "Is the child process
 * alive" would NOT have caught the observed failure — the process was alive.
 * Only "does it still answer" distinguishes a long prefill from a dead engine,
 * so a big model's prefill can stay silent for as long as it needs. */
void idletoken_llama_http_watch(idletoken_llama_conn *c, const char *endpoint);

/* Next de-chunked payload bytes. >0 = bytes written to dst, 0 = end of body,
 * -1 = transport error. */
ssize_t idletoken_llama_http_read(idletoken_llama_conn *c, void *dst, size_t cap);

/* Read the whole body (caps at max_bytes). Returns a malloc'd NUL-terminated
 * buffer (caller frees) and sets *out_len, or NULL on error. */
char *idletoken_llama_http_read_all(idletoken_llama_conn *c, size_t *out_len,
                                    size_t max_bytes);

void idletoken_llama_http_close(idletoken_llama_conn *c);

#endif /* IDLETOKEN_LLAMA_SIDECAR_H */
