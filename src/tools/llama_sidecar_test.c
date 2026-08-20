/* llama_sidecar_test.c — drive the engine sidecar directly, both transports.
 *
 * WHY A HARNESS AND NOT "start coord and read the log": on a machine the
 * scheduler refuses (this repo's Mac probes 0 GiB usable VRAM, so the privacy
 * invariant "layer 0 stays on the coordinator" stops it before any engine
 * exists) the coordinator never reaches this code, and the log stops earlier.
 * Judging the transport from that log yields a false negative — the same shape
 * of mistake as a grep pattern that cannot match.
 *
 * What it proves (P0-4, gate G-SHARED-1 in docs/shared-mode-plan-2026-08.md):
 *
 *   shared  — the engine answers a real completion over an AF_UNIX socket, owns
 *             no listening TCP socket at all, refuses GET /slots (which hands
 *             back every live prompt verbatim), and ignores LLAMA_ARG_* in the
 *             environment;
 *   local   — the same request works over loopback TCP, the listener IS there,
 *             /slots answers, and LLAMA_ARG_ALIAS takes effect.
 *
 * The local run is the POSITIVE CONTROL, every line of it. "No TCP listener",
 * "/slots refused" and "the alias did not apply" each prove nothing on their
 * own: a check that cannot go red is decoration, and this repo has shipped two
 * of those (see CLAUDE.md, "known traps: methodology"). The control is what
 * makes the shared results mean something.
 *
 * Usage:  llama-sidecar-test <engine-bin> <model.gguf> [marker]
 * Contract: prints exactly one SIDECAR_(OK|FAIL): line last.
 */
#include "idletoken_llama_sidecar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

static void sleep_ms(int ms) {
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* Does this pid hold a listening TCP socket? lsof is on every platform we
 * develop on and needs no privileges for one's own processes — unlike a packet
 * capture, which needs root and so could not run unattended here. */
static int has_tcp_listener(long long pid) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "lsof -nP -a -p %lld -iTCP -sTCP:LISTEN 2>/dev/null | grep -c LISTEN",
             pid);
    FILE *f = popen(cmd, "r");
    if (!f) return -1;
    char line[64] = "";
    if (!fgets(line, sizeof(line), f)) { pclose(f); return -1; }
    pclose(f);
    return atoi(line);
}

static int wait_ready(idletoken_llama *lc, int timeout_s) {
    for (int i = 0; i < timeout_s * 5; i++) {
        idletoken_llama_state st = idletoken_llama_get_state(lc);
        if (st == IDLETOKEN_LLAMA_READY) return 0;
        if (st == IDLETOKEN_LLAMA_FAILED) {
            char why[256] = "";
            idletoken_llama_fail_reason(lc, why, sizeof(why));
            fprintf(stderr, "  engine FAILED: %s\n", why);
            return -1;
        }
        sleep_ms(200);
    }
    return -1;
}

/* GET `path`; returns the HTTP status (-1 if unreachable) and, when `body_out`
 * is given, up to cap-1 bytes of the body. */
static int http_get(const char *endpoint, const char *path,
                    char *body_out, size_t cap) {
    idletoken_llama_conn c;
    if (body_out && cap) body_out[0] = '\0';
    if (idletoken_llama_http_open(endpoint, "GET", path, NULL, 0, 15000, &c) != 0)
        return -1;
    size_t n = 0;
    char *resp = idletoken_llama_http_read_all(&c, &n, 1u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    if (resp && body_out && cap) snprintf(body_out, cap, "%s", resp);
    free(resp);
    return status;
}

/* One real completion through the sidecar's own HTTP client, carrying `marker`
 * in the prompt. Returns 0 when the engine answered 200. */
static int completion_ok(const char *endpoint, const char *marker) {
    char body[512];
    snprintf(body, sizeof(body),
             "{\"prompt\":\"%s\",\"n_predict\":4,\"temperature\":0}", marker);
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(endpoint, "POST", "/completion",
                                  body, strlen(body), 120000, &c) != 0) {
        fprintf(stderr, "  cannot reach the engine at %s\n", endpoint);
        return -1;
    }
    size_t n = 0;
    char *resp = idletoken_llama_http_read_all(&c, &n, 1u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    free(resp);
    if (status != 200) { fprintf(stderr, "  engine HTTP %d\n", status); return -1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <engine-bin> <model.gguf> [marker]\n", argv[0]);
        return 2;
    }
    const char *bin = argv[1], *gguf = argv[2];
    /* The caller normally mints a per-run marker (see scripts/shared_mode_gate.sh):
     * a constant would also be a literal in this file, and the disk sweep that
     * follows would then match our own source. */
    const char *marker = argc > 3 && argv[3][0] ? argv[3] : "IdleTokenCanaryDefault";

    const char *home = getenv("HOME");
    if (!home || !home[0]) { printf("SIDECAR_FAIL: no HOME\n"); return 1; }
    char dir[400], sock[440], log_shared[440], log_local[440];
    snprintf(dir, sizeof(dir), "%s/.idletoken", home);
    mkdir(dir, 0700);
    snprintf(sock, sizeof(sock), "%s/engine-test.sock", dir);
    snprintf(log_shared, sizeof(log_shared), "%s/sidecar-test-shared.log", dir);
    snprintf(log_local, sizeof(log_local), "%s/sidecar-test-local.log", dir);

    /* Set for BOTH runs. llama.cpp applies the environment before argv
     * (common/arg.cpp), so these are exactly the levers a provider has once
     * our command line is locked. ALIAS is the observable one: it renames the
     * model in /v1/models and we never pass -a ourselves, so whether it took
     * effect isolates the environment path. */
    const char *ENV_ALIAS = "scrub-canary";
    setenv("LLAMA_ARG_ALIAS", ENV_ALIAS, 1);
    setenv("LLAMA_ARG_ENDPOINT_SLOTS", "1", 1);
    /* And the argv lever, G-SHARED-2. --metrics is the observable one that
     * does not collide with the alias above: it turns on an endpoint that is
     * off by default, and we set no environment variable for it, so /metrics
     * answering can only mean IDLETOKEN_LLAMA_ARGS was honoured. */
    setenv("IDLETOKEN_LLAMA_ARGS", "--metrics", 1);

    int failed = 0;

    /* --- shared: AF_UNIX, and no TCP listener anywhere in the engine ------ */
    printf("shared mode (engine socket %s)\n", sock);
    {
        char err[256] = "";
        idletoken_llama *lc = idletoken_llama_start(bin, gguf, 0, sock, 2048, 1,
                                                    NULL, log_shared, 1,
                                                    err, sizeof(err));
        if (!lc) { printf("SIDECAR_FAIL: shared start: %s\n", err); return 1; }
        printf("  endpoint: %s\n", idletoken_llama_endpoint_of(lc));
        if (wait_ready(lc, 120) != 0) {
            printf("SIDECAR_FAIL: engine never became ready over the socket "
                   "(log %s)\n", log_shared);
            idletoken_llama_shutdown(lc);
            return 1;
        }
        if (completion_ok(idletoken_llama_endpoint_of(lc), marker) != 0) {
            printf("SIDECAR_FAIL: no completion over the socket\n");
            idletoken_llama_shutdown(lc);
            return 1;
        }
        printf("  [ok] completion answered over AF_UNIX\n");

        /* GET /slots returns each live slot's prompt AND generated text. */
        int slots = http_get(idletoken_llama_endpoint_of(lc), "/slots", NULL, 0);
        if (slots == 200) {
            printf("  [BAD] /slots answered 200 — live prompts readable\n");
            failed = 1;
        } else {
            printf("  [ok] /slots refused (HTTP %d)\n", slots);
        }

        int metrics = http_get(idletoken_llama_endpoint_of(lc), "/metrics", NULL, 0);
        if (metrics == 200) {
            printf("  [BAD] IDLETOKEN_LLAMA_ARGS reached the engine "
                   "(/metrics is on)\n");
            failed = 1;
        } else {
            printf("  [ok] IDLETOKEN_LLAMA_ARGS did not reach the engine "
                   "(/metrics HTTP %d)\n", metrics);
        }

        char models[4096];
        http_get(idletoken_llama_endpoint_of(lc), "/v1/models", models, sizeof models);
        if (strstr(models, ENV_ALIAS)) {
            printf("  [BAD] LLAMA_ARG_ALIAS reached the engine — the "
                   "environment still configures it\n");
            failed = 1;
        } else {
            printf("  [ok] LLAMA_ARG_ALIAS did not reach the engine\n");
        }

        /* The engine's pid is not exported; find it by the socket it holds. */
        char cmd[600];
        snprintf(cmd, sizeof(cmd), "lsof -nPt %s 2>/dev/null | head -1", sock);
        FILE *f = popen(cmd, "r");
        long long pid = 0;
        if (f) { char l[64] = ""; if (fgets(l, sizeof(l), f)) pid = atoll(l); pclose(f); }
        if (pid <= 0) {
            printf("  [BAD] could not find the engine pid — the listener check "
                   "below would silently pass without it\n");
            failed = 1;
        } else {
            int n = has_tcp_listener(pid);
            if (n != 0) {
                printf("  [BAD] engine pid %lld holds %d listening TCP socket(s) "
                       "in shared mode\n", pid, n);
                failed = 1;
            } else {
                printf("  [ok] engine pid %lld has no listening TCP socket\n", pid);
            }
        }
        idletoken_llama_shutdown(lc);
        if (access(sock, F_OK) == 0) {
            printf("  [BAD] socket file survived shutdown: %s\n", sock);
            failed = 1;
        } else {
            printf("  [ok] socket file removed on shutdown\n");
        }
    }

    /* --- local: the control. Same check MUST come out the other way. ------ */
    printf("local mode (loopback TCP) — positive control\n");
    {
        char err[256] = "";
        idletoken_llama *lc = idletoken_llama_start(bin, gguf, 18711, "", 2048, 1,
                                                    NULL, log_local, 0,
                                                    err, sizeof(err));
        if (!lc) { printf("SIDECAR_FAIL: local start: %s\n", err); return 1; }
        printf("  endpoint: %s\n", idletoken_llama_endpoint_of(lc));
        if (wait_ready(lc, 120) != 0) {
            printf("SIDECAR_FAIL: engine never became ready over TCP (log %s)\n",
                   log_local);
            idletoken_llama_shutdown(lc);
            return 1;
        }
        if (completion_ok(idletoken_llama_endpoint_of(lc), marker) != 0) {
            printf("SIDECAR_FAIL: no completion over TCP\n");
            idletoken_llama_shutdown(lc);
            return 1;
        }
        printf("  [ok] completion answered over loopback TCP\n");

        int slots = http_get(idletoken_llama_endpoint_of(lc), "/slots", NULL, 0);
        if (slots != 200) {
            printf("  [BAD] control failed: /slots is not reachable in LOCAL "
                   "mode either (HTTP %d), so \"refused\" above says nothing\n",
                   slots);
            failed = 1;
        } else {
            printf("  [ok] /slots answers 200 here — the checker can see it "
                   "when it is on\n");
        }

        int metrics = http_get(idletoken_llama_endpoint_of(lc), "/metrics", NULL, 0);
        if (metrics != 200) {
            printf("  [BAD] control failed: IDLETOKEN_LLAMA_ARGS did not apply "
                   "in LOCAL mode either (/metrics HTTP %d), so \"ignored\" "
                   "above says nothing\n", metrics);
            failed = 1;
        } else {
            printf("  [ok] IDLETOKEN_LLAMA_ARGS applies here — the variable "
                   "really does reconfigure an unlocked engine\n");
        }

        char models[4096];
        http_get(idletoken_llama_endpoint_of(lc), "/v1/models", models, sizeof models);
        if (!strstr(models, ENV_ALIAS)) {
            printf("  [BAD] control failed: LLAMA_ARG_ALIAS did not apply in "
                   "LOCAL mode either, so the scrub above proves nothing\n");
            failed = 1;
        } else {
            printf("  [ok] LLAMA_ARG_ALIAS applies here — the environment does "
                   "configure an unscrubbed engine\n");
        }

        char cmd[128];
        snprintf(cmd, sizeof(cmd), "lsof -nPt -iTCP:18711 -sTCP:LISTEN 2>/dev/null | head -1");
        FILE *f = popen(cmd, "r");
        long long pid = 0;
        if (f) { char l[64] = ""; if (fgets(l, sizeof(l), f)) pid = atoll(l); pclose(f); }
        if (pid <= 0 || has_tcp_listener(pid) < 1) {
            printf("  [BAD] control failed: no listening TCP socket found in "
                   "LOCAL mode either, so the shared-mode result above proves "
                   "nothing about the checker\n");
            failed = 1;
        } else {
            printf("  [ok] engine pid %lld holds a listening TCP socket — the "
                   "checker can see one when there is one\n", pid);
        }
        idletoken_llama_shutdown(lc);
    }

    if (failed) { printf("SIDECAR_FAIL: see [BAD] lines above\n"); return 1; }
    printf("SIDECAR_OK\n");
    return 0;
}
