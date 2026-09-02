/* Shipped worker-join contract regression.
 *
 * This is deliberately a call-site test, not a UI-state test. It reads the
 * Tauri source that constructs the real worker argv, confirms the packaged
 * sidecar is the same binary, then executes that binary's retired network path
 * and requires a refusal before any network operation. The in-memory mutation
 * controls prove the source oracle goes red when either guarantee is removed. */

#include "idletoken_resource.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks;
static int failures;

static void check(int ok, const char *what) {
    checks++;
    if (ok) {
        printf("[ok] %s\n", what);
    } else {
        printf("[FAIL] %s\n", what);
        failures++;
    }
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "worker-join-test: open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0 || n > 4 * 1024 * 1024 || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "worker-join-test: invalid source size for %s\n", path);
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

static int count_text(const char *haystack, const char *needle) {
    int n = 0;
    size_t step = strlen(needle);
    if (!step) return 0;
    for (const char *p = haystack; (p = strstr(p, needle)) != NULL; p += step)
        n++;
    return n;
}

static const char *find_before(const char *begin, const char *end,
                               const char *needle) {
    const char *p = strstr(begin, needle);
    return p && p < end ? p : NULL;
}

static int validate_pairing_source(const char *src, char *why, size_t why_cap) {
    static const char fn_mark[] = "fn materialize_engine(app: &AppHandle)";
    static const char vec_mark[] = "let mut worker_args = vec![";
    static const char flag_mark[] = "\"--rpc-supervisor\".into(),";
    static const char engine_mark[] = "\"--engine-dir\".into(),";
    static const char call_mark[] =
        "crate::engine::start_engine(app, \"worker\".into(), worker_args, worker_env)";
    const char *fn = strstr(src, fn_mark);
    if (!fn || count_text(src, fn_mark) != 1) {
        snprintf(why, why_cap, "materialize_engine is missing or ambiguous");
        return 0;
    }
    const char *fn_end = strstr(fn, "\n/// Blocking HTTP GET");
    if (!fn_end) {
        snprintf(why, why_cap, "cannot bound materialize_engine");
        return 0;
    }
    const char *vec = find_before(fn, fn_end, vec_mark);
    if (!vec) {
        snprintf(why, why_cap, "worker argv vector is missing");
        return 0;
    }
    const char *vec_body = vec + strlen(vec_mark);
    const char *vec_end = find_before(vec_body, fn_end, "\n        ];");
    if (!vec_end) {
        snprintf(why, why_cap, "worker argv vector is not bounded");
        return 0;
    }
    const char *flag = find_before(vec_body, vec_end, flag_mark);
    const char *engine = find_before(vec_body, vec_end, engine_mark);
    const char *first_quote = (const char *)memchr(vec_body, '"', (size_t)(vec_end - vec_body));
    if (!flag || flag != first_quote || !engine || flag >= engine) {
        snprintf(why, why_cap,
                 "--rpc-supervisor is not the first shipped worker argument");
        return 0;
    }
    const char *call = find_before(vec_end, fn_end, call_mark);
    if (!call || count_text(fn, call_mark) != 1) {
        snprintf(why, why_cap, "worker argv is not passed to the worker sidecar exactly once");
        return 0;
    }
    static const char *forbidden[] = {
        "worker_args =", "worker_args.clear(", "worker_args.remove(",
        "worker_args.retain(", "worker_args.drain(", "worker_args.pop(",
        "worker_args.truncate(", "worker_args.swap_remove("
    };
    for (size_t i = 0; i < sizeof(forbidden) / sizeof(forbidden[0]); i++) {
        if (find_before(vec_end, call, forbidden[i])) {
            snprintf(why, why_cap, "worker argv is rewritten before sidecar launch");
            return 0;
        }
    }
    why[0] = '\0';
    return 1;
}

static int validate_bundle_config(const char *src, char *why, size_t why_cap) {
    static const char external[] = "\"externalBin\": [";
    static const char worker[] = "\"binaries/idletoken-worker\"";
    const char *list = strstr(src, external);
    if (!list || count_text(src, external) != 1) {
        snprintf(why, why_cap, "externalBin is missing or ambiguous");
        return 0;
    }
    const char *end = strchr(list, ']');
    const char *entry = end ? find_before(list, end, worker) : NULL;
    if (!entry || count_text(src, worker) != 1) {
        snprintf(why, why_cap, "the worker sidecar is not packaged exactly once");
        return 0;
    }
    why[0] = '\0';
    return 1;
}

static int run_retired_path(const char *worker, char *out, size_t out_cap,
                            int *exit_out) {
    int fds[2];
    if (pipe(fds) != 0) return 0;
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]); close(fds[1]);
        return 0;
    }
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        if (fds[1] > STDERR_FILENO) close(fds[1]);
        execl(worker, worker, "--coordinator", "127.0.0.1:1", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    size_t used = 0;
    while (used + 1 < out_cap) {
        ssize_t n = read(fds[0], out + used, out_cap - used - 1);
        if (n > 0) used += (size_t)n;
        else if (n < 0 && errno == EINTR) continue;
        else break;
    }
    out[used] = '\0';
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 0;
    *exit_out = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s PAIRING_RS TAURI_CONF WORKER_BIN\n", argv[0]);
        return 2;
    }
    size_t pairing_len = 0, config_len = 0;
    char *pairing = read_file(argv[1], &pairing_len);
    char *config = read_file(argv[2], &config_len);
    if (!pairing || !config) {
        free(pairing); free(config);
        return 2;
    }

    char why[192] = "";
    int pairing_ok = validate_pairing_source(pairing, why, sizeof(why));
    check(pairing_ok, pairing_ok
          ? "shipped materialize_engine passes --rpc-supervisor first to the worker sidecar"
          : why);
    int bundle_ok = validate_bundle_config(config, why, sizeof(why));
    check(bundle_ok, bundle_ok
          ? "the Tauri externalBin bundle contains that worker sidecar"
          : why);

    char *mut = (char *)malloc(pairing_len + 1);
    if (mut) memcpy(mut, pairing, pairing_len + 1);
    char *flag = mut ? strstr(mut, "\"--rpc-supervisor\".into(),") : NULL;
    if (flag) flag[2] = '_';
    int mutated_pairing_red = flag && !validate_pairing_source(mut, why, sizeof(why));
    check(mutated_pairing_red,
          "control: removing the shipped --rpc-supervisor argument makes the oracle red");
    free(mut);

    mut = (char *)malloc(config_len + 1);
    if (mut) memcpy(mut, config, config_len + 1);
    char *entry = mut ? strstr(mut, "\"binaries/idletoken-worker\"") : NULL;
    if (entry) entry[1] = '_';
    int mutated_bundle_red = entry && !validate_bundle_config(mut, why, sizeof(why));
    check(mutated_bundle_red,
          "control: removing the bundled worker entry makes the oracle red");
    free(mut);

    char output[8192];
    int exit_code = -1;
    int ran = run_retired_path(argv[3], output, sizeof(output), &exit_code);
    check(ran, "the built worker legacy-path probe executed");
    if (ran) {
        check(exit_code == IDLETOKEN_EXIT_JOIN_REFUSED,
              "a non-supervisor network launch exits with the refusal code");
        check(count_text(output, IDLETOKEN_JOIN_REFUSED_MARK) == 1 &&
              strstr(output, "legacy INFER join mode is retired; use --rpc-supervisor") != NULL,
              "the refusal is one explicit, actionable JOIN_REFUSED line");
        check(strstr(output, "dialing coord") == NULL &&
              strstr(output, "discovering coordinator") == NULL &&
              strstr(output, "pairing auth") == NULL &&
              strstr(output, "sent HELLO") == NULL,
              "retirement happens before discovery, pairing, or HELLO");
    }

    free(pairing);
    free(config);
    printf("worker join contract: %d checks, %d failures\n", checks, failures);
    if (failures) return 1;
    printf("WORKER_JOIN_CONTRACT_TEST_OK\n");
    return 0;
}
