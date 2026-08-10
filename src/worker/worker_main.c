/* IdleToken worker — inference shard.
 *
 * v0.1 scope (this stub): parse args, prove the binary links against the
 * ds4 vendor objects and the idletoken proto/net helpers. Inference and the
 * coordinator handshake land in the next steps. */

#include "idletoken_proto.h"
#include "idletoken_net.h"
#include "idletoken_discovery.h"
#include "idletoken_model.h"
#include "idletoken_resource.h"
#include "idletoken_advise.h"
#include "idletoken_weights.h"
#include "idletoken_gguf.h"   /* idletoken_gguf_identity — model identity check */
#include "idletoken_ds4x.h"   /* generic GQA/MLA CPU backend (small models) */
#ifdef IDLETOKEN_DS4X_CUDA
  #include "idletoken_ds4x_cuda.h"   /* ds4x_cuda_set_budget */
#endif
#include "ds4.h"

#ifndef DS4_NO_GPU
/* Declared here rather than by including ds4_gpu.h, which would pull in the GPU
 * tensor types this file has no other use for. The budget MUST travel by call:
 * this binary is MinGW, ds4cuda.dll is nvcc/MSVC, and their CRTs do not share
 * an environment block — the setenv that used to carry it was inert. */
void ds4_gpu_set_hybrid_vram_budget(uint64_t bytes);
int  ds4_gpu_synchronize(void);
void ds4_gpu_set_moe_cache(uint64_t bytes, uint32_t n_layers);
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
  #include <winsock2.h>      /* before windows.h */
  #include <windows.h>
  #include <bcrypt.h>        /* BCryptGenRandom — link -lbcrypt */
  #include <process.h>       /* _getpid */
  /* Every fd closed in this file is a network socket. */
  #define close(fd)     closesocket((SOCKET)(fd))
  #define getpid()      _getpid()
  #define usleep(us)    Sleep((DWORD)((us) / 1000))
  #define setenv(n,v,o) _putenv_s((n), (v))
  static int idletoken_getrandom(void *buf, size_t n) {
      return BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)n,
                             BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0 ? (int)n : -1;
  }
  #define getrandom(b, n, f) idletoken_getrandom((b), (n))
#else
  #include <dirent.h>
  #include <sys/random.h>
  #include <unistd.h>
#endif
#ifdef __linux__
  #include <signal.h>
  #include <sys/prctl.h>
#endif
#ifdef __APPLE__
  #include <limits.h>
  #include <mach-o/dyld.h>   /* _NSGetExecutablePath: where this binary lives */
#endif

#define IDLETOKEN_WORKER_VERSION "idletoken-worker v0.1.0-pre"

/* The model this worker was assigned (from ASSIGN_PLAN's model_id, resolved
 * against the registry). All tensor dimensions (n_embd/hc_streams/n_vocab)
 * come from here — never from DS4_* compile-time constants. Set right after
 * the plan is parsed, before any buffer sizing. */
static const idletoken_model_spec *g_model;

static void fill_uuid(uint8_t uuid[16]) {
#if defined(__APPLE__)
    /* macOS has no getrandom(2). getentropy(3) is the same kernel CSPRNG and
     * is documented never to fail for <= 256 bytes; the fallback below stays
     * reachable anyway rather than assuming that. */
    if (getentropy(uuid, 16) == 0) return;
#else
    if (getrandom(uuid, 16, 0) == 16) return;
#endif
    /* Fallback: time ^ pid. */
    uint64_t t = (uint64_t)time(NULL);
    uint64_t p = (uint64_t)getpid();
    for (int i = 0; i < 8; i++) uuid[i]   = (uint8_t)(t >> (8 * i));
    for (int i = 0; i < 8; i++) uuid[i+8] = (uint8_t)(p >> (8 * i));
}

/* ---- KV warm-cache maintenance (client settings, acceptance P5) -----------
 * The on-disk KV cache holds processed-context snapshots named `<sha40>.kv`
 * (same convention as ds4's kv_disk_cache; `.kv.tmp` = interrupted write).
 * The client's "clear my cache now" button runs `--kv-clear`; it only touches
 * those suffixes, is idempotent, and a missing directory counts as success. */

static void idletoken_default_kv_dir(char *buf, size_t n) {
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (base && base[0]) snprintf(buf, n, "%s\\IdleToken\\kv", base);
    else                 snprintf(buf, n, "C:\\IdleToken\\kv");
#else
    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) { snprintf(buf, n, "%s/idletoken/kv", xdg); return; }
    const char *home = getenv("HOME");
    if (home && home[0]) snprintf(buf, n, "%s/.cache/idletoken/kv", home);
    else                 snprintf(buf, n, "/tmp/idletoken-kv");
#endif
}

static int idletoken_is_kv_file(const char *name) {
    size_t len = strlen(name);
    return (len > 3 && !strcmp(name + len - 3, ".kv")) ||
           (len > 7 && !strcmp(name + len - 7, ".kv.tmp"));
}

static int idletoken_kv_clear(const char *dir) {
    unsigned removed = 0, failed = 0;
    unsigned long long freed = 0;
#ifdef _WIN32
    char pattern[1400];
    if (snprintf(pattern, sizeof pattern, "%s\\*", dir) >= (int)sizeof pattern) {
        fprintf(stderr, "idletoken-worker: kv-clear: directory path too long\n");
        return 1;
    }
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        if (e == ERROR_PATH_NOT_FOUND || e == ERROR_FILE_NOT_FOUND) {
            fprintf(stderr, "idletoken-worker: kv-clear: %s does not exist; nothing to clear\n", dir);
            return 0;
        }
        fprintf(stderr, "idletoken-worker: kv-clear: cannot open %s (error %lu)\n", dir, (unsigned long)e);
        return 1;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        if (!idletoken_is_kv_file(fd.cFileName)) continue;
        char path[1400];
        if (snprintf(path, sizeof path, "%s\\%s", dir, fd.cFileName) >= (int)sizeof path) {
            failed++;
            continue;
        }
        if (DeleteFileA(path)) {
            removed++;
            freed += ((unsigned long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        } else {
            failed++;
            fprintf(stderr, "idletoken-worker: kv-clear: cannot remove %s (error %lu)\n",
                    path, (unsigned long)GetLastError());
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    if (!d) {
        if (errno == ENOENT) {
            fprintf(stderr, "idletoken-worker: kv-clear: %s does not exist; nothing to clear\n", dir);
            return 0;
        }
        fprintf(stderr, "idletoken-worker: kv-clear: cannot open %s: %s\n", dir, strerror(errno));
        return 1;
    }
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!idletoken_is_kv_file(ent->d_name)) continue;
        char path[1400]; /* fits a 1023-byte dir + '/' + a 255-byte name */
        if (snprintf(path, sizeof path, "%s/%s", dir, ent->d_name) >= (int)sizeof path) {
            failed++;
            continue;
        }
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        unsigned long long sz = (unsigned long long)st.st_size;
        if (unlink(path) == 0) {
            removed++;
            freed += sz;
        } else {
            failed++;
            fprintf(stderr, "idletoken-worker: kv-clear: cannot remove %s: %s\n", path, strerror(errno));
        }
    }
    closedir(d);
#endif
    printf("idletoken-worker: kv-clear: removed %u file(s), freed %.1f MiB from %s%s\n",
           removed, (double)freed / (1024.0 * 1024.0), dir,
           failed ? " (some removals failed)" : "");
    return failed ? 1 : 0;
}

static void usage(FILE *out) {
    fprintf(out,
"idletoken-worker  DSv4-Flash inference shard (v0.1)\n"
"Usage: idletoken-worker (--coordinator <host:port> | --pair-code CODE) [--model <path>]\n"
"       idletoken-worker --probe-only [--gguf-dir <dir>]\n"
"\n"
"Required (run mode) — one of:\n"
"  --coordinator H:P   address of the cluster coordinator (manual)\n"
"  --pair-code CODE    discover the coordinator on the LAN by join code (no\n"
"                      manual address needed; --coordinator, if also given,\n"
"                      is used as an instant fallback)\n"
"\n"
"Account pairing (same-account machines self-assemble):\n"
"  --pair-account C    cloud cluster label C (+ --account-token, --rendezvous)\n"
"  --account-token JWT bearer token proving the account\n"
"  --rendezvous H:P    cloud rendezvous endpoint\n"
"  --discovery-port N  UDP discovery port (default 14097)\n"
"\n"
"Optional:\n"
"  --model PATH        DSv4-Flash GGUF (default: $IDLETOKEN_MODEL or ./ds4flash.gguf)\n"
"  --bind H:P          accept inter-stage TCP on this address (default: 0.0.0.0:14101)\n"
"  --probe-only        run hardware probe and exit without joining cluster\n"
"  --probe-json        like --probe-only but emit one line of JSON (for the client)\n"
"  --advise            print which models/precisions THIS machine can run\n"
"  --advise-json       same as --advise as one line of JSON (for the client)\n"
"  --advise-peers L    judge this machine PLUS peers; L = vramGiB:ramGiB:unified,...\n"
"  --gguf-dir DIR      directory to statvfs() for disk-avail (default: ./)\n"
"  --max-vram-mb N     cap usable VRAM at N MiB (client setting; 0 = no cap)\n"
"  --max-ram-mb N      cap usable RAM at N MiB (client setting; 0 = no cap)\n"
"  --kv-clear          wipe the on-disk KV warm cache and exit\n"
"  --kv-dir DIR        KV cache directory (default: platform cache dir)\n"
"  -h, --help          show this help\n"
"\n"
"v0.1 only runs PP (segment_id always 0). v0.2 will add SP=2.\n");
}

/* Pure compute time accumulated by this stage (printed before exit when
 * IDLETOKEN_DS4X_PROF=1). Owned exclusively by the single-threaded inference
 * loop, so no atomics are needed. */
static double   g_compute_s = 0.0;
static uint64_t g_compute_calls = 0;
/* The last stage's output projection (lm_head, [n_vocab x n_embd]) is timed
 * separately. It is **not** inside ds4x_runner_run, yet it can dominate the
 * per-token cost: across machines, which node ends up as the last stage moved a
 * single token from 106 ms to 2872 ms, while layer compute in both
 * configurations stayed in the tens of milliseconds -- the difference could only
 * be here. */
static double   g_logits_s = 0.0;
static uint64_t g_logits_calls = 0;
/* prefill and decode are accounted separately. Mixed together, a single
 * multi-second prefill drags "avg_ms" up until it means nothing -- and decode is
 * the number to watch when chasing latency. */
static double   g_prefill_s = 0.0;
static uint64_t g_prefill_calls = 0;
/* Harvesting the output (HC tensors / logits) is timed separately. CUDA encode
 * is an asynchronous submission, so the real synchronization point lands here;
 * only by splitting the two can you tell "compute is slow" from "readback is
 * slow", and merging them points the conclusion at the wrong half.
 * **Decode only**: one prefill readback is two orders of magnitude larger than
 * one decode readback, and mixing it into the average destroys the only useful
 * quantity, milliseconds per token (which is exactly how the first version of
 * this counter became worthless). */
static double   g_harvest_s = 0.0;
static uint64_t g_harvest_calls = 0;
/* GPU execution time, kept **separate** from readback.
 * The comment above claimed the two were split, but the code did not split
 * them: encode submits asynchronously, and the first blocking call afterwards
 * is the synchronous cudaMemcpy inside readback -- so **all GPU execution was
 * being charged to harvest**. Read that way, the data said "readback is 86% of
 * the time" and pointed optimization at transfer, even though logits are only
 * about 516 KB and no D2H copy that small takes 80 ms. There is now an explicit
 * synchronize before readback: its duration is GPU execution, and the memcpy
 * that follows is the real readback. */
static double   g_gpuexec_s = 0.0;
static uint64_t g_gpuexec_calls = 0;
/* Time spent blocked waiting for the next unit of work. In cross-machine
 * pipeline parallelism every token is one serial chain, so this stage's wait is
 * roughly the peer's compute plus two network trips. Subtract the compute time
 * the peer reports and you have the transport plus coordinator overhead, with
 * no timestamps added to the protocol. */
static double   g_wait_s = 0.0;
static uint64_t g_wait_calls = 0;
/* Wait and readback from prefill rounds are collected here, out of the decode
 * average. The first round's wait also includes idling for an HTTP request to
 * arrive; folding that in would add hundreds of milliseconds of phantom wait. */
static double   g_pf_other_s = 0.0;

static double now_monotonic_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* **Total** VRAM budget for the MoE expert cache (0 = off).
 *
 * It has to be a total rather than a per-layer slot count: `vram_usable` is
 * reported in RESOURCE_REPORT, which happens **before** ASSIGN_PLAN arrives and
 * the node learns how many layers it got. A total is independent of layer count
 * and can be subtracted from `vram_usable` on the spot, so the planner sees the
 * real headroom. A per-layer slot count cannot do that -- the first version did
 * it that way, and the split on identical hardware drifted from 28/15 to 16/27,
 * making two measurements incomparable. */
static uint64_t g_moe_cache_actual_bytes = 0;   /* actual value after trimming to this machine's pinned-memory headroom */

static uint64_t worker_moe_cache_budget(void) {
    const char *e = getenv("IDLETOKEN_MOE_CACHE_GB");
    if (!e || !e[0]) return 0;
    char *end = NULL;
    double gib = strtod(e, &end);
    if (end == e || gib <= 0.0 || gib > 64.0) return 0;
    return (uint64_t)(gib * 1073741824.0);
}

/* Pinned-memory ceiling: measure once, then remember it.
 *
 * Why it must be cached: the only way to measure it is to keep allocating until
 * allocation fails, which on this machine briefly reaches 46.5 GiB. Measuring
 * on every start would mean deliberately driving the machine out of memory at
 * startup -- precisely the class of problem we just fixed.
 *
 * The cache is keyed on ram_total: swapping RAM warrants a re-measure, while
 * other changes (a new driver, a different GPU) are far more likely to leave
 * this ceiling alone than to move it. Failing to read or write the cache only
 * costs time; correctness is unaffected. */
#ifndef DS4_NO_GPU
uint64_t ds4_gpu_probe_pinned_max(void);

static void pinnable_cache_path(char *out, size_t cap) {
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base || !base[0]) base = getenv("TEMP");
    snprintf(out, cap, "%s\\IdleToken-pinned-max.txt", base ? base : ".");
#else
    const char *base = getenv("XDG_CACHE_HOME");
    if (base && base[0]) snprintf(out, cap, "%s/idletoken-pinned-max", base);
    else {
        const char *home = getenv("HOME");
        snprintf(out, cap, "%s/.cache/idletoken-pinned-max", home ? home : ".");
    }
#endif
}

static uint64_t worker_measure_pinnable(uint64_t ram_total) {
    char path[512];
    pinnable_cache_path(path, sizeof(path));

    FILE *f = fopen(path, "r");
    if (f) {
        unsigned long long key = 0, val = 0;
        int n = fscanf(f, "%llu %llu", &key, &val);
        fclose(f);
        if (n == 2 && key == (unsigned long long)ram_total && val > 0) {
            fprintf(stderr, "idletoken-worker: pinned ceiling %.2f GiB (cached)\n",
                    (double)val / 1073741824.0);
            return (uint64_t)val;
        }
    }

    fprintf(stderr, "idletoken-worker: measuring pinned-memory ceiling once "
                    "(brief high memory use; result is cached in %s)\n", path);
    uint64_t got = ds4_gpu_probe_pinned_max();
    if (got == 0) return 0;   /* not measurable = unknown; callers treat it as unconstrained */

    f = fopen(path, "w");
    if (f) {
        fprintf(f, "%llu %llu\n", (unsigned long long)ram_total, (unsigned long long)got);
        fclose(f);
    }
    return got;
}
#endif /* !DS4_NO_GPU */

/* The one-line PROF2 dump. It **must be printable periodically**, not only at
 * exit: cross-machine benchmarks are torn down with `Stop-Process -Force`, the
 * exit path never runs, and you end up having added timing that measured
 * nothing. This project has already paid for that once (the stagebench round). */
static void prof2_dump(uint32_t stage_id, uint32_t layer_lo, uint32_t layer_hi) {
    fprintf(stderr,
            "idletoken-worker: PROF2 stage=%u layers=[%u,%u) "
            "decode: n=%llu enc=%.1fms gpuexec=%.1fms harvest=%.1fms wait=%.1fms (%.1fms per token on this stage)  "
            "prefill: n=%llu enc=%.0fms other=%.0fms\n",
            stage_id, layer_lo, layer_hi,
            (unsigned long long)g_compute_calls,
            g_compute_calls ? g_compute_s * 1000.0 / (double)g_compute_calls : 0.0,
            g_gpuexec_calls ? g_gpuexec_s * 1000.0 / (double)g_gpuexec_calls : 0.0,
            g_harvest_calls ? g_harvest_s * 1000.0 / (double)g_harvest_calls : 0.0,
            g_wait_calls ? g_wait_s * 1000.0 / (double)g_wait_calls : 0.0,
            g_compute_calls ? (g_compute_s + g_gpuexec_s + g_harvest_s) * 1000.0 / (double)g_compute_calls : 0.0,
            (unsigned long long)g_prefill_calls,
            g_prefill_calls ? g_prefill_s * 1000.0 / (double)g_prefill_calls : 0.0,
            g_pf_other_s * 1000.0);
    fflush(stderr);
}

/* v4 multi-sequence (scheduler-design §6-E2): fetch the ds4_session /
 * ds4x_runner for a given seq_id, **creating it on first use**.
 *
 * Why lazily: every sequence preallocates a full KV cache sized to ctx. The
 * coordinator uses a single slot by default, so preallocating N sequences wastes
 * N times the VRAM. When creation fails (it does not fit) we return -1, and the
 * caller fails this round so the coordinator or the platform can move the work
 * elsewhere -- rather than driving the machine into OOM.
 *
 * `real_ds4` / `real_ds4x` are the test for which backend this machine actually
 * loaded (i.e. session / xr non-NULL in main). On the MOCK path both are false,
 * the output is all zeros, and no sequence state is needed. */
static int seq_resolve(uint8_t seq_id,
                       ds4_session **sessions, ds4x_runner **runners,
                       ds4_engine *engine, ds4x_model *xm, uint32_t ctx_size,
                       int real_ds4, int real_ds4x,
                       ds4_session **out_session, ds4x_runner **out_runner) {
    *out_session = NULL;
    *out_runner  = NULL;
    if (seq_id >= IDLETOKEN_MAX_SEQ_SLOTS) return -1;
    if (real_ds4) {
        if (!sessions[seq_id]) {
            ds4_session *ns = NULL;
            int rc = ds4_session_create(&ns, engine, (int)ctx_size);
            if (rc != 0 || !ns) {
                fprintf(stderr, "idletoken-worker: ds4_session_create(seq=%u) failed (rc=%d)"
                                " — not enough room for this many sequences\n", seq_id, rc);
                return -1;
            }
            sessions[seq_id] = ns;
            fprintf(stderr, "idletoken-worker: created KV sequence slot %u\n", seq_id);
        }
        *out_session = sessions[seq_id];
        return 0;
    }
    if (real_ds4x) {
        if (!runners[seq_id]) {
            char xerr2[256] = "";
            ds4x_runner *nr = ds4x_runner_create(xm, ctx_size, xerr2, sizeof(xerr2));
            if (!nr) {
                fprintf(stderr, "idletoken-worker: ds4x_runner_create(seq=%u) failed: %.180s\n",
                        seq_id, xerr2);
                return -1;
            }
            runners[seq_id] = nr;
            fprintf(stderr, "idletoken-worker: created ds4x sequence slot %u\n", seq_id);
        }
        *out_runner = runners[seq_id];
        return 0;
    }
    return 0;   /* MOCK: no sequence state needed */
}

#ifdef __APPLE__
/* Point the ds4 Metal backend at its shader sources.
 *
 * ds4 compiles the metal/ shader sources at ds4_gpu_init() time and looks for
 * them relative to the CURRENT WORKING DIRECTORY. That works for `./ds4` run
 * from its checkout and fails for us: the worker is started by the client, or
 * by a script, or from the user's home — and the failure ("Metal source not
 * found") looks like a broken install rather than a wrong cwd.
 *
 * So resolve it from where THIS BINARY lives, which is stable no matter who
 * launched it, and hand ds4 a single base directory (patch 0011). An explicit
 * DS4_METAL_SOURCE_DIR always wins, so a packaged app can point elsewhere. */
static void mac_locate_metal_sources(void) {
    if (getenv("DS4_METAL_SOURCE_DIR")) return;

    char exe[PATH_MAX];
    uint32_t n = sizeof(exe);
    if (_NSGetExecutablePath(exe, &n) != 0) return;

    char real[PATH_MAX];
    if (!realpath(exe, real)) snprintf(real, sizeof(real), "%s", exe);
    char *slash = strrchr(real, '/');
    if (!slash) return;
    *slash = '\0';                       /* real = directory of the binary */

    /* Installed layout first, then the repo layout (make drops the binary in
     * the repo root, where the vendored engine sits under vendor/ds4). */
    const char *rel[] = { "metal", "vendor/ds4/metal", "../vendor/ds4/metal" };
    for (size_t i = 0; i < sizeof(rel) / sizeof(rel[0]); i++) {
        char cand[PATH_MAX], probe[PATH_MAX];
        snprintf(cand, sizeof(cand), "%s/%s", real, rel[i]);
        snprintf(probe, sizeof(probe), "%s/flash_attn.metal", cand);
        if (access(probe, R_OK) == 0) {
            setenv("DS4_METAL_SOURCE_DIR", cand, 1);
            /* Logged, not silent: "which shaders did it actually compile" is
             * the first question when a Metal build behaves oddly. */
            fprintf(stderr, "idletoken-worker: ds4 Metal shaders from %s\n", cand);
            return;
        }
    }
    /* Not found: say so now, with the directories tried, instead of letting
     * Metal init fail later with only the file name. */
    fprintf(stderr, "idletoken-worker: ds4 Metal shader sources not found near %s "
                    "— set DS4_METAL_SOURCE_DIR if they live elsewhere\n", real);
}
#endif /* __APPLE__ */

int main(int argc, char **argv) {
#ifdef __APPLE__
    mac_locate_metal_sources();
#endif
#ifdef __linux__
    /* Opt-in (set by the client supervisor): die with the launching client so
     * even a SIGKILLed client never orphans the engine. Must NOT be default —
     * scripted deploys launch us under nohup and outlive their shell. */
    if (getenv("IDLETOKEN_DIE_WITH_PARENT")) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() == 1) _exit(0); /* parent already gone before prctl */
    }
#elif defined(_WIN32)
    /* Windows side of parent-death: watch the client's process handle. */
    idletoken_die_with_parent();
    /* Unbuffered stderr: a hard crash in the CUDA DLL (access violation /
     * stack-guard fastfail) otherwise loses the last diagnostic line, since
     * Windows full-buffers stderr when it is redirected to a file. */
    setvbuf(stderr, NULL, _IONBF, 0);
#endif
    const char *coord_addr = NULL;
    /* Pairing / discovery: with --pair-code (or account mode) the worker finds
     * the coordinator over the LAN instead of needing --coordinator. --pair-code
     * can be passed via env for the Tauri sidecar without arg quoting. */
    const char *pair_code  = getenv("IDLETOKEN_PAIR_CODE");
    const char *pair_acct  = getenv("IDLETOKEN_PAIR_ACCOUNT");
    const char *acct_token = getenv("IDLETOKEN_ACCOUNT_TOKEN");
    const char *rendezvous = getenv("IDLETOKEN_RENDEZVOUS");
    int         disc_port  = IDLETOKEN_DISCOVERY_PORT;
    const char *model_path = getenv("IDLETOKEN_MODEL");
    const char *bind_addr  = "0.0.0.0:14101";
    const char *gguf_dir   = "./";
    /* Layer-shard weight repo (IdleToken): when set, this worker loads only its
     * assigned layers instead of the whole 80GB GGUF. Value is a base — a local
     * directory now, an http(s):// repo URL later — under which per-range
     * partial GGUFs live as `L<lo>-<hi>.gguf`. Overrides the plan's model_path.
     * A local dir must already hold the materialized partial (scripts/gguf_shard.py);
     * an http base triggers a byte-range fetch into the local cache. */
    const char *shard_repo = getenv("IDLETOKEN_SHARD_REPO");
    /* Optional DSpark draft module (docs/dspark-design.md). Env var as well
     * as a flag so the Tauri sidecar can set it without arg quoting. */
    const char *dspark_path = getenv("IDLETOKEN_DSPARK_GGUF");
    /* Coordinator-side weight repo server (IdleToken): serve `--gguf-dir` (the
     * master GGUF + its .idx) over HTTP byte-range so remote workers fetch only
     * their layers. Runs as an isolated idletoken-worker sidecar. */
    const char *serve_weights = NULL;   /* GGUF file to serve (its dir is the repo) */
    int weights_port = 8001;
    int probe_only = 0;
    int probe_json = 0;
    int advise = 0;       /* --advise: "what can this machine run?" table */
    int advise_json = 0;
    /* --advise-peers: judge a HYPOTHETICAL cluster, not just this machine.
     * Format: "vramGiB:ramGiB:unified,..." per extra machine. Answers the
     * question a user actually asks ("what if I add my other PC?") and lets the
     * topology matrix ask the advisor whether a subset SHOULD fit before it
     * spends ten minutes proving that it does not. */
    const char *advise_peers = NULL;
    int kv_clear = 0;
    const char *kv_dir = NULL;
    /* Usage caps from the client's settings panel (MiB; 0 = no cap). Also
     * accept env vars so the Tauri sidecar can pass them without arg quoting. */
    const char *env_vram = getenv("IDLETOKEN_MAX_VRAM_MB");
    const char *env_ram  = getenv("IDLETOKEN_MAX_RAM_MB");
    uint64_t max_vram_mb = env_vram ? strtoull(env_vram, NULL, 10) : 0;
    uint64_t max_ram_mb  = env_ram  ? strtoull(env_ram,  NULL, 10) : 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--coordinator") && i + 1 < argc) coord_addr = argv[++i];
        else if (!strcmp(a, "--model")       && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(a, "--bind")        && i + 1 < argc) bind_addr  = argv[++i];
        else if (!strcmp(a, "--gguf-dir")    && i + 1 < argc) gguf_dir   = argv[++i];
        else if (!strcmp(a, "--shard-repo")  && i + 1 < argc) shard_repo = argv[++i];
        else if (!strcmp(a, "--dspark")      && i + 1 < argc) dspark_path = argv[++i];
        else if (!strcmp(a, "--serve-weights") && i + 1 < argc) serve_weights = argv[++i];
        else if (!strcmp(a, "--weights-port") && i + 1 < argc) weights_port = atoi(argv[++i]);
        else if (!strcmp(a, "--max-vram-mb") && i + 1 < argc) max_vram_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--max-ram-mb")  && i + 1 < argc) max_ram_mb  = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--probe-only"))                  probe_only = 1;
        else if (!strcmp(a, "--probe-json"))                  probe_json = 1;
        else if (!strcmp(a, "--advise"))                      advise = 1;
        else if (!strcmp(a, "--advise-json"))                 advise_json = 1;
        else if (!strcmp(a, "--advise-peers") && i + 1 < argc) advise_peers = argv[++i];
        else if (!strcmp(a, "--kv-clear"))                    kv_clear = 1;
        else if (!strcmp(a, "--kv-dir")      && i + 1 < argc) kv_dir = argv[++i];
        else if (!strcmp(a, "--pair-code")   && i + 1 < argc) pair_code   = argv[++i];
        else if (!strcmp(a, "--pair-account")&& i + 1 < argc) pair_acct   = argv[++i];
        else if (!strcmp(a, "--account-token")&& i + 1 < argc) acct_token = argv[++i];
        else if (!strcmp(a, "--rendezvous")  && i + 1 < argc) rendezvous  = argv[++i];
        else if (!strcmp(a, "--discovery-port") && i + 1 < argc) disc_port = atoi(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "idletoken-worker: unknown argument: %s\n\n", a); usage(stderr); return 2; }
    }
    if (!model_path) model_path = "./ds4flash.gguf";

    if (kv_clear) {
        char defdir[1024];
        if (!kv_dir) {
            idletoken_default_kv_dir(defdir, sizeof defdir);
            kv_dir = defdir;
        }
        return idletoken_kv_clear(kv_dir);
    }

    /* Weight repo server mode: generate the .idx if missing, then serve the
     * GGUF's directory over HTTP byte-range (blocks). The coordinator runs this
     * so remote workers fetch only their layers. */
    if (serve_weights) {
        char idx_path[1200];
        snprintf(idx_path, sizeof idx_path, "%s.idx", serve_weights);
        /* Rebuild when the index is missing OR describes a different file. The
         * missing-only check shipped a repo whose .idx had been built from a
         * half-downloaded GGUF: right offsets, wrong file_size, so every worker
         * re-fetched its entire shard on every single run and nothing ever said
         * why (idletoken_idx_stale). */
        if (idletoken_idx_stale(serve_weights, idx_path)) {
            fprintf(stderr, "idletoken-worker: (re)building weight index %s\n", idx_path);
            if (idletoken_write_idx(serve_weights, idx_path) != 0) {
                fprintf(stderr, "idletoken-worker: could not build weight index for %s\n", serve_weights);
                return 1;
            }
        }
        /* Serve the directory that holds the GGUF (+ its .idx). */
        char dir[1024];
        snprintf(dir, sizeof dir, "%s", serve_weights);
        char *slash = strrchr(dir, '/');
#ifdef _WIN32
        char *bslash = strrchr(dir, '\\');
        if (bslash > slash) slash = bslash;
#endif
        if (slash) *slash = '\0'; else snprintf(dir, sizeof dir, ".");
        char bind[64];
        snprintf(bind, sizeof bind, "0.0.0.0:%d", weights_port);
#ifdef _WIN32
        {   /* remote workers must reach this HTTP range server */
            char rule[64];
            snprintf(rule, sizeof rule, "IdleToken weights TCP %d", weights_port);
            idletoken_win_ensure_firewall_rule(rule, "TCP", weights_port);
        }
#endif
        return idletoken_serve_weights(dir, bind);
    }

    const uint64_t max_vram_bytes = max_vram_mb * 1024ull * 1024ull;
    const uint64_t max_ram_bytes  = max_ram_mb  * 1024ull * 1024ull;

    /* --advise: the answer to "can my computer run any of this?". Same probe,
     * but the output is the capability table rather than raw numbers — the
     * verdicts come from the planner (idletoken_advise), never from a second
     * estimate, so the table cannot promise what the cluster then refuses. */
    if (advise || advise_json) {
        idletoken_resource_report rr;
        if (idletoken_resource_probe(&rr, gguf_dir) != 0 && !advise_json)
            fprintf(stderr, "idletoken-worker: probe encountered errors (report below is partial)\n");
        idletoken_resource_apply_caps(&rr, max_vram_bytes, max_ram_bytes);

        char why[256] = "";
        idletoken_hw_status hw = idletoken_hw_check(&rr, why, sizeof(why));

        /* Zero-init on purpose: `idletoken_node_mem` has a `ram_pinnable` field
         * that this path never measures (the pinned-memory probe runs later,
         * and --advise must stay cheap). Leaving it as stack garbage made
         * idletoken_plan_layers() clamp a discrete node's host share to a
         * random ceiling — nondeterministic capability advice, no diagnostic.
         * 0 = unknown/unconstrained, which is what the planner's guard expects. */
        idletoken_node_mem pool[17] = {0};
        int n_pool = 1;
        pool[0].vram_usable  = rr.vram_usable;
        pool[0].ram_usable   = rr.ram_usable;
        pool[0].ram_pinnable = rr.ram_pinnable;  /* 0 here; picked up free if probe ever measures it */
        pool[0].unified      = rr.unified_memory ? 1u : 0u;
        if (advise_peers && *advise_peers) {
            const char *p = advise_peers;
            while (*p && n_pool < (int)(sizeof(pool) / sizeof(pool[0]))) {
                double v = 0, r = 0; int u = 0;
                if (sscanf(p, "%lf:%lf:%d", &v, &r, &u) >= 2) {
                    pool[n_pool].vram_usable  = (uint64_t)(v * 1073741824.0);
                    pool[n_pool].ram_usable   = (uint64_t)(r * 1073741824.0);
                    pool[n_pool].ram_pinnable = 0;  /* peer string carries v:r:unified only */
                    pool[n_pool].unified      = u ? 1u : 0u;
                    n_pool++;
                }
                const char *comma = strchr(p, ',');
                if (!comma) break;
                p = comma + 1;
            }
        }
        idletoken_advice_row rows[IDLETOKEN_ADVISE_MAX_ROWS];
        int n = idletoken_advise(pool, n_pool, rows, IDLETOKEN_ADVISE_MAX_ROWS);
        if (n < 0) { fprintf(stderr, "idletoken-worker: advise failed\n"); return 1; }

        if (advise_json) {
            idletoken_advise_print_json(rows, n, n_pool);
        } else {
            char hw_line[320];
            snprintf(hw_line, sizeof hw_line,
                     "%s (cc %u.%u, driver %s) · %.1f GB usable VRAM · %.1f GB usable RAM",
                     rr.gpu_name[0] ? rr.gpu_name : "no GPU",
                     rr.cc_major, rr.cc_minor,
                     rr.driver_version[0] ? rr.driver_version : "unknown",
                     (double)rr.vram_usable / 1073741824.0,
                     (double)rr.ram_usable  / 1073741824.0);
            idletoken_advise_print(rows, n, n_pool, hw_line);
            if (hw != IDLETOKEN_HW_OK)
                printf("\n  NOTE: %s\n  Nothing in the table above can run until that is fixed.\n", why);
        }
        return hw == IDLETOKEN_HW_OK ? 0 : 2;
    }

    if (probe_only || probe_json) {
        idletoken_resource_report rr;
        if (idletoken_resource_probe(&rr, gguf_dir) != 0) {
            fprintf(stderr, "idletoken-worker: probe encountered errors (partial report below)\n");
        }
        idletoken_resource_apply_caps(&rr, max_vram_bytes, max_ram_bytes);
        if (probe_json) idletoken_resource_print_json(&rr);
        else            idletoken_resource_print(&rr);
        /* Verdict on the hardware floor (G-HW). Exit non-zero when this machine
         * cannot serve layers, so `--probe-only` doubles as the installer/CI
         * check and a caller cannot mistake "printed a report" for "usable". */
        {
            char why[256] = "";
            idletoken_hw_status hw = idletoken_hw_check(&rr, why, sizeof(why));
            if (hw != IDLETOKEN_HW_OK) {
                if (!probe_json)
                    fprintf(stderr, "\nidletoken-worker: this machine cannot join a "
                                    "cluster — %s\n", why);
                return 2;
            }
        }
        return 0;
    }

#ifdef _WIN32
    /* Self-provision inbound firewall rules (architecture §9, productization).
     * Pairing on real machines failed twice on silently filtered inbound ports
     * (the UDP beacon on 14097, the HC ports 14322/14323), and until now the fix
     * was a manual netsh invocation. The rule name carries the port, so changing
     * ports adds a new rule automatically; without elevation we print the exact
     * command instead. Set IDLETOKEN_NO_FIREWALL_RULE=1 to skip. */
    {
        const char *colon = strrchr(bind_addr, ':');
        int bind_port = colon ? atoi(colon + 1) : 0;
        char rule[64];
        if (bind_port > 0) {
            snprintf(rule, sizeof rule, "IdleToken worker TCP %d", bind_port);
            idletoken_win_ensure_firewall_rule(rule, "TCP", bind_port);
        }
        if (pair_code || pair_acct) {
            snprintf(rule, sizeof rule, "IdleToken discovery UDP %d", disc_port);
            idletoken_win_ensure_firewall_rule(rule, "UDP", disc_port);
        }
    }
#endif

    printf("idletoken-worker v0.1.0-pre  (proto v%u, header=%zuB)\n",
           (unsigned)IDLETOKEN_PROTO_VERSION, sizeof(idletoken_msg_header));
    printf("  coordinator : %s\n",
           coord_addr ? coord_addr
                      : (pair_code || pair_acct) ? "(discover by pairing)" : "(unset — required)");
    printf("  model       : %s\n", model_path);
    printf("  bind        : %s\n", bind_addr);

    /* Smoke test: pack/unpack a HELLO header through the wire codec. */
    idletoken_msg_header h0 = {
        .magic         = IDLETOKEN_PROTO_MAGIC,
        .version       = IDLETOKEN_PROTO_VERSION,
        .msg_type      = IDLETOKEN_MSG_HELLO,
        .payload_bytes = 0,
        .request_id    = 0x0123456789ABCDEFULL,
        .stage_id      = 0,
        .segment_id    = IDLETOKEN_SEGMENT_NONE,
    };
    uint8_t wire[48];
    idletoken_header_pack(&h0, wire);

    idletoken_msg_header h1;
    idletoken_header_unpack(wire, &h1);
    int ok = (h1.magic == h0.magic) && (h1.version == h0.version)
          && (h1.msg_type == h0.msg_type) && (h1.request_id == h0.request_id);
    printf("  wire codec  : %s (request_id=0x%016llx)\n",
           ok ? "OK" : "FAIL", (unsigned long long)h1.request_id);

    /* Build the pairing identity (code or account), if any. */
    idletoken_pair_id pair_id;
    int pairing = 0;
    if (pair_acct) {
        if (!acct_token || !rendezvous) {
            fprintf(stderr, "\nidletoken-worker: --pair-account needs --account-token and --rendezvous\n");
            return 1;
        }
        if (idletoken_pair_id_from_account(&pair_id, pair_acct, acct_token, rendezvous) != 0) {
            fprintf(stderr, "idletoken-worker: bad account pairing spec\n"); return 1;
        }
        pairing = 1;
    } else if (pair_code) {
        if (!idletoken_pair_code_valid(pair_code)) {
            fprintf(stderr, "\nidletoken-worker: invalid join code '%s'\n", pair_code); return 1;
        }
        if (idletoken_pair_id_from_code(&pair_id, pair_code) != 0) {
            fprintf(stderr, "idletoken-worker: bad join code\n"); return 1;
        }
        pairing = 1;
    }

    if (!coord_addr && !pairing) {
        fprintf(stderr, "\nidletoken-worker: need --coordinator <host:port> OR --pair-code CODE "
                        "(or account pairing) to run.\n");
        return 1;
    }

    /* Run resource probe once up front; HELLO is followed by RESOURCE_REPORT. */
    idletoken_resource_report rr;
    if (idletoken_resource_probe(&rr, gguf_dir) != 0) {
        fprintf(stderr, "idletoken-worker: probe encountered errors; continuing with partial report\n");
    }
    /* HYBRID puts every layer that does not fit in VRAM into pinned memory, so
     * the real bound on the host-side share is that ceiling, not ram_usable. It
     * sits far below physical RAM and cannot be derived (see
     * idletoken_resource.h); it can only be measured, once, and cached.
     * Which backend will run is still unknown here (plan_backend waits for
     * ASSIGN_PLAN), so we measure regardless of backend -- after the first time
     * it is just a file read anyway. */
#ifndef DS4_NO_GPU
    rr.ram_pinnable = worker_measure_pinnable(rr.ram_total);
    /* The cache budget is **not** subtracted from `vram_usable`.
     *
     * `vram_usable` decides how many layers this node accepts, and the cache
     * does not reduce that capacity -- experts that do not fit already spill to
     * host memory. What the cache actually competes for is the **resident share
     * of experts in VRAM**, so it belongs in the auto-HYBRID VRAM budget instead
     * (see the ds4_gpu_set_hybrid_vram_budget section below).
     *
     * The first version subtracted it here and measurement punished it at once:
     * the split stopped drifting, but the resident weights we gave up exactly
     * cancelled the cache's gain -- 8.298 (no cache) -> 8.342 (subtracted in the
     * wrong place) -> 6.856 (not subtracted; there was VRAM headroom anyway). */
#endif

    /* Honor the client's usage caps before reporting, so the coordinator's
     * layer split respects this machine's configured limits. */
    idletoken_resource_apply_caps(&rr, max_vram_bytes, max_ram_bytes);

    /* Hardware floor (G-HW): refuse BEFORE the handshake. A pre-Turing card or
     * a driver too old for the shipped CUDA does not fail cleanly at load time
     * — it produces garbage tokens or a silent MOCK fallback, which then looks
     * like a cluster bug. Say what is required and stop. */
    {
        char why[256] = "";
        idletoken_hw_status hw = idletoken_hw_check(&rr, why, sizeof(why));
        if (hw != IDLETOKEN_HW_OK) {
            fprintf(stderr, "idletoken-worker: refusing to join — %s\n", why);
            return 2;
        }
    }

    /* --- resolve the coordinator address --------------------------------- */
    /* With pairing, discover the coordinator over the LAN (broadcast -> manual
     * -> subnet -> rendezvous). A given --coordinator becomes the instant
     * manual fallback. Without pairing, use --coordinator directly. */
    char resolved_addr[80] = "";
    if (pairing) {
        idletoken_discovery *disc = idletoken_discovery_multi((uint16_t)disc_port, coord_addr);
        if (!disc) { fprintf(stderr, "idletoken-worker: discovery init failed\n"); return 1; }
        fprintf(stderr, "idletoken-worker: discovering coordinator for %s pairing (udp/%d, up to 60s)...\n",
                pair_id.mode == IDLETOKEN_PAIR_MODE_ACCOUNT ? "account" : "code", disc_port);
        int rc = disc->resolve(disc, &pair_id, resolved_addr, sizeof(resolved_addr), 60000);
        disc->destroy(disc);
        if (rc != 0) {
            fprintf(stderr, "idletoken-worker: no coordinator found for that code/account on this LAN\n");
            return 1;
        }
        coord_addr = resolved_addr;
        fprintf(stderr, "idletoken-worker: discovered coordinator at %s\n", coord_addr);
    }

    fprintf(stderr, "idletoken-worker: dialing coord at %s\n", coord_addr);
    int fd = idletoken_connect_tcp(coord_addr);
    if (fd < 0) {
        fprintf(stderr, "idletoken-worker: connect: %s\n", strerror(errno));
        return 1;
    }

    /* Pairing auth preamble: prove both sides know the shared secret and derive
     * a session key, before the normal HELLO. Rejected => wrong code/account. */
    if (pairing) {
        uint8_t session_key[IDLETOKEN_SESSION_KEY_BYTES];
        if (idletoken_pair_client_auth(fd, &pair_id, session_key) != 0) {
            fprintf(stderr, "idletoken-worker: pairing auth failed (%s) — wrong code?\n",
                    strerror(errno));
            close(fd); return 1;
        }
        fprintf(stderr, "idletoken-worker: pairing auth OK (mutual)\n");
    }

    /* --- HELLO ----------------------------------------------------------- */
    uint8_t uuid[16];
    fill_uuid(uuid);

    uint8_t hello_payload[1024];
    idletoken_buf b;
    idletoken_buf_init(&b, hello_payload, sizeof(hello_payload));
    idletoken_buf_put_bytes(&b, uuid, 16);
    idletoken_buf_put_str(&b, rr.hostname);
    idletoken_buf_put_str(&b, IDLETOKEN_WORKER_VERSION);
    idletoken_buf_put_str(&b, bind_addr);
    idletoken_buf_put_u8(&b, 1);  /* os_family Linux */
    uint8_t pad3[3] = {0};
    if (idletoken_buf_put_bytes(&b, pad3, 3) != 0 || b.err) {
        fprintf(stderr, "idletoken-worker: HELLO payload pack overflow\n");
        close(fd); return 1;
    }

    idletoken_msg_header hello = {
        .magic         = IDLETOKEN_PROTO_MAGIC,
        .version       = IDLETOKEN_PROTO_VERSION,
        .msg_type      = IDLETOKEN_MSG_HELLO,
        .payload_bytes = b.pos,
        .request_id    = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32),
        .stage_id      = 0,
        .segment_id    = IDLETOKEN_SEGMENT_NONE,
    };
    if (idletoken_send_msg(fd, &hello, hello_payload, b.pos) != 0) {
        fprintf(stderr, "idletoken-worker: send HELLO: %s\n", strerror(errno));
        close(fd); return 1;
    }
    fprintf(stderr, "idletoken-worker: sent HELLO (%zu B payload, request_id=0x%016llx)\n",
            b.pos, (unsigned long long)hello.request_id);

    /* --- HELLO_ACK ------------------------------------------------------- */
    uint8_t ack_payload[1024];
    idletoken_msg_header ack;
    if (idletoken_recv_msg(fd, &ack, ack_payload, sizeof(ack_payload)) != 0) {
        fprintf(stderr, "idletoken-worker: recv HELLO_ACK: %s\n", strerror(errno));
        close(fd); return 1;
    }
    if (ack.msg_type != IDLETOKEN_MSG_HELLO_ACK) {
        fprintf(stderr, "idletoken-worker: expected HELLO_ACK, got msg_type=0x%04x\n",
                (unsigned)ack.msg_type);
        close(fd); return 1;
    }
    fprintf(stderr, "idletoken-worker: got HELLO_ACK (request_id=0x%016llx)\n",
            (unsigned long long)ack.request_id);

    /* --- RESOURCE_REPORT ------------------------------------------------- */
    uint8_t rep[512];
    idletoken_buf rb;
    idletoken_buf_init(&rb, rep, sizeof(rep));
    idletoken_buf_put_str(&rb, rr.gpu_name);
    idletoken_buf_put_u8 (&rb, rr.cc_major);
    idletoken_buf_put_u8 (&rb, rr.cc_minor);
    idletoken_buf_put_u8 (&rb, rr.unified_memory ? 1 : 0);
    idletoken_buf_put_u8 (&rb, 0);                   /* reserved */
    idletoken_buf_put_u64(&rb, rr.vram_total);
    idletoken_buf_put_u64(&rb, rr.vram_used_other);
    idletoken_buf_put_u64(&rb, rr.vram_usable);
    idletoken_buf_put_u64(&rb, rr.ram_total);
    idletoken_buf_put_u64(&rb, rr.ram_used_other);
    idletoken_buf_put_u64(&rb, rr.ram_usable);
    idletoken_buf_put_u32(&rb, rr.cpu_count);
    /* The measured pinned-memory ceiling in MiB, carried in what used to be a
     * reserved u32. Not appending a new field is deliberate: the parser checks
     * b.err after reading the fixed fields, so a new coordinator can read extra
     * trailing bytes, but a datagram from an old worker would make a new
     * coordinator read past the end. Reusing reserved is compatible in both
     * directions -- 0 has always meant "unknown", which is exactly its original
     * meaning. A u32 of MiB tops out at 4 PiB, and the probe itself has 512 MiB
     * granularity. */
    idletoken_buf_put_u32(&rb, (uint32_t)(rr.ram_pinnable >> 20));
    idletoken_buf_put_u64(&rb, rr.disk_avail);
    idletoken_buf_put_u32(&rb, 0);                   /* net_link_mbps; v0.1 unknown */
    idletoken_buf_put_u32(&rb, 0);                   /* reserved */
    idletoken_buf_put_u8 (&rb, rr.vram_usable >= (10ull << 30) ? 1 : 0);  /* can_run_ds4 rough gate */
    uint8_t pad7[7] = {0};
    if (idletoken_buf_put_bytes(&rb, pad7, 7) != 0 || rb.err) {
        fprintf(stderr, "idletoken-worker: RESOURCE_REPORT pack overflow\n");
        close(fd); return 1;
    }
    idletoken_msg_header rep_hdr = {
        .magic         = IDLETOKEN_PROTO_MAGIC,
        .version       = IDLETOKEN_PROTO_VERSION,
        .msg_type      = IDLETOKEN_MSG_RESOURCE_REPORT,
        .payload_bytes = rb.pos,
        .request_id    = hello.request_id,
        .stage_id      = 0,
        .segment_id    = IDLETOKEN_SEGMENT_NONE,
    };
    if (idletoken_send_msg(fd, &rep_hdr, rep, rb.pos) != 0) {
        fprintf(stderr, "idletoken-worker: send RESOURCE_REPORT: %s\n", strerror(errno));
        close(fd); return 1;
    }
    fprintf(stderr, "idletoken-worker: sent RESOURCE_REPORT (%zu B payload)\n", rb.pos);

    /* --- ASSIGN_PLAN ----------------------------------------------------- */
    uint8_t plan_payload[2048];
    idletoken_msg_header plan_hdr;
    if (idletoken_recv_msg(fd, &plan_hdr, plan_payload, sizeof(plan_payload)) != 0) {
        fprintf(stderr, "idletoken-worker: recv ASSIGN_PLAN: %s\n", strerror(errno));
        close(fd); return 1;
    }
    if (plan_hdr.msg_type != IDLETOKEN_MSG_ASSIGN_PLAN) {
        fprintf(stderr, "idletoken-worker: expected ASSIGN_PLAN, got msg_type=0x%04x\n",
                (unsigned)plan_hdr.msg_type);
        close(fd); return 1;
    }
    /* v2 changed this payload's layout (model id + u16 layer range); a v1
     * coordinator's plan would silently mis-parse. net.c only rejects NEWER
     * versions, so check equality here (mirror of the coordinator's HELLO
     * check). */
    if (plan_hdr.version != IDLETOKEN_PROTO_VERSION) {
        fprintf(stderr, "idletoken-worker: coordinator speaks protocol v%u, we need "
                        "v%u — upgrade the older side\n",
                (unsigned)plan_hdr.version, (unsigned)IDLETOKEN_PROTO_VERSION);
        close(fd); return 1;
    }

    idletoken_buf pb;
    idletoken_buf_init(&pb, plan_payload, plan_hdr.payload_bytes);
    uint8_t cluster_size = 0, stage_id = 0, segment_id = 0, mode = 0;
    uint16_t layer_lo = 0, layer_hi = 0, plan_n_layers = 0;
    uint8_t plan_backend = 0, pad1 = 0, hc_dtype = 0;
    uint8_t pad7_in[7];
    uint32_t plan_ctx_size = 0, prefill_cap = 0;
    uint8_t sha256[32];
    char plan_model_id[64];
    char plan_quant[32];
    char plan_model_path[512];
    char prev_addr[64], next_addr[64], coord_inbox[64];

    idletoken_buf_get_u8   (&pb, &cluster_size);
    idletoken_buf_get_u8   (&pb, &stage_id);
    idletoken_buf_get_u8   (&pb, &segment_id);
    idletoken_buf_get_u8   (&pb, &mode);
    idletoken_buf_get_u16  (&pb, &layer_lo);
    idletoken_buf_get_u16  (&pb, &layer_hi);
    idletoken_buf_get_u16  (&pb, &plan_n_layers);
    idletoken_buf_get_u8   (&pb, &plan_backend);
    idletoken_buf_get_u8   (&pb, &pad1);
    idletoken_buf_get_u32  (&pb, &plan_ctx_size);
    idletoken_buf_get_u32  (&pb, &prefill_cap);
    idletoken_buf_get_u8   (&pb, &hc_dtype);
    idletoken_buf_get_bytes(&pb, pad7_in, 7);
    idletoken_buf_get_bytes(&pb, sha256, 32);
    idletoken_buf_get_str  (&pb, plan_model_id,   sizeof(plan_model_id));
    idletoken_buf_get_str  (&pb, plan_quant,      sizeof(plan_quant));
    idletoken_buf_get_str  (&pb, plan_model_path, sizeof(plan_model_path));
    idletoken_buf_get_str  (&pb, prev_addr,       sizeof(prev_addr));
    idletoken_buf_get_str  (&pb, next_addr,       sizeof(next_addr));
    idletoken_buf_get_str  (&pb, coord_inbox,     sizeof(coord_inbox));
    if (pb.err) {
        fprintf(stderr, "idletoken-worker: ASSIGN_PLAN payload malformed\n");
        close(fd); return 1;
    }

    fprintf(stderr,
"idletoken-worker: ASSIGN_PLAN received\n"
"  cluster_size : %u\n"
"  stage_id     : %u  (segment_id=%u)\n"
"  mode         : %u  (1=GPU_ONLY, 2=HYBRID)\n"
"  model        : %s  (backend=%u, %u layers total)\n"
"  quant        : %s\n"
"  layer range  : [%u, %u)  (%u layers)\n"
"  ctx_size     : %u\n"
"  prefill_cap  : %u  (0 = worker picks)\n"
"  hc_dtype     : %u  (1=F32, 2=F16)\n"
"  model_path   : %s\n"
"  prev_stage   : %s\n"
"  next_stage   : %s\n"
"  coord_inbox  : %s\n",
            cluster_size, stage_id, segment_id, mode,
            plan_model_id[0] ? plan_model_id : "(empty)",
            plan_backend, plan_n_layers,
            plan_quant[0] ? plan_quant : "(default)",
            layer_lo, layer_hi, (unsigned)(layer_hi - layer_lo),
            plan_ctx_size, prefill_cap, hc_dtype,
            plan_model_path[0] ? plan_model_path : "(empty)",
            prev_addr[0] ? prev_addr : "(none — stage 0)",
            next_addr[0] ? next_addr : "(none — last stage)",
            coord_inbox[0] ? coord_inbox : "(empty)");

    /* sha256 hex dump for the operator to eyeball against the file. */
    fprintf(stderr, "  model_sha256 : ");
    for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", sha256[i]);
    fprintf(stderr, "\n");

    /* Sanity-check segment_id (v0.1 must be 0; coord should already enforce). */
    if (segment_id != 0) {
        fprintf(stderr, "idletoken-worker: segment_id=%u rejected (v0.1 SP=1 only)\n", segment_id);
        close(fd); return 1;
    }
    /* Resolve the model this plan names. An unknown id means the coordinator
     * runs a newer registry than this build — refuse loudly rather than load
     * the wrong weights. Backend gating: only ds4 (DSv4-Flash) is implemented
     * until Phase B lands the generic ds4x path. */
    g_model = idletoken_model_get(plan_model_id);
    if (!g_model) {
        fprintf(stderr, "idletoken-worker: unknown model id '%s' in ASSIGN_PLAN — "
                        "upgrade this worker\n", plan_model_id);
        close(fd); return 1;
    }
    if (plan_backend != g_model->backend) {
        fprintf(stderr, "idletoken-worker: ASSIGN_PLAN backend %u != registry backend %u "
                        "for '%s' — mismatched builds\n",
                (unsigned)plan_backend, (unsigned)g_model->backend, plan_model_id);
        close(fd); return 1;
    }
    if (plan_backend != IDLETOKEN_BACKEND_DS4 && plan_backend != IDLETOKEN_BACKEND_DS4X) {
        fprintf(stderr, "idletoken-worker: backend %u not implemented by this build\n",
                (unsigned)plan_backend);
        close(fd); return 1;
    }
    /* ds4x = the generic CPU MLA/GQA path (small models like Qwen3). It runs on
     * CPU (no CUDA kernel yet — that's a speed follow-up), so a ds4x worker
     * serves correctly, just slower. */
    const int use_ds4x = (plan_backend == IDLETOKEN_BACKEND_DS4X);
    if (plan_n_layers != g_model->n_layers) {
        fprintf(stderr, "idletoken-worker: plan says %u layers but registry says %u "
                        "for '%s' — mismatched builds\n",
                plan_n_layers, (unsigned)g_model->n_layers, plan_model_id);
        close(fd); return 1;
    }
    if (layer_hi <= layer_lo || layer_hi > plan_n_layers) {
        fprintf(stderr, "idletoken-worker: layer range [%u,%u) out of bounds\n", layer_lo, layer_hi);
        close(fd); return 1;
    }

    /* --- Load model + send LOAD_MODEL_DONE -------------------------------
     *
     * Two paths:
     *
     *   REAL  — GGUF file looks legit and `ds4_engine_open` + `ds4_session_create`
     *           succeed: this worker holds the real model, and the INFER loop
     *           below calls ds4_session_encode_layer_range / hc_tensor_* /
     *           logits_read.  This is the production path.
     *
     *   MOCK  — preflight fails (file missing/small) or engine_open fails:
     *           we still send LOAD_MODEL_DONE (ok=1, mock-only message) and
     *           the INFER loop runs zero-filled payloads.  Keeps the wire
     *           path smoke-testable without GGUF.
     *
     * The PP-range patch (main #6) means our `ds4_engine_open` per worker
     * loads the full model but encode_layer_range only touches `[lo,hi)`,
     * so per-worker VRAM still scales sub-linearly (KV cache is per-layer).
     * Memory optimization to alloc only `[lo,hi)` graph state is a TODO. */

    char resolved_model[1024];
    if (shard_repo && shard_repo[0]) {
        /* Layer-shard repo: this worker loads only its assigned range. A local
         * dir must already hold `L<lo>-<hi>.gguf` (materialized sparse partial);
         * an http(s) base is fetched by byte-range into the local cache. */
        if (!strncmp(shard_repo, "http://", 7) || !strncmp(shard_repo, "https://", 8)) {
            if (idletoken_shard_fetch(shard_repo, layer_lo, layer_hi, gguf_dir,
                                   resolved_model, sizeof(resolved_model)) != 0) {
                fprintf(stderr, "idletoken-worker: shard fetch failed from %s [%u,%u)\n",
                        shard_repo, layer_lo, layer_hi);
                close(fd); return 1;
            }
        } else {
            size_t dlen = strlen(shard_repo);
            const char *sep = (dlen > 0 && shard_repo[dlen-1] == '/') ? "" : "/";
            snprintf(resolved_model, sizeof(resolved_model), "%s%sL%u-%u.gguf",
                     shard_repo, sep, layer_lo, layer_hi);
        }
        fprintf(stderr, "idletoken-worker: shard-repo load [%u,%u) -> %s\n",
                layer_lo, layer_hi, resolved_model);
    } else {
        /* The coord-sent path may be absolute on the COORD's filesystem (e.g.
         * its tokenizer model path) — meaningless on a remote worker whose GGUF
         * lives elsewhere. Resolve by BASENAME under our own --gguf-dir first;
         * only if that isn't present fall back to the coord-sent path verbatim
         * (covers a coord and worker sharing one machine). Handles both '/' and
         * '\\' separators for cross-OS clusters. */
        const char *base = plan_model_path;
        for (const char *p = plan_model_path; *p; p++)
            if (*p == '/' || *p == '\\') base = p + 1;
        size_t dlen = strlen(gguf_dir);
        const char *sep = (dlen > 0 && (gguf_dir[dlen-1] == '/' ||
                                        gguf_dir[dlen-1] == '\\')) ? "" : "/";
        snprintf(resolved_model, sizeof(resolved_model), "%s%s%s",
                 gguf_dir, sep, base);
        struct stat stb;
        if (stat(resolved_model, &stb) != 0) {
            snprintf(resolved_model, sizeof(resolved_model), "%s", plan_model_path);
        }
    }

    uint8_t  load_ok         = 0;
    /* Fits in the LOAD_MODEL_DONE payload (done_payload[512]); messages embed
     * resolved_model with an explicit %.180s so they can never overflow it. */
    char     load_err[256]   = "";
    uint64_t vram_used_after = 0;
    uint64_t ram_used_after  = 0;
    uint32_t raw_cap         = 0;
    uint32_t comp_cap        = 0;

    ds4_engine  *engine  = NULL;
    ds4_session *session = NULL;
    /* v4 multi-sequence (scheduler-design §6-E2): one **independent** KV
     * sequence per seq_id. `session` / `xr` remain sequence 0 (historical
     * behaviour, warnings and the REAL/MOCK verdict all still go through them),
     * and the other slots are **created lazily on first use** -- never
     * preallocated, or a coordinator using a single slot would still pay N times
     * the KV memory.
     *
     * Known limitation: the VRAM headroom estimate in this file assumes **one**
     * sequence, and multiple slots exceed it. That is why `--seq-slots` defaults
     * to 1 on the coordinator side and is an explicit opt-in; when creation
     * fails we report honestly and fail the round rather than driving the
     * machine into OOM. */
    ds4_session *seq_sessions[IDLETOKEN_MAX_SEQ_SLOTS] = { NULL };
    ds4x_runner *seq_runners [IDLETOKEN_MAX_SEQ_SLOTS] = { NULL };
    /* ds4x (small-model CPU backend) handles; NULL unless use_ds4x + real load. */
    ds4x_model  *xm = NULL;
    ds4x_runner *xr = NULL;
    float       *xhidden = NULL;   /* [chunk_cap * n_embd] reused per step */

    struct stat st;
    int preflight_ok = 0;
    /* Minimum plausible file size: DSv4 Q2 is ~80GB; small ds4x models (Qwen3-8B
     * Q4_K_M ≈ 5GB, down to ~1GB at low quant) are far smaller — gate per
     * backend so a legit small GGUF isn't dismissed as a stub. */
    const uint64_t min_bytes = use_ds4x ? (256ull << 20) : (60ull << 30);
    if (stat(resolved_model, &st) != 0) {
        snprintf(load_err, sizeof(load_err),
                 "stat(%.180s): %s — running mock", resolved_model, strerror(errno));
    } else if (!S_ISREG(st.st_mode)) {
        snprintf(load_err, sizeof(load_err),
                 "%.180s not a regular file — running mock", resolved_model);
    } else if ((uint64_t)st.st_size < min_bytes) {
        snprintf(load_err, sizeof(load_err),
                 "%.180s is %llu bytes (< min) — running mock",
                 resolved_model, (unsigned long long)st.st_size);
    } else {
        preflight_ok = 1;
    }

    /* Model identity check (both backends). The coordinator sends sha256 of its
     * GGUF's metadata region; we hash ours and compare. All-zero = the coord
     * could not read its own copy, so there is nothing to check against.
     *
     * This replaces a `v0.1 skips verification` printf that had been there
     * since the field was added: the cluster would happily assemble stages from
     * DIFFERENT models or quant variants and only show it as garbage output.
     * A mismatch is a hard refusal — serving wrong weights is worse than not
     * serving (principle 9: better to go red on the spot). See
     * docs/review/findings.md R-06 for what
     * this covers (model/quant/layout) and what it does not (tensor-byte
     * corruption). */
    if (preflight_ok) {
        int sha_all_zero = 1;
        for (int i = 0; i < 32; i++) { if (sha256[i]) { sha_all_zero = 0; break; } }
        if (sha_all_zero) {
            fprintf(stderr, "idletoken-worker: coord sent no model identity — "
                            "skipping verification\n");
        } else {
            uint8_t mine[32];
            char ierr[256] = "";
            if (idletoken_gguf_identity(resolved_model, mine, ierr, sizeof ierr) != 0) {
                snprintf(load_err, sizeof(load_err),
                         "cannot hash %.120s for identity check (%.80s)", resolved_model, ierr);
                preflight_ok = 0;
            } else if (memcmp(mine, sha256, 32) != 0) {
                char a[17], b[17];
                for (int i = 0; i < 8; i++) {
                    snprintf(a + i * 2, 3, "%02x", sha256[i]);
                    snprintf(b + i * 2, 3, "%02x", mine[i]);
                }
                snprintf(load_err, sizeof(load_err),
                         "model identity mismatch: coord %s… vs local %s… (%.100s) "
                         "— wrong model or quant variant",
                         a, b, resolved_model);
                preflight_ok = 0;
                fprintf(stderr, "idletoken-worker: %s\n", load_err);
            } else {
                fprintf(stderr, "idletoken-worker: model identity verified against coord\n");
            }
        }
    }

    /* Auto-HYBRID: if this worker's assigned layers won't fit its usable VRAM
     * (leaving headroom for the CUDA context, KV cache and prefill batch
     * buffers), keep what fits in VRAM and spill the rest to pinned host RAM.
     * The CUDA DLL reads IDLETOKEN_HYBRID / IDLETOKEN_HYBRID_VRAM_GB; the worker sets
     * them automatically from its own probe + layer count — no manual config.
     * Unified-memory hosts (DGX) have a single pool and never trigger this. A
     * user-set IDLETOKEN_HYBRID is respected as an override. */
    /* Not under IDLETOKEN_DS4_CPU=1: the CPU backend never touches the CUDA
     * weight cache, so announcing "auto-HYBRID — N layers need X GiB" there
     * would describe a path the run is not taking. Misleading log lines cost
     * real debugging time (see the diagnostic notes on this file's CPU switch). */
    if (preflight_ok && !use_ds4x && !rr.unified_memory && !getenv("IDLETOKEN_HYBRID")
        && !(getenv("IDLETOKEN_DS4_CPU") && getenv("IDLETOKEN_DS4_CPU")[0] == '1')) {
        const double GiBd = 1073741824.0;
        uint32_t nlay = layer_hi - layer_lo;
        /* ~1.85 GiB per layer (80 GiB / 43) + ~1.6 GiB shared (embd + output). */
        double need_gib = nlay * 1.85 + 1.6;
        double vram_gib = (double)rr.vram_usable / GiBd;
        double headroom = 2.0;   /* CUDA ctx + KV + prefill batch buffers */
        double budget_gib = vram_gib > headroom ? vram_gib - headroom : 1.0;
        if (need_gib > budget_gib) {
            /* The expert cache and the resident weights compete for the same
             * VRAM, so this is the correct place to subtract it.
             *
             * **But the two machines cannot share one number**: lowering the
             * VRAM budget pushes more weights into pinned host memory, and the
             * pinned ceiling **differs per machine and cannot be derived**
             * (measured: 46.5 GiB on one host, 23.0 GiB on another, same GPU
             * model). Give both the same 2.3 GiB and the second host's budget
             * drops to 8 GiB, which then needs 22.9 GiB pinned -- grazing its
             * 23.0 GiB ceiling, failing, and taking the whole cluster with it.
             *
             * So how much the cache may take is derived from **this machine's
             * pinned-memory headroom**: work out the minimum VRAM this host
             * needs to stay under its pinned ceiling, and only what is left over
             * may go to the cache. */
            {
                double cache_gib = (double)worker_moe_cache_budget() / GiBd;
                if (cache_gib > 0.0) {
                    if (rr.ram_pinnable > 0) {
                        /* Whatever of the shard does not fit in VRAM must be pinned,
                         * hence VRAM >= shard - pinned ceiling. */
                        const double pin_gib = (double)rr.ram_pinnable / GiBd;
                        const double min_vram = need_gib - pin_gib + 1.0;   /* 1 GiB of slack */
                        const double room = budget_gib - (min_vram > 1.0 ? min_vram : 1.0);
                        if (room < cache_gib) cache_gib = room > 0.0 ? room : 0.0;
                    }
                    if (cache_gib > 0.0) {
                        budget_gib -= cache_gib;
                        if (budget_gib < 1.0) budget_gib = 1.0;
                    }
                    fprintf(stderr, "idletoken-worker: MoE expert cache gets %.2f GiB"
                                    " (derived from a %.1f GiB pinned ceiling), HYBRID VRAM budget %.1f GiB\n",
                            cache_gib, (double)rr.ram_pinnable / GiBd, budget_gib);
                    g_moe_cache_actual_bytes = (uint64_t)(cache_gib * GiBd);
                }
            }
            const int bg_gib = (int)budget_gib > 0 ? (int)budget_gib : 1;
            char bg[32];
            snprintf(bg, sizeof(bg), "%d", bg_gib);
            /* setenv is kept for the ds4x path and for anything that reads the
             * variable inside THIS binary, but it is NOT how ds4cuda.dll learns
             * the budget. This worker is built with MinGW and that DLL with
             * nvcc/MSVC: the two CRTs keep separate environment blocks, so the
             * DLL's getenv() never saw these. The budget was therefore inert on
             * Windows for as long as this code has existed — and it is a SAFETY
             * mechanism, not a tuning knob: without it cudaMalloc oversubscribes
             * VRAM and WDDM pages the overflow, fighting the whole machine for
             * memory. Measured with it dead: 3.46 s/token and two hard hangs;
             * with it applied: 0.49 s/token. Pass it by call. */
            setenv("IDLETOKEN_HYBRID", "1", 1);
            setenv("IDLETOKEN_HYBRID_VRAM_GB", bg, 1);
#ifndef DS4_NO_GPU
            ds4_gpu_set_hybrid_vram_budget((uint64_t)bg_gib << 30);
#endif
            fprintf(stderr,
                    "idletoken-worker: auto-HYBRID — %u layers need ~%.1f GiB > %.1f GiB "
                    "usable VRAM; keeping %s GiB in VRAM, overflow to host RAM\n",
                    nlay, need_gib, vram_gib, bg);
        }
    }

#ifndef DS4_NO_GPU
    /* This must come **after** auto-HYBRID: that section is where the cache
     * allowance gets trimmed to this machine's pinned-memory headroom. The first
     * version ran before it and always read 0. */
    {
        /* bytes=0 lets the DLL size the cache from **the VRAM this machine has
         * free right now**. Headroom differs a lot between machines (different
         * layer counts, different resident weights), so one shared number is
         * guaranteed to push one of them off a cliff: at 16 slots one host saw
         * gpuexec blow up 5x while the other, in the same run, was untouched. */
        const uint64_t cb = g_moe_cache_actual_bytes;   /* 0 = automatic */
        const uint32_t nlay_c = (layer_hi > layer_lo) ? (uint32_t)(layer_hi - layer_lo) : 0u;
        if (nlay_c > 0 && !use_ds4x) ds4_gpu_set_moe_cache(cb, nlay_c);
    }
#endif

#ifdef IDLETOKEN_DS4X_CUDA
    /* Give ds4x the same VRAM discipline the ds4 path gets from auto-HYBRID
     * above: weights may occupy usable VRAM minus headroom for the CUDA
     * context, KV cache and prefill buffers. `rr.vram_usable` already has the
     * user's --max-vram-mb / settings-panel cap applied (resource.c
     * idletoken_resource_apply_caps), so this is the line that turns that setting
     * into an actual limit instead of a smaller number in a report. Anything
     * that does not fit stays on the CPU — slower, never wrong. */
    if (preflight_ok && use_ds4x) {
        const uint64_t headroom = 2ull << 30;   /* same 2 GiB as auto-HYBRID */
        uint64_t budget = rr.vram_usable > headroom ? rr.vram_usable - headroom
                                                    : rr.vram_usable / 2;
        ds4x_cuda_set_budget(budget);
        fprintf(stderr, "idletoken-worker: ds4x VRAM budget %.2f GiB "
                        "(usable %.2f GiB - %.1f GiB headroom)\n",
                (double)budget / 1073741824.0,
                (double)rr.vram_usable / 1073741824.0,
                (double)headroom / 1073741824.0);
    }
#endif

    if (preflight_ok && use_ds4x) {
        /* ds4x (small models): load the generic GGUF backend and create the PP
         * runner over this worker's [layer_lo,layer_hi). ds4x_model_load
         * validates arch/dims vs the GGUF and, on a CUDA build with a usable
         * device, uploads the big projections to VRAM within the budget set
         * just above (it prints which of the two it ended up doing). */
        char xerr[256] = "";
        xm = ds4x_model_load(resolved_model, layer_lo, layer_hi, xerr, sizeof(xerr));
        if (!xm) {
            snprintf(load_err, sizeof(load_err), "ds4x load failed: %.180s — running mock", xerr);
        } else {
            xr = ds4x_runner_create(xm, plan_ctx_size, xerr, sizeof(xerr));
            if (!xr) {
                snprintf(load_err, sizeof(load_err), "ds4x runner failed: %.180s — running mock", xerr);
                ds4x_model_free(xm); xm = NULL;
            } else {
                seq_runners[0] = xr;   /* seq 0 = the historical single sequence */
                idletoken_resource_report rr2;
                if (idletoken_resource_probe(&rr2, gguf_dir) == 0) {
                    vram_used_after = (rr2.vram_total > rr2.vram_usable) ? (rr2.vram_total - rr2.vram_usable) : 0;
                    ram_used_after  = (rr2.ram_total  > rr2.ram_usable)  ? (rr2.ram_total  - rr2.ram_usable)  : 0;
                }
                /* Don't claim a compute backend here — whether the GPU matvec
                 * is active is decided inside ds4x_model_load and printed by
                 * it ("ds4x: CUDA on ..."). Hard-coding "CPU" here was wrong
                 * as soon as the CUDA path landed. */
                snprintf(load_err, sizeof(load_err), "real ds4x backend ready (ctx=%u)", plan_ctx_size);
            }
        }
    } else if (preflight_ok) {
        /* Try real load. (Identity was verified above, for both backends.) */

        /* ds4 uses /tmp/ds4.lock by default to prevent two ds4 processes
         * stomping each other on a single-machine setup.  For our cluster,
         * multiple workers on the same host is a normal case (also useful
         * for testing).  Give each worker its own lock file keyed by PID.
         * `DS4_LOCK_FILE` is read in ds4_acquire_instance_lock() at engine
         * open time. */
        char ds4_lock_path[512];
#ifdef _WIN32
        /* No /tmp on Windows — use %TEMP% (always set) so the lock file can
         * actually be created; a bad path would abort ds4_engine_open. */
        const char *tmpdir = getenv("TEMP");
        if (!tmpdir || !tmpdir[0]) tmpdir = ".";
        snprintf(ds4_lock_path, sizeof(ds4_lock_path),
                 "%s\\ds4-%ld.lock", tmpdir, (long)getpid());
#else
        snprintf(ds4_lock_path, sizeof(ds4_lock_path),
                 "/tmp/ds4-%ld.lock", (long)getpid());
#endif
        setenv("DS4_LOCK_FILE", ds4_lock_path, 1);

        /* DSpark only belongs on the LAST stage: the drafter consumes the
         * hidden states of the final layers, which exist nowhere else. Loading
         * it on any other stage would burn 5.6 GiB for a module that can never
         * run — so the placement rule is enforced here, not left to whoever
         * writes the launch command. */
        const bool is_last_stage = (layer_hi == plan_n_layers);
        const char *eo_dspark = (dspark_path && dspark_path[0] && is_last_stage)
                              ? dspark_path : NULL;
        if (dspark_path && dspark_path[0] && !is_last_stage) {
            fprintf(stderr,
                    "idletoken-worker: DSpark module not loaded on this stage "
                    "(layers [%u,%u) of %u; the drafter needs the last layers)\n",
                    layer_lo, layer_hi, plan_n_layers);
        }
        /* IDLETOKEN_DS4_CPU=1 forces ds4's CPU backend for this worker.
         *
         * Why this exists (2026-08-05): our HYBRID path is "GPU computes
         * everything; weights past the VRAM budget live in pinned MAPPED host
         * memory that the GPU reads over PCIe" (ds4_cuda.cu). Measured on two
         * Windows nodes that is ~30 s/token with the GPU sitting at 0%
         * utilisation — the SMs stall on fine-grained PCIe reads. llama.cpp's
         * partial offload is fast for the opposite reason: the layers it cannot
         * fit are COMPUTED ON THE CPU at memory bandwidth, with no per-token
         * PCIe traffic. ds4 already has the multithreaded CPU kernels
         * (matmul_q8_0_* over ds4_parallel_for); what it lacks is per-layer
         * backend dispatch. Before paying for that, this switch lets us measure
         * what the CPU path actually does on real hardware — the number that
         * decides whether the refactor is worth it. Diagnostic, not a product
         * setting. */
        const char *cpu_env = getenv("IDLETOKEN_DS4_CPU");
        const int force_cpu = (cpu_env && cpu_env[0] == '1');
        if (force_cpu) {
            fprintf(stderr, "idletoken-worker: IDLETOKEN_DS4_CPU=1 — ds4 CPU backend "
                            "(diagnostic A/B against the PCIe-mapped HYBRID path)\n");
        }
        ds4_engine_options eo = {
            .model_path = resolved_model,
            .mtp_path = NULL,
            .dspark_path = eo_dspark,
            .backend = force_cpu ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA,
            .n_threads = 0,                /* ds4 picks */
            .mtp_draft_tokens = 0,
            .mtp_margin = 0.0f,
            .directional_steering_file = NULL,
            .directional_steering_attn = 0.0f,
            .directional_steering_ffn  = 0.0f,
            .warm_weights = false,
            .quality = false,
            /* IdleToken PP: only prefetch this worker's layer range. Non-layer
             * tensors (token_embd, output_*, mtp.0.*) load regardless. */
            .load_layer_lo = (int)layer_lo,
            .load_layer_hi = (int)layer_hi,
        };
        fprintf(stderr,
                "idletoken-worker: opening ds4 engine at %s (CUDA backend, layers [%u,%u))...\n",
                resolved_model, layer_lo, layer_hi);
        time_t t0 = time(NULL);
        int rc = ds4_engine_open(&engine, &eo);
        if (rc != 0 || !engine) {
            snprintf(load_err, sizeof(load_err),
                     "ds4_engine_open failed (rc=%d) — running mock", rc);
            engine = NULL;
        } else {
            fprintf(stderr, "idletoken-worker: ds4_engine_open OK in %lds\n",
                    (long)(time(NULL) - t0));
            rc = ds4_session_create(&session, engine, (int)plan_ctx_size);
            if (rc != 0 || !session) {
                snprintf(load_err, sizeof(load_err),
                         "ds4_session_create failed (rc=%d) — running mock", rc);
                ds4_engine_close(engine);
                engine = NULL;
                session = NULL;
            } else {
                seq_sessions[0] = session;   /* seq 0 = the historical single sequence */
                /* Re-probe to report post-load memory. */
                idletoken_resource_report rr2;
                if (idletoken_resource_probe(&rr2, gguf_dir) == 0) {
                    vram_used_after = (rr2.vram_total > rr2.vram_usable)
                                      ? (rr2.vram_total - rr2.vram_usable)
                                      : 0;
                    ram_used_after  = (rr2.ram_total  > rr2.ram_usable)
                                      ? (rr2.ram_total  - rr2.ram_usable)
                                      : 0;
                }
                snprintf(load_err, sizeof(load_err),
                         "real ds4 engine + session ready (ctx=%u)",
                         plan_ctx_size);
            }
        }
    }
    /* Mock is a smoke-test channel, NOT a fallback for a model we were told to
     * load. Reporting ok=1 while serving zeros is how a cluster comes up
     * "ready" and then answers garbage — it cost two full debug rounds today
     * (a stale CUDA DLL, then a truncated 28 GB copy of an 80 GB GGUF). When a
     * model path was given and we could not load it, say so: the coordinator
     * already knows how to abort loudly on a failed worker.
     *
     * Mock stays legitimate when there is nothing to load at all (no model path
     * in the plan — how the platform e2e smoke-tests the wire path), or when a
     * test asks for it explicitly. */
    {
        const int real_engine  = (session != NULL) || (xr != NULL);
        const int nothing_todo = (resolved_model[0] == '\0');
        const int mock_allowed = nothing_todo || getenv("IDLETOKEN_ALLOW_MOCK") != NULL;
        load_ok = (real_engine || mock_allowed) ? 1 : 0;
        if (!load_ok) {
            fprintf(stderr,
                    "idletoken-worker: refusing to serve — %s was requested but could "
                    "not be loaded (%s). Set IDLETOKEN_ALLOW_MOCK=1 only if you want a "
                    "zero-output smoke test.\n", resolved_model, load_err);
            /* That actionable hint has to travel **back to the coordinator
             * inside load_err**, or it only ever lives in this process's stderr.
             * The coordinator merely says "N workers failed to load" and
             * forwards load_err, so the real cause sits two log files away from
             * the user: the gate says "coord API never became healthy", the
             * coordinator says "1 worker failed to load model", and the line
             * that actually matters is in a third file. Diagnosing it that way
             * once was expensive enough. */
            const size_t used = strlen(load_err);
            if (used + 1 < sizeof(load_err))
                snprintf(load_err + used, sizeof(load_err) - used,
                         " [worker refuses to serve from mock; set IDLETOKEN_ALLOW_MOCK=1 if you really only want to exercise the path]");
        }
    }

    /* REAL covers both engines: a ds4 session OR a loaded ds4x runner. Saying
     * MOCK while ds4x is actually serving is the single most misleading log we
     * can emit (grep "load REAL|MOCK" is the first triage step). */
    fprintf(stderr, "idletoken-worker: model load %s — %s\n",
            (session || xr) ? "REAL" : "MOCK", load_err);

    /* Pack LOAD_MODEL_DONE payload per wire-protocol.md §LOAD_MODEL_DONE. */
    uint8_t done_payload[512];
    idletoken_buf db;
    idletoken_buf_init(&db, done_payload, sizeof(done_payload));
    idletoken_buf_put_u8 (&db, load_ok);
    uint8_t pad7_out[7] = {0};
    idletoken_buf_put_bytes(&db, pad7_out, 7);
    idletoken_buf_put_u64(&db, vram_used_after);
    idletoken_buf_put_u64(&db, ram_used_after);
    idletoken_buf_put_u32(&db, raw_cap);
    idletoken_buf_put_u32(&db, comp_cap);
    idletoken_buf_put_str(&db, load_err);
    if (db.err) {
        fprintf(stderr, "idletoken-worker: LOAD_MODEL_DONE pack overflow\n");
        close(fd); return 1;
    }

    idletoken_msg_header done_hdr = {
        .magic         = IDLETOKEN_PROTO_MAGIC,
        .version       = IDLETOKEN_PROTO_VERSION,
        .msg_type      = IDLETOKEN_MSG_LOAD_MODEL_DONE,
        .payload_bytes = db.pos,
        .request_id    = plan_hdr.request_id,
        .stage_id      = stage_id,
        .segment_id    = IDLETOKEN_SEGMENT_NONE,
    };
    if (idletoken_send_msg(fd, &done_hdr, done_payload, db.pos) != 0) {
        fprintf(stderr, "idletoken-worker: send LOAD_MODEL_DONE: %s\n", strerror(errno));
        close(fd); return 1;
    }
    fprintf(stderr, "idletoken-worker: sent LOAD_MODEL_DONE (ok=%u, %zu B payload)\n",
            load_ok, db.pos);

    if (!load_ok) {
        close(fd);
        return 1;
    }

    /* --- INFER event loop -------------------------------------------------
     *
     * v0.1 PP topology:
     *   coord  --INFER_BEGIN-->         stage 0
     *   stage k  --INFER_HC_FORWARD-->  stage k+1
     *   last stage  --INFER_LOGITS-->   coord
     * (Since v5 there is no return TOKEN_ACK: it was a pure barrier -- see the
     *  version notes in idletoken_proto.h.)
     *
     * Each non-first worker listens on its own bind_addr for the prev stage's
     * HC connection.  Each non-last worker connects to next_stage_addr to ship
     * HC.  The control connection (`fd`) to coord stays open for
     * INFER_BEGIN / INFER_LOGITS.
     *
     * Mock mode: ds4_session_* calls (encode / hc_tensor_read|write /
     * logits_read) are stubbed.  HC payloads are zero-filled, logits are
     * zero-filled.  Real ds4 calls land after engine_open is wired
     * (requires GGUF + main #6 + main #7 patch 0001 from scripts/ds4-patches).
     */

    int is_stage0   = (stage_id == 0);
    int is_last     = (stage_id == cluster_size - 1);
    int hc_in_fd    = -1;
    int hc_out_fd   = -1;
    int loop_rc     = 0;

    /* Open inter-stage HC links. Listener first (non-first stages), then
     * connector (non-last stages) with a short retry loop to absorb startup
     * race against the next stage's listen(). */
    if (!is_stage0) {
        int hc_listener = idletoken_listen_tcp(bind_addr);
        if (hc_listener < 0) {
            fprintf(stderr, "idletoken-worker: listen(%s) for HC: %s\n",
                    bind_addr, strerror(errno));
            close(fd); return 1;
        }
        fprintf(stderr, "idletoken-worker: listening on %s for HC from prev stage\n", bind_addr);
        hc_in_fd = idletoken_accept_tcp(hc_listener);
        close(hc_listener);
        if (hc_in_fd < 0) {
            fprintf(stderr, "idletoken-worker: accept HC: %s\n", strerror(errno));
            close(fd); return 1;
        }
        fprintf(stderr, "idletoken-worker: HC link accepted from prev stage\n");
    }
    if (!is_last) {
        /* The next stage may still be loading — and with layer-shard fetch it
         * can pull tens of GB before it listens. Retry on ANY transient
         * failure until a wall-clock deadline, not just ECONNREFUSED: a
         * not-yet-listening port refuses on Linux but a Windows host silently
         * drops SYNs (firewall / no listener) so the connect TIMES OUT instead.
         * Bounding by wall clock (not iteration count) keeps this correct even
         * though each timed-out connect can itself block for many seconds. */
        time_t hc_deadline = time(NULL) + 900;  /* up to 15 min for a slow next stage */
        int announced = 0;
        for (;;) {
            hc_out_fd = idletoken_connect_tcp(next_addr);
            if (hc_out_fd >= 0) break;
            if (time(NULL) >= hc_deadline) break;
            if (!announced) {
                fprintf(stderr, "idletoken-worker: next stage %s not ready (%s); "
                        "retrying until it listens...\n", next_addr, strerror(errno));
                announced = 1;
            }
            usleep(200000); /* 200 ms between attempts (connect itself may block) */
        }
        if (hc_out_fd < 0) {
            fprintf(stderr, "idletoken-worker: connect HC to %s: %s (gave up after 15 min)\n",
                    next_addr, strerror(errno));
            if (hc_in_fd >= 0) close(hc_in_fd);
            close(fd); return 1;
        }
        fprintf(stderr, "idletoken-worker: HC link to next stage %s open\n", next_addr);
    }

    /* Buffer big enough for any v0.1 INFER message. Decode-step extremes:
     *   INFER_HC_FORWARD F32  : 16 + 1 * N_HC * N_EMBD * 4   = 64 KiB + 16 B
     *   INFER_LOGITS          : 8  + DS4_N_VOCAB * 4         = 505 KiB + 8 B
     * Prefill chunks are the big case: chunk_cap tokens × 64 KiB F32 HC rows
     * + the chunk token ids appended for the next stage (the MoE router is
     * token-aware). chunk_cap mirrors official ds4's prefill chunking
     * (≤ 2048), so worst case ≈ 128 MiB host-side. */
    const uint32_t chunk_cap = ds4_prefill_chunk_cap_for_ctx((int)plan_ctx_size);
    size_t msg_buf_cap = 1024 * 1024;
    {
        const size_t prefill_need =
            16 + (size_t)chunk_cap * (size_t)g_model->hc_streams * g_model->n_embd * 4
               + 4 * (size_t)chunk_cap + 64;
        if (prefill_need > msg_buf_cap) msg_buf_cap = prefill_need;
    }
    uint8_t *msg_buf = malloc(msg_buf_cap);
    int *chunk_tokens = malloc((size_t)(chunk_cap ? chunk_cap : 1) * sizeof(int));
    /* ds4x hidden-state working buffer: [chunk_cap][n_embd] fp32. It holds the
     * tensor that crosses the PP seam (== the HC payload when hc_streams=1). */
    if (xr) xhidden = malloc((size_t)(chunk_cap ? chunk_cap : 1) * g_model->n_embd * sizeof(float));
    if (!msg_buf || !chunk_tokens || (xr && !xhidden)) {
        fprintf(stderr, "idletoken-worker: msg buf malloc failed\n");
        free(msg_buf); free(chunk_tokens); free(xhidden);
        if (hc_in_fd  >= 0) close(hc_in_fd);
        if (hc_out_fd >= 0) close(hc_out_fd);
        close(fd); return 1;
    }

    /* Multi-step decode loop: each iteration handles one INFER_BEGIN → HC
     * chain → INFER_LOGITS. Coord drives `--n-predict` and
     * closes the control connection when done; worker detects EOF and exits
     * cleanly. */
    int steps_done = 0;

    uint64_t last_prof_at = 0;
    while (loop_rc == 0) {
        idletoken_msg_header in_hdr;
        uint8_t phase    = 0;
        uint8_t is_first = 0;   /* is_first_chunk: a new sequence restarts at pos0 (rewind signal) */
        uint8_t seq_id   = 0;   /* v4: whose KV this round reads and writes (0 = the historical one) */
        ds4_session *cur_session = NULL;   /* KV holder for this round's sequence (filled by seq_resolve) */
        ds4x_runner *cur_xr      = NULL;
        uint32_t pos0    = 0;
        uint32_t n_tokens = 1;
        uint32_t in_token = 0;
        uint64_t req_id  = 0;
        double   round_wait = 0.0;   /* this round's idle time; classified once n_tokens is known */

        /* Block waiting for the next unit of work. Both branches feed the same
         * counter: stage 0 waits for the coordinator's INFER_BEGIN, the others
         * wait for HC_FORWARD from upstream, and both mean "this stage is
         * idle". */
        const double t_wait0 = now_monotonic_s();
        if (is_stage0) {
            if (idletoken_recv_msg(fd, &in_hdr, msg_buf, msg_buf_cap) != 0) {
                if (errno == ECONNRESET) {
                    fprintf(stderr, "idletoken-worker: coord closed control conn — clean shutdown\n");
                    break;
                }
                fprintf(stderr, "idletoken-worker: recv INFER_BEGIN: %s\n", strerror(errno));
                loop_rc = 1; break;
            }
            round_wait = now_monotonic_s() - t_wait0;
            if (in_hdr.msg_type != IDLETOKEN_MSG_INFER_BEGIN) {
                fprintf(stderr, "idletoken-worker: expected INFER_BEGIN got 0x%04x\n",
                        in_hdr.msg_type);
                loop_rc = 1; break;
            }
            idletoken_buf pb2;
            idletoken_buf_init(&pb2, msg_buf, in_hdr.payload_bytes);
            uint8_t is_last_chunk = 0, wire_seq = 0;
            uint32_t _rp2;
            idletoken_buf_get_u8 (&pb2, &phase);
            idletoken_buf_get_u8 (&pb2, &is_first);
            idletoken_buf_get_u8 (&pb2, &is_last_chunk);
            idletoken_buf_get_u8 (&pb2, &wire_seq);   /* v4: sequence slot */
            idletoken_buf_get_u32(&pb2, &pos0);
            idletoken_buf_get_u32(&pb2, &n_tokens);
            idletoken_buf_get_u32(&pb2, &_rp2);
            if (wire_seq >= IDLETOKEN_MAX_SEQ_SLOTS) {
                fprintf(stderr, "idletoken-worker: INFER_BEGIN seq_id=%u out of range (max %d)\n",
                        wire_seq, IDLETOKEN_MAX_SEQ_SLOTS - 1);
                loop_rc = 1; break;
            }
            seq_id = wire_seq;
            if (seq_resolve(seq_id, seq_sessions, seq_runners, engine, xm, plan_ctx_size,
                            session != NULL, xr != NULL, &cur_session, &cur_xr) != 0) {
                loop_rc = 1; break;
            }
            if (n_tokens < 1 || n_tokens > chunk_cap) {
                fprintf(stderr, "idletoken-worker: INFER_BEGIN n_tokens=%u out of range (cap %u)\n",
                        n_tokens, chunk_cap);
                loop_rc = 1; break;
            }
            for (uint32_t ti = 0; ti < n_tokens; ti++) {
                uint32_t t = 0;
                idletoken_buf_get_u32(&pb2, &t);
                chunk_tokens[ti] = (int)t;
            }
            in_token = (uint32_t)chunk_tokens[0];
            if (pb2.err) {
                fprintf(stderr, "idletoken-worker: INFER_BEGIN payload malformed\n");
                loop_rc = 1; break;
            }
            /* ds4x stage 0: embed the chunk tokens into the hidden buffer that
             * the runner will transform (int==int32 on all targets). */
            if (xr) {
                if (ds4x_embed_tokens(xm, (const int32_t *)chunk_tokens, n_tokens, xhidden) != 0) {
                    fprintf(stderr, "idletoken-worker: ds4x_embed_tokens failed\n");
                    loop_rc = 1; break;
                }
            }
            req_id = in_hdr.request_id;
            fprintf(stderr,
                    "idletoken-worker: INFER_BEGIN phase=%u pos0=%u n_tokens=%u token0=%u "
                    "(is_first=%u is_last=%u)\n",
                    phase, pos0, n_tokens, in_token, is_first, is_last_chunk);
        } else {
            if (idletoken_recv_msg(hc_in_fd, &in_hdr, msg_buf, msg_buf_cap) != 0) {
                if (errno == ECONNRESET) {
                    fprintf(stderr, "idletoken-worker: prev stage closed HC — clean shutdown\n");
                    break;
                }
                fprintf(stderr, "idletoken-worker: recv INFER_HC_FORWARD: %s\n", strerror(errno));
                loop_rc = 1; break;
            }
            round_wait = now_monotonic_s() - t_wait0;
            if (in_hdr.msg_type != IDLETOKEN_MSG_INFER_HC_FORWARD) {
                fprintf(stderr, "idletoken-worker: expected HC_FORWARD got 0x%04x\n",
                        in_hdr.msg_type);
                loop_rc = 1; break;
            }
            idletoken_buf pb2;
            idletoken_buf_init(&pb2, msg_buf, in_hdr.payload_bytes);
            uint8_t in_dtype = 0, _rp[2];
            uint32_t in_hc_bytes = 0;
            idletoken_buf_get_u8   (&pb2, &phase);
            idletoken_buf_get_u8   (&pb2, &in_dtype);
            idletoken_buf_get_bytes(&pb2, _rp, 2);
            /* Reserved byte [0] = is_first_chunk, propagated by the upstream
             * stage. An older build sends 0, meaning no rewind, which matches
             * pre-upgrade behaviour -- reserved fields are backward compatible
             * by construction. */
            is_first = _rp[0];
            /* Reserved byte [1] = seq_id (v4). An older build sends 0 and lands
             * on sequence 0, as it did before the upgrade. */
            if (_rp[1] >= IDLETOKEN_MAX_SEQ_SLOTS) {
                fprintf(stderr, "idletoken-worker: HC_FORWARD seq_id=%u out of range\n", _rp[1]);
                loop_rc = 1; break;
            }
            seq_id = _rp[1];
            if (seq_resolve(seq_id, seq_sessions, seq_runners, engine, xm, plan_ctx_size,
                            session != NULL, xr != NULL, &cur_session, &cur_xr) != 0) {
                loop_rc = 1; break;
            }
            idletoken_buf_get_u32  (&pb2, &pos0);
            idletoken_buf_get_u32  (&pb2, &n_tokens);
            idletoken_buf_get_u32  (&pb2, &in_hc_bytes);
            if (pb2.err) {
                fprintf(stderr, "idletoken-worker: HC_FORWARD payload malformed\n");
                loop_rc = 1; break;
            }
            req_id = in_hdr.request_id;
            fprintf(stderr,
                    "idletoken-worker: INFER_HC_FORWARD phase=%u pos0=%u n_tokens=%u hc_bytes=%u\n",
                    phase, pos0, n_tokens, in_hc_bytes);
            if (n_tokens < 1 || n_tokens > chunk_cap) {
                fprintf(stderr, "idletoken-worker: HC_FORWARD n_tokens=%u out of range (cap %u)\n",
                        n_tokens, chunk_cap);
                loop_rc = 1; break;
            }
            if (n_tokens > 1) {
                /* Prefill chunk: batch HC rows + the chunk token ids appended
                 * after the HC data (MoE router is token-aware). */
                const uint64_t want_hc =
                    (uint64_t)n_tokens * (size_t)g_model->hc_streams * g_model->n_embd * 4;
                if (in_hc_bytes != want_hc ||
                    (uint64_t)16 + in_hc_bytes + 4ull * n_tokens > in_hdr.payload_bytes) {
                    fprintf(stderr,
                            "idletoken-worker: HC_FORWARD prefill payload malformed "
                            "(hc_bytes=%u want=%llu payload=%llu)\n",
                            in_hc_bytes, (unsigned long long)want_hc,
                            (unsigned long long)in_hdr.payload_bytes);
                    loop_rc = 1; break;
                }
                const uint8_t *tp = msg_buf + 16 + in_hc_bytes;
                for (uint32_t ti = 0; ti < n_tokens; ti++) {
                    chunk_tokens[ti] = (int)((uint32_t)tp[4*ti] |
                                             ((uint32_t)tp[4*ti+1] << 8) |
                                             ((uint32_t)tp[4*ti+2] << 16) |
                                             ((uint32_t)tp[4*ti+3] << 24));
                }
                in_token = (uint32_t)chunk_tokens[0];
                if (cur_session) {
                    if (!ds4_session_batch_hc_write(cur_session, msg_buf + 16, n_tokens)) {
                        fprintf(stderr, "idletoken-worker: ds4_session_batch_hc_write failed\n");
                        loop_rc = 1; break;
                    }
                }
            } else if (cur_session && in_hc_bytes > 0) {
                if (!ds4_session_hc_tensor_write(cur_session, msg_buf + 16, in_hc_bytes)) {
                    fprintf(stderr, "idletoken-worker: ds4_session_hc_tensor_write failed\n");
                    loop_rc = 1; break;
                }
            }
            /* ds4x: the incoming HC payload IS the hidden tensor (hc_streams=1);
             * copy it into the runner's working buffer to transform in place. */
            if (cur_xr) {
                const size_t want = (size_t)n_tokens * g_model->n_embd * sizeof(float);
                if (in_hc_bytes < want) {
                    fprintf(stderr, "idletoken-worker: ds4x HC underrun (%u < %zu)\n", in_hc_bytes, want);
                    loop_rc = 1; break;
                }
                memcpy(xhidden, msg_buf + 16, want);
            }
        }

        /* Real encode if engine loaded; otherwise mock path returns zeros.
         * n_tokens > 1 → prefill chunk through ds4's batched kernels (same
         * numerics as official single-machine prefill); n_tokens == 1 →
         * incremental decode. */
        if (cur_session) {
            /* A new sequence (the coordinator saw a KV miss and restarted from
             * zero): rewind to pos0 first and clear this shard's rolling
             * compression state. Blindly writing after the previous session's KV
             * would make the compressed layers append the wrong pooled rows (KV
             * prefix reuse design, docs/kv-cache-design.md §A). */
            if (is_first && phase == 1 && pos0 == 0) {
                ds4_session_rewind(cur_session, 0);
                fprintf(stderr, "idletoken-worker: is_first_chunk -> session rewind(0) [seq %u]\n", seq_id);
            }
            bool enc_ok;
            const double t_enc0 = now_monotonic_s();
            if (n_tokens > 1) {
                enc_ok = ds4_session_prefill_layer_range(cur_session, chunk_tokens,
                                                         n_tokens, pos0,
                                                         layer_lo, layer_hi);
            } else {
                enc_ok = ds4_session_encode_layer_range(cur_session, (int)in_token,
                                                        pos0, layer_lo, layer_hi);
            }
            /* The ds4 path had no timing at all: PROF only wrapped the ds4x
             * branch, and DSv4 runs through here. So when DSv4 ran across
             * machines, per-stage cost was invisible and optimization was
             * guesswork. */
            const double t_enc = now_monotonic_s() - t_enc0;
            if (n_tokens > 1) { g_prefill_s += t_enc; g_prefill_calls++; g_pf_other_s += round_wait; }
            else              { g_compute_s += t_enc; g_compute_calls++; g_wait_s += round_wait; g_wait_calls++; }
            if (!enc_ok) {
                fprintf(stderr,
                        "idletoken-worker: ds4 %s(%u,%u) failed\n",
                        n_tokens > 1 ? "prefill_layer_range" : "encode_layer_range",
                        layer_lo, layer_hi);
                loop_rc = 1; break;
            }
        } else if (cur_xr) {
            /* ds4x: transform xhidden in place through this stage's layers,
             * appending KV at absolute positions pos0..pos0+n_tokens-1. A new
             * sequence (pos0==0) overwrites the cache from the start — no
             * explicit rewind needed (the runner indexes KV by absolute pos). */
            const double t_run0 = now_monotonic_s();
            if (ds4x_runner_run(cur_xr, xhidden, n_tokens, pos0) != 0) {
                fprintf(stderr, "idletoken-worker: ds4x_runner_run(pos0=%u,n=%u) failed: %s\n",
                        pos0, n_tokens, ds4x_runner_last_error());
                loop_rc = 1; break;
            }
            /* How long each stage computed for. This is the key input to E3:
             * how balanced the stages are decides how much filling the pipeline
             * can buy. Previously only the single-machine tool ds4x_infer
             * honoured IDLETOKEN_DS4X_PROF, so per-stage cost inside a cluster
             * was invisible. */
            g_compute_s += now_monotonic_s() - t_run0;
            g_compute_calls++;
        }
        (void)in_token;

        /* Produce output. Last stage → LOGITS to coord. Else → HC_FORWARD. */
        const uint64_t hc_elem_bytes = (hc_dtype == 1) ? 4ull : 2ull;
        if (n_tokens > 1 && hc_dtype != 1) {
            fprintf(stderr, "idletoken-worker: prefill chunks require F32 HC (v0.1)\n");
            loop_rc = 1; break;
        }
        const uint64_t hc_bytes      =
            (uint64_t)n_tokens * (size_t)g_model->hc_streams * g_model->n_embd * hc_elem_bytes;

        if (is_last) {
            /* Send INFER_LOGITS to coord (zero-filled in mock). */
            const size_t logits_bytes = g_model->n_vocab * sizeof(float);
            const size_t logits_payload_bytes = 8 + logits_bytes;
            if (logits_payload_bytes > msg_buf_cap) {
                fprintf(stderr, "idletoken-worker: logits payload too large\n");
                loop_rc = 1; break;
            }
            idletoken_buf lb;
            idletoken_buf_init(&lb, msg_buf, msg_buf_cap);
            idletoken_buf_put_u32(&lb, pos0);
            idletoken_buf_put_u32(&lb, g_model->n_vocab);
            if (cur_session) {
                if (n_tokens == 1) {
#ifndef DS4_NO_GPU
                    const double t_s0 = now_monotonic_s();
                    (void)ds4_gpu_synchronize();
                    g_gpuexec_s += now_monotonic_s() - t_s0;
                    g_gpuexec_calls++;
#endif
                }
                const double t_h0 = now_monotonic_s();
                if (!ds4_session_logits_read(cur_session, (float *)(msg_buf + 8))) {
                    fprintf(stderr, "idletoken-worker: ds4_session_logits_read failed\n");
                    loop_rc = 1; break;
                }
                if (n_tokens > 1) { g_pf_other_s += now_monotonic_s() - t_h0; }
                else { g_harvest_s += now_monotonic_s() - t_h0; g_harvest_calls++; }
            } else if (cur_xr) {
                /* last stage: project the final token's hidden → vocab logits */
                const double t_lg0 = now_monotonic_s();
                if (ds4x_output_logits(xm, xhidden + (size_t)(n_tokens - 1) * g_model->n_embd,
                                       (float *)(msg_buf + 8)) != 0) {
                    fprintf(stderr, "idletoken-worker: ds4x_output_logits failed\n");
                    loop_rc = 1; break;
                }
                g_logits_s += now_monotonic_s() - t_lg0;
                g_logits_calls++;
            } else {
                memset(msg_buf + 8, 0, logits_bytes);  /* mock: zeros */
            }
            lb.pos = logits_payload_bytes;
            idletoken_msg_header lh = {
                .magic = IDLETOKEN_PROTO_MAGIC,
                .version = IDLETOKEN_PROTO_VERSION,
                .msg_type = IDLETOKEN_MSG_INFER_LOGITS,
                .payload_bytes = lb.pos,
                .request_id = req_id,
                .stage_id = stage_id,
                .segment_id = IDLETOKEN_SEGMENT_NONE,
            };
            if (idletoken_send_msg(fd, &lh, msg_buf, lb.pos) != 0) {
                fprintf(stderr, "idletoken-worker: send INFER_LOGITS: %s\n", strerror(errno));
                loop_rc = 1; break;
            }
            fprintf(stderr,
                    "idletoken-worker: sent INFER_LOGITS (pos=%u, %zu B incl. %zu B logits)\n",
                    pos0, lb.pos, logits_bytes);
        } else {
            /* Send INFER_HC_FORWARD to next stage (zero-filled cur_hc in mock).
             * Prefill chunks (n_tokens > 1) append the chunk token ids after
             * the batch HC rows for the next stage's token-aware kernels. */
            const size_t hc_payload_bytes =
                16 + hc_bytes + (n_tokens > 1 ? 4ull * n_tokens : 0);
            if (hc_payload_bytes > msg_buf_cap) {
                fprintf(stderr, "idletoken-worker: HC payload (%zu B) > buffer; bump msg_buf_cap\n",
                        hc_payload_bytes);
                loop_rc = 1; break;
            }
            idletoken_buf hb;
            idletoken_buf_init(&hb, msg_buf, msg_buf_cap);
            idletoken_buf_put_u8 (&hb, phase);
            idletoken_buf_put_u8 (&hb, hc_dtype);
            /* Reserved byte [0] propagates is_first_chunk: downstream stages
             * must rewind their own shard too. */
            uint8_t hb_pad2[2] = { is_first, seq_id };   /* v4: downstream stages must honour this sequence as well */
            idletoken_buf_put_bytes(&hb, hb_pad2, 2);
            idletoken_buf_put_u32(&hb, pos0);
            idletoken_buf_put_u32(&hb, n_tokens);
            idletoken_buf_put_u32(&hb, (uint32_t)hc_bytes);
            if (cur_session) {
                if (n_tokens == 1) {
#ifndef DS4_NO_GPU
                    const double t_s0 = now_monotonic_s();
                    (void)ds4_gpu_synchronize();
                    g_gpuexec_s += now_monotonic_s() - t_s0;
                    g_gpuexec_calls++;
#endif
                }
                const double t_h0 = now_monotonic_s();
                bool read_ok = (n_tokens > 1)
                    ? ds4_session_batch_hc_read(cur_session, msg_buf + 16, n_tokens)
                    : ds4_session_hc_tensor_read(cur_session, msg_buf + 16, hc_bytes);
                if (n_tokens > 1) { g_pf_other_s += now_monotonic_s() - t_h0; }
                else { g_harvest_s += now_monotonic_s() - t_h0; g_harvest_calls++; }
                if (!read_ok) {
                    fprintf(stderr, "idletoken-worker: ds4 HC read failed (n_tokens=%u)\n",
                            n_tokens);
                    loop_rc = 1; break;
                }
            } else if (cur_xr) {
                /* middle stage: the transformed hidden IS the HC payload */
                memcpy(msg_buf + 16, xhidden, hc_bytes);
            } else {
                memset(msg_buf + 16, 0, hc_bytes);  /* mock: zeros */
            }
            if (n_tokens > 1) {
                uint8_t *tp = msg_buf + 16 + hc_bytes;
                for (uint32_t ti = 0; ti < n_tokens; ti++) {
                    const uint32_t t = (uint32_t)chunk_tokens[ti];
                    tp[4*ti]   = (uint8_t)(t & 0xff);
                    tp[4*ti+1] = (uint8_t)((t >> 8) & 0xff);
                    tp[4*ti+2] = (uint8_t)((t >> 16) & 0xff);
                    tp[4*ti+3] = (uint8_t)((t >> 24) & 0xff);
                }
            }
            hb.pos = hc_payload_bytes;
            idletoken_msg_header hh = {
                .magic = IDLETOKEN_PROTO_MAGIC,
                .version = IDLETOKEN_PROTO_VERSION,
                .msg_type = IDLETOKEN_MSG_INFER_HC_FORWARD,
                .payload_bytes = hb.pos,
                .request_id = req_id,
                .stage_id = stage_id,
                .segment_id = IDLETOKEN_SEGMENT_NONE,
            };
            if (idletoken_send_msg(hc_out_fd, &hh, msg_buf, hb.pos) != 0) {
                fprintf(stderr, "idletoken-worker: send INFER_HC_FORWARD: %s\n", strerror(errno));
                loop_rc = 1; break;
            }
            fprintf(stderr,
                    "idletoken-worker: sent INFER_HC_FORWARD pos0=%u n_tokens=%u hc_bytes=%llu\n",
                    pos0, n_tokens, (unsigned long long)hc_bytes);
        }

        /* v5: this used to block on the coordinator's INFER_TOKEN_ACK before a
         * round counted as finished. Removed -- the pos/token it carried was
         * never used (parsing it only produced a log line), while the cost was
         * pinning the entire pipeline: this stage could not accept the next
         * INFER_BEGIN until the coordinator had collected LOGITS, i.e. until
         * every stage had run. We now go straight back to the top of the loop
         * for the next round. */
        steps_done++;
        /* Print every 8 decode rounds, so the data survives a benchmark script
         * that kills the process with -Force. */
        if (g_compute_calls && (g_compute_calls % 8) == 0 && g_compute_calls != last_prof_at
            && getenv("IDLETOKEN_DS4X_PROF")) {
            last_prof_at = g_compute_calls;
            prof2_dump(stage_id, layer_lo, layer_hi);
        }
    }

    /* IDLETOKEN_DS4X_PROF=1: print the cost breakdown for this stage. In
     * cross-machine pipeline parallelism, which stage is the bottleneck and
     * whether the stages are balanced used to be guesswork -- and that is
     * exactly what decides whether PP micro-batching (E3) is worth it. */
    if (getenv("IDLETOKEN_DS4X_PROF") && (g_compute_calls > 0 || g_prefill_calls > 0))
        prof2_dump(stage_id, layer_lo, layer_hi);
    if (getenv("IDLETOKEN_DS4X_PROF") && g_compute_calls > 0) {
        /* Report only **this stage's wall-clock compute time**: that is the one
         * quantity needed to judge whether the stages are balanced. The
         * CUDA-kernel-level breakdown comes from the single-machine tool
         * ds4x_infer; duplicating it here would drag in the cross-platform
         * header and link dependencies of ds4x_cuda_stats. */
        fprintf(stderr,
                "idletoken-worker: PROF stage=%u layers=[%u,%u) rounds=%llu "
                "compute_s=%.3f avg_ms=%.1f  logits_calls=%llu logits_avg_ms=%.1f\n",
                stage_id, layer_lo, layer_hi,
                (unsigned long long)g_compute_calls, g_compute_s,
                g_compute_s * 1000.0 / (double)g_compute_calls,
                (unsigned long long)g_logits_calls,
                g_logits_calls ? g_logits_s * 1000.0 / (double)g_logits_calls : 0.0);
    }

    free(msg_buf);
    free(chunk_tokens);
    free(xhidden);
    if (hc_in_fd  >= 0) close(hc_in_fd);
    if (hc_out_fd >= 0) close(hc_out_fd);
    close(fd);
    if (session) ds4_session_free(session);
    if (engine)  ds4_engine_close(engine);
    /* Multi-sequence slots (slot 0 is session/xr itself -- do not double free). */
    for (int si = 1; si < IDLETOKEN_MAX_SEQ_SLOTS; si++) {
        if (seq_sessions[si]) ds4_session_free(seq_sessions[si]);
        if (seq_runners[si])  ds4x_runner_free(seq_runners[si]);
    }
    if (xr) ds4x_runner_free(xr);
    if (xm) ds4x_model_free(xm);
    fprintf(stderr, "idletoken-worker: INFER loop done (%d step%s, rc=%d). Exiting.\n",
            steps_done, steps_done == 1 ? "" : "s", loop_rc);
    return loop_rc;
}
