/* Dynamic local-model cache gate. Uses a tiny synthetic indexed file because
 * the cache layer only needs byte ranges; GGUF parsing belongs to autotest. */
#include "idletoken_net.h"
#include "idletoken_weights.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
#define getpid _getpid
#define rmdir _rmdir
#else
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static int checks, failures;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) printf("  [ok] %s\n", what);
    else { failures++; printf("  [FAIL] %s\n", what); }
}

static int mkdir_one(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST ? 0 : -1;
#else
    return mkdir(path, 0700) == 0 || errno == EEXIST ? 0 : -1;
#endif
}

static int write_fixture(const char *path, unsigned seed, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (size_t i = 0; i < bytes; i++) {
        unsigned char b = (unsigned char)((i * 17u + seed) % 251u);
        if (fwrite(&b, 1, 1, f) != 1) { fclose(f); return -1; }
    }
    return fclose(f) == 0 ? 0 : -1;
}

static int put_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int put_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

/* Minimal GGUF v3 fixture: no metadata, one-dimensional F32 tensors. The
 * production indexer sees the same per-part tensor directories and relative
 * offsets as it does in a standard llama.cpp split model. */
static int write_gguf_part(const char *path, const char *const *names,
                           const unsigned *seeds, size_t count,
                           uint64_t *absolute_offsets) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int fail = put_u32(f, UINT32_C(0x46554747)) || put_u32(f, 3) ||
               put_u64(f, (uint64_t)count) || put_u64(f, 0);
    for (size_t i = 0; !fail && i < count; i++) {
        const uint64_t len = (uint64_t)strlen(names[i]);
        fail = put_u64(f, len) || fwrite(names[i], 1, (size_t)len, f) != len ||
               put_u32(f, 1) || put_u64(f, 4) || put_u32(f, 0) ||
               put_u64(f, (uint64_t)(i * 16));
    }
    long pos = ftell(f);
    if (pos < 0) fail = 1;
    const uint64_t tdp = ((uint64_t)(pos < 0 ? 0 : pos) + 31) / 32 * 32;
    while (!fail && (uint64_t)ftell(f) < tdp)
        fail = fputc(0, f) == EOF;
    for (size_t i = 0; !fail && i < count; i++) {
        if (absolute_offsets) absolute_offsets[i] = tdp + i * 16;
        for (size_t j = 0; j < 16; j++) {
            unsigned char b = (unsigned char)((j * 17u + seeds[i]) % 251u);
            if (fwrite(&b, 1, 1, f) != 1) { fail = 1; break; }
        }
    }
    if (fclose(f) != 0) fail = 1;
    return fail ? -1 : 0;
}

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    if (fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s); fclose(f); return NULL;
    }
    s[n] = '\0';
    fclose(f);
    return s;
}

static uint64_t test_fnv1a(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= UINT64_C(0x100000001b3);
    }
    return h;
}

static int write_index(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "4096 256 6 2\n");
    fprintf(f, "-1 256 32 token_embd.weight\n");
    fprintf(f, "0 512 64 blk.0.weight\n");
    fprintf(f, "1 768 64 blk.1.weight\n");
    fprintf(f, "2 1024 64 blk.2.weight\n");
    fprintf(f, "3 1280 64 blk.3.weight\n");
    fprintf(f, "-1 1536 32 output.weight\n");
    return fclose(f) == 0 ? 0 : -1;
}

typedef struct {
    char dir[512];
    char bind[80];
} server_args;

#ifdef _WIN32
static void *serve(void *opaque) {
    server_args *a = (server_args *)opaque;
    idletoken_serve_weights(a->dir, a->bind);
    return NULL;
}
#endif

static int read_at(const char *path, uint64_t off, unsigned char *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
#ifdef _WIN32
    if (_fseeki64(f, (long long)off, SEEK_SET) != 0) { fclose(f); return -1; }
#else
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) { fclose(f); return -1; }
#endif
    int rc = fread(buf, 1, n, f) == n ? 0 : -1;
    fclose(f);
    return rc;
}

static int same_range(const char *a, const char *b, uint64_t off, size_t n) {
    unsigned char aa[256], bb[256];
    if (n > sizeof(aa) || read_at(a, off, aa, n) != 0 ||
        read_at(b, off, bb, n) != 0) return 0;
    return memcmp(aa, bb, n) == 0;
}

static int same_cross_range(const char *a, uint64_t aoff,
                            const char *b, uint64_t boff, size_t n) {
    unsigned char aa[256], bb[256];
    if (n > sizeof(aa) || read_at(a, aoff, aa, n) != 0 ||
        read_at(b, boff, bb, n) != 0) return 0;
    return memcmp(aa, bb, n) == 0;
}

static int sibling_path(const char *primary, const char *name,
                        char *out, size_t cap) {
    const char *slash = strrchr(primary, '/');
#ifdef _WIN32
    const char *back = strrchr(primary, '\\');
    if (!slash || (back && back > slash)) slash = back;
#endif
    size_t prefix = slash ? (size_t)(slash - primary + 1) : 0;
    int n = snprintf(out, cap, "%.*s%s", (int)prefix, primary, name);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static int zero_range(const char *path, uint64_t off, size_t n) {
    unsigned char b[256];
    if (n > sizeof(b) || read_at(path, off, b, n) != 0) return 0;
    for (size_t i = 0; i < n; i++) if (b[i] != 0) return 0;
    return 1;
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = getenv("TEMP");
    if (!tmp || !tmp[0]) tmp = "/tmp";

    char root[512], model[640], index[672], cache[640];
    snprintf(root, sizeof(root), "%s/idletoken-weights-%d", tmp, (int)getpid());
    snprintf(model, sizeof(model), "%s/master.gguf", root);
    snprintf(index, sizeof(index), "%s.idx", model);
    snprintf(cache, sizeof(cache), "%s/cache/nested", root);
    ok(mkdir_one(root) == 0, "creates an isolated fixture directory");
    ok(write_fixture(model, 23, 4096) == 0 && write_index(index) == 0,
       "writes the synthetic indexed model");

    server_args args;
    snprintf(args.dir, sizeof(args.dir), "%s", root);
    const int port = 30000 + ((int)getpid() % 20000);
    snprintf(args.bind, sizeof(args.bind), "127.0.0.1:%d", port);
    int started = 0;
#ifdef _WIN32
    pthread_t tid;
    started = pthread_create(&tid, NULL, serve, &args) == 0;
    if (started) pthread_detach(tid);
#else
    pid_t server_pid = fork();
    if (server_pid == 0) {
        idletoken_serve_weights(args.dir, args.bind);
        _exit(1);
    }
    started = server_pid > 0;
#endif
    int ready = 0;
    for (int i = 0; started && i < 100; i++) {
        int fd = idletoken_connect_tcp(args.bind);
        if (fd >= 0) { idletoken_close_fd(fd); ready = 1; break; }
#ifdef _WIN32
        Sleep(20);
#else
        usleep(20000);
#endif
    }
    ok(ready, "starts the range repository");

    char url[256], view[800], view2[800];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d/master.gguf", port);
    int cold = ready ? idletoken_local_model_prepare(url, 2, cache,
                                                      view, sizeof(view)) : -1;
    ok(cold == 0, "builds a scheduler-selected [0,2) local view");
    ok(cold == 0 && same_range(model, view, 0, 256) &&
                       same_range(model, view, 256, 32) &&
                       same_range(model, view, 512, 64) &&
                       same_range(model, view, 768, 64) &&
                       same_range(model, view, 1536, 32),
       "the view contains its header, shared tensors, and assigned layers");
    ok(cold == 0 && zero_range(view, 1024, 64) && zero_range(view, 1280, 64),
       "unassigned remote layers remain sparse holes");

    int smaller = cold == 0 ? idletoken_local_model_prepare(
        url, 1, cache, view2, sizeof(view2)) : -1;
    ok(smaller == 0 && strcmp(view, view2) == 0 &&
                       same_range(model, view2, 768, 64),
       "a smaller heterogeneous re-plan reuses the high-water view");

    int larger = smaller == 0 ? idletoken_local_model_prepare(
        url, 4, cache, view2, sizeof(view2)) : -1;
    ok(larger == 0 && strcmp(view, view2) == 0 &&
                      same_range(model, view2, 1024, 64) &&
                      same_range(model, view2, 1280, 64),
       "a larger re-plan extends the same view with only the missing suffix");

    char marker[840];
    snprintf(marker, sizeof(marker), "%s.done", view);
    FILE *mf = fopen(marker, "r");
    unsigned long long hash = 0, size = 0;
    unsigned coverage = 0;
    int marker_ok = mf && fscanf(mf, "IDLETOKEN_LOCAL_VIEW_V1 %llx %u %llu",
                                 &hash, &coverage, &size) == 3;
    if (mf) fclose(mf);
    ok(marker_ok && hash != 0 && coverage == 4 && size == 4096,
       "the durable marker records model identity and reusable coverage");

    /* Standard split GGUF: the first file carries metadata and may contain no
     * tensors at all. This is the production shape which originally produced
     * a valid-looking 0-layer index and made cluster startup fail on Windows. */
    char split1[700], split2[700], split3[700], split_idx[740];
    snprintf(split1, sizeof(split1), "%s/Tiny-00001-of-00003.gguf", root);
    snprintf(split2, sizeof(split2), "%s/Tiny-00002-of-00003.gguf", root);
    snprintf(split3, sizeof(split3), "%s/Tiny-00003-of-00003.gguf", root);
    snprintf(split_idx, sizeof(split_idx), "%s.idx", split1);
    const char *names2[] = {
        "token_embd.weight", "blk.0.weight", "blk.2.weight"
    };
    const unsigned seeds2[] = {31, 41, 61};
    const char *names3[] = {
        "blk.1.weight", "blk.3.weight", "output.weight"
    };
    const unsigned seeds3[] = {51, 71, 81};
    uint64_t off2[3] = {0}, off3[3] = {0};
    int split_written =
        write_gguf_part(split1, NULL, NULL, 0, NULL) == 0 &&
        write_gguf_part(split2, names2, seeds2, 3, off2) == 0 &&
        write_gguf_part(split3, names3, seeds3, 3, off3) == 0;
    ok(split_written, "writes a metadata-only primary plus two tensor parts");
    int indexed = split_written
        ? idletoken_write_idx(split1, split_idx) == 0 : 0;
    char *split_text = indexed ? read_text(split_idx) : NULL;
    ok(indexed && split_text && strstr(split_text, " 3 3") &&
           strstr(split_text, "P 1 ") && strstr(split_text, "P 2 ") &&
           strstr(split_text, "T 2 0 ") && strstr(split_text, "T 3 3 "),
       "builds one v3 index over every split tensor directory");
    ok(indexed && idletoken_idx_stale(split1, split_idx) == 0,
       "accepts a split index only while every part size matches");

    char split_cache[700] = "", split_url[300] = "", split_view[900] = "";
    snprintf(split_cache, sizeof(split_cache), "%s/split-cache", root);
    snprintf(split_url, sizeof(split_url),
             "http://127.0.0.1:%d/Tiny-00001-of-00003.gguf", port);
    int split_local = indexed
        ? idletoken_local_model_prepare(split_url, 2, split_cache,
                                        split_view, sizeof(split_view)) : -1;
    char local2[900] = "", local3[900] = "";
    sibling_path(split_view, "Tiny-00002-of-00003.gguf",
                 local2, sizeof(local2));
    sibling_path(split_view, "Tiny-00003-of-00003.gguf",
                 local3, sizeof(local3));
    ok(split_local == 0 && strstr(split_view, "Tiny-00001-of-00003.gguf") &&
           same_cross_range(split1, 0, split_view, 0, 32) &&
           same_cross_range(split2, 0, local2, 0, (size_t)off2[0]) &&
           same_cross_range(split3, 0, local3, 0, (size_t)off3[0]),
       "materializes all split headers under their original sibling names");
    ok(split_local == 0 &&
           same_cross_range(split2, off2[0], local2, off2[0], 16) &&
           same_cross_range(split2, off2[1], local2, off2[1], 16) &&
           zero_range(local2, off2[2], 16) &&
           same_cross_range(split3, off3[0], local3, off3[0], 16) &&
           zero_range(local3, off3[1], 16) &&
           same_cross_range(split3, off3[2], local3, off3[2], 16),
       "keeps shared plus [0,2) tensors and leaves other split layers sparse");

    char direct_cache[700] = "", direct_view[900] = "";
    char direct2[900] = "", direct3[900] = "";
    snprintf(direct_cache, sizeof(direct_cache), "%s/direct-cache", root);
    int direct_local = indexed
        ? idletoken_local_model_prepare_from_file(
              split_url, split1, 2, direct_cache,
              direct_view, sizeof(direct_view)) : -1;
    sibling_path(direct_view, "Tiny-00002-of-00003.gguf",
                 direct2, sizeof(direct2));
    sibling_path(direct_view, "Tiny-00003-of-00003.gguf",
                 direct3, sizeof(direct3));
    ok(direct_local == 0 &&
           same_cross_range(split2, off2[1], direct2, off2[1], 16) &&
           same_cross_range(split3, off3[0], direct3, off3[0], 16) &&
           zero_range(direct2, off2[2], 16),
       "copies coordinator-owned split ranges directly without loopback data");
    int prefetched = direct_local == 0
        ? idletoken_local_model_prefetch(split_url, 1, direct_view) : -1;
    ok(prefetched == 0,
       "prefetches only the coordinator CPU prefix from a sparse split view");

    char rpc_cache[700];
    snprintf(rpc_cache, sizeof(rpc_cache), "%s/rpc-cache", root);
    uint64_t rpc_bytes = 0;
    unsigned rpc_tensors = 0;
    int rpc_seeded = indexed
        ? idletoken_rpc_cache_fetch(split_url, 2, 4, NULL, rpc_cache,
                                    NULL, NULL, &rpc_bytes, &rpc_tensors) : -1;
    char named2[800], named3[800];
    const uint64_t nh2 = test_fnv1a(names2[2], strlen(names2[2]));
    const uint64_t nh3 = test_fnv1a(names3[1], strlen(names3[1]));
    snprintf(named2, sizeof(named2), "%s/n-%016llx", rpc_cache,
             (unsigned long long)nh2);
    snprintf(named3, sizeof(named3), "%s/n-%016llx", rpc_cache,
             (unsigned long long)nh3);
    ok(rpc_seeded == 0 && rpc_tensors == 4 && rpc_bytes == 64 &&
           same_cross_range(split2, off2[2], named2, 0, 16) &&
           same_cross_range(split3, off3[1], named3, 0, 16),
       "seeds a worker's assigned tensors from the correct source parts");

    char rpc_local_cache[700], local_named2[800], local_named3[800];
    snprintf(rpc_local_cache, sizeof(rpc_local_cache), "%s/rpc-local-cache", root);
    rpc_bytes = 0;
    rpc_tensors = 0;
    int rpc_local_seeded = indexed
        ? idletoken_rpc_cache_fetch(split_url, 2, 4, split1,
                                    rpc_local_cache, NULL, NULL,
                                    &rpc_bytes, &rpc_tensors) : -1;
    snprintf(local_named2, sizeof(local_named2), "%s/n-%016llx",
             rpc_local_cache, (unsigned long long)nh2);
    snprintf(local_named3, sizeof(local_named3), "%s/n-%016llx",
             rpc_local_cache, (unsigned long long)nh3);
    ok(rpc_local_seeded == 0 && rpc_tensors == 4 && rpc_bytes == 64 &&
           same_cross_range(split2, off2[2], local_named2, 0, 16) &&
           same_cross_range(split3, off3[1], local_named3, 0, 16),
       "seeds only assigned RPC tensors from this node's complete split GGUF");

    FILE *changed = fopen(split3, "ab");
    if (changed) { fputc(0, changed); fclose(changed); }
    ok(changed && idletoken_idx_stale(split1, split_idx) == 1,
       "invalidates the aggregate index when any sibling part changes");
    char rpc_bad_cache[700];
    snprintf(rpc_bad_cache, sizeof(rpc_bad_cache), "%s/rpc-bad-cache", root);
    ok(changed && idletoken_rpc_cache_fetch(split_url, 2, 4, split1,
                                             rpc_bad_cache, NULL, NULL,
                                             NULL, NULL) != 0,
       "refuses a local GGUF set that does not match the creator's index");

    /* Remove split fixtures and their tiny caches. */
    if (split_text) {
        const uint64_t ih = test_fnv1a(split_text, strlen(split_text));
        const char *selected_names[] = {
            names2[0], names2[2], names3[1], names3[2]
        };
        const char *selected_files[] = { split2, split2, split3, split3 };
        const uint64_t selected_offs[] = { off2[0], off2[2], off3[1], off3[2] };
        for (size_t i = 0; i < 4; i++) {
            unsigned char b[16];
            if (read_at(selected_files[i], selected_offs[i], b, sizeof(b)) == 0) {
                char p[850];
                uint64_t ch = test_fnv1a(b, sizeof(b));
                uint64_t nh = test_fnv1a(selected_names[i],
                                         strlen(selected_names[i]));
                snprintf(p, sizeof(p), "%s/%016llx", rpc_cache,
                         (unsigned long long)ch); remove(p);
                snprintf(p, sizeof(p), "%s/n-%016llx", rpc_cache,
                         (unsigned long long)nh); remove(p);
                snprintf(p, sizeof(p), "%s/r-%016llx", rpc_cache,
                         (unsigned long long)nh); remove(p);
            }
        }
        char p[850];
        snprintf(p, sizeof(p), "%s/idletoken-%016llx-L2-4.done", rpc_cache,
                 (unsigned long long)ih); remove(p);
    }
    free(split_text);
    rmdir(rpc_cache);
    if (split_local == 0) {
        char split_marker[940], split_dir[900];
        snprintf(split_dir, sizeof(split_dir), "%s", split_view);
        char *slash = strrchr(split_dir, '/');
#ifdef _WIN32
        { char *back = strrchr(split_dir, '\\');
          if (!slash || (back && back > slash)) slash = back; }
#endif
        if (slash) {
            *slash = '\0';
            snprintf(split_marker, sizeof(split_marker),
                     "%s/idletoken.done", split_dir);
            remove(split_marker); remove(split_view); remove(local2); remove(local3);
            rmdir(split_dir);
        }
    }
    rmdir(split_cache);
    if (direct_local == 0) {
        char direct_marker[940], direct_dir[900];
        snprintf(direct_dir, sizeof(direct_dir), "%s", direct_view);
        char *slash = strrchr(direct_dir, '/');
#ifdef _WIN32
        { char *back = strrchr(direct_dir, '\\');
          if (!slash || (back && back > slash)) slash = back; }
#endif
        if (slash) {
            *slash = '\0';
            snprintf(direct_marker, sizeof(direct_marker),
                     "%s/idletoken.done", direct_dir);
            remove(direct_marker); remove(direct_view); remove(direct2); remove(direct3);
            rmdir(direct_dir);
        }
    }
    rmdir(direct_cache);
    remove(split_idx); remove(split1); remove(split2); remove(split3);

#ifndef _WIN32
    if (started) {
        kill(server_pid, SIGTERM);
        waitpid(server_pid, NULL, 0);
    }
#endif
    remove(marker);
    remove(view);
    remove(index);
    remove(model);
    rmdir(cache);
    {
        char p[640];
        snprintf(p, sizeof(p), "%s/cache", root);
        rmdir(p);
    }
    rmdir(root);

    printf("\n%d checks, %d failures\n", checks, failures);
    puts(failures ? "WEIGHTS_TEST_FAIL" : "WEIGHTS_TEST_OK");
    return failures ? 1 : 0;
}
