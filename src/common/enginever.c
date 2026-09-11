/* enginever.c — see include/idletoken_enginever.h. C only. No C++. */
#include "idletoken_enginever.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int idletoken_engine_version(const char *llama_server_bin,
                             char *out, size_t cap) {
    if (!out || cap == 0) return -1;
    out[0] = '\0';

    const char *fake = getenv("IDLETOKEN_TEST_ENGINE_VERSION");
    if (fake && fake[0]) {
        fprintf(stderr,
                "*** TEST OVERRIDE *** IDLETOKEN_TEST_ENGINE_VERSION=%s replaces "
                "the real engine version of this node. Never set this outside a "
                "test harness.\n", fake);
        snprintf(out, cap, "%s", fake);
        return 0;
    }

#ifdef _WIN32
    /* MSVCRT spells it _popen/_pclose, and cmd.exe wants DOUBLE quotes around a
     * path with spaces (C:\Users\... is fine, C:\Program Files\... is not) --
     * the single quotes the POSIX branch uses would be passed through as part
     * of the filename. `2>&1` works in cmd as well.
     *
     * Until 2026-08-15 this branch just printed "not implemented" and returned
     * -1, which meant a WINDOWS MACHINE COULD NEVER BE THE COORDINATOR: the
     * cluster refuses to form when the engine-version invariant (hard
     * constraint #4) cannot be checked, so every all-Windows household was
     * blocked at the first handshake. The refusal was right; the missing probe
     * was the bug. Found by G-TOPO's "Windows-coordinated cluster" cell. */
    if (!llama_server_bin || !llama_server_bin[0]) return -1;
    char cmd[1200];
    if (snprintf(cmd, sizeof(cmd), "\"\"%s\" --version 2>&1\"",
                 llama_server_bin) >= (int)sizeof(cmd))
        return -1;
    FILE *p = _popen(cmd, "r");
    if (!p) return -1;
    char line[256] = "";
    int found = -1;
    while (fgets(line, sizeof(line), p)) {
        const char *v = strstr(line, "version:");
        if (v) {
            v += strlen("version:");
            while (*v == ' ' || *v == '\t') v++;
            size_t n = strcspn(v, "\r\n");
            if (n >= cap) n = cap - 1;
            memcpy(out, v, n);
            out[n] = '\0';
            found = out[0] ? 0 : -1;
            break;
        }
    }
    _pclose(p);
    return found;
#else
    if (!llama_server_bin || !llama_server_bin[0]) return -1;
    char cmd[1200];
    /* --version prints "version: N (sha)" on its first line. Merge stderr:
     * which stream it lands on is a llama.cpp implementation detail. */
    if (snprintf(cmd, sizeof(cmd), "'%s' --version 2>&1",
                 llama_server_bin) >= (int)sizeof(cmd))
        return -1;
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    char line[256] = "";
    int found = -1;
    while (fgets(line, sizeof(line), p)) {
        const char *v = strstr(line, "version:");
        if (v) {
            v += strlen("version:");
            while (*v == ' ' || *v == '\t') v++;
            size_t n = strcspn(v, "\r\n");
            if (n >= cap) n = cap - 1;
            memcpy(out, v, n);
            out[n] = '\0';
            found = out[0] ? 0 : -1;
            break;
        }
    }
    pclose(p);
    return found;
#endif
}

/* Scan an engine binary for a string that only the patched sources contain.
 * Overlap the chunks by the needle length so a match straddling a chunk
 * boundary is not missed. 1 found, 0 absent, -1 unreadable. */
static int engine_has_needle(const char *engine_bin, const char *needle) {
    if (!engine_bin || !engine_bin[0]) return -1;
    FILE *f = fopen(engine_bin, "rb");
    if (!f) return -1;
    const size_t nlen = strlen(needle);
    enum { CHUNK = 1 << 20 };
    unsigned char *buf = (unsigned char *)malloc(CHUNK + nlen);
    if (!buf) { fclose(f); return -1; }
    size_t carry = 0;
    int found = 0;
    for (;;) {
        size_t got = fread(buf + carry, 1, CHUNK, f);
        if (got == 0) break;
        size_t have = carry + got;
        for (size_t i = 0; i + nlen <= have; i++) {
            if (buf[i] == (unsigned char)needle[0] &&
                memcmp(buf + i, needle, nlen) == 0) { found = 1; break; }
        }
        if (found) break;
        carry = have >= nlen ? nlen - 1 : have;
        memmove(buf, buf + have - carry, carry);
    }
    free(buf);
    fclose(f);
    return found;
}

int idletoken_engine_has_node_local_moe(const char *engine_bin) {
    /* A log format string compiled into both llama-server and ggml-rpc-server
     * by patch 0005; absent from every earlier series. */
    return engine_has_needle(engine_bin, "selected-expert ranges");
}

int idletoken_engine_has_host_staging(const char *engine_bin) {
    /* The staged allocator's log line, patch 0007 (ggml-cuda is linked
     * statically into both engine binaries). */
    return engine_has_needle(engine_bin, "staged host buffer:");
}

/* --- page-lockable host memory (see the header) -------------------------- */

uint64_t idletoken_engine_probe_pinned(const char *rpc_server_bin) {
    if (!rpc_server_bin || !rpc_server_bin[0]) return 0;
    char cmd[1200];
#ifdef _WIN32
    if (snprintf(cmd, sizeof(cmd), "\"\"%s\" --pinned-probe 2>&1\"", rpc_server_bin) >= (int)sizeof(cmd))
        return 0;
    FILE *p = _popen(cmd, "r");
#else
    if (snprintf(cmd, sizeof(cmd), "'%s' --pinned-probe 2>&1", rpc_server_bin) >= (int)sizeof(cmd))
        return 0;
    FILE *p = popen(cmd, "r");
#endif
    if (!p) return 0;
    char line[512];
    uint64_t mib = 0;
    int found = 0;
    while (fgets(line, sizeof(line), p)) {
        const char *k = strstr(line, "PINNED_MAX_MIB=");
        if (k) {
            mib = (uint64_t)strtoull(k + strlen("PINNED_MAX_MIB="), NULL, 10);
            found = 1;
        }
    }
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    if (!found) {
        fprintf(stderr, "idletoken: %s printed no PINNED_MAX_MIB (an engine without the "
                        "probe, or it failed to start); the pinned ceiling stays unknown\n",
                rpc_server_bin);
        return 0;
    }
    return mib << 20;
}

static void pinned_cache_path(char *out, size_t cap) {
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

uint64_t idletoken_engine_cached_pinned_ceiling(uint64_t ram_total) {
    char path[512];
    pinned_cache_path(path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    unsigned long long key = 0, val = 0;
    const int n = fscanf(f, "%llu %llu", &key, &val);
    fclose(f);
    if (n != 2 || key != (unsigned long long)ram_total || val == 0) return 0;
    return (uint64_t)val;
}

uint64_t idletoken_engine_pinned_ceiling(const char *rpc_server_bin, uint64_t ram_total) {
    char path[512];
    pinned_cache_path(path, sizeof(path));
    const uint64_t cached = idletoken_engine_cached_pinned_ceiling(ram_total);
    if (cached > 0) {
        fprintf(stderr, "idletoken: pinned ceiling %.2f GiB (cached in %s)\n",
                (double)cached / 1073741824.0, path);
        return cached;
    }
    fprintf(stderr, "idletoken: measuring how much RAM this GPU can page-lock, once "
                    "(brief high memory use, up to a minute; cached in %s)\n", path);
    const uint64_t got = idletoken_engine_probe_pinned(rpc_server_bin);
    if (got == 0) return 0;   /* caller applies the platform fallback, if any */
    fprintf(stderr, "idletoken: pinned ceiling %.2f GiB of %.2f GiB RAM\n",
            (double)got / 1073741824.0, (double)ram_total / 1073741824.0);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%llu %llu\n", (unsigned long long)ram_total, (unsigned long long)got);
        fclose(f);
    }
    return got;
}
