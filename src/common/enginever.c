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
