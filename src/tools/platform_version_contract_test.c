/* DIST-07 build/source contract. Pure C so all release builders can run it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_all(const char *path) {
    FILE *f = fopen(path, "rb");
    long n;
    char *out;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    out = (char *)malloc((size_t)n + 1);
    if (!out || fread(out, 1, (size_t)n, f) != (size_t)n) {
        free(out); fclose(f); return NULL;
    }
    out[n] = 0;
    fclose(f);
    return out;
}

static int has(const char *text, const char *needle) {
    return text && needle && strstr(text, needle) != NULL;
}

static int count_occurrences(const char *text, const char *needle) {
    int n = 0;
    size_t step = strlen(needle);
    const char *at = text;
    while (step > 0 && (at = strstr(at, needle)) != NULL) {
        n++;
        at += step;
    }
    return n;
}

static int require(const char *path, const char *text, const char *needle) {
    if (has(text, needle)) return 1;
    fprintf(stderr, "PLATFORM_VERSION_CONTRACT_FAIL: %s lacks %s\n", path, needle);
    return 0;
}

static int remove_once(char *text, const char *needle) {
    char *at = strstr(text, needle);
    if (!at) return 0;
    memmove(at, at + strlen(needle), strlen(at + strlen(needle)) + 1);
    return 1;
}

static int mutation_goes_red(const char *path, const char *original,
                             const char *needle) {
    char *copy = (char *)malloc(strlen(original) + 1);
    int red;
    if (!copy) return 0;
    strcpy(copy, original);
    red = remove_once(copy, needle) && !has(copy, needle);
    free(copy);
    if (!red)
        fprintf(stderr, "PLATFORM_VERSION_CONTRACT_FAIL: mutation stayed green: %s\n", path);
    return red;
}

int main(int argc, char **argv) {
    static const char *needles[] = {
        "\"version\"",                          /* package.json */
        "[IDLETOKEN_VERSION_HEADER]: PLATFORM_CLIENT_VERSION", /* browser */
        ".header(IDLETOKEN_VERSION_HEADER, env!(\"CARGO_PKG_VERSION\"))", /* Tauri */
        "IDLETOKEN_VERSION_HTTP_HEADER, body",   /* C request */
        "client/package.json",                   /* POSIX */
        "Get-Content -Raw 'client/package.json'", /* Windows package */
        "Get-Content -Raw 'client/package.json'"  /* Windows root */
    };
    char *files[7] = {0};
    int ok = 1;
    if (argc != 8) {
        fprintf(stderr, "usage: %s package ts rust c make win-script win-root\n", argv[0]);
        return 2;
    }
    for (int i = 0; i < 7; i++) {
        files[i] = read_all(argv[i + 1]);
        if (!files[i]) { fprintf(stderr, "cannot read %s\n", argv[i + 1]); ok = 0; }
    }
    for (int i = 0; i < 7 && ok; i++) {
        ok &= require(argv[i + 1], files[i], needles[i]);
        ok &= mutation_goes_red(argv[i + 1], files[i], needles[i]);
    }
    if (ok) {
        ok &= require(argv[4], files[3], "#error \"IDLETOKEN_CLIENT_VERSION");
        ok &= require(argv[4], files[3], "X-IdleToken-Version: ");
        ok &= require(argv[5], files[4], "-include $(VERSION_HEADER)");
        ok &= require(argv[6], files[5], "-include %VERSION_HEADER%");
        ok &= require(argv[7], files[6], "-include %VERSION_HEADER%");
        ok &= require(argv[6], files[5], "set \"IDLETOKEN_CLIENT_VERSION=\"");
        ok &= require(argv[7], files[6], "set \"IDLETOKEN_CLIENT_VERSION=\"");
        if (count_occurrences(files[3], "http_post_json(platform_addr") != 5) {
            fprintf(stderr, "PLATFORM_VERSION_CONTRACT_FAIL: expected five platform POST call sites\n");
            ok = 0;
        }
        if (!has(files[3], "http_request_json(\"GET\", addr, path, NULL, NULL")) {
            fprintf(stderr, "PLATFORM_VERSION_CONTRACT_FAIL: generic local GET was unexpectedly version-tagged\n");
            ok = 0;
        }
    }
    for (int i = 0; i < 7; i++) free(files[i]);
    if (!ok) return 1;
    puts("PLATFORM_VERSION_CONTRACT_OK (7 source checks, 7 known-bad controls)");
    return 0;
}
