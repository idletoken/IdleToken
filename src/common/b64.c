/* IdleToken — base64 (libsodium ORIGINAL variant). See include/idletoken_b64.h.
 *
 * Lifted verbatim from src/tools/platform_agent.c, which had carried the only
 * copy since the sealed envelope existed; the agent now calls in here so the
 * provider side and the coordinator's consumer side cannot drift. */

#include "idletoken_b64.h"

#include <stdlib.h>
#include <string.h>

static const char b64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char *idletoken_b64_encode(const uint8_t *in, size_t n) {
    size_t olen = 4 * ((n + 2) / 3);
    char *out = malloc(olen + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < n) v |= (uint32_t)in[i + 1] << 8;
        if (i + 2 < n) v |= (uint32_t)in[i + 2];
        out[o++] = b64_alphabet[(v >> 18) & 63];
        out[o++] = b64_alphabet[(v >> 12) & 63];
        out[o++] = (i + 1 < n) ? b64_alphabet[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? b64_alphabet[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

uint8_t *idletoken_b64_decode(const char *s, size_t slen, size_t *out_len) {
    if (!s || slen == 0 || slen % 4 != 0) return NULL;
    int8_t rev[256];
    memset(rev, -1, sizeof(rev));
    for (int i = 0; i < 64; i++) rev[(uint8_t)b64_alphabet[i]] = (int8_t)i;

    size_t pad = 0;
    if (s[slen - 1] == '=') pad++;
    if (s[slen - 2] == '=') pad++;
    size_t olen = slen / 4 * 3 - pad;
    uint8_t *out = malloc(olen ? olen : 1);
    if (!out) return NULL;

    size_t o = 0;
    for (size_t i = 0; i < slen; i += 4) {
        int last = (i + 4 == slen);
        uint32_t v = 0;
        int npad = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i + k];
            if (c == '=') {
                /* '=' only allowed as the final group's tail: "xx==" / "xxx=" */
                if (!last || k < 2 || (k == 2 && s[i + 3] != '=')) { free(out); return NULL; }
                npad = 4 - k;
                v <<= 6 * npad;
                break;
            }
            int8_t d = rev[(uint8_t)c];
            if (d < 0) { free(out); return NULL; }
            v = (v << 6) | (uint32_t)d;
        }
        int nout = 3 - npad;
        for (int k = 0; k < nout; k++) {
            if (o >= olen) { free(out); return NULL; }
            out[o++] = (uint8_t)(v >> (16 - 8 * k));
        }
    }
    if (o != olen) { free(out); return NULL; }
    if (out_len) *out_len = olen;
    return out;
}
