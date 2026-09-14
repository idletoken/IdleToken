/* Inflate side of the deflated sealed payload (see idletoken_deflate_wire.h).
 *
 * Only the inflate half lives here: the platform compresses (Node's zlib), the
 * agent decompresses. One direction means one implementation to vendor and one
 * to audit — and the reply leg travels home→cloud, which measured fine.
 *
 * The inflater is vendor/puff (Mark Adler, zlib licence): inflate-only, no
 * dynamic allocation of its own, and small enough to read end to end. It is
 * slower than zlib's inflate by about 4x, which for a payload of this size is
 * a few milliseconds — and this code path runs once per request, next to a
 * public-key unseal and an LLM.
 */

#include "idletoken_deflate_wire.h"
#include "idletoken_privacy.h"   /* idletoken_secure_zero */
#include "puff.h"

#include <stdlib.h>
#include <string.h>

int idletoken_wire_is_deflated(const void *buf, size_t len) {
    return buf && len >= IDLETOKEN_DEFLATE_MAGIC_LEN &&
           memcmp(buf, IDLETOKEN_DEFLATE_MAGIC, IDLETOKEN_DEFLATE_MAGIC_LEN) == 0;
}

void idletoken_wire_free(uint8_t *buf, size_t len) {
    if (!buf) return;
    idletoken_secure_zero(buf, len);
    free(buf);
}

int idletoken_wire_inflate(const void *in, size_t in_len,
                           uint8_t **out, size_t *out_len, size_t max_out) {
    if (out) *out = NULL;
    if (out_len) *out_len = 0;
    if (!out || !out_len || !idletoken_wire_is_deflated(in, in_len)) return -1;

    const unsigned char *src = (const unsigned char *)in + IDLETOKEN_DEFLATE_MAGIC_LEN;
    unsigned long src_len = (unsigned long)(in_len - IDLETOKEN_DEFLATE_MAGIC_LEN);
    if (src_len == 0) return -1;

    /* Pass 1: ask how big it claims to be, writing nothing. This is what makes
     * the ceiling meaningful — refusing after allocating would be refusing
     * after the damage. */
    unsigned long want = 0, consumed = src_len;
    if (puff(NIL, &want, src, &consumed) != 0) return -1;
    if (want == 0 || want > max_out) return -1;

    unsigned char *buf = malloc((size_t)want);
    if (!buf) return -1;

    /* Pass 2: the real one. Both passes must agree; if the second reads a
     * different length out of the same bytes, something is wrong with our
     * assumptions and the payload does not get used. */
    unsigned long got = want, read = src_len;
    if (puff(buf, &got, src, &read) != 0 || got != want) {
        idletoken_wire_free(buf, (size_t)want);
        return -1;
    }

    *out = buf;
    *out_len = (size_t)want;
    return 0;
}
