/* ds4x_tok_test.c — GGUF byte-level BPE tokenizer unit tests.
 * Fixture: build/fixtures/tokenizer.gguf (scripts/make_test_gguf.py).
 * Verifies vocab load, decode (byte-unicode reversal), special tokens, and the
 * BPE merge encode (round-trip). */
#include "idletoken_ds4x_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) printf("  [ok] %s\n", what);
    else { failures++; printf("  [FAIL] %s\n", what); }
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "build/fixtures";
    char path[512], err[256];
    snprintf(path, sizeof(path), "%s/tokenizer.gguf", dir);

    ds4x_tokenizer *t = ds4x_tok_load(path, err, sizeof(err));
    ok(t != NULL, "tokenizer loads");
    if (!t) { printf("  err: %s\n  DS4X_TOK_TEST_FAIL\n", err); return 1; }

    ok(ds4x_tok_vocab_size(t) == 264, "vocab = 256 bytes + 4 merges + bos/eos + 2 chatml");
    ok(ds4x_tok_bos(t) == 260 && ds4x_tok_eos(t) == 261, "bos/eos ids read");

    /* decode: single byte tokens → raw text */
    int32_t hi[] = { 'h', 'i' };   /* ascii printable = byte-unicode self, id==byte */
    char *d = ds4x_tok_decode(t, hi, 2, 0);
    ok(d && !strcmp(d, "hi"), "decode single-byte tokens → 'hi'");
    free(d);

    /* decode: a space (byte 32, non-printable → byte-unicode glyph) round-trips */
    int32_t sp[] = { ' ', 'a', ' ', 'b' };
    d = ds4x_tok_decode(t, sp, 4, 0);
    ok(d && !strcmp(d, " a b"), "decode reverses non-printable byte glyphs (spaces)");
    free(d);

    /* decode: control tokens are dropped unless kept */
    int32_t withctrl[] = { 260, 'h', 'i', 261 };   /* <bos> h i <eos> */
    d = ds4x_tok_decode(t, withctrl, 4, 0);
    ok(d && !strcmp(d, "hi"), "control tokens dropped by default");
    free(d);
    d = ds4x_tok_decode(t, withctrl, 4, 1);
    ok(d && !strcmp(d, "<bos>hi<eos>"), "control tokens kept when asked");
    free(d);

    /* encode: BPE merges h+e, l+l, he+ll, hell+o → single "hello" token */
    int32_t ids[16];
    int64_t n = ds4x_tok_encode(t, "hello", ids, 16);
    ok(n == 1, "encode 'hello' merges to a single token");
    if (n == 1) {
        char *back = ds4x_tok_decode(t, ids, 1, 0);
        ok(back && !strcmp(back, "hello"), "encode→decode round-trips 'hello'");
        free(back);
    }

    /* encode round-trip on a longer string (partial merges + raw bytes) */
    const char *s = "he ll hello";
    n = ds4x_tok_encode(t, s, ids, 16);
    ok(n > 0, "encode mixed string produces tokens");
    char *rt = ds4x_tok_decode(t, ids, (uint32_t)n, 0);
    ok(rt && !strcmp(rt, s), "encode→decode round-trips a mixed string");
    free(rt);

    /* special-token-aware encode: <|im_start|> and <|im_end|> are added tokens
     * (ids 262/263) → emitted as single ids; the "hi" between them byte-BPEs. */
    ok(ds4x_tok_special_id(t, "<|im_start|>") == 262 &&
       ds4x_tok_special_id(t, "<|im_end|>") == 263, "special ids resolve");
    ok(ds4x_tok_special_id(t, "hello") == -1, "normal token is not 'special'");
    {
        int64_t m = ds4x_tok_encode(t, "<|im_start|>hi<|im_end|>", ids, 16);
        /* [262, 'h', 'i', 263] */
        ok(m == 4 && ids[0] == 262 && ids[1] == 'h' && ids[2] == 'i' && ids[3] == 263,
           "encode splits on special tokens, byte-BPEs the gap");
    }

    /* ChatML chat_apply: renders <|im_start|>role\n...<|im_end|>\n turns and a
     * trailing assistant open; decode(keep_special) must reproduce the string. */
    {
        const char *roles[]    = { "user" };
        const char *contents[] = { "hello" };
        int32_t cids[64];
        int64_t cn = ds4x_tok_chat_apply(t, roles, contents, 1, 1, cids, 64);
        ok(cn > 0 && cids[0] == 262, "chat_apply starts with <|im_start|>");
        char *cd = ds4x_tok_decode(t, cids, (uint32_t)cn, 1);
        ok(cd && !strcmp(cd, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"),
           "chat_apply renders the ChatML prompt");
        free(cd);
    }

    ds4x_tok_free(t);
    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("DS4X_TOK_TEST_FAIL\n"); return 1; }
    printf("DS4X_TOK_TEST_OK\n");
    return 0;
}
