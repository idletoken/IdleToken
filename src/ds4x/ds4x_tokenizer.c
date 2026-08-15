/* ds4x_tokenizer.c — GGUF byte-level BPE tokenizer. See idletoken_ds4x_tok.h.
 *
 * Vocabulary strings live in the GPT-2 "byte-unicode" alphabet: each raw byte
 * maps to a printable Unicode codepoint (bytes_to_unicode), so a token string
 * is a sequence of those codepoints encoded as UTF-8. Decode reverses the map
 * back to raw bytes; encode maps input bytes into the alphabet, runs BPE
 * merges, then looks up token ids. */
/* strdup is POSIX, not C99 — declare it under -std=c99 on glibc (else it is
 * implicitly int, truncating the returned pointer to 32 bits → crash). */
#define _POSIX_C_SOURCE 200809L
#include "idletoken_ds4x_tok.h"
#include "idletoken_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* token_type values (llama.cpp LLAMA_TOKEN_TYPE_*). */
enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4,
       TT_UNUSED = 5, TT_BYTE = 6 };

struct ds4x_tokenizer {
    uint32_t n_vocab;
    char   **tokens;        /* [n_vocab] byte-unicode strings (owned) */
    int8_t  *types;         /* [n_vocab] token_type */
    int32_t  bos, eos;

    /* string → id hash (open addressing) */
    int32_t *ht;            /* [ht_cap] ids, -1 empty */
    uint32_t ht_cap;

    /* merge rank map: key "A B" → rank. Stored as a parallel string/rank array
     * with the same hash table shape (linear list is fine — merges are read
     * once per pair during encode; we hash them). */
    char   **merge_key;     /* [merge_cap] "A B" or NULL */
    int32_t *merge_rank;    /* [merge_cap] */
    uint32_t merge_cap, n_merges;

    /* byte-unicode maps */
    int32_t byte_to_uni[256];       /* raw byte → codepoint */
    int16_t uni_to_byte[0x200];     /* codepoint → byte (-1 none), enough for GPT2 */

    /* special / added tokens (types CONTROL / USER_DEFINED): stored as raw
     * literals so encode can match them in the input verbatim (added tokens
     * are NOT byte-unicode encoded). Sorted longest-first for greedy match. */
    int32_t *special;       /* [n_special] ids, longest literal first */
    uint32_t n_special;
};

/* FNV-1a over a NUL-terminated string. */
static uint64_t fnv(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return h;
}

static void ht_put(ds4x_tokenizer *t, const char *key, int32_t id) {
    uint32_t i = (uint32_t)(fnv(key) & (t->ht_cap - 1));
    while (t->ht[i] != -1) {
        if (!strcmp(t->tokens[t->ht[i]], key)) { t->ht[i] = id; return; }
        i = (i + 1) & (t->ht_cap - 1);
    }
    t->ht[i] = id;
}
static int32_t ht_get(const ds4x_tokenizer *t, const char *key) {
    uint32_t i = (uint32_t)(fnv(key) & (t->ht_cap - 1));
    while (t->ht[i] != -1) {
        if (!strcmp(t->tokens[t->ht[i]], key)) return t->ht[i];
        i = (i + 1) & (t->ht_cap - 1);
    }
    return -1;
}

static int32_t merge_get(const ds4x_tokenizer *t, const char *key) {
    if (t->merge_cap == 0) return -1;
    uint32_t i = (uint32_t)(fnv(key) & (t->merge_cap - 1));
    while (t->merge_key[i]) {
        if (!strcmp(t->merge_key[i], key)) return t->merge_rank[i];
        i = (i + 1) & (t->merge_cap - 1);
    }
    return -1;
}
static void merge_put(ds4x_tokenizer *t, const char *key, int32_t rank) {
    uint32_t i = (uint32_t)(fnv(key) & (t->merge_cap - 1));
    while (t->merge_key[i]) i = (i + 1) & (t->merge_cap - 1);
    t->merge_key[i] = strdup(key);
    t->merge_rank[i] = rank;
}

/* Append codepoint cp as UTF-8 to buf[*pos], returns bytes written. */
static int utf8_put(char *buf, int cp) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) { buf[0] = (char)(0xC0 | (cp >> 6)); buf[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    buf[0] = (char)(0xE0 | (cp >> 12));
    buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}
/* Decode one UTF-8 codepoint at s, advance *i. Returns cp or -1. */
static int utf8_get(const char *s, size_t len, size_t *i) {
    unsigned char c = (unsigned char)s[*i];
    if (c < 0x80) { (*i)++; return c; }
    if ((c & 0xE0) == 0xC0 && *i + 1 < len) {
        int cp = ((c & 0x1F) << 6) | ((unsigned char)s[*i + 1] & 0x3F);
        *i += 2; return cp;
    }
    if ((c & 0xF0) == 0xE0 && *i + 2 < len) {
        int cp = ((c & 0x0F) << 12) | (((unsigned char)s[*i + 1] & 0x3F) << 6) |
                 ((unsigned char)s[*i + 2] & 0x3F);
        *i += 3; return cp;
    }
    (*i)++; return -1;
}

/* GPT-2 bytes_to_unicode: printable byte ranges map to themselves, the rest to
 * 256 + running index. Builds both directions. */
static void build_byte_unicode(ds4x_tokenizer *t) {
    for (int i = 0; i < 0x200; i++) t->uni_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        int cp = printable ? b : (256 + n++);
        t->byte_to_uni[b] = cp;
        if (cp < 0x200) t->uni_to_byte[cp] = (int16_t)b;
    }
}

ds4x_tokenizer *ds4x_tok_load(const char *gguf_path, char *err, size_t errlen) {
#define FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(gguf_path, err, errlen);
    if (!m) return NULL;
    ds4x_tokenizer *t = (ds4x_tokenizer *)calloc(1, sizeof(*t));
    if (!t) { idletoken_gguf_meta_close(m); return NULL; }
    build_byte_unicode(t);

    uint64_t nv = 0;
    if (idletoken_gguf_meta_arr_len(m, "tokenizer.ggml.tokens", &nv) != 0 || nv == 0)
        FAIL("no tokenizer.ggml.tokens array");
    t->n_vocab = (uint32_t)nv;
    t->tokens = (char **)calloc(nv, sizeof(char *));
    t->types  = (int8_t *)calloc(nv, 1);
    if (!t->tokens || !t->types) FAIL("oom vocab");

    /* In-order walk: indexed access is O(idx) per element, so reading a
     * 151k-token vocab that way is quadratic (~30 s for Qwen3 — it dominated
     * the whole model load before this). */
    char tokbuf[512];
    idletoken_gguf_str_iter tit;
    if (idletoken_gguf_meta_arr_str_begin(m, "tokenizer.ggml.tokens", &tit, NULL) != 0)
        FAIL("cannot open tokenizer.ggml.tokens");
    for (uint64_t i = 0; i < nv; i++) {
        int64_t l = idletoken_gguf_meta_arr_str_next(&tit, tokbuf, sizeof(tokbuf));
        if (l < 0) FAIL("bad token %llu", (unsigned long long)i);
        t->tokens[i] = strdup(tokbuf);
        int32_t ty = TT_NORMAL;
        idletoken_gguf_meta_arr_i32(m, "tokenizer.ggml.token_type", i, &ty);
        t->types[i] = (int8_t)ty;
    }

    /* vocab hash (power-of-two ≥ 2×n) */
    t->ht_cap = 1;
    while (t->ht_cap < nv * 2) t->ht_cap <<= 1;
    t->ht = (int32_t *)malloc(t->ht_cap * sizeof(int32_t));
    if (!t->ht) FAIL("oom hash");
    for (uint32_t i = 0; i < t->ht_cap; i++) t->ht[i] = -1;
    for (uint64_t i = 0; i < nv; i++) ht_put(t, t->tokens[i], (int32_t)i);

    /* merges (optional) */
    uint64_t nm = 0;
    idletoken_gguf_meta_arr_len(m, "tokenizer.ggml.merges", &nm);
    t->n_merges = (uint32_t)nm;
    if (nm > 0) {
        t->merge_cap = 1;
        while (t->merge_cap < nm * 2) t->merge_cap <<= 1;
        t->merge_key  = (char **)calloc(t->merge_cap, sizeof(char *));
        t->merge_rank = (int32_t *)calloc(t->merge_cap, sizeof(int32_t));
        if (!t->merge_key || !t->merge_rank) FAIL("oom merges");
        char mb[1024];
        idletoken_gguf_str_iter mit;
        if (idletoken_gguf_meta_arr_str_begin(m, "tokenizer.ggml.merges", &mit, NULL) != 0)
            FAIL("cannot open tokenizer.ggml.merges");
        for (uint64_t i = 0; i < nm; i++) {
            if (idletoken_gguf_meta_arr_str_next(&mit, mb, sizeof(mb)) < 0)
                FAIL("bad merge %llu", (unsigned long long)i);
            merge_put(t, mb, (int32_t)i);   /* rank = position */
        }
    }

    int32_t v;
    t->bos = (idletoken_gguf_meta_u32(m, "tokenizer.ggml.bos_token_id", (uint32_t *)&v) == 0) ? v : -1;
    t->eos = (idletoken_gguf_meta_u32(m, "tokenizer.ggml.eos_token_id", (uint32_t *)&v) == 0) ? v : -1;

    /* collect special/added tokens (CONTROL / USER_DEFINED) for verbatim
     * matching in encode, sorted longest literal first (greedy longest-match). */
    for (uint32_t i = 0; i < t->n_vocab; i++)
        if (t->types[i] == TT_CONTROL || t->types[i] == TT_USER_DEFINED) t->n_special++;
    if (t->n_special) {
        t->special = (int32_t *)malloc(t->n_special * sizeof(int32_t));
        if (!t->special) FAIL("oom special");
        uint32_t j = 0;
        for (uint32_t i = 0; i < t->n_vocab; i++)
            if (t->types[i] == TT_CONTROL || t->types[i] == TT_USER_DEFINED)
                t->special[j++] = (int32_t)i;
        /* insertion sort by descending token string length (n_special is small) */
        for (uint32_t a = 1; a < t->n_special; a++) {
            int32_t id = t->special[a];
            size_t la = strlen(t->tokens[id]);
            uint32_t b = a;
            while (b > 0 && strlen(t->tokens[t->special[b - 1]]) < la) {
                t->special[b] = t->special[b - 1]; b--;
            }
            t->special[b] = id;
        }
    }

    idletoken_gguf_meta_close(m);
    return t;
fail:
    ds4x_tok_free(t);
    idletoken_gguf_meta_close(m);
    return NULL;
#undef FAIL
}

void ds4x_tok_free(ds4x_tokenizer *t) {
    if (!t) return;
    if (t->tokens) for (uint32_t i = 0; i < t->n_vocab; i++) free(t->tokens[i]);
    free(t->tokens); free(t->types); free(t->ht);
    if (t->merge_key) for (uint32_t i = 0; i < t->merge_cap; i++) free(t->merge_key[i]);
    free(t->merge_key); free(t->merge_rank);
    free(t->special);
    free(t);
}

int32_t ds4x_tok_special_id(const ds4x_tokenizer *t, const char *s) {
    int32_t id = ht_get(t, s);
    if (id < 0) return -1;
    return (t->types[id] == TT_CONTROL || t->types[id] == TT_USER_DEFINED) ? id : -1;
}

uint32_t ds4x_tok_vocab_size(const ds4x_tokenizer *t) { return t->n_vocab; }
int32_t  ds4x_tok_bos(const ds4x_tokenizer *t) { return t->bos; }
int32_t  ds4x_tok_eos(const ds4x_tokenizer *t) { return t->eos; }

char *ds4x_tok_decode(const ds4x_tokenizer *t, const int32_t *ids, uint32_t n,
                      int keep_special) {
    /* worst case: every token a few UTF-8 chars → size generously, grow if
     * needed. */
    size_t cap = 256, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    for (uint32_t k = 0; k < n; k++) {
        int32_t id = ids[k];
        if (id < 0 || (uint32_t)id >= t->n_vocab) continue;
        if (!keep_special && (t->types[id] == TT_CONTROL || t->types[id] == TT_UNKNOWN))
            continue;
        const char *s = t->tokens[id];
        size_t si = 0, sl = strlen(s);
        while (si < sl) {
            int cp = utf8_get(s, sl, &si);
            int b = (cp >= 0 && cp < 0x200) ? t->uni_to_byte[cp] : -1;
            if (len + 4 >= cap) { cap *= 2; char *nb = realloc(out, cap); if (!nb) { free(out); return NULL; } out = nb; }
            if (b >= 0) out[len++] = (char)b;           /* byte-unicode → raw byte */
            else len += (size_t)utf8_put(out + len, cp); /* non-byte glyph: keep as-is */
        }
    }
    out[len] = '\0';
    return out;
}

/* Byte-BPE one plain-text run: map bytes → byte-unicode symbols, greedily
 * merge the lowest-rank adjacent pair until none remain, then emit ids.
 * Appends to out at *produced; only writes while *produced < cap.
 *
 * THIS IS THE REFERENCE, not the shipped path — encode_segment below picks.
 * Kept because it is short enough to read as a definition of what byte-BPE
 * means here, and because the fast version is asserted against it. */
static int encode_segment_scan(const ds4x_tokenizer *t, const char *text, size_t tlen,
                               int32_t *out, uint32_t cap, int64_t *produced) {
    if (tlen == 0) return 0;
    char **sym = (char **)malloc(tlen * sizeof(char *));
    if (!sym) return -1;
    size_t nsym = 0;
    for (size_t i = 0; i < tlen; i++) {
        int cp = t->byte_to_uni[(unsigned char)text[i]];
        char b[8]; int bl = utf8_put(b, cp); b[bl] = '\0';
        sym[nsym++] = strdup(b);
    }
    for (;;) {
        int best_rank = -1; size_t best_i = 0;
        char pair[1024];
        for (size_t i = 0; i + 1 < nsym; i++) {
            snprintf(pair, sizeof(pair), "%s %s", sym[i], sym[i + 1]);
            int32_t r = merge_get(t, pair);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
        }
        if (best_rank < 0) break;
        char *merged = (char *)malloc(strlen(sym[best_i]) + strlen(sym[best_i + 1]) + 1);
        strcpy(merged, sym[best_i]); strcat(merged, sym[best_i + 1]);
        free(sym[best_i]); free(sym[best_i + 1]);
        sym[best_i] = merged;
        memmove(&sym[best_i + 1], &sym[best_i + 2], (nsym - best_i - 2) * sizeof(char *));
        nsym--;
    }
    for (size_t i = 0; i < nsym; i++) {
        int32_t id = ht_get(t, sym[i]);
        if (id >= 0 && *produced < (int64_t)cap) out[*produced] = id;
        if (id >= 0) (*produced)++;
        free(sym[i]);
    }
    free(sym);
    return 0;
}

/* ---- byte-BPE, priority-queue form --------------------------------------
 * The scan above rescans EVERY adjacent pair to find the lowest rank, merges
 * one pair, and repeats: O(n) merges x an O(n) scan, with a snprintf and a hash
 * lookup inside the inner loop. Measured on Qwen3.5's vocabulary (2026-08-13,
 * GB10): 99 tokens 5 ms, 395 tokens 74 ms, 789 tokens 310 ms, 1578 tokens
 * 1.183 s. That last one is 35% of the whole request — and being quadratic it
 * would dominate everything else at 4K context, which is an ordinary prompt.
 *
 * A merge only changes the two pairs ADJACENT to it; the rest of the scan is
 * rediscovering what it already knew. So keep the candidates in a heap: pop the
 * best, merge it, push the at most two pairs that merge created. O(n log n).
 *
 * Symbols are SPANS of one converted buffer rather than individual strings,
 * because merging adjacent spans is just "extend the left one" — no malloc per
 * symbol and no copy per merge, which is most of the remaining constant.
 *
 * Stale heap entries are not removed; they are caught on pop by comparing each
 * symbol's current length against the length recorded when the candidate was
 * pushed. A symbol only ever grows, so a mismatch means the pair is gone. (Same
 * trick as llama.cpp's BPE.)
 *
 * ⚠ TIE-BREAKING IS PART OF THE CONTRACT. The scan takes the FIRST pair of the
 * lowest rank — strict `<` while walking left to right — so the heap orders by
 * (rank, then left index) to match. Symbol indices stay in text order because a
 * merge keeps the LEFT symbol's index, so "lowest index" still means "leftmost".
 * Get this wrong and tokenization changes on real prose while every short test
 * still passes. */
typedef struct { size_t start, len; int prev, next; } bpe_sym;
typedef struct { int32_t rank; int left, right; size_t lsz, rsz; } bpe_big;

static int bpe_less(const bpe_big *a, const bpe_big *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->left < b->left;
}

static void bpe_heap_push(bpe_big *h, size_t *n, bpe_big v) {
    size_t i = (*n)++;
    while (i > 0) {
        const size_t p = (i - 1) / 2;
        if (!bpe_less(&v, &h[p])) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = v;
}

static bpe_big bpe_heap_pop(bpe_big *h, size_t *n) {
    const bpe_big top = h[0];
    const bpe_big last = h[--(*n)];
    if (*n) {
        size_t i = 0;
        for (;;) {
            const size_t l = 2 * i + 1, r = l + 1;
            size_t best = i;
            int have = 0;
            bpe_big cand = last;
            if (l < *n && bpe_less(&h[l], &cand)) { cand = h[l]; best = l; have = 1; }
            if (r < *n && bpe_less(&h[r], &cand)) { best = r; have = 1; }
            if (!have) break;
            h[i] = h[best];
            i = best;
        }
        h[i] = last;
    }
    return top;
}

/* Build "<left> <right>" exactly as the scan's snprintf(pair, 1024, "%s %s")
 * did, TRUNCATION INCLUDED — a longer buffer here would look like a fix and
 * would silently retokenize inputs that produce very long merged symbols. */
static void bpe_add(const ds4x_tokenizer *t, const char *conv, const bpe_sym *sym,
                    bpe_big *heap, size_t *nheap, int l, int r) {
    char pair[1024];
    size_t n = 0;
    const bpe_sym *A = &sym[l], *B = &sym[r];
    for (size_t i = 0; i < A->len && n + 1 < sizeof(pair); i++) pair[n++] = conv[A->start + i];
    if (n + 1 < sizeof(pair)) pair[n++] = ' ';
    for (size_t i = 0; i < B->len && n + 1 < sizeof(pair); i++) pair[n++] = conv[B->start + i];
    pair[n] = '\0';
    const int32_t rank = merge_get(t, pair);
    if (rank < 0) return;
    bpe_big v; v.rank = rank; v.left = l; v.right = r; v.lsz = A->len; v.rsz = B->len;
    bpe_heap_push(heap, nheap, v);
}

static int encode_segment_heap(const ds4x_tokenizer *t, const char *text, size_t tlen,
                               int32_t *out, uint32_t cap, int64_t *produced) {
    if (tlen == 0) return 0;   /* the emit walk below starts at symbol 0 unconditionally */
    /* Each initial pair pushes once, and each of the <= tlen-1 merges pushes at
     * most two more; 3*tlen is a safe bound with room to spare. */
    char    *conv = (char *)malloc(tlen * 4 + 1);
    bpe_sym *sym  = (bpe_sym *)malloc(tlen * sizeof(bpe_sym));
    bpe_big *heap = (bpe_big *)malloc((3 * tlen + 8) * sizeof(bpe_big));
    if (!conv || !sym || !heap) { free(conv); free(sym); free(heap); return -1; }

    size_t clen = 0;
    for (size_t i = 0; i < tlen; i++) {
        const int cp = t->byte_to_uni[(unsigned char)text[i]];
        const size_t st = clen;
        clen += (size_t)utf8_put(conv + clen, cp);
        sym[i].start = st;
        sym[i].len   = clen - st;
        sym[i].prev  = (int)i - 1;
        sym[i].next  = (i + 1 < tlen) ? (int)(i + 1) : -1;
    }
    conv[clen] = '\0';

    size_t nheap = 0;
    for (size_t i = 0; i + 1 < tlen; i++)
        bpe_add(t, conv, sym, heap, &nheap, (int)i, (int)(i + 1));

    while (nheap) {
        const bpe_big b = bpe_heap_pop(heap, &nheap);
        bpe_sym *L = &sym[b.left], *R = &sym[b.right];
        if (L->len == 0 || R->len == 0) continue;            /* already consumed */
        if (L->len != b.lsz || R->len != b.rsz) continue;    /* stale candidate  */
        if (L->next != b.right) continue;                    /* no longer adjacent */
        L->len += R->len;                                    /* spans are contiguous */
        R->len = 0;
        L->next = R->next;
        if (R->next >= 0) sym[R->next].prev = b.left;
        if (L->prev >= 0) bpe_add(t, conv, sym, heap, &nheap, L->prev, b.left);
        if (L->next >= 0) bpe_add(t, conv, sym, heap, &nheap, b.left, L->next);
    }

    /* Emit in list order. ht_get needs a NUL-terminated key and the spans are
     * contiguous, so borrow the following byte and put it back. */
    for (int i = 0; i >= 0; i = sym[i].next) {
        const size_t end = sym[i].start + sym[i].len;
        const char saved = conv[end];
        conv[end] = '\0';
        const int32_t id = ht_get(t, conv + sym[i].start);
        conv[end] = saved;
        if (id >= 0 && *produced < (int64_t)cap) out[*produced] = id;
        if (id >= 0) (*produced)++;
    }

    free(conv); free(sym); free(heap);
    return 0;
}

/* IDLETOKEN_DS4X_TOK_SCAN=1 selects the reference loop. ds4x_tok_test asserts
 * the two agree token for token, so the switch is what makes that assertion
 * possible rather than a leftover. */
static int encode_segment(const ds4x_tokenizer *t, const char *text, size_t tlen,
                          int32_t *out, uint32_t cap, int64_t *produced) {
    /* Read every call, not cached in a static: the differential test flips this
     * between two encodes IN ONE PROCESS, and a cached value would make it
     * compare the fast path against itself and pass unconditionally. The other
     * runtime switches in this project (ds4x_dp4a_wanted and friends) re-read
     * for the same reason. getenv is nothing next to a BPE segment. */
    const char *e = getenv("IDLETOKEN_DS4X_TOK_SCAN");
    return (e && e[0] == '1') ? encode_segment_scan(t, text, tlen, out, cap, produced)
                              : encode_segment_heap(t, text, tlen, out, cap, produced);
}

/* Longest special/added token matching the input at `text` (verbatim), or -1.
 * `special` is pre-sorted longest-first so the first hit is the longest. */
static int32_t special_match_at(const ds4x_tokenizer *t, const char *text) {
    for (uint32_t j = 0; j < t->n_special; j++) {
        const char *lit = t->tokens[t->special[j]];
        size_t ll = strlen(lit);
        if (ll && strncmp(text, lit, ll) == 0) return t->special[j];
    }
    return -1;
}

/* Encode: split the input on verbatim special/added tokens (emitted as their
 * single id), byte-BPE the plain runs between them. */
int64_t ds4x_tok_encode(const ds4x_tokenizer *t, const char *text,
                        int32_t *out, uint32_t cap) {
    const size_t tlen = strlen(text);
    if (tlen == 0) return 0;
    int64_t produced = 0;
    size_t run_start = 0, i = 0;
    while (i < tlen) {
        int32_t sid = special_match_at(t, text + i);
        if (sid >= 0) {
            if (i > run_start &&
                encode_segment(t, text + run_start, i - run_start, out, cap, &produced) != 0)
                return -1;
            if (produced < (int64_t)cap) out[produced] = sid;
            produced++;
            i += strlen(t->tokens[sid]);
            run_start = i;
        } else {
            i++;
        }
    }
    if (tlen > run_start &&
        encode_segment(t, text + run_start, tlen - run_start, out, cap, &produced) != 0)
        return -1;
    return produced;
}

/* ChatML render → encode. Builds the prompt string then runs it through the
 * special-token-aware encoder (so <|im_start|>/<|im_end|> land as single ids). */
int64_t ds4x_tok_chat_apply(const ds4x_tokenizer *t,
                            const char *const *roles,
                            const char *const *contents, uint32_t n_msgs,
                            int add_generation_prompt,
                            int32_t *out, uint32_t cap) {
    if (ds4x_tok_special_id(t, "<|im_start|>") < 0 ||
        ds4x_tok_special_id(t, "<|im_end|>") < 0)
        return -1;   /* not a ChatML vocab — caller falls back */
    /* size the buffer: each message ≈ markers + role + content + newlines */
    size_t need = 64;
    for (uint32_t k = 0; k < n_msgs; k++)
        need += strlen(roles[k]) + strlen(contents[k]) + 40;
    char *buf = (char *)malloc(need);
    if (!buf) return -1;
    size_t p = 0;
    for (uint32_t k = 0; k < n_msgs; k++)
        p += (size_t)snprintf(buf + p, need - p, "<|im_start|>%s\n%s<|im_end|>\n",
                              roles[k], contents[k]);
    if (add_generation_prompt)
        p += (size_t)snprintf(buf + p, need - p, "<|im_start|>assistant\n");
    int64_t n = ds4x_tok_encode(t, buf, out, cap);
    free(buf);
    return n;
}
