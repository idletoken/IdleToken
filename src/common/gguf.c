/* gguf.c — read-only GGUF metadata access. See include/idletoken_gguf.h.
 * Cursor logic mirrors weights.c's embedded parser (kept private there);
 * consolidation of the two is deliberate future cleanup, not a behavior gap. */
#include "idletoken_gguf.h"
#include "idletoken_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u  /* "GGUF" little-endian */

/* GGUF metadata value type ids (spec order). */
enum {
    GT_U8 = 0, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL,
    GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64,
};

static uint64_t scalar_size(uint32_t t) {
    switch (t) {
        case GT_U8: case GT_I8: case GT_BOOL: return 1;
        case GT_U16: case GT_I16: return 2;
        case GT_U32: case GT_I32: case GT_F32: return 4;
        case GT_U64: case GT_I64: case GT_F64: return 8;
        default: return 0;
    }
}

typedef struct { const uint8_t *b; size_t p, n; int err; } gcur;
static uint32_t gc_u32(gcur *c) {
    if (c->p + 4 > c->n) { c->err = 1; return 0; }
    uint32_t v; memcpy(&v, c->b + c->p, 4); c->p += 4; return v;
}
static uint64_t gc_u64(gcur *c) {
    if (c->p + 8 > c->n) { c->err = 1; return 0; }
    uint64_t v; memcpy(&v, c->b + c->p, 8); c->p += 8; return v;
}
static void gc_skip(gcur *c, uint64_t n) {
    if (c->p + n > c->n) c->err = 1; else c->p += (size_t)n;
}
static uint64_t gc_str(gcur *c, char *out, size_t cap) {
    uint64_t len = gc_u64(c);
    if (c->err || c->p + len > c->n) { c->err = 1; return 0; }
    if (out && cap) {
        size_t k = (len < cap - 1) ? (size_t)len : cap - 1;
        memcpy(out, c->b + c->p, k);
        out[k] = '\0';
    }
    c->p += (size_t)len;
    return len;
}
static void gc_skip_value(gcur *c, uint32_t vtype, int depth) {
    if (c->err || depth > 8) { c->err = 1; return; }
    if (vtype == GT_STR) { gc_str(c, NULL, 0); return; }
    if (vtype == GT_ARR) {
        uint32_t it = gc_u32(c); uint64_t len = gc_u64(c);
        uint64_t sz = scalar_size(it);
        if (sz) gc_skip(c, sz * len);
        else for (uint64_t i = 0; i < len && !c->err; i++) gc_skip_value(c, it, depth + 1);
        return;
    }
    uint64_t sz = scalar_size(vtype);
    if (!sz) { c->err = 1; return; }
    gc_skip(c, sz);
}

typedef struct {
    char     key[128];
    uint32_t vtype;
    size_t   vpos;    /* offset of the value (for GT_ARR: of the item-type u32) */
} gguf_kv;

struct idletoken_gguf_meta {
    uint8_t *buf;      /* header region (owned) */
    size_t   len;
    gguf_kv *kv;
    uint64_t n_kv;
    uint64_t n_tensors;
    idletoken_gguf_tensor *tensors;   /* parsed directory */
    uint64_t data_offset;          /* absolute file offset of tensor data */
};

static const gguf_kv *find_kv(const idletoken_gguf_meta *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (strcmp(m->kv[i].key, key) == 0) return &m->kv[i];
    return NULL;
}

idletoken_gguf_meta *idletoken_gguf_meta_open(const char *path, char *err, size_t errlen) {
#define FAILF(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)
    FILE *f = fopen(path, "rb");
    uint8_t *buf = NULL;
    gguf_kv *kv = NULL;
    idletoken_gguf_meta *m = NULL;
    if (!f) { if (err) snprintf(err, errlen, "open %s failed", path); return NULL; }

    /* Read a header-sized prefix; grow and retry if the KV section (chat
     * templates can be 100s of KB) overruns it. Same strategy as weights.c. */
    size_t cap = 1u << 20;
    for (;;) {
        uint8_t *nb = (uint8_t *)realloc(buf, cap);
        if (!nb) FAILF("oom");
        buf = nb;
        rewind(f);
        size_t got = fread(buf, 1, cap, f);
        gcur c = { buf, 0, got, 0 };
        if (gc_u32(&c) != GGUF_MAGIC) FAILF("not a GGUF: %s", path);
        uint32_t version = gc_u32(&c);
        if (version < 2) FAILF("GGUF v%u too old (need >=2)", version);
        uint64_t n_tensors = gc_u64(&c);
        uint64_t n_kv = gc_u64(&c);
        if (n_kv > 65536) FAILF("implausible KV count %llu", (unsigned long long)n_kv);

        if (n_tensors > (1u << 20)) FAILF("implausible tensor count");

        free(kv);
        kv = (gguf_kv *)calloc(n_kv ? (size_t)n_kv : 1, sizeof(gguf_kv));
        if (!kv) FAILF("oom");
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv && !c.err; i++) {
            gc_str(&c, kv[i].key, sizeof(kv[i].key));
            kv[i].vtype = gc_u32(&c);
            kv[i].vpos = c.p;
            if (!strcmp(kv[i].key, "general.alignment") && kv[i].vtype == GT_U32) {
                gcur ac = c;
                alignment = gc_u32(&ac);
                if (alignment == 0) alignment = 32;
            }
            gc_skip_value(&c, kv[i].vtype, 0);
        }
        /* tensor directory follows the KVs */
        idletoken_gguf_tensor *ts = (idletoken_gguf_tensor *)
            calloc(n_tensors ? (size_t)n_tensors : 1, sizeof(*ts));
        if (!ts) FAILF("oom");
        for (uint64_t i = 0; i < n_tensors && !c.err; i++) {
            gc_str(&c, ts[i].name, sizeof(ts[i].name));
            ts[i].ndim = gc_u32(&c);
            if (ts[i].ndim > 4) { c.err = 1; break; }
            for (uint32_t d = 0; d < ts[i].ndim && !c.err; d++)
                ts[i].dims[d] = gc_u64(&c);
            ts[i].type = gc_u32(&c);
            ts[i].offset = gc_u64(&c);
        }
        if (c.err) {
            free(ts);
            if (got < cap) FAILF("truncated GGUF header: %s", path);
            cap *= 2;
            continue;   /* grow and retry */
        }
        m = (idletoken_gguf_meta *)calloc(1, sizeof(*m));
        if (!m) { free(ts); FAILF("oom"); }
        m->buf = buf; m->len = c.p; m->kv = kv;
        m->n_kv = n_kv; m->n_tensors = n_tensors;
        m->tensors = ts;
        m->data_offset = (c.p + alignment - 1) / alignment * alignment;
        fclose(f);
        return m;
    }
fail:
    if (f) fclose(f);
    free(buf);
    free(kv);
    free(m);
    return NULL;
#undef FAILF
}

void idletoken_gguf_meta_close(idletoken_gguf_meta *m) {
    if (!m) return;
    free(m->buf);
    free(m->kv);
    free(m->tensors);
    free(m);
}

int idletoken_gguf_tensor_info(const idletoken_gguf_meta *m, uint64_t idx,
                            idletoken_gguf_tensor *out) {
    if (idx >= m->n_tensors) return -1;
    *out = m->tensors[idx];
    return 0;
}

int idletoken_gguf_tensor_find(const idletoken_gguf_meta *m, const char *name,
                            idletoken_gguf_tensor *out) {
    for (uint64_t i = 0; i < m->n_tensors; i++)
        if (!strcmp(m->tensors[i].name, name)) { *out = m->tensors[i]; return 0; }
    return -1;
}

uint64_t idletoken_gguf_data_offset(const idletoken_gguf_meta *m) {
    return m->data_offset;
}

uint64_t idletoken_gguf_meta_n_tensors(const idletoken_gguf_meta *m) { return m->n_tensors; }

int idletoken_gguf_meta_str(const idletoken_gguf_meta *m, const char *key,
                         char *out, size_t cap) {
    const gguf_kv *k = find_kv(m, key);
    if (!k || k->vtype != GT_STR) return -1;
    gcur c = { m->buf, k->vpos, m->len, 0 };
    gc_str(&c, out, cap);
    return c.err ? -1 : 0;
}

/* Read the scalar at pos as a double (integers widened, f32/f64 as-is). */
static int scalar_as_double(const idletoken_gguf_meta *m, const gguf_kv *k, double *out) {
    gcur c = { m->buf, k->vpos, m->len, 0 };
    switch (k->vtype) {
        case GT_U8:  case GT_BOOL: { gc_skip(&c, 0); if (c.p + 1 > c.n) return -1; *out = m->buf[c.p]; return 0; }
        case GT_I8:  { if (c.p + 1 > c.n) return -1; *out = (int8_t)m->buf[c.p]; return 0; }
        case GT_U16: { uint16_t v; if (c.p + 2 > c.n) return -1; memcpy(&v, m->buf + c.p, 2); *out = v; return 0; }
        case GT_I16: { int16_t v;  if (c.p + 2 > c.n) return -1; memcpy(&v, m->buf + c.p, 2); *out = v; return 0; }
        case GT_U32: { uint32_t v = gc_u32(&c); if (c.err) return -1; *out = v; return 0; }
        case GT_I32: { uint32_t v = gc_u32(&c); if (c.err) return -1; *out = (int32_t)v; return 0; }
        case GT_F32: { uint32_t v = gc_u32(&c); if (c.err) return -1; float f; memcpy(&f, &v, 4); *out = f; return 0; }
        case GT_U64: { uint64_t v = gc_u64(&c); if (c.err) return -1; *out = (double)v; return 0; }
        case GT_I64: { uint64_t v = gc_u64(&c); if (c.err) return -1; *out = (double)(int64_t)v; return 0; }
        case GT_F64: { uint64_t v = gc_u64(&c); if (c.err) return -1; double d; memcpy(&d, &v, 8); *out = d; return 0; }
        default: return -1;
    }
}

int idletoken_gguf_meta_u32(const idletoken_gguf_meta *m, const char *key, uint32_t *out) {
    const gguf_kv *k = find_kv(m, key);
    double d;
    if (!k || scalar_as_double(m, k, &d) != 0) return -1;
    if (d < 0 || d > (double)UINT32_MAX) return -1;
    *out = (uint32_t)d;
    return 0;
}

int idletoken_gguf_meta_f32(const idletoken_gguf_meta *m, const char *key, float *out) {
    const gguf_kv *k = find_kv(m, key);
    double d;
    if (!k || scalar_as_double(m, k, &d) != 0) return -1;
    *out = (float)d;
    return 0;
}

int idletoken_gguf_meta_arr_len(const idletoken_gguf_meta *m, const char *key, uint64_t *out) {
    const gguf_kv *k = find_kv(m, key);
    if (!k || k->vtype != GT_ARR) return -1;
    gcur c = { m->buf, k->vpos, m->len, 0 };
    (void)gc_u32(&c);            /* item type */
    uint64_t len = gc_u64(&c);
    if (c.err) return -1;
    *out = len;
    return 0;
}

/* Position a cursor at array `key`, returning item type + count, or -1. */
static int arr_open(const idletoken_gguf_meta *m, const char *key, gcur *c,
                    uint32_t *item_type, uint64_t *count) {
    const gguf_kv *k = find_kv(m, key);
    if (!k || k->vtype != GT_ARR) return -1;
    c->b = m->buf; c->p = k->vpos; c->n = m->len; c->err = 0;
    *item_type = gc_u32(c);
    *count = gc_u64(c);
    return c->err ? -1 : 0;
}

int64_t idletoken_gguf_meta_arr_str(const idletoken_gguf_meta *m, const char *key,
                                 uint64_t idx, char *out, size_t cap) {
    gcur c; uint32_t it; uint64_t n;
    if (arr_open(m, key, &c, &it, &n) != 0 || it != GT_STR || idx >= n) return -1;
    for (uint64_t i = 0; i < idx; i++) { gc_str(&c, NULL, 0); if (c.err) return -1; }
    int64_t len = (int64_t)gc_str(&c, out, cap);
    return c.err ? -1 : len;
}

int idletoken_gguf_meta_arr_str_begin(const idletoken_gguf_meta *m, const char *key,
                                   idletoken_gguf_str_iter *it, uint64_t *count) {
    gcur c; uint32_t item; uint64_t n;
    if (!it || arr_open(m, key, &c, &item, &n) != 0 || item != GT_STR) return -1;
    it->b = c.b; it->p = c.p; it->n = c.n; it->err = 0;
    it->left = (unsigned long long)n;
    if (count) *count = n;
    return 0;
}

int64_t idletoken_gguf_meta_arr_str_next(idletoken_gguf_str_iter *it, char *out, size_t cap) {
    if (!it || it->err || it->left == 0) return -1;
    gcur c = { it->b, it->p, it->n, 0 };
    const int64_t len = (int64_t)gc_str(&c, out, cap);
    if (c.err) { it->err = 1; return -1; }
    it->p = c.p;
    it->left--;
    return len;
}

int idletoken_gguf_meta_arr_i32(const idletoken_gguf_meta *m, const char *key,
                             uint64_t idx, int32_t *out) {
    gcur c; uint32_t it; uint64_t n;
    if (arr_open(m, key, &c, &it, &n) != 0 || idx >= n) return -1;
    const uint64_t sz = scalar_size(it);
    if (sz == 0) return -1;                       /* string/nested array */
    gc_skip(&c, sz * idx);
    if (c.err) return -1;
    switch (it) {
        case GT_U8:  case GT_BOOL: *out = m->buf[c.p]; return 0;
        case GT_I8:  *out = (int8_t)m->buf[c.p]; return 0;
        case GT_U16: { uint16_t v; memcpy(&v, m->buf + c.p, 2); *out = v; return 0; }
        case GT_I16: { int16_t v;  memcpy(&v, m->buf + c.p, 2); *out = v; return 0; }
        case GT_U32: case GT_I32: { uint32_t v = gc_u32(&c); return c.err ? -1 : (*out = (int32_t)v, 0); }
        case GT_U64: case GT_I64: { uint64_t v = gc_u64(&c); return c.err ? -1 : (*out = (int32_t)v, 0); }
        default: return -1;
    }
}

/* ---- model identity ------------------------------------------------------
 * See idletoken_gguf.h for what this covers and what it deliberately does not.
 * Streams the metadata region in 1 MiB chunks: on a sparse layer shard the
 * region is present (it is what the fetcher needs to parse the index), while
 * reading the whole 80 GiB file is neither affordable nor possible there. */
/* Whole-file digest — the value curated manifests pin. See the header for why
 * it is separate from the metadata identity above and must stay off startup
 * paths. */
int idletoken_gguf_file_sha256(const char *path, uint8_t out[32],
                               unsigned progress_mib,
                               char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err && errlen) snprintf(err, errlen, "open %s failed", path);
        return -1;
    }
    enum { CHUNK = 1u << 20 };
    unsigned char *buf = (unsigned char *)malloc(CHUNK);
    if (!buf) { fclose(f); if (err && errlen) snprintf(err, errlen, "oom"); return -1; }

    idletoken_sha256_ctx c;
    idletoken_sha256_init(&c);
    uint64_t done = 0, next_report = (uint64_t)progress_mib << 20;
    for (;;) {
        size_t got = fread(buf, 1, CHUNK, f);
        if (got == 0) break;
        idletoken_sha256_update(&c, buf, got);
        done += got;
        if (progress_mib > 0 && done >= next_report) {
            fprintf(stderr, "coord: hashing GGUF… %llu MiB\n",
                    (unsigned long long)(done >> 20));
            fflush(stderr);
            next_report = done + ((uint64_t)progress_mib << 20);
        }
    }
    int read_err = ferror(f);
    free(buf);
    fclose(f);
    if (read_err) {
        if (err && errlen) snprintf(err, errlen, "read error on %s", path);
        return -1;
    }
    if (done == 0) {
        if (err && errlen) snprintf(err, errlen, "%s is empty", path);
        return -1;
    }
    idletoken_sha256_final(&c, out);
    return 0;
}

int idletoken_gguf_identity(const char *path, uint8_t out[32],
                         char *err, size_t errlen) {
    char merr[256] = "";
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(path, merr, sizeof merr);
    if (!m) {
        if (err && errlen) snprintf(err, errlen, "gguf open: %s", merr);
        return -1;
    }
    const uint64_t n = idletoken_gguf_data_offset(m);
    idletoken_gguf_meta_close(m);
    if (n == 0) {
        if (err && errlen) snprintf(err, errlen, "gguf has no data offset");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err && errlen) snprintf(err, errlen, "open %s failed", path);
        return -1;
    }
    enum { CHUNK = 1u << 20 };
    unsigned char *buf = (unsigned char *)malloc(CHUNK);
    if (!buf) { fclose(f); if (err && errlen) snprintf(err, errlen, "oom"); return -1; }

    idletoken_sha256_ctx c;
    idletoken_sha256_init(&c);
    /* Bind the declared boundary into the digest before any file bytes: a
     * metadata-only fixture ends BEFORE data_offset (the alignment padding is
     * simply absent), so we hash to EOF rather than demanding data_offset
     * bytes. Hashing the two lengths as well means a truncated file can never
     * collide with the full one just because it shares a prefix. */
    uint8_t hdr[16];
    for (int i = 0; i < 8; i++) hdr[i] = (uint8_t)(n >> (8 * i));
    uint64_t hashed = 0;
    while (hashed < n) {
        uint64_t left = n - hashed;
        size_t want = left > CHUNK ? CHUNK : (size_t)left;
        size_t got = fread(buf, 1, want, f);
        if (got == 0) break;                     /* EOF: metadata-only file */
        idletoken_sha256_update(&c, buf, got);
        hashed += got;
        if (got < want) break;
    }
    if (hashed == 0) {
        free(buf); fclose(f);
        if (err && errlen) snprintf(err, errlen, "no metadata bytes readable");
        return -1;
    }
    for (int i = 0; i < 8; i++) hdr[8 + i] = (uint8_t)(hashed >> (8 * i));
    idletoken_sha256_update(&c, hdr, sizeof hdr);
    idletoken_sha256_final(&c, out);
    free(buf);
    fclose(f);
    return 0;
}
