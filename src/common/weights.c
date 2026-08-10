/* IdleToken layer-shard weight fetch — see include/idletoken_weights.h.
 *
 * A pipeline worker fetches ONLY the byte ranges it needs (header + shared
 * tensors + its [lo,hi) layers) from the weight repo over HTTP byte-range, and
 * materializes a sparse partial GGUF locally (original apparent size, holes for
 * skipped layers). ds4's loader accepts this unchanged (bounds checks are
 * arithmetic against the apparent size; skipped layers' bytes are never read).
 *
 * The repo hosts the master GGUF at <base_url> plus a line-based index at
 * <base_url>.idx (scripts/gguf_shard.py idx):
 *     line 1: "<file_size> <tensor_data_pos> <n_tensors>"
 *     then n: "<layer> <offset> <bytes>"   (layer -1 = shared/global)
 * We reuse the same needed-ranges logic as the Python `ranges` command. */
#include "idletoken_weights.h"
#include "idletoken_net.h"   /* idletoken_connect_tcp */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #define OPEN_FLAGS (O_CREAT | O_WRONLY | O_BINARY)
  #define RDONLY_FLAGS (O_RDONLY | O_BINARY)
#else
  #include <signal.h>
  #include <sys/socket.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
  #include <fcntl.h>
  #define OPEN_FLAGS (O_CREAT | O_WRONLY)
  #define RDONLY_FLAGS (O_RDONLY)
#endif

/* Last socket error. Winsock does not set errno, so a bare strerror(errno)
 * after a failed send/recv on Windows prints a stale or bogus reason — which is
 * why a dropped weight transfer used to report nothing usable at all. */
static int sock_err(void) {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

/* ---- tiny portable 64-bit file helpers ---------------------------------- */

/* Make the file sparse so the unwritten holes cost no disk (critical on
 * Windows/NTFS, where extending a file otherwise allocates every byte — a PC
 * with 41GB free could not hold an 80GB apparent file otherwise). No-op where
 * holes are automatic (Linux/ext4/xfs). */
static void file_make_sparse(int fd) {
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h != INVALID_HANDLE_VALUE) {
        DWORD ret = 0;
        DeviceIoControl(h, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &ret, NULL);
    }
#else
    (void)fd; /* ftruncate + gaps are naturally sparse */
#endif
}

static int file_truncate64(int fd, uint64_t size) {
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    LARGE_INTEGER li; li.QuadPart = (LONGLONG)size;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) return -1;
    return SetEndOfFile(h) ? 0 : -1;
#else
    return ftruncate(fd, (off_t)size);
#endif
}

static int file_seek64(int fd, uint64_t off) {
#ifdef _WIN32
    return _lseeki64(fd, (long long)off, SEEK_SET) < 0 ? -1 : 0;
#else
    return lseek(fd, (off_t)off, SEEK_SET) < 0 ? -1 : 0;
#endif
}

static uint64_t file_size_of(const char *path) {
#ifdef _WIN32
    struct _stati64 st;
    if (_stati64(path, &st) != 0) return UINT64_MAX;
#else
    struct stat st;
    if (stat(path, &st) != 0) return UINT64_MAX;
#endif
    return (uint64_t)st.st_size;
}

/* ---- minimal HTTP/1.1 client (raw socket, LAN, no TLS) ------------------- */

/* Parse http://host[:port]/path into host_port ("host:port") and path. */
static int parse_http_url(const char *url, char *host_port, size_t hp_cap,
                          char *path, size_t path_cap) {
    if (strncmp(url, "http://", 7) != 0) return -1;
    const char *rest = url + 7;
    const char *slash = strchr(rest, '/');
    const char *host_end = slash ? slash : rest + strlen(rest);
    size_t hlen = (size_t)(host_end - rest);
    if (hlen == 0 || hlen >= hp_cap) return -1;
    memcpy(host_port, rest, hlen);
    host_port[hlen] = '\0';
    if (!strchr(host_port, ':')) {  /* default port 80 */
        if (hlen + 3 >= hp_cap) return -1;
        strcat(host_port, ":80");
    }
    snprintf(path, path_cap, "%s", slash ? slash : "/");
    return 0;
}

/* Send "GET <path>" with an optional Range header. r0>=0 requests bytes r0..r1. */
static int http_send_get(int fd, const char *host_port, const char *path,
                         long long r0, long long r1) {
    char req[1024];
    int n;
    if (r0 >= 0) {
        n = snprintf(req, sizeof(req),
            "GET %s HTTP/1.1\r\nHost: %s\r\nRange: bytes=%lld-%lld\r\n"
            "Connection: close\r\n\r\n", path, host_port, r0, r1);
    } else {
        n = snprintf(req, sizeof(req),
            "GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n",
            path, host_port);
    }
    if (n <= 0) return -1;
    size_t off = 0;
    while (off < (size_t)n) {
        int w = (int)send(fd, req + off, (int)((size_t)n - off), 0);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

/* Read and parse the response head. Returns HTTP status (>=100) on success,
 * -1 on error. Fills *content_len (-1 if absent). body_pre and body_pre_len get
 * any body bytes already read past the header terminator (caller owns copying
 * them out before draining more). buf is a scratch of size bufcap. */
static int http_read_head(int fd, char *buf, size_t bufcap, long long *content_len,
                          char **body_pre, size_t *body_pre_len) {
    size_t have = 0;
    char *hdr_end = NULL;
    while (have < bufcap - 1) {
        int r = (int)recv(fd, buf + have, (int)(bufcap - 1 - have), 0);
        if (r <= 0) break;
        have += (size_t)r;
        buf[have] = '\0';
        hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) break;
    }
    if (!hdr_end) return -1;
    int status = -1;
    if (strncmp(buf, "HTTP/1.", 7) == 0) status = atoi(buf + 9);
    *content_len = -1;
    /* case-insensitive-ish search for Content-Length */
    for (char *p = buf; p < hdr_end; p++) {
        if ((p[0] == 'C' || p[0] == 'c') &&
            (strncmp(p, "Content-Length:", 15) == 0 ||
             strncmp(p, "content-length:", 15) == 0)) {
            *content_len = atoll(p + 15);
            break;
        }
    }
    *body_pre = hdr_end + 4;
    *body_pre_len = have - (size_t)(*body_pre - buf);
    return status;
}

/* GET the whole small resource at `path` into a malloc'd NUL-terminated buffer.
 * Returns buffer (caller frees) or NULL. */
static char *http_get_all(const char *host_port, const char *path) {
    int fd = idletoken_connect_tcp(host_port);
    if (fd < 0) return NULL;
    if (http_send_get(fd, host_port, path, -1, -1) != 0) { idletoken_close_fd(fd); return NULL; }
    char head[4096];
    long long clen; char *pre; size_t pre_len;
    int status = http_read_head(fd, head, sizeof(head), &clen, &pre, &pre_len);
    if (status != 200) { idletoken_close_fd(fd); return NULL; }
    size_t cap = (clen > 0) ? (size_t)clen + 1 : 65536;
    char *out = (char *)malloc(cap);
    if (!out) { idletoken_close_fd(fd); return NULL; }
    size_t len = 0;
    if (pre_len) { if (pre_len >= cap) { free(out); idletoken_close_fd(fd); return NULL; }
                   memcpy(out, pre, pre_len); len = pre_len; }
    for (;;) {
        if (len + 1 >= cap) { cap *= 2; char *n = realloc(out, cap); if (!n) { free(out); idletoken_close_fd(fd); return NULL; } out = n; }
        int r = (int)recv(fd, out + len, (int)(cap - 1 - len), 0);
        if (r <= 0) break;
        len += (size_t)r;
        if (clen > 0 && len >= (size_t)clen) break;
    }
    out[len] = '\0';
    idletoken_close_fd(fd);
    return out;
}

/* GET bytes [r0,r1] and write them to fd at offset r0. Returns 0 on success. */
static int http_get_range_to_file(const char *host_port, const char *path,
                                  uint64_t r0, uint64_t r1, int outfd) {
    int fd = idletoken_connect_tcp(host_port);
    if (fd < 0) return -1;
    if (http_send_get(fd, host_port, path, (long long)r0, (long long)r1) != 0) {
        idletoken_close_fd(fd); return -1;
    }
    char head[4096];
    long long clen; char *pre; size_t pre_len;
    int status = http_read_head(fd, head, sizeof(head), &clen, &pre, &pre_len);
    /* 206 = partial (Range honored); 200 = server ignored Range (still ok if it
     * streams the whole file, but that defeats the purpose) — accept only 206. */
    if (status != 206) {
        fprintf(stderr, "idletoken-weights: %llu-%llu: HTTP status %d (want 206)\n",
                (unsigned long long)r0, (unsigned long long)r1, status);
        idletoken_close_fd(fd); return -1;
    }
    uint64_t want = r1 - r0 + 1;
    if (file_seek64(outfd, r0) != 0) { idletoken_close_fd(fd); return -1; }
    uint64_t written = 0;
    if (pre_len) {
        size_t take = (pre_len > want) ? (size_t)want : pre_len;
        if (write(outfd, pre, (unsigned)take) != (int)take) { idletoken_close_fd(fd); return -1; }
        written += take;
    }
    char chunk[1 << 20];
    while (written < want) {
        int r = (int)recv(fd, chunk, (int)sizeof(chunk), 0);
        if (r <= 0) {
            /* Short read = dropped/reset connection, not a protocol error. Say
             * WHICH error and how far we got — a silent truncation here used to
             * surface only as "range fetch failed" with no way to tell a reset
             * from a disk-full write. */
            fprintf(stderr,
                    "idletoken-weights: %llu-%llu: recv %s after %llu/%llu bytes (err %d)\n",
                    (unsigned long long)r0, (unsigned long long)r1,
                    r == 0 ? "EOF" : "error", (unsigned long long)written,
                    (unsigned long long)want, sock_err());
            break;
        }
        size_t take = ((uint64_t)r > want - written) ? (size_t)(want - written) : (size_t)r;
        if (write(outfd, chunk, (unsigned)take) != (int)take) {
            fprintf(stderr, "idletoken-weights: %llu-%llu: write failed at %llu: %s\n",
                    (unsigned long long)r0, (unsigned long long)r1,
                    (unsigned long long)(r0 + written), strerror(errno));
            idletoken_close_fd(fd); return -1;
        }
        written += take;
    }
    idletoken_close_fd(fd);
    return (written == want) ? 0 : -1;
}

/* One needed range can be tens of GB (a 2-node split hands the last stage 20
 * contiguous layers = 37 GB). Asking for that in a single HTTP request means a
 * single transient socket error throws away everything transferred so far, and
 * a sustained multi-GB send is exactly where transient errors live (Windows
 * WSAENOBUFS under memory pressure being the one that bit us). Fetch in bounded
 * pieces instead: each piece writes at its own absolute offset, so a retry is
 * idempotent and costs one piece, not the whole range. */
#define SHARD_PIECE_BYTES   (512ull << 20)
#define SHARD_PIECE_RETRIES 4

static void shard_backoff(int attempt) {
#ifdef _WIN32
    Sleep((DWORD)(1000 * attempt));
#else
    sleep((unsigned)attempt);
#endif
}

/* Fetch [s,e) in SHARD_PIECE_BYTES pieces, retrying each piece. Returns 0 ok. */
static int fetch_range_pieces(const char *host_port, const char *path,
                              uint64_t s, uint64_t e, int outfd,
                              uint64_t *done_bytes, uint64_t total_bytes) {
    for (uint64_t p = s; p < e; ) {
        uint64_t q = (e - p > SHARD_PIECE_BYTES) ? p + SHARD_PIECE_BYTES : e;
        int ok = 0;
        for (int attempt = 1; attempt <= SHARD_PIECE_RETRIES; attempt++) {
            if (http_get_range_to_file(host_port, path, p, q - 1, outfd) == 0) { ok = 1; break; }
            fprintf(stderr, "idletoken-weights: piece %llu-%llu failed (attempt %d/%d)\n",
                    (unsigned long long)p, (unsigned long long)q,
                    attempt, SHARD_PIECE_RETRIES);
            if (attempt < SHARD_PIECE_RETRIES) shard_backoff(attempt);
        }
        if (!ok) return -1;
        *done_bytes += q - p;
        fprintf(stderr, "idletoken-weights: %.2f/%.2f GB (%.0f%%)\n",
                *done_bytes / 1e9, total_bytes / 1e9,
                100.0 * (double)*done_bytes / (double)(total_bytes ? total_bytes : 1));
        p = q;
    }
    return 0;
}

/* ---- range set from the .idx manifest ----------------------------------- */

typedef struct { uint64_t s, e; } range_t;

/* Parse the .idx, compute the merged byte ranges this [lo,hi) worker needs, and
 * return them (caller frees). Sets *file_size. Returns count, -1 on error. */
static int compute_ranges(const char *idx, unsigned lo, unsigned hi,
                          uint64_t *file_size, range_t **out_ranges) {
    const char *p = idx;
    uint64_t fsz = 0, tdp = 0; long long ntensors = 0;
    if (sscanf(p, "%llu %llu %lld", (unsigned long long *)&fsz,
               (unsigned long long *)&tdp, &ntensors) != 3 || ntensors <= 0)
        return -1;
    range_t *r = (range_t *)malloc(sizeof(range_t) * (size_t)(ntensors + 1));
    if (!r) return -1;
    int nr = 0;
    r[nr].s = 0; r[nr].e = tdp; nr++;          /* header + tensor directory */
    /* advance past line 1 */
    p = strchr(p, '\n'); if (!p) { free(r); return -1; } p++;
    for (long long i = 0; i < ntensors && *p; i++) {
        long long layer; unsigned long long off, bytes;
        if (sscanf(p, "%lld %llu %llu", &layer, &off, &bytes) != 3) { free(r); return -1; }
        if (bytes > 0 && (layer < 0 || ((unsigned)layer >= lo && (unsigned)layer < hi))) {
            r[nr].s = off; r[nr].e = off + bytes; nr++;
        }
        p = strchr(p, '\n'); if (!p) break; p++;
    }
    /* sort by start (insertion sort — nr is small, ~ shared + range tensors) */
    for (int i = 1; i < nr; i++) {
        range_t k = r[i]; int j = i - 1;
        while (j >= 0 && r[j].s > k.s) { r[j + 1] = r[j]; j--; }
        r[j + 1] = k;
    }
    /* merge adjacent/overlapping */
    int m = 0;
    for (int i = 0; i < nr; i++) {
        if (m > 0 && r[i].s <= r[m - 1].e) {
            if (r[i].e > r[m - 1].e) r[m - 1].e = r[i].e;
        } else {
            r[m++] = r[i];
        }
    }
    *file_size = fsz;
    *out_ranges = r;
    return m;
}

/* ---- GGUF index (coordinator side) --------------------------------------
 * Emit the C-friendly `.idx` manifest (`<file_size> <tensor_data_pos> <n>` then
 * per-tensor `<layer> <offset> <bytes>`) so the coordinator's repo is
 * self-contained — no Python at runtime. Block geometry is verbatim from ds4.c
 * gguf_types[]; the same table drives scripts/gguf_shard.py. */

static const struct { unsigned be, bb; } GGUF_GEOM[] = {
    {1,4},{1,2},{32,18},{32,20},{0,0},{0,0},{32,22},{32,24},{32,34},{32,40},
    {256,84},{256,110},{256,144},{256,176},{256,210},{256,292},{256,66},{256,74},
    {256,98},{256,110},{256,50},{256,110},{256,82},{256,136},{1,1},{1,2},{1,4},
    {1,8},{1,8},{256,56},{1,2},
};
#define GGUF_GEOM_N (sizeof(GGUF_GEOM)/sizeof(GGUF_GEOM[0]))

/* GGUF metadata value type scalar sizes (0 = string/array/unknown). */
static uint64_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;          /* u8/i8/bool */
        case 2: case 3: return 2;                  /* u16/i16 */
        case 4: case 5: case 6: return 4;          /* u32/i32/f32 */
        case 10: case 11: case 12: return 8;       /* u64/i64/f64 */
        default: return 0;
    }
}

/* Cursor over the in-memory GGUF header. */
typedef struct { const uint8_t *b; size_t p, n; int err; } gcur;
static uint32_t gc_u32(gcur *c) {
    if (c->p + 4 > c->n) { c->err = 1; return 0; }
    uint32_t v; memcpy(&v, c->b + c->p, 4); c->p += 4; return v;
}
static uint64_t gc_u64(gcur *c) {
    if (c->p + 8 > c->n) { c->err = 1; return 0; }
    uint64_t v; memcpy(&v, c->b + c->p, 8); c->p += 8; return v;
}
static void gc_skip(gcur *c, uint64_t n) { if (c->p + n > c->n) c->err = 1; else c->p += n; }
static uint64_t gc_str(gcur *c, char *out, size_t cap) {  /* returns length; copies up to cap-1 */
    uint64_t len = gc_u64(c);
    if (c->err || c->p + len > c->n) { c->err = 1; return 0; }
    if (out && cap) { size_t k = (len < cap - 1) ? (size_t)len : cap - 1; memcpy(out, c->b + c->p, k); out[k] = '\0'; }
    c->p += len;
    return len;
}
static void gc_skip_value(gcur *c, uint32_t vtype, int depth) {
    if (c->err || depth > 8) { c->err = 1; return; }
    if (vtype == 8) { gc_str(c, NULL, 0); return; }               /* string */
    if (vtype == 9) {                                             /* array */
        uint32_t it = gc_u32(c); uint64_t len = gc_u64(c);
        uint64_t sz = gguf_scalar_size(it);
        if (sz) gc_skip(c, sz * len);
        else for (uint64_t i = 0; i < len && !c->err; i++) gc_skip_value(c, it, depth + 1);
        return;
    }
    uint64_t sz = gguf_scalar_size(vtype);
    if (!sz) { c->err = 1; return; }
    gc_skip(c, sz);
}

static int layer_of_name(const char *name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    const char *r = name + 4; int v = 0, any = 0;
    while (*r >= '0' && *r <= '9') { v = v * 10 + (*r - '0'); r++; any = 1; }
    return (any && *r == '.') ? v : -1;
}

int idletoken_write_idx(const char *gguf_path, const char *idx_path) {
    uint64_t fsz = file_size_of(gguf_path);
    if (fsz == UINT64_MAX) { fprintf(stderr, "idletoken-weights: idx: cannot stat %s\n", gguf_path); return -1; }
    /* Read the header region (grow if the tensor directory is bigger). */
    size_t cap = (fsz < (64u << 20)) ? (size_t)fsz : (64u << 20);
    uint8_t *buf = NULL; FILE *f = fopen(gguf_path, "rb");
    if (!f) { fprintf(stderr, "idletoken-weights: idx: open %s failed\n", gguf_path); return -1; }
    FILE *out = NULL;
    for (;;) {
        buf = (uint8_t *)realloc(buf, cap);
        if (!buf) { fclose(f); return -1; }
        rewind(f);
        size_t got = fread(buf, 1, cap, f);
        gcur c = { buf, 0, got, 0 };
        uint32_t magic = gc_u32(&c);
        if (magic != 0x46554747u) { fprintf(stderr, "idletoken-weights: idx: not a GGUF (%s)\n", gguf_path); free(buf); fclose(f); return -1; }
        gc_u32(&c);                       /* version */
        uint64_t n_tensors = gc_u64(&c);
        uint64_t n_kv = gc_u64(&c);
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv && !c.err; i++) {
            char key[128]; gc_str(&c, key, sizeof(key));
            uint32_t vt = gc_u32(&c);
            if (!strcmp(key, "general.alignment") && vt == 4) alignment = gc_u32(&c);
            else gc_skip_value(&c, vt, 0);
        }
        /* tensor directory */
        typedef struct { long layer; uint64_t off, bytes; } tent;
        tent *t = (tent *)malloc(sizeof(tent) * (size_t)(n_tensors ? n_tensors : 1));
        if (!t) { free(buf); fclose(f); return -1; }
        int ok = 1;
        for (uint64_t i = 0; i < n_tensors && !c.err; i++) {
            char nm[256]; gc_str(&c, nm, sizeof(nm));
            uint32_t ndim = gc_u32(&c);
            uint64_t elems = 1;
            for (uint32_t d = 0; d < ndim && !c.err; d++) elems *= gc_u64(&c);
            uint32_t ty = gc_u32(&c);
            uint64_t rel = gc_u64(&c);
            if (ty >= GGUF_GEOM_N || GGUF_GEOM[ty].be == 0) { ok = 0; break; }
            uint64_t blocks = (elems + GGUF_GEOM[ty].be - 1) / GGUF_GEOM[ty].be;
            t[i].bytes = blocks * GGUF_GEOM[ty].bb;
            t[i].layer = layer_of_name(nm);
            t[i].off = rel;   /* + tensor_data_pos below */
        }
        if (c.err) {
            /* header bigger than we read — grow and retry (unless whole file). */
            free(t);
            if (cap >= fsz) { fprintf(stderr, "idletoken-weights: idx: parse error\n"); free(buf); fclose(f); return -1; }
            cap = (cap * 2 < fsz) ? cap * 2 : (size_t)fsz;
            continue;
        }
        uint64_t tdp = (c.p + alignment - 1) / alignment * alignment;
        out = fopen(idx_path, "w");
        if (!out) { fprintf(stderr, "idletoken-weights: idx: write %s failed\n", idx_path); free(t); free(buf); fclose(f); return -1; }
        fprintf(out, "%llu %llu %llu\n", (unsigned long long)fsz,
                (unsigned long long)tdp, (unsigned long long)n_tensors);
        for (uint64_t i = 0; i < n_tensors; i++)
            fprintf(out, "%ld %llu %llu\n", t[i].layer,
                    (unsigned long long)(tdp + t[i].off), (unsigned long long)t[i].bytes);
        fclose(out);
        free(t);
        (void)ok;
        break;
    }
    free(buf);
    fclose(f);
    fprintf(stderr, "idletoken-weights: wrote index %s\n", idx_path);
    return 0;
}

/* ---- weight repo server (coordinator side) ------------------------------ */

/* Send a small text response head. */
static void srv_send_head(int fd, int status, const char *reason,
                          long long clen, long long r0, long long r1, long long total) {
    char h[512];
    int n;
    if (status == 206) {
        n = snprintf(h, sizeof(h),
            "HTTP/1.1 206 %s\r\nContent-Type: application/octet-stream\r\n"
            "Accept-Ranges: bytes\r\nContent-Range: bytes %lld-%lld/%lld\r\n"
            "Content-Length: %lld\r\nConnection: close\r\n\r\n",
            reason, r0, r1, total, clen);
    } else {
        n = snprintf(h, sizeof(h),
            "HTTP/1.1 %d %s\r\nContent-Type: application/octet-stream\r\n"
            "Accept-Ranges: bytes\r\nContent-Length: %lld\r\nConnection: close\r\n\r\n",
            status, reason, clen);
    }
    if (n > 0) { ssize_t w = send(fd, h, (int)n, 0); (void)w; }
}

/* Handle one GET (with optional Range) for a file under `dir`. */
static void srv_handle(int cfd, const char *dir) {
    char buf[8192];
    size_t have = 0;
    char *he = NULL;
    while (have < sizeof(buf) - 1) {
        int r = (int)recv(cfd, buf + have, (int)(sizeof(buf) - 1 - have), 0);
        if (r <= 0) break;
        have += (size_t)r;
        buf[have] = '\0';
        if ((he = strstr(buf, "\r\n\r\n")) != NULL) break;
    }
    if (!he) return;
    if (strncmp(buf, "GET ", 4) != 0) { srv_send_head(cfd, 405, "Method Not Allowed", 0, 0, 0, 0); return; }
    /* Parse the Range header first — before we carve up the request line. */
    long long r0 = -1, r1 = -1;
    char *rng = strstr(buf, "Range:");
    if (!rng) rng = strstr(buf, "range:");
    if (rng && rng < he) sscanf(rng, "%*[^=]=%lld-%lld", &r0, &r1);
    /* Extract the request-target (GET <path> HTTP/1.1) without mutating buf. */
    char *p = buf + 4;
    char *sp = strchr(p, ' ');
    if (!sp || sp > he) return;
    char reltmp[1024];
    size_t plen = (size_t)(sp - p);
    if (plen >= sizeof(reltmp)) plen = sizeof(reltmp) - 1;
    memcpy(reltmp, p, plen); reltmp[plen] = '\0';
    const char *rel = (reltmp[0] == '/') ? reltmp + 1 : reltmp;
    if (strstr(rel, "..")) { srv_send_head(cfd, 403, "Forbidden", 0, 0, 0, 0); return; }
    char path[1200];
    size_t dl = strlen(dir);
    const char *sep = (dl && (dir[dl-1] == '/' || dir[dl-1] == '\\')) ? "" : "/";
    snprintf(path, sizeof(path), "%s%s%s", dir, sep, rel);

    uint64_t fsz = file_size_of(path);
    if (fsz == UINT64_MAX) { srv_send_head(cfd, 404, "Not Found", 0, 0, 0, 0); return; }
    int ffd = open(path, RDONLY_FLAGS);
    if (ffd < 0) { srv_send_head(cfd, 404, "Not Found", 0, 0, 0, 0); return; }

    uint64_t start = 0, end = fsz ? fsz - 1 : 0;
    int partial = 0;
    if (r0 >= 0) {
        partial = 1;
        start = (uint64_t)r0;
        end = (r1 >= 0) ? (uint64_t)r1 : fsz - 1;
        if (end >= fsz) end = fsz - 1;
        if (start > end) { srv_send_head(cfd, 416, "Range Not Satisfiable", 0, 0, 0, (long long)fsz); close(ffd); return; }
    }
    uint64_t len = (fsz == 0) ? 0 : end - start + 1;
    srv_send_head(cfd, partial ? 206 : 200, partial ? "Partial Content" : "OK",
                  (long long)len, (long long)start, (long long)end, (long long)fsz);
    if (file_seek64(ffd, start) == 0) {
        char chunk[1 << 20];
        uint64_t sent = 0;
        while (sent < len) {
            int want = (int)((len - sent > sizeof(chunk)) ? sizeof(chunk) : (len - sent));
            int r = (int)read(ffd, chunk, (unsigned)want);
            if (r <= 0) {
                fprintf(stderr, "idletoken-weights: read %s at %llu (+%llu/%llu): %s\n",
                        r == 0 ? "EOF" : "error", (unsigned long long)start,
                        (unsigned long long)sent, (unsigned long long)len, strerror(errno));
                break;
            }
            int off = 0;
            while (off < r) { int w = (int)send(cfd, chunk + off, r - off, 0); if (w <= 0) { r = -1; break; } off += w; }
            if (r < 0) {
                /* The client sees only a truncated body, so without this line a
                 * server-side send failure is invisible on both ends. */
                fprintf(stderr, "idletoken-weights: send failed at %llu (+%llu/%llu bytes, err %d)\n",
                        (unsigned long long)start, (unsigned long long)sent,
                        (unsigned long long)len, sock_err());
                break;
            }
            sent += (uint64_t)off;
        }
    }
    close(ffd);
}

int idletoken_idx_stale(const char *gguf_path, const char *idx_path) {
    uint64_t gsz = file_size_of(gguf_path);
    if (gsz == UINT64_MAX) return 1;
    FILE *f = fopen(idx_path, "r");
    if (!f) return 1;
    unsigned long long isz = 0;
    int got = fscanf(f, "%llu", &isz);
    fclose(f);
    if (got != 1) return 1;
    return ((uint64_t)isz != gsz) ? 1 : 0;
}

int idletoken_serve_weights(const char *dir, const char *bind_addr) {
    int lfd = idletoken_listen_tcp(bind_addr);
    if (lfd < 0) {
        fprintf(stderr, "idletoken-weights: serve listen(%s) failed\n", bind_addr);
        return -1;
    }
#ifndef _WIN32
    signal(SIGCHLD, SIG_IGN);   /* auto-reap forked handlers */
    signal(SIGPIPE, SIG_IGN);
#endif
    fprintf(stderr, "idletoken-weights: serving %s on %s (range-capable)\n", dir, bind_addr);
    for (;;) {
        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) continue;
#ifndef _WIN32
        pid_t pid = fork();
        if (pid == 0) {           /* child: serve one request, exit */
            idletoken_close_fd(lfd);
            srv_handle(cfd, dir);
            idletoken_close_fd(cfd);
            _exit(0);
        }
        idletoken_close_fd(cfd);      /* parent keeps accepting */
#else
        srv_handle(cfd, dir);      /* Windows: serialized */
        idletoken_close_fd(cfd);
#endif
    }
}

/* ---- public entry -------------------------------------------------------- */

int idletoken_shard_fetch(const char *base_url, unsigned layer_lo, unsigned layer_hi,
                       const char *cache_dir, char *out_path, size_t out_cap) {
    char host_port[256], path[1024], idx_path[1088];
    if (parse_http_url(base_url, host_port, sizeof(host_port), path, sizeof(path)) != 0) {
        fprintf(stderr, "idletoken-weights: bad repo url: %s\n", base_url);
        return -1;
    }
    snprintf(idx_path, sizeof(idx_path), "%s.idx", path);

    const char *cd = (cache_dir && cache_dir[0]) ? cache_dir : ".";
    size_t cdl = strlen(cd);
    const char *sep = (cdl && (cd[cdl - 1] == '/' || cd[cdl - 1] == '\\')) ? "" : "/";
    snprintf(out_path, out_cap, "%s%sL%u-%u.gguf", cd, sep, layer_lo, layer_hi);

    /* Fetch + parse the index. */
    fprintf(stderr, "idletoken-weights: fetching index %s%s\n", host_port, idx_path);
    char *idx = http_get_all(host_port, idx_path);
    if (!idx) { fprintf(stderr, "idletoken-weights: could not fetch %s\n", idx_path); return -1; }
    uint64_t file_size = 0; range_t *ranges = NULL;
    int nr = compute_ranges(idx, layer_lo, layer_hi, &file_size, &ranges);
    free(idx);
    if (nr < 0) { fprintf(stderr, "idletoken-weights: bad index format\n"); return -1; }

    uint64_t need = 0;
    for (int i = 0; i < nr; i++) need += ranges[i].e - ranges[i].s;

    /* Idempotent: a cached partial of the right apparent size is reused. A
     * ".done" sidecar guards against a partial/interrupted previous fetch. */
    char done_path[1152];
    snprintf(done_path, sizeof(done_path), "%s.done", out_path);
    if (file_size_of(out_path) == file_size && file_size_of(done_path) != UINT64_MAX) {
        fprintf(stderr, "idletoken-weights: reuse cached %s (%.2f GB apparent)\n",
                out_path, file_size / 1e9);
        free(ranges);
        return 0;
    }

    fprintf(stderr,
            "idletoken-weights: fetching layers [%u,%u): %d ranges, %.2f GB of %.2f GB -> %s\n",
            layer_lo, layer_hi, nr, need / 1e9, file_size / 1e9, out_path);

    int fd = open(out_path, OPEN_FLAGS, 0644);
    if (fd < 0) { fprintf(stderr, "idletoken-weights: open %s: %s\n", out_path, strerror(errno)); free(ranges); return -1; }
    file_make_sparse(fd);
    if (file_truncate64(fd, file_size) != 0) {
        fprintf(stderr, "idletoken-weights: truncate %s to %llu failed\n", out_path,
                (unsigned long long)file_size);
        close(fd); free(ranges); return -1;
    }
    uint64_t done_bytes = 0;
    for (int i = 0; i < nr; i++) {
        if (fetch_range_pieces(host_port, path, ranges[i].s, ranges[i].e, fd,
                               &done_bytes, need) != 0) {
            fprintf(stderr, "idletoken-weights: range %llu-%llu fetch failed\n",
                    (unsigned long long)ranges[i].s, (unsigned long long)ranges[i].e);
            close(fd); free(ranges); return -1;
        }
    }
    close(fd);
    free(ranges);

    /* Mark complete. */
    FILE *df = fopen(done_path, "w");
    if (df) { fprintf(df, "%llu\n", (unsigned long long)file_size); fclose(df); }
    fprintf(stderr, "idletoken-weights: shard ready %s\n", out_path);
    return 0;
}
