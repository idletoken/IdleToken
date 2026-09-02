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
 *     line 1: "<file_size> <tensor_data_pos> <n_tensors> 2"
 *     then n: "<layer> <offset> <bytes> <name>" (layer -1 = shared/global)
 * We reuse the same needed-ranges logic as the Python `ranges` command. */
#include "idletoken_weights.h"
#include "idletoken_net.h"   /* idletoken_connect_tcp */

#include <errno.h>
#include <limits.h>
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
  #include <direct.h>
  #define OPEN_FLAGS (O_CREAT | O_WRONLY | O_BINARY)
  #define CACHE_OPEN_FLAGS (O_CREAT | O_TRUNC | O_WRONLY | O_BINARY)
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
  #define CACHE_OPEN_FLAGS (O_CREAT | O_TRUNC | O_WRONLY)
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
static uint64_t fnv1a_update(uint64_t hash, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= p[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

/* GET bytes [r0,r1], writing at `dst_off`.  When hash_io is non-NULL, update
 * the same FNV-1a hash ggml-RPC uses for RPC_CMD_SET_TENSOR_HASH.  A failed
 * request restores the incoming hash so a piece retry is byte-exact. */
static int http_get_range_to_file_at(const char *host_port, const char *path,
                                     uint64_t r0, uint64_t r1, int outfd,
                                     uint64_t dst_off, uint64_t *hash_io) {
    const uint64_t hash_before = hash_io ? *hash_io : 0;
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
    if (file_seek64(outfd, dst_off) != 0) { idletoken_close_fd(fd); return -1; }
    uint64_t written = 0;
    if (pre_len) {
        size_t take = (pre_len > want) ? (size_t)want : pre_len;
        if (write(outfd, pre, (unsigned)take) != (int)take) { idletoken_close_fd(fd); return -1; }
        if (hash_io) *hash_io = fnv1a_update(*hash_io, pre, take);
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
            if (hash_io) *hash_io = hash_before;
            idletoken_close_fd(fd); return -1;
        }
        if (hash_io) *hash_io = fnv1a_update(*hash_io, chunk, take);
        written += take;
    }
    idletoken_close_fd(fd);
    if (written != want) {
        if (hash_io) *hash_io = hash_before;
        return -1;
    }
    return 0;
}

/* GET bytes [r0,r1] and write them to fd at their absolute offset. */
static int http_get_range_to_file(const char *host_port, const char *path,
                                  uint64_t r0, uint64_t r1, int outfd) {
    return http_get_range_to_file_at(host_port, path, r0, r1, outfd, r0, NULL);
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

/* Coordinator fast path: it already owns the source GGUF, so routing tens of
 * GiB through a loopback TCP server only adds copies and scheduler wakeups.
 * Keep the same bounded progress/checkpoint cadence as remote range fetches. */
static int copy_range_pieces(const char *source, uint64_t s, uint64_t e,
                             int outfd, uint64_t *done_bytes,
                             uint64_t total_bytes) {
    int infd = open(source, RDONLY_FLAGS);
    if (infd < 0) {
        fprintf(stderr, "idletoken-weights: open local source %s: %s\n",
                source, strerror(errno));
        return -1;
    }
    unsigned char *buf = (unsigned char *)malloc(8u << 20);
    if (!buf) { close(infd); return -1; }
    int rc = 0;
    for (uint64_t p = s; p < e && rc == 0; ) {
        uint64_t q = (e - p > SHARD_PIECE_BYTES) ? p + SHARD_PIECE_BYTES : e;
        if (file_seek64(infd, p) != 0 || file_seek64(outfd, p) != 0) {
            rc = -1;
            break;
        }
        uint64_t copied = 0;
        while (p + copied < q) {
            size_t want = (size_t)((q - p - copied > (8u << 20))
                                       ? (8u << 20) : q - p - copied);
            int got = (int)read(infd, buf, (unsigned)want);
            if (got <= 0) { rc = -1; break; }
            int off = 0;
            while (off < got) {
                int wrote = (int)write(outfd, buf + off,
                                       (unsigned)(got - off));
                if (wrote <= 0) { rc = -1; break; }
                off += wrote;
            }
            copied += (uint64_t)got;
        }
        if (rc != 0) break;
        *done_bytes += q - p;
        fprintf(stderr, "idletoken-weights: local copy %.2f/%.2f GB (%.0f%%)\n",
                *done_bytes / 1e9, total_bytes / 1e9,
                100.0 * (double)*done_bytes /
                    (double)(total_bytes ? total_bytes : 1));
        p = q;
    }
    free(buf);
    close(infd);
    if (rc != 0)
        fprintf(stderr, "idletoken-weights: local copy %s range %llu-%llu "
                        "failed: %s\n", source,
                (unsigned long long)s, (unsigned long long)e, strerror(errno));
    return rc;
}

/* Fetch one tensor into a standalone cache file, preserving the content hash
 * across bounded piece retries. */
static int fetch_tensor_pieces(const char *host_port, const char *path,
                               uint64_t s, uint64_t e, int outfd,
                               uint64_t *hash_io,
                               uint64_t *done_bytes, uint64_t total_bytes,
                               idletoken_weights_progress_fn progress,
                               void *progress_opaque) {
    for (uint64_t p = s; p < e; ) {
        uint64_t q = (e - p > SHARD_PIECE_BYTES) ? p + SHARD_PIECE_BYTES : e;
        int ok = 0;
        for (int attempt = 1; attempt <= SHARD_PIECE_RETRIES; attempt++) {
            if (http_get_range_to_file_at(host_port, path, p, q - 1, outfd,
                                          p - s, hash_io) == 0) {
                ok = 1;
                break;
            }
            fprintf(stderr, "idletoken-weights: cache piece %llu-%llu failed "
                            "(attempt %d/%d)\n",
                    (unsigned long long)p, (unsigned long long)q,
                    attempt, SHARD_PIECE_RETRIES);
            if (attempt < SHARD_PIECE_RETRIES) shard_backoff(attempt);
        }
        if (!ok) return -1;
        *done_bytes += q - p;
        fprintf(stderr, "idletoken-weights: RPC cache %.2f/%.2f GB (%.0f%%)\n",
                *done_bytes / 1e9, total_bytes / 1e9,
                100.0 * (double)*done_bytes /
                    (double)(total_bytes ? total_bytes : 1));
        if (progress) progress(*done_bytes, total_bytes, progress_opaque);
        p = q;
    }
    return 0;
}

/* Copy one assigned tensor from this node's complete GGUF into the standalone
 * content-addressed RPC cache file. The input offset is absolute in the GGUF;
 * the output starts at zero because ggml-RPC stores one file per tensor. */
static int copy_tensor_pieces(const char *source, uint64_t s, uint64_t e,
                              int outfd, uint64_t *hash_io,
                              uint64_t *done_bytes, uint64_t total_bytes,
                              idletoken_weights_progress_fn progress,
                              void *progress_opaque) {
    int infd = open(source, RDONLY_FLAGS);
    if (infd < 0) {
        fprintf(stderr, "idletoken-weights: open local tensor source %s: %s\n",
                source, strerror(errno));
        return -1;
    }
    unsigned char *buf = (unsigned char *)malloc(8u << 20);
    if (!buf) { close(infd); return -1; }
    int rc = 0;
    for (uint64_t p = s; p < e && rc == 0; ) {
        const uint64_t q = (e - p > SHARD_PIECE_BYTES)
            ? p + SHARD_PIECE_BYTES : e;
        if (file_seek64(infd, p) != 0 || file_seek64(outfd, p - s) != 0) {
            rc = -1;
            break;
        }
        uint64_t copied = 0;
        while (p + copied < q) {
            const size_t want = (size_t)((q - p - copied > (8u << 20))
                                             ? (8u << 20) : q - p - copied);
            const int got = (int)read(infd, buf, (unsigned)want);
            if (got <= 0) { rc = -1; break; }
            int off = 0;
            while (off < got) {
                const int wrote = (int)write(outfd, buf + off,
                                             (unsigned)(got - off));
                if (wrote <= 0) { rc = -1; break; }
                off += wrote;
            }
            if (rc != 0) break;
            *hash_io = fnv1a_update(*hash_io, buf, (size_t)got);
            copied += (uint64_t)got;
        }
        if (rc != 0) break;
        *done_bytes += q - p;
        fprintf(stderr, "idletoken-weights: local RPC cache %.2f/%.2f GB (%.0f%%)\n",
                *done_bytes / 1e9, total_bytes / 1e9,
                100.0 * (double)*done_bytes /
                    (double)(total_bytes ? total_bytes : 1));
        if (progress) progress(*done_bytes, total_bytes, progress_opaque);
        p = q;
    }
    free(buf);
    close(infd);
    if (rc != 0)
        fprintf(stderr, "idletoken-weights: local tensor copy %s range "
                        "%llu-%llu failed: %s\n", source,
                (unsigned long long)s, (unsigned long long)e, strerror(errno));
    return rc;
}

/* ---- range set from the .idx manifest ----------------------------------- */

typedef struct { uint64_t s, e; } range_t;

typedef struct {
    unsigned part;
    long long layer;
    uint64_t off;
    uint64_t bytes;
    char name[128];
    char part_name[256];
} cache_tensor_t;

typedef struct {
    uint64_t file_size;
    uint64_t tensor_data_pos;
    char name[256];
} idx_part_t;

typedef struct {
    unsigned part;
    long long layer;
    uint64_t off;
    uint64_t bytes;
    char name[128];
} idx_tensor_t;

typedef struct {
    unsigned version;
    unsigned n_parts;
    size_t n_tensors;
    idx_part_t *parts;
    idx_tensor_t *tensors;
} idx_manifest_t;

static void idx_manifest_clear(idx_manifest_t *m) {
    if (!m) return;
    free(m->parts);
    free(m->tensors);
    memset(m, 0, sizeof(*m));
}

/* Index v2 describes one GGUF. Index v3 describes a standard llama.cpp split
 * GGUF set. All names are basename-only: the repository deliberately serves a
 * single directory, so neither an index nor a request can escape it. */
static int idx_manifest_parse(const char *idx, idx_manifest_t *m) {
    if (!idx || !m) return -1;
    memset(m, 0, sizeof(*m));
    unsigned long long fsz = 0, tdp = 0, nt = 0;
    unsigned version = 0, n_parts = 0;
    if (sscanf(idx, "%llu %llu %llu %u %u", &fsz, &tdp, &nt,
               &version, &n_parts) < 4 || nt == 0 ||
        (version != 2 && version != 3)) return -1;
    if (version == 2) n_parts = 1;
    if (n_parts == 0 || n_parts > 999 || nt > SIZE_MAX / sizeof(idx_tensor_t))
        return -1;
    m->version = version;
    m->n_parts = n_parts;
    m->n_tensors = (size_t)nt;
    m->parts = (idx_part_t *)calloc(n_parts, sizeof(*m->parts));
    m->tensors = (idx_tensor_t *)calloc((size_t)nt, sizeof(*m->tensors));
    if (!m->parts || !m->tensors) {
        idx_manifest_clear(m);
        return -1;
    }
    const char *p = strchr(idx, '\n');
    if (!p) { idx_manifest_clear(m); return -1; }
    p++;
    if (version == 2) {
        if (tdp == 0 || tdp > fsz) { idx_manifest_clear(m); return -1; }
        m->parts[0].file_size = (uint64_t)fsz;
        m->parts[0].tensor_data_pos = (uint64_t)tdp;
    } else {
        for (unsigned i = 0; i < n_parts; i++) {
            unsigned part = 0;
            unsigned long long psz = 0, ptdp = 0;
            char name[256] = "";
            if (!*p || sscanf(p, "P %u %llu %llu %255s", &part, &psz,
                              &ptdp, name) != 4 || part != i + 1 ||
                !name[0] || strstr(name, "..") || strchr(name, '/') ||
                strchr(name, '\\') || ptdp == 0 || ptdp > psz) {
                idx_manifest_clear(m);
                return -1;
            }
            m->parts[i].file_size = (uint64_t)psz;
            m->parts[i].tensor_data_pos = (uint64_t)ptdp;
            snprintf(m->parts[i].name, sizeof(m->parts[i].name), "%s", name);
            const char *nl = strchr(p, '\n');
            if (!nl) { idx_manifest_clear(m); return -1; }
            p = nl + 1;
        }
    }
    for (size_t i = 0; i < m->n_tensors; i++) {
        unsigned part = 1;
        long long layer = 0;
        unsigned long long off = 0, bytes = 0;
        char name[128] = "";
        int got = version == 2
            ? sscanf(p, "%lld %llu %llu %127s", &layer, &off, &bytes, name)
            : sscanf(p, "T %u %lld %llu %llu %127s", &part, &layer,
                     &off, &bytes, name);
        if (!*p || got != (version == 2 ? 4 : 5) || part == 0 ||
            part > n_parts || layer < -1 || !name[0] || bytes == 0 ||
            off > m->parts[part - 1].file_size ||
            bytes > m->parts[part - 1].file_size - off) {
            idx_manifest_clear(m);
            return -1;
        }
        m->tensors[i].part = part - 1;
        m->tensors[i].layer = layer;
        m->tensors[i].off = (uint64_t)off;
        m->tensors[i].bytes = (uint64_t)bytes;
        snprintf(m->tensors[i].name, sizeof(m->tensors[i].name), "%s", name);
        const char *nl = strchr(p, '\n');
        if (!nl && i + 1 < m->n_tensors) {
            idx_manifest_clear(m);
            return -1;
        }
        p = nl ? nl + 1 : p + strlen(p);
    }
    return 0;
}

static int ensure_dir_one(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(path, 0700) == 0 || errno == EEXIST) return 0;
#endif
    return -1;
}

static int ensure_dir_tree(const char *path) {
    if (!path || !path[0]) return -1;
    char tmp[1200];
    if (snprintf(tmp, sizeof(tmp), "%s", path) < 0 ||
        strlen(path) >= sizeof(tmp)) return -1;
    size_t n = strlen(tmp);
    while (n > 1 && (tmp[n - 1] == '/' || tmp[n - 1] == '\\')) tmp[--n] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/' && *p != '\\') continue;
#ifdef _WIN32
        if (p == tmp + 2 && tmp[1] == ':') continue;
#endif
        char sep = *p;
        *p = '\0';
        if (ensure_dir_one(tmp) != 0) return -1;
        *p = sep;
    }
    return ensure_dir_one(tmp);
}

static int cache_file_ok(const char *dir, uint64_t hash, uint64_t bytes) {
    char p[1200];
    snprintf(p, sizeof(p), "%s/%016llx", dir, (unsigned long long)hash);
    return file_size_of(p) == bytes;
}

/* A name link is published only after its content-addressed tensor is complete.
 * Persist the name -> content hash beside it so a changed layer boundary can
 * reuse an overlapping warm shard without rereading tens of GiB merely to
 * recompute hashes. Size checks on both links keep a torn record fail-closed. */
static int cache_write_resume_record(const char *dir, uint64_t idx_hash,
                                     uint64_t name_hash,
                                     uint64_t content_hash, uint64_t bytes) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/r-%016llx", dir,
             (unsigned long long)name_hash);
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    int wrote = fprintf(f, "IDLETOKEN_RPC_NAME_V2 %016llx %016llx %llu\n",
                        (unsigned long long)idx_hash,
                        (unsigned long long)content_hash,
                        (unsigned long long)bytes) > 0;
    int ok = fclose(f) == 0 && wrote;
    if (!ok) remove(path);
    return ok;
}

static int cache_read_resume_record(const char *dir, uint64_t idx_hash,
                                    uint64_t name_hash,
                                    uint64_t bytes, uint64_t *content_hash_out) {
    char path[1200], named[1200];
    snprintf(path, sizeof(path), "%s/r-%016llx", dir,
             (unsigned long long)name_hash);
    snprintf(named, sizeof(named), "%s/n-%016llx", dir,
             (unsigned long long)name_hash);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    unsigned long long model_hash = 0, hash = 0, recorded_bytes = 0;
    int valid = fscanf(f, "IDLETOKEN_RPC_NAME_V2 %llx %llx %llu",
                       &model_hash, &hash, &recorded_bytes) == 3;
    fclose(f);
    if (!valid || (uint64_t)model_hash != idx_hash ||
        (uint64_t)recorded_bytes != bytes ||
        file_size_of(named) != bytes ||
        !cache_file_ok(dir, (uint64_t)hash, bytes)) {
        return 0;
    }
    *content_hash_out = (uint64_t)hash;
    return 1;
}

/* Recover completed tensors from an interrupted cold seed. The name hard link
 * is published only after a tensor has been fetched and closed, so its presence
 * at the expected size is a safe per-tensor checkpoint even when the final V2
 * marker has not been written yet. Prefer its persisted content-hash record;
 * old caches fall back to one local hash pass and are upgraded in place. */
static int cache_resume_name(const char *dir, uint64_t idx_hash,
                             uint64_t name_hash,
                             uint64_t bytes, uint64_t *content_hash_out) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/n-%016llx", dir,
             (unsigned long long)name_hash);
    if (file_size_of(path) != bytes) return 0;
    /* A same-name, same-size tensor from another model is not proof of equal
     * content.  V1 records had no model identity, so they are deliberately not
     * migrated by hashing the local file: that proves what the bytes are, not
     * that they belong to this repository.  A valid range marker upgrades warm
     * V1 caches without downloading because it already records the content
     * hashes under this idx_hash. */
    return cache_read_resume_record(dir, idx_hash, name_hash, bytes,
                                    content_hash_out);
}

static int cache_activate_name(const char *dir, uint64_t content_hash,
                               uint64_t idx_hash, uint64_t name_hash,
                               uint64_t bytes) {
    char content[1200], named[1200];
    snprintf(content, sizeof(content), "%s/%016llx", dir,
             (unsigned long long)content_hash);
    snprintf(named, sizeof(named), "%s/n-%016llx", dir,
             (unsigned long long)name_hash);
    if (file_size_of(content) != bytes) return 0;
    remove(named);
#ifdef _WIN32
    if (!CreateHardLinkA(named, content, NULL)) return 0;
#else
    if (link(content, named) != 0) return 0;
#endif
    if (file_size_of(named) != bytes) return 0;
    cache_write_resume_record(dir, idx_hash, name_hash, content_hash, bytes);
    return 1;
}

/* Validate a completed seed marker and every content-addressed tensor it
 * names.  A marker without its files is not a cache hit. */
static int rpc_cache_marker_valid(const char *marker, const char *cache_dir,
                                  uint64_t idx_hash, unsigned lo, unsigned hi,
                                  uint64_t total, unsigned count) {
    FILE *f = fopen(marker, "r");
    if (!f) return 0;
    unsigned long long mh = 0, mt = 0;
    unsigned mlo = 0, mhi = 0, mc = 0;
    if (fscanf(f, "IDLETOKEN_RPC_CACHE_V2 %llx %u %u %u %llu",
               &mh, &mlo, &mhi, &mc, &mt) != 5 ||
        (uint64_t)mh != idx_hash || mlo != lo || mhi != hi ||
        mc != count || (uint64_t)mt != total) {
        fclose(f);
        return 0;
    }
    for (unsigned i = 0; i < count; i++) {
        unsigned long long h = 0, n = 0, nh = 0;
        if (fscanf(f, "%llx %llu %llx", &h, &n, &nh) != 3 ||
            !cache_file_ok(cache_dir, (uint64_t)h, (uint64_t)n) ||
            !cache_activate_name(cache_dir, (uint64_t)h, idx_hash,
                                 (uint64_t)nh,
                                 (uint64_t)n)) {
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return 1;
}

/* Parse the .idx, compute the merged byte ranges this [lo,hi) worker needs, and
 * return them (caller frees). Sets *file_size. Returns count, -1 on error. */
static int manifest_ranges(const idx_manifest_t *m, unsigned part,
                           unsigned lo, unsigned hi,
                           int include_header, int include_shared,
                           uint64_t *file_size, range_t **out_ranges) {
    if (!m || part >= m->n_parts || !file_size || !out_ranges) return -1;
    range_t *r = (range_t *)malloc(sizeof(range_t) * (m->n_tensors + 1));
    if (!r) return -1;
    int nr = 0;
    if (include_header) {
        r[nr].s = 0;
        r[nr].e = m->parts[part].tensor_data_pos;
        nr++;
    }
    for (size_t i = 0; i < m->n_tensors; i++) {
        const idx_tensor_t *t = &m->tensors[i];
        if (t->part != part) continue;
        if ((include_shared && t->layer < 0) ||
            (t->layer >= 0 && (unsigned)t->layer >= lo &&
             (unsigned)t->layer < hi)) {
            r[nr].s = t->off;
            r[nr].e = t->off + t->bytes;
            nr++;
        }
    }
    /* sort by start (insertion sort — nr is small, ~ shared + range tensors) */
    for (int i = 1; i < nr; i++) {
        range_t k = r[i]; int j = i - 1;
        while (j >= 0 && r[j].s > k.s) { r[j + 1] = r[j]; j--; }
        r[j + 1] = k;
    }
    /* merge adjacent/overlapping */
    int merged = 0;
    for (int i = 0; i < nr; i++) {
        if (merged > 0 && r[i].s <= r[merged - 1].e) {
            if (r[i].e > r[merged - 1].e) r[merged - 1].e = r[i].e;
        } else {
            r[merged++] = r[i];
        }
    }
    *file_size = m->parts[part].file_size;
    *out_ranges = r;
    return merged;
}

static int compute_ranges_mode(const char *idx, unsigned lo, unsigned hi,
                               int include_header, int include_shared,
                               uint64_t *file_size, range_t **out_ranges) {
    idx_manifest_t m;
    if (idx_manifest_parse(idx, &m) != 0 || m.version != 2) return -1;
    int nr = manifest_ranges(&m, 0, lo, hi, include_header, include_shared,
                             file_size, out_ranges);
    idx_manifest_clear(&m);
    return nr;
}

static int compute_ranges(const char *idx, unsigned lo, unsigned hi,
                          uint64_t *file_size, range_t **out_ranges) {
    return compute_ranges_mode(idx, lo, hi, 1, 1, file_size, out_ranges);
}

static int idx_layer_count(const char *idx, unsigned *layers_out) {
    if (!idx || !layers_out) return -1;
    idx_manifest_t m;
    if (idx_manifest_parse(idx, &m) != 0) return -1;
    long long max_layer = -1;
    for (size_t i = 0; i < m.n_tensors; i++)
        if (m.tensors[i].layer > max_layer) max_layer = m.tensors[i].layer;
    idx_manifest_clear(&m);
    if (max_layer < 0 || (unsigned long long)max_layer >= UINT_MAX) return -1;
    *layers_out = (unsigned)max_layer + 1;
    return 0;
}

/* ---- GGUF index (coordinator side) --------------------------------------
 * Emit the C-friendly `.idx` manifest (`<file_size> <tensor_data_pos> <n> 2`
 * then per-tensor `<layer> <offset> <bytes> <name>`) so the coordinator's repo is
 * self-contained — no Python at runtime. Block geometry follows the pinned
 * llama.cpp ggml_type enum; the same table drives scripts/gguf_shard.py. */

static const struct { unsigned be, bb; } GGUF_GEOM[] = {
    {1,4},{1,2},{32,18},{32,20},{0,0},{0,0},{32,22},{32,24},{32,34},{32,40},
    {256,84},{256,110},{256,144},{256,176},{256,210},{256,292},{256,66},{256,74},
    {256,98},{256,110},{256,50},{256,110},{256,82},{256,136},{1,1},{1,2},{1,4},
    {1,8},{1,8},{256,56},{1,2},{0,0},{0,0},{0,0},{256,54},{256,66},{0,0},
    {0,0},{0,0},{32,17},{64,36},{128,18},{64,18},
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

typedef struct {
    idx_part_t part;
    idx_tensor_t *tensors;
    size_t n_tensors;
} gguf_scan_t;

static void gguf_scan_clear(gguf_scan_t *s) {
    if (!s) return;
    free(s->tensors);
    memset(s, 0, sizeof(*s));
}

static int split_primary_count(const char *path, unsigned *count_out,
                               size_t *prefix_len_out) {
    const char *hit = NULL;
    for (const char *p = path; (p = strstr(p, "-00001-of-")) != NULL; p++)
        hit = p;
    if (!hit) return 0;
    const char *digits = hit + strlen("-00001-of-");
    for (int i = 0; i < 5; i++)
        if (digits[i] < '0' || digits[i] > '9') return 0;
    if (strcmp(digits + 5, ".gguf") != 0) return 0;
    unsigned n = (unsigned)strtoul(digits, NULL, 10);
    if (n < 2 || n > 999) return 0;
    if (count_out) *count_out = n;
    if (prefix_len_out) *prefix_len_out = (size_t)(hit - path);
    return 1;
}

static int split_part_path(const char *primary, size_t prefix_len,
                           unsigned part, unsigned count,
                           char *out, size_t cap) {
    int n = snprintf(out, cap, "%.*s-%05u-of-%05u.gguf",
                     (int)prefix_len, primary, part, count);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static const char *path_basename(const char *path) {
    const char *base = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    return base;
}

static int gguf_scan_file(const char *gguf_path, gguf_scan_t *scan) {
    memset(scan, 0, sizeof(*scan));
    uint64_t fsz = file_size_of(gguf_path);
    if (fsz == UINT64_MAX || fsz == 0) {
        fprintf(stderr, "idletoken-weights: idx: cannot stat %s\n", gguf_path);
        return -1;
    }
    /* Read the header region (grow if the tensor directory is bigger). */
    size_t cap = (fsz < (64u << 20)) ? (size_t)fsz : (64u << 20);
    uint8_t *buf = NULL; FILE *f = fopen(gguf_path, "rb");
    if (!f) { fprintf(stderr, "idletoken-weights: idx: open %s failed\n", gguf_path); return -1; }
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
        idx_tensor_t *t = (idx_tensor_t *)calloc(
            (size_t)(n_tensors ? n_tensors : 1), sizeof(*t));
        if (!t) { free(buf); fclose(f); return -1; }
        int ok = 1;
        for (uint64_t i = 0; i < n_tensors && !c.err; i++) {
            char nm[256]; gc_str(&c, nm, sizeof(nm));
            uint32_t ndim = gc_u32(&c);
            uint64_t elems = 1;
            for (uint32_t d = 0; d < ndim && !c.err; d++) {
                uint64_t dim = gc_u64(&c);
                if (dim && elems > UINT64_MAX / dim) { c.err = 1; break; }
                elems *= dim;
            }
            uint32_t ty = gc_u32(&c);
            uint64_t rel = gc_u64(&c);
            if (ty >= GGUF_GEOM_N || GGUF_GEOM[ty].be == 0) {
                fprintf(stderr, "idletoken-weights: idx: unsupported GGUF "
                                "tensor type %u for %s in %s\n",
                        ty, nm, gguf_path);
                ok = 0;
                break;
            }
            uint64_t blocks = (elems + GGUF_GEOM[ty].be - 1) / GGUF_GEOM[ty].be;
            if (blocks > UINT64_MAX / GGUF_GEOM[ty].bb) { ok = 0; break; }
            t[i].bytes = blocks * GGUF_GEOM[ty].bb;
            t[i].layer = layer_of_name(nm);
            t[i].off = rel;   /* + tensor_data_pos below */
            snprintf(t[i].name, sizeof(t[i].name), "%s", nm);
        }
        if (c.err) {
            /* header bigger than we read — grow and retry (unless whole file). */
            free(t);
            if (cap >= fsz) { fprintf(stderr, "idletoken-weights: idx: parse error\n"); free(buf); fclose(f); return -1; }
            cap = (cap * 2 < fsz) ? cap * 2 : (size_t)fsz;
            continue;
        }
        if (!ok) {
            fprintf(stderr, "idletoken-weights: idx: unsupported or invalid "
                            "tensor geometry in %s\n", gguf_path);
            free(t); free(buf); fclose(f); return -1;
        }
        uint64_t tdp = (c.p + alignment - 1) / alignment * alignment;
        if (tdp == 0 || tdp > fsz) {
            fprintf(stderr, "idletoken-weights: idx: invalid tensor data "
                            "offset in %s\n", gguf_path);
            free(t); free(buf); fclose(f); return -1;
        }
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (t[i].off > fsz - tdp || t[i].bytes > fsz - tdp - t[i].off) {
                fprintf(stderr, "idletoken-weights: idx: tensor %s exceeds %s\n",
                        t[i].name, gguf_path);
                free(t); free(buf); fclose(f); return -1;
            }
            t[i].off += tdp;
        }
        scan->part.file_size = fsz;
        scan->part.tensor_data_pos = tdp;
        snprintf(scan->part.name, sizeof(scan->part.name), "%s",
                 path_basename(gguf_path));
        scan->tensors = t;
        scan->n_tensors = (size_t)n_tensors;
        break;
    }
    free(buf);
    fclose(f);
    return 0;
}

int idletoken_write_idx(const char *gguf_path, const char *idx_path) {
    if (!gguf_path || !idx_path) return -1;
    unsigned n_parts = 1;
    size_t prefix_len = 0;
    const int is_split = split_primary_count(gguf_path, &n_parts, &prefix_len);
    gguf_scan_t *parts = (gguf_scan_t *)calloc(n_parts, sizeof(*parts));
    if (!parts) return -1;
    size_t total_tensors = 0;
    int rc = -1;
    for (unsigned i = 0; i < n_parts; i++) {
        char path[1600];
        if (is_split) {
            if (split_part_path(gguf_path, prefix_len, i + 1, n_parts,
                                path, sizeof(path)) != 0) goto done;
        } else {
            if (snprintf(path, sizeof(path), "%s", gguf_path) < 0 ||
                strlen(gguf_path) >= sizeof(path)) goto done;
        }
        if (gguf_scan_file(path, &parts[i]) != 0) goto done;
        if (SIZE_MAX - total_tensors < parts[i].n_tensors) goto done;
        total_tensors += parts[i].n_tensors;
    }
    if (total_tensors == 0) {
        fprintf(stderr, "idletoken-weights: idx: the GGUF set has no tensors\n");
        goto done;
    }
    char tmp[1600];
    if (snprintf(tmp, sizeof(tmp), "%s.part", idx_path) < 0 ||
        strlen(idx_path) + 5 >= sizeof(tmp)) goto done;
    FILE *out = fopen(tmp, "w");
    if (!out) {
        fprintf(stderr, "idletoken-weights: idx: write %s failed\n", tmp);
        goto done;
    }
    int write_ok = 1;
    if (!is_split) {
        write_ok = fprintf(out, "%llu %llu %llu 2\n",
                           (unsigned long long)parts[0].part.file_size,
                           (unsigned long long)parts[0].part.tensor_data_pos,
                           (unsigned long long)total_tensors) > 0;
    } else {
        uint64_t total_size = 0;
        for (unsigned i = 0; i < n_parts; i++) {
            if (UINT64_MAX - total_size < parts[i].part.file_size) write_ok = 0;
            total_size += parts[i].part.file_size;
        }
        if (write_ok)
            write_ok = fprintf(out, "%llu 0 %llu 3 %u\n",
                               (unsigned long long)total_size,
                               (unsigned long long)total_tensors, n_parts) > 0;
        for (unsigned i = 0; write_ok && i < n_parts; i++)
            write_ok = fprintf(out, "P %u %llu %llu %s\n", i + 1,
                               (unsigned long long)parts[i].part.file_size,
                               (unsigned long long)parts[i].part.tensor_data_pos,
                               parts[i].part.name) > 0;
    }
    for (unsigned i = 0; write_ok && i < n_parts; i++) {
        for (size_t j = 0; write_ok && j < parts[i].n_tensors; j++) {
            idx_tensor_t *t = &parts[i].tensors[j];
            if (is_split)
                write_ok = fprintf(out, "T %u %lld %llu %llu %s\n", i + 1,
                                   t->layer, (unsigned long long)t->off,
                                   (unsigned long long)t->bytes, t->name) > 0;
            else
                write_ok = fprintf(out, "%lld %llu %llu %s\n", t->layer,
                                   (unsigned long long)t->off,
                                   (unsigned long long)t->bytes, t->name) > 0;
        }
    }
    if (fclose(out) != 0) write_ok = 0;
    if (!write_ok) { remove(tmp); goto done; }
    remove(idx_path);
    if (rename(tmp, idx_path) != 0) {
        fprintf(stderr, "idletoken-weights: idx: rename %s -> %s failed: %s\n",
                tmp, idx_path, strerror(errno));
        remove(tmp);
        goto done;
    }
    rc = 0;
    fprintf(stderr, "idletoken-weights: wrote index %s\n", idx_path);
done:
    for (unsigned i = 0; i < n_parts; i++) gguf_scan_clear(&parts[i]);
    free(parts);
    return rc;
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
    unsigned long long isz = 0, tdp = 0, nt = 0;
    unsigned version = 0, indexed_parts = 0;
    int got = fscanf(f, "%llu %llu %llu %u %u", &isz, &tdp, &nt,
                     &version, &indexed_parts);
    unsigned actual_parts = 1;
    size_t prefix_len = 0;
    const int is_split = split_primary_count(gguf_path, &actual_parts,
                                             &prefix_len);
    if (got < 4 || nt == 0 || (is_split && version != 3) ||
        (!is_split && version != 2)) {
        fclose(f);
        return 1;
    }
    if (!is_split) {
        fclose(f);
        return tdp == 0 || (uint64_t)isz != gsz;
    }
    if (got != 5 || indexed_parts != actual_parts) {
        fclose(f);
        return 1;
    }
    for (unsigned i = 0; i < actual_parts; i++) {
        unsigned part = 0;
        unsigned long long psz = 0, ptdp = 0;
        char indexed_name[256] = "", path[1600];
        if (fscanf(f, " P %u %llu %llu %255s", &part, &psz, &ptdp,
                   indexed_name) != 4 || part != i + 1 || ptdp == 0 ||
            split_part_path(gguf_path, prefix_len, i + 1, actual_parts,
                            path, sizeof(path)) != 0 ||
            strcmp(indexed_name, path_basename(path)) != 0 ||
            file_size_of(path) != (uint64_t)psz) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
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

static int repo_sibling_path(const char *primary_path, const char *name,
                             char *out, size_t cap) {
    const char *slash = strrchr(primary_path, '/');
    size_t dir_len = slash ? (size_t)(slash - primary_path + 1) : 1;
    int n = snprintf(out, cap, "%.*s%s", (int)dir_len,
                     slash ? primary_path : "/", name);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static int file_sibling_path(const char *primary, const char *name,
                             char *out, size_t cap) {
    const char *slash = strrchr(primary, '/');
#ifdef _WIN32
    const char *back = strrchr(primary, '\\');
    if (!slash || (back && back > slash)) slash = back;
#endif
    size_t dir_len = slash ? (size_t)(slash - primary + 1) : 0;
    int n = snprintf(out, cap, "%.*s%s", (int)dir_len, primary, name);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static int local_split_marker_valid(const char *marker, const char *dir,
                                    const idx_manifest_t *m,
                                    uint64_t idx_hash,
                                    unsigned *coverage_out) {
    FILE *f = fopen(marker, "r");
    if (!f) return 0;
    unsigned long long mh = 0;
    unsigned coverage = 0, n_parts = 0;
    int valid = fscanf(f, "IDLETOKEN_LOCAL_SPLIT_V1 %llx %u %u",
                       &mh, &coverage, &n_parts) == 3 &&
                (uint64_t)mh == idx_hash && n_parts == m->n_parts;
    for (unsigned i = 0; valid && i < m->n_parts; i++) {
        unsigned part = 0;
        unsigned long long size = 0;
        char path[1600];
        valid = fscanf(f, "%u %llu", &part, &size) == 2 && part == i + 1 &&
                (uint64_t)size == m->parts[i].file_size &&
                snprintf(path, sizeof(path), "%s/%s", dir,
                         m->parts[i].name) > 0 &&
                file_size_of(path) == m->parts[i].file_size;
    }
    fclose(f);
    if (!valid) return 0;
    *coverage_out = coverage;
    return 1;
}

static int local_split_prepare(const char *host_port, const char *primary_path,
                               const char *local_primary,
                               const idx_manifest_t *m, uint64_t idx_hash,
                               unsigned layer_hi, const char *cache_dir,
                               char *out_path, size_t out_cap) {
    char dir[1400], marker[1500];
    int n = snprintf(dir, sizeof(dir), "%s%slocal-%016llx", cache_dir,
                     (cache_dir[strlen(cache_dir) - 1] == '/' ||
                      cache_dir[strlen(cache_dir) - 1] == '\\') ? "" : "/",
                     (unsigned long long)idx_hash);
    if (n <= 0 || (size_t)n >= sizeof(dir) || ensure_dir_tree(dir) != 0)
        return -1;
    n = snprintf(out_path, out_cap, "%s/%s", dir, m->parts[0].name);
    if (n <= 0 || (size_t)n >= out_cap) return -1;
    snprintf(marker, sizeof(marker), "%s/idletoken.done", dir);

    unsigned coverage = 0;
    int warm = local_split_marker_valid(marker, dir, m, idx_hash, &coverage);
    if (warm && coverage >= layer_hi) {
        fprintf(stderr, "idletoken-weights: reuse split local model %s "
                        "(cached [0,%u), current plan [0,%u))\n",
                out_path, coverage, layer_hi);
        return 0;
    }
    uint64_t total_need = 0;
    for (unsigned part = 0; part < m->n_parts; part++) {
        uint64_t fsz = 0;
        range_t *ranges = NULL;
        int nr = manifest_ranges(m, part, warm ? coverage : 0, layer_hi,
                                 warm ? 0 : 1, warm ? 0 : 1,
                                 &fsz, &ranges);
        if (nr < 0) return -1;
        for (int i = 0; i < nr; i++) total_need += ranges[i].e - ranges[i].s;
        free(ranges);
    }
    fprintf(stderr, "idletoken-weights: %s split local model to [0,%u): "
                    "%u files, %.2f GB new bytes -> %s\n",
            warm ? "extending" : "building", layer_hi, m->n_parts,
            total_need / 1e9, out_path);

    uint64_t done_bytes = 0;
    for (unsigned part = 0; part < m->n_parts; part++) {
        uint64_t fsz = 0;
        range_t *ranges = NULL;
        int nr = manifest_ranges(m, part, warm ? coverage : 0, layer_hi,
                                 warm ? 0 : 1, warm ? 0 : 1,
                                 &fsz, &ranges);
        if (nr < 0) return -1;
        char local[1600], remote[1200], source[1600];
        snprintf(local, sizeof(local), "%s/%s", dir, m->parts[part].name);
        if (repo_sibling_path(primary_path, m->parts[part].name,
                              remote, sizeof(remote)) != 0) {
            free(ranges);
            return -1;
        }
        if (!warm) remove(local);
        int fd = open(local, OPEN_FLAGS, 0600);
        if (fd < 0) {
            fprintf(stderr, "idletoken-weights: open split local view %s: %s\n",
                    local, strerror(errno));
            free(ranges);
            return -1;
        }
        file_make_sparse(fd);
        if (file_truncate64(fd, fsz) != 0) {
            close(fd); free(ranges); return -1;
        }
        for (int i = 0; i < nr; i++) {
            int range_rc;
            if (local_primary) {
                range_rc = file_sibling_path(local_primary,
                                              m->parts[part].name,
                                              source, sizeof(source)) == 0
                    ? copy_range_pieces(source, ranges[i].s, ranges[i].e, fd,
                                        &done_bytes, total_need)
                    : -1;
            } else {
                range_rc = fetch_range_pieces(host_port, remote, ranges[i].s,
                                               ranges[i].e, fd, &done_bytes,
                                               total_need);
            }
            if (range_rc != 0) {
                fprintf(stderr, "idletoken-weights: split part %u range "
                                "%llu-%llu fetch failed\n", part + 1,
                        (unsigned long long)ranges[i].s,
                        (unsigned long long)ranges[i].e);
                close(fd); free(ranges); return -1;
            }
        }
        free(ranges);
#ifdef _WIN32
        if (_commit(fd) != 0) {
#else
        if (fsync(fd) != 0) {
#endif
            close(fd);
            return -1;
        }
        close(fd);
    }

    char marker_tmp[1540];
    snprintf(marker_tmp, sizeof(marker_tmp), "%s.part", marker);
    FILE *mf = fopen(marker_tmp, "w");
    int marker_ok = mf &&
        fprintf(mf, "IDLETOKEN_LOCAL_SPLIT_V1 %016llx %u %u\n",
                (unsigned long long)idx_hash, layer_hi, m->n_parts) > 0;
    for (unsigned i = 0; marker_ok && i < m->n_parts; i++)
        marker_ok = fprintf(mf, "%u %llu\n", i + 1,
                            (unsigned long long)m->parts[i].file_size) > 0;
    if (mf && fclose(mf) != 0) marker_ok = 0;
    if (!marker_ok) { remove(marker_tmp); return -1; }
    remove(marker);
    if (rename(marker_tmp, marker) != 0) {
        remove(marker_tmp);
        return -1;
    }
    fprintf(stderr, "idletoken-weights: split local model ready %s, "
                    "reusable coverage [0,%u)\n", out_path, layer_hi);
    return 0;
}

static int local_model_prepare_impl(const char *base_url,
                                    const char *local_primary,
                                    unsigned layer_hi,
                                    const char *cache_dir,
                                    char *out_path, size_t out_cap) {
    if (!base_url || !base_url[0] || !cache_dir || !cache_dir[0] ||
        !out_path || out_cap == 0) return -1;

    char host_port[256], path[1024], idx_path[1088];
    if (parse_http_url(base_url, host_port, sizeof(host_port),
                       path, sizeof(path)) != 0) {
        fprintf(stderr, "idletoken-weights: bad local model repo url: %s\n",
                base_url);
        return -1;
    }
    snprintf(idx_path, sizeof(idx_path), "%s.idx", path);
    fprintf(stderr, "idletoken-weights: fetching local model index %s%s\n",
            host_port, idx_path);
    char *idx = http_get_all(host_port, idx_path);
    if (!idx) {
        fprintf(stderr, "idletoken-weights: could not fetch %s\n", idx_path);
        return -1;
    }

    const uint64_t idx_hash = fnv1a_update(
        UINT64_C(0xcbf29ce484222325), idx, strlen(idx));
    idx_manifest_t manifest;
    if (idx_manifest_parse(idx, &manifest) != 0) {
        fprintf(stderr, "idletoken-weights: bad local model index format\n");
        free(idx);
        return -1;
    }
    unsigned n_layers = 0;
    if (idx_layer_count(idx, &n_layers) != 0 || layer_hi > n_layers) {
        fprintf(stderr, "idletoken-weights: local model plan [0,%u) does not "
                        "match the repository index (%u repeating layers)\n",
                layer_hi, n_layers);
        idx_manifest_clear(&manifest);
        free(idx);
        return -1;
    }
    if (ensure_dir_tree(cache_dir) != 0) {
        fprintf(stderr, "idletoken-weights: cannot create local model cache "
                        "%s: %s\n", cache_dir, strerror(errno));
        idx_manifest_clear(&manifest);
        free(idx);
        return -1;
    }
    if (manifest.version == 3) {
        int rc = local_split_prepare(host_port, path, local_primary,
                                     &manifest, idx_hash,
                                     layer_hi, cache_dir, out_path, out_cap);
        idx_manifest_clear(&manifest);
        free(idx);
        return rc;
    }
    idx_manifest_clear(&manifest);

    const size_t cdl = strlen(cache_dir);
    const char *sep = cdl && (cache_dir[cdl - 1] == '/' ||
                              cache_dir[cdl - 1] == '\\') ? "" : "/";
    int pn = snprintf(out_path, out_cap, "%s%slocal-%016llx.gguf",
                      cache_dir, sep, (unsigned long long)idx_hash);
    if (pn < 0 || (size_t)pn >= out_cap) {
        free(idx);
        return -1;
    }
    char marker[1280];
    snprintf(marker, sizeof(marker), "%s.done", out_path);

    unsigned coverage = 0;
    uint64_t file_size = 0;
    int warm = 0;
    FILE *mf = fopen(marker, "r");
    if (mf) {
        unsigned long long mh = 0, msz = 0;
        unsigned mcov = 0;
        if (fscanf(mf, "IDLETOKEN_LOCAL_VIEW_V1 %llx %u %llu",
                   &mh, &mcov, &msz) == 3 &&
            (uint64_t)mh == idx_hash && mcov <= n_layers &&
            file_size_of(out_path) == (uint64_t)msz) {
            coverage = mcov;
            file_size = (uint64_t)msz;
            warm = 1;
        }
        fclose(mf);
    }
    if (warm && coverage >= layer_hi) {
        fprintf(stderr, "idletoken-weights: reuse local model view %s "
                        "(cached [0,%u), current plan [0,%u))\n",
                out_path, coverage, layer_hi);
        free(idx);
        return 0;
    }

    range_t *ranges = NULL;
    int nr = compute_ranges_mode(idx,
                                 warm ? coverage : 0,
                                 layer_hi,
                                 warm ? 0 : 1,
                                 warm ? 0 : 1,
                                 &file_size, &ranges);
    free(idx);
    if (nr < 0) {
        fprintf(stderr, "idletoken-weights: bad local model index format\n");
        return -1;
    }
    uint64_t need = 0;
    for (int i = 0; i < nr; i++) need += ranges[i].e - ranges[i].s;

    if (!warm) {
        remove(marker);
        remove(out_path);
    }
    int fd = open(out_path, OPEN_FLAGS, 0600);
    if (fd < 0) {
        fprintf(stderr, "idletoken-weights: open local model view %s: %s\n",
                out_path, strerror(errno));
        free(ranges);
        return -1;
    }
    file_make_sparse(fd);
    if (file_truncate64(fd, file_size) != 0) {
        fprintf(stderr, "idletoken-weights: truncate local model view %s to "
                        "%llu failed\n", out_path,
                (unsigned long long)file_size);
        close(fd);
        free(ranges);
        return -1;
    }
    fprintf(stderr, "idletoken-weights: %s local model view to [0,%u): "
                    "%d ranges, %.2f GB new bytes -> %s\n",
            warm ? "extending" : "building", layer_hi, nr, need / 1e9,
            out_path);
    uint64_t done_bytes = 0;
    for (int i = 0; i < nr; i++) {
        int range_rc = local_primary
            ? copy_range_pieces(local_primary, ranges[i].s, ranges[i].e, fd,
                                &done_bytes, need)
            : fetch_range_pieces(host_port, path, ranges[i].s, ranges[i].e, fd,
                                 &done_bytes, need);
        if (range_rc != 0) {
            fprintf(stderr, "idletoken-weights: local model range %llu-%llu "
                            "fetch failed\n",
                    (unsigned long long)ranges[i].s,
                    (unsigned long long)ranges[i].e);
            close(fd);
            free(ranges);
            return -1;
        }
    }
    free(ranges);
#ifdef _WIN32
    if (_commit(fd) != 0) {
#else
    if (fsync(fd) != 0) {
#endif
        fprintf(stderr, "idletoken-weights: sync local model view %s failed: %s\n",
                out_path, strerror(errno));
        close(fd);
        return -1;
    }
    close(fd);

    char marker_tmp[1320];
    snprintf(marker_tmp, sizeof(marker_tmp), "%s.part", marker);
    mf = fopen(marker_tmp, "w");
    int marker_ok = 0;
    if (mf) {
        marker_ok = fprintf(mf,
                            "IDLETOKEN_LOCAL_VIEW_V1 %016llx %u %llu\n",
                            (unsigned long long)idx_hash, layer_hi,
                            (unsigned long long)file_size) > 0;
        if (fclose(mf) != 0) marker_ok = 0;
    }
    if (!marker_ok) {
        remove(marker_tmp);
        fprintf(stderr, "idletoken-weights: cannot publish local model marker %s\n",
                marker);
        return -1;
    }
    remove(marker);
    if (rename(marker_tmp, marker) != 0) {
        fprintf(stderr, "idletoken-weights: rename %s -> %s: %s\n",
                marker_tmp, marker, strerror(errno));
        remove(marker_tmp);
        return -1;
    }
    fprintf(stderr, "idletoken-weights: local model ready %s, reusable "
                    "coverage [0,%u)\n", out_path, layer_hi);
    return 0;
}

int idletoken_local_model_prepare(const char *base_url,
                                  unsigned layer_hi,
                                  const char *cache_dir,
                                  char *out_path, size_t out_cap) {
    return local_model_prepare_impl(base_url, NULL, layer_hi, cache_dir,
                                    out_path, out_cap);
}

int idletoken_local_model_prepare_from_file(const char *base_url,
                                            const char *local_primary_gguf,
                                            unsigned layer_hi,
                                            const char *cache_dir,
                                            char *out_path, size_t out_cap) {
    if (!local_primary_gguf || !local_primary_gguf[0]) return -1;
    return local_model_prepare_impl(base_url, local_primary_gguf, layer_hi,
                                    cache_dir, out_path, out_cap);
}

static int prefetch_local_ranges(const char *path, const range_t *ranges,
                                 int nr, uint64_t *bytes_done) {
#ifdef _WIN32
    /* ReadFile warms the Cache Manager, but that is not the same residency as
     * llama.cpp's mapped-file section on Windows. On the two-PC DSv4 bed a
     * complete 25.7 GiB ReadFile pass still left the first prompt doing about
     * 1.3 GiB of hard faults. Map and touch the exact local tensor ranges
     * instead, and intentionally keep those views alive for the coordinator's
     * lifetime. The server maps the same sparse files, so both processes then
     * reference the same physical pages; remote-layer holes are never mapped
     * or touched. */
    typedef struct pinned_view {
        void *base;
        struct pinned_view *next;
    } pinned_view;
    static pinned_view *kept_views;
    static volatile uint8_t touch_sink;

    HANDLE file = CreateFileA(path, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE |
                                  FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                              NULL);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "idletoken-weights: open local prefetch view %s: "
                        "Windows error %lu\n",
                path, (unsigned long)GetLastError());
        return -1;
    }
    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        fprintf(stderr, "idletoken-weights: map local prefetch view %s: "
                        "Windows error %lu\n",
                path, (unsigned long)GetLastError());
        CloseHandle(file);
        return -1;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;
    const size_t page = si.dwPageSize;
    int rc = 0;
    for (int i = 0; i < nr && rc == 0; i++) {
        const uint64_t aligned = ranges[i].s - ranges[i].s % gran;
        const uint64_t delta64 = ranges[i].s - aligned;
        const uint64_t data64 = ranges[i].e - ranges[i].s;
        const uint64_t view64 = delta64 + data64;
        if (view64 == 0 || view64 > (uint64_t)SIZE_MAX) {
            rc = -1;
            break;
        }
        void *view = MapViewOfFile(mapping, FILE_MAP_READ,
                                   (DWORD)(aligned >> 32), (DWORD)aligned,
                                   (SIZE_T)view64);
        if (!view) {
            fprintf(stderr, "idletoken-weights: mapped prefetch of %s at %llu "
                            "failed: Windows error %lu\n",
                    path, (unsigned long long)ranges[i].s,
                    (unsigned long)GetLastError());
            rc = -1;
            break;
        }
        pinned_view *keep = (pinned_view *)malloc(sizeof(*keep));
        if (!keep) {
            UnmapViewOfFile(view);
            rc = -1;
            break;
        }
        keep->base = view;
        keep->next = kept_views;
        kept_views = keep;

        volatile const uint8_t *data =
            (volatile const uint8_t *)view + (size_t)delta64;
        uint8_t sink = touch_sink;
        for (uint64_t off = 0; off < data64; off += page)
            sink ^= data[(size_t)off];
        sink ^= data[(size_t)(data64 - 1)];
        touch_sink = sink;
        *bytes_done += data64;
    }
    CloseHandle(mapping); /* views remain valid until UnmapViewOfFile */
    CloseHandle(file);
    return rc;
#else
    int fd = open(path, RDONLY_FLAGS);
    if (fd < 0) {
        fprintf(stderr, "idletoken-weights: open local prefetch view %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    const size_t cap = 8u << 20;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) {
        close(fd);
        return -1;
    }
    int rc = 0;
    for (int i = 0; i < nr && rc == 0; i++) {
        if (file_seek64(fd, ranges[i].s) != 0) {
            rc = -1;
            break;
        }
        uint64_t left = ranges[i].e - ranges[i].s;
        while (left > 0) {
            unsigned want = (unsigned)(left < cap ? left : cap);
            int got = (int)read(fd, buf, want);
            if (got <= 0) {
                fprintf(stderr, "idletoken-weights: local prefetch read %s at "
                                "%llu failed: %s\n", path,
                        (unsigned long long)(ranges[i].e - left),
                        got == 0 ? "unexpected EOF" : strerror(errno));
                rc = -1;
                break;
            }
            left -= (uint64_t)got;
            *bytes_done += (uint64_t)got;
        }
    }
    free(buf);
    close(fd);
    return rc;
#endif
}

int idletoken_local_model_prefetch(const char *base_url,
                                   unsigned layer_hi,
                                   const char *local_view_primary_gguf) {
    if (!base_url || !base_url[0] || !local_view_primary_gguf ||
        !local_view_primary_gguf[0]) return -1;

    char host_port[256], path[1024], idx_path[1088];
    if (parse_http_url(base_url, host_port, sizeof(host_port),
                       path, sizeof(path)) != 0) return -1;
    snprintf(idx_path, sizeof(idx_path), "%s.idx", path);
    char *idx = http_get_all(host_port, idx_path);
    if (!idx) {
        fprintf(stderr, "idletoken-weights: could not fetch local prefetch "
                        "index %s\n", idx_path);
        return -1;
    }

    idx_manifest_t manifest;
    if (idx_manifest_parse(idx, &manifest) != 0) {
        free(idx);
        fprintf(stderr, "idletoken-weights: bad local prefetch index format\n");
        return -1;
    }
    unsigned n_layers = 0;
    if (idx_layer_count(idx, &n_layers) != 0 || layer_hi > n_layers) {
        idx_manifest_clear(&manifest);
        free(idx);
        return -1;
    }

    uint64_t bytes = 0;
    int rc = 0;
    if (manifest.version == 3) {
        for (unsigned part = 0; part < manifest.n_parts && rc == 0; part++) {
            uint64_t fsz = 0;
            range_t *ranges = NULL;
            int nr = manifest_ranges(&manifest, part, 0, layer_hi, 1, 1,
                                     &fsz, &ranges);
            char local[1600];
            if (nr < 0 || file_sibling_path(local_view_primary_gguf,
                                             manifest.parts[part].name,
                                             local, sizeof(local)) != 0) {
                free(ranges);
                rc = -1;
                break;
            }
            rc = prefetch_local_ranges(local, ranges, nr, &bytes);
            free(ranges);
        }
    } else {
        uint64_t fsz = 0;
        range_t *ranges = NULL;
        int nr = compute_ranges_mode(idx, 0, layer_hi, 1, 1,
                                     &fsz, &ranges);
        if (nr < 0) rc = -1;
        else rc = prefetch_local_ranges(local_view_primary_gguf, ranges,
                                        nr, &bytes);
        free(ranges);
    }
    idx_manifest_clear(&manifest);
    free(idx);
    if (rc == 0)
        fprintf(stderr, "idletoken-weights: prefetched local CPU layers "
                        "[0,%u): %.2f GiB resident before inference readiness\n",
                layer_hi, (double)bytes / 1073741824.0);
    return rc;
}

int idletoken_rpc_cache_fetch(const char *base_url,
                              unsigned layer_lo, unsigned layer_hi,
                              const char *local_primary_gguf,
                              const char *cache_dir,
                              idletoken_weights_progress_fn progress,
                              void *progress_opaque,
                              uint64_t *bytes_out,
                              unsigned *tensors_out) {
    if (bytes_out) *bytes_out = 0;
    if (tensors_out) *tensors_out = 0;
    if (!base_url || !base_url[0] || !cache_dir || !cache_dir[0] ||
        layer_hi < layer_lo) return -1;

    char host_port[256], path[1024], idx_path[1088];
    if (parse_http_url(base_url, host_port, sizeof(host_port),
                       path, sizeof(path)) != 0) {
        fprintf(stderr, "idletoken-weights: bad RPC cache repo url: %s\n",
                base_url);
        return -1;
    }
    snprintf(idx_path, sizeof(idx_path), "%s.idx", path);
    fprintf(stderr, "idletoken-weights: fetching RPC cache index %s%s\n",
            host_port, idx_path);
    char *idx = http_get_all(host_port, idx_path);
    if (!idx) {
        fprintf(stderr, "idletoken-weights: could not fetch %s\n", idx_path);
        return -1;
    }

    const uint64_t idx_hash = fnv1a_update(
        UINT64_C(0xcbf29ce484222325), idx, strlen(idx));
    idx_manifest_t manifest;
    if (idx_manifest_parse(idx, &manifest) != 0) {
        fprintf(stderr, "idletoken-weights: malformed RPC cache index header\n");
        free(idx);
        return -1;
    }
    if (local_primary_gguf && local_primary_gguf[0]) {
        // The coordinator's index is authoritative for both identity and
        // offsets. A local file is usable only if every sibling has the exact
        // indexed size; an explicit but wrong copy fails loudly instead of
        // falling back to re-downloading tensors over the LAN.
        for (unsigned part = 0; part < manifest.n_parts; part++) {
            char source[1600];
            const char *candidate = local_primary_gguf;
            if (manifest.version == 3) {
                if (file_sibling_path(local_primary_gguf,
                                      manifest.parts[part].name,
                                      source, sizeof(source)) != 0) {
                    idx_manifest_clear(&manifest);
                    free(idx);
                    return -1;
                }
                candidate = source;
            }
            if (file_size_of(candidate) != manifest.parts[part].file_size) {
                fprintf(stderr, "idletoken-weights: local GGUF part does not "
                                "match the cluster index: %s\n", candidate);
                idx_manifest_clear(&manifest);
                free(idx);
                return -1;
            }
        }
    }
    cache_tensor_t *selected = (cache_tensor_t *)calloc(
        manifest.n_tensors, sizeof(*selected));
    if (!selected) {
        idx_manifest_clear(&manifest);
        free(idx);
        return -1;
    }
    unsigned nsel = 0;
    uint64_t total = 0;
    for (size_t i = 0; i < manifest.n_tensors; i++) {
        const idx_tensor_t *t = &manifest.tensors[i];
        if (t->layer < 0 || ((unsigned long long)t->layer >= layer_lo &&
                             (unsigned long long)t->layer < layer_hi)) {
            if (UINT64_MAX - total < t->bytes) {
                free(selected);
                idx_manifest_clear(&manifest);
                free(idx);
                return -1;
            }
            selected[nsel].part = t->part;
            selected[nsel].layer = t->layer;
            selected[nsel].off = t->off;
            selected[nsel].bytes = t->bytes;
            snprintf(selected[nsel].name, sizeof(selected[nsel].name), "%s",
                     t->name);
            if (manifest.version == 3)
                snprintf(selected[nsel].part_name,
                         sizeof(selected[nsel].part_name), "%s",
                         manifest.parts[t->part].name);
            total += t->bytes;
            nsel++;
        }
    }
    idx_manifest_clear(&manifest);
    free(idx);

    if (ensure_dir_one(cache_dir) != 0) {
        fprintf(stderr, "idletoken-weights: cannot create RPC cache %s: %s\n",
                cache_dir, strerror(errno));
        free(selected);
        return -1;
    }

    char marker[1200];
    snprintf(marker, sizeof(marker), "%s/idletoken-%016llx-L%u-%u.done",
             cache_dir, (unsigned long long)idx_hash, layer_lo, layer_hi);
    if (rpc_cache_marker_valid(marker, cache_dir, idx_hash,
                               layer_lo, layer_hi, total, nsel)) {
        fprintf(stderr, "idletoken-weights: reuse warm RPC cache for layers "
                        "[%u,%u): %u tensors, %.2f GB\n",
                layer_lo, layer_hi, nsel, total / 1e9);
        if (progress) progress(total, total, progress_opaque);
        if (bytes_out) *bytes_out = total;
        if (tensors_out) *tensors_out = nsel;
        free(selected);
        return 0;
    }

    fprintf(stderr, "idletoken-weights: seeding RPC cache layers [%u,%u): "
                    "%u tensors, %.2f GB from %s\n",
            layer_lo, layer_hi, nsel, total / 1e9,
            local_primary_gguf && local_primary_gguf[0]
                ? "this node's complete GGUF" : "the cluster repository");
    uint64_t *hashes = (uint64_t *)calloc(nsel ? nsel : 1, sizeof(*hashes));
    uint64_t *name_hashes = (uint64_t *)calloc(nsel ? nsel : 1, sizeof(*name_hashes));
    if (!hashes || !name_hashes) {
        free(name_hashes); free(hashes); free(selected); return -1;
    }
    uint64_t done = 0;
    for (unsigned i = 0; i < nsel; i++) {
        name_hashes[i] = fnv1a_update(UINT64_C(0xcbf29ce484222325),
                                      selected[i].name,
                                      strlen(selected[i].name));
        if (cache_resume_name(cache_dir, idx_hash, name_hashes[i],
                              selected[i].bytes,
                              &hashes[i])) {
            done += selected[i].bytes;
            if (progress) progress(done, total, progress_opaque);
            continue;
        }
        char tmp[1200], final[1200], tensor_path[1200];
        const char *fetch_path = path;
        const char *copy_source = local_primary_gguf;
        char local_tensor_path[1600];
        if (selected[i].part_name[0]) {
            if (repo_sibling_path(path, selected[i].part_name,
                                  tensor_path, sizeof(tensor_path)) != 0) {
                free(name_hashes); free(hashes); free(selected); return -1;
            }
            fetch_path = tensor_path;
            if (local_primary_gguf && local_primary_gguf[0]) {
                if (file_sibling_path(local_primary_gguf,
                                      selected[i].part_name,
                                      local_tensor_path,
                                      sizeof(local_tensor_path)) != 0) {
                    free(name_hashes); free(hashes); free(selected); return -1;
                }
                copy_source = local_tensor_path;
            }
        }
        snprintf(tmp, sizeof(tmp), "%s/.part-%03u-%016llx-%016llx",
                 cache_dir,
                 selected[i].part + 1,
                 (unsigned long long)selected[i].off,
                 (unsigned long long)selected[i].bytes);
        int ofd = open(tmp, CACHE_OPEN_FLAGS, 0600);
        if (ofd < 0) {
            fprintf(stderr, "idletoken-weights: open %s: %s\n", tmp, strerror(errno));
            free(name_hashes); free(hashes); free(selected); return -1;
        }
        uint64_t hash = UINT64_C(0xcbf29ce484222325);
        const uint64_t end = selected[i].off + selected[i].bytes;
        int frc = -1;
        if (end >= selected[i].off) {
            frc = copy_source && copy_source[0]
                ? copy_tensor_pieces(copy_source, selected[i].off, end, ofd,
                                     &hash, &done, total,
                                     progress, progress_opaque)
                : fetch_tensor_pieces(host_port, fetch_path,
                                      selected[i].off, end, ofd,
                                      &hash, &done, total,
                                      progress, progress_opaque);
        }
        close(ofd);
        if (frc != 0) {
            remove(tmp);
            fprintf(stderr, "idletoken-weights: tensor range %llu-%llu failed\n",
                    (unsigned long long)selected[i].off,
                    (unsigned long long)end);
            free(name_hashes); free(hashes); free(selected); return -1;
        }
        snprintf(final, sizeof(final), "%s/%016llx", cache_dir,
                 (unsigned long long)hash);
        if (file_size_of(final) == selected[i].bytes) {
            remove(tmp); /* content-addressed duplicate/already warm */
        } else {
            remove(final);
            if (rename(tmp, final) != 0) {
                fprintf(stderr, "idletoken-weights: rename %s -> %s: %s\n",
                        tmp, final, strerror(errno));
                remove(tmp);
                free(name_hashes); free(hashes); free(selected); return -1;
            }
        }
        hashes[i] = hash;
        if (!cache_activate_name(cache_dir, hashes[i], idx_hash,
                                 name_hashes[i],
                                 selected[i].bytes)) {
            fprintf(stderr, "idletoken-weights: cannot activate local tensor %s\n",
                    selected[i].name);
            free(name_hashes); free(hashes); free(selected); return -1;
        }
    }

    char marker_tmp[1240];
    snprintf(marker_tmp, sizeof(marker_tmp), "%s.part", marker);
    FILE *mf = fopen(marker_tmp, "w");
    if (!mf) {
        fprintf(stderr, "idletoken-weights: cannot write cache marker %s: %s\n",
                marker_tmp, strerror(errno));
        free(name_hashes); free(hashes); free(selected); return -1;
    }
    fprintf(mf, "IDLETOKEN_RPC_CACHE_V2 %016llx %u %u %u %llu\n",
            (unsigned long long)idx_hash, layer_lo, layer_hi, nsel,
            (unsigned long long)total);
    for (unsigned i = 0; i < nsel; i++)
        fprintf(mf, "%016llx %llu %016llx\n",
                (unsigned long long)hashes[i],
                (unsigned long long)selected[i].bytes,
                (unsigned long long)name_hashes[i]);
    if (fclose(mf) != 0) {
        remove(marker_tmp);
        free(name_hashes); free(hashes); free(selected); return -1;
    }
    remove(marker);
    if (rename(marker_tmp, marker) != 0) {
        fprintf(stderr, "idletoken-weights: cannot publish cache marker %s: %s\n",
                marker, strerror(errno));
        remove(marker_tmp);
        free(name_hashes); free(hashes); free(selected); return -1;
    }
    fprintf(stderr, "idletoken-weights: RPC cache ready for layers [%u,%u): "
                    "%u tensors, %.2f GB\n",
            layer_lo, layer_hi, nsel, total / 1e9);
    if (bytes_out) *bytes_out = total;
    if (tensors_out) *tensors_out = nsel;
    free(name_hashes);
    free(hashes);
    free(selected);
    return 0;
}
