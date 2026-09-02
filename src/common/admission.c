/* IdleToken — local admission capabilities. See include/idletoken_admission.h
 * for what this is for and, more importantly, for what it is NOT.
 *
 * C99. Links into both the coordinator and the platform agent. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "idletoken_admission.h"
#include "idletoken_privacy.h"
#include "idletoken_sha256.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#ifdef _WIN32
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  if defined(_MSC_VER)
#    define IDLETOKEN_ADM_TLS __declspec(thread)
#  else
#    define IDLETOKEN_ADM_TLS _Thread_local
#  endif
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define IDLETOKEN_ADM_TLS __thread
#endif

/* One process-wide lock, on EVERY platform. Minting and spending are rare (once
 * per dispatched job) and the critical sections are a few hundred bytes of
 * hashing, so a single mutex is the right shape: a reader/writer split here
 * would buy nothing and add a second way to get the replay table wrong.
 *
 * ⚠ This was `((void)0)` on Windows until 2026-08-30, on the stated grounds
 * that "the coordinator's Windows build is single-threaded through this path".
 * It is not. The coordinator serves HTTP from a thread pool
 * (`llama_pool_worker` in coord_main.c, `pthread_create` on all platforms) and
 * `build_coord_win.bat` links winpthread on purpose. So the replay table and
 * the in-flight table were racing on the one platform most providers run —
 * where two concurrent requests carrying the SAME capability could both pass
 * adm_spend_nonce(), which is exactly the single-use property this module
 * exists to provide.
 *
 * pthreads rather than a Windows CRITICAL_SECTION because both binaries that
 * link this file already depend on them there: coord_main.c and
 * platform_agent.c both call pthread_create/pthread_mutex_lock unconditionally,
 * and the Windows link line already carries -lwinpthread. Reaching for a second
 * threading API to avoid a dependency that is already present would be the
 * costlier choice. */
static pthread_mutex_t g_adm_mu = PTHREAD_MUTEX_INITIALIZER;
#define ADM_LOCK()   pthread_mutex_lock(&g_adm_mu)
#define ADM_UNLOCK() pthread_mutex_unlock(&g_adm_mu)

#define ADM_PREFIX      "itadm1"
#define ADM_NONCE_BYTES 16
#define ADM_MAC_BYTES   32

/* Live tickets, for the "used exactly once" half of the contract.
 *
 * 1024 entries against a 120 s TTL is a sustained 8.5 admissions a second — two
 * orders of magnitude above a coordinator that runs one sequence slot. A table
 * that fills anyway is not waved through: see ADM_NO_ROOM below. */
#define ADM_SPENT_SLOTS 1024

/* In-flight platform jobs. One entry per consumed capability, cleared by
 * idletoken_admission_request_end() on the thread that consumed it. */
#define ADM_INFLIGHT_SLOTS 64

typedef struct {
    uint8_t   nonce[ADM_NONCE_BYTES];
    long long expires;          /* 0 = free slot */
} adm_spent;

static struct {
    int       armed;
    uint8_t   channel_key[IDLETOKEN_ADM_KEY_BYTES];
    char      channel_hex[IDLETOKEN_ADM_KEYHEX_CAP];
    char      local_hex[IDLETOKEN_ADM_KEYHEX_CAP];
    adm_spent spent[ADM_SPENT_SLOTS];
    long long inflight_since[ADM_INFLIGHT_SLOTS];   /* 0 = free */
    long long minted, consumed, rejected;
} g_adm;

/* Which in-flight slot this thread is holding, +1 (0 = none). Thread-local so
 * request_end() needs no bookkeeping at the call site beyond being called. */
static IDLETOKEN_ADM_TLS int g_adm_my_slot;

const char *idletoken_admission_rc_str(idletoken_adm_rc rc) {
    switch (rc) {
    case IDLETOKEN_ADM_OK:            return "ok";
    case IDLETOKEN_ADM_ABSENT:        return "no capability presented";
    case IDLETOKEN_ADM_NOT_ARMED:     return "this coordinator mints no capabilities";
    case IDLETOKEN_ADM_MALFORMED:     return "not a v1 admission capability";
    case IDLETOKEN_ADM_BAD_MAC:       return "signature does not verify";
    case IDLETOKEN_ADM_EXPIRED:       return "expired";
    case IDLETOKEN_ADM_REPLAYED:      return "already spent";
    case IDLETOKEN_ADM_BODY_MISMATCH: return "minted for a different request body";
    case IDLETOKEN_ADM_NO_ROOM:       return "too many live capabilities";
    }
    return "unknown";
}

void idletoken_admission_hex(const uint8_t *in, size_t n, char *out, size_t cap) {
    static const char *H = "0123456789abcdef";
    size_t i = 0;
    if (!out || cap == 0) return;
    for (; i < n && (i * 2 + 2) < cap; i++) {
        out[i * 2]     = H[(in[i] >> 4) & 0xf];
        out[i * 2 + 1] = H[in[i] & 0xf];
    }
    out[i * 2] = '\0';
}

static int adm_hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Exactly `want` bytes of hex and then the terminator; anything else is -1.
 *
 * The two nibbles are read ONE AT A TIME and each is judged before the next is
 * touched. The obvious spelling reads both first, and on an odd-length string
 * that means reading one byte past the NUL — harmless on the stack buffers this
 * had at its two call sites, and a heap overread the first time somebody passes
 * a short malloc'd string. Fixed here rather than documented as safe-for-now. */
static int adm_unhex(const char *hex, uint8_t *out, size_t want) {
    size_t i;
    if (!hex) return -1;
    for (i = 0; i < want; i++) {
        int hi = adm_hexval(hex[i * 2]);
        int lo;
        if (hi < 0) return -1;                 /* covers the NUL: not a digit */
        lo = adm_hexval(hex[i * 2 + 1]);
        if (lo < 0) return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return hex[want * 2] == '\0' ? 0 : -1;
}

/* Constant time over the whole buffer: a compare that returns early leaks how
 * many leading bytes were right, which is enough to walk a MAC out one byte at
 * a time given enough tries. */
static int adm_eq_ct(const uint8_t *a, const uint8_t *b, size_t n) {
    uint8_t d = 0;
    size_t i;
    for (i = 0; i < n; i++) d |= (uint8_t)(a[i] ^ b[i]);
    return d == 0;
}

static int adm_streq_ct(const char *a, const char *b) {
    size_t la, lb, i;
    uint8_t d;
    if (!a || !b) return 0;
    la = strlen(a); lb = strlen(b);
    if (la != lb) return 0;             /* length is not a secret */
    d = 0;
    for (i = 0; i < la; i++) d |= (uint8_t)(a[i] ^ b[i]);
    return d == 0;
}

/* HMAC-SHA256. Written here rather than reused from discovery.c because the
 * platform agent's link (Makefile.platform) does not include discovery.c, and
 * pulling the cluster's discovery/pairing machinery into the agent to get one
 * hash would be a much larger change than 30 lines of RFC 2104. */
static void adm_hmac(const uint8_t *key, size_t key_len,
                     const uint8_t *msg, size_t msg_len, uint8_t out[32]) {
    uint8_t k[64], ipad[64], opad[64], inner[32];
    idletoken_sha256_ctx c;
    size_t i;

    memset(k, 0, sizeof k);
    if (key_len > 64) idletoken_sha256(key, key_len, k);
    else if (key_len)  memcpy(k, key, key_len);

    for (i = 0; i < 64; i++) { ipad[i] = (uint8_t)(k[i] ^ 0x36); opad[i] = (uint8_t)(k[i] ^ 0x5c); }

    idletoken_sha256_init(&c);
    idletoken_sha256_update(&c, ipad, 64);
    idletoken_sha256_update(&c, msg, msg_len);
    idletoken_sha256_final(&c, inner);

    idletoken_sha256_init(&c);
    idletoken_sha256_update(&c, opad, 64);
    idletoken_sha256_update(&c, inner, 32);
    idletoken_sha256_final(&c, out);

    idletoken_secure_zero(k, sizeof k);
    idletoken_secure_zero(ipad, sizeof ipad);
    idletoken_secure_zero(opad, sizeof opad);
    idletoken_secure_zero(inner, sizeof inner);
}

void idletoken_admission_body_hash(const void *body, size_t len, uint8_t out[32]) {
    idletoken_sha256(body ? body : "", body ? len : 0, out);
}

/* The signed string. Domain-separated and field-separated by '\n', with every
 * field drawn from a charset that cannot contain '\n' — so no two different
 * (job, exp, nonce, body) tuples can produce the same input. The domain prefix
 * is what stops a MAC computed for some other purpose under the same key from
 * counting here.
 *
 * The body hash is a FIELD OF THE TICKET as well as an input to the MAC. It
 * could have been left implicit (recompute it from the request and hash that
 * in), and that was the first shape — but then "forged" and "minted for another
 * request" are literally the same event, a MAC that does not match, and the
 * gate cannot tell a verifier that refuses everything from one that refuses the
 * right things. Carrying it lets the MAC be checked first, on the ticket's own
 * claim, and the claim then compared to the request: two failures, two names,
 * both authenticated. */
static void adm_mac(const uint8_t key[IDLETOKEN_ADM_KEY_BYTES],
                    const char *job, long long exp, const char *nonce_hex,
                    const char *body_hex, uint8_t out[ADM_MAC_BYTES]) {
    char msg[512];
    int n = snprintf(msg, sizeof msg,
                     "idletoken-admission-v1\n%s\n%lld\n%s\n%s\n",
                     job, exp, nonce_hex, body_hex);
    if (n < 0) n = 0;
    if ((size_t)n >= sizeof msg) n = (int)sizeof msg - 1;
    adm_hmac(key, IDLETOKEN_ADM_KEY_BYTES, (const uint8_t *)msg, (size_t)n, out);
}

/* Job ids are used verbatim inside the MAC input, so the charset is a
 * correctness requirement, not tidiness: an id containing the separator would
 * let two different tickets hash identically. Refused rather than escaped —
 * escaping is a second encoding to get wrong. */
static int adm_job_ok(const char *job) {
    size_t i;
    if (!job || !job[0]) return 0;
    for (i = 0; job[i]; i++) {
        char ch = job[i];
        if (i >= IDLETOKEN_ADM_JOB_CAP - 1) return 0;
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == ':')
            continue;
        return 0;
    }
    return 1;
}

/* --- arming --------------------------------------------------------------- */

/* 0600 on POSIX. On Windows the file inherits the user's profile ACL, which is
 * the same boundary the settings file and the engine log already sit behind;
 * there is no weaker claim being made here than elsewhere in the product. */
static int adm_write_private(const char *path, const char *hex, char *err, size_t err_cap) {
    FILE *f;
    int fd;
#ifdef _WIN32
    fd = _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC, _S_IREAD | _S_IWRITE);
#else
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
#endif
    if (fd < 0) {
        if (err && err_cap) snprintf(err, err_cap, "cannot write %s: %s", path, strerror(errno));
        return -1;
    }
#ifdef _WIN32
    f = _fdopen(fd, "w");
#else
    f = fdopen(fd, "w");
#endif
    if (!f) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        if (err && err_cap) snprintf(err, err_cap, "cannot write %s: %s", path, strerror(errno));
        return -1;
    }
    fprintf(f, "%s\n", hex);
    fclose(f);
#ifndef _WIN32
    /* open()'s mode is masked by umask, and an upgraded install may already own
     * the file with a wider mode. Say what we want outright. */
    chmod(path, 0600);
#endif
    return 0;
}

static int adm_read_hex(const char *path, char *out, size_t cap) {
    FILE *f = fopen(path, "r");
    char line[IDLETOKEN_ADM_KEYHEX_CAP + 8];
    uint8_t probe[IDLETOKEN_ADM_KEY_BYTES];
    size_t n;
    if (!f) return -1;
    if (!fgets(line, sizeof line, f)) { fclose(f); return -1; }
    fclose(f);
    n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = '\0';
    if (adm_unhex(line, probe, IDLETOKEN_ADM_KEY_BYTES) != 0) return -1;
    idletoken_secure_zero(probe, sizeof probe);
    if (n + 1 > cap) return -1;
    memcpy(out, line, n + 1);
    return 0;
}

int idletoken_admission_init(const char *channel_path, const char *local_path,
                             char *err, size_t err_cap) {
    uint8_t local_key[IDLETOKEN_ADM_KEY_BYTES];
    int rc = 0;
    if (err && err_cap) err[0] = '\0';

    ADM_LOCK();
    memset(g_adm.spent, 0, sizeof g_adm.spent);
    memset(g_adm.inflight_since, 0, sizeof g_adm.inflight_since);
    /* Fresh every start. The channel key authenticates minting, so a key that
     * outlived the process it was minted for would let a ticket survive a
     * restart it has no business surviving. */
    idletoken_random_bytes(g_adm.channel_key, sizeof g_adm.channel_key);
    idletoken_admission_hex(g_adm.channel_key, sizeof g_adm.channel_key,
                            g_adm.channel_hex, sizeof g_adm.channel_hex);
    g_adm.armed = 1;
    ADM_UNLOCK();

    /* The local-origin marker is REUSED across restarts when one is already on
     * disk. It is an attribution marker, not a secret, and a client that
     * cached it must not be silently demoted to "unattributed" because the
     * coordinator restarted under it — that failure would look exactly like the
     * attack this whole mechanism exists to stop. */
    {
        char local_hex[IDLETOKEN_ADM_KEYHEX_CAP];
        int  reused = local_path && local_path[0] &&
                      adm_read_hex(local_path, local_hex, sizeof local_hex) == 0;
        if (!reused) {
            idletoken_random_bytes(local_key, sizeof local_key);
            idletoken_admission_hex(local_key, sizeof local_key,
                                    local_hex, sizeof local_hex);
            idletoken_secure_zero(local_key, sizeof local_key);
            if (local_path && local_path[0] &&
                adm_write_private(local_path, local_hex, err, err_cap) != 0)
                rc = -1;
        }
        ADM_LOCK();
        memcpy(g_adm.local_hex, local_hex, sizeof g_adm.local_hex);
        ADM_UNLOCK();
    }

    if (channel_path && channel_path[0] &&
        adm_write_private(channel_path, g_adm.channel_hex, err, err_cap) != 0)
        rc = -1;

    return rc;
}

/* --- attaching (the platform agent's side) ---------------------------------
 *
 * The agent does not have its own channel: it mints under the coordinator's, so
 * that the coordinator beside it can recognise the ticket. Which is the whole
 * asymmetry of this mechanism and is worth being explicit about — the agent
 * proves it can talk to THIS coordinator, not that it is honest. A modified
 * agent on the same machine can read the same file (accepted HOST-02/HOST-16
 * boundary); what it can no longer do is omit the marker and be READ AS LOCAL,
 * because under the strict policy absence is refused rather than promoted. */
int idletoken_admission_attach(const char *channel_hex, char *err, size_t err_cap) {
    uint8_t key[IDLETOKEN_ADM_KEY_BYTES];
    if (err && err_cap) err[0] = '\0';
    if (!channel_hex || adm_unhex(channel_hex, key, sizeof key) != 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "the admission channel key is not %d hex bytes",
                     IDLETOKEN_ADM_KEY_BYTES);
        idletoken_secure_zero(key, sizeof key);
        return -1;
    }
    ADM_LOCK();
    memset(g_adm.spent, 0, sizeof g_adm.spent);
    memset(g_adm.inflight_since, 0, sizeof g_adm.inflight_since);
    memcpy(g_adm.channel_key, key, sizeof g_adm.channel_key);
    idletoken_admission_hex(g_adm.channel_key, sizeof g_adm.channel_key,
                            g_adm.channel_hex, sizeof g_adm.channel_hex);
    /* No local-origin marker on this side. Attaching is for MINTING; a process
     * that could also hand out the local marker would be handing out the very
     * thing that says "a human at this machine asked for this". */
    g_adm.local_hex[0] = '\0';
    g_adm.armed = 1;
    ADM_UNLOCK();
    idletoken_secure_zero(key, sizeof key);
    return 0;
}

int idletoken_admission_attach_file(const char *channel_path,
                                    char *err, size_t err_cap) {
    char hex[IDLETOKEN_ADM_KEYHEX_CAP];
    if (err && err_cap) err[0] = '\0';
    if (!channel_path || !channel_path[0] ||
        adm_read_hex(channel_path, hex, sizeof hex) != 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "cannot read an admission channel key from %s",
                     channel_path && channel_path[0] ? channel_path : "(no path)");
        return -1;
    }
    {
        int rc = idletoken_admission_attach(hex, err, err_cap);
        idletoken_secure_zero(hex, sizeof hex);
        return rc;
    }
}

/* See the header for why an attachment is a cache and not a latch. The compare
 * is `channel_ok`, i.e. the same constant-time comparison the coordinator uses,
 * rather than a mtime or a size: a coordinator that restarts twice inside one
 * filesystem timestamp tick would defeat a mtime check silently, and silence is
 * the failure mode this whole function exists to remove. */
int idletoken_admission_attach_file_if_changed(const char *channel_path,
                                               char *err, size_t err_cap) {
    char hex[IDLETOKEN_ADM_KEYHEX_CAP];
    int rc;
    if (err && err_cap) err[0] = '\0';
    if (!channel_path || !channel_path[0] ||
        adm_read_hex(channel_path, hex, sizeof hex) != 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "cannot read an admission channel key from %s",
                     channel_path && channel_path[0] ? channel_path : "(no path)");
        return -1;
    }
    if (idletoken_admission_channel_ok(hex)) {
        idletoken_secure_zero(hex, sizeof hex);
        return 0;                       /* already attached to this very key */
    }
    rc = idletoken_admission_attach(hex, err, err_cap);
    idletoken_secure_zero(hex, sizeof hex);
    return rc == 0 ? 1 : -1;
}

/* One definition of where the files live, used by the coordinator that writes
 * them and the agent that reads them. Two copies of a path that must match is
 * how a rename becomes "the agent silently stopped being able to mint", which
 * on a strict coordinator reads as "the platform stopped sending me work". */
int idletoken_admission_default_paths(char *channel, size_t channel_cap,
                                      char *local, size_t local_cap) {
    const char *home = getenv("IDLETOKEN_STATE_DIR");
    char base[320];
    if (home && home[0]) {
        snprintf(base, sizeof base, "%s", home);
    } else {
#ifdef _WIN32
        const char *h = getenv("USERPROFILE");
#else
        const char *h = getenv("HOME");
#endif
        if (!h || !h[0]) return -1;
        snprintf(base, sizeof base, "%s/.idletoken", h);
    }
    if (channel && channel_cap)
        snprintf(channel, channel_cap, "%s/coord-admission.key", base);
    if (local && local_cap)
        snprintf(local, local_cap, "%s/coord-local-origin.key", base);
    return 0;
}

int idletoken_admission_ready(void) {
    int on;
    ADM_LOCK(); on = g_adm.armed; ADM_UNLOCK();
    return on;
}

int idletoken_admission_channel_ok(const char *presented_hex) {
    int ok;
    ADM_LOCK();
    ok = g_adm.armed && adm_streq_ct(presented_hex, g_adm.channel_hex);
    ADM_UNLOCK();
    return ok;
}

int idletoken_admission_local_ok(const char *presented_hex) {
    int ok;
    ADM_LOCK();
    ok = g_adm.armed && g_adm.local_hex[0] && adm_streq_ct(presented_hex, g_adm.local_hex);
    ADM_UNLOCK();
    return ok;
}

int idletoken_admission_local_marker(char *out, size_t cap) {
    int rc = -1;
    ADM_LOCK();
    if (g_adm.armed && g_adm.local_hex[0] && strlen(g_adm.local_hex) + 1 <= cap) {
        memcpy(out, g_adm.local_hex, strlen(g_adm.local_hex) + 1);
        rc = 0;
    }
    ADM_UNLOCK();
    return rc;
}

/* --- minting -------------------------------------------------------------- */

int idletoken_admission_mint(const char *job_id, const uint8_t body_sha256[32],
                             long long now_unix,
                             char *out, size_t out_cap,
                             char *err, size_t err_cap) {
    uint8_t nonce[ADM_NONCE_BYTES], mac[ADM_MAC_BYTES];
    char nonce_hex[ADM_NONCE_BYTES * 2 + 1], mac_hex[ADM_MAC_BYTES * 2 + 1];
    char body_hex[65];
    long long exp;
    int n;

    if (err && err_cap) err[0] = '\0';
    if (!out || out_cap == 0) return -1;
    out[0] = '\0';
    if (!idletoken_admission_ready()) {
        if (err && err_cap) snprintf(err, err_cap, "admission is not armed");
        return -1;
    }
    if (!adm_job_ok(job_id)) {
        if (err && err_cap)
            snprintf(err, err_cap,
                     "job id must be 1..%d chars of [A-Za-z0-9_:-]",
                     IDLETOKEN_ADM_JOB_CAP - 1);
        return -1;
    }
    if (!body_sha256) {
        if (err && err_cap) snprintf(err, err_cap, "a body hash is required");
        return -1;
    }

    idletoken_random_bytes(nonce, sizeof nonce);
    idletoken_admission_hex(nonce, sizeof nonce, nonce_hex, sizeof nonce_hex);
    idletoken_admission_hex(body_sha256, 32, body_hex, sizeof body_hex);
    exp = now_unix + IDLETOKEN_ADM_TTL_S;

    ADM_LOCK();
    adm_mac(g_adm.channel_key, job_id, exp, nonce_hex, body_hex, mac);
    g_adm.minted++;
    ADM_UNLOCK();

    idletoken_admission_hex(mac, sizeof mac, mac_hex, sizeof mac_hex);
    n = snprintf(out, out_cap, ADM_PREFIX ".%s.%lld.%s.%s.%s",
                 job_id, exp, nonce_hex, body_hex, mac_hex);
    if (n < 0 || (size_t)n >= out_cap) {
        out[0] = '\0';
        if (err && err_cap) snprintf(err, err_cap, "capability does not fit");
        return -1;
    }
    return 0;
}

/* --- spending ------------------------------------------------------------- */

/* Caller holds the lock. Returns 1 if the nonce was recorded, 0 if the table is
 * full of live entries (fail closed), and -1 if it has been seen already. */
static int adm_spend_nonce(const uint8_t nonce[ADM_NONCE_BYTES], long long exp,
                           long long now_unix) {
    int free_slot = -1;
    int i;
    for (i = 0; i < ADM_SPENT_SLOTS; i++) {
        adm_spent *s = &g_adm.spent[i];
        if (s->expires != 0 && s->expires <= now_unix) { s->expires = 0; }   /* aged out */
        if (s->expires == 0) { if (free_slot < 0) free_slot = i; continue; }
        if (adm_eq_ct(s->nonce, nonce, ADM_NONCE_BYTES)) return -1;
    }
    if (free_slot < 0) return 0;
    memcpy(g_adm.spent[free_slot].nonce, nonce, ADM_NONCE_BYTES);
    g_adm.spent[free_slot].expires = exp;
    return 1;
}

/* Caller holds the lock. */
static int adm_inflight_take(long long now_unix) {
    int i;
    for (i = 0; i < ADM_INFLIGHT_SLOTS; i++) {
        if (g_adm.inflight_since[i] != 0 &&
            now_unix - g_adm.inflight_since[i] > IDLETOKEN_ADM_INFLIGHT_MAX_S)
            g_adm.inflight_since[i] = 0;                /* self-heal */
        if (g_adm.inflight_since[i] == 0) {
            g_adm.inflight_since[i] = now_unix;
            return i + 1;
        }
    }
    return 0;
}

idletoken_adm_rc idletoken_admission_consume(const char *ticket,
                                             const uint8_t body_sha256[32],
                                             long long now_unix,
                                             char *out_job, size_t job_cap) {
    char job[IDLETOKEN_ADM_JOB_CAP];
    char nonce_hex[ADM_NONCE_BYTES * 2 + 1];
    char mac_hex[ADM_MAC_BYTES * 2 + 1];
    char body_hex[65];
    uint8_t nonce[ADM_NONCE_BYTES], want[ADM_MAC_BYTES], got[ADM_MAC_BYTES];
    const char *p, *q;
    long long exp;
    size_t n;
    idletoken_adm_rc rc = IDLETOKEN_ADM_MALFORMED;
    int spent;

    if (out_job && job_cap) out_job[0] = '\0';
    if (!ticket || !ticket[0])     return IDLETOKEN_ADM_ABSENT;
    if (!idletoken_admission_ready()) return IDLETOKEN_ADM_NOT_ARMED;

    /* Every parse failure below lands on the SAME exit, which counts a
     * rejection. There is no path out of this function that says "could not
     * parse it, so treat it as an ordinary request": a capability that is
     * present and does not verify is a refusal, because the alternative is that
     * corrupting a ticket becomes an easier way to strip it than deleting it. */
    p = ticket;
    if (strncmp(p, ADM_PREFIX ".", sizeof(ADM_PREFIX)) != 0) goto reject;
    p += sizeof(ADM_PREFIX);

    q = strchr(p, '.');
    if (!q) goto reject;
    n = (size_t)(q - p);
    if (n == 0 || n >= sizeof job) goto reject;
    memcpy(job, p, n); job[n] = '\0';
    if (!adm_job_ok(job)) goto reject;
    p = q + 1;

    q = strchr(p, '.');
    if (!q) goto reject;
    {
        char expbuf[24];
        char *endp = NULL;
        n = (size_t)(q - p);
        if (n == 0 || n >= sizeof expbuf) goto reject;
        memcpy(expbuf, p, n); expbuf[n] = '\0';
        exp = strtoll(expbuf, &endp, 10);
        if (!endp || *endp != '\0') goto reject;
    }
    p = q + 1;

    q = strchr(p, '.');
    if (!q) goto reject;
    n = (size_t)(q - p);
    if (n != ADM_NONCE_BYTES * 2) goto reject;
    memcpy(nonce_hex, p, n); nonce_hex[n] = '\0';
    if (adm_unhex(nonce_hex, nonce, ADM_NONCE_BYTES) != 0) goto reject;
    p = q + 1;

    q = strchr(p, '.');
    if (!q) goto reject;
    n = (size_t)(q - p);
    if (n != 64) goto reject;
    memcpy(body_hex, p, n); body_hex[n] = '\0';
    {
        uint8_t probe[32];
        if (adm_unhex(body_hex, probe, 32) != 0) goto reject;
    }
    p = q + 1;

    n = strlen(p);
    if (n != ADM_MAC_BYTES * 2) goto reject;
    memcpy(mac_hex, p, n); mac_hex[n] = '\0';
    if (adm_unhex(mac_hex, got, ADM_MAC_BYTES) != 0) goto reject;

    /* The MAC is checked FIRST, over the ticket's own fields, before anything
     * those fields say is allowed to influence a decision. Order matters twice:
     * an unauthenticated field must never drive an outcome, and answering
     * "expired" to a forged ticket would confirm that the rest of the shape was
     * right. */
    ADM_LOCK();
    adm_mac(g_adm.channel_key, job, exp, nonce_hex, body_hex, want);
    ADM_UNLOCK();
    if (!adm_eq_ct(want, got, ADM_MAC_BYTES)) { rc = IDLETOKEN_ADM_BAD_MAC; goto reject; }
    if (exp <= now_unix) { rc = IDLETOKEN_ADM_EXPIRED; goto reject; }

    /* Now the body. The hash above is authenticated, so this compares what the
     * coordinator was ASKED to admit against what actually arrived: a ticket
     * minted for one request cannot admit a different one. */
    {
        char actual_hex[65];
        idletoken_admission_hex(body_sha256, 32, actual_hex, sizeof actual_hex);
        if (!body_sha256 || strcmp(actual_hex, body_hex) != 0) {
            rc = IDLETOKEN_ADM_BODY_MISMATCH;
            goto reject;
        }
    }

    ADM_LOCK();
    spent = adm_spend_nonce(nonce, exp, now_unix);
    if (spent == 1) {
        g_adm.consumed++;
        g_adm_my_slot = adm_inflight_take(now_unix);
    } else {
        g_adm.rejected++;
    }
    ADM_UNLOCK();
    if (spent == -1) return IDLETOKEN_ADM_REPLAYED;
    if (spent == 0)  return IDLETOKEN_ADM_NO_ROOM;

    if (out_job && job_cap) snprintf(out_job, job_cap, "%s", job);
    return IDLETOKEN_ADM_OK;

reject:
    ADM_LOCK(); g_adm.rejected++; ADM_UNLOCK();
    return rc;
}

void idletoken_admission_request_end(void) {
    int slot = g_adm_my_slot;
    if (slot <= 0) return;
    g_adm_my_slot = 0;
    ADM_LOCK();
    if (slot <= ADM_INFLIGHT_SLOTS) g_adm.inflight_since[slot - 1] = 0;
    ADM_UNLOCK();
}

int idletoken_admission_platform_busy(long long now_unix) {
    int i, busy = 0;
    ADM_LOCK();
    for (i = 0; i < ADM_INFLIGHT_SLOTS; i++) {
        if (g_adm.inflight_since[i] == 0) continue;
        if (now_unix - g_adm.inflight_since[i] > IDLETOKEN_ADM_INFLIGHT_MAX_S) {
            g_adm.inflight_since[i] = 0;                /* self-heal */
            continue;
        }
        busy++;
    }
    ADM_UNLOCK();
    return busy;
}

void idletoken_admission_counters(long long *minted, long long *consumed,
                                  long long *rejected, int *inflight) {
    ADM_LOCK();
    if (minted)   *minted   = g_adm.minted;
    if (consumed) *consumed = g_adm.consumed;
    if (rejected) *rejected = g_adm.rejected;
    ADM_UNLOCK();
    if (inflight) *inflight = idletoken_admission_platform_busy((long long)time(NULL));
}

/* --- self-test ------------------------------------------------------------ */

int idletoken_admission_selftest(void) {
    int fails = 0;
    char err[160], t_good[IDLETOKEN_ADM_TICKET_CAP], t2[IDLETOKEN_ADM_TICKET_CAP];
    char job[IDLETOKEN_ADM_JOB_CAP];
    uint8_t h1[32], h2[32];
    long long now = 1756500000LL;   /* fixed: a test whose result depends on the
                                     * clock is a test that fails at midnight */

#define AST(cond, name) do { \
        if (cond) fprintf(stderr, "selftest PASS %s\n", name); \
        else      { fprintf(stderr, "selftest FAIL %s\n", name); fails++; } \
    } while (0)

    idletoken_admission_init(NULL, NULL, err, sizeof err);
    idletoken_admission_body_hash("{\"messages\":[]}", 15, h1);
    idletoken_admission_body_hash("{\"messages\":[1]}", 16, h2);

    AST(idletoken_admission_mint("job-A1", h1, now, t_good, sizeof t_good,
                                 err, sizeof err) == 0,
        "admission: a well-formed job mints a capability");

    /* The one that must work, and it must work only once. */
    AST(idletoken_admission_consume(t_good, h1, now + 1, job, sizeof job) == IDLETOKEN_ADM_OK,
        "admission: a fresh capability is accepted");
    AST(!strcmp(job, "job-A1"),
        "admission: the accepted capability names its own job");
    AST(idletoken_admission_platform_busy(now + 1) == 1,
        "admission: a consumed capability marks a platform job in flight");
    idletoken_admission_request_end();
    AST(idletoken_admission_platform_busy(now + 1) == 0,
        "admission: the in-flight mark clears when the request ends");
    AST(idletoken_admission_consume(t_good, h1, now + 2, NULL, 0) == IDLETOKEN_ADM_REPLAYED,
        "admission: the same capability cannot be spent twice");

    /* Bound to THIS body: a capability minted for one request may not admit
     * another. Without this, one ticket plus a rewritten body is a free pass.
     * The reason must be its OWN reason and not a generic bad signature, or the
     * gate cannot tell this check from the one above it. */
    idletoken_admission_mint("job-B", h1, now, t2, sizeof t2, err, sizeof err);
    AST(idletoken_admission_consume(t2, h2, now + 1, NULL, 0) == IDLETOKEN_ADM_BODY_MISMATCH,
        "admission: a capability minted for another body is refused as such");
    AST(idletoken_admission_consume(t2, h1, now + 1, NULL, 0) == IDLETOKEN_ADM_OK,
        "admission: ...and the SAME capability still works on its own body "
        "(a refusal above is the body check, not a dead verifier)");
    idletoken_admission_request_end();

    /* Expiry is inside the MAC, so moving it forward invalidates the ticket
     * rather than extending it; the honest expiry path is tested by waiting. */
    idletoken_admission_mint("job-C", h1, now, t2, sizeof t2, err, sizeof err);
    AST(idletoken_admission_consume(t2, h1, now + IDLETOKEN_ADM_TTL_S + 1, NULL, 0)
        == IDLETOKEN_ADM_EXPIRED,
        "admission: a capability past its expiry is refused");

    /* Forged: same shape, wrong MAC. */
    idletoken_admission_mint("job-D", h1, now, t2, sizeof t2, err, sizeof err);
    {
        size_t L = strlen(t2);
        t2[L - 1] = (t2[L - 1] == 'a') ? 'b' : 'a';
        AST(idletoken_admission_consume(t2, h1, now + 1, NULL, 0) == IDLETOKEN_ADM_BAD_MAC,
            "admission: a tampered capability is refused as a bad signature");
    }

    /* Minted by ANOTHER coordinator: re-arming rolls the channel key, so a
     * ticket from before the restart must not verify after it. */
    idletoken_admission_mint("job-E", h1, now, t2, sizeof t2, err, sizeof err);
    idletoken_admission_init(NULL, NULL, err, sizeof err);
    AST(idletoken_admission_consume(t2, h1, now + 1, NULL, 0) == IDLETOKEN_ADM_BAD_MAC,
        "admission: a capability from a previous process does not verify");

    /* Shape rejections, each named so a verifier that has collapsed into
     * refusing everything cannot pass this block. */
    AST(idletoken_admission_consume("", h1, now, NULL, 0) == IDLETOKEN_ADM_ABSENT,
        "admission: an empty capability reads as absent, not as valid");
    AST(idletoken_admission_consume("hello", h1, now, NULL, 0) == IDLETOKEN_ADM_MALFORMED,
        "admission: a non-capability string is refused as malformed");
    AST(idletoken_admission_consume("itadm1.a.b.c.d", h1, now, NULL, 0) == IDLETOKEN_ADM_MALFORMED,
        "admission: a capability with unparsable fields is refused");

    /* A job id carrying the field separator would let two different tuples hash
     * the same; it is refused at mint rather than escaped. */
    AST(idletoken_admission_mint("job.with.dots", h1, now, t2, sizeof t2,
                                 err, sizeof err) != 0,
        "admission: a job id containing the field separator is refused at mint");
    AST(idletoken_admission_mint("", h1, now, t2, sizeof t2, err, sizeof err) != 0,
        "admission: an empty job id is refused at mint");

    /* The channel key gates minting. Both directions, because a comparison that
     * accepts everything and one that accepts nothing are equally broken. The
     * POSITIVE control is the one that matters: `channel_ok` had only its
     * negative asserted, and a function that returns 0 unconditionally would
     * have passed that while locking the agent out of minting for good. */
    {
        char hex[IDLETOKEN_ADM_KEYHEX_CAP] = "";
        AST(idletoken_admission_channel_ok(g_adm.channel_hex) == 1,
            "admission: the channel key matches itself");
        AST(idletoken_admission_channel_ok("0000") == 0,
            "admission: a wrong channel key is refused");
        AST(idletoken_admission_local_marker(hex, sizeof hex) == 0 &&
            idletoken_admission_local_ok(hex) == 1,
            "admission: the local-origin marker matches itself");
        AST(idletoken_admission_local_ok("deadbeef") == 0,
            "admission: a wrong local-origin marker is refused");
    }

    /* --- the two-process shape, which is the one that actually ships --------
     *
     * The agent mints; the coordinator spends. They are different processes, so
     * every assertion above (single process, one global) could hold while the
     * real pair does not — that is the classic shared-assumption oracle this
     * repo has been burned by. Simulated here by snapshotting the channel,
     * re-arming (a "different process"), and attaching the snapshot back. */
    {
        char channel[IDLETOKEN_ADM_KEYHEX_CAP];
        char t3[IDLETOKEN_ADM_TICKET_CAP];
        char foreign[IDLETOKEN_ADM_TICKET_CAP];

        idletoken_admission_init(NULL, NULL, err, sizeof err);
        snprintf(channel, sizeof channel, "%s", g_adm.channel_hex);   /* coordinator's */

        /* A ticket minted under a DIFFERENT channel, kept for the negative
         * below. Minted first, while the wrong key is loaded. */
        AST(idletoken_admission_attach(
                "1111111111111111111111111111111111111111111111111111111111111111",
                err, sizeof err) == 0,
            "admission: attaching a well-formed channel key succeeds");
        idletoken_admission_mint("job-F", h1, now, foreign, sizeof foreign, err, sizeof err);

        /* Now the agent attaches the coordinator's real channel and mints. */
        AST(idletoken_admission_attach(channel, err, sizeof err) == 0,
            "admission: the agent attaches the coordinator's channel");
        AST(idletoken_admission_mint("job-G", h1, now, t3, sizeof t3,
                                     err, sizeof err) == 0,
            "admission: an attached process can mint");

        /* Back on the coordinator side: same channel, so the ticket verifies. */
        AST(idletoken_admission_attach(channel, err, sizeof err) == 0 &&
            idletoken_admission_consume(t3, h1, now + 1, job, sizeof job) == IDLETOKEN_ADM_OK &&
            !strcmp(job, "job-G"),
            "admission: a ticket minted by the agent is spent by the coordinator");
        idletoken_admission_request_end();

        AST(idletoken_admission_consume(foreign, h1, now + 1, NULL, 0) == IDLETOKEN_ADM_BAD_MAC,
            "admission: a ticket minted under another channel is refused");

        AST(idletoken_admission_attach("not-hex", err, sizeof err) == -1 &&
            strstr(err, "hex bytes") != NULL,
            "admission: a malformed channel key is refused, with a reason");
    }

    /* --- the coordinator restarting under a live agent ----------------------
     *
     * The one that was missing, and the one that cost a provider its listing.
     * The agent outlives the coordinator by design (engine.rs::stop_engine), the
     * coordinator rolls its key on every start, and minting is a local HMAC that
     * cannot tell it is using yesterday's key. Every assertion above holds
     * perfectly while that is broken, because every one of them attaches
     * immediately before it mints.
     *
     * So this block does the thing the others do not: it lets a key go stale
     * ON DISK and asks whether the agent notices. The first assertion is the
     * NEGATIVE control — an unchanged file must NOT re-attach, or a function
     * that re-attaches unconditionally would pass the rest of this block while
     * throwing the cache away on every job. */
    {
        char path[400];
        char keyA[IDLETOKEN_ADM_KEYHEX_CAP], keyB[IDLETOKEN_ADM_KEYHEX_CAP];
        char tA[IDLETOKEN_ADM_TICKET_CAP], tB[IDLETOKEN_ADM_TICKET_CAP];
        const char *tmp = getenv("TMPDIR");
        if (!tmp || !tmp[0]) tmp = getenv("TEMP");
        if (!tmp || !tmp[0]) tmp = getenv("TMP");
        if (!tmp || !tmp[0]) tmp = ".";
        snprintf(path, sizeof path, "%s/idletoken-admission-selftest.key", tmp);

        /* Coordinator start #1 — publishes key A. */
        AST(idletoken_admission_init(path, NULL, err, sizeof err) == 0,
            "admission: the coordinator publishes its channel key to a file");
        snprintf(keyA, sizeof keyA, "%s", g_adm.channel_hex);

        /* Agent side: attach from that file, then mint while it is current. */
        AST(idletoken_admission_attach_file(path, err, sizeof err) == 0 &&
            idletoken_admission_channel_ok(keyA) == 1,
            "admission: attaching from the file loads exactly what was published");
        AST(idletoken_admission_attach_file_if_changed(path, err, sizeof err) == 0,
            "admission: an UNCHANGED key file does not re-attach "
            "(negative control: the cache is still a cache)");
        idletoken_admission_mint("job-H", h1, now, tA, sizeof tA, err, sizeof err);

        /* Coordinator start #2 — same path, brand new key. */
        AST(idletoken_admission_init(path, NULL, err, sizeof err) == 0,
            "admission: a restarted coordinator republishes to the same path");
        snprintf(keyB, sizeof keyB, "%s", g_adm.channel_hex);
        AST(strcmp(keyA, keyB) != 0,
            "admission: a restart rolls the channel key");
        AST(idletoken_admission_consume(tA, h1, now + 1, NULL, 0) == IDLETOKEN_ADM_BAD_MAC,
            "admission: the ticket minted before the restart no longer verifies "
            "(this is the failure a cached attachment produces forever)");

        /* Back on the agent, still holding the pre-restart key. */
        AST(idletoken_admission_attach(keyA, err, sizeof err) == 0,
            "admission: (the agent is put back on the stale key)");
        AST(idletoken_admission_attach_file_if_changed(path, err, sizeof err) == 1 &&
            idletoken_admission_channel_ok(keyB) == 1,
            "admission: a ROLLED key file re-attaches, to the new key");
        AST(idletoken_admission_mint("job-I", h1, now, tB, sizeof tB,
                                     err, sizeof err) == 0,
            "admission: the re-attached agent can mint again");

        /* And the restarted coordinator accepts it. Minting is a local HMAC, so
         * without this last step the block would prove only that the agent
         * changed its mind, not that the two processes agree again. */
        AST(idletoken_admission_attach(keyB, err, sizeof err) == 0 &&
            idletoken_admission_consume(tB, h1, now + 1, job, sizeof job) == IDLETOKEN_ADM_OK &&
            !strcmp(job, "job-I"),
            "admission: the restarted coordinator spends the re-minted ticket");
        idletoken_admission_request_end();

        AST(idletoken_admission_attach_file_if_changed(
                "no/such/idletoken-admission.key", err, sizeof err) == -1 && err[0],
            "admission: a missing key file is a named failure, not a silent success");

        remove(path);
    }
#undef AST
    return fails;
}
