/* IdleToken privacy pipeline — real implementation (see include/idletoken_privacy.h
 * and docs/privacy-design.md). Crypto primitives come from the vendored,
 * self-contained TweetNaCl (vendor/tweetnacl): X25519 + XSalsa20-Poly1305 +
 * SHA-512. No external dependency, builds on Win/Linux/macOS with plain cc.
 *
 * C only. No C++.
 */

#include "idletoken_privacy.h"
#include "tweetnacl.h"

#include <string.h>
#include <stdlib.h>   /* malloc/free/abort */
#include <errno.h>

/* crypto_secretbox / crypto_box use the NaCl zero-padding convention:
 *   ZEROBYTES(32) leading zeros on the plaintext, BOXZEROBYTES(16) leading
 *   zeros on the ciphertext. We hide that behind the compact wrappers below
 *   so callers deal in tight buffers. */
#define ZB   crypto_secretbox_ZEROBYTES      /* 32 */
#define BZB  crypto_secretbox_BOXZEROBYTES   /* 16 */

/* ==========================================================================
 * randombytes — required by TweetNaCl (crypto_box_keypair etc.) and by our
 * nonce/keygen. Must be a CSPRNG. Abort the process on failure: silently
 * proceeding with weak/zero randomness would be a catastrophic privacy bug.
 * ========================================================================== */
#if defined(_WIN32)
  #include <windows.h>
  #include <bcrypt.h>
  void randombytes(unsigned char *p, unsigned long long n) {
      if (BCryptGenRandom(NULL, p, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
          /* No safe fallback; refuse to run with bad entropy. */
          abort();
      }
  }
#else
  #include <unistd.h>
  #include <fcntl.h>
  #if defined(__linux__)
    #include <sys/random.h>
  #endif
  void randombytes(unsigned char *p, unsigned long long n) {
  #if defined(__linux__)
      size_t off = 0;
      while (off < n) {
          ssize_t r = getrandom(p + off, (size_t)n - off, 0);
          if (r > 0) { off += (size_t)r; continue; }
          if (r < 0 && (errno == EINTR)) continue;
          break;  /* fall through to /dev/urandom */
      }
      if (off == n) return;
  #endif
      int fd = open("/dev/urandom", O_RDONLY);
      if (fd < 0) abort();
      size_t off2 = 0;
      while (off2 < n) {
          ssize_t r = read(fd, p + off2, (size_t)n - off2);
          if (r > 0) { off2 += (size_t)r; continue; }
          if (r < 0 && errno == EINTR) continue;
          close(fd); abort();
      }
      close(fd);
  }
#endif

/* ==========================================================================
 * PlaintextGuard
 * ========================================================================== */

/* Volatile function pointer to memset so the optimizer cannot elide the wipe
 * (a plain memset on a soon-to-be-freed buffer is a classic dead-store the
 * compiler removes — exactly the bug we must avoid for key material). */
static void *(*const volatile idletoken_memset_v)(void *, int, size_t) = memset;

void idletoken_secure_zero(void *p, size_t n) {
    if (p && n) idletoken_memset_v(p, 0, n);
}

int idletoken_ct_eq(const void *a, const void *b, size_t n) {
    const volatile unsigned char *pa = (const volatile unsigned char *)a;
    const volatile unsigned char *pb = (const volatile unsigned char *)b;
    unsigned char diff = 0;
    for (size_t i = 0; i < n; i++) diff |= (unsigned char)(pa[i] ^ pb[i]);
    return diff == 0 ? 1 : 0;
}

#if defined(_WIN32)
int idletoken_mlock(void *p, size_t n)   { return VirtualLock(p, n) ? 0 : -1; }
int idletoken_munlock(void *p, size_t n) { return VirtualUnlock(p, n) ? 0 : -1; }
#else
  #include <sys/mman.h>
  #include <sys/resource.h>
  #if defined(__linux__)
    #include <sys/prctl.h>
  #endif
int idletoken_mlock(void *p, size_t n)   { return mlock(p, n); }
int idletoken_munlock(void *p, size_t n) { return munlock(p, n); }
#endif

/* Tier-1 process hardening — raise the cost of catching the plaintext window.
 * Best-effort; honest scope in the header (does NOT stop the machine's root). */
idletoken_priv_rc idletoken_harden_process(void) {
#if defined(_WIN32)
    /* No ptrace on Windows; crash dumps flow through WER. Suppress the crash
     * UI/dump for this process. Full anti-dump needs more — Tier-1 is limited
     * on Windows. */
    SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
    return IDLETOKEN_PRIV_OK;
#else
    int ok = 1;
    struct rlimit rl;
    rl.rlim_cur = 0;
    rl.rlim_max = 0;
    if (setrlimit(RLIMIT_CORE, &rl) != 0) ok = 0;   /* no core dumps */
  #if defined(__linux__)
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) ok = 0;  /* block non-root ptrace */
  #endif
    return ok ? IDLETOKEN_PRIV_OK : IDLETOKEN_PRIV_EINVAL;
#endif
}

int idletoken_process_dumpable(void) {
#if defined(__linux__)
    int r = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
    return r < 0 ? -1 : (r ? 1 : 0);
#else
    return -1;   /* not queryable / not applicable on this platform */
#endif
}

long idletoken_core_limit(void) {
#if defined(_WIN32)
    return -1;
#else
    struct rlimit rl;
    if (getrlimit(RLIMIT_CORE, &rl) != 0) return -1;
    return (long)rl.rlim_cur;
#endif
}

/* ==========================================================================
 * PayloadCipher — XSalsa20-Poly1305 (crypto_secretbox), compact I/O.
 * ========================================================================== */

void idletoken_random_bytes(uint8_t *buf, size_t n) {
    if (buf && n) randombytes(buf, n);
}

idletoken_priv_rc idletoken_random_nonce(uint8_t nonce[IDLETOKEN_NONCE_BYTES]) {
    if (!nonce) return IDLETOKEN_PRIV_EINVAL;
    randombytes(nonce, IDLETOKEN_NONCE_BYTES);
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_cipher_encrypt(const uint8_t key[IDLETOKEN_SYMKEY_BYTES],
                                     const uint8_t nonce[IDLETOKEN_NONCE_BYTES],
                                     const uint8_t *plain, size_t plen,
                                     uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!key || !nonce || (!plain && plen) || !out) return IDLETOKEN_PRIV_EINVAL;
    if (out_cap < plen + IDLETOKEN_CIPHER_OVERHEAD) return IDLETOKEN_PRIV_EBUF;

    size_t mlen = ZB + plen;
    unsigned char *m = (unsigned char *)malloc(mlen);
    unsigned char *c = (unsigned char *)malloc(mlen);
    if (!m || !c) { free(m); free(c); return IDLETOKEN_PRIV_EINVAL; }
    memset(m, 0, ZB);
    if (plen) memcpy(m + ZB, plain, plen);

    int rc = crypto_secretbox(c, m, mlen, nonce, key);
    if (rc != 0) { idletoken_secure_zero(m, mlen); free(m); free(c); return IDLETOKEN_PRIV_EINVAL; }

    /* compact ciphertext = c[BZB .. mlen), length plen + MAC */
    memcpy(out, c + BZB, plen + IDLETOKEN_CIPHER_OVERHEAD);
    if (out_len) *out_len = plen + IDLETOKEN_CIPHER_OVERHEAD;

    idletoken_secure_zero(m, mlen);
    free(m); free(c);
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_cipher_decrypt(const uint8_t key[IDLETOKEN_SYMKEY_BYTES],
                                     const uint8_t nonce[IDLETOKEN_NONCE_BYTES],
                                     const uint8_t *cipher, size_t clen,
                                     uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!key || !nonce || !cipher || !out) return IDLETOKEN_PRIV_EINVAL;
    if (clen < IDLETOKEN_CIPHER_OVERHEAD) return IDLETOKEN_PRIV_EAUTH;
    size_t plen = clen - IDLETOKEN_CIPHER_OVERHEAD;
    if (out_cap < plen) return IDLETOKEN_PRIV_EBUF;

    size_t buflen = BZB + clen;   /* = ZB + plen */
    unsigned char *c = (unsigned char *)malloc(buflen);
    unsigned char *m = (unsigned char *)malloc(buflen);
    if (!c || !m) { free(c); free(m); return IDLETOKEN_PRIV_EINVAL; }
    memset(c, 0, BZB);
    memcpy(c + BZB, cipher, clen);

    int rc = crypto_secretbox_open(m, c, buflen, nonce, key);
    if (rc != 0) { free(c); idletoken_secure_zero(m, buflen); free(m); return IDLETOKEN_PRIV_EAUTH; }

    if (plen) memcpy(out, m + ZB, plen);
    if (out_len) *out_len = plen;

    idletoken_secure_zero(m, buflen);
    free(c); free(m);
    return IDLETOKEN_PRIV_OK;
}

/* ==========================================================================
 * SealedTransport — X25519 sealed box (libsodium crypto_box_seal shape).
 * ========================================================================== */

idletoken_priv_rc idletoken_keypair_generate(idletoken_keypair *kp) {
    if (!kp) return IDLETOKEN_PRIV_EINVAL;
    crypto_box_keypair(kp->pk, kp->sk);
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_keypair_from_sk(const uint8_t sk[IDLETOKEN_SK_BYTES],
                                      idletoken_keypair *kp) {
    if (!sk || !kp) return IDLETOKEN_PRIV_EINVAL;
    memcpy(kp->sk, sk, IDLETOKEN_SK_BYTES);
    crypto_scalarmult_base(kp->pk, kp->sk);   /* pk = sk * basepoint */
    return IDLETOKEN_PRIV_OK;
}

/* nonce = SHA512(eph_pk || recip_pk)[:24] — self-consistent sealed-box nonce
 * (libsodium uses blake2b; we use TweetNaCl's SHA-512, documented deviation). */
static void seal_nonce(const uint8_t eph_pk[IDLETOKEN_PK_BYTES],
                       const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                       uint8_t nonce[IDLETOKEN_NONCE_BYTES]) {
    unsigned char cat[2 * IDLETOKEN_PK_BYTES];
    unsigned char h[64];
    memcpy(cat, eph_pk, IDLETOKEN_PK_BYTES);
    memcpy(cat + IDLETOKEN_PK_BYTES, recip_pk, IDLETOKEN_PK_BYTES);
    crypto_hash(h, cat, sizeof(cat));   /* SHA-512 */
    memcpy(nonce, h, IDLETOKEN_NONCE_BYTES);
    idletoken_secure_zero(h, sizeof(h));
}

idletoken_priv_rc idletoken_seal(const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                           const uint8_t *plain, size_t plen,
                           uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!recip_pk || (!plain && plen) || !out) return IDLETOKEN_PRIV_EINVAL;
    if (out_cap < plen + IDLETOKEN_SEAL_OVERHEAD) return IDLETOKEN_PRIV_EBUF;

    idletoken_keypair eph;
    crypto_box_keypair(eph.pk, eph.sk);

    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    seal_nonce(eph.pk, recip_pk, nonce);

    size_t mlen = ZB + plen;
    unsigned char *m = (unsigned char *)malloc(mlen);
    unsigned char *c = (unsigned char *)malloc(mlen);
    if (!m || !c) { free(m); free(c); idletoken_secure_zero(eph.sk, sizeof(eph.sk)); return IDLETOKEN_PRIV_EINVAL; }
    memset(m, 0, ZB);
    if (plen) memcpy(m + ZB, plain, plen);

    int rc = crypto_box(c, m, mlen, nonce, recip_pk, eph.sk);
    idletoken_secure_zero(eph.sk, sizeof(eph.sk));   /* ephemeral sk dies now */
    if (rc != 0) { idletoken_secure_zero(m, mlen); free(m); free(c); return IDLETOKEN_PRIV_EINVAL; }

    /* out = eph.pk || compact_ciphertext */
    memcpy(out, eph.pk, IDLETOKEN_PK_BYTES);
    memcpy(out + IDLETOKEN_PK_BYTES, c + BZB, plen + IDLETOKEN_MAC_BYTES);
    if (out_len) *out_len = plen + IDLETOKEN_SEAL_OVERHEAD;

    idletoken_secure_zero(m, mlen);
    free(m); free(c);
    return IDLETOKEN_PRIV_OK;
}

idletoken_priv_rc idletoken_seal_open(const idletoken_keypair *recip,
                                const uint8_t *sealed, size_t slen,
                                uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!recip || !sealed || !out) return IDLETOKEN_PRIV_EINVAL;
    if (slen < IDLETOKEN_SEAL_OVERHEAD) return IDLETOKEN_PRIV_EAUTH;

    const uint8_t *eph_pk = sealed;
    const uint8_t *ct     = sealed + IDLETOKEN_PK_BYTES;
    size_t ctlen = slen - IDLETOKEN_PK_BYTES;          /* = plen + MAC */
    size_t plen  = ctlen - IDLETOKEN_MAC_BYTES;
    if (out_cap < plen) return IDLETOKEN_PRIV_EBUF;

    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    seal_nonce(eph_pk, recip->pk, nonce);

    size_t buflen = BZB + ctlen;   /* = ZB + plen */
    unsigned char *c = (unsigned char *)malloc(buflen);
    unsigned char *m = (unsigned char *)malloc(buflen);
    if (!c || !m) { free(c); free(m); return IDLETOKEN_PRIV_EINVAL; }
    memset(c, 0, BZB);
    memcpy(c + BZB, ct, ctlen);

    int rc = crypto_box_open(m, c, buflen, nonce, eph_pk, recip->sk);
    if (rc != 0) { free(c); idletoken_secure_zero(m, buflen); free(m); return IDLETOKEN_PRIV_EAUTH; }

    if (plen) memcpy(out, m + ZB, plen);
    if (out_len) *out_len = plen;

    idletoken_secure_zero(m, buflen);
    free(c); free(m);
    return IDLETOKEN_PRIV_OK;
}

/* ==========================================================================
 * Obfuscator — NoOp default (v0.1). The DP landing spot (design §4).
 * ========================================================================== */

static int obf_noop_apply(idletoken_obfuscator *self, float *hidden, size_t n) {
    (void)self; (void)hidden; (void)n;   /* identity: hidden states untouched */
    return 0;
}

static idletoken_obfuscator g_obf_noop = { "noop", obf_noop_apply, NULL };

idletoken_obfuscator *idletoken_obfuscator_noop(void) { return &g_obf_noop; }

/* ==========================================================================
 * Boundary — Worker-side no-secret assertion (design §3.4.5).
 * ========================================================================== */

idletoken_priv_rc idletoken_boundary_assert_worker_clean(const idletoken_worker_privacy_view *v) {
    if (!v) return IDLETOKEN_PRIV_EINVAL;
    if (v->has_decrypt_key || v->has_vocab || v->has_plaintext)
        return IDLETOKEN_PRIV_EINVAL;
    return IDLETOKEN_PRIV_OK;
}

const char *idletoken_privacy_backend(void) {
    return "tweetnacl-x25519-xsalsa20poly1305-sha512";
}
