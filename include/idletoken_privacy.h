/* IdleToken — privacy / token-encryption pipeline (G-PRIV).
 *
 * Implements docs/privacy-design.md: keep the user's prompt in an *untrusted*
 * cluster out of plain sight everywhere except the one in-memory instant the
 * Coordinator is actively running the forward pass. Everything else — the
 * wire, logs, disk, and every Worker — sees ciphertext or opaque activations
 * only. The goal is "raise the cost / stop casual leakage", NOT full defence
 * (a determined insider dumping live Coordinator RAM is explicitly out of
 * scope; so are learning-based inversion and metadata side channels).
 *
 * ---------------------------------------------------------------------------
 * Interface-first (docs/architecture.md §1.4). The pipeline is five pluggable layers,
 * each a small vtable so a real impl can be swapped without touching callers:
 *
 *   SealedTransport : envelope seal/open   (v0.1: X25519 sealed box)   <- "token encryption"
 *                                          (future: HPKE / threshold k-of-n)
 *   PayloadCipher   : authenticated symmetric cipher (v0.1: XSalsa20-Poly1305)
 *                                          (future: AES-256-GCM via libsodium)
 *   Obfuscator      : metadata / DP hook   (v0.1: NoOp)                <- where DP lands
 *                                          (future: padding / dummy tokens / noise)
 *   PlaintextGuard  : no-log + zeroize + mlock + encrypt-at-rest discipline
 *   Boundary        : coord decrypts+embeds; workers only see hidden states
 *
 * The hard limit (design §2): hidden states that a transformer must *compute*
 * on cannot be encrypted — only obfuscated. So Obfuscator is the only privacy
 * lever on the activation path; SealedTransport/PayloadCipher protect the
 * request/response payloads and anything that touches disk.
 *
 * C only. No C++. Matches vendor/ds4 + include/ style.
 */

#ifndef IDLETOKEN_PRIVACY_H
#define IDLETOKEN_PRIVACY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- sizes (match the vendored X25519 + XSalsa20-Poly1305 primitives) ----- */
#define IDLETOKEN_PK_BYTES      32   /* X25519 public key                        */
#define IDLETOKEN_SK_BYTES      32   /* X25519 secret key                        */
#define IDLETOKEN_SYMKEY_BYTES  32   /* symmetric key (PayloadCipher)            */
#define IDLETOKEN_NONCE_BYTES   24   /* XSalsa20 nonce                           */
#define IDLETOKEN_MAC_BYTES     16   /* Poly1305 tag                             */

/* A sealed box adds an ephemeral public key + the MAC to the plaintext. */
#define IDLETOKEN_SEAL_OVERHEAD (IDLETOKEN_PK_BYTES + IDLETOKEN_MAC_BYTES)   /* 48 */
/* crypto_secretbox output adds only the MAC. */
#define IDLETOKEN_CIPHER_OVERHEAD (IDLETOKEN_MAC_BYTES)                   /* 16 */

typedef enum {
    IDLETOKEN_PRIV_OK             = 0,
    IDLETOKEN_PRIV_EAUTH          = -1,  /* MAC failed: forged/corrupt/wrong key   */
    IDLETOKEN_PRIV_EBUF           = -2,  /* output buffer too small                */
    IDLETOKEN_PRIV_EINVAL         = -3,  /* bad argument                           */
    IDLETOKEN_PRIV_ERANDOM        = -4,  /* RNG failure                            */
    IDLETOKEN_PRIV_ENOTIMPL       = -5   /* layer not implemented in this build    */
} idletoken_priv_rc;

/* An X25519 keypair. The secret key is sensitive: mlock it, zeroize on drop. */
typedef struct {
    uint8_t pk[IDLETOKEN_PK_BYTES];
    uint8_t sk[IDLETOKEN_SK_BYTES];
} idletoken_keypair;

/* =========================================================================
 * PlaintextGuard -- the "an ordinary person cannot see it" discipline (design
 * §3.4). Not crypto;
 * memory hygiene. Callers wrap every transient plaintext buffer with these.
 * ========================================================================= */

/* Constant-time-ish secure wipe that the compiler may not elide (unlike a
 * plain memset the optimizer can drop). Use on every plaintext/key buffer as
 * soon as it is no longer needed. */
void idletoken_secure_zero(void *p, size_t n);

/* Constant-time equality (for comparing tags / tokens without a timing leak).
 * Returns 1 if equal, 0 otherwise. */
int idletoken_ct_eq(const void *a, const void *b, size_t n);

/* Pin `n` bytes at `p` so they never hit swap/hibernation (design §3.4.3).
 * Best-effort: returns 0 on success, -1 if the OS refused (logged by caller,
 * not fatal). Pair with idletoken_munlock. */
int idletoken_mlock(void *p, size_t n);
int idletoken_munlock(void *p, size_t n);

/* Harden the current process against external inspection (Tier-1 of the
 * "raise the cost, stop the merely curious" goal, design §3.4). Best-effort, applied once at
 * startup by the plaintext-window owner (the decrypting node):
 *   - Linux: prctl(PR_SET_DUMPABLE, 0) — blocks ptrace-attach by a non-root
 *     process and suppresses core dumps, so a curious operator can't gdb-attach
 *     the running engine or read a crash dump to fish out the plaintext window.
 *   - POSIX: setrlimit(RLIMIT_CORE, 0) — no core file carrying plaintext ever
 *     hits disk.
 *   - Windows: SetErrorMode suppresses the WER crash dump/dialog (limited).
 *
 * HONEST SCOPE (design §7): this raises the cost against co-tenants, other
 * processes, crash dumps, swap, and casual snooping. It does NOT — and cannot
 * — stop the machine's own root/owner, who can re-enable ptrace or snapshot
 * RAM. Software on a box can't hide from that box's root; that threat is out
 * of scope by design. The aim is only to make the plaintext instant expensive
 * to catch, not impossible.
 *
 * Returns IDLETOKEN_PRIV_OK if the applicable hardening was applied. */
idletoken_priv_rc idletoken_harden_process(void);

/* Introspection used by the selftest oracle to prove hardening took effect. */
int  idletoken_process_dumpable(void);  /* 1=dumpable, 0=protected, -1=unknown/NA */
long idletoken_core_limit(void);        /* RLIMIT_CORE soft limit; 0=off, -1=NA   */

/* =========================================================================
 * PayloadCipher — authenticated symmetric encryption (design §3.3).
 * Role of AES-256-GCM; v0.1 impl is XSalsa20-Poly1305 (crypto_secretbox).
 * Used for: session-keyed response encryption, and encrypt-at-rest of any
 * spilled KV cache / snapshot / cache file (design §3.4.4).
 * ========================================================================= */

/* Encrypt `plain[plen]` under 32-byte `key` with 24-byte `nonce`.
 * Writes plen + IDLETOKEN_CIPHER_OVERHEAD bytes to `out`; sets *out_len.
 * Nonce MUST be unique per key (caller supplies a fresh random one and sends
 * it alongside the ciphertext — the nonce is not secret). */
idletoken_priv_rc idletoken_cipher_encrypt(const uint8_t key[IDLETOKEN_SYMKEY_BYTES],
                                     const uint8_t nonce[IDLETOKEN_NONCE_BYTES],
                                     const uint8_t *plain, size_t plen,
                                     uint8_t *out, size_t out_cap, size_t *out_len);

/* Inverse. On MAC failure returns IDLETOKEN_PRIV_EAUTH and writes nothing. */
idletoken_priv_rc idletoken_cipher_decrypt(const uint8_t key[IDLETOKEN_SYMKEY_BYTES],
                                     const uint8_t nonce[IDLETOKEN_NONCE_BYTES],
                                     const uint8_t *cipher, size_t clen,
                                     uint8_t *out, size_t out_cap, size_t *out_len);

/* Fill `nonce` with 24 fresh random bytes. */
idletoken_priv_rc idletoken_random_nonce(uint8_t nonce[IDLETOKEN_NONCE_BYTES]);

/* Fill `buf[n]` with cryptographically secure random bytes (CSPRNG). Aborts
 * the process on RNG failure rather than proceeding with weak entropy. */
void idletoken_random_bytes(uint8_t *buf, size_t n);

/* =========================================================================
 * SealedTransport — envelope encryption to a recipient's public key
 * (design §3.3, libsodium crypto_box_seal shape). The client holds ONLY the
 * Coordinator's public key; it needs no pre-shared secret. Anonymous sender:
 * a fresh ephemeral keypair per message gives the envelope forward secrecy
 * against later theft of that ephemeral key.
 *
 * Construction: eph = keypair(); nonce = SHA512(eph.pk || recip_pk)[:24];
 *   out = eph.pk || crypto_box(plain, nonce, recip_pk, eph.sk).
 * The recipient recomputes the nonce and opens with its secret key.
 * ========================================================================= */

/* Generate a fresh X25519 keypair (Coordinator identity key, or a per-session
 * client key). Caller should mlock kp->sk. */
idletoken_priv_rc idletoken_keypair_generate(idletoken_keypair *kp);

/* Derive the public key from an existing secret key (for reloading a pinned
 * Coordinator identity from disk-at-rest). */
idletoken_priv_rc idletoken_keypair_from_sk(const uint8_t sk[IDLETOKEN_SK_BYTES],
                                      idletoken_keypair *kp);

/* Seal `plain[plen]` to `recip_pk`. Writes plen + IDLETOKEN_SEAL_OVERHEAD bytes. */
idletoken_priv_rc idletoken_seal(const uint8_t recip_pk[IDLETOKEN_PK_BYTES],
                           const uint8_t *plain, size_t plen,
                           uint8_t *out, size_t out_cap, size_t *out_len);

/* Open a sealed box with the recipient's keypair. IDLETOKEN_PRIV_EAUTH if the
 * box was not sealed to this key (i.e. wrong/forged key ⇒ request rejected,
 * which is G-PRIV oracle item 4). */
idletoken_priv_rc idletoken_seal_open(const idletoken_keypair *recip,
                                const uint8_t *sealed, size_t slen,
                                uint8_t *out, size_t out_cap, size_t *out_len);

/* =========================================================================
 * Obfuscator — the differential-privacy / metadata-defence landing spot
 * (design §4). v0.1 is NoOp (identity). The interface takes the boundary
 * hidden-state tensor so a future impl can pad length / add calibrated noise
 * / inject dummy tokens WITHOUT changing any caller. Kept as a vtable so the
 * real DP impl is a drop-in.
 * ========================================================================= */

typedef struct idletoken_obfuscator idletoken_obfuscator;
struct idletoken_obfuscator {
    const char *name;
    /* Transform `n` float32 hidden-state elements in place. Returns 0 on ok.
     * NoOp leaves them untouched. */
    int (*apply)(idletoken_obfuscator *self, float *hidden, size_t n);
    void *state;
};

/* The v0.1 default: identity. Safe to share; holds no state. */
idletoken_obfuscator *idletoken_obfuscator_noop(void);

/* =========================================================================
 * Boundary — architectural guarantee that Workers hold no secret material
 * (design §3.4.5). This is enforced by construction (the Coordinator never
 * ships keys/vocab/text to Workers, only hidden states). The helper below is
 * a runtime self-check used by the G-PRIV oracle: a Worker-side privacy view
 * must report zero keys, zero vocab, zero plaintext.
 * ========================================================================= */

typedef struct {
    int has_decrypt_key;   /* must be 0 on a Worker */
    int has_vocab;         /* must be 0 on a Worker */
    int has_plaintext;     /* must be 0 on a Worker */
} idletoken_worker_privacy_view;

/* Returns IDLETOKEN_PRIV_OK iff the Worker view is clean (all zero). */
idletoken_priv_rc idletoken_boundary_assert_worker_clean(const idletoken_worker_privacy_view *v);

/* --- build/impl identity (for honest reporting in logs & selftest) -------- */
const char *idletoken_privacy_backend(void);   /* e.g. "tweetnacl-x25519-xsalsa20poly1305" */

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_PRIVACY_H */
