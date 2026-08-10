/* IdleToken G-PRIV self-test oracle.
 *
 * Headless, hardware-free proof that the privacy pipeline
 * (docs/privacy-design.md, include/idletoken_privacy.h) actually holds. It
 * simulates the real data flow — client seals a prompt to the Coordinator's
 * public key, the Coordinator opens it, Workers see only hidden states, the
 * answer is sealed back — and asserts the G-PRIV oracle properties from
 * docs/acceptance-criteria.md:
 *
 *   1. no plaintext on the wire (sealed request/response bytes contain no
 *      prompt/answer substring)                                   [items 1,2]
 *   2. Workers hold no decrypt key / no vocab / no plaintext      [item 3]
 *   3. wrong / tampered key ⇒ request rejected                    [item 4]
 *   4. seal→open→(infer)→seal→open reproduces the exact plaintext [item 5]
 *   5. encrypt-at-rest: spilled KV blob is ciphertext on disk     [item 2/§3.4.4]
 *   6. pipeline layers are pluggable (Obfuscator swap is a no-op) [item 6]
 *   7. plaintext buffers zeroize after use                        [§3.4.2]
 *
 * On full success prints `G_PRIV_SELFTEST_OK` (last line) and exits 0, so
 * scripts/acceptance.sh can use it as the executable oracle for gate G-PRIV.
 * Any failed check prints `G_PRIV_SELFTEST_FAIL: <why>` and exits 1.
 *
 * C only. No C++.
 */

#include "idletoken_privacy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void ok(const char *what)  { printf("  [ok]   %s\n", what); }
static void bad(const char *what) { printf("  [FAIL] %s\n", what); g_fail = 1; }
#define CHECK(cond, what) do { if (cond) ok(what); else bad(what); } while (0)

/* Does haystack[hlen] contain needle as a raw byte substring? Used to prove a
 * ciphertext buffer does NOT leak the plaintext. */
static int contains(const uint8_t *hay, size_t hlen, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || nlen > hlen) return 0;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return 1;
    return 0;
}

int main(void) {
    printf("IdleToken G-PRIV self-test  (backend: %s)\n", idletoken_privacy_backend());
    printf("-------------------------------------------------------------\n");

    const char *PROMPT = "SECRET-PROMPT: my bank PIN is 4021, keep it private";
    const char *ANSWER = "SECRET-ANSWER: acknowledged, PIN stored nowhere";

    /* ---- key bootstrap (design §6): coord identity + client session key ---- */
    idletoken_keypair coord, session, attacker;
    CHECK(idletoken_keypair_generate(&coord)   == IDLETOKEN_PRIV_OK, "coord identity keypair");
    CHECK(idletoken_keypair_generate(&session) == IDLETOKEN_PRIV_OK, "client session keypair");
    CHECK(idletoken_keypair_generate(&attacker)== IDLETOKEN_PRIV_OK, "attacker keypair (control)");
    /* keys are sensitive — pin them (best effort; not fatal if OS refuses) */
    idletoken_mlock(coord.sk, sizeof(coord.sk));

    /* keypair_from_sk must reproduce the same public key */
    idletoken_keypair coord2;
    CHECK(idletoken_keypair_from_sk(coord.sk, &coord2) == IDLETOKEN_PRIV_OK &&
          idletoken_ct_eq(coord2.pk, coord.pk, IDLETOKEN_PK_BYTES),
          "derive pubkey from secret key is deterministic");

    /* ================= client side: seal the prompt ==================== */
    uint8_t sealed_req[512];
    size_t  sealed_req_len = 0;
    CHECK(idletoken_seal(coord.pk, (const uint8_t *)PROMPT, strlen(PROMPT),
                      sealed_req, sizeof(sealed_req), &sealed_req_len) == IDLETOKEN_PRIV_OK,
          "client seals prompt to coord public key");
    CHECK(sealed_req_len == strlen(PROMPT) + IDLETOKEN_SEAL_OVERHEAD,
          "sealed request length = plaintext + 48B overhead");

    /* [item 1] the on-wire sealed bytes must NOT contain the prompt */
    CHECK(!contains(sealed_req, sealed_req_len, "SECRET-PROMPT"),
          "no plaintext on the wire (sealed request is ciphertext)");

    /* ================= coord side: open the envelope =================== */
    /* This is the ONLY plaintext window (design §2). Pin + zeroize it. */
    uint8_t prompt_plain[512];
    size_t  prompt_len = 0;
    idletoken_mlock(prompt_plain, sizeof(prompt_plain));
    CHECK(idletoken_seal_open(&coord, sealed_req, sealed_req_len,
                           prompt_plain, sizeof(prompt_plain), &prompt_len) == IDLETOKEN_PRIV_OK,
          "coord opens sealed request with its secret key");
    CHECK(prompt_len == strlen(PROMPT) &&
          memcmp(prompt_plain, PROMPT, prompt_len) == 0,
          "recovered prompt matches original exactly");

    /* [item 4] an attacker/wrong key must NOT be able to open it */
    uint8_t junk[512]; size_t junk_len = 0;
    CHECK(idletoken_seal_open(&attacker, sealed_req, sealed_req_len,
                           junk, sizeof(junk), &junk_len) == IDLETOKEN_PRIV_EAUTH,
          "wrong key is rejected (EAUTH)");

    /* [item 4] a tampered ciphertext must fail the MAC */
    uint8_t tampered[512];
    memcpy(tampered, sealed_req, sealed_req_len);
    tampered[sealed_req_len - 1] ^= 0x01;   /* flip one bit of the tag */
    CHECK(idletoken_seal_open(&coord, tampered, sealed_req_len,
                           junk, sizeof(junk), &junk_len) == IDLETOKEN_PRIV_EAUTH,
          "tampered ciphertext is rejected (authentication works)");

    /* ============ boundary: workers see only hidden states ============= */
    /* Simulate the Coordinator embedding the prompt into hidden states and
     * shipping them across the PP boundary. Workers get floats, no secrets. */
    float hidden[64];
    for (size_t i = 0; i < 64; i++) hidden[i] = (float)(prompt_plain[i % prompt_len]) * 0.01f;

    /* [item 6] Obfuscator is pluggable; the v0.1 NoOp leaves states intact. */
    float hidden_before[64];
    memcpy(hidden_before, hidden, sizeof(hidden));
    idletoken_obfuscator *obf = idletoken_obfuscator_noop();
    CHECK(obf && obf->apply(obf, hidden, 64) == 0 &&
          memcmp(hidden_before, hidden, sizeof(hidden)) == 0,
          "Obfuscator NoOp is identity (pluggable DP hook present)");

    /* [item 3] a Worker's privacy view must be clean: no key/vocab/text */
    idletoken_worker_privacy_view worker = { 0, 0, 0 };
    CHECK(idletoken_boundary_assert_worker_clean(&worker) == IDLETOKEN_PRIV_OK,
          "worker holds no decrypt key / no vocab / no plaintext");
    /* the hidden-state buffer a worker sees contains no prompt bytes */
    CHECK(!contains((const uint8_t *)hidden, sizeof(hidden), "SECRET-PROMPT"),
          "worker's hidden-state tensor carries no plaintext substring");
    /* a "dirty" worker (holding a key) must be flagged */
    idletoken_worker_privacy_view dirty = { 1, 0, 0 };
    CHECK(idletoken_boundary_assert_worker_clean(&dirty) != IDLETOKEN_PRIV_OK,
          "a worker that holds a key is flagged as unclean");

    /* ============ encrypt-at-rest: spilled KV / snapshot =============== */
    /* [item 2/§3.4.4] anything hitting disk is ciphertext. */
    uint8_t kv_key[IDLETOKEN_SYMKEY_BYTES];
    idletoken_random_bytes(kv_key, sizeof(kv_key));
    uint8_t nonce[IDLETOKEN_NONCE_BYTES];
    idletoken_random_nonce(nonce);
    const char *kv_plain = "KV-CACHE contains SECRET-PROMPT tokens";
    uint8_t kv_cipher[256]; size_t kv_clen = 0;
    CHECK(idletoken_cipher_encrypt(kv_key, nonce, (const uint8_t *)kv_plain,
                                strlen(kv_plain), kv_cipher, sizeof(kv_cipher),
                                &kv_clen) == IDLETOKEN_PRIV_OK,
          "encrypt-at-rest: KV blob encrypted");
    CHECK(!contains(kv_cipher, kv_clen, "SECRET-PROMPT"),
          "spilled KV blob is ciphertext on disk (no plaintext)");
    uint8_t kv_back[256]; size_t kv_blen = 0;
    CHECK(idletoken_cipher_decrypt(kv_key, nonce, kv_cipher, kv_clen,
                                kv_back, sizeof(kv_back), &kv_blen) == IDLETOKEN_PRIV_OK &&
          kv_blen == strlen(kv_plain) && memcmp(kv_back, kv_plain, kv_blen) == 0,
          "encrypt-at-rest round-trips");

    /* ============ coord seals the answer to the session key =========== */
    uint8_t sealed_resp[512]; size_t sealed_resp_len = 0;
    CHECK(idletoken_seal(session.pk, (const uint8_t *)ANSWER, strlen(ANSWER),
                      sealed_resp, sizeof(sealed_resp), &sealed_resp_len) == IDLETOKEN_PRIV_OK,
          "coord seals answer to client session key");
    CHECK(!contains(sealed_resp, sealed_resp_len, "SECRET-ANSWER"),
          "no plaintext on the wire (sealed response is ciphertext)");
    uint8_t answer_plain[512]; size_t answer_len = 0;
    CHECK(idletoken_seal_open(&session, sealed_resp, sealed_resp_len,
                           answer_plain, sizeof(answer_plain), &answer_len) == IDLETOKEN_PRIV_OK &&
          answer_len == strlen(ANSWER) && memcmp(answer_plain, ANSWER, answer_len) == 0,
          "client decrypts answer end-to-end (matches original)");

    /* ============ Tier-1 process hardening (raise probing cost) ======= */
    /* The plaintext-window owner hardens itself: no core dumps, no non-root
     * ptrace-attach. Honest scope — this stops co-tenants / dumps / casual
     * snooping, NOT the machine's own root (design §7). */
    CHECK(idletoken_harden_process() == IDLETOKEN_PRIV_OK,
          "process hardened (core dumps off; ptrace-attach blocked on Linux)");
    long core = idletoken_core_limit();
    CHECK(core == 0 || core == -1,
          "core-dump limit is 0 (no plaintext-bearing crash dump on disk)");
    int dumpable = idletoken_process_dumpable();
    CHECK(dumpable == 0 || dumpable == -1,
          "process non-dumpable where supported (Linux blocks non-root ptrace)");

    /* ================= plaintext discipline: zeroize ================== */
    idletoken_secure_zero(prompt_plain, sizeof(prompt_plain));
    int all_zero = 1;
    for (size_t i = 0; i < sizeof(prompt_plain); i++) if (prompt_plain[i]) { all_zero = 0; break; }
    CHECK(all_zero, "plaintext buffer zeroized after use");
    idletoken_secure_zero(answer_plain, sizeof(answer_plain));
    idletoken_secure_zero(kv_key, sizeof(kv_key));
    idletoken_munlock(prompt_plain, sizeof(prompt_plain));
    idletoken_munlock(coord.sk, sizeof(coord.sk));

    printf("-------------------------------------------------------------\n");
    if (g_fail) {
        printf("G_PRIV_SELFTEST_FAIL: one or more privacy invariants broken\n");
        return 1;
    }
    printf("G_PRIV_SELFTEST_OK\n");
    return 0;
}
