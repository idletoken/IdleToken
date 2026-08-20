/* Unit tests for LAN discovery + verification-code pairing (src/common/
 * discovery.c). Pure C — runs on the Mac control machine and Linux nodes with
 * plain cc (+ -lpthread). No GPU, no model.
 *
 *   make disctest      # builds build/discovery_test and runs it
 *
 * Covers: SHA-256/HMAC known-answer vectors, code mint/validate, pair-id
 * derivation determinism + sensitivity, the PAIR_HELLO/PAIR_ACCEPT mutual-auth
 * preamble over a real loopback socket (right code succeeds + derives a shared
 * session key; wrong code is rejected), and broadcast + mock provider e2e. */

#include "idletoken_discovery.h"
#include "idletoken_net.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

static int g_fail = 0;
/* Mirrors THROTTLE_FREE_TRIES in discovery.c. Deliberately a separate literal:
 * the free allowance is a promise to users ("a few typos cost nothing"), so a
 * change to it should break this test and be re-decided, not silently follow. */
#define THROTTLE_FREE_TRIES_T 5

#define CHECK(cond, msg) do { \
    if (cond) { printf("  [ok] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); g_fail = 1; } \
} while (0)

static void hex(const uint8_t *b, size_t n, char *out) {
    static const char *h = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) { out[i*2]=h[b[i]>>4]; out[i*2+1]=h[b[i]&15]; }
    out[n*2] = '\0';
}

/* ---- crypto known-answer vectors ---------------------------------------- */
static void test_crypto(void) {
    printf("crypto vectors:\n");
    /* HMAC-SHA256, RFC 4231 test case 2: key="Jefe" data="what do ya want for nothing?" */
    uint8_t out[32]; char hx[65];
    idletoken_hmac_sha256((const uint8_t *)"Jefe", 4,
                       (const uint8_t *)"what do ya want for nothing?", 28, out);
    hex(out, 32, hx);
    CHECK(strcmp(hx, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843") == 0,
          "HMAC-SHA256 RFC4231 case 2");
}

/* ---- code helpers -------------------------------------------------------- */
static void test_code(void) {
    printf("code helpers:\n");
    char c[16];
    CHECK(idletoken_pair_code_mint(c, sizeof(c)) == 0 && strlen(c) == 6, "mint 6-char code");
    CHECK(idletoken_pair_code_valid(c) == 1, "minted code is valid");
    CHECK(idletoken_pair_code_valid("ABC234") == 1, "ABC234 valid");
    CHECK(idletoken_pair_code_valid("abc234") == 1, "lowercase accepted");
    CHECK(idletoken_pair_code_valid("ABC23O") == 0, "O rejected (ambiguous)");
    CHECK(idletoken_pair_code_valid("ABC231") == 0, "1 rejected (ambiguous)");
    CHECK(idletoken_pair_code_valid("ABC23") == 0, "too short rejected");
    CHECK(idletoken_pair_code_valid("ABC2345") == 0, "too long rejected");
}

/* ---- pair-id derivation -------------------------------------------------- */
static void test_pairid(void) {
    printf("pair-id derivation:\n");
    idletoken_pair_id a, b, c;
    idletoken_pair_id_from_code(&a, "ABC234");
    idletoken_pair_id_from_code(&b, "abc234");   /* case-insensitive */
    idletoken_pair_id_from_code(&c, "XYZ789");
    CHECK(memcmp(a.group_id, b.group_id, 32) == 0, "same code -> same group_id");
    CHECK(memcmp(a.psk, b.psk, 32) == 0, "same code -> same psk");
    CHECK(memcmp(a.group_id, c.group_id, 32) != 0, "diff code -> diff group_id");
    CHECK(memcmp(a.group_id, a.psk, 32) != 0, "group_id != psk (distinct labels)");
}

/* ---- auth preamble over a real loopback socket --------------------------- */
struct srv_arg { int fd; const char *code; int rc; uint8_t key[32]; };
static void *srv_thread(void *p) {
    struct srv_arg *a = (struct srv_arg *)p;
    idletoken_pair_id id; idletoken_pair_id_from_code(&id, a->code);
    a->rc = idletoken_pair_server_auth(a->fd, &id, a->key);
    return NULL;
}

/* Run one auth handshake: server uses srv_code, client uses cli_code. Returns
 * client rc; fills the two derived keys. */
static int run_auth(const char *srv_code, const char *cli_code,
                    uint8_t skey[32], uint8_t ckey[32]) {
    int lfd = idletoken_listen_tcp("127.0.0.1:0");
    if (lfd < 0) { perror("listen"); return -2; }
    /* discover the ephemeral port */
    struct sockaddr_in sa; socklen_t sl = sizeof(sa);
    getsockname(lfd, (struct sockaddr *)&sa, &sl);
    char addr[64]; snprintf(addr, sizeof(addr), "127.0.0.1:%d", ntohs(sa.sin_port));

    int cfd = idletoken_connect_tcp(addr);
    int afd = idletoken_accept_tcp(lfd);
    if (cfd < 0 || afd < 0) { return -2; }

    struct srv_arg sar = { afd, srv_code, -1, {0} };
    pthread_t th; pthread_create(&th, NULL, srv_thread, &sar);

    idletoken_pair_id cid; idletoken_pair_id_from_code(&cid, cli_code);
    int crc = idletoken_pair_client_auth(cfd, &cid, ckey);

    pthread_join(th, NULL);
    memcpy(skey, sar.key, 32);
    idletoken_close_fd(cfd); idletoken_close_fd(afd); idletoken_close_fd(lfd);
    (void)sar.rc;
    return crc;
}

static void test_auth(void) {
    printf("pairing auth preamble:\n");
    uint8_t skey[32], ckey[32];
    int rc = run_auth("ABC234", "ABC234", skey, ckey);
    CHECK(rc == 0, "matching code -> mutual auth success");
    CHECK(memcmp(skey, ckey, 32) == 0, "both sides derive the same session key");

    uint8_t s2[32], c2[32];
    int rc2 = run_auth("ABC234", "WRONG9", s2, c2);
    CHECK(rc2 != 0, "wrong code -> auth rejected");
}

/* ---- online-guessing throttle (D2) ---------------------------------------
 *
 * The security property is "wrong codes get slower"; the usability property is
 * "a person who mistypes a few times is not locked out". Both are asserted,
 * and in that order — a throttle that never fires would pass a test that only
 * checked the happy path. */
static void test_throttle(void) {
    printf("pairing throttle:\n");
    idletoken_pair_throttle_reset();

    /* The curve itself: free allowance, then doubling to a ceiling. */
    CHECK(idletoken_pair_backoff_ms(1) == 0, "first failure is free (typo)");
    CHECK(idletoken_pair_backoff_ms(5) == 0, "five failures still free");
    CHECK(idletoken_pair_backoff_ms(6) == 1000, "sixth failure costs 1 s");
    CHECK(idletoken_pair_backoff_ms(7) == 2000, "then it doubles");
    CHECK(idletoken_pair_backoff_ms(40) == 30000, "and is capped (no permanent lockout)");

    uint8_t sk[32], ck[32];
    /* Baseline: with a clean record the right code works. Without this the
     * "blocked" assertion below could pass for the wrong reason. */
    CHECK(run_auth("ABC234", "ABC234", sk, ck) == 0, "clean record: correct code works");

    idletoken_pair_throttle_reset();
    /* Spend the free allowance and one more, exactly as a guesser would. The
     * allowance counts FAILURES, so the Nth wrong code is still answered
     * normally; the penalty starts once it has failed N+1 times. */
    for (int i = 0; i < THROTTLE_FREE_TRIES_T + 1; i++) run_auth("ABC234", "WRONG9", sk, ck);
    /* Past the allowance the source is refused even WITH the right code —
     * this is the property that makes walking 2^30 impractical. */
    CHECK(run_auth("ABC234", "ABC234", sk, ck) != 0,
          "past the free allowance even a correct code is refused (guessing is throttled)");

    /* And the lockout is not permanent state damage: clearing the record (what
     * the backoff window does once it expires) restores normal service. */
    idletoken_pair_throttle_reset();
    CHECK(run_auth("ABC234", "ABC234", sk, ck) == 0,
          "once the window passes, the correct code works again");
}

/* ---- broadcast provider e2e (real UDP on a test port) -------------------- */
static void test_broadcast(void) {
    printf("broadcast provider e2e:\n");
    const uint16_t port = 24097;   /* test-only discovery port */
    idletoken_pair_id id; idletoken_pair_id_from_code(&id, "HOME24");

    idletoken_discovery *coord = idletoken_discovery_broadcast(port);
    int adv = coord->advertise(coord, &id, "192.168.1.50:14100");
    CHECK(adv == 0, "coordinator advertises");

    idletoken_discovery *worker = idletoken_discovery_broadcast(port);
    char addr[64] = "";
    int rc = worker->resolve(worker, &id, addr, sizeof(addr), 4000);
    CHECK(rc == 0 && strcmp(addr, "192.168.1.50:14100") == 0, "worker resolves coord addr by code");

    /* wrong code must NOT resolve (auth tag mismatch), short timeout */
    idletoken_pair_id wrong; idletoken_pair_id_from_code(&wrong, "NOPE55");
    char addr2[64] = "";
    int rc2 = worker->resolve(worker, &wrong, addr2, sizeof(addr2), 1200);
    CHECK(rc2 != 0, "wrong code does not resolve");

    worker->destroy(worker);
    coord->destroy(coord);
}

/* ---- mock provider parity ------------------------------------------------ */
static void test_mock(void) {
    printf("mock provider:\n");
    idletoken_pair_id id; idletoken_pair_id_from_code(&id, "MOCK22");
    idletoken_discovery *m = idletoken_discovery_mock();
    m->advertise(m, &id, "10.0.0.1:14100");
    char addr[64] = "";
    int rc = m->resolve(m, &id, addr, sizeof(addr), 0);
    CHECK(rc == 0 && strcmp(addr, "10.0.0.1:14100") == 0, "mock resolve returns advertised addr");
    idletoken_pair_id other; idletoken_pair_id_from_code(&other, "OTHER3");
    int rc2 = m->resolve(m, &other, addr, sizeof(addr), 0);
    CHECK(rc2 != 0, "mock resolve misses unknown group");
    m->destroy(m);
}

int main(void) {
    printf("=== discovery unit tests ===\n");
    test_crypto();
    test_code();
    test_pairid();
    test_auth();
    test_throttle();
    test_mock();
    test_broadcast();
    printf("\n%s\n", g_fail ? "DISCOVERY_TEST_FAIL" : "DISCOVERY_TEST_OK");
    return g_fail ? 1 : 0;
}
