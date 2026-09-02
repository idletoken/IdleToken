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
#include <time.h>
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

/* ---- CLUS-02: the cluster-wide meter, isolated from the per-source curve --
 *
 * The per-source backoff prices ONE attacker at ONE address. The same attacker
 * across N addresses buys N free allowances, which is why the second meter
 * exists. It is also the half that CANNOT be driven from a single loopback
 * address — the per-source throttle refuses long before a cluster-wide count
 * could climb to the threshold — so the evidence is split in two: the threshold
 * asserted directly through the pure predicate, and the live counter asserted
 * separately to prove that predicate is wired to real failures rather than
 * sitting beside them. Neither half is sufficient alone. */
static void test_global_meter(void) {
    printf("cluster-wide guessing meter (CLUS-02):\n");
    uint8_t sk[32], ck[32];

    /* Part 1 — the threshold. A home cluster forming for the first time can
     * produce a handful of mistyped codes across a few machines; the meter must
     * be nowhere near that, and must actually fire above it. */
    CHECK(idletoken_pair_global_would_block(1) == 0, "one failed join does not trip it");
    CHECK(idletoken_pair_global_would_block(20) == 0,
          "twenty failures across a forming cluster are still typing mistakes");
    CHECK(idletoken_pair_global_would_block(119) == 0, "just under the budget is allowed");
    CHECK(idletoken_pair_global_would_block(120) == 1, "at the budget, new attempts are refused");
    CHECK(idletoken_pair_global_would_block(100000) == 1, "and it stays tripped above it");

    /* Part 2 — the wiring. Drive actual handshakes and watch the live meter
     * move. Three is inside the per-source free allowance, so every one of them
     * is really evaluated rather than short-circuited by the backoff. */
    idletoken_pair_throttle_reset();
    int fails = -1; long long blocked = -1;
    idletoken_pair_global_state(&fails, &blocked);
    CHECK(fails == 0 && blocked == 0, "the meter starts clear after a reset");

    for (int i = 0; i < 3; i++) run_auth("ABC234", "WRONG9", sk, ck);
    idletoken_pair_global_state(&fails, &blocked);
    CHECK(fails == 3, "each real failed join increments the cluster-wide count");
    CHECK(blocked == 0, "and three failures do not refuse anybody");

    /* A SUCCESSFUL join must not silently discount the meter: clearing the
     * source's own record is right, resetting a cluster-wide attack signal
     * because one guess happened to land is not. */
    CHECK(run_auth("ABC234", "ABC234", sk, ck) == 0, "a correct code still joins at this point");
    idletoken_pair_global_state(&fails, &blocked);
    CHECK(fails == 3, "a success does not erase the cluster-wide failure history");

    idletoken_pair_throttle_reset();
    idletoken_pair_global_state(&fails, &blocked);
    CHECK(fails == 0 && blocked == 0, "reset clears the meter for the tests that follow");
}

/* ---- per-source table eviction (CLUS-02) --------------------------------
 *
 * The table is keyed by source address, and its real question is: does a source
 * that has just earned a penalty keep it when other machines show up? Plain LRU
 * answered "no" — the attacker's next address evicted the penalty the previous
 * one had just earned, so the table erased its own work under exactly the
 * behaviour it exists to price.
 *
 * This cannot be driven through sockets. A loopback test has one source
 * address, which is precisely the case where the bug is invisible: every
 * single-address test passes on both the broken and the fixed table. Hence the
 * synthetic-address seam, which calls the same key-based core the live path
 * calls.
 *
 * The flood is made of SUCCESSFUL joins on purpose. "A penalty must survive an
 * attacker spraying addresses" is the security claim, but "a penalty must
 * survive the neighbours' laptops joining normally" is the stronger one, and it
 * is the one an honest home LAN actually produces. */
static void addr_v4(uint8_t out[16], unsigned a, unsigned b, unsigned c, unsigned d) {
    memset(out, 0, 16);
    out[10] = 0xff; out[11] = 0xff;          /* v4-mapped, as peer_key() writes */
    out[12] = (uint8_t)a; out[13] = (uint8_t)b;
    out[14] = (uint8_t)c; out[15] = (uint8_t)d;
}

static void test_throttle_eviction(void) {
    printf("per-source table eviction (CLUS-02):\n");
    idletoken_pair_throttle_reset();

    uint8_t attacker[16], neighbour[16], other[16];
    addr_v4(attacker,  192, 168, 1, 66);
    addr_v4(neighbour, 192, 168, 1, 67);

    /* Eleven consecutive wrong codes: past the free allowance and far enough up
     * the curve to sit at the 30 s ceiling, which is comfortably above anything
     * the cluster-wide meter could contribute. Keeping the two apart is what
     * makes the assertion below mean "this SOURCE is still penalised" rather
     * than "something, somewhere, is refusing". */
    for (int i = 0; i < 11; i++) idletoken_pair_throttle_test_record(attacker, 1);
    long long before = idletoken_pair_throttle_test_check(attacker);
    CHECK(before > 25000, "a source past the free allowance is refused for the ceiling");

    /* A second known source that is NOT blocked — five failures is still inside
     * the free allowance. It is the control: if the flood below does not
     * actually evict anything, this one survives too and the attacker's
     * survival proves nothing. */
    for (int i = 0; i < 5; i++) idletoken_pair_throttle_test_record(neighbour, 1);
    CHECK(idletoken_pair_throttle_test_check(neighbour) == 0,
          "control: five failures are free, so this source is not blocked");

    /* 300 other machines join, successfully. More than the table holds. */
    for (int i = 0; i < 300; i++) {
        addr_v4(other, 10, 0, (unsigned)(i / 250), (unsigned)(i % 250));
        idletoken_pair_throttle_test_record(other, 0);
    }

    CHECK(idletoken_pair_throttle_test_check(attacker) > 25000,
          "the penalised source keeps its penalty through 300 other joins");

    /* Positive control on the flood itself. The neighbour had five failures on
     * record; one more would block it IF the table still remembered it. It does
     * not block, which is only possible because the flood really did evict —
     * so the table was genuinely under pressure and the survival asserted above
     * is a policy decision, not an artefact of a table that never filled. */
    idletoken_pair_throttle_test_record(neighbour, 1);
    CHECK(idletoken_pair_throttle_test_check(neighbour) == 0,
          "control: the flood really does evict — the unblocked source was forgotten");

    /* And the meter stayed out of the way, so the numbers above are the
     * per-source policy talking and nothing else. */
    int gfails = -1; long long gblocked = -1;
    idletoken_pair_global_state(&gfails, &gblocked);
    CHECK(gblocked == 0, "the cluster-wide meter did not fire, so this measured one source");

    idletoken_pair_throttle_reset();
}

/* ---- handshake deadline (CLUS-05) ---------------------------------------
 *
 * The attack this closes needs no join code: open TCP to the coordinator and
 * then say nothing. The join loop is serial, so one silent socket used to pin
 * it in recv() forever — and IDLETOKEN_JOIN_WAIT_S could not fire, because that
 * deadline is measured around accept(), not around the read.
 *
 * Asserted from the peer's own fd while the server thread is parked inside
 * recv(): that is the only moment the deadline is observable, and observing it
 * there costs milliseconds instead of the 15 s a real timeout would take. */
struct deadline_arg { int fd; const char *code; int rc; };

static void *deadline_srv(void *p) {
    struct deadline_arg *a = (struct deadline_arg *)p;
    idletoken_pair_id id; idletoken_pair_id_from_code(&id, a->code);
    uint8_t key[32];
    a->rc = idletoken_pair_server_auth(a->fd, &id, key);
    return NULL;
}

static void test_pair_deadline(void) {
    printf("handshake deadline:\n");
    /* A source still serving a penalty is refused before it ever reads, which
     * would make the observation below race against the restore. */
    idletoken_pair_throttle_reset();

    int lfd = idletoken_listen_tcp("127.0.0.1:0");
    if (lfd < 0) { CHECK(0, "listen for the deadline test"); return; }
    struct sockaddr_in sa; socklen_t sl = sizeof(sa);
    getsockname(lfd, (struct sockaddr *)&sa, &sl);
    char addr[64]; snprintf(addr, sizeof(addr), "127.0.0.1:%d", ntohs(sa.sin_port));
    int cfd = idletoken_connect_tcp(addr);
    int afd = idletoken_accept_tcp(lfd);
    if (cfd < 0 || afd < 0) { CHECK(0, "connect/accept for the deadline test"); return; }

    /* A distinctive caller-owned deadline, so "restored" cannot be confused
     * with "never changed" or "left at the handshake's own 15 s". */
    const int caller_ms = 4321;
    CHECK(idletoken_set_recv_timeout(afd, caller_ms) == 0, "a caller can set its own recv deadline");
    int read_back = -1;
    CHECK(idletoken_get_recv_timeout(afd, &read_back) == 0 && read_back == caller_ms,
          "and read it back unchanged");

    struct deadline_arg arg = { afd, "ABC234", -1 };
    pthread_t th; pthread_create(&th, NULL, deadline_srv, &arg);

    /* The server is now parked in recv() waiting for a PAIR_HELLO that this
     * test deliberately has not sent yet. */
    usleep(250 * 1000);
    int during = -1;
    CHECK(idletoken_get_recv_timeout(afd, &during) == 0 && during == 15000,
          "a silent peer cannot park the join loop: the handshake imposes 15 s");

    /* Unblock it the honest way — a real PAIR_HELLO carrying the wrong group —
     * so the server takes its normal reject path rather than an fd yanked out
     * from under it. */
    idletoken_pair_id wrong; idletoken_pair_id_from_code(&wrong, "WRONG9");
    uint8_t ck[32];
    idletoken_pair_client_auth(cfd, &wrong, ck);
    pthread_join(th, NULL);

    CHECK(arg.rc != 0, "the wrong group is still refused with the deadline in force");
    int after = -1;
    CHECK(idletoken_get_recv_timeout(afd, &after) == 0 && after == caller_ms,
          "and the caller's own deadline is put back afterwards");

    idletoken_close_fd(cfd); idletoken_close_fd(afd); idletoken_close_fd(lfd);
}

/* ---- peer-supplied string gates (CLUS-14, CHAIN-08 LAN side) -------------
 *
 * TLS says who may speak, not what they may say. The gate's whole job is to
 * refuse a hostname that is really a fragment of JSON or a forged log line, so
 * the test leads with that exact payload rather than with well-formed input. */
static void test_peer_strings(void) {
    printf("peer string gates:\n");

    /* The attack, spelled out: an authorized-but-modified worker reports this
     * as its hostname, and the coordinator splices it into the cluster status
     * body with %s. If either gate accepted it, the client would parse a
     * document in which this machine claims to be the coordinator. */
    const char *inject = "a\",\"role\":\"coordinator\",\"x\":\"";
    CHECK(idletoken_peer_label_ok(inject, 64) == 0, "JSON-injecting hostname is refused (label)");
    CHECK(idletoken_peer_host_ok(inject, 64) == 0, "JSON-injecting hostname is refused (host)");
    CHECK(idletoken_peer_label_ok("box-a\nJan 01 00:00:00 coord: all clear", 64) == 0,
          "a forged log line embedded in a name is refused");

    CHECK(idletoken_peer_label_ok("NVIDIA GeForce RTX 4090", 64) == 1, "a real GPU name passes");
    CHECK(idletoken_peer_label_ok(NULL, 64) == 0, "NULL is refused");
    CHECK(idletoken_peer_label_ok("", 64) == 0, "empty is refused");
    CHECK(idletoken_peer_label_ok("a\\b", 64) == 0, "a backslash is refused");
    CHECK(idletoken_peer_label_ok("caf\xc3\xa9", 64) == 0, "non-ASCII is refused (UTF-8 truncation)");
    CHECK(idletoken_peer_label_ok("a\x7f" "b", 64) == 0, "DEL is refused");

    char long_name[80];
    memset(long_name, 'a', sizeof(long_name)); long_name[64] = '\0';
    CHECK(idletoken_peer_label_ok(long_name, 64) == 1, "exactly max_len is accepted");
    long_name[64] = 'a'; long_name[65] = '\0';
    CHECK(idletoken_peer_label_ok(long_name, 64) == 0, "one over max_len is refused, not clamped");

    CHECK(idletoken_peer_host_ok("192.168.1.50:50052", 64) == 1, "an ip:port endpoint passes");
    CHECK(idletoken_peer_host_ok("box-a_1.local", 64) == 1, "a normal hostname passes");
    CHECK(idletoken_peer_host_ok("box a", 64) == 0, "a space in an address is refused");
    CHECK(idletoken_peer_host_ok("a/../b", 64) == 0, "path separators in an address are refused");
    CHECK(idletoken_peer_host_ok("box-a;rm -rf /", 64) == 0,
          "shell metacharacters in an address are refused");
    CHECK(idletoken_peer_label_ok("bell\x07here", 64) == 0,
          "a low control byte inside a label is refused");

    /* The gate has to keep passing what real machines actually send, or it
     * ships as an outage rather than a defence. This is the version banner the
     * rpc-supervisor worker puts in HELLO today. */
    CHECK(idletoken_peer_label_ok("0.1.15 (rpc-cache-v1+rpc-cpu-v1)", 64) == 1,
          "a real worker version banner still passes");
}

/* ---- JSON escaping at egress (CLUS-14 belt-and-braces) ------------------- */
static void test_json_escape(void) {
    printf("json escaping:\n");
    char out[128];
    size_t n = idletoken_json_escape(out, sizeof(out), "a\"b\\c", 5);
    CHECK(strcmp(out, "a\\\"b\\\\c") == 0, "quote and backslash are escaped");
    CHECK(n == strlen(out), "the returned length matches the string written");

    idletoken_json_escape(out, sizeof(out), "x\ny", 3);
    CHECK(strcmp(out, "x\\ny") == 0, "a newline cannot forge a log line");
    idletoken_json_escape(out, sizeof(out), "x\x01y", 3);
    CHECK(strcmp(out, "x\\u0001y") == 0, "a control byte becomes a \\u escape");

    /* Truncation must not cut an escape in half: "\\u00" is not shorter JSON,
     * it is broken JSON, and the parser on the other end fails on the whole
     * document rather than on one field. */
    char tight[8];
    size_t t = idletoken_json_escape(tight, sizeof(tight), "ab\x01" "cd", 5);
    CHECK(t == strlen(tight), "a truncated body is still NUL-terminated and self-consistent");
    CHECK(strstr(tight, "\\u00") == NULL || strcmp(tight, "ab\\u0001") == 0,
          "truncation never lands inside an escape sequence");

    CHECK(idletoken_json_escape(out, 0, "x", 1) == 0, "a zero-capacity destination writes nothing");

    /* One byte is capacity for the terminator and nothing else. The result must
     * still be a valid empty C string, not an unterminated buffer. */
    char one[1] = { 'Z' };
    CHECK(idletoken_json_escape(one, sizeof(one), "abc", 3) == 0 && one[0] == '\0',
          "a 1-byte destination yields an empty, terminated string");
}

/* ---- length-prefixed string reads (CLUS-06, CLUS-14) --------------------- */
static void test_buf_strings(void) {
    printf("wire string reads:\n");

    /* Two hostnames that differ only past byte 63 — the shape of the CLUS-06
     * impersonation. The lenient reader is what makes them the same name. */
    char a[200], b[200];
    memset(a, 'h', sizeof(a)); a[199] = '\0';
    memcpy(b, a, sizeof(b)); b[100] = 'X';

    uint8_t wire[512];
    idletoken_buf w; idletoken_buf_init(&w, wire, sizeof(wire));
    idletoken_buf_put_str(&w, a);
    idletoken_buf_put_str(&w, b);

    char ra[64], rb[64];
    idletoken_buf r; idletoken_buf_init(&r, wire, w.pos);
    idletoken_buf_get_str(&r, ra, sizeof(ra));
    idletoken_buf_get_str(&r, rb, sizeof(rb));
    CHECK(!r.err && strcmp(ra, rb) == 0,
          "control: the lenient reader collapses two different names into one");

    /* The strict reader refuses instead, and poisons the cursor so the caller
     * cannot go on parsing a message it has already mis-read. */
    char sa[64];
    idletoken_buf s; idletoken_buf_init(&s, wire, w.pos);
    CHECK(idletoken_buf_get_str_strict(&s, sa, sizeof(sa)) != 0 && s.err,
          "the strict reader refuses an over-long identity");
    CHECK(sa[0] == '\0', "and leaves no partial name behind");

    /* It must still be an ordinary reader for values that fit. */
    idletoken_buf w2; idletoken_buf_init(&w2, wire, sizeof(wire));
    idletoken_buf_put_str(&w2, "box-a");
    idletoken_buf r2; idletoken_buf_init(&r2, wire, w2.pos);
    char fits[64] = "";
    CHECK(idletoken_buf_get_str_strict(&r2, fits, sizeof(fits)) == 0 &&
          !r2.err && strcmp(fits, "box-a") == 0, "a name that fits reads normally");
}

/* ---- cursor bounds under a hostile length (CLUS-14) ---------------------- */
static void test_buf_bounds(void) {
    printf("cursor bounds:\n");
    uint8_t backing[32];
    uint8_t dst[8];
    idletoken_buf b;
    idletoken_buf_init(&b, backing, sizeof(backing));
    b.pos = 8;
    /* The length comes off the wire, so it is the attacker's to choose. The old
     * check computed `pos + n > cap`, which WRAPS: 8 + (SIZE_MAX - 4) is 3, and
     * 3 > 32 is false, so the read was allowed and the memcpy that followed ran
     * off the end. Running this against the pre-fix cursor is the positive
     * control — it does not report a failure, it crashes the process. */
    CHECK(idletoken_buf_get_bytes(&b, dst, SIZE_MAX - 4) != 0 && b.err,
          "a length that wraps pos+n is refused, not wrapped into a small number");

    idletoken_buf c;
    idletoken_buf_init(&c, backing, sizeof(backing));
    CHECK(idletoken_buf_get_bytes(&c, dst, sizeof(dst)) == 0 && !c.err,
          "control: an in-bounds read still works");

    /* The same class of hostile length, but arriving the way a real peer sends
     * one: as the u32 length prefix of a string. SIZE_MAX above only wraps on
     * a 64-bit cursor; 0xffffffff is what wraps a 32-bit one, and the Windows
     * MinGW worker was 32-bit until recently. Both paths, or the fix is only
     * proven on the machine the test happened to run on. */
    uint8_t raw[8] = { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0 };
    idletoken_buf ov; idletoken_buf_init(&ov, raw, sizeof(raw));
    char ov_out[16] = "";
    CHECK(idletoken_buf_get_str(&ov, ov_out, sizeof(ov_out)) != 0 && ov.err,
          "a 4 GiB string length is refused rather than wrapping the bounds check");
}

/* ---- CLUS-05: the deadline actually fires ---------------------------------
 *
 * test_pair_deadline above proves the handshake IMPOSES 15 s and puts the
 * caller's own value back. It does not prove a timeout ever expires — a
 * setsockopt that silently did nothing would pass it. This is the other half:
 * a real silent peer, a short deadline, and a read that returns instead of
 * parking the thread. "Blocks forever" cannot be asserted directly, so the
 * assertion is the positive one. */
static void test_recv_deadline(void) {
    printf("recv deadline fires (CLUS-05):\n");
    int lfd = idletoken_listen_tcp("127.0.0.1:0");
    if (lfd < 0) { CHECK(0, "listen for deadline test"); return; }
    struct sockaddr_in sa; socklen_t sl = sizeof(sa);
    getsockname(lfd, (struct sockaddr *)&sa, &sl);
    char addr[64]; snprintf(addr, sizeof(addr), "127.0.0.1:%d", ntohs(sa.sin_port));

    int cfd = idletoken_connect_tcp(addr);       /* connects, then says nothing */
    int afd = idletoken_accept_tcp(lfd);
    if (cfd < 0 || afd < 0) { CHECK(0, "socket pair for deadline test"); return; }

    int back = -1;
    CHECK(idletoken_set_recv_timeout(afd, 400) == 0 &&
          idletoken_get_recv_timeout(afd, &back) == 0 && back >= 300 && back <= 500,
          "the deadline round-trips through the socket option");

    time_t t0 = time(NULL);
    idletoken_msg_header h; uint8_t pay[64];
    int rc = idletoken_recv_msg(afd, &h, pay, sizeof(pay));
    double waited = difftime(time(NULL), t0);
    CHECK(rc != 0 && waited < 5.0,
          "a silent peer's read returns instead of pinning the join loop");

    /* 0 restores the historical unbounded behaviour, so a caller that saved and
     * put back "no timeout" really does put it back. */
    CHECK(idletoken_set_recv_timeout(afd, 0) == 0 &&
          idletoken_get_recv_timeout(afd, &back) == 0 && back == 0,
          "zero clears the deadline again");

    idletoken_close_fd(cfd); idletoken_close_fd(afd); idletoken_close_fd(lfd);
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
    test_global_meter();
    test_throttle_eviction();
    test_pair_deadline();
    test_peer_strings();
    test_json_escape();
    test_buf_strings();
    test_buf_bounds();
    test_recv_deadline();
    test_mock();
    test_broadcast();
    printf("\n%s\n", g_fail ? "DISCOVERY_TEST_FAIL" : "DISCOVERY_TEST_OK");
    return g_fail ? 1 : 0;
}
