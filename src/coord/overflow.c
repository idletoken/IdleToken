/* IdleToken — overflow routing, coordinator side. See include/idletoken_overflow.h
 * for the three rules this file exists to keep.
 *
 * Three parts, in the order they have to be right: the trust anchor (a
 * substituted platform key fails silently and looks exactly like success, so it
 * comes first), the policy that decides whether a given request may go out at
 * all, and the sealed exchange itself.
 *
 * Deliberately NOT here: how an OpenAI or an Anthropic request body is read,
 * and how an answer is written back in either shape. That is the coordinator's
 * protocol knowledge and it already exists there; this file is transport,
 * crypto and policy. */

#include "idletoken_overflow.h"
#include "idletoken_b64.h"
#include "idletoken_http.h"
#include "idletoken_privacy.h"
#include "idletoken_sodium_seal.h"
/* The coordinator's existing HTTP/1.1 client. It is named after the engine it
 * was written for, but it speaks plain HTTP to a "host:port" and nothing about
 * it is llama-specific — reusing it is what keeps the plan's "no new
 * dependency" promise, and it already has the bounded-read discipline that the
 * engine link learned the hard way (results/coord-wedge-20260817.md).
 *
 * It reads no proxy environment variables, so the local-proxy hijack that bites
 * the CLIENT's loopback SSE (Clash and friends) cannot reach this connection.
 * A TUN-mode VPN still can, and that is out of any application's hands. */
#include "idletoken_llama_sidecar.h"

#include "tweetnacl.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The string the platform signs. Its engine-side twin is
 * KeyAttestationService.signedMessage() in
 * platform/packages/gateway/src/crypto/key-attestation.service.ts — the two must
 * agree byte for byte, so if one is edited the other has to be edited with it. */
#define OVF_KEY_DOMAIN "idletoken-platform-key-v1"

/* --- the pinned verify key ------------------------------------------------ */

const uint8_t *idletoken_overflow_verify_key(const char **out_source) {
    static uint8_t key[IDLETOKEN_OVF_PK_BYTES];
    static int     resolved = 0;   /* 0 = not looked at, 1 = have one, -1 = none */
    static const char *source = "none";

    if (resolved == 0) {
        resolved = -1;
        /* Escape hatch, and loud about it. Development gateways mint an
         * EPHEMERAL signing key per process and print its verify key (see the
         * attestation service), so without this no test could ever exercise the
         * happy path against a real gateway. It is announced every time for the
         * same reason GGML_RPC_ALLOW_PLAINTEXT is: an unusual trust anchor must
         * never be a quiet one. */
        const char *env = getenv("IDLETOKEN_PLATFORM_VERIFY_KEY");
        const char *pinned = IDLETOKEN_PLATFORM_VERIFY_KEY_B64;
        const char *src_b64 = NULL;
        if (env && env[0]) {
            src_b64 = env;
            source  = "env";
        } else if (pinned[0]) {
            src_b64 = pinned;
            source  = "pinned";
        }
        if (src_b64) {
            size_t n = 0;
            uint8_t *raw = idletoken_b64_decode(src_b64, strlen(src_b64), &n);
            if (raw && n == IDLETOKEN_OVF_PK_BYTES) {
                memcpy(key, raw, sizeof(key));
                resolved = 1;
                if (!strcmp(source, "env"))
                    fprintf(stderr, "coord: overflow: IDLETOKEN_PLATFORM_VERIFY_KEY is set — "
                                    "trusting a platform signing key this build did not pin\n");
            } else {
                /* Loud, and still refused: a malformed pin is a configuration
                 * error, and continuing "unpinned" would turn it into a machine
                 * that seals prompts to whatever the network offers. */
                fprintf(stderr, "coord: overflow: platform verify key (%s) is not 32 "
                                "base64 bytes — refusing to use it\n", source);
                source = "none";
            }
            free(raw);
        }
    }
    if (out_source) *out_source = source;
    return resolved == 1 ? key : NULL;
}

/* --- ISO-8601 (UTC) -> unix seconds --------------------------------------- */

/* Days since 1970-01-01 for a proleptic-Gregorian y/m/d (Howard Hinnant's
 * days_from_civil). Written out rather than calling timegm(), which MinGW does
 * not have and whose absence would only surface on a Windows build. */
static long long days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    long long era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - (int)(era * 400));
    unsigned doy = (153u * (m + (m > 2 ? -3u : 9u)) + 2u) / 5u + d - 1u;
    unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097LL + (long long)doe - 719468LL;
}

/* "YYYY-MM-DDTHH:MM:SS[.mmm]Z" -> unix seconds, or -1 if it is not that.
 *
 * The trailing Z is REQUIRED. The platform emits Date.toISOString(), which
 * always ends in Z; accepting a local-time spelling would mean silently reading
 * an expiry hours away from the one that was signed. */
static long long iso8601_utc_to_unix(const char *s) {
    if (!s) return -1;
    int y, mo, d, h, mi, sec;
    if (sscanf(s, "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &sec) != 6)
        return -1;
    if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || sec > 60)
        return -1;
    size_t n = strlen(s);
    if (n == 0 || s[n - 1] != 'Z') return -1;
    long long days = days_from_civil(y, (unsigned)mo, (unsigned)d);
    return days * 86400LL + h * 3600LL + mi * 60LL + sec;
}

/* --- attestation verification --------------------------------------------- */

#define OVF_FAIL(...) do { \
        if (err && err_cap) snprintf(err, err_cap, __VA_ARGS__); \
        return -1; \
    } while (0)

int idletoken_overflow_key_verify(const uint8_t verify_pk[IDLETOKEN_OVF_PK_BYTES],
                                  const char *json, size_t json_len,
                                  long long now_unix,
                                  idletoken_overflow_key *out,
                                  char *err, size_t err_cap) {
    if (err && err_cap) err[0] = '\0';
    if (!verify_pk)
        OVF_FAIL("this build pinned no platform signing key");
    if (!json || json_len == 0)
        OVF_FAIL("empty platform-key response");

    char alg[48] = "", sign_alg[32] = "", pubkey_b64[128] = "",
         not_after[64] = "", sig_b64[128] = "", key_id[IDLETOKEN_OVF_KEYID_CAP] = "";
    idletoken_http_json_extract_str(json, json_len, "alg", alg, sizeof alg);
    idletoken_http_json_extract_str(json, json_len, "sign_alg", sign_alg, sizeof sign_alg);
    idletoken_http_json_extract_str(json, json_len, "pubkey", pubkey_b64, sizeof pubkey_b64);
    idletoken_http_json_extract_str(json, json_len, "not_after", not_after, sizeof not_after);
    idletoken_http_json_extract_str(json, json_len, "sig", sig_b64, sizeof sig_b64);
    idletoken_http_json_extract_str(json, json_len, "key_id", key_id, sizeof key_id);

    /* An algorithm we do not implement must be refused, not assumed to be the
     * one we do. "sig" over an unknown scheme is a field we cannot judge. */
    if (strcmp(alg, "x25519-sealedbox") != 0)
        OVF_FAIL("unsupported alg \"%s\" (want x25519-sealedbox)", alg);
    if (strcmp(sign_alg, "ed25519") != 0)
        OVF_FAIL("unsupported sign_alg \"%s\" (want ed25519)", sign_alg);
    if (!pubkey_b64[0]) OVF_FAIL("response carries no pubkey");
    if (!not_after[0])  OVF_FAIL("response carries no not_after");
    /* No signature at all is its own reason. The platform is specified to
     * answer 503 rather than serve an unsigned key, and a client that ever
     * accepts one has no verification left — so name this case in the log. */
    if (!sig_b64[0])    OVF_FAIL("response carries no signature");

    size_t pk_len = 0, sig_len = 0;
    uint8_t *pk  = idletoken_b64_decode(pubkey_b64, strlen(pubkey_b64), &pk_len);
    uint8_t *sig = idletoken_b64_decode(sig_b64, strlen(sig_b64), &sig_len);
    if (!pk || pk_len != IDLETOKEN_OVF_PK_BYTES) {
        free(pk); free(sig);
        OVF_FAIL("pubkey is not %d base64 bytes", IDLETOKEN_OVF_PK_BYTES);
    }
    if (!sig || sig_len != IDLETOKEN_OVF_SIGN_BYTES) {
        free(pk); free(sig);
        OVF_FAIL("signature is not %d base64 bytes", IDLETOKEN_OVF_SIGN_BYTES);
    }

    /* Rebuild the signed bytes from the response's OWN fields, always with the
     * domain prefix. Because we prepend it unconditionally, a signature made
     * over the bare "<pubkey>\n<not_after>" simply does not verify — the
     * separation is enforced by construction, not by a check that could be
     * deleted. */
    char msg[320];
    int msg_len = snprintf(msg, sizeof msg, OVF_KEY_DOMAIN "\n%s\n%s",
                           pubkey_b64, not_after);
    if (msg_len < 0 || (size_t)msg_len >= sizeof msg) {
        free(pk); free(sig);
        OVF_FAIL("platform-key fields are implausibly long");
    }

    /* TweetNaCl only offers the combined form: sm = sig || message. */
    size_t sm_len = (size_t)msg_len + IDLETOKEN_OVF_SIGN_BYTES;
    uint8_t *sm = malloc(sm_len);
    uint8_t *opened = malloc(sm_len);
    if (!sm || !opened) {
        free(pk); free(sig); free(sm); free(opened);
        OVF_FAIL("out of memory verifying the platform key");
    }
    memcpy(sm, sig, IDLETOKEN_OVF_SIGN_BYTES);
    memcpy(sm + IDLETOKEN_OVF_SIGN_BYTES, msg, (size_t)msg_len);
    unsigned long long opened_len = 0;
    int ok = crypto_sign_open(opened, &opened_len, sm, (unsigned long long)sm_len,
                              verify_pk) == 0;
    free(sm); free(opened); free(sig);
    if (!ok) {
        free(pk);
        OVF_FAIL("signature does not verify against the pinned platform key");
    }

    long long exp = iso8601_utc_to_unix(not_after);
    if (exp < 0) {
        free(pk);
        OVF_FAIL("not_after \"%s\" is not UTC ISO-8601", not_after);
    }
    if (exp <= now_unix) {
        free(pk);
        OVF_FAIL("platform key expired at %s", not_after);
    }

    memcpy(out->pubkey, pk, IDLETOKEN_OVF_PK_BYTES);
    free(pk);
    snprintf(out->key_id, sizeof out->key_id, "%s", key_id);
    out->not_after = exp;
    return 0;
}

/* --- configuration and the forwarding decision -----------------------------
 *
 * One lock covers the config and the day's spend. It is taken for a handful of
 * comparisons on the 429 path only — never around a network call — and the
 * llamacpp relay pool means several threads really can arrive at once. */

/* Defined with the exchange further down; declared here because switching
 * overflow on has to VERIFY the platform key, not merely intend to. */
static int ovf_url_to_addr(const char *url, char *out, size_t cap);
static int ovf_ensure_key(const char *addr, int *out_unreachable,
                          char *err, size_t err_cap);

static pthread_mutex_t g_ovf_mu = PTHREAD_MUTEX_INITIALIZER;
static struct {
    int       on;
    char      url[256];
    char      api_key[192];
    long long wait_ms;
    long long daily_cap_milli;
    long long spent_milli;   /* today's, per the platform's own figures */
    long long day;           /* unix day number the spend belongs to */
} g_ovf;

/* UTC calendar day. The platform's own key cap resets on the UTC day
 * (api-surface §5.2) and two ceilings that reset at different moments would be
 * impossible to explain to whoever hit one of them. */
static long long ovf_today(void) { return (long long)time(NULL) / 86400; }

/* Caller holds the lock. */
static void ovf_roll_day(void) {
    long long d = ovf_today();
    if (d != g_ovf.day) { g_ovf.day = d; g_ovf.spent_milli = 0; }
}

int idletoken_overflow_configure(const idletoken_overflow_cfg *cfg,
                                 char *err, size_t err_cap) {
    if (err && err_cap) err[0] = '\0';
    if (!cfg) OVF_FAIL("no overflow configuration");

    /* RULE 2 — fail closed. api_token_ok() waves everything through when no
     * token is set, which is a fine default for a machine that can only spend
     * its OWN hardware. The moment it can spend credits instead, that same
     * default means anyone on the WiFi can empty the balance and the owner
     * finds out from the ledger (api-surface §5.3). A warning would not do:
     * the whole point is that the dangerous configuration must not run. */
    if (!cfg->api_token_set)
        OVF_FAIL("overflow needs a local API token first — generate one in the "
                 "client's settings (without it anyone on this network could "
                 "spend your credits)");
    if (!cfg->url || !cfg->url[0])
        OVF_FAIL("overflow needs the platform URL");
    if (!cfg->api_key || !cfg->api_key[0])
        OVF_FAIL("overflow needs an account key to spend against");
    /* Checked before the trust anchor: a URL this binary cannot dial is a
     * configuration mistake, and reporting it as a missing signing key would
     * send whoever hit it looking in the wrong place. https:// in particular is
     * refused rather than downgraded to plaintext on port 443 -- that would be
     * the worst of the two, since the configuration would still LOOK encrypted. */
    char addr[160];
    if (ovf_url_to_addr(cfg->url, addr, sizeof addr) != 0)
        OVF_FAIL("cannot use platform URL \"%s\" (https:// needs a TLS client "
                 "this binary has not got; put a plain-HTTP hop in front)", cfg->url);

    /* RULE 3 — no silent fallback. Without the pinned signing key there is no
     * way to tell the platform's encryption key from an attacker's, and sealing
     * to the wrong one looks exactly like success. Refuse to switch on rather
     * than switch on something that only appears to be encrypted. */
    const char *src = "none";
    if (!idletoken_overflow_verify_key(&src))
        OVF_FAIL("this build pinned no platform signing key, so a platform "
                 "encryption key cannot be told apart from a substituted one");

    /* Fetch and verify the platform's encryption key NOW (plan O1). Switching
     * overflow on is the moment to find out that the key does not verify — the
     * alternative is discovering it on the first busy minute of the day, which
     * is when nobody is reading logs.
     *
     * Unreachable is not the same answer as unverifiable, and the difference
     * decides whether this machine starts. A home coordinator regularly comes
     * up before its uplink does; refusing to run then would turn a slow router
     * into a broken product. A key that ANSWERS and does not verify is the
     * attack this whole mechanism exists for, and that one is fatal. Either
     * way nothing is ever sealed to an unverified key: the exchange runs the
     * same check again before it sends. */
    {
        int unreachable = 0;
        char kerr[240];
        if (ovf_ensure_key(addr, &unreachable, kerr, sizeof kerr) != 0) {
            if (!unreachable)
                OVF_FAIL("%s", kerr);
            fprintf(stderr, "coord: overflow: %s — carrying on; the key is "
                            "fetched and verified again before anything is sent\n", kerr);
        }
    }

    pthread_mutex_lock(&g_ovf_mu);
    snprintf(g_ovf.url, sizeof g_ovf.url, "%s", cfg->url);
    snprintf(g_ovf.api_key, sizeof g_ovf.api_key, "%s", cfg->api_key);
    g_ovf.wait_ms = cfg->wait_ms > 0 ? cfg->wait_ms : 0;
    g_ovf.daily_cap_milli = cfg->daily_cap_milli > 0
        ? cfg->daily_cap_milli : IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI;
    g_ovf.day = ovf_today();
    g_ovf.spent_milli = 0;
    g_ovf.on = 1;
    long long cap = g_ovf.daily_cap_milli, wait = g_ovf.wait_ms;
    pthread_mutex_unlock(&g_ovf_mu);

    fprintf(stderr, "coord: overflow: on — platform %s, verify key %s, "
                    "forward when the wait is >= %lldms, at most %lld "
                    "milli-credits a day\n", cfg->url, src, wait, cap);
    return 0;
}

int idletoken_overflow_enabled(void) {
    pthread_mutex_lock(&g_ovf_mu);
    int on = g_ovf.on;
    pthread_mutex_unlock(&g_ovf_mu);
    return on;
}

void idletoken_overflow_spend_today(long long *spent_milli, long long *cap_milli) {
    pthread_mutex_lock(&g_ovf_mu);
    ovf_roll_day();
    if (spent_milli) *spent_milli = g_ovf.spent_milli;
    if (cap_milli)   *cap_milli   = g_ovf.daily_cap_milli;
    pthread_mutex_unlock(&g_ovf_mu);
}

void idletoken_overflow_note_spend(long long milli) {
    if (milli <= 0) return;
    pthread_mutex_lock(&g_ovf_mu);
    ovf_roll_day();
    g_ovf.spent_milli += milli;
    pthread_mutex_unlock(&g_ovf_mu);
}

int idletoken_overflow_should_forward(int from_platform, int want_stream,
                                      long long est_wait_ms, const char **why) {
    const char *reason = "off";
    int yes = 0;

    /* RULE 1, and it is checked FIRST so that no setting, threshold or budget
     * can be read as an exception to it: a job the platform dispatched ends
     * here or is refused here, and is never handed on. That one rule removes
     * every loop — provider ping-pong, and a request finding its way back to
     * the cluster it came from — without a hop counter to get wrong.
     *
     * The sense of the test is the subtle half. "No marker means local" is what
     * makes the feature work at all: the curl and the Claude Code on this LAN
     * have no reason to send our private header, so treating unknown origins as
     * un-forwardable would forward nothing, ever, and look like a feature that
     * was never switched on. Which puts the entire weight of this rule on the
     * agent really setting the header — hence the gate that asserts against the
     * real agent binary rather than against this function (design §3). */
    if (from_platform) {
        reason = "platform work is never forwarded";
    } else if (want_stream) {
        /* §5.4: refuse a stream honestly rather than relay one we cannot
         * finish. "The upstream died and the caller already has half an answer"
         * has no agreed meaning yet, and half an answer is worse than a 429. */
        reason = "stream:true is refused locally, not forwarded";
    } else {
        pthread_mutex_lock(&g_ovf_mu);
        if (!g_ovf.on) {
            reason = "off";
        } else {
            ovf_roll_day();
            if (g_ovf.spent_milli >= g_ovf.daily_cap_milli) {
                reason = "daily spend cap reached";
            } else if (est_wait_ms < g_ovf.wait_ms) {
                reason = "the estimated wait is below the threshold";
            } else {
                reason = "busy, and borrowing is allowed";
                yes = 1;
            }
        }
        pthread_mutex_unlock(&g_ovf_mu);
    }
    if (why) *why = reason;
    return yes;
}

/* --- the sealed exchange --------------------------------------------------- */

/* "http://host[:port]/..." or "host:port" -> "host:port". Mirrors the platform
 * agent's url_to_addr, including the refusal of https:// — there is no TLS
 * client in this binary either, and a silent downgrade to port 443 plaintext
 * would be the worst of both. */
static int ovf_url_to_addr(const char *url, char *out, size_t cap) {
    const char *p = url;
    if (!strncmp(p, "http://", 7)) p += 7;
    else if (!strncmp(p, "https://", 8)) return -1;
    size_t n = strcspn(p, "/");
    if (n == 0 || n >= cap) return -1;
    memcpy(out, p, n);
    out[n] = '\0';
    if (!strchr(out, ':')) {
        if (n + 4 > cap) return -1;
        memcpy(out + n, ":80", 4);
    }
    return 0;
}

/* The raw, still-escaped span of a JSON string field: same first-occurrence
 * scan as everywhere else in this codebase (see idletoken_http.h's note on the
 * tolerance level). Returns 0 and sets val and val_len, or -1. */
static int ovf_str_span(const char *json, size_t len, const char *key,
                        const char **val, size_t *val_len) {
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen + 3 < len; i++) {
        if (json[i] != '"' || memcmp(json + i + 1, key, klen) != 0 ||
            json[i + 1 + klen] != '"') continue;
        size_t p = i + klen + 2;
        while (p < len && (json[p] == ' ' || json[p] == '\t')) p++;
        if (p >= len || json[p] != ':') continue;
        p++;
        while (p < len && (json[p] == ' ' || json[p] == '\t')) p++;
        if (p >= len || json[p] != '"') return -1;
        p++;
        size_t start = p;
        int esc = 0;
        while (p < len) {
            char c = json[p];
            if (esc) esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == '"') break;
            p++;
        }
        if (p >= len) return -1;
        *val = json + start;
        *val_len = p - start;
        return 0;
    }
    return -1;
}

/* First integer value for `key`, or `dflt`. Reaches into nested objects by
 * design: the reply's numbers live under "usage" and "credits", and the field
 * names are unique across the whole body. */
static long long ovf_int_field(const char *json, size_t len, const char *key,
                               long long dflt) {
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen + 3 < len; i++) {
        if (json[i] != '"' || memcmp(json + i + 1, key, klen) != 0 ||
            json[i + 1 + klen] != '"') continue;
        size_t p = i + klen + 2;
        while (p < len && (json[p] == ' ' || json[p] == '\t')) p++;
        if (p >= len || json[p] != ':') continue;
        p++;
        while (p < len && (json[p] == ' ' || json[p] == '\t')) p++;
        if (p < len && (json[p] == '-' || (json[p] >= '0' && json[p] <= '9')))
            return strtoll(json + p, NULL, 10);
        return dflt;
    }
    return dflt;
}

/* The attested platform key, cached until it expires. Refetched when missing or
 * within an hour of not_after — a key that lapses between the check and the use
 * would show up as an unexplained refusal in the middle of a busy minute. */
static struct {
    idletoken_overflow_key key;
    int                    have;
} g_ovf_key;

/* `out_unreachable` (may be NULL) separates "the platform did not answer" from
 * "the platform answered and the answer did not verify". The caller treats them
 * differently at start-up: a network that is not up yet is a reason to try
 * again later, a key that does not verify is a reason to refuse outright. */
static int ovf_ensure_key(const char *addr, int *out_unreachable,
                          char *err, size_t err_cap) {
    if (out_unreachable) *out_unreachable = 0;
    long long now = (long long)time(NULL);
    pthread_mutex_lock(&g_ovf_mu);
    int fresh = g_ovf_key.have && g_ovf_key.key.not_after > now + 3600;
    pthread_mutex_unlock(&g_ovf_mu);
    if (fresh) return 0;

    idletoken_llama_conn c;
    if (idletoken_llama_http_open(addr, "GET", IDLETOKEN_NS "/platform-key",
                                  NULL, 0, 15000, &c) != 0) {
        if (out_unreachable) *out_unreachable = 1;
        OVF_FAIL("cannot reach the platform at %s to fetch its key", addr);
    }
    int status = c.status;
    size_t rlen = 0;
    char *body = idletoken_llama_http_read_all(&c, &rlen, 64u * 1024u);
    idletoken_llama_http_close(&c);
    if (status != 200 || !body) {
        free(body);
        /* 503 is the platform's specified answer when it has no signing key
         * configured. It must never answer with an unsigned one, so this is the
         * end of the road rather than a reason to relax. A 5xx is still the
         * platform being unwell rather than lying, so it counts as unreachable
         * and does not veto the start. */
        if (out_unreachable) *out_unreachable = status >= 500 || status == 0;
        OVF_FAIL("platform returned %d for its signing-attested key", status);
    }

    const char *src = NULL;
    const uint8_t *vk = idletoken_overflow_verify_key(&src);
    idletoken_overflow_key k;
    char verr[200];
    if (idletoken_overflow_key_verify(vk, body, rlen, now, &k, verr, sizeof verr) != 0) {
        free(body);
        /* RULE 3: loud, and no fallback. Everything downstream of a bad key
         * still "works" — which is exactly the failure this refuses to have. */
        fprintf(stderr, "coord: overflow: REFUSING the platform's key: %s\n", verr);
        OVF_FAIL("platform key rejected: %s", verr);
    }
    free(body);

    pthread_mutex_lock(&g_ovf_mu);
    g_ovf_key.key = k;
    g_ovf_key.have = 1;
    pthread_mutex_unlock(&g_ovf_mu);
    fprintf(stderr, "coord: overflow: platform key %s verified against the %s "
                    "signing key, good until %lld\n",
            k.key_id[0] ? k.key_id : "(no id)", src ? src : "?", k.not_after);
    return 0;
}

void idletoken_overflow_reply_free(idletoken_overflow_reply *r) {
    if (!r) return;
    free(r->text_escaped);
    r->text_escaped = NULL;
}

int idletoken_overflow_exchange(const char *messages_json,
                                const char *model, int max_tokens,
                                idletoken_overflow_reply *out,
                                char *err, size_t err_cap) {
    if (err && err_cap) err[0] = '\0';
    if (!out || !messages_json) OVF_FAIL("nothing to forward");
    memset(out, 0, sizeof(*out));

    char url[256], api_key[192];
    pthread_mutex_lock(&g_ovf_mu);
    int on = g_ovf.on;
    snprintf(url, sizeof url, "%s", g_ovf.url);
    snprintf(api_key, sizeof api_key, "%s", g_ovf.api_key);
    pthread_mutex_unlock(&g_ovf_mu);
    if (!on) OVF_FAIL("overflow is off");

    char addr[160];
    if (ovf_url_to_addr(url, addr, sizeof addr) != 0)
        OVF_FAIL("cannot use platform URL \"%s\" (https:// needs a TLS client "
                 "this binary has not got)", url);
    if (ovf_ensure_key(addr, NULL, err, err_cap) != 0) return -1;

    idletoken_overflow_key pk;
    pthread_mutex_lock(&g_ovf_mu);
    pk = g_ovf_key.key;
    pthread_mutex_unlock(&g_ovf_mu);

    /* Anti-replay (cleanup-t9 §A1). The sealed envelope stops anyone on the way
     * from READING the prompt; it does nothing to stop them from sending the
     * same bytes a second time, and a second time is a second charge against
     * this user's credits and a second slot taken on somebody's machine. So
     * every exchange carries a fresh nonce and the time it was minted, and the
     * platform refuses a nonce it has already seen inside the window.
     *
     * Minted per call, never reused: a retry of a failed request gets a new
     * nonce and goes through, which is exactly the line between "our retry" and
     * "somebody else's copy of our request". Both fields go INSIDE the seal —
     * outside they would be an attacker's to rewrite. */
    uint8_t nonce_raw[16];
    idletoken_random_bytes(nonce_raw, sizeof nonce_raw);
    char nonce_hex[sizeof nonce_raw * 2 + 1];
    for (size_t i = 0; i < sizeof nonce_raw; i++)
        snprintf(nonce_hex + i * 2, 3, "%02x", nonce_raw[i]);
    long long issued_at = (long long)time(NULL);

    /* The plaintext, and the only place it exists outside this machine's own
     * memory is nowhere: it is sealed before the socket is opened. */
    size_t inner_cap = strlen(messages_json) + strlen(api_key) +
                       (model ? strlen(model) : 0) + 256;
    char *inner = malloc(inner_cap);
    if (!inner) OVF_FAIL("out of memory");
    int inner_len;
    if (max_tokens > 0)
        inner_len = snprintf(inner, inner_cap,
                             "{\"api_key\":\"%s\",\"model\":\"%s\","
                             "\"messages\":%s,\"max_tokens\":%d,"
                             "\"nonce\":\"%s\",\"issued_at\":%lld}",
                             api_key, model ? model : "", messages_json, max_tokens,
                             nonce_hex, issued_at);
    else
        inner_len = snprintf(inner, inner_cap,
                             "{\"api_key\":\"%s\",\"model\":\"%s\",\"messages\":%s,"
                             "\"nonce\":\"%s\",\"issued_at\":%lld}",
                             api_key, model ? model : "", messages_json,
                             nonce_hex, issued_at);
    if (inner_len < 0 || (size_t)inner_len >= inner_cap) {
        free(inner);
        OVF_FAIL("request too large to seal");
    }

    /* A reply-to key pair per request. Nothing else can open the answer, and
     * neither can a later request of ours. */
    idletoken_keypair reply_kp;
    if (idletoken_keypair_generate(&reply_kp) != IDLETOKEN_PRIV_OK) {
        idletoken_secure_zero(inner, (size_t)inner_len);
        free(inner);
        OVF_FAIL("could not generate a reply key pair");
    }

    size_t sealed_cap = (size_t)inner_len + IDLETOKEN_SEAL_OVERHEAD;
    uint8_t *sealed = malloc(sealed_cap);
    size_t sealed_len = 0;
    idletoken_priv_rc seal_rc = sealed
        ? idletoken_sodium_seal(pk.pubkey, (const uint8_t *)inner, (size_t)inner_len,
                                sealed, sealed_cap, &sealed_len)
        : IDLETOKEN_PRIV_EBUF;
    idletoken_secure_zero(inner, (size_t)inner_len);
    free(inner);
    if (seal_rc != IDLETOKEN_PRIV_OK) {
        free(sealed);
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        OVF_FAIL("could not seal the request to the platform key");
    }

    char *sealed_b64 = idletoken_b64_encode(sealed, sealed_len);
    char *reply_b64  = idletoken_b64_encode(reply_kp.pk, IDLETOKEN_PK_BYTES);
    free(sealed);
    char *body = NULL;
    int bl = 0;
    if (sealed_b64 && reply_b64) {
        size_t cap = strlen(sealed_b64) + strlen(reply_b64) + 64;
        body = malloc(cap);
        if (body)
            bl = snprintf(body, cap, "{\"sealed_request\":\"%s\",\"reply_to\":\"%s\"}",
                          sealed_b64, reply_b64);
    }
    free(sealed_b64);
    free(reply_b64);
    if (!body || bl <= 0) {
        free(body);
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        OVF_FAIL("out of memory");
    }

    /* 120 s: long enough for a borrowed machine to answer a real prompt, short
     * enough that a silent platform costs one slot rather than the machine.
     * Unbounded waiting on a remote peer is the failure this coordinator has
     * already been taken down by once. */
    idletoken_llama_conn c;
    int opened = idletoken_llama_http_open(addr, "POST", IDLETOKEN_NS "/sealed/chat",
                                           body, (size_t)bl, 120000, &c);
    free(body);
    if (opened != 0) {
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        OVF_FAIL("platform %s did not answer", addr);
    }
    int status = c.status;
    size_t rlen = 0;
    char *resp = idletoken_llama_http_read_all(&c, &rlen, 8u * 1024u * 1024u);
    idletoken_llama_http_close(&c);
    if (status != 200 || !resp) {
        free(resp);
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        /* The platform's own body is deliberately not quoted: it may describe
         * the platform's insides, and it is not the local caller's business. */
        OVF_FAIL("platform answered %d", status);
    }

    const char *sr = NULL;
    size_t sr_len = 0;
    if (ovf_str_span(resp, rlen, "sealed_response", &sr, &sr_len) != 0) {
        free(resp);
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        OVF_FAIL("platform reply carried no sealed_response");
    }
    size_t enc_len = 0;
    uint8_t *enc = idletoken_b64_decode(sr, sr_len, &enc_len);
    free(resp);
    if (!enc || enc_len < IDLETOKEN_SEAL_OVERHEAD) {
        free(enc);
        idletoken_secure_zero(&reply_kp, sizeof reply_kp);
        OVF_FAIL("platform reply is not a sealed envelope");
    }
    uint8_t *plain = malloc(enc_len);
    size_t plain_len = 0;
    idletoken_priv_rc orc = plain
        ? idletoken_sodium_seal_open(&reply_kp, enc, enc_len, plain, enc_len, &plain_len)
        : IDLETOKEN_PRIV_EBUF;
    free(enc);
    idletoken_secure_zero(&reply_kp, sizeof reply_kp);
    if (orc != IDLETOKEN_PRIV_OK) {
        free(plain);
        OVF_FAIL("the platform's answer would not open (wrong key or corrupt)");
    }

    /* No "text" means the platform sealed an error back instead of an answer
     * (insufficient credits, a key past its cap, moderation). That is still a
     * failure for the local caller: it gets the ordinary 429, and only the log
     * learns which one. Handing "insufficient credits" back as the reply to a
     * chat request would make a billing problem look like the model talking. */
    const char *txt = NULL;
    size_t txt_len = 0;
    if (ovf_str_span((const char *)plain, plain_len, "text", &txt, &txt_len) != 0) {
        const char *etype = NULL, *emsg = NULL;
        size_t et_len = 0, em_len = 0;
        if (ovf_str_span((const char *)plain, plain_len, "type", &etype, &et_len) == 0 &&
            ovf_str_span((const char *)plain, plain_len, "message", &emsg, &em_len) == 0) {
            fprintf(stderr, "coord: overflow: the platform refused: %.*s - %.*s\n",
                    (int)et_len, etype, (int)em_len, emsg);
            idletoken_secure_zero(plain, plain_len);
            free(plain);
            OVF_FAIL("platform refused the request (%.*s)", (int)et_len, etype);
        }
        idletoken_secure_zero(plain, plain_len);
        free(plain);
        OVF_FAIL("the platform's answer carried neither text nor an error");
    }
    out->text_escaped = malloc(txt_len + 1);
    if (!out->text_escaped) {
        free(plain);
        OVF_FAIL("out of memory");
    }
    memcpy(out->text_escaped, txt, txt_len);
    out->text_escaped[txt_len] = '\0';
    out->in_tokens     = (int)ovf_int_field((const char *)plain, plain_len, "input_tokens", 0);
    out->out_tokens    = (int)ovf_int_field((const char *)plain, plain_len, "output_tokens", 0);
    out->charged_milli = ovf_int_field((const char *)plain, plain_len, "charged_milli", 0);
    idletoken_secure_zero(plain, plain_len);
    free(plain);

    /* Count the platform's OWN figure, not a local estimate. When the two
     * ledgers disagree the platform's is the real one; this one is a brake. */
    idletoken_overflow_note_spend(out->charged_milli);

    long long spent = 0, cap = 0;
    idletoken_overflow_spend_today(&spent, &cap);
    fprintf(stderr, "coord: overflow: borrowed %d in / %d out tok for %lld "
                    "milli-credits (%lld/%lld today)\n",
            out->in_tokens, out->out_tokens, out->charged_milli, spent, cap);
    return 0;
}

/* --- self-test ------------------------------------------------------------
 *
 * The bad samples are the same three shapes the platform's own G_SEALED_INTAKE
 * suite builds: a substituted key, a tampered expiry, and a signature made
 * without the domain prefix. Each is asserted to be refused FOR ITS OWN REASON
 * — "rejected" alone would still pass if the verifier had degenerated into
 * refusing everything, and a verifier that refuses everything is as broken as
 * one that accepts everything (it takes overflow off the air silently). */

static char *ovf_sign_response(const uint8_t sk[64], const char *pubkey_b64,
                               const char *not_after, const char *sign_prefix) {
    /* sign_prefix: the domain string to sign under. Passing "" produces the
     * "signature over the bare fields" bad sample. */
    char msg[320];
    int msg_len = snprintf(msg, sizeof msg, "%s%s\n%s",
                           sign_prefix[0] ? sign_prefix : "",
                           pubkey_b64, not_after);
    if (msg_len < 0) return NULL;

    uint8_t sm[512];
    unsigned long long sm_len = 0;
    if ((size_t)msg_len + IDLETOKEN_OVF_SIGN_BYTES > sizeof sm) return NULL;
    crypto_sign(sm, &sm_len, (const uint8_t *)msg, (unsigned long long)msg_len, sk);
    char *sig_b64 = idletoken_b64_encode(sm, IDLETOKEN_OVF_SIGN_BYTES);
    if (!sig_b64) return NULL;

    char *json = malloc(1024);
    if (json) {
        snprintf(json, 1024,
                 "{\"alg\":\"x25519-sealedbox\",\"sign_alg\":\"ed25519\","
                 "\"pubkey\":\"%s\",\"not_after\":\"%s\","
                 "\"key_id\":\"abc123\",\"sig\":\"%s\"}",
                 pubkey_b64, not_after, sig_b64);
    }
    free(sig_b64);
    return json;
}

int idletoken_overflow_selftest(void) {
    int fails = 0;
#define OST(cond, name) do { \
        if (cond) fprintf(stderr, "selftest PASS %s\n", name); \
        else      { fprintf(stderr, "selftest FAIL %s\n", name); fails++; } \
    } while (0)

    /* A signing key pair for the "platform", and a second one standing in for
     * an attacker who substitutes their own. */
    uint8_t sign_pk[32], sign_sk[64], evil_pk[32], evil_sk[64];
    crypto_sign_keypair(sign_pk, sign_sk);
    crypto_sign_keypair(evil_pk, evil_sk);

    uint8_t enc_pk[IDLETOKEN_OVF_PK_BYTES];
    for (int i = 0; i < IDLETOKEN_OVF_PK_BYTES; i++) enc_pk[i] = (uint8_t)(i * 7 + 1);
    char *enc_b64 = idletoken_b64_encode(enc_pk, sizeof enc_pk);

    const long long now = 1755000000LL;                  /* 2025-08-12, fixed */
    const char *future  = "2030-01-01T00:00:00.000Z";
    const char *past    = "2020-01-01T00:00:00.000Z";
    char err[200];
    idletoken_overflow_key got;

    /* Positive control. Without it, all six negatives below would also pass on
     * a verifier that rejects unconditionally. */
    char *good = ovf_sign_response(sign_sk, enc_b64, future, OVF_KEY_DOMAIN "\n");
    OST(good && idletoken_overflow_key_verify(sign_pk, good, strlen(good), now,
                                              &got, err, sizeof err) == 0,
        "overflow: a correctly signed platform key is accepted");
    OST(good && memcmp(got.pubkey, enc_pk, sizeof enc_pk) == 0,
        "overflow: the accepted key is the one that was signed");

    /* Bad sample 1 — a substituted key: correctly signed, by the wrong signer.
     * This is the whole attack the pin exists to stop. */
    char *swapped = ovf_sign_response(evil_sk, enc_b64, future, OVF_KEY_DOMAIN "\n");
    OST(swapped && idletoken_overflow_key_verify(sign_pk, swapped, strlen(swapped),
                                                 now, &got, err, sizeof err) == -1 &&
        strstr(err, "does not verify") != NULL,
        "overflow: a key signed by someone else is refused");

    /* Bad sample 2a — the expiry edited after signing. */
    char *tampered = ovf_sign_response(sign_sk, enc_b64, future, OVF_KEY_DOMAIN "\n");
    if (tampered) {
        char *p = strstr(tampered, future);
        if (p) memcpy(p, "2031-01-01T00:00:00.000Z", strlen(future));
    }
    OST(tampered && idletoken_overflow_key_verify(sign_pk, tampered, strlen(tampered),
                                                  now, &got, err, sizeof err) == -1 &&
        strstr(err, "does not verify") != NULL,
        "overflow: an edited not_after breaks the signature");

    /* Bad sample 2b — honestly signed, but expired. Without this, not_after
     * would be a field that is merely carried rather than obeyed. */
    char *expired = ovf_sign_response(sign_sk, enc_b64, past, OVF_KEY_DOMAIN "\n");
    OST(expired && idletoken_overflow_key_verify(sign_pk, expired, strlen(expired),
                                                 now, &got, err, sizeof err) == -1 &&
        strstr(err, "expired") != NULL,
        "overflow: an expired platform key is refused");

    /* Bad sample 3 — signed without the domain prefix. */
    char *nodomain = ovf_sign_response(sign_sk, enc_b64, future, "");
    OST(nodomain && idletoken_overflow_key_verify(sign_pk, nodomain, strlen(nodomain),
                                                  now, &got, err, sizeof err) == -1 &&
        strstr(err, "does not verify") != NULL,
        "overflow: a signature without the domain prefix is refused");

    /* An unsigned response — the shape the platform is specified never to
     * produce (it answers 503 instead), and the one that would quietly end
     * verification if it were ever accepted. */
    {
        char unsigned_json[512];
        snprintf(unsigned_json, sizeof unsigned_json,
                 "{\"alg\":\"x25519-sealedbox\",\"sign_alg\":\"ed25519\","
                 "\"pubkey\":\"%s\",\"not_after\":\"%s\",\"key_id\":\"abc123\"}",
                 enc_b64 ? enc_b64 : "", future);
        OST(idletoken_overflow_key_verify(sign_pk, unsigned_json, strlen(unsigned_json),
                                          now, &got, err, sizeof err) == -1 &&
            strstr(err, "no signature") != NULL,
            "overflow: an unsigned platform key is refused");
    }

    /* An algorithm nobody here implements. */
    {
        char other_alg[512];
        snprintf(other_alg, sizeof other_alg,
                 "{\"alg\":\"rsa-oaep\",\"sign_alg\":\"ed25519\","
                 "\"pubkey\":\"%s\",\"not_after\":\"%s\",\"sig\":\"AA==\"}",
                 enc_b64 ? enc_b64 : "", future);
        OST(idletoken_overflow_key_verify(sign_pk, other_alg, strlen(other_alg),
                                          now, &got, err, sizeof err) == -1 &&
            strstr(err, "unsupported alg") != NULL,
            "overflow: an algorithm we do not implement is refused");
    }

    /* A build with no pin trusts nothing, rather than trusting the network. */
    OST(idletoken_overflow_key_verify(NULL, good ? good : "{}",
                                      good ? strlen(good) : 2, now,
                                      &got, err, sizeof err) == -1 &&
        strstr(err, "pinned no platform signing key") != NULL,
        "overflow: an unpinned build accepts no platform key at all");

    /* not_after has to be UTC. A local-time spelling would be read hours off
     * the moment that was actually signed. */
    OST(iso8601_utc_to_unix("2030-01-01T00:00:00.000Z") == 1893456000LL,
        "overflow: ISO-8601 UTC parses to the right instant");
    OST(iso8601_utc_to_unix("2030-01-01T00:00:00+08:00") == -1,
        "overflow: a non-UTC not_after is refused");
    OST(iso8601_utc_to_unix("tomorrow") == -1,
        "overflow: a non-date not_after is refused");

    /* ---- RULE 2: what "fail closed" refuses ------------------------------
     * These run before the pin check inside configure(), so they hold in a
     * pinned and an unpinned build alike. */
    {
        /* 127.0.0.1:1 refuses instantly, so the "unreachable platform" branch
         * is exercised without a DNS lookup or a timeout in a unit test. */
        idletoken_overflow_cfg cfg = { "http://127.0.0.1:1", "sk-test",
                                       0, 0, /*api_token_set=*/0 };
        OST(idletoken_overflow_configure(&cfg, err, sizeof err) == -1 &&
            strstr(err, "local API token") != NULL,
            "overflow: without a local API token, overflow refuses to switch on");
        OST(!idletoken_overflow_enabled(),
            "overflow: a refused configure leaves it OFF, not half on");

        cfg.api_token_set = 1;
        cfg.url = "";
        OST(idletoken_overflow_configure(&cfg, err, sizeof err) == -1 &&
            strstr(err, "platform URL") != NULL,
            "overflow: no platform URL is refused");

        cfg.url = "http://127.0.0.1:1";
        cfg.api_key = "";
        OST(idletoken_overflow_configure(&cfg, err, sizeof err) == -1 &&
            strstr(err, "account key") != NULL,
            "overflow: no account key is refused");

        /* RULE 3: with everything else in order, switching on must succeed
         * exactly when this build has a trust anchor — and must fail when it
         * has none. Written as an equivalence so the assertion is real in both
         * a pinned release build and an unpinned development one. */
        cfg.api_key = "sk-test";
        int have_anchor = idletoken_overflow_verify_key(NULL) != NULL;
        int cfg_ok = idletoken_overflow_configure(&cfg, err, sizeof err) == 0;
        OST(cfg_ok == have_anchor,
            "overflow: switching on succeeds exactly when a platform signing key is pinned");
        OST(cfg_ok || strstr(err, "pinned no platform signing key") != NULL,
            "overflow: an unpinned build says so instead of sealing to anything");

        /* https:// has no client in this binary. Refused rather than quietly
         * spoken to in the clear on port 443, which would be the worst of the
         * two (it would LOOK encrypted in the configuration). */
        cfg.url = "https://platform.example";
        OST(idletoken_overflow_configure(&cfg, err, sizeof err) == -1 &&
            strstr(err, "TLS client") != NULL,
            "overflow: an https:// platform URL is refused, not downgraded");
    }

    /* ---- the forwarding decision ----------------------------------------
     * Driven by writing the state directly rather than through configure(), so
     * the policy is tested on both kinds of build. */
    {
        const char *why = NULL;
        g_ovf.on = 1;
        g_ovf.wait_ms = 0;
        g_ovf.daily_cap_milli = 1000;
        g_ovf.spent_milli = 0;
        g_ovf.day = ovf_today();

        /* Positive control first: without it every "refused" below would also
         * hold on a predicate that returns 0 unconditionally — which is what a
         * silently disabled feature looks like. */
        OST(idletoken_overflow_should_forward(0, 0, 0, &why) == 1,
            "overflow: a local non-streaming request on a full machine forwards");

        /* RULE 1, the one that costs the most if it inverts. */
        OST(idletoken_overflow_should_forward(1, 0, 0, &why) == 0 &&
            strstr(why, "platform work") != NULL,
            "overflow: platform-dispatched work is never forwarded");
        g_ovf.wait_ms = 60000;
        OST(idletoken_overflow_should_forward(1, 0, 999999, &why) == 0,
            "overflow: no threshold or estimate makes platform work forwardable");
        g_ovf.wait_ms = 0;

        OST(idletoken_overflow_should_forward(0, 1, 0, &why) == 0 &&
            strstr(why, "stream") != NULL,
            "overflow: a streaming request is refused here, not relayed");

        /* The threshold: "only bother paying if I would have waited a while". */
        g_ovf.wait_ms = 5000;
        OST(idletoken_overflow_should_forward(0, 0, 4999, &why) == 0 &&
            strstr(why, "threshold") != NULL,
            "overflow: an estimated wait under the threshold stays local");
        OST(idletoken_overflow_should_forward(0, 0, 5000, &why) == 1,
            "overflow: at the threshold it forwards");
        g_ovf.wait_ms = 0;

        /* The daily ceiling, and that it is a ceiling on the DAY. */
        idletoken_overflow_note_spend(999);
        OST(idletoken_overflow_should_forward(0, 0, 0, &why) == 1,
            "overflow: under the daily cap it still forwards");
        idletoken_overflow_note_spend(1);
        OST(idletoken_overflow_should_forward(0, 0, 0, &why) == 0 &&
            strstr(why, "daily spend cap") != NULL,
            "overflow: at the daily cap it stops forwarding");
        g_ovf.day -= 1;          /* pretend the UTC day turned over */
        OST(idletoken_overflow_should_forward(0, 0, 0, &why) == 1,
            "overflow: the cap resets when the UTC day turns over");

        g_ovf.on = 0;
        OST(idletoken_overflow_should_forward(0, 0, 999999, &why) == 0 &&
            strstr(why, "off") != NULL,
            "overflow: switched off, nothing forwards");
        memset(&g_ovf, 0, sizeof g_ovf);
    }

    free(good); free(swapped); free(tampered); free(expired); free(nodomain);
    free(enc_b64);
#undef OST
    return fails;
}
