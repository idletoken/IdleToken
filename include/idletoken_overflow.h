/* IdleToken — overflow routing: "this machine is full, borrow someone else's".
 *
 * Design: docs/overflow-routing-design.md + docs/api-surface.md §5.
 * Plan:   docs/overflow-b2b-plan-2026-08.md.
 *
 * THREE RULES THAT MAY NOT BE VIOLATED (plan §1). They are repeated at the code
 * that implements each one, because a rule stated only in a document is a rule
 * that gets refactored away.
 *
 *   1. A job the PLATFORM dispatched has exactly two endings: finished here, or
 *      refused here with a 429. It is never forwarded back out. That single
 *      rule kills every loop class — provider ping-pong, and a request coming
 *      back around to the cluster that sent it — with no hop counter.
 *      The predicate's direction is "no marker = local", and writing it the
 *      other way round disables the whole feature silently (design §3): a curl
 *      or a Claude Code on the LAN has no reason to send our private header,
 *      so "unknown origin = do not forward" would refuse to forward anything.
 *
 *   2. FAIL CLOSED. Overflow may only be enabled on a coordinator that has an
 *      --api-token. api_token_ok() lets everything through when no token is
 *      configured, and a coordinator that can spend credits by itself plus an
 *      open door means anyone on the LAN can drain the balance (api-surface
 *      §5.3). Not a warning — a refusal to start.
 *
 *   3. NO SILENT FALLBACK. Signature verification failing, an envelope that
 *      will not open, a platform that cannot be reached: each is reported
 *      loudly and answered with a local 429. Never plaintext on the wire, never
 *      a quietly swallowed error.
 *
 * C99. */

#ifndef IDLETOKEN_OVERFLOW_H
#define IDLETOKEN_OVERFLOW_H

#include <stddef.h>
#include <stdint.h>

#define IDLETOKEN_OVF_PK_BYTES   32   /* X25519 public key (sealed box recipient) */
#define IDLETOKEN_OVF_SIGN_BYTES 64   /* ed25519 detached signature */
#define IDLETOKEN_OVF_KEYID_CAP  32   /* short fingerprint, log/UI only */

/* --- the pinned trust anchor ----------------------------------------------
 *
 * The coordinator has NO TLS client (api-surface §5.1b), so it fetches the
 * platform's X25519 encryption key over plaintext HTTP. Fetching an encryption
 * key in the clear makes the encryption worth nothing on its own: an active
 * attacker on the path swaps in their own key, opens everything the coordinator
 * seals, re-seals it to the real key, and NOTHING anywhere reports an error.
 * The overflow API key travels inside that same envelope, so the same failure
 * hands over the credentials as well.
 *
 * So the encryption key is not trusted for being reachable — it is trusted for
 * being signed by an ed25519 key compiled into this binary, exactly as the
 * client's self-updater trusts its minisign key. The pin has to live in the
 * ENGINE: sealing happens in the coordinator, and a copy pinned in the client
 * would not be on the path at all (the coordinator is often headless).
 *
 * Build with -DIDLETOKEN_PLATFORM_VERIFY_KEY_B64='"<base64>"' to pin one.
 * An unpinned build refuses to enable overflow rather than trusting whatever
 * the network hands it. */
#ifndef IDLETOKEN_PLATFORM_VERIFY_KEY_B64
#define IDLETOKEN_PLATFORM_VERIFY_KEY_B64 ""
#endif

/* The ed25519 verify key this build trusts, or NULL when the build pinned none
 * and the loud IDLETOKEN_PLATFORM_VERIFY_KEY escape hatch is unset.
 * `out_source` (may be NULL) is set to a short word naming where it came from
 * ("pinned" / "env") so callers can say so in a log line. */
const uint8_t *idletoken_overflow_verify_key(const char **out_source);

/* An attested platform encryption key, once it has survived verification. */
typedef struct {
    uint8_t   pubkey[IDLETOKEN_OVF_PK_BYTES];
    char      key_id[IDLETOKEN_OVF_KEYID_CAP];
    long long not_after;   /* unix seconds, from the SIGNED not_after field */
} idletoken_overflow_key;

/* Verify one `GET /idletoken/v1/platform-key` response body.
 *
 * Checks, in this order, and every one of them can reject:
 *   - alg == "x25519-sealedbox" and sign_alg == "ed25519" (an algorithm we do
 *     not implement must not be waved through as if it were the one we do);
 *   - pubkey decodes to 32 bytes, sig to 64;
 *   - the ed25519 signature over the DOMAIN-SEPARATED canonical string
 *     "idletoken-platform-key-v1\n<pubkey_b64>\n<not_after>", rebuilt here from
 *     the response's own fields. The prefix is what stops a signature made for
 *     some other purpose from counting here (and vice versa); because we always
 *     prepend it, a signature computed over the bare "pubkey\nnot_after" fails;
 *   - not_after parses as UTC ISO-8601 and is still in the future at `now`.
 *     not_after is inside the signature on purpose: with no revocation list it
 *     is the only way to retire a key, and an expiry the signature does not
 *     cover is not an expiry at all.
 *
 * `verify_pk` is passed in rather than read from the pin so the self-test can
 * drive it with a key pair it generated — a verifier that can only be called
 * one way is a verifier whose failure paths never get exercised.
 *
 * Returns 0 and fills *out on success. On rejection returns -1 and writes a
 * one-line reason into `err` (which names WHICH check failed; "signature
 * verification failed" for all three bad-sample shapes would make the gate
 * unable to tell them apart). */
int idletoken_overflow_key_verify(const uint8_t verify_pk[IDLETOKEN_OVF_PK_BYTES],
                                  const char *json, size_t json_len,
                                  long long now_unix,
                                  idletoken_overflow_key *out,
                                  char *err, size_t err_cap);

/* --- configuration --------------------------------------------------------
 *
 * Overflow is off unless configured, and configuring it can fail. Everything
 * that must hold before this coordinator is allowed to spend money by itself is
 * checked once, here, rather than at the moment of the first burst. */

/* Requests are refused past this many milli-credits of platform spend in one
 * UTC day. A daily ceiling is NOT optional (api-surface §5.4): "forward to the
 * platform when busy" means "spend money without being asked", and one runaway
 * benchmark overnight would empty the balance before anyone looked.
 *
 * 50 credits/day, raised from 5 on 2026-08-19 once there were measured prices
 * to judge it against (platform/pricing/anchor-proposal-v1.md, decision D2).
 * The old 5 was picked when nothing could say how much a credit buys; against
 * the measured rate card it bought ONE 27B-class conversation and not even half
 * a Claude Code turn, i.e. a user who switched overflow on got a feature that
 * looked broken. 50 buys roughly ten 27B-class conversations or 3-5 Claude Code
 * turns and is still a guardrail: a machine that shares for a day earns ~5800.
 *
 * Prices are set per deployment, so no number here is universal — this one is
 * anchored to OUR rate card. It stays deliberately finite: the switch shows it
 * and the user can raise it, and a default that surprises someone by refusing
 * is recoverable in a way that a default which surprises them by spending
 * is not. */
#define IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI 50000

typedef struct {
    const char *url;              /* platform base URL or host:port */
    const char *api_key;          /* the account's overflow key; sealed, never a header */
    long long   wait_ms;          /* forward only when the estimated wait is at least this */
    long long   daily_cap_milli;  /* <= 0 means the default above; there is no "off" */
    int         api_token_set;    /* does this coordinator have an --api-token? (RULE 2) */
} idletoken_overflow_cfg;

/* Turn overflow on. Returns 0, or -1 with a reason in `err` and overflow left
 * OFF. Refuses when: no --api-token (RULE 2 — an open local API plus automatic
 * spending is a drainable balance); no platform URL or key; or this build
 * pinned no platform signing key and none was supplied (RULE 3 — without the
 * anchor the envelope protects nothing). */
int idletoken_overflow_configure(const idletoken_overflow_cfg *cfg,
                                 char *err, size_t err_cap);

int idletoken_overflow_enabled(void);

/* Milli-credits spent today, and the ceiling in force (either may be NULL). */
void idletoken_overflow_spend_today(long long *spent_milli, long long *cap_milli);

/* Record what the platform said it charged. Called with the figure out of the
 * sealed reply, so the local ledger is the platform's own number rather than a
 * second estimate — when the two disagree the platform's is the real one and
 * this one is only a brake. */
void idletoken_overflow_note_spend(long long milli);

/* May THIS request be forwarded, right now?
 *
 * `*why` (may be NULL) always receives a short phrase naming the deciding
 * reason, including on the yes path, so the log line says why rather than only
 * what. The order of the checks is the order of the rules:
 *
 *   RULE 1  from_platform      -> never. Not a threshold, not a setting.
 *   §5.4    want_stream        -> never; refuse locally with an estimate
 *                                 instead. Handing back half a stream whose
 *                                 upstream died has no defined meaning yet.
 *   §5.4    daily cap reached  -> never, until the UTC day turns over.
 *   §4      est_wait < wait_ms -> not yet: the user asked to wait this long
 *                                 before paying someone else.
 *
 * est_wait is the same queue-depth x service-time estimate that already goes
 * into the 429's X-IdleToken-Est-Wait-Ms header. GPU utilisation and free VRAM
 * are deliberately NOT consulted (design §4): neither has a stable relationship
 * with "how long will this request wait". */
int idletoken_overflow_should_forward(int from_platform, int want_stream,
                                      long long est_wait_ms, const char **why);

/* --- the sealed exchange ---------------------------------------------------
 *
 * What actually leaves the machine, and the shape it leaves in:
 *
 *   POST {platform}/idletoken/v1/sealed/chat
 *     { sealed_request: b64(crypto_box_seal(inner, platform_pk)),
 *       reply_to:       b64(a fresh X25519 public key) }
 *   inner = { api_key, model, messages:[{role,content}], max_tokens? }
 *   <- 200 { sealed_response: b64(crypto_box_seal(json, reply_to)) }
 *
 * Byte-for-byte the envelope the platform agent already opens in the other
 * direction, and the same libsodium construction — no new crypto is written for
 * this, only a second caller of what exists.
 *
 * The account key travels INSIDE the envelope, never as a header: it is the
 * credential, and the transport is plaintext HTTP. Which is also why the
 * signature check on the platform key is this function's security premise
 * rather than a nicety — seal to a substituted key and the key goes with it.
 *
 * The reply-to key pair is fresh per request and lives only for that request:
 * the answer is unopenable by anyone else, including a later request of ours.
 */
typedef struct {
    /* The assistant's reply, as the RAW still-JSON-escaped span from the sealed
     * body — handed on to the response builder as-is. Unescaping it here only
     * to re-escape it there would be two chances to disagree about \uXXXX for
     * no gain. malloc'd; free with idletoken_overflow_reply_free. */
    char     *text_escaped;
    int       in_tokens;
    int       out_tokens;
    long long charged_milli;
} idletoken_overflow_reply;

void idletoken_overflow_reply_free(idletoken_overflow_reply *r);

/* Seal one request to the platform and open its answer.
 *
 * `messages_json` is a complete JSON array — `[{"role":"user","content":"..."}]`
 * with the content already escaped — because building it means understanding
 * OpenAI and Anthropic request bodies, which is the coordinator's job, not this
 * module's. Here it is transport and crypto only.
 *
 * Returns 0 with `*out` filled, or -1 with a one-line reason in `err`. EVERY
 * failure lands here: a platform 4xx/5xx, an unreachable host, an envelope that
 * will not open, an error the platform sealed back. The caller answers the
 * local client with its ordinary 429 in all of those cases — "this machine is
 * busy and could not borrow one" is the honest local meaning, and the
 * platform's own error text is never passed through, since it may describe the
 * platform's insides rather than anything the caller did.
 *
 * On success the charge the platform REPORTED is added to the day's spend, so
 * the local ceiling counts the same currency the platform bills in. */
int idletoken_overflow_exchange(const char *messages_json,
                                const char *model, int max_tokens,
                                idletoken_overflow_reply *out,
                                char *err, size_t err_cap);

/* Self-test for everything above: builds its own signing key pair, signs a good
 * sample, and asserts the good one passes and each bad one is refused for the
 * stated reason. Returns the number of FAILED assertions (0 = all pass); prints
 * one PASS/FAIL line each, in coord --selftest's format. */
int idletoken_overflow_selftest(void);

#endif /* IDLETOKEN_OVERFLOW_H */
