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

/* No coordinator-side daily ceiling by default. The account's Spark balance is
 * the product's spend gate: while it is positive, an eligible busy request may
 * keep borrowing. Operators can still set a positive --overflow-daily-cap as
 * an explicit local stop-loss; zero means no additional local ceiling. */
#define IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI 0

typedef struct {
    const char *url;              /* platform base URL or host:port */
    const char *api_key;          /* the account's overflow key; sealed, never a header */
    long long   wait_ms;          /* forward only when the estimated wait is at least this */
    long long   daily_cap_milli;  /* > 0 explicit stop-loss; <= 0 means no local cap */
    int         api_token_set;    /* retained config field; tokenless loopback is supported */
} idletoken_overflow_cfg;

/* Turn overflow on. Returns 0, or -1 with a reason in `err` and overflow left
 * OFF. Refuses when there is no platform URL/key, or this build pinned no
 * platform signing key and none was supplied (without the trust anchor the
 * envelope protects nothing). A local API token is optional: the API is
 * loopback-only and browser Origin requests are rejected. */
int idletoken_overflow_configure(const idletoken_overflow_cfg *cfg,
                                 char *err, size_t err_cap);

int idletoken_overflow_enabled(void);

/* Where borrowed requests are sent, verbatim, or "" when overflow is off.
 *
 * Exposed so the coordinator can DISCLOSE it rather than only dial it. A
 * borrowed prompt is decrypted by whatever is at this URL and then by whichever
 * provider that endpoint chooses (PRIV-01/PRIV-02), and when the URL is not the
 * IdleToken platform the user's mental model of "still inside IdleToken's
 * envelope" is simply wrong (PRIV-08). The client can only show that if the
 * engine will say it; and saying it here, from the value that is actually
 * dialled, is the difference between a disclosure and a caption.
 *
 * Returns 0 on success. Note the honest scope: this is the OFFICIAL
 * coordinator answering about itself. A modified one can answer anything, which
 * is why HOST-06 stays open — see docs/modified-client-threat-register-2026-08.md. */
int idletoken_overflow_endpoint(char *out, size_t cap);

/* Milli-credits spent today, and the ceiling in force (either may be NULL). */
void idletoken_overflow_spend_today(long long *spent_milli, long long *cap_milli);

/* Record what the platform said it charged. Called with the figure out of the
 * sealed reply, so the local ledger is the platform's own number rather than a
 * second estimate — when the two disagree the platform's is the real one and
 * this one is only a brake. */
void idletoken_overflow_note_spend(long long milli);

/* --- where a request came from --------------------------------------------
 *
 * RULE 1 used to read one header and call the answer a boolean. It is not a
 * boolean, because the header is set by the sender, and one of the senders is
 * an agent running on a provider's own machine (threat register PROV-28): an
 * agent that simply omits the marker had every dispatched job treated as local
 * work and forwarded on, spending a second fee and showing the prompt to one
 * more stranger. Four states, not two, so the difference between "proven" and
 * "merely claimed" and "not said at all" survives to the decision. */
typedef enum {
    /* Attributed to a caller on this machine — either it presented the
     * coordinator's local-origin marker, or the machine is not a provider and
     * "no marker" therefore has only one possible meaning. */
    IDLETOKEN_ORIGIN_LOCAL = 0,
    /* Spent a single-use capability this coordinator minted (idletoken_admission.h).
     * The only state that CANNOT be produced by editing the sender. */
    IDLETOKEN_ORIGIN_PLATFORM_PROVEN,
    /* Sent the legacy X-IdleToken-Origin header and nothing else. Still treated
     * as platform work — believing a claim in the SAFE direction costs nothing,
     * and refusing it would break every agent older than this change. */
    IDLETOKEN_ORIGIN_PLATFORM_CLAIMED,
    /* No marker at all, on a machine that is also serving the platform. This is
     * exactly the shape of a stripped header, and it is exactly the shape of an
     * ordinary curl. Which is why it is its own state: under the strict policy
     * it is not forwardable, and the log says so in those words. */
    IDLETOKEN_ORIGIN_UNATTRIBUTED
} idletoken_origin;

const char *idletoken_origin_name(idletoken_origin o);

/* How much of the above the coordinator acts on.
 *
 * LEGACY reproduces the pre-2026-08-30 behaviour exactly (no marker = local =
 * forwardable). It exists so the gate can demonstrate the vulnerability it is
 * closing on the same binary that closes it — an attack oracle that has to be
 * run against an older build is an oracle nobody runs.
 *
 * CAPABILITY recognises and consumes capabilities but still forwards
 * unattributed loopback work. It is the product default in both private and
 * shared mode: OpenAI/Anthropic compatibility means third-party local clients
 * cannot be required to read an IdleToken state file and add a private header.
 * Proven platform work, legacy-marked platform work, a platform job already in
 * flight, and a spent hop budget remain independently non-forwardable.
 *
 * STRICT additionally refuses to forward unattributed work. It is an explicit
 * operator opt-in for a provider that accepts the trade-off: ordinary Claude
 * Code, Codex, Nimbalyst and curl requests receive 429 instead of borrowing
 * while the local slot is busy. */
typedef enum {
    IDLETOKEN_OVF_ORIGIN_LEGACY = 0,
    IDLETOKEN_OVF_ORIGIN_CAPABILITY,
    IDLETOKEN_OVF_ORIGIN_STRICT
} idletoken_ovf_policy;

void idletoken_overflow_set_policy(idletoken_ovf_policy p);
idletoken_ovf_policy idletoken_overflow_policy(void);
const char *idletoken_overflow_policy_name(idletoken_ovf_policy p);

/* How many times one request may be handed on before somebody refuses.
 *
 * RULE 1 kills the loop that goes through the platform and back. It does NOT
 * kill a chain that grows one machine at a time — A borrows from the platform,
 * the platform dispatches to B, B is full and borrows again — because from B's
 * side the job is platform work and RULE 1 already stops it, but only while B
 * can TELL. A hop budget is the belt to that brace: it does not depend on
 * anyone's honesty about origin, only on a counter that travels with the
 * request, and one hop is all the feature was ever specified to need. */
#define IDLETOKEN_OVF_MAX_HOPS 1

/* May THIS request be forwarded, right now?
 *
 * `*why` (may be NULL) always receives a short phrase naming the deciding
 * reason, including on the yes path, so the log line says why rather than only
 * what. The order of the checks is the order of the rules:
 *
 *   RULE 1  platform work        -> never. Not a threshold, not a setting.
 *   RULE 1b unattributed, strict  -> never. This opt-in also disables overflow
 *                                    for ordinary third-party API clients.
 *   RULE 1c a platform job is in flight -> never: a machine in the middle of
 *                                    somebody else's paid job must not pay a
 *                                    third machine for what may be the same
 *                                    prompt (fee expansion, CHAIN-05).
 *   hops    already forwarded once -> never; the chain ends here.
 *   §5.4    explicit cap reached   -> never, until the UTC day turns over.
 *   §4      est_wait < wait_ms     -> not yet: the user asked to wait this long
 *                                    before paying someone else.
 *
 * stream requests are eligible: the coordinator obtains the complete sealed
 * answer before opening the local SSE response, so an upstream failure still
 * produces an ordinary 429 and a success is re-emitted as a complete
 * OpenAI/Anthropic event sequence.
 *
 * est_wait is the same queue-depth x service-time estimate that already goes
 * into the 429's X-IdleToken-Est-Wait-Ms header. GPU utilisation and free VRAM
 * are deliberately NOT consulted (design §4): neither has a stable relationship
 * with "how long will this request wait". */
int idletoken_overflow_should_forward(idletoken_origin origin, int want_stream,
                                      long long est_wait_ms, int hops_in,
                                      const char **why);

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
    /* Optional OpenAI tool_calls array, brackets included, copied verbatim
     * from the sealed reply. NULL means this was a text-only turn. */
    char     *tool_calls_json;
    /* OpenAI finish_reason. Empty only for an old platform response; callers
     * then derive the historical default from tool_calls/text. */
    char      finish_reason[24];
    int       in_tokens;
    int       out_tokens;
    long long charged_milli;
} idletoken_overflow_reply;

void idletoken_overflow_reply_free(idletoken_overflow_reply *r);

/* Seal one request to the platform and open its answer.
 *
 * `messages_json` is a complete JSON array — `[{"role":"user","content":"..."}]`
 * with the content already escaped. `tools_json` and `tool_choice_json` are
 * optional complete JSON values from the same normalized OpenAI request.
 * `model` and `quant` are the exact service identity this coordinator is
 * running; carrying both prevents the platform from applying the public
 * bare-name quality floor to a user's explicitly selected lower precision.
 * The explicit lengths let callers pass spans without copying them. Building
 * these values means understanding API request bodies, which is the
 * coordinator's job, not this module's; here they are transport and crypto
 * only.
 *
 * Returns IDLETOKEN_OVF_EXCHANGE_OK with `*out` filled,
 * IDLETOKEN_OVF_EXCHANGE_REFUSED when the trusted platform sealed back a
 * caller-actionable refusal, or IDLETOKEN_OVF_EXCHANGE_ERROR for transport,
 * crypto and malformed-response failures. On REFUSED, `err` contains one of a
 * small allowlisted set of stable reason codes, never the platform's free-form
 * message. That lets the caller report quota/request failures honestly without
 * leaking platform internals; ERROR still becomes the ordinary local busy 429.
 *
 * On success the charge the platform REPORTED is added to the day's spend, so
 * the local ceiling counts the same currency the platform bills in.
 *
 * `hops_in` is how many times this request had already been forwarded when it
 * arrived here (0 for a request that started on this machine). It is not used
 * to decide anything at this point — should_forward() already did that — but it
 * is written INSIDE the envelope as `hops`, together with `max_hops` and a
 * stable pseudonymous `origin_id` for this installation, so that:
 *
 *   - the platform can refuse to dispatch a request back to the machine that
 *     forwarded it (the loop RULE 1 cannot see, because from each machine's own
 *     side of it every hop looks like a first one), and
 *   - the exposure set of one prompt has a declared ceiling rather than an
 *     emergent one (PRIV-04).
 *
 * The coordinator emits and honours these fields; the platform side of the
 * contract is a cross-owner request, recorded in
 * results/security-hardening-overflow-privacy-20260830.md. Emitting them before
 * anyone reads them is deliberate: an unread field costs bytes, while a field
 * added later means every deployed coordinator is exempt from the rule. */
#define IDLETOKEN_OVF_EXCHANGE_OK       0
#define IDLETOKEN_OVF_EXCHANGE_ERROR   -1
#define IDLETOKEN_OVF_EXCHANGE_REFUSED -2

int idletoken_overflow_exchange(const char *messages_json, size_t messages_len,
                                const char *tools_json, size_t tools_len,
                                const char *tool_choice_json,
                                size_t tool_choice_len,
                                const char *model, const char *quant,
                                int max_tokens,
                                int hops_in,
                                idletoken_overflow_reply *out,
                                char *err, size_t err_cap);

/* A stable, pseudonymous id for this installation, hex, for the `origin_id`
 * field above. Derived from the persisted local-origin marker, so it survives a
 * restart and reveals nothing about the machine.
 *
 * It adds no linkability the platform did not already have: every overflow
 * request carries the account's own overflow key inside the same envelope, so
 * the account is identified exactly either way (PRIV-10 is unchanged by this
 * field). Returns 0 on success. */
int idletoken_overflow_origin_id(char *out, size_t cap);

/* Self-test for everything above: builds its own signing key pair, signs a good
 * sample, and asserts the good one passes and each bad one is refused for the
 * stated reason. Returns the number of FAILED assertions (0 = all pass); prints
 * one PASS/FAIL line each, in coord --selftest's format. */
int idletoken_overflow_selftest(void);

#endif /* IDLETOKEN_OVERFLOW_H */
