/* IdleToken — local admission capabilities: proving where a loopback request
 * came from, to something better than a header the sender chose to send.
 *
 * WHY THIS EXISTS (threat register PROV-28 / PRIV-05 / CHAIN-05)
 *
 * Overflow routing rests on one rule: a job the PLATFORM dispatched is finished
 * here or refused here, and is never forwarded back out. Until now the
 * coordinator learned which requests those were from `X-IdleToken-Origin:
 * platform`, a header the platform agent sets on itself. That is an ATTACKER-
 * CONTROLLED claim: the agent runs on the provider's own machine, and a
 * provider who deletes one line from it gets every dispatched job treated as
 * local work — free to be forwarded to a third machine, charged a second time,
 * and shown to one more stranger. Nothing anywhere reports an error.
 *
 * A header cannot be fixed by adding a second header. What this module adds is
 * a capability the coordinator MINTS and then CONSUMES:
 *
 *   1. the coordinator generates a channel key at startup and publishes it to a
 *      0600 file that the agent beside it reads;
 *   2. before forwarding a job the agent asks the coordinator to mint a ticket
 *      for THAT job id and THAT exact request body;
 *   3. the ticket rides on the request; the coordinator verifies the MAC, the
 *      expiry, the body hash and the fact that this ticket has never been used
 *      before, and consumes it.
 *
 * So a ticket cannot be minted for one job and spent on another, cannot be
 * replayed to mark a second request, and cannot be produced at all by something
 * that never spoke to the coordinator.
 *
 * WHAT IT DOES NOT DO — say this out loud, because the register's first
 * immutable judgement is that "official client" is not a security boundary:
 *
 *   - The key file sits on the provider's own disk, under the provider's own
 *     user. A same-user process can read it. This is the accepted HOST-02 /
 *     HOST-16 boundary, not a gap this module closes.
 *   - A provider who modifies the COORDINATOR as well as the agent removes the
 *     check entirely. No code running on the attacker's machine can bind the
 *     attacker. The residual enforcement for that case belongs to the platform,
 *     which is the only party the provider does not control (see the sealed
 *     provenance fields in idletoken_overflow.h and the cross-owner request in
 *     results/security-hardening-overflow-privacy-20260830.md).
 *
 * What it DOES close: the exact attack in PROV-28 — an agent that simply omits
 * the marker — because omission no longer reads as "local". Under the strict
 * origin policy (the default while `--shared` is on) a request that carries no
 * capability at all is not forwardable, so dropping the header stops the
 * forward instead of enabling it, and a capability that is present but does not
 * verify is refused outright rather than quietly demoted to "local".
 *
 * C99, no external dependencies: HMAC-SHA256 over the header-only SHA-256 in
 * idletoken_sha256.h, so the same object links into the coordinator and into
 * the platform agent (whose Makefile.platform link does not include
 * discovery.c, where the cluster's own HMAC lives). */

#ifndef IDLETOKEN_ADMISSION_H
#define IDLETOKEN_ADMISSION_H

#include <stddef.h>
#include <stdint.h>

/* The capability travels in its own header, never in Authorization: the
 * coordinator's --api-token is a different thing with a different lifetime, and
 * one field carrying two meanings is how the next person merges them. */
#define IDLETOKEN_HDR_ADMISSION    "X-IdleToken-Job-Admission"

/* The other direction: a marker that says "this request started on this
 * machine, in the client the user is looking at". Deliberately NOT single-use
 * and deliberately not called authentication — it is an ATTRIBUTION marker, and
 * a program on this machine can read the same file. Its job is to let the
 * coordinator distinguish the honest local caller from an unattributed one when
 * the strict origin policy is in force, so that closing the platform hole does
 * not also switch the feature off for the user who paid for it. */
#define IDLETOKEN_HDR_LOCAL_ORIGIN "X-IdleToken-Local-Origin"

/* Exposure hops already spent by this request (see idletoken_overflow.h).
 * Present on the way IN so a coordinator can refuse to extend a chain that has
 * already been extended once, and emitted on the way OUT so the next hop can do
 * the same. */
#define IDLETOKEN_HDR_HOPS         "X-IdleToken-Hops"

#define IDLETOKEN_ADM_KEY_BYTES    32
#define IDLETOKEN_ADM_JOB_CAP      65     /* 64 chars + NUL */
#define IDLETOKEN_ADM_TICKET_CAP   256
#define IDLETOKEN_ADM_KEYHEX_CAP   (IDLETOKEN_ADM_KEY_BYTES * 2 + 1)

/* How long a minted ticket stays spendable. Long enough for the agent to open
 * an envelope and post it, short enough that a ticket stolen off the disk is
 * worth little on its own. */
#define IDLETOKEN_ADM_TTL_S        120

/* Every way a presented capability can fail, each its own value.
 *
 * They are distinct because the gate has to be able to tell them apart: a
 * verifier that has degenerated into refusing everything would pass a test that
 * only asserts "rejected", and a coordinator that refuses every capability
 * silently stops serving the platform at all. */
typedef enum {
    IDLETOKEN_ADM_OK = 0,
    IDLETOKEN_ADM_ABSENT,          /* nothing was presented */
    IDLETOKEN_ADM_NOT_ARMED,       /* this coordinator mints no capabilities */
    IDLETOKEN_ADM_MALFORMED,       /* not a v1 ticket at all */
    IDLETOKEN_ADM_BAD_MAC,         /* forged, or minted by another coordinator */
    IDLETOKEN_ADM_EXPIRED,
    IDLETOKEN_ADM_REPLAYED,        /* already spent */
    IDLETOKEN_ADM_BODY_MISMATCH,   /* minted for a different request body */
    IDLETOKEN_ADM_NO_ROOM          /* replay table full of live entries */
} idletoken_adm_rc;

/* A short phrase naming the outcome, for logs and for the refusal body. */
const char *idletoken_admission_rc_str(idletoken_adm_rc rc);

/* --- arming ---------------------------------------------------------------
 *
 * `channel_path` receives the freshly generated per-process channel key (hex,
 * 0600); the agent reads it to mint. `local_path` receives the local-origin
 * marker; unlike the channel key this one is REUSED when the file already holds
 * a valid marker, so a coordinator restart does not invalidate the copy a
 * running client is already sending. Either path may be NULL/"" (tests, and a
 * machine with no writable state directory) — the corresponding value then
 * exists only in memory, and callers that needed the file must say so.
 *
 * Returns 0, or -1 with a one-line reason in `err`. A failure to publish the
 * channel key is NOT fatal to the coordinator by itself; it is fatal to
 * capability-based origin, and the caller decides which of those it is. */
int idletoken_admission_init(const char *channel_path, const char *local_path,
                             char *err, size_t err_cap);

/* Non-zero once init()/attach() has produced a channel key (in memory at
 * least). */
int idletoken_admission_ready(void);

/* --- attaching: the minting side (the platform agent) ---------------------
 *
 * The agent does not get a channel of its own. It ATTACHES to the coordinator's
 * — reading the hex the coordinator published — so that what it mints is what
 * that coordinator will recognise. Anything already armed is replaced, and the
 * local-origin marker is deliberately NOT loaded: a process that mints
 * job capabilities has no business also being able to hand out the marker that
 * says "a human at this machine asked for this".
 *
 * Returns 0, or -1 with a one-line reason in `err`. An agent that cannot attach
 * must say so and carry on: its jobs then arrive unattributed, which on a
 * strict coordinator means "serve it, do not forward it" — the safe direction. */
int idletoken_admission_attach(const char *channel_hex, char *err, size_t err_cap);
int idletoken_admission_attach_file(const char *channel_path, char *err, size_t err_cap);

/* Re-attach ONLY when the key on disk is no longer the one in use.
 *
 * The coordinator rolls its channel key on EVERY start (see init()), and the
 * client deliberately leaves the agent running across a coordinator restart
 * (engine.rs::stop_engine) — a model switch, a manual stop/start, a crash
 * respawn. An agent that attached once and cached the result therefore keeps
 * minting under a key nobody recognises any more, and because minting is a
 * local HMAC it SUCCEEDS: the failure only appears at the coordinator, as
 * BAD_MAC, on every dispatched job forever after. Measured on a discrete-GPU
 * Windows test node 2026-09-01: one coordinator restart turned every job into
 * `403 admission capability rejected: signature does not verify`, which the
 * platform then read as a provider fault — cooldown, strikes, and eventually a
 * blacklist for a machine that was serving perfectly well.
 *
 * So the attachment is a cache that must be INVALIDATED, not a latch. Cheap
 * enough to check per job: one ~65-byte file read next to an inference request.
 *
 * Returns 1 when it re-attached, 0 when the key was already current, and -1
 * when no key could be read from `channel_path` (reason in `err`).
 *
 * Agent-side only: it discards the spent/in-flight tables the way attach()
 * does, which is right for a process that mints and wrong for one that spends. */
int idletoken_admission_attach_file_if_changed(const char *channel_path,
                                               char *err, size_t err_cap);

/* Where the two files live, in ONE place, because the coordinator writes them
 * and a different binary reads them. `IDLETOKEN_STATE_DIR` overrides; otherwise
 * `$HOME/.idletoken` (`%USERPROFILE%` on Windows). Either output may be NULL.
 * Returns 0, or -1 when there is no home directory to hang them off. */
int idletoken_admission_default_paths(char *channel, size_t channel_cap,
                                      char *local, size_t local_cap);

/* Constant-time comparison of a presented bearer against the channel key (hex)
 * / the local-origin marker (hex). Both return 1 on match, 0 otherwise, and 0
 * for NULL or for a module that was never armed. */
int idletoken_admission_channel_ok(const char *presented_hex);
int idletoken_admission_local_ok(const char *presented_hex);

/* The local-origin marker, hex, so the coordinator can report it to a caller
 * that is entitled to it. Returns 0 on success. */
int idletoken_admission_local_marker(char *out, size_t cap);

/* --- minting and spending -------------------------------------------------
 *
 * `job_id` must be 1..64 chars of [A-Za-z0-9_:-]; anything else is refused
 * rather than escaped, because the id is a field inside the MAC input and an id
 * containing the field separator would make two different tickets hash the
 * same. `body_sha256` binds the ticket to the exact bytes that will be posted.
 */
int idletoken_admission_mint(const char *job_id, const uint8_t body_sha256[32],
                             long long now_unix,
                             char *out, size_t out_cap,
                             char *err, size_t err_cap);

/* Verify AND consume, in one step, so there is no window in which a caller
 * could check a ticket and forget to spend it. On IDLETOKEN_ADM_OK the ticket's
 * job id is copied into `out_job` (may be NULL) and the in-flight counter is
 * raised until idletoken_admission_request_end() runs on this thread. */
idletoken_adm_rc idletoken_admission_consume(const char *ticket,
                                             const uint8_t body_sha256[32],
                                             long long now_unix,
                                             char *out_job, size_t job_cap);

/* Called once per served request, on the thread that served it: drops the
 * in-flight mark this thread's capability took, if any. Safe to call when
 * nothing was consumed. */
void idletoken_admission_request_end(void);

/* Is this machine executing a platform-dispatched job right now?
 *
 * Overflow asks, because a machine in the middle of somebody else's paid job
 * must not simultaneously pay a third machine for what may be the same prompt —
 * that is the fee-expansion half of CHAIN-05, and unlike the origin question it
 * is decidable locally from facts the coordinator minted itself.
 *
 * The count self-heals: an entry older than IDLETOKEN_ADM_INFLIGHT_MAX_S is
 * treated as finished, so a thread that dies without calling request_end()
 * costs a bounded window rather than switching overflow off for good. */
#define IDLETOKEN_ADM_INFLIGHT_MAX_S 900
int idletoken_admission_platform_busy(long long now_unix);

/* Counters for /idletoken/v1/stats and for the gate. Any pointer may be NULL. */
void idletoken_admission_counters(long long *minted, long long *consumed,
                                  long long *rejected, int *inflight);

/* SHA-256 of a request body, the value both sides bind the ticket to. */
void idletoken_admission_body_hash(const void *body, size_t len, uint8_t out[32]);

/* Hex helper shared by both sides (lower case, NUL-terminated). */
void idletoken_admission_hex(const uint8_t *in, size_t n, char *out, size_t cap);

/* Self-test: mints against a known key and asserts that the good ticket is
 * spent exactly once and that each bad shape is refused FOR ITS OWN REASON.
 * Returns the number of failed assertions (0 = all pass) and prints one
 * PASS/FAIL line each, in coord --selftest's format. */
int idletoken_admission_selftest(void);

#endif /* IDLETOKEN_ADMISSION_H */
