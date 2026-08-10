/* IdleToken Cluster — LAN discovery + verification-code / account pairing.
 *
 * Turns "manual --coordinator <ip:port>" into "create/join a cluster by code
 * (or account)". Interface-first (architecture.md philosophy 4): the coord and
 * worker talk to a pluggable `idletoken_discovery` provider, so the four discovery
 * mechanisms (broadcast / manual / subnet / cloud rendezvous) and a test mock
 * all share one contract.
 *
 * Two roles:
 *   Coordinator: advertise() — announce "I coordinate group G at <addr>".
 *   Worker:      resolve()   — find a coordinator's <addr> for group G.
 *
 * A pairing group is identified by a 32-byte `group_id` and authenticated with
 * a 32-byte pre-shared key `psk`, both derived from either a 6-char join code
 * (pure-LAN, offline) or a cloud account identity (see idletoken_pair_id_*).
 *
 * Discovery only finds the coordinator's address; the TCP session that follows
 * is the existing wire protocol (docs/wire-protocol.md), optionally prefixed by
 * the pairing auth preamble (PAIR_HELLO / PAIR_ACCEPT) when a psk is present.
 *
 * C only. No C++. No gRPC/protobuf. */

#ifndef IDLETOKEN_DISCOVERY_H
#define IDLETOKEN_DISCOVERY_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_proto.h"

/* ---- pairing identity ---------------------------------------------------- */

typedef enum {
    IDLETOKEN_PAIR_MODE_CODE    = 0,  /* 6-char verification code, pure LAN */
    IDLETOKEN_PAIR_MODE_ACCOUNT = 1,  /* same-account, cloud identity match */
} idletoken_pair_mode;

/* Everything the discovery + auth layers need to identify and authenticate a
 * cluster group, independent of which mode produced it. */
typedef struct {
    idletoken_pair_mode mode;
    uint8_t group_id[IDLETOKEN_GROUP_ID_BYTES];   /* match key on the beacon/rendezvous */
    uint8_t psk[IDLETOKEN_SESSION_KEY_BYTES];      /* mutual-auth + session-key secret */
    /* Account mode only (empty in code mode): */
    char account_token[512];                    /* bearer JWT proving the account */
    char rendezvous_url[256];                    /* platform base "host:port" */
    char account_cluster[128];                   /* cloud-assigned cluster label */
} idletoken_pair_id;

/* ---- code helpers (mirror client/src/pairing.ts) ------------------------- */

#define IDLETOKEN_CODE_LEN 6
/* Unambiguous alphabet — no O/0/I/1 — for a code people read aloud. */
extern const char IDLETOKEN_CODE_ALPHABET[];   /* "ABCDEFGHJKLMNPQRSTUVWXYZ23456789", 32 chars */

/* Mint a fresh IDLETOKEN_CODE_LEN code into `out` (>= IDLETOKEN_CODE_LEN+1 bytes),
 * NUL-terminated, uppercase. Returns 0 / -1 (no entropy). */
int idletoken_pair_code_mint(char *out, size_t cap);

/* True if `code` is a syntactically valid join code (len, alphabet). Accepts
 * lowercase (callers should uppercase). */
int idletoken_pair_code_valid(const char *code);

/* Derive a pairing identity from a join code (code mode). `code` is
 * canonicalized (trim + uppercase) internally. Returns 0 / -1. */
int idletoken_pair_id_from_code(idletoken_pair_id *id, const char *code);

/* Derive a pairing identity from a cloud account identity (account mode).
 * `account_cluster` is the cloud-assigned cluster label shared by same-account
 * machines; `token` is the bearer JWT; `rendezvous_url` is the platform base
 * ("host:port"). group_id/psk are derived from account_cluster + token so only
 * same-account nodes match and can authenticate. Returns 0 / -1. */
int idletoken_pair_id_from_account(idletoken_pair_id *id,
                                const char *account_cluster,
                                const char *token,
                                const char *rendezvous_url);

/* ---- HMAC / session key (self-contained SHA-256, no external deps) -------- */

/* HMAC-SHA256(key, msg) -> out[32]. */
void idletoken_hmac_sha256(const uint8_t *key, size_t key_len,
                        const uint8_t *msg, size_t msg_len,
                        uint8_t out[32]);

/* Fill `buf` with `n` cryptographically-strong random bytes. Returns 0 / -1. */
int idletoken_random_bytes(void *buf, size_t n);

/* ---- discovery provider vtable ------------------------------------------- */

typedef enum {
    IDLETOKEN_DISCOVERY_BROADCAST  = 0,  /* UDP broadcast beacon (home main path) */
    IDLETOKEN_DISCOVERY_MANUAL     = 1,  /* fixed coordinator addr (always works) */
    IDLETOKEN_DISCOVERY_SUBNET     = 2,  /* known-port + /24 unicast scan */
    IDLETOKEN_DISCOVERY_RENDEZVOUS = 3,  /* cloud rendezvous (account mode) */
    IDLETOKEN_DISCOVERY_MULTI      = 4,  /* orchestrator: broadcast->manual->subnet->rendezvous */
    IDLETOKEN_DISCOVERY_MOCK       = 5,  /* deterministic in-memory (unit tests) */
} idletoken_discovery_kind;

typedef struct idletoken_discovery idletoken_discovery;
struct idletoken_discovery {
    idletoken_discovery_kind kind;

    /* Coordinator: begin advertising `self_addr` ("host:port" workers should
     * HELLO to) for group `id`. Non-blocking — spawns a background beacon
     * thread / posts to the rendezvous. Returns 0 / -1. */
    int (*advertise)(idletoken_discovery *d, const idletoken_pair_id *id,
                     const char *self_addr);

    /* Worker: block up to `timeout_ms` resolving a coordinator address for
     * group `id` into `out_addr` (cap `out_cap`). Returns 0 on found, -1 on
     * timeout/error (errno=ETIMEDOUT on timeout). */
    int (*resolve)(idletoken_discovery *d, const idletoken_pair_id *id,
                   char *out_addr, size_t out_cap, int timeout_ms);

    /* Named `destroy` (not `close`) on purpose: worker_main.c #defines `close`
     * to closesocket() on Windows, which would mangle a `->close` member call. */
    void (*destroy)(idletoken_discovery *d);
    void *state;
};

/* ---- provider constructors ----------------------------------------------- */

/* UDP broadcast beacon on `udp_port` (0 = IDLETOKEN_DISCOVERY_PORT). Coordinator
 * beacons every ~1s AND answers unicast DISCOVER_QUERY; worker listens for
 * beacons and also actively broadcasts a query. */
idletoken_discovery *idletoken_discovery_broadcast(uint16_t udp_port);

/* Manual: resolve() always returns `coord_addr` immediately; advertise() is a
 * no-op. The always-available fallback. */
idletoken_discovery *idletoken_discovery_manual(const char *coord_addr);

/* Subnet scan: unicast DISCOVER_QUERY to every host on this machine's /24 at
 * `udp_port`; coordinator replies unicast. Punches through broadcast-blocked
 * LANs (not client isolation). advertise() shares the broadcast beacon path. */
idletoken_discovery *idletoken_discovery_subnet(uint16_t udp_port);

/* Cloud rendezvous (account mode): advertise() POSTs {group,addr} to the
 * platform; resolve() polls the platform for a peer coordinator's LAN addr,
 * then connects directly over the LAN. Uses id->rendezvous_url + account_token.
 * The only automatic path through client isolation. */
idletoken_discovery *idletoken_discovery_rendezvous(void);

/* Orchestrator: try providers in the architecture §6.2 degrade order. For
 * resolve(): broadcast -> (manual if manual_addr) -> subnet -> rendezvous (if
 * account mode). For advertise(): fan out to broadcast + rendezvous (account).
 * `manual_addr` may be NULL. Owns and frees the sub-providers. */
idletoken_discovery *idletoken_discovery_multi(uint16_t udp_port, const char *manual_addr);

/* Mock: advertise()/resolve() a single in-process address keyed by group_id.
 * Deterministic, no sockets — for unit tests and the mock/real parity check. */
idletoken_discovery *idletoken_discovery_mock(void);

/* ---- TCP pairing auth preamble (PAIR_HELLO / PAIR_ACCEPT) ----------------
 *
 * Runs on the fresh TCP connection BEFORE the normal HELLO, only when a psk is
 * present (code or account mode). A symmetric HMAC challenge-response proves
 * both peers know the psk (mutual recognition) and derives a shared session key. A
 * node with the wrong code fails here and is dropped before HELLO. */

/* Worker side: send PAIR_HELLO, await PAIR_ACCEPT, verify the coordinator's
 * proof, derive `session_key`. Returns 0 on mutual-auth success, -1 otherwise
 * (errno=EACCES on auth failure). */
int idletoken_pair_client_auth(int fd, const idletoken_pair_id *id,
                            uint8_t session_key[IDLETOKEN_SESSION_KEY_BYTES]);

/* Coordinator side: read PAIR_HELLO, verify the worker's proof, reply
 * PAIR_ACCEPT with our proof, derive `session_key`. Returns 0 on success, -1 on
 * auth failure (a rejecting PAIR_ACCEPT is still sent so the worker learns why). */
int idletoken_pair_server_auth(int fd, const idletoken_pair_id *id,
                            uint8_t session_key[IDLETOKEN_SESSION_KEY_BYTES]);

#endif /* IDLETOKEN_DISCOVERY_H */
