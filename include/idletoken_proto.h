/* IdleToken Cluster — worker <-> coordinator wire protocol (v0.1).
 *
 * Length-prefixed binary messages over TCP. All multi-byte fields are
 * little-endian. The header is fixed 48 bytes; payload layout depends on
 * msg_type.
 *
 * Node addressing is two-dimensional: (stage_id, segment_id). In v0.1
 * segment_id is always 0 (PP only). v0.2 will start using segment_id for
 * sequence parallelism (SP). Header fields below are forward-compatible:
 * `reserved` bytes never get repurposed; new fields arrive by bumping
 * IDLETOKEN_PROTO_VERSION and adding payload variants. See docs/ds4-anatomy.md
 * for what flows across the HC boundary.
 *
 * C only. No C++. Matches vendor/ds4 AGENT.md style.
 */

#ifndef IDLETOKEN_PROTO_H
#define IDLETOKEN_PROTO_H

#include <stdint.h>

#define IDLETOKEN_PROTO_MAGIC   0x31494148u  /* 'HAI1' in little-endian bytes */
/* v2 (multi-model, 2026-07): ASSIGN_PLAN carries the model identity
 * (model_id string + backend byte + total n_layers) and the layer range grew
 * u8 -> u16 (models may exceed 255 layers). Mixed-version clusters are
 * refused at HELLO — the coordinator requires exact version equality.
 * v3 (small models, 2026-07-27): ASSIGN_PLAN adds a `quant` string after
 * `model_id` (selected precision, may be empty) so workers load the matching
 * GGUF and validate its quant. small-model-design.md §3.3.
 * v4 (multi-sequence, 2026-07-30): INFER_BEGIN and INFER_HC_FORWARD carry a
 * `seq_id` in what used to be a reserved pad byte, so one cluster can hold
 * several **persistent** sequences (each with its own KV) and serve interleaved
 * conversations without destroying each other's prefix cache. seq_id 0 is the
 * historical single sequence, so a v4 cluster with seq_slots=1 behaves exactly
 * like v3. Execution stays one-round-at-a-time — concurrent micro-batching
 * across sequences is a later step. See docs/scheduler-design.md §6-E2.
 * v5 (pipeline barrier removed, 2026-08-03): INFER_TOKEN_ACK is **gone**.
 * It carried no information any worker used (every stage recv'd it, parsed it,
 * logged it, discarded it) but it was a hard pipeline barrier: a worker could
 * not begin round N+1 until coord had received LOGITS for round N — i.e. until
 * *every* stage had finished — so stage 0 sat idle for the whole tail of the
 * pipeline. Removing it is the precondition for E3 micro-batching, and on its
 * own it saves N_stages send + N_stages blocking recv **per token**. Workers
 * now loop straight back to recv INFER_BEGIN / INFER_HC_FORWARD. Coord MUST
 * NOT send it: a stray ACK would land where the worker expects the next
 * INFER_BEGIN and be rejected as a protocol error. See scheduler-design.md
 * §6-E3.
 * v6 (argmax at the last stage, 2026-08-11): INFER_LOGITS gains a SHORT FORM.
 * `n_vocab == 0` means "the last stage already took the argmax"; the payload is
 * then `u32 pos, u32 0, u32 token_id` — 12 bytes instead of 8 + 4*n_vocab.
 *
 * Why: the coordinator's only use of that vector has always been one argmax
 * (coord_main.c), so a 248,320-entry vocab shipped 993,280 B per token to
 * decide 4 bytes' worth of answer. Measured on one machine over loopback that
 * round trip plus the argmax cost ~6.9 ms of a 32.8 ms token — the socket has
 * to move ~1 MB through default-sized buffers every step. Across a real LAN it
 * is worse: 970 KB per token is ~8 ms on gigabit before anything else.
 *
 * The long form is NOT retired — it stays the wire format whenever the
 * coordinator needs the distribution rather than the winner (temperature /
 * top-p / logprobs are all future callers), and IDLETOKEN_FULL_LOGITS=1 forces
 * it for debugging. A v6 coordinator accepts both forms; that is what makes
 * adding sampling later a coordinator-side change instead of a wire change. */
#define IDLETOKEN_PROTO_VERSION 6u

/* Upper bound on persistent sequence slots per cluster. seq_id is a single
 * byte on the wire; the platform scheduler caps concurrency at 64 anyway
 * (scheduler.service.ts slotsFor). Real clusters are limited far below this
 * by KV memory, not by the protocol. */
#define IDLETOKEN_MAX_SEQ_SLOTS 32

#define IDLETOKEN_STAGE_COORD   0xFFFFFFFFu  /* sentinel stage_id for the coordinator */
#define IDLETOKEN_SEGMENT_NONE  0u           /* v0.1 always 0 */

/* Default UDP port for LAN discovery beacons/queries (distinct from the TCP
 * worker-facing port 14100 and the client-layer beacon 14099). Coordinators
 * broadcast IDLETOKEN_MSG_DISCOVER_BEACON here; workers listen and/or send
 * IDLETOKEN_MSG_DISCOVER_QUERY here. See docs/wire-protocol.md §Discovery. */
#define IDLETOKEN_DISCOVERY_PORT 14097u

/* A pairing group is addressed by a 32-byte id (HMAC-SHA256 of the join code,
 * or the cloud-assigned account cluster id). Nonces/tags in the pairing auth
 * preamble are fixed-width. */
#define IDLETOKEN_GROUP_ID_BYTES 32
#define IDLETOKEN_PAIR_NONCE_BYTES 16
#define IDLETOKEN_PAIR_TAG_BYTES 16
#define IDLETOKEN_SESSION_KEY_BYTES 32

/* OS family, as reported by the worker in HELLO.
 *
 * A cluster MUST be homogeneous: every compute node runs the same OS family.
 * The coordinator enforces this at handshake (see coord_main.c), because a
 * mixed cluster has no oracle — the numeric gates compare token ids against a
 * single-machine ds4 baseline, and CUDA (--use_fast_math) vs Metal differ
 * slightly per layer, so greedy decoding eventually flips an argmax. A mixed
 * cluster that "works" would be a green we cannot falsify. See CLAUDE.md
 * hard constraint #2 and docs/macos-node.md §5. */
typedef enum {
    IDLETOKEN_OS_UNKNOWN = 0,
    IDLETOKEN_OS_LINUX   = 1,
    IDLETOKEN_OS_WINDOWS = 2,
    IDLETOKEN_OS_MACOS   = 3
} idletoken_os_family;

/* This build's OS family — what a worker puts on the wire. Compile-time, not
 * probed: the binary cannot run on an OS it was not built for. */
#if defined(_WIN32)
#  define IDLETOKEN_OS_FAMILY_SELF IDLETOKEN_OS_WINDOWS
#elif defined(__APPLE__)
#  define IDLETOKEN_OS_FAMILY_SELF IDLETOKEN_OS_MACOS
#elif defined(__linux__)
#  define IDLETOKEN_OS_FAMILY_SELF IDLETOKEN_OS_LINUX
#else
#  define IDLETOKEN_OS_FAMILY_SELF IDLETOKEN_OS_UNKNOWN
#endif

/* Human-readable name for log lines and reject messages. Never NULL. */
static inline const char *idletoken_os_family_name(unsigned f) {
    switch (f) {
        case IDLETOKEN_OS_LINUX:   return "Linux";
        case IDLETOKEN_OS_WINDOWS: return "Windows";
        case IDLETOKEN_OS_MACOS:   return "macOS";
        default:                   return "unknown";
    }
}

typedef struct {
    uint32_t magic;          /* must be IDLETOKEN_PROTO_MAGIC */
    uint16_t version;        /* must be IDLETOKEN_PROTO_VERSION */
    uint16_t msg_type;       /* enum idletoken_msg_type */
    uint64_t payload_bytes;  /* length of payload following the header */
    uint64_t request_id;     /* opaque correlator chosen by initiator */
    uint32_t stage_id;       /* PP stage 0..N-1, or IDLETOKEN_STAGE_COORD */
    uint32_t segment_id;     /* SP segment, MUST be 0 in v0.1 */
    uint8_t  reserved[16];   /* zero on send, ignored on receive */
} idletoken_msg_header;
/* sizeof(idletoken_msg_header) must be 48 bytes; static-asserted in net.c */

typedef enum {
    /* --- handshake ----------------------------------------------------- */
    IDLETOKEN_MSG_HELLO              = 0x0001,  /* worker -> coord on connect */
    IDLETOKEN_MSG_HELLO_ACK          = 0x0002,  /* coord -> worker */

    /* --- LAN discovery (UDP datagrams, same 48-byte framed header) ------ */
    IDLETOKEN_MSG_DISCOVER_BEACON    = 0x0030,  /* coord -> broadcast/unicast: "I coordinate group G at addr" */
    IDLETOKEN_MSG_DISCOVER_QUERY     = 0x0031,  /* worker -> broadcast/unicast: "any coordinator for group G?" */

    /* --- pairing auth preamble (TCP, before HELLO, only when paired) ---- */
    IDLETOKEN_MSG_PAIR_HELLO         = 0x0032,  /* worker -> coord: group id + challenge */
    IDLETOKEN_MSG_PAIR_ACCEPT        = 0x0033,  /* coord -> worker: accept/reject + proof */

    /* --- resource reporting -------------------------------------------- */
    IDLETOKEN_MSG_RESOURCE_REPORT    = 0x0010,  /* worker -> coord, periodic */

    /* --- planning / loading -------------------------------------------- */
    IDLETOKEN_MSG_ASSIGN_PLAN        = 0x0020,  /* coord -> worker: model id/backend,
                                              * u16 layer range, model path, ctx */
    IDLETOKEN_MSG_LOAD_MODEL         = 0x0021,  /* coord -> worker: begin model load */
    IDLETOKEN_MSG_LOAD_MODEL_DONE    = 0x0022,  /* worker -> coord */

    /* --- inference -----------------------------------------------------
     * Payload prefixes (v4). Both carry seq_id at byte 3 — the slot whose KV
     * this round reads/writes. Workers keep one ds4_session / ds4x_runner per
     * slot, created lazily on first use.
     *   INFER_BEGIN:     u8 phase, u8 is_first_chunk, u8 is_last_chunk,
     *                    u8 seq_id, u32 pos0, u32 n_tokens, u32 reserved,
     *                    u32 tokens[n_tokens]
     *   INFER_HC_FORWARD:u8 phase, u8 hc_dtype, u8 is_first_chunk, u8 seq_id,
     *                    u32 pos0, u32 n_tokens, u32 hc_bytes, <hc payload>
     *                    [, u32 tokens[n_tokens] when n_tokens > 1]
     * INFER_LOGITS still carries **no** seq_id: coord drives one round at a
     * time and therefore always knows which slot it is in. Concurrent
     * micro-batching (E3) needs to claim logits out of order and will add it.
     *
     * 0x0043 (INFER_TOKEN_ACK) is **retired in v5** — see the version note at
     * the top. The value stays listed and MUST NOT be reused: a v4 worker that
     * somehow reached a v5 coord would otherwise mis-decode a new message as
     * an ACK. Version equality is enforced at HELLO, so this is belt-and-braces
     * for out-of-tree/mock implementations. */
    IDLETOKEN_MSG_INFER_BEGIN        = 0x0040,  /* coord -> stage0: prefill/decode tokens */
    IDLETOKEN_MSG_INFER_HC_FORWARD   = 0x0041,  /* stage k -> stage k+1: cur_hc tensor */
    IDLETOKEN_MSG_INFER_LOGITS       = 0x0042,  /* last stage -> coord */
    IDLETOKEN_MSG_INFER_TOKEN_ACK_RETIRED_V5 = 0x0043,  /* reserved, never reuse */

    /* --- ops / liveness ------------------------------------------------ */
    IDLETOKEN_MSG_HEARTBEAT          = 0x0080,
    IDLETOKEN_MSG_ERROR              = 0x00FF
} idletoken_msg_type;

/* Payload layouts (declared, defined alongside message handlers).
 *
 * v0.1 keeps payloads simple: fixed-size little-endian structs where
 * possible, length-prefixed byte arrays where variable. Strings are
 * length-prefixed UTF-8, not NUL-terminated.
 *
 * Detailed payload structs are TBD as each handler is implemented; see
 * docs/wire-protocol.md (to be written) for the canonical layout. */

/* Reasonable upper bound on a single message payload. Enough for an HC
 * tensor at prefill chunk_size=4096 in F32 (4 * 4096 * 4096 * 4 = 256MB).
 * Tighter caps will be enforced per message type at handler level. */
#define IDLETOKEN_MAX_PAYLOAD_BYTES (512ull * 1024ull * 1024ull)

#endif /* IDLETOKEN_PROTO_H */
