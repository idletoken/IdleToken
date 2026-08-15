/* IdleToken coordinator — node discovery, planning, OpenAI/Anthropic API.
 *
 * v0.1 scope (this version):
 *   1. Listen for `--num-workers` workers on `--bind` (default 1).
 *   2. For each: HELLO -> HELLO_ACK -> RESOURCE_REPORT (synchronous,
 *      one connection at a time; workers wait their turn). Keep fds open.
 *   3. Rank workers by election score (vram_usable*1000 + ram_usable).
 *      The "strongest" worker is stage 0 by convention.
 *   4. Equal-split DS4's 43 layers across stages and send ASSIGN_PLAN
 *      to each worker.
 *
 * Election + planning is intentionally simple. The DP allocator from
 * The DP allocator (Jupiter-style) lands later once we have real timing
 * data. v0.1 just proves the wire plumbing end-to-end. */

#include "idletoken_proto.h"
#include "idletoken_net.h"
#include "idletoken_http.h"
#include "idletoken_model.h"
#include "idletoken_gguf.h"   /* idletoken_gguf_identity — model identity */
#include "idletoken_plan.h"
#include "idletoken_resource.h"   /* IDLETOKEN_JOIN_REFUSED_MARK */
#include "idletoken_advise.h"
#include "idletoken_discovery.h"
#include "idletoken_sha256.h"   /* session-key fingerprint for logs (never the key) */
#include "idletoken_nodecrypt.h"
#include "idletoken_privacy.h"   /* idletoken_secure_zero */
#include "idletoken_ds4x_tok.h"   /* GGUF byte-BPE tokenizer for ds4x models */
#include "idletoken_llama_sidecar.h"   /* llamacpp single-machine mode (WS-B1) */
#include "idletoken_model_auto.h"   /* open model intake: GGUF -> spec (WS-B4) */
#include "idletoken_enginever.h"   /* engine version invariant (WS-C3) */
#include "idletoken_apiconv.h"   /* Anthropic <-> OpenAI body translation */
#include "ds4.h"

#include <errno.h>
#include <signal.h>
#include <stdarg.h>

/* WINDOWS: close() is the CRT's, and it takes a FILE DESCRIPTOR. Every fd in
 * this file is a SOCKET handle from accept()/socket(), which the CRT knows
 * nothing about — so `close(sock)` returned -1/EBADF and **left the connection
 * open**. No FIN was ever sent, so a client that reads until end-of-stream sat
 * there until its own timeout: our desktop client showed "the cluster went
 * silent for 300s" AFTER a reply that had in fact completed normally. Handles
 * leaked on top of that.
 *
 * src/worker/worker_main.c has had this exact guard since the Windows port;
 * the coordinator never got it, and Linux/macOS hid the omission completely
 * because there close() is correct for sockets. src/common/net.c also exports
 * idletoken_close_fd() for the same purpose — this mirrors the worker so the
 * ~25 existing call sites stay as they are. */
#ifdef _WIN32
  #include <winsock2.h>
  #include <direct.h>
  #undef close
  #define close(fd) closesocket((SOCKET)(fd))
#else
  #include <poll.h>   /* llama_lfd_readable: the accept loop's 1 s pulse */
#endif
#ifdef __linux__
  #include <sys/prctl.h>
  #include <unistd.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define IDLETOKEN_MAX_WORKERS 16

/* The model this cluster run serves (--model-id, default DSv4-Flash). Set in
 * main() before any worker/HTTP traffic; the fallback keeps early error paths
 * safe. All layer counts / weight sizes come from here — never hard-coded
 * (multi-model design §3.3). */
static const idletoken_model_spec *g_model;

/* Cluster salt + the token-encryption key derived from it (proto v7,
 * docs/inter-node-encryption.md §3). Minted once per coordinator run, shipped
 * in every ASSIGN_PLAN. Stays all-zero when there is no pairing psk, and then
 * g_cluster_key_ok is 0 — this cluster simply cannot encrypt, which N2 turns
 * into a refusal to serve platform traffic rather than a silent cleartext
 * fallback. */
static uint8_t g_cluster_salt[IDLETOKEN_CLUSTER_SALT_BYTES];
static uint8_t g_cluster_key[IDLETOKEN_SESSION_KEY_BYTES];
static int     g_cluster_key_ok;
static const idletoken_model_spec *coord_model(void) {
    return g_model ? g_model : idletoken_model_default();
}

/* llamacpp single-machine mode (v2 rebuild WS-B1+B3): non-NULL when this
 * coordinator drives a local llama-server sidecar instead of a worker cluster.
 * The chat/tokenize/count_tokens routes then relay to it; every other route
 * (health, stats, models, cluster, capability) is served as before, so the
 * HTTP surface stays byte-compatible. Set once in main() before serving. */
static idletoken_llama *g_llama;

/* llama.cpp cluster membership exposed to the client status endpoint. Unlike
 * the legacy pipeline these nodes own tensor shares, not contiguous layer
 * ranges. Keep hostname + endpoint so the UI can map its frozen roster to the
 * engine topology without pretending an old PP plan still exists. */
typedef struct {
    char endpoint[64];
    char hostname[64];
    int  fd;   /* worker control fd (pairing channel), kept open for the
                * lifetime of the cluster: HEARTBEAT frames go down it every
                * IDLETOKEN_HB_INTERVAL_MS so workers can tell a dead-but-not-
                * closed coordinator from a live one. -1 once the link breaks;
                * 0/unset in single-machine mode (no peers). */
} idletoken_rpc_peer;
static idletoken_rpc_peer g_rpc_peers[IDLETOKEN_LLPLAN_MAX_NODES];
static int g_n_rpc_peers = 0;

/* The precision this run serves (--quant, default = model's default variant).
 * Empty string for models with no variant menu. Carried in ASSIGN_PLAN so the
 * worker loads the matching GGUF and validates its quant (small-model §3.3). */
static const char *g_quant = "";
static const char *coord_quant(void) { return g_quant ? g_quant : ""; }

/* SHA-256 of our GGUF's metadata region, sent in ASSIGN_PLAN so each worker can
 * prove it opened the same model+quant+layout. All-zero (ok=0) means "we could
 * not read our own copy" — workers must then skip the check, not invent one. */
static uint8_t g_model_identity[32] = {0};
static int     g_model_identity_ok = 0;

typedef struct {
    int      fd;
    uint8_t  uuid[16];
    char     hostname[64];
    char     version[64];
    char     bind_addr[64];
    uint8_t  os_family;
    /* llama.cpp build version from HELLO ("" = worker predates the field, or
     * legacy INFER worker). WS-C3: in rpc-cluster mode this must equal the
     * coordinator's own engine version or the worker is refused by name. */
    char     engine_version[IDLETOKEN_ENGINE_VERSION_MAX];

    char     gpu_name[64];
    uint8_t  cc_major, cc_minor, unified, _r0;
    uint64_t vram_total, vram_used_other, vram_usable;
    uint64_t ram_total,  ram_used_other,  ram_usable;
    uint64_t ram_pinnable;   /* measured pinned-memory ceiling, 0 = unknown (see idletoken_resource.h) */
    uint32_t cpu_count;
    uint64_t disk_avail;
    uint32_t net_link_mbps;
    uint8_t  can_run_ds4;

    /* derived */
    uint64_t score;
    uint8_t  stage_id;
    uint16_t layer_lo;   /* u16 on the wire too — models may exceed 255 layers */
    uint16_t layer_hi;
    /* Post-load usage as reported by LOAD_MODEL_DONE. This used to be parsed and
     * discarded; D2's slot self-sizing needs it to work out how much room the
     * machine has left for KV. */
    uint64_t vram_used_after, ram_used_after;

    /* Session key from the pairing preamble (docs/inter-node-encryption.md N0).
     *
     * PAIR_HELLO/PAIR_ACCEPT has always derived one — from the psk plus both
     * nonces, so it is fresh per pairing — and both sides then threw it away as
     * a local variable. Keeping it is the whole of N0: nothing encrypts yet, but
     * the material the token-id encryption (N1) needs now survives to where the
     * inference messages are actually sent.
     *
     * has_session_key = 0 for a cluster formed with a plain --coordinator: that
     * path has no shared secret at all, so it can never be encrypted, and N2
     * turns that into a refusal to serve platform traffic rather than a silent
     * cleartext fallback. */
    uint8_t  session_key[IDLETOKEN_SESSION_KEY_BYTES];
    uint8_t  has_session_key;

    /* Token-field crypto for this link (proto v7). One state per peer: it holds
     * both a send counter and a receive high-water mark, and the coordinator
     * talks to stage 0 (INFER_BEGIN) and hears from the last stage
     * (INFER_LOGITS), which are different workers unless the cluster has one
     * stage. Keyed with the cluster key, not the pairwise session key, so the
     * same nonce rules cover the stage<->stage links too. */
    idletoken_nodecrypt nc;
} idletoken_worker_info;

/* Short, non-reversible fingerprint of a session key, for logs and for the N0
 * gate to compare the two ends with. Never log the key itself: the whole point
 * of N0 is that this value stops being ephemeral, and a log line is exactly the
 * kind of place a secret gets left behind (see G_NO_PROMPT_LOG for the last
 * time that happened here). */
static void session_key_fp(const uint8_t *key, char out[9]) {
    uint8_t d[32];
    idletoken_sha256_ctx c;
    idletoken_sha256_init(&c);
    idletoken_sha256_update(&c, key, IDLETOKEN_SESSION_KEY_BYTES);
    idletoken_sha256_final(&c, d);
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 4; i++) { out[i * 2] = hex[d[i] >> 4]; out[i * 2 + 1] = hex[d[i] & 15]; }
    out[8] = '\0';
}

static void usage(FILE *out) {
    fprintf(out,
"idletoken-coord  cluster coordinator + OpenAI/Anthropic API (v0.1)\n"
"Usage: idletoken-coord [--bind H:P] [--num-workers N] [--ctx-size N]\n"
"                    [--model-id ID] [--model-path P] [--quant Q] [--n-predict N]\n"
"                    [--max-decode N]\n"
"\n"
"Optional:\n"
"  --bind H:P          worker-facing TCP (default: 0.0.0.0:14100)\n"
"  --num-workers N     wait for N workers, then plan (default: 1)\n"
"  --ctx-size N        context window in ASSIGN_PLAN (default: 8192)\n"
"  --model-id ID       model to serve, from the model registry\n"
"                      (default: deepseek-v4-flash; other registered models\n"
"                      need their backend implemented first)\n"
"  --model-path P      GGUF path each worker should load. Absolute path is\n"
"                      taken as-is; relative path is resolved by each worker\n"
"                      under its --gguf-dir.  (default: the model's own name)\n"
"  --gguf-dir DIR      where THIS process finds its own copy for the tokenizer\n"
"                      when --model-path is a bare filename (which is what\n"
"                      --quant leaves you with). Workers are unaffected.\n"
"  --quant Q           precision for small models (e.g. Q4_K_M, Q8_0, BF16);\n"
"                      default = the model's default variant. Ignored for\n"
"                      models with a single precision.\n"
"  --max-decode N      ceiling on tokens generated per request (default: 4096,\n"
"                      0 = bounded only by the remaining context). The client\n"
"                      passes the user's setting here, so it is configuration,\n"
"                      not a compiled-in constant -- and it ships 0, so this\n"
"                      4096 applies only when the engine is run by hand, where\n"
"                      a bounded default is the friendlier accident.\n"
"  --n-predict N       decode steps to drive after cluster_ready (default: 1;\n"
"                      0 = skip warmup so the first HTTP request starts at pos 0,\n"
"                      needed for token-exact comparison against single-node ds4)\n"
"  --seq-slots N|auto  persistent sequence slots (default: auto). Each slot is one\n"
"                      independent KV sequence on every worker, so N interleaved\n"
"                      conversations keep their prefix caches instead of\n"
"                      overwriting each other. Costs N x KV memory per worker —\n"
"                      raise it only if the workers have the headroom.\n"
"                      'auto' derives it from what is actually left after the\n"
"                      weights load: min over workers of (free / KV-per-seq),\n"
"                      using half the free memory and capped at stages x 2.\n"
"                      Pass 1 to get exactly the pre-v4 single-sequence behaviour.\n"
"  --concurrent-decode N|auto  interleave up to N requests to fill the PP\n"
"                      pipeline bubble (default: auto). While the coordinator\n"
"                      waits for request A's logits, stage 0 sits idle; with\n"
"                      N>1 it starts B in that window (measured 1.41x on a\n"
"                      2-machine cluster). 'auto' turns it on only when the\n"
"                      stages span >=2 machines: several stages on ONE box share\n"
"                      one GPU, so the bubble is not idle hardware and\n"
"                      interleaving is a net loss there (measured 0.91x).\n"
"                      Capped by --seq-slots — two requests must never share one\n"
"                      KV slot. Pass 0 to force strictly-one-at-a-time.\n"
"  --api-bind H:P      HTTP API bind addr (default: 0.0.0.0:8000)\n"
"  --http              serve HTTP API on --api-bind after warmup\n"
"  --tokenizer-only    no cluster: open the vocab and serve ONLY /health +\n"
"                      /idletoken/v1/tokenize on --api-bind (platform metering instance;\n"
"                      works with a layer-free sparse vocab shard)\n"
"  --api-token TOK     require `Authorization: Bearer TOK` or `x-api-key: TOK`\n"
"                      on /v1/messages and /v1/chat/completions (401 otherwise).\n"
"                      /health and /idletoken/v1/cluster/status stay open — clients and\n"
"                      peers poll them for liveness/pairing. Also via env\n"
"                      IDLETOKEN_API_TOKEN. (default: no auth — open on the LAN)\n"
"\n"
"llama.cpp single-machine mode (v2; give BOTH paths to enable):\n"
"  --llama-server-bin P  llama-server binary to spawn as the local engine\n"
"                      (env IDLETOKEN_LLAMA_SERVER_BIN)\n"
"  --llama-gguf P      GGUF the engine loads (env IDLETOKEN_LLAMA_GGUF)\n"
"  --max-vram-mb N     cap this machine's usable VRAM at N MiB before planning\n"
"                      (client usage slider; 0 = no cap; llamacpp mode)\n"
"  --max-ram-mb N      cap this machine's usable RAM at N MiB (same contract\n"
"                      as the worker flag of the same name)\n"
"  --llama-port N      engine port on 127.0.0.1 (default 18099; env\n"
"                      IDLETOKEN_LLAMA_PORT). The engine is loopback-only;\n"
"                      the coordinator API (--api-bind, --api-token) is the\n"
"                      only external surface. Extra engine args via\n"
"                      IDLETOKEN_LLAMA_ARGS (whitespace-split).\n"
"                      IDLETOKEN_FORCE_BACKEND=ds4 skips this mode (loud).\n"
"                      With an explicit --num-workers N (and pairing), this\n"
"                      becomes llamacpp CLUSTER mode: N idletoken-worker\n"
"                      --rpc-supervisor nodes join, each gets the cluster TLS\n"
"                      PSK through the pairing channel, and the local\n"
"                      llama-server spans them via --rpc/--tensor-split with\n"
"                      layer 0 pinned to this machine. Local device name via\n"
"                      IDLETOKEN_LLAMA_DEVICE (default MTL0 on macOS, CUDA0\n"
"                      elsewhere); PSK file via IDLETOKEN_RPC_PSK_FILE\n"
"                      (default ~/.idletoken/rpc_psk).\n"
"\n"
"Pairing / LAN discovery (workers join by code — no manual --coordinator):\n"
"  --create            mint a fresh join code and advertise on the LAN\n"
"  --pair-code CODE    advertise this cluster under join code CODE\n"
"  --pair-account C    account mode: cloud cluster label C (+ --account-token,\n"
"                      --rendezvous) so same-account machines self-assemble\n"
"  --account-token JWT bearer token proving the account (account mode)\n"
"  --rendezvous H:P    cloud rendezvous endpoint (account mode)\n"
"  --discovery-port N  UDP discovery port (default 14097)\n"
"  -h, --help          show this help\n");
}

static int do_hello(int fd, idletoken_worker_info *w, uint64_t *out_request_id) {
    uint8_t hp[1024];
    idletoken_msg_header h;
    if (idletoken_recv_msg(fd, &h, hp, sizeof(hp)) != 0) {
        fprintf(stderr, "coord: recv HELLO: %s\n", strerror(errno));
        return -1;
    }
    if (h.msg_type != IDLETOKEN_MSG_HELLO) {
        fprintf(stderr, "coord: expected HELLO, got 0x%04x\n", h.msg_type);
        return -1;
    }
    /* net.c only rejects NEWER versions; refuse older workers explicitly —
     * v2 changed the ASSIGN_PLAN layout (model_id + u16 layer range), silent
     * mixed-version clusters would mis-parse it. */
    if (h.version != IDLETOKEN_PROTO_VERSION) {
        fprintf(stderr, "coord: worker speaks protocol v%u, we need v%u — "
                        "upgrade the older side\n",
                (unsigned)h.version, (unsigned)IDLETOKEN_PROTO_VERSION);
        return -1;
    }

    idletoken_buf b;
    idletoken_buf_init(&b, hp, h.payload_bytes);
    uint8_t pad3[3];
    idletoken_buf_get_bytes(&b, w->uuid, 16);
    idletoken_buf_get_str  (&b, w->hostname,  sizeof(w->hostname));
    idletoken_buf_get_str  (&b, w->version,   sizeof(w->version));
    idletoken_buf_get_str  (&b, w->bind_addr, sizeof(w->bind_addr));
    idletoken_buf_get_u8   (&b, &w->os_family);
    idletoken_buf_get_bytes(&b, pad3, 3);
    if (b.err) { fprintf(stderr, "coord: HELLO payload malformed\n"); return -1; }
    /* Optional trailing field (WS-C3): the worker's llama.cpp build version.
     * Read only when bytes remain, so a pre-C3 worker's HELLO still parses;
     * a failed optional read must not poison the verdict above. */
    w->engine_version[0] = '\0';
    if (b.pos + 2 <= h.payload_bytes) {
        idletoken_buf ev = b;   /* copy: keep b's err state pristine */
        char ver[IDLETOKEN_ENGINE_VERSION_MAX] = "";
        if (idletoken_buf_get_str(&ev, ver, sizeof(ver)) == 0 && !ev.err)
            snprintf(w->engine_version, sizeof(w->engine_version), "%s", ver);
    }
    *out_request_id = h.request_id;
    return 0;
}

static int send_hello_ack(int fd, uint64_t request_id) {
    idletoken_msg_header h = {
        .magic = IDLETOKEN_PROTO_MAGIC,
        .version = IDLETOKEN_PROTO_VERSION,
        .msg_type = IDLETOKEN_MSG_HELLO_ACK,
        .payload_bytes = 0,
        .request_id = request_id,
        .stage_id = IDLETOKEN_STAGE_COORD,
        .segment_id = IDLETOKEN_SEGMENT_NONE,
    };
    if (idletoken_send_msg(fd, &h, NULL, 0) != 0) {
        fprintf(stderr, "coord: send HELLO_ACK: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/* Refusing ACK: the accept path used to just close the socket, which left the
 * rejected machine's operator staring at "connection reset" with no reason.
 * docs/wire-protocol.md has always specified accepted/reject_message here; an
 * empty payload still means "accepted" so older coordinators stay compatible. */
static void send_hello_reject(int fd, uint64_t request_id, uint8_t reasoncode,
                              const char *why) {
    uint8_t p[512];
    idletoken_buf b;
    idletoken_buf_init(&b, p, sizeof(p));
    idletoken_buf_put_u8 (&b, 0);            /* accepted = no */
    idletoken_buf_put_u8 (&b, reasoncode);
    idletoken_buf_put_u8 (&b, 0);
    idletoken_buf_put_u8 (&b, 0);            /* reserved[2] */
    idletoken_buf_put_u16(&b, (uint16_t)IDLETOKEN_PROTO_VERSION);
    idletoken_buf_put_u16(&b, 0);            /* reserved */
    idletoken_buf_put_u32(&b, 0);            /* heartbeat_secs: n/a, not joining */
    idletoken_buf_put_str(&b, "idletoken-coord v0.1.0-pre");
    idletoken_buf_put_str(&b, why);
    if (b.err) return;

    idletoken_msg_header h = {
        .magic = IDLETOKEN_PROTO_MAGIC,
        .version = IDLETOKEN_PROTO_VERSION,
        .msg_type = IDLETOKEN_MSG_HELLO_ACK,
        .payload_bytes = b.pos,
        .request_id = request_id,
        .stage_id = IDLETOKEN_STAGE_COORD,
        .segment_id = IDLETOKEN_SEGMENT_NONE,
    };
    (void)idletoken_send_msg(fd, &h, p, b.pos);
}

static int recv_resource_report(int fd, idletoken_worker_info *w) {
    uint8_t rp[1024];
    idletoken_msg_header h;
    if (idletoken_recv_msg(fd, &h, rp, sizeof(rp)) != 0) {
        fprintf(stderr, "coord: recv RESOURCE_REPORT: %s\n", strerror(errno));
        return -1;
    }
    if (h.msg_type != IDLETOKEN_MSG_RESOURCE_REPORT) {
        fprintf(stderr, "coord: expected RESOURCE_REPORT, got 0x%04x\n", h.msg_type);
        return -1;
    }

    idletoken_buf b;
    idletoken_buf_init(&b, rp, h.payload_bytes);
    uint32_t r1 = 0, r2 = 0;
    uint8_t pad7[7];
    idletoken_buf_get_str  (&b, w->gpu_name, sizeof(w->gpu_name));
    idletoken_buf_get_u8   (&b, &w->cc_major);
    idletoken_buf_get_u8   (&b, &w->cc_minor);
    idletoken_buf_get_u8   (&b, &w->unified);
    idletoken_buf_get_u8   (&b, &w->_r0);
    idletoken_buf_get_u64  (&b, &w->vram_total);
    idletoken_buf_get_u64  (&b, &w->vram_used_other);
    idletoken_buf_get_u64  (&b, &w->vram_usable);
    idletoken_buf_get_u64  (&b, &w->ram_total);
    idletoken_buf_get_u64  (&b, &w->ram_used_other);
    idletoken_buf_get_u64  (&b, &w->ram_usable);
    idletoken_buf_get_u32  (&b, &w->cpu_count);
    idletoken_buf_get_u32  (&b, &r1);            /* = measured pinned ceiling in MiB, 0 = unknown */
    w->ram_pinnable = (uint64_t)r1 << 20;
    idletoken_buf_get_u64  (&b, &w->disk_avail);
    idletoken_buf_get_u32  (&b, &w->net_link_mbps);
    idletoken_buf_get_u32  (&b, &r2);
    idletoken_buf_get_u8   (&b, &w->can_run_ds4);
    idletoken_buf_get_bytes(&b, pad7, 7);
    if (b.err) { fprintf(stderr, "coord: RESOURCE_REPORT payload malformed\n"); return -1; }

    w->score = w->vram_usable * 1000ull + w->ram_usable;
    return 0;
}

static int cmp_worker_desc(const void *pa, const void *pb) {
    const idletoken_worker_info *a = (const idletoken_worker_info *)pa;
    const idletoken_worker_info *b = (const idletoken_worker_info *)pb;
    if (a->score < b->score) return  1;
    if (a->score > b->score) return -1;
    return 0;
}

/* Resource-proportional split + mode decision live in src/common/plan.c as
 * pure unit-tested functions (src/tools/plan_test.c). This wrapper adapts
 * idletoken_worker_info (callers pass it sorted strongest-first) and stamps
 * stage ids / contiguous layer ranges. In HYBRID the split sizes by VRAM+RAM
 * so weak-VRAM/ample-RAM nodes get more layers and offload the overflow. */
static void plan_layers(idletoken_worker_info *ws, int n, idletoken_mode mode,
                        uint32_t ctx_size) {
    if (n <= 0) return;

    idletoken_node_mem nodes[IDLETOKEN_MAX_WORKERS];
    for (int i = 0; i < n && i < IDLETOKEN_MAX_WORKERS; i++) {
        nodes[i].vram_usable = ws[i].vram_usable;
        nodes[i].ram_usable  = ws[i].ram_usable;
        nodes[i].ram_pinnable= ws[i].ram_pinnable;
        nodes[i].unified     = ws[i].unified;
    }

    int counts[IDLETOKEN_MAX_WORKERS];

    const int n_layers = (int)coord_model()->n_layers;

    /* Test/override hook: IDLETOKEN_FORCE_LAYERS="c0,c1,..." pins each stage's
     * layer count (strongest-first order), bypassing the proportional split.
     * Used to deliberately assign a discrete card MORE layers than fit in its
     * VRAM to exercise HYBRID (managed-memory VRAM+RAM). Counts must sum to
     * the model's layer count. */
    const char *force = getenv("IDLETOKEN_FORCE_LAYERS");
    if (force && force[0]) {
        int sum = 0, k = 0;
        const char *p = force;
        while (*p && k < n) {
            counts[k++] = (int)strtol(p, NULL, 10);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        for (int i = 0; i < k; i++) sum += counts[i];
        if (k == n && sum == n_layers) {
            fprintf(stderr, "coord: IDLETOKEN_FORCE_LAYERS override active: %s\n", force);
            int lo2 = 0;
            for (int i = 0; i < n; i++) {
                ws[i].stage_id = (uint8_t)i;
                ws[i].layer_lo = (uint16_t)lo2;
                ws[i].layer_hi = (uint16_t)(lo2 + counts[i]);
                lo2 += counts[i];
            }
            return;
        }
        fprintf(stderr, "coord: IDLETOKEN_FORCE_LAYERS ignored (need %d counts summing to %d; got %d summing to %d)\n",
                n, n_layers, k, sum);
    }

    if (idletoken_plan_layers(coord_model(), nodes, n, ctx_size, counts, mode) != 0) {
        /* More workers than layers: give everything to the strongest.
         * (Upstream should reject such clusters at join time.) */
        fprintf(stderr, "coord: plan_layers rejected n=%d; single-stage fallback\n", n);
        ws[0].stage_id = 0;
        ws[0].layer_lo = 0;
        ws[0].layer_hi = (uint16_t)n_layers;
        return;
    }

    int lo = 0;
    for (int i = 0; i < n; i++) {
        ws[i].stage_id = (uint8_t)i;
        ws[i].layer_lo = (uint16_t)lo;
        ws[i].layer_hi = (uint16_t)(lo + counts[i]);
        lo += counts[i];
    }
}

static int send_assign_plan(const idletoken_worker_info *w, int cluster_size,
                            const idletoken_worker_info *prev, const idletoken_worker_info *next,
                            uint32_t ctx_size, const char *model_path,
                            const char *coord_inbox, uint8_t mode) {
    uint8_t buf[2048];
    idletoken_buf b;
    idletoken_buf_init(&b, buf, sizeof(buf));

    /* v2 layout: u16 layer range + model identity (id/backend/n_layers) so a
     * worker can pick its backend and validate the range — see proto.h. */
    idletoken_buf_put_u8(&b, (uint8_t)cluster_size);
    idletoken_buf_put_u8(&b, w->stage_id);
    idletoken_buf_put_u8(&b, 0);                  /* segment_id, v0.1 */
    idletoken_buf_put_u8(&b, mode);               /* idletoken_mode: 1=GPU_ONLY, 2=HYBRID */
    idletoken_buf_put_u16(&b, w->layer_lo);
    idletoken_buf_put_u16(&b, w->layer_hi);
    idletoken_buf_put_u16(&b, coord_model()->n_layers);
    idletoken_buf_put_u8(&b, coord_model()->backend);
    idletoken_buf_put_u8(&b, 0);                  /* pad */
    idletoken_buf_put_u32(&b, ctx_size);
    idletoken_buf_put_u32(&b, 0);                 /* prefill_cap: 0 = worker picks */
    idletoken_buf_put_u8(&b, 1);                  /* hc_dtype: 1=F32 (v0.1) */
    uint8_t pad7[7] = {0};
    idletoken_buf_put_bytes(&b, pad7, 7);
    /* Model identity: sha256 of the GGUF metadata region (see gguf.h). Zeros
     * when this coordinator could not read its own copy — the worker reads
     * all-zero as "not vouched for" and skips, exactly as before. */
    idletoken_buf_put_bytes(&b, g_model_identity, 32);
    (void)g_model_identity_ok;
    idletoken_buf_put_str(&b, coord_model()->id);
    idletoken_buf_put_str(&b, coord_quant());     /* selected precision (may be "") */
    idletoken_buf_put_str(&b, model_path);
    idletoken_buf_put_str(&b, prev ? prev->bind_addr : "");
    idletoken_buf_put_str(&b, next ? next->bind_addr : "");
    idletoken_buf_put_str(&b, coord_inbox ? coord_inbox : "");
    /* v7: cluster salt, last field so older parsers stop before it. Every node
     * derives the token-encryption key from psk + this salt, so it needs no
     * distribution (docs/inter-node-encryption.md §3). All-zero when this
     * cluster has no shared secret at all (a --coordinator cluster). */
    idletoken_buf_put_bytes(&b, g_cluster_salt, IDLETOKEN_CLUSTER_SALT_BYTES);
    if (b.err) { fprintf(stderr, "coord: ASSIGN_PLAN pack overflow\n"); return -1; }

    idletoken_msg_header h = {
        .magic = IDLETOKEN_PROTO_MAGIC,
        .version = IDLETOKEN_PROTO_VERSION,
        .msg_type = IDLETOKEN_MSG_ASSIGN_PLAN,
        .payload_bytes = b.pos,
        .request_id = 0,
        .stage_id = IDLETOKEN_STAGE_COORD,
        .segment_id = IDLETOKEN_SEGMENT_NONE,
    };
    if (idletoken_send_msg(w->fd, &h, buf, b.pos) != 0) {
        fprintf(stderr, "coord: send ASSIGN_PLAN to %s: %s\n", w->hostname, strerror(errno));
        return -1;
    }
    return 0;
}

/* Drive one INFER round (single decode token, or one prefill chunk) across the
 * cluster: INFER_BEGIN(pos0, tokens[n_tokens]) -> stage 0, INFER_LOGITS <- last
 * stage, argmax. n_tokens == 1 is the decode path; n_tokens > 1 is a prefill
 * chunk (workers take ds4's batched prefill kernels — numerics identical to
 * official single-machine chunked prefill). Chunk size must respect
 * ds4_prefill_chunk_cap_for_ctx(ctx). Returns 0 with `*out_token` = argmax of
 * the last position's logits. `lbuf` must be ≥ 8 + N_VOCAB*4 bytes.
 * (v5 dropped the INFER_TOKEN_ACK broadcast that used to close each round —
 * it was a pure pipeline barrier; see idletoken_proto.h.)
 *
 * --- Split into send / recv halves (E3.2) ---
 * The split exists so the executor can **send a B** during the window where A
 * has gone out but A's logits have not come back -- and that window is exactly
 * the pipeline bubble: stage 0 has handed A downstream and is sitting idle.
 * coord_infer_round() remains the send+recv combination; prefill and today's
 * serial decode keep using it, unchanged. */
static int coord_round_send(idletoken_worker_info *ws, int n,
                            uint64_t req_id,
                            uint32_t pos0,
                            const uint32_t *tokens, uint32_t n_tokens,
                            int first_chunk, uint8_t seq_id) {
    (void)n;
    if (n_tokens == 0) return -1;
    idletoken_worker_info *stage0 = &ws[0];

    /* 1. INFER_BEGIN -> stage 0 */
    const size_t begin_cap = 20 + (size_t)n_tokens * 4 + IDLETOKEN_NODECRYPT_OVERHEAD;
    uint8_t *buf = malloc(begin_cap);
    if (!buf) return -1;
    idletoken_buf b;
    idletoken_buf_init(&b, buf, begin_cap);
    idletoken_buf_put_u8 (&b, n_tokens > 1 ? 1 : 2);  /* phase: 1=PREFILL_CHUNK, 2=DECODE_TOKEN */
    /* is_first_chunk = this call starts a new sequence at pos0. A REAL worker
     * that sees first together with pos0==0 calls ds4_session_rewind(0) before
     * writing KV, clearing the compressed layers' rolling state so one session
     * cannot contaminate the next. */
    idletoken_buf_put_u8 (&b, first_chunk ? 1 : 0);
    idletoken_buf_put_u8 (&b, 0);             /* is_last_chunk (reserved) */
    idletoken_buf_put_u8 (&b, seq_id);        /* v4: which sequence slot's KV this round uses */
    idletoken_buf_put_u32(&b, pos0);
    idletoken_buf_put_u32(&b, n_tokens);
    idletoken_buf_put_u32(&b, 0);             /* reserved */
    /* Token ids: encrypted when this cluster has a key (proto v7). The receiver
     * needs no flag -- it derived the same key from the same salt, so it knows
     * which form to expect. A cluster with no pairing secret sends them in the
     * clear, which is the honest state for a --coordinator cluster and is what
     * N2 refuses to serve platform traffic on. */
    if (stage0->nc.ready) {
        /* Sized from n_tokens rather than a compile-time cap: prefill chunk
         * sizes come from ds4_prefill_chunk_cap_for_ctx(), so a fixed array here
         * would be a second, quieter limit on how big a chunk may be. */
        size_t tcap = (size_t)n_tokens * 4;
        uint8_t *tbuf = malloc(tcap);
        uint8_t *wrapped = malloc(tcap + IDLETOKEN_NODECRYPT_OVERHEAD);
        if (!tbuf || !wrapped) { free(tbuf); free(wrapped); free(buf); return -1; }
        idletoken_buf tb;
        idletoken_buf_init(&tb, tbuf, tcap);
        for (uint32_t i = 0; i < n_tokens; i++) idletoken_buf_put_u32(&tb, tokens[i]);
        size_t wlen = 0;
        idletoken_nc_rc nrc = tb.err ? IDLETOKEN_NC_EINVAL
            : idletoken_nodecrypt_wrap(&stage0->nc, tbuf, tb.pos,
                                       wrapped, tcap + IDLETOKEN_NODECRYPT_OVERHEAD, &wlen);
        if (nrc == IDLETOKEN_NC_OK) idletoken_buf_put_bytes(&b, wrapped, wlen);
        idletoken_secure_zero(tbuf, tcap);
        free(tbuf); free(wrapped);
        if (nrc != IDLETOKEN_NC_OK) {
            fprintf(stderr, "coord: could not encrypt INFER_BEGIN token ids\n");
            free(buf); return -1;
        }
    } else {
        for (uint32_t i = 0; i < n_tokens; i++) idletoken_buf_put_u32(&b, tokens[i]);
    }
    if (b.err) { free(buf); return -1; }
    idletoken_msg_header bh = {
        .magic = IDLETOKEN_PROTO_MAGIC,
        .version = IDLETOKEN_PROTO_VERSION,
        .msg_type = IDLETOKEN_MSG_INFER_BEGIN,
        .payload_bytes = b.pos,
        .request_id = req_id,
        .stage_id = IDLETOKEN_STAGE_COORD,
        .segment_id = IDLETOKEN_SEGMENT_NONE,
    };
    int send_rc = idletoken_send_msg(stage0->fd, &bh, buf, b.pos);
    free(buf);
    if (send_rc != 0) {
        fprintf(stderr, "coord: send INFER_BEGIN: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static double coord_prof_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Receive one INFER_LOGITS and take the argmax. `out_req_id` reports which
 * request these logits belong to: the worker copies the request_id from the
 * INFER_BEGIN header straight into LOGITS (worker_main.c), so interleaved
 * requests can be claimed **without touching the protocol**. */
static int coord_round_recv(idletoken_worker_info *ws, int n,
                            uint8_t *lbuf, size_t lbuf_cap,
                            uint32_t *out_token, uint64_t *out_req_id) {
    if (out_token) *out_token = 0;
    if (out_req_id) *out_req_id = 0;
    idletoken_worker_info *last = &ws[n - 1];

    /* 2. recv INFER_LOGITS from last stage */
    idletoken_msg_header lh;
    if (idletoken_recv_msg(last->fd, &lh, lbuf, lbuf_cap) != 0) {
        fprintf(stderr, "coord: recv INFER_LOGITS: %s\n", strerror(errno));
        return -1;
    }
    if (lh.msg_type != IDLETOKEN_MSG_INFER_LOGITS) {
        fprintf(stderr, "coord: expected LOGITS got 0x%04x\n", lh.msg_type);
        return -1;
    }
    idletoken_buf lb;
    idletoken_buf_init(&lb, lbuf, lh.payload_bytes);
    uint32_t got_pos = 0, got_n_vocab = 0;
    idletoken_buf_get_u32(&lb, &got_pos);
    idletoken_buf_get_u32(&lb, &got_n_vocab);
    /* v6 SHORT FORM: n_vocab == 0 means the last stage already took the argmax
     * and the payload is just the winning token id. That is the normal case —
     * this function's only use of a full distribution has always been to pick
     * its maximum, so shipping ~1 MB per token to learn 4 bytes was the single
     * largest non-compute cost per token (~6.9 ms of 32.8 on one machine over
     * loopback; worse across a LAN).
     *
     * The long form is still accepted, and must stay accepted: it is what
     * IDLETOKEN_FULL_LOGITS=1 produces, and it is what any non-greedy sampling
     * will need. Handling both here is what keeps "add temperature/top-p" a
     * change to this file rather than a change to the wire. */
    uint32_t argmax = 0;
    if (got_n_vocab == 0) {
        /* 8 header bytes + the token field, which is 4 in the clear and
         * 4 + overhead when encrypted (proto v7). Checking the wrong minimum
         * would reject every encrypted frame as "too small". */
        const uint32_t need = 8 + (last->nc.ready ? 4 + IDLETOKEN_NODECRYPT_OVERHEAD : 4);
        if (lb.err || lh.payload_bytes < need) {
            fprintf(stderr, "coord: INFER_LOGITS short form too small (%u B, need %u)\n",
                    (unsigned)lh.payload_bytes, need);
            return -1;
        }
        if (last->nc.ready) {
            uint8_t tw[4 + IDLETOKEN_NODECRYPT_OVERHEAD], tp[4];
            size_t got = 0;
            idletoken_buf_get_bytes(&lb, tw, sizeof(tw));
            idletoken_nc_rc nrc = lb.err ? IDLETOKEN_NC_EINVAL
                : idletoken_nodecrypt_unwrap(&last->nc, tw, sizeof(tw), tp, sizeof(tp), &got);
            if (nrc != IDLETOKEN_NC_OK || got != 4) {
                fprintf(stderr, "coord: INFER_LOGITS token failed to decrypt (rc=%d)\n", (int)nrc);
                return -1;
            }
            idletoken_buf tb; idletoken_buf_init(&tb, tp, got);
            idletoken_buf_get_u32(&tb, &argmax);
        } else {
            idletoken_buf_get_u32(&lb, &argmax);
        }
        if (lb.err || argmax >= coord_model()->n_vocab) {
            fprintf(stderr, "coord: INFER_LOGITS short form token %u out of range "
                            "(n_vocab=%u)\n", argmax, coord_model()->n_vocab);
            return -1;
        }
    } else {
    /* vocab must match THIS model's registry entry — never a hard-coded DSv4
     * constant (multi-model §3.3: no model-specific numbers in the planner). */
    if (lb.err || got_n_vocab != coord_model()->n_vocab ||
        lh.payload_bytes < 8 + (uint64_t)got_n_vocab * sizeof(float)) {
        fprintf(stderr, "coord: INFER_LOGITS malformed (pos=%u n_vocab=%u)\n",
                got_pos, got_n_vocab);
        return -1;
    }
    const float *logits = (const float *)(lbuf + 8);
    float    best   = logits[0];
    for (uint32_t i = 1; i < got_n_vocab; i++) {
        if (logits[i] > best) { best = logits[i]; argmax = i; }
    }
    (void)best;
    }

    /* v5: this used to broadcast INFER_TOKEN_ACK to **every** stage. Removed --
     * no worker had any use for the pos/token it carried (received, parsed,
     * logged, dropped), yet it acted as a hard barrier: each worker blocked at
     * the end of the round waiting for the ACK, so stage 0 could not start the
     * next round until the coordinator had collected LOGITS (i.e. until every
     * stage had finished), forcing the whole pipeline to run serially. Removing
     * it is a precondition for E3 micro-batching, and it also saves N sends plus
     * N blocking receives per token.
     * WARNING: the coordinator must never send it again -- a stray ACK would
     * arrive where the worker expects the next INFER_BEGIN and be treated as a
     * protocol error. */
    (void)got_pos;
    if (out_token)  *out_token  = argmax;
    if (out_req_id) *out_req_id = lh.request_id;
    return 0;
}

/* The send + recv combination, equivalent to coord_infer_round before the split. */
static int coord_infer_round(idletoken_worker_info *ws, int n,
                             uint64_t req_id,
                             uint32_t pos0,
                             const uint32_t *tokens, uint32_t n_tokens,
                             int first_chunk, uint8_t seq_id,
                             uint8_t *lbuf, size_t lbuf_cap,
                             uint32_t *out_token) {
    if (out_token) *out_token = 0;
    if (coord_round_send(ws, n, req_id, pos0, tokens, n_tokens, first_chunk, seq_id) != 0)
        return -1;
    return coord_round_recv(ws, n, lbuf, lbuf_cap, out_token, NULL);
}

/* Back-compat single-token wrapper (warmup loop + decode loop use this). */
static int coord_decode_step(idletoken_worker_info *ws, int n,
                             uint64_t req_id,
                             uint32_t pos, uint32_t in_token, uint8_t seq_id,
                             uint8_t *lbuf, size_t lbuf_cap,
                             uint32_t *out_token) {
    return coord_infer_round(ws, n, req_id, pos, &in_token, 1, 0, seq_id,
                             lbuf, lbuf_cap, out_token);
}

/* SIGPIPE flooding the process when a client closes early is annoying; the
 * sendall path checks return codes so we just ignore the signal.
 * Windows has no SIGPIPE at all — a send() to a closed socket just returns
 * WSAECONNRESET, which the sendall path already handles. */
static void ignore_sigpipe(void) {
#ifndef _WIN32
    struct sigaction sa = {0};
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, NULL);
#endif
}

/* ---- Request intake queue (E1, docs/scheduler-design.md §6-E1) ------------
 *
 * This used to be a straight line: `accept -> handle -> close`. **Concurrent
 * requests were absorbed by the TCP backlog alone**, so the platform could
 * neither see how many were queued nor be told "I am busy". It is now two
 * halves:
 *   intake thread   accept -> enqueue (queue full -> immediate 429 plus an
 *                   estimated wait, so the platform picks another machine)
 *   executor thread dequeue -> handle_http_request (**still one at a time**)
 *
 * Why execution stays serial: real concurrent execution means advancing rounds
 * of several sequences out of phase inside the PP pipeline (E3 micro-batching),
 * which requires rewriting the round driver and the workers' synchronization
 * points. E1 only makes concurrency **explicit**: queue depth is observable, a
 * full queue can be refused honestly, and E3 gets its precondition of having
 * more than one request in hand (decode within a single sequence is inherently
 * serial, and with no second request there is no bubble to fill).
 *
 * The thread boundary is deliberately narrow: only this fd queue crosses
 * threads. `g_slots`, `g_stats` and the prefix history are still **touched by
 * the executor thread only**, so they need no locks -- deliberately, because
 * putting locks on the inference path is far too easy to get wrong.
 *
 * The queue length cap is **purely a memory backstop** (a home machine queueing
 * too much would OOM), not a scheduling input: the latency-budget decision lives
 * on the platform side (§4.2b). The coordinator's job is to say "full" when it
 * is full. */
#define COORD_INTAKE_MAX 8

typedef struct {
    int      fds[COORD_INTAKE_MAX];
    long long at_ms[COORD_INTAKE_MAX];   /* enqueue timestamp, for the real queueing delay */
    int      head, len;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int      cap;          /* effective cap = min(COORD_INTAKE_MAX, seq_slots*2) */
    int      stop;
} coord_intake;

static coord_intake g_intake;

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void intake_init(int cap) {
    memset(&g_intake, 0, sizeof(g_intake));
    g_intake.cap = cap < 1 ? 1 : (cap > COORD_INTAKE_MAX ? COORD_INTAKE_MAX : cap);
    pthread_mutex_init(&g_intake.mu, NULL);
    pthread_cond_init(&g_intake.cv, NULL);
}

/* Current queue length (for /idletoken/v1/stats; an instantaneous snapshot, lock held
 * briefly). */
static int intake_depth(void) {
    pthread_mutex_lock(&g_intake.mu);
    int d = g_intake.len;
    pthread_mutex_unlock(&g_intake.mu);
    return d;
}

/* Enqueue. Returns -1 when the queue is full; the caller is responsible for
 * replying 429 and closing. */
static int intake_push(int fd) {
    pthread_mutex_lock(&g_intake.mu);
    if (g_intake.len >= g_intake.cap) {
        pthread_mutex_unlock(&g_intake.mu);
        return -1;
    }
    int slot = (g_intake.head + g_intake.len) % COORD_INTAKE_MAX;
    g_intake.fds[slot]   = fd;
    g_intake.at_ms[slot] = now_ms();
    g_intake.len++;
    pthread_cond_signal(&g_intake.cv);
    pthread_mutex_unlock(&g_intake.mu);
    return 0;
}

/* Dequeue, blocking until there is work or stop is set. `*queued_ms` returns
 * the real queueing delay. */
/* Non-blocking variant: returns -1 immediately when the queue is empty. The
 * executor must not block here while it already has requests in flight -- doing
 * so would starve those in-flight requests until new work arrived. */
static int intake_try_pop(long long *queued_ms) {
    pthread_mutex_lock(&g_intake.mu);
    if (g_intake.len == 0) { pthread_mutex_unlock(&g_intake.mu); return -1; }
    int fd = g_intake.fds[g_intake.head];
    long long at = g_intake.at_ms[g_intake.head];
    g_intake.head = (g_intake.head + 1) % COORD_INTAKE_MAX;
    g_intake.len--;
    pthread_mutex_unlock(&g_intake.mu);
    if (queued_ms) *queued_ms = now_ms() - at;
    return fd;
}

static int intake_pop(long long *queued_ms) {
    pthread_mutex_lock(&g_intake.mu);
    while (g_intake.len == 0 && !g_intake.stop)
        pthread_cond_wait(&g_intake.cv, &g_intake.mu);
    if (g_intake.len == 0) { pthread_mutex_unlock(&g_intake.mu); return -1; }
    int fd = g_intake.fds[g_intake.head];
    long long at = g_intake.at_ms[g_intake.head];
    g_intake.head = (g_intake.head + 1) % COORD_INTAKE_MAX;
    g_intake.len--;
    pthread_mutex_unlock(&g_intake.mu);
    if (queued_ms) *queued_ms = now_ms() - at;
    return fd;
}

/* Live serving counters for GET /idletoken/v1/stats — the dashboard's "how is my
 * cluster doing" panel. Same trust level as /health (no token gate): counts
 * only, never content. Single-threaded HTTP loop → plain fields, no locking.
 * last_tok_per_s = decode-loop throughput of the most recent chat request
 * (excludes prefill; the first token is the prefill's sample). */
static struct {
    uint64_t  requests;         /* completed chat requests (both protocols) */
    uint64_t  in_tokens;
    uint64_t  out_tokens;
    uint64_t  cache_hits;       /* requests that hit KV prefix reuse (P1, see kv-cache-design.md) */
    /* EWMA of end-to-end service time (E1). The platform's cost function turns
     * this into "how long a request would wait on this machine" (§4.2b).
     * Previously the platform could only infer it from the round-trip latency it
     * observed itself, which folded in network time and queueing. */
    double    service_ms_ewma;
    /* EWMA of TTFT (**time to first token**). The platform's admission decision
     * is bounded by a TTFT budget (the performance targets table),
     * whereas service_ms_ewma is the duration of a **whole request** -- an order
     * of magnitude apart (generating a few hundred tokens takes tens of seconds,
     * while the 8K tier's budget is 5 seconds). The platform used to compare the
     * full duration against the TTFT budget, so any honestly-reporting real
     * machine was judged unable to keep up.
     * Definition: **start of execution -> first token**, excluding the
     * coordinator's own intake queueing. The platform accounts for queueing
     * separately as queue_depth x avg_service_ms, so including it here would
     * double count. */
    double    ttft_ms_ewma;
    uint64_t  queued_total_ms;  /* cumulative real queueing time; divide by requests for the mean wait */
    uint64_t  cached_tokens;    /* cumulative prefill tokens saved */
    long long started_at;       /* unix s, set when API serving starts */
    long long last_request_at;  /* unix s of the last completed request */
    double    last_tok_per_s;
} g_stats;

/* Listening fd for the intake thread (read by it alone; the executor thread
 * never touches it). */
static int g_intake_lfd = -1;

/**
 * The intake thread: accept and enqueue, nothing else. When the queue is full it
 * replies 429 **immediately**, with an estimated wait, so the platform picks
 * another machine -- "switching machines is cheaper than queueing" (§4.2b, first
 * principle). Letting the request sit in the backlog instead would let the
 * platform believe it still has a chance, and then time out with the rest.
 */
static void *intake_accept_thread(void *ud) {
    (void)ud;
    for (;;) {
        int cfd = idletoken_accept_tcp(g_intake_lfd);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            pthread_mutex_lock(&g_intake.mu);
            g_intake.stop = 1;
            pthread_cond_broadcast(&g_intake.cv);
            pthread_mutex_unlock(&g_intake.mu);
            return NULL;
        }
        if (intake_push(cfd) != 0) {
            /* Estimated wait = queued requests x mean service time. The platform
             * uses it to decide between retrying and switching machines. */
            double svc = g_stats.service_ms_ewma > 0 ? g_stats.service_ms_ewma : 1000.0;
            long long est = (long long)(svc * (double)g_intake.cap);
            /* Header and body are written separately and Content-Length comes
             * from strlen: a hardcoded length that does not match yields half a
             * response, which the client sees only as a truncated connection --
             * harder to diagnose than the 429 itself. */
            static const char busy_body[] = "{\"error\":{\"message\":\"coordinator busy\"}}";
            char hdr[256];
            int hl = snprintf(hdr, sizeof(hdr),
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n"
                "Retry-After: %lld\r\nX-IdleToken-Est-Wait-Ms: %lld\r\n"
                "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                (est + 999) / 1000, est, sizeof(busy_body) - 1);
            if (hl > 0 && hl < (int)sizeof(hdr)) {
                ssize_t w1 = write(cfd, hdr, (size_t)hl);            (void)w1;
                ssize_t w2 = write(cfd, busy_body, sizeof(busy_body) - 1); (void)w2;
            }
            close(cfd);
            fprintf(stderr, "coord: intake full (cap %d) -> 429, est_wait=%lldms\n",
                    g_intake.cap, est);
        }
    }
}


/* ---- KV prefix reuse (engine half of P1, docs/kv-cache-design.md §A/§6) ---
 *
 * Semantics: the coordinator remembers the token sequence actually materialized
 * in the workers' KV -- the previous request's prompt plus the generated tokens
 * already fed back in, occupying absolute KV positions [base, base+len). If a
 * new request's prompt **strictly extends** that history (token for token equal
 * and longer), only the suffix is sent to prefill (pos0 = base+len), skipping
 * recomputation of the common prefix. That is the real basis for
 * cache_hit=true / cached_tokens=len in the sealed response contract, which the
 * platform discounts against.
 *
 * Why only strict extension: on the worker side the rolling state of
 * ds4_session's compressed layers (CSA/HCA) is append-only, and going backwards
 * requires ds4_session_rewind -- but the v1 wire protocol has no rewind message
 * and the worker never calls it. Append-only reuse is semantically identical to
 * what already happens (today we blindly append across requests anyway); the
 * only change is that we continue **only when the history really matches**,
 * which makes the regression risk zero. A mismatch takes the old path and
 * recomputes the whole prompt (KV positions keep growing in that case; the
 * ceiling and a rewind message are left for the real-hardware phase, see TODO).
 *
 * Token-level matching is naturally conservative about re-tokenization drift
 * (generated ids disagreeing with a re-rendered template): a divergence is a
 * miss, correctness is unaffected, we simply save a little less. The HTTP loop
 * is single threaded, so no locks. */
typedef struct {
    uint32_t *v;      /* the materialized token sequence */
    uint32_t  len;
    uint32_t  cap;
    uint32_t  base;   /* absolute KV position corresponding to v[0] */
    int       valid;
    /* --- v4 multi-sequence (E2) ----------------------------------------
     * Each slot corresponds to a ds4_session / ds4x_runner that **really
     * exists** on the worker side. pos is that slot's own KV cursor; there is no
     * global infer_pos any more, which was the root cause of "two interleaved
     * sessions overwrite each other".
     *
     * Hard rule: the coordinator's slot count must be <= the number of sessions
     * a worker can actually create, and seq_ids must correspond one to one. A
     * history describes what **that slot alone** materialized. If the
     * coordinator believes slot 1 has history while the worker's slot 1 session
     * does not exist or has been recycled into a different sequence, it will
     * skip prefill for tokens that were never computed -- and the output is
     * silently wrong. So slots are "declared by the coordinator, created lazily
     * by the worker per seq_id", with both sides aligned. */
    uint32_t  pos;    /* this slot's KV cursor (next absolute position to write) */
    uint64_t  used_at;/* monotonically increasing use counter, for LRU eviction */
    /* E3.3: this slot is held by a request in flight. When requests interleave,
     * **two of them must never select the same slot** -- each would write the KV
     * of the same worker sequence according to its own history, overwrite the
     * other, and then produce **silently** wrong output (no error, because each
     * side's own ledger is self-consistent). */
    int       in_flight;
} coord_kv_slot;

/* Slot count: 1 by default, so behaviour is identical to v3 (zero regression).
 * --seq-slots N enables multiple sequences. The protocol ceiling is
 * IDLETOKEN_MAX_SEQ_SLOTS; the real ceiling is KV memory, not the protocol. */
static coord_kv_slot g_slots[IDLETOKEN_MAX_SEQ_SLOTS];
/* 0 = auto (computed from resources after load, see coord_auto_seq_slots);
 * >0 = explicitly set by the user. The automatic value is computed once the
 * workers have reported LOAD_MODEL_DONE -- only then do we know how much memory
 * is left for KV. */
static int           g_n_slots = 0;
static int           g_n_slots_auto = 0;   /* the computed value, reported even when the user overrides it (the client shows "auto: N") */
/* E3.3: how many requests the executor drives at once (filling PP pipeline
 * bubbles). -1 = auto (the default, decided from the topology after load);
 * 0 = forced off; >0 = set by the user. The resolved effective value lives in
 * g_concurrent_live, and that is what `/idletoken/v1/stats` reports -- the platform's rate
 * limiting and wait estimates need the **real concurrency**, and using the KV
 * slot count would make a serially-executing machine look like it runs N in
 * parallel. */
static int           g_concurrent_decode = -1;
static int           g_concurrent_live   = 0;
static uint64_t      g_slot_clock = 0;

static void kv_slot_reset(int s) {
    free(g_slots[s].v);
    uint32_t keep_pos  = g_slots[s].pos;      /* the cursor must not rewind just because the history was invalidated */
    /* The LRU timestamp must not be cleared either. Clearing it makes the slot
     * we just used instantly look "least recently used", so every miss picks the
     * same slot and overwrites it again while the others stay idle -- multiple
     * sequences would be enabled in name only. (Caught by e2e: with
     * --seq-slots 4, six interleaved requests all landed on slot 0.) */
    uint64_t keep_used = g_slots[s].used_at;
    /* The in-flight flag must survive for the same reason: reset only
     * invalidates the history ledger, it does not mean the request finished.
     * Clearing it would mark a slot that is being written as free, and another
     * request would select it straight away -- the same class of bug as above. */
    int      keep_busy = g_slots[s].in_flight;
    memset(&g_slots[s], 0, sizeof(g_slots[s]));
    g_slots[s].pos       = keep_pos;
    g_slots[s].used_at   = keep_used;
    g_slots[s].in_flight = keep_busy;
}

static void kv_slots_reset_all(void) {
    for (int s = 0; s < IDLETOKEN_MAX_SEQ_SLOTS; s++) {
        free(g_slots[s].v);
        memset(&g_slots[s], 0, sizeof(g_slots[s]));
    }
    g_slot_clock = 0;
}

static int kv_hist_push_slot(int s, uint32_t tok) {
    coord_kv_slot *h = &g_slots[s];
    if (h->len == h->cap) {
        uint32_t ncap = h->cap ? h->cap * 2 : 1024;
        uint32_t *nv = realloc(h->v, (size_t)ncap * sizeof(uint32_t));
        if (!nv) { kv_slot_reset(s); return -1; }
        h->v = nv;
        h->cap = ncap;
    }
    h->v[h->len++] = tok;
    return 0;
}

/* Length of the common prefix between a new prompt and a slot's history
 * (token level, compared from the start). */
static uint32_t kv_hist_common_slot(int s, const int *prompt, uint32_t n_prompt) {
    const coord_kv_slot *h = &g_slots[s];
    uint32_t i = 0, lim = h->len < n_prompt ? h->len : n_prompt;
    while (i < lim && (uint32_t)prompt[i] == h->v[i]) i++;
    return i;
}

/* Whether this slot can serve the prompt as a strict extension (i.e. prefill
 * only the suffix). */
static int kv_slot_extends(int s, const int *prompt, uint32_t n_prompt) {
    const coord_kv_slot *h = &g_slots[s];
    return h->valid && h->len > 0 &&
           h->pos == h->base + h->len &&
           n_prompt > h->len &&
           kv_hist_common_slot(s, prompt, n_prompt) == h->len;
}

/**
 * How many bytes of KV **each additional** sequence costs (the denominator of
 * decision D2, §4.5b).
 *
 * Note this is the **marginal** cost, not `idletoken_model_overhead()`: that one
 * includes CUDA context and workspace, base costs **shared by all sequences**,
 * and multiplying them by the sequence count would compute far fewer slots than
 * the machine can actually run.
 *
 * It is only computable for models whose KV scales with the sequence count:
 * - GQA / MLA: `kv_bytes_per_token_layer x ctx x layers on this machine`.
 * - HYBRID (Qwen3.5 linear attention): only 1 layer in N is full attention and
 *   grows linearly with ctx; the linear layers hold a **fixed-size** recurrent
 *   state. Account for them per cycle, **rounding the full-attention layer count
 *   up** -- underestimating KV means OOM.
 * - DSV4: `kv_kind == IDLETOKEN_KV_DSV4` uses the per-tier calibrated table
 *   (docs/architecture.md §5), where base and KV are entangled and the
 *   **marginal term cannot be separated** -> return 0 and let the caller fall
 *   back to a single slot. Enabling multiple sequences for DSv4 first requires
 *   the **measured** `kv_bytes_per_token` report that §4.5 calls for; until then
 *   we would rather not enable it.
 */
static uint64_t coord_kv_bytes_per_seq(const idletoken_model_spec *m,
                                       uint32_t ctx, uint32_t layers) {
    if (!m || layers == 0 || ctx == 0) return 0;
    if (m->kv_kind == IDLETOKEN_KV_HYBRID) {
        const uint32_t iv = m->full_attn_interval ? m->full_attn_interval : 1;
        const uint64_t n_full = ((uint64_t)layers + iv - 1) / iv;
        const uint64_t n_lin  = (uint64_t)layers - n_full;
        return (uint64_t)m->kv_bytes_per_token_layer * ctx * n_full
             + (uint64_t)m->state_bytes_per_layer * n_lin;
    }
    if (m->kv_kind == IDLETOKEN_KV_MLA || m->kv_kind == IDLETOKEN_KV_GQA) {
        if (m->kv_bytes_per_token_layer == 0) return 0;
        return (uint64_t)m->kv_bytes_per_token_layer * ctx * layers;
    }
    return 0;   /* DSV4 calibrated table: no separable marginal KV */
}

/**
 * Compute the number of concurrent sequence slots from available resources
 * (decision D2, §4.5b).
 *
 * Users will generally never set this, so it needs a default that is **usable as
 * is**. Every input is something the coordinator already has, so the wire
 * protocol does not change:
 *   free_i = usable_i - used after load (from LOAD_MODEL_DONE)
 *   slots  = clamp( min_i floor(free_i x SHARE / kv_per_seq_i), 1, stages x 2 )
 *
 * Three deliberately conservative choices:
 * - **min, not sum**: under PP every stage stores the KV of its own layers for
 *   the same sequence, so the tightest stage decides how many can run. Summing
 *   would produce a number that cannot actually run.
 * - **cap at stages x 2**: filling the pipeline takes roughly N micro-batches in
 *   flight, and 2N leaves room for jitter; beyond that we only tie up KV without
 *   buying throughput (B1 saturates well before).
 * - **use only half of free** (`KV_SHARE`): concurrency also raises activation
 *   and workspace usage, which kv_per_seq does not cover. Better to run one
 *   sequence fewer than to have sequence N hit OOM halfway through.
 *
 * On HYBRID deployments (unified memory, or VRAM already full) the computation
 * uses RAM headroom -- which is exactly where such machines keep their KV.
 */
#define KV_SHARE_NUM 1
#define KV_SHARE_DEN 2

static int coord_auto_seq_slots(const idletoken_model_spec *mi,
                                const idletoken_worker_info *ws, int n, uint32_t ctx_size) {
    if (!mi || n <= 0 || ctx_size == 0) return 1;
    int best = INT_MAX;
    for (int i = 0; i < n; i++) {
        uint32_t layers = (ws[i].layer_hi > ws[i].layer_lo)
                        ? (uint32_t)(ws[i].layer_hi - ws[i].layer_lo) : 0;
        uint64_t kv_per_seq = coord_kv_bytes_per_seq(mi, ctx_size, layers);
        if (kv_per_seq == 0) return 1;   /* no marginal KV for this model: stick to one slot */
        uint64_t usable = ws[i].vram_usable;
        uint64_t used   = ws[i].vram_used_after;
        if (ws[i].unified || used > usable) {   /* unified memory, or VRAM is full: KV lands in host RAM */
            usable = ws[i].ram_usable;
            used   = ws[i].ram_used_after;
        }
        uint64_t freeb = usable > used ? usable - used : 0;
        int slots = (int)((freeb / KV_SHARE_DEN * KV_SHARE_NUM) / kv_per_seq);
        if (slots < best) best = slots;
    }
    if (best == INT_MAX) return 1;
    int cap = n * 2;
    if (cap > IDLETOKEN_MAX_SEQ_SLOTS) cap = IDLETOKEN_MAX_SEQ_SLOTS;
    if (best < 1) best = 1;
    if (best > cap) best = cap;
    return best;
}

/**
 * Select a slot for a prompt (the core decision of E2).
 *
 * 1. Prefer, among the slots that can **strictly extend**, the one with the
 *    longest history -- that is the only case where prefill is really saved.
 * 2. If none can extend, take the **least recently used** slot, discard its
 *    history and start over.
 *
 * Evicting by LRU rather than by "longest common prefix" is deliberate: when a
 * slot cannot extend, however long the common prefix is it saves not one token
 * (the compressed layers' rolling state is append-only and the wire protocol has
 * no rewind), so that information is noise here; LRU instead preserves the
 * active sessions that may still be extended on the next turn.
 */
/* Returns the slot to use; **returns -1 when every slot is in flight**, on which
 * the caller makes the request come back later rather than grabbing a slot that
 * is being written. */
/* Whether interleaved execution is worth enabling is decided by the
 * **topology**, not by the slot count:
 *   - One stage: there is no pipeline bubble at all.
 *   - Several stages on the same machine: they contend for the same GPU, the
 *     "bubble" is not idle hardware, and interleaving only gets in the way
 *     (measured 0.91x with two workers on one machine -- slower than serial).
 *   - Several stages across >=2 machines: while the coordinator waits for A's
 *     logits, stage 0 really is idle (measured 1.41x across machines, saturating
 *     the end-to-end / slowest-stage bound).
 * Concurrency is min(stage count, slot count): covering the pipeline depth is
 * enough, and more only adds queueing. */
static int coord_auto_concurrent_decode(const idletoken_worker_info *ws, int n) {
    if (!ws || n < 2) return 0;
    int distinct = 0;
    for (int i = 0; i < n; i++) {
        int seen = 0;
        for (int j = 0; j < i; j++)
            if (!strcmp(ws[i].hostname, ws[j].hostname)) { seen = 1; break; }
        if (!seen) distinct++;
    }
    if (distinct < 2) return 0;
    const int slots = g_n_slots > 0 ? g_n_slots : 1;
    const int c = n < slots ? n : slots;
    return c >= 2 ? c : 0;
}

static int kv_pick_slot(const int *prompt, uint32_t n_prompt) {
    int best = -1;
    uint32_t best_len = 0;
    /* g_n_slots may still be 0 (auto not derived yet, e.g. the --tokenizer-only
     * path), in which case treat it as 1. */
    const int nslots = g_n_slots > 0 ? g_n_slots : 1;
    for (int s = 0; s < nslots; s++) {
        if (g_slots[s].in_flight) continue;
        if (!kv_slot_extends(s, prompt, n_prompt)) continue;
        if (best < 0 || g_slots[s].len > best_len) { best = s; best_len = g_slots[s].len; }
    }
    if (best >= 0) return best;
    int lru = -1;
    for (int s = 0; s < nslots; s++) {
        if (g_slots[s].in_flight) continue;
        if (lru < 0 || g_slots[s].used_at < g_slots[lru].used_at) lru = s;
    }
    return lru;   /* -1 = every slot is in flight */
}

/* ---- Parsing the messages array (multi-turn) ------------------------------
 * A hand-written scan, in the same "fixed-shape JSON extraction" style as the
 * rest of this file: find "messages":[ and cut out each top-level {...} object
 * by brace depth (string- and escape-aware), then use
 * idletoken_http_json_extract_str inside the object for "role" and "content"
 * (when content is an Anthropic array of text blocks, take the first "text" --
 * a v0.1 single-block limitation; multi-block support comes later). */
typedef int (*chat_msg_fn)(void *ud, const char *role, const char *content);

static int for_each_chat_message(const char *body, size_t len, chat_msg_fn fn, void *ud) {
    if (!body || len == 0) return 0;
    /* Locate the '[' following the "messages" key. */
    size_t i = 0, start = 0;
    int found = 0;
    for (i = 0; i + 10 <= len; i++) {
        if (body[i] == '"' && !memcmp(body + i + 1, "messages\"", 9)) {
            size_t p = i + 10;
            while (p < len && (body[p] == ' ' || body[p] == ':' || body[p] == '\t' ||
                               body[p] == '\n' || body[p] == '\r')) p++;
            if (p < len && body[p] == '[') { start = p + 1; found = 1; break; }
        }
    }
    if (!found) return 0;

    int count = 0;
    size_t p = start;
    while (p < len) {
        /* Skip to the next object, or the end of the array. */
        while (p < len && body[p] != '{' && body[p] != ']') p++;
        if (p >= len || body[p] == ']') break;
        /* Cut the object by brace depth (string- and escape-aware). */
        size_t obj0 = p;
        int depth = 0, in_str = 0, esc = 0;
        while (p < len) {
            char c = body[p];
            if (esc)            { esc = 0; }
            else if (c == '\\') { esc = 1; }
            else if (in_str)    { if (c == '"') in_str = 0; }
            else if (c == '"')  { in_str = 1; }
            else if (c == '{')  { depth++; }
            else if (c == '}')  { if (--depth == 0) { p++; break; } }
            p++;
        }
        if (depth != 0) return -1;  /* truncated or malformed */
        size_t obj_len = p - obj0;

        char role[32] = "";
        char *content = malloc(obj_len + 1);
        if (!content) return -1;
        content[0] = '\0';
        idletoken_http_json_extract_str(body + obj0, obj_len, "role", role, sizeof(role));
        int have = idletoken_http_json_extract_str(body + obj0, obj_len, "content",
                                                   content, obj_len + 1) == 0;
        if (!have) {
            /* Anthropic content block array: {"content":[{"type":"text","text":"..."}]} */
            have = idletoken_http_json_extract_str(body + obj0, obj_len, "text",
                                                   content, obj_len + 1) == 0;
        }
        /* Keyed on "was there a content field", not on "is it non-empty". An
         * assistant turn whose text is "" is a real turn -- a generation the
         * user stopped before its first token, which stays in the stored
         * transcript. Dropping it silently deleted one side of the exchange and
         * handed the model two consecutive user messages, so from that turn on
         * the conversation it saw was not the conversation on screen. */
        if (role[0] && have) {
            if (fn(ud, role, content) != 0) { free(content); return -1; }
            count++;
        }
        free(content);
    }
    return count;
}

/* Multi-turn template callback for handle_http_request: feed each message into
 * the ds4 chat template, and keep a summary of the first message's content for
 * the log along the way. */
typedef struct {
    ds4_engine *e;
    ds4_tokens *t;
    char       *first;
    size_t      fcap;
} coord_chat_ud;

static int coord_chat_append_cb(void *u, const char *role, const char *content) {
    coord_chat_ud *ud = (coord_chat_ud *)u;
    ds4_chat_append_message(ud->e, ud->t, role, content);
    if (ud->first && !ud->first[0]) snprintf(ud->first, ud->fcap, "%s", content);
    return 0;
}

/* ds4x chat collector: gather (role, content) copies for ds4x_tok_chat_apply,
 * which renders the ChatML prompt itself (the ds4 chat template lives inside
 * ds4_engine; ds4x has no engine, only a GGUF tokenizer). Bounded at 128 msgs. */
#define COORD_XCHAT_MAX 128
typedef struct {
    char  *roles[COORD_XCHAT_MAX];
    char  *contents[COORD_XCHAT_MAX];
    int    n;
    char  *first; size_t fcap;
} coord_xchat_ud;

static int coord_xchat_cb(void *u, const char *role, const char *content) {
    coord_xchat_ud *c = (coord_xchat_ud *)u;
    if (c->n >= COORD_XCHAT_MAX) return 0;
    c->roles[c->n]    = strdup(role ? role : "user");
    c->contents[c->n] = strdup(content ? content : "");
    if (!c->roles[c->n] || !c->contents[c->n]) { free(c->roles[c->n]); free(c->contents[c->n]); return 0; }
    c->n++;
    if (c->first && !c->first[0]) snprintf(c->first, c->fcap, "%s", content ? content : "");
    return 0;
}
static void coord_xchat_free(coord_xchat_ud *c) {
    for (int i = 0; i < c->n; i++) { free(c->roles[i]); free(c->contents[i]); }
    c->n = 0;
}

/* Render a chat request body into prompt token ids, applying the chat template
 * — the full multi-turn messages array when present (the precondition for KV
 * prefix reuse: each turn of a continuing conversation must render the same
 * prefix). With no messages array we fall back to the old single-content path,
 * for compatibility with older clients and scripts.
 *
 * Extracted from the chat handler so that `POST /v1/messages/count_tokens` can
 * answer with the count of the prompt a REAL request would prefill. Counting
 * the raw message text instead would be a second implementation of the same
 * thing, off by the chat template's framing, and its whole purpose is context
 * budgeting — a count that does not match what actually gets prefilled is worse
 * than no count at all.
 *
 * Returns 0 on success (caller owns `*prompt` and must ds4_tokens_free it),
 * -1 when the body carried no usable message — the caller answers 400, because
 * this helper has no socket and must not decide how a request fails. */
static int coord_encode_request_prompt(ds4_engine *coord_engine,
                                       ds4x_tokenizer *coord_xtok,
                                       const uint8_t *body, size_t body_len,
                                       int is_anthropic, ds4_tokens *prompt,
                                       char *first_text, size_t first_cap,
                                       int *out_n_msgs) {
    int n_msgs = 0;
  if (coord_xtok) {
    /* ds4x: collect messages, render the ChatML prompt via the tokenizer, and
     * push the ids into `prompt` (the rest of the handler is token-id agnostic
     * — KV reuse, prefill/decode, sampling all work unchanged). */
    coord_xchat_ud xc = {0};
    xc.first = first_text; xc.fcap = first_cap;
    if (body && body_len > 0) {
        if (is_anthropic) {
            char *sys = malloc(body_len + 1);
            if (sys) {
                sys[0] = '\0';
                idletoken_http_json_extract_str((const char *)body, body_len,
                                             "system", sys, body_len + 1);
                if (sys[0] && xc.n < COORD_XCHAT_MAX) {
                    xc.roles[xc.n] = strdup("system");
                    xc.contents[xc.n] = strdup(sys);
                    if (xc.roles[xc.n] && xc.contents[xc.n]) xc.n++;
                }
                free(sys);
            }
        }
        for_each_chat_message((const char *)body, body_len, coord_xchat_cb, &xc);
    }
    n_msgs = xc.n;
    if (xc.n == 0) {
        /* single-content fallback (old clients) */
        char user_text[2048] = "";
        if (body && body_len > 0)
            idletoken_http_json_extract_str((const char *)body, body_len,
                                         "content", user_text, sizeof(user_text));
        if (!user_text[0]) { coord_xchat_free(&xc); return -1; }
        snprintf(first_text, first_cap, "%s", user_text);
        xc.roles[0] = strdup("user"); xc.contents[0] = strdup(user_text);
        xc.n = (xc.roles[0] && xc.contents[0]) ? 1 : 0;
    }
    const uint32_t idcap = 16384;
    int32_t *ids = (int32_t *)malloc((size_t)idcap * sizeof(int32_t));
    int64_t nid = (ids && xc.n > 0)
        ? ds4x_tok_chat_apply(coord_xtok, (const char *const *)xc.roles,
                              (const char *const *)xc.contents, (uint32_t)xc.n,
                              1, ids, idcap)
        : -1;
    for (int64_t i = 0; i < nid && i < (int64_t)idcap; i++)
        ds4_tokens_push(prompt, (int)ids[i]);
    free(ids);
    coord_xchat_free(&xc);
  } else {
    if (body && body_len > 0) {
        ds4_chat_begin(coord_engine, prompt);
        if (is_anthropic) {
            /* Anthropic's system prompt is a top-level field, not part of messages. */
            char *sys = malloc(body_len + 1);
            if (sys) {
                sys[0] = '\0';
                idletoken_http_json_extract_str((const char *)body, body_len,
                                             "system", sys, body_len + 1);
                if (sys[0]) ds4_chat_append_message(coord_engine, prompt, "system", sys);
                free(sys);
            }
        }
        coord_chat_ud ud = { coord_engine, prompt, first_text, first_cap };
        n_msgs = for_each_chat_message((const char *)body, body_len,
                                       coord_chat_append_cb, &ud);
        if (n_msgs > 0) {
            ds4_chat_append_assistant_prefix(coord_engine, prompt, DS4_THINK_NONE);
        }
    }
    if (n_msgs <= 0) {
        /* The old path: a single "content" field (the first occurrence), as in
         * v0.1. */
        ds4_tokens_free(prompt);
        memset(prompt, 0, sizeof(*prompt));
        char user_text[2048] = "";
        if (body && body_len > 0) {
            idletoken_http_json_extract_str((const char *)body, body_len,
                                         "content", user_text, sizeof(user_text));
        }
        if (!user_text[0]) return -1;
        snprintf(first_text, first_cap, "%s", user_text);
        ds4_encode_chat_prompt(coord_engine, NULL, user_text, DS4_THINK_NONE, prompt);
    }
  }
    if (out_n_msgs) *out_n_msgs = n_msgs;
    return 0;
}

/* Backend-branched EOS id: ds4x tokenizer or ds4 engine. */
static int coord_tok_eos(ds4_engine *e, ds4x_tokenizer *xt) {
    return xt ? (int)ds4x_tok_eos(xt) : ds4_token_eos(e);
}
/* Backend-branched single-token detokenize into the caller's buf (always
 * NUL-terminated); returns the byte length written.
 * BOTH backends return heap memory (ds4x_tok_decode and ds4_token_text each
 * malloc), so this copies into `buf` and frees the source here — the caller
 * owns only its own buffer and must NOT free the result. */
static size_t coord_tok_text(ds4_engine *e, ds4x_tokenizer *xt, int tok,
                             char *buf, size_t cap) {
    if (cap == 0) return 0;
    char  *src = NULL;
    size_t n   = 0;
    if (xt) {
        int32_t id = tok;
        src = ds4x_tok_decode(xt, &id, 1, 0);
        if (src) n = strlen(src);
    } else {
        src = ds4_token_text(e, tok, &n);
    }
    if (!src) { buf[0] = '\0'; return 0; }
    if (n >= cap) n = cap - 1;
    memcpy(buf, src, n);
    buf[n] = '\0';
    free(src);
    return n;
}

/* Did this request come from the platform, or from a client on this LAN?
 *
 * Both arrive on this same endpoint: the platform agent opens the sealed
 * envelope and forwards the plaintext here over loopback (platform_agent.c),
 * which until the agent started marking it was indistinguishable from a request
 * a user sent directly.
 *
 * Overflow routing needs the distinction (docs/overflow-routing-design.md §2): a
 * job the platform dispatched is finished here or refused, and is never
 * forwarded back out. Only requests that started locally may ever be forwarded.
 *
 * A request with no marker counts as local, because a local client -- curl,
 * Claude Code, anything -- has no reason to send one. That puts the entire
 * weight of the rule on the agent actually setting the header, which is why the
 * gate asserts against the real agent rather than against this function alone.
 * Nothing reads this yet; O0 records the fact and changes no behaviour. */
static int request_from_platform(const idletoken_http_req *req) {
    char origin[32] = "";
    if (idletoken_http_header_get(req, IDLETOKEN_HDR_ORIGIN, origin, sizeof(origin)) != 0)
        return 0;
    return strcmp(origin, IDLETOKEN_ORIGIN_PLATFORM) == 0;
}

/* --selftest: unit tests for the two blocks of pure logic above (no GGUF or
 * engine dependency), runnable on a Mac as well as on real hardware. */
static int msg_collect_cb(void *ud, const char *role, const char *content) {
    char *buf = (char *)ud;
    size_t off = strlen(buf);
    snprintf(buf + off, 512 - off, "[%s:%s]", role, content);
    return 0;
}

static int coord_selftest(void) {
    int fails = 0;
#define ST(cond, name) do { \
        if (cond) fprintf(stderr, "selftest PASS %s\n", name); \
        else      { fprintf(stderr, "selftest FAIL %s\n", name); fails++; } \
    } while (0)

    /* Multi-turn parsing: role/content extraction, escapes, nested quotes,
     * Anthropic text blocks. */
    {
        char got[512] = "";
        const char *body =
            "{\"model\":\"m\",\"messages\":["
            "{\"role\":\"system\",\"content\":\"be {nice}\"},"
            "{\"role\":\"user\",\"content\":\"say \\\"hi\\\"\\nplease\"},"
            "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}"
            "],\"max_tokens\":8}";
        int n = for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(n == 3, "parse: 3 messages");
        ST(strstr(got, "[system:be {nice}]") != NULL, "parse: braces inside string");
        ST(strstr(got, "[user:say \"hi\"\nplease]") != NULL, "parse: escapes");
        ST(strstr(got, "[assistant:ok]") != NULL, "parse: anthropic text block");
    }
    /* What CONTINUING AN OLD CONVERSATION actually sends. A first turn is one
     * user message; every later turn also carries assistant messages holding a
     * real model's output — escapes, CRLF, reasoning tags, JSON-looking text.
     * That is the only structural difference between "new chat works" and "old
     * chat fails", so it is the shape worth pinning down. */
    {
        char got[512] = "";
        const char *body =
            "{\"messages\":["
            "{\"role\":\"user\",\"content\":\"什么是流水线并行？\"},"
            "{\"role\":\"assistant\",\"content\":\"<think>用户在问 PP。</think>按层切分。\"},"
            "{\"role\":\"user\",\"content\":\"那张量并行呢？\"}"
            "]}";
        int n = for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(n == 3, "resume: 3 messages incl. an assistant turn");
        ST(strstr(got, "[assistant:<think>用户在问 PP。</think>按层切分。]") != NULL,
           "resume: reasoning tags and CJK survive verbatim");
    }
    {   /* A reply that talks about JSON. The object cutter counts braces and the
         * key search takes the first hit, so a reply containing "role"/"content"
         * and braces must not be able to steer either. */
        char got[512] = "";
        const char *body =
            "{\"messages\":["
            "{\"role\":\"assistant\",\"content\":\"send {\\\"role\\\":\\\"user\\\"} in the body\"}"
            "]}";
        int n = for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(n == 1, "resume: a reply quoting JSON is still one message");
        ST(strstr(got, "[assistant:send {\"role\":\"user\"} in the body]") != NULL,
           "resume: quoted JSON in a reply does not steer role/content extraction");
    }
    {   /* CRLF and a unicode escape: both appear in real replies, and neither is
         * in the extractor's escape table. */
        char got[512] = "";
        const char *body =
            "{\"messages\":["
            "{\"role\":\"assistant\",\"content\":\"line one\\r\\nline two\\u00e9\"}"
            "]}";
        for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(strstr(got, "line one\r\nline two") != NULL,
           "resume: CRLF in a reply survives as CRLF");
        ST(strstr(got, "u00e9") == NULL,
           "resume: \\uXXXX is decoded, not spelled out into the prompt");
    }
    {   /* An assistant turn with empty text -- a generation that was stopped
         * before its first token, which stays in the stored transcript. */
        char got[512] = "";
        const char *body =
            "{\"messages\":["
            "{\"role\":\"user\",\"content\":\"a\"},"
            "{\"role\":\"assistant\",\"content\":\"\"},"
            "{\"role\":\"user\",\"content\":\"b\"}"
            "]}";
        int n = for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(n == 3, "resume: an empty assistant turn is not silently dropped");
    }

    {   /* Python's json.dumps escapes ALL non-ASCII by default, so this is what
         * an ordinary OpenAI-compatible client sends for Chinese. It reached the
         * model as the literal text "u4f60u597d" until 2026-08-12. */
        char got[512] = "";
        const char *body =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\\u4f60\\u597d\"}]}";
        for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(strstr(got, "[user:你好]") != NULL, "parse: ensure_ascii CJK decodes to UTF-8");
    }
    {   /* Emoji: a surrogate PAIR, which is two \u escapes for one character. */
        char got[512] = "";
        const char *body =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\\ud83d\\ude80 go\"}]}";
        for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(strstr(got, "[user:\xf0\x9f\x9a\x80 go]") != NULL,
           "parse: surrogate pair becomes one UTF-8 code point");
    }
    {   /* A lone high surrogate is not a character; it must not become invalid
         * UTF-8, because the tokenizer downstream has to be able to read it. */
        char got[512] = "";
        const char *body =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\\ud83d ok\"}]}";
        for_each_chat_message(body, strlen(body), msg_collect_cb, got);
        ST(strstr(got, "\xef\xbf\xbd") != NULL, "parse: unpaired surrogate -> U+FFFD");
    }

    {   /* No messages array -> 0 (falls back to the old single-turn path). */
        const char *body = "{\"content\":\"hello\"}";
        char got[512] = "";
        ST(for_each_chat_message(body, strlen(body), msg_collect_cb, got) == 0,
           "parse: no messages array -> 0");
    }
    {   /* Truncated or malformed -> -1. */
        const char *body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"";
        char got[512] = "";
        ST(for_each_chat_message(body, strlen(body), msg_collect_cb, got) == -1,
           "parse: truncated -> error");
    }

    /* Prefix history: push/common/reset (single slot, i.e. v3 behaviour). */
    {
        kv_slots_reset_all();
        g_n_slots = 1;
        g_slots[0].base = 0; g_slots[0].valid = 1;
        for (uint32_t t = 100; t < 110; t++) kv_hist_push_slot(0, t);
        g_slots[0].pos = 10;                     /* base+len: only then is the cursor "unmoved by anything else" */
        int ext[14], div[14];
        for (int k = 0; k < 14; k++) { ext[k] = 100 + k; div[k] = 100 + k; }
        div[5] = 999;
        ST(kv_hist_common_slot(0, ext, 14) == 10, "hist: strict extension matches all 10");
        ST(kv_hist_common_slot(0, div, 14) == 5,  "hist: divergence stops at 5");
        ST(kv_hist_common_slot(0, ext, 4)  == 4,  "hist: shorter prompt caps at its len");
        ST(kv_slot_extends(0, ext, 14) == 1,      "hist: extends -> hit");
        ST(kv_slot_extends(0, div, 14) == 0,      "hist: divergence -> miss");
        ST(kv_slot_extends(0, ext, 10) == 0,      "hist: equal length is not an extension");
        g_slots[0].pos = 11;                      /* cursor moved by something else -> do not risk reuse */
        ST(kv_slot_extends(0, ext, 14) == 0,      "hist: moved cursor -> miss");
        g_slots[0].pos = 10;
        kv_slot_reset(0);
        ST(g_slots[0].len == 0 && !g_slots[0].valid, "hist: reset clears");
        ST(g_slots[0].pos == 10, "hist: reset keeps the KV cursor (it must not rewind)");
        g_slots[0].used_at = 7;
        kv_slot_reset(0);
        ST(g_slots[0].used_at == 7, "hist: reset keeps the LRU stamp (else one slot hogs everything)");
    }

    /* D2 automatic slot sizing (§4.5b). Driven by an **explicitly constructed
     * spec** rather than the global model: the first version wrote
     * `if (coord_model()->kv_bytes_per_token_layer > 0)`, and that field is 0 for
     * the default model (DSv4 uses the calibrated table), so the whole block was
     * **silently skipped** while the selftest still reported ALL PASS.
     * A conditional self-test is no self-test. */
    {
        const uint32_t ctx = 8192;
        idletoken_model_spec gqa;
        memset(&gqa, 0, sizeof(gqa));
        gqa.kv_kind = IDLETOKEN_KV_GQA;
        gqa.kv_bytes_per_token_layer = 4096;
        const uint64_t per_seq = (uint64_t)4096 * ctx * 10;   /* 10 layers */

        idletoken_worker_info tw[2];
        memset(tw, 0, sizeof(tw));
        for (int i = 0; i < 2; i++) { tw[i].layer_lo = 0; tw[i].layer_hi = 10; }
        /* A has room for 8, B for only 2 (we spend half the headroom, hence the
         * doubled free memory) -> take the min. */
        tw[0].vram_usable = per_seq * 16;
        tw[1].vram_usable = per_seq * 4;
        ST(coord_auto_seq_slots(&gqa, tw, 2, ctx) == 2, "auto-slots: min over workers, not sum");
        tw[1].vram_usable = per_seq * 1000;
        ST(coord_auto_seq_slots(&gqa, tw, 2, ctx) == 4, "auto-slots: capped at stages x 2");
        tw[0].vram_used_after = tw[0].vram_usable;
        tw[1].vram_used_after = tw[1].vram_usable;
        ST(coord_auto_seq_slots(&gqa, tw, 2, ctx) == 1, "auto-slots: never returns 0");
        /* Spending only half the headroom: room for exactly 2 yields 1. */
        memset(tw, 0, sizeof(tw));
        for (int i = 0; i < 2; i++) { tw[i].layer_lo = 0; tw[i].layer_hi = 10; }
        tw[0].vram_usable = per_seq * 2; tw[1].vram_usable = per_seq * 2;
        ST(coord_auto_seq_slots(&gqa, tw, 2, ctx) == 1, "auto-slots: only spends half the free memory");
        /* Unified memory (DGX/Spark) reads the RAM budget. */
        memset(tw, 0, sizeof(tw));
        tw[0].layer_lo = 0; tw[0].layer_hi = 10;
        tw[0].unified = 1; tw[0].ram_usable = per_seq * 8;
        ST(coord_auto_seq_slots(&gqa, tw, 1, ctx) == 2, "auto-slots: unified memory reads the RAM budget");

        /* DSv4's calibrated table has no separable marginal KV -> fall back to a
         * single slot, never guess. */
        idletoken_model_spec dsv4;
        memset(&dsv4, 0, sizeof(dsv4));
        dsv4.kv_kind = IDLETOKEN_KV_DSV4;
        memset(tw, 0, sizeof(tw));
        tw[0].layer_lo = 0; tw[0].layer_hi = 10; tw[0].vram_usable = per_seq * 1000;
        ST(coord_auto_seq_slots(&dsv4, tw, 1, ctx) == 1,
           "auto-slots: DSv4 calibrated table has no marginal KV -> stay at 1");
        ST(coord_kv_bytes_per_seq(&dsv4, ctx, 10) == 0, "auto-slots: DSv4 marginal KV is undefined");

        /* HYBRID: only 1 layer in `interval` is full attention and grows with
         * ctx; the linear layers hold a fixed-size state. */
        idletoken_model_spec hyb;
        memset(&hyb, 0, sizeof(hyb));
        hyb.kv_kind = IDLETOKEN_KV_HYBRID;
        hyb.kv_bytes_per_token_layer = 4096;
        hyb.state_bytes_per_layer = 1000;
        hyb.full_attn_interval = 4;
        /* 8 layers -> 2 full-attention layers (rounded up) + 6 linear layers. */
        ST(coord_kv_bytes_per_seq(&hyb, ctx, 8) == (uint64_t)4096 * ctx * 2 + 1000ull * 6,
           "auto-slots: hybrid charges ctx only to the full-attention layers");
        ST(coord_kv_bytes_per_seq(&hyb, ctx, 8) < coord_kv_bytes_per_seq(&gqa, ctx, 8),
           "auto-slots: hybrid is cheaper than pure GQA at the same layer count");
    }

    /* v4 multi-sequence slot selection (E2): two interleaved sessions each keep
     * their own prefix. */
    {
        kv_slots_reset_all();
        g_n_slots = 4;
        /* Slot 0 = session A (100..109), slot 2 = session B (200..204). */
        g_slots[0].valid = 1; g_slots[0].base = 0;
        for (uint32_t t = 100; t < 110; t++) kv_hist_push_slot(0, t);
        g_slots[0].pos = 10; g_slots[0].used_at = 1;
        g_slots[2].valid = 1; g_slots[2].base = 0;
        for (uint32_t t = 200; t < 205; t++) kv_hist_push_slot(2, t);
        g_slots[2].pos = 5; g_slots[2].used_at = 2;

        int a_next[12], b_next[7], fresh[3] = { 900, 901, 902 };
        for (int k = 0; k < 12; k++) a_next[k] = 100 + k;
        for (int k = 0; k < 7;  k++) b_next[k] = 200 + k;
        ST(kv_pick_slot(a_next, 12) == 0, "slots: A's continuation picks A's slot");
        ST(kv_pick_slot(b_next, 7)  == 2, "slots: B's continuation picks B's slot");
        /* Interleaving must not clobber either: asking A, B, A in a row all hit. */
        ST(kv_pick_slot(a_next, 12) == 0 && kv_pick_slot(b_next, 7) == 2 &&
           kv_pick_slot(a_next, 12) == 0, "slots: interleaved A/B/A all hit");
        /* A brand-new session extends nothing, so the least recently used empty
         * slot is chosen (1 or 3, both with used_at=0). */
        int nsel = kv_pick_slot(fresh, 3);
        ST(nsel == 1 || nsel == 3, "slots: fresh prompt evicts an LRU slot");
        ST(kv_slot_extends(nsel, fresh, 3) == 0, "slots: LRU pick is a miss (full prefill)");
        /* When two slots can extend, take the longer history (it saves more). */
        kv_slots_reset_all();
        g_n_slots = 2;
        for (int sl = 0; sl < 2; sl++) {
            g_slots[sl].valid = 1; g_slots[sl].base = 0;
        }
        for (uint32_t t = 100; t < 103; t++) kv_hist_push_slot(0, t);   /* 3 tok */
        g_slots[0].pos = 3;
        for (uint32_t t = 100; t < 108; t++) kv_hist_push_slot(1, t);   /* 8 tok, same prefix */
        g_slots[1].pos = 8;
        ST(kv_pick_slot(a_next, 12) == 1, "slots: prefers the longer reusable history");

        /* Consecutive brand-new sessions must rotate across slots rather than
         * hammering the same one (which is what happened when reset cleared
         * used_at). This simulates the real write-back path: pick a slot, stamp
         * it, miss, kv_slot_reset, write the new history. */
        kv_slots_reset_all();
        g_n_slots = 3;
        int seen[3] = { 0, 0, 0 };
        for (int r = 0; r < 3; r++) {
            int fresh_prompt[4] = { 500 + r * 10, 501 + r * 10, 502 + r * 10, 503 + r * 10 };
            int sl = kv_pick_slot(fresh_prompt, 4);
            g_slots[sl].used_at = (uint64_t)(r + 1);
            kv_slot_reset(sl);                 /* miss: invalidate the old history */
            g_slots[sl].valid = 1; g_slots[sl].base = 0;
            for (int k = 0; k < 4; k++) kv_hist_push_slot(sl, (uint32_t)fresh_prompt[k]);
            g_slots[sl].pos = 4;
            if (sl >= 0 && sl < 3) seen[sl] = 1;
        }
        ST(seen[0] && seen[1] && seen[2], "slots: consecutive fresh sessions rotate across slots");

        /* --- E3.3: in-flight slots are mutually exclusive ------------------
         * If two interleaved requests select the same slot, each writes the KV
         * of the same worker sequence according to its own history, they clobber
         * each other, and the output is **silently** wrong. These three
         * assertions are a hard constraint, not an optimization. */
        kv_slots_reset_all();
        g_n_slots = 3;
        int p_a[4] = { 900, 901, 902, 903 };
        /* Build A's history in slot 0, then mark it in flight. */
        int sa = kv_pick_slot(p_a, 4);
        g_slots[sa].valid = 1; g_slots[sa].base = 0;
        for (int k = 0; k < 4; k++) kv_hist_push_slot(sa, (uint32_t)p_a[k]);
        g_slots[sa].pos = 4;
        g_slots[sa].used_at = 1;
        int busy_next[5] = { 900, 901, 902, 903, 904 };
        ST(kv_pick_slot(busy_next, 5) == sa, "slots: while idle, A's continuation hits A's slot (control)");
        g_slots[sa].in_flight = 1;
        ST(kv_pick_slot(busy_next, 5) != sa,
           "slots: while A's slot is in flight it must not be chosen, prefix hit or not");
        /* reset only invalidates the history; it must not clear the in-flight
         * flag either (same reasoning as pos/used_at). */
        kv_slot_reset(sa);
        ST(g_slots[sa].in_flight == 1, "slots: kv_slot_reset preserves in_flight");
        ST(g_slots[sa].pos == 4,       "slots: kv_slot_reset preserves pos");
        /* All in flight -> -1, on which the caller replies 429 instead of
         * grabbing a slot. */
        for (int s = 0; s < 3; s++) g_slots[s].in_flight = 1;
        ST(kv_pick_slot(busy_next, 5) == -1, "slots: every slot in flight -> -1 (caller replies 429)");
        for (int s = 0; s < 3; s++) g_slots[s].in_flight = 0;
        ST(kv_pick_slot(busy_next, 5) >= 0, "slots: selectable again once released");

        kv_slots_reset_all();
        g_n_slots = 1;
    }

    /* Request origin (docs/overflow-routing-design.md §3). Overflow routing
     * hangs entirely off this predicate, so it is tested directly rather than
     * only through the agent. */
    {
        idletoken_http_req r;
        memset(&r, 0, sizeof(r));

        snprintf(r.headers, sizeof(r.headers),
                 "Host: 127.0.0.1:8000\r\n" IDLETOKEN_HDR_ORIGIN ": " IDLETOKEN_ORIGIN_PLATFORM "\r\n");
        ST(request_from_platform(&r) == 1, "origin: the agent's marker is recognized");

        /* Header names are case-insensitive on the wire; a proxy may rewrite
         * them, and the answer must not change. */
        snprintf(r.headers, sizeof(r.headers), "x-idletoken-origin: platform\r\n");
        ST(request_from_platform(&r) == 1, "origin: marker matches case-insensitively");

        snprintf(r.headers, sizeof(r.headers), "Content-Type: application/json\r\n");
        ST(request_from_platform(&r) == 0, "origin: no marker means local");

        r.headers[0] = 0;
        ST(request_from_platform(&r) == 0, "origin: no headers at all means local");

        /* A value we did not write must never be read as the platform: an
         * unknown marker is not a licence to skip the rule. */
        snprintf(r.headers, sizeof(r.headers), IDLETOKEN_HDR_ORIGIN ": platform-ish\r\n");
        ST(request_from_platform(&r) == 0, "origin: an unknown value is not the platform");

        /* A LAN client can spoof the marker, and that is harmless by
         * construction: claiming to be the platform only forfeits its own
         * request's right to be forwarded. It can never gain one. */
        snprintf(r.headers, sizeof(r.headers), IDLETOKEN_HDR_ORIGIN ": local\r\n");
        ST(request_from_platform(&r) == 0, "origin: an explicit local marker stays local");
    }

    /* --- Automatic decision on interleaved execution (E3.4) -------------
     * The test is the **topology**, not the slot count. Several stages on one
     * machine contend for the same GPU and measured 0.91x, slower than serial,
     * so "there is more than one stage" is not enough -- they must sit on
     * different machines. */
    {
        idletoken_worker_info w[3];
        memset(w, 0, sizeof(w));
        snprintf(w[0].hostname, sizeof(w[0].hostname), "boxA");
        snprintf(w[1].hostname, sizeof(w[1].hostname), "boxA");
        snprintf(w[2].hostname, sizeof(w[2].hostname), "boxB");
        const int saved = g_n_slots;
        g_n_slots = 4;
        ST(coord_auto_concurrent_decode(w, 1) == 0, "conc: one stage has no bubble -> 0");
        ST(coord_auto_concurrent_decode(w, 2) == 0, "conc: two stages on one machine share a GPU -> 0");
        ST(coord_auto_concurrent_decode(w, 3) == 3, "conc: 3 stages across machines -> 3");
        g_n_slots = 2;
        ST(coord_auto_concurrent_decode(w, 3) == 2, "conc: clamped by slot count (3 stages, 2 slots)");
        g_n_slots = 1;
        ST(coord_auto_concurrent_decode(w, 3) == 0, "conc: a single slot cannot interleave");
        ST(coord_auto_concurrent_decode(NULL, 3) == 0, "conc: no topology information -> conservative 0");
        g_n_slots = saved;
    }
    /* ASSIGN_PLAN v7 framing: the cluster salt must come off the wire at the
     * same offset it went on.
     *
     * This is the one part of N1b-1 that a Mac cannot check live -- the sealed
     * worker reports 0 usable memory, so planning refuses and ASSIGN_PLAN is
     * never sent. The risk it leaves is purely positional: a field order that
     * disagrees between the two sides. So both real sequences are replayed here
     * -- the coordinator's puts and the worker's gets, in their actual order --
     * and a mismatch shows up as a wrong salt rather than as a cluster that
     * mysteriously will not form on someone else's Linux box. */
    {
        uint8_t pbuf[2048];
        idletoken_buf pb;
        idletoken_buf_init(&pb, pbuf, sizeof(pbuf));
        uint8_t salt_in[IDLETOKEN_CLUSTER_SALT_BYTES];
        for (size_t i = 0; i < sizeof(salt_in); i++) salt_in[i] = (uint8_t)(0x40 + i);
        uint8_t ident[32]; memset(ident, 0xCD, sizeof(ident));
        uint8_t pad7z[7] = {0};

        /* --- exactly the coordinator's put order --- */
        idletoken_buf_put_u8(&pb, 2); idletoken_buf_put_u8(&pb, 1);
        idletoken_buf_put_u8(&pb, 0); idletoken_buf_put_u8(&pb, 1);
        idletoken_buf_put_u16(&pb, 7); idletoken_buf_put_u16(&pb, 13);
        idletoken_buf_put_u16(&pb, 43); idletoken_buf_put_u8(&pb, 2);
        idletoken_buf_put_u8(&pb, 0);
        idletoken_buf_put_u32(&pb, 8192); idletoken_buf_put_u32(&pb, 0);
        idletoken_buf_put_u8(&pb, 1);
        idletoken_buf_put_bytes(&pb, pad7z, 7);
        idletoken_buf_put_bytes(&pb, ident, 32);
        idletoken_buf_put_str(&pb, "some-model");
        idletoken_buf_put_str(&pb, "Q4_K_M");
        idletoken_buf_put_str(&pb, "/tmp/m.gguf");
        idletoken_buf_put_str(&pb, "10.0.0.1:1");
        idletoken_buf_put_str(&pb, "10.0.0.2:2");
        idletoken_buf_put_str(&pb, "10.0.0.3:3");
        idletoken_buf_put_bytes(&pb, salt_in, sizeof(salt_in));
        ST(!pb.err, "assign_plan v7: packs without overflow");

        /* --- exactly the worker's get order --- */
        idletoken_buf gb;
        idletoken_buf_init(&gb, pbuf, pb.pos);
        uint8_t u8a, u8b, u8c, u8d, u8e, u8f, u8g;
        uint16_t u16a, u16b, u16c;
        uint32_t u32a, u32b;
        uint8_t pad7_in[7], sha_in[32], salt_out[IDLETOKEN_CLUSTER_SALT_BYTES];
        char s1[64], s2[64], s3[256], s4[64], s5[64], s6[64];
        idletoken_buf_get_u8(&gb, &u8a); idletoken_buf_get_u8(&gb, &u8b);
        idletoken_buf_get_u8(&gb, &u8c); idletoken_buf_get_u8(&gb, &u8d);
        idletoken_buf_get_u16(&gb, &u16a); idletoken_buf_get_u16(&gb, &u16b);
        idletoken_buf_get_u16(&gb, &u16c); idletoken_buf_get_u8(&gb, &u8e);
        idletoken_buf_get_u8(&gb, &u8f);
        idletoken_buf_get_u32(&gb, &u32a); idletoken_buf_get_u32(&gb, &u32b);
        idletoken_buf_get_u8(&gb, &u8g);
        idletoken_buf_get_bytes(&gb, pad7_in, 7);
        idletoken_buf_get_bytes(&gb, sha_in, 32);
        idletoken_buf_get_str(&gb, s1, sizeof(s1));
        idletoken_buf_get_str(&gb, s2, sizeof(s2));
        idletoken_buf_get_str(&gb, s3, sizeof(s3));
        idletoken_buf_get_str(&gb, s4, sizeof(s4));
        idletoken_buf_get_str(&gb, s5, sizeof(s5));
        idletoken_buf_get_str(&gb, s6, sizeof(s6));
        idletoken_buf_get_bytes(&gb, salt_out, sizeof(salt_out));
        ST(!gb.err, "assign_plan v7: parses without error");
        ST(memcmp(salt_in, salt_out, sizeof(salt_in)) == 0,
           "assign_plan v7: the cluster salt survives the round trip at the right offset");
        ST(u16a == 7 && u16b == 13 && u32a == 8192 && !strcmp(s1, "some-model"),
           "assign_plan v7: the fields before the salt are unshifted");
    }
    /* INFER_BEGIN v7 framing, both forms. The wire-level gate
     * (G_NODE_CRYPT_TOKENS) needs two real paired processes and therefore a
     * Linux box; this covers the part that can be got wrong without one --
     * whether the encrypted token field is written and read at the same offset,
     * and whether the ids really leave the buffer. */
    {
        const uint32_t toks[4] = { 0x1234, 0x5678, 0x9abc, 0xdef0 };
        uint8_t ckey[IDLETOKEN_SESSION_KEY_BYTES];
        for (size_t i = 0; i < sizeof(ckey); i++) ckey[i] = (uint8_t)(i ^ 0x5a);

        idletoken_nodecrypt tx, rx;
        idletoken_nodecrypt_init(&tx, ckey, IDLETOKEN_NC_ID_COORD, 0);
        idletoken_nodecrypt_init(&rx, ckey, 0, IDLETOKEN_NC_ID_COORD);

        /* --- the coordinator's send path, verbatim --- */
        uint8_t wire[256];
        idletoken_buf wb;
        idletoken_buf_init(&wb, wire, sizeof(wire));
        idletoken_buf_put_u8(&wb, 1); idletoken_buf_put_u8(&wb, 1);
        idletoken_buf_put_u8(&wb, 0); idletoken_buf_put_u8(&wb, 0);
        idletoken_buf_put_u32(&wb, 0); idletoken_buf_put_u32(&wb, 4);
        idletoken_buf_put_u32(&wb, 0);
        uint8_t tplain[16], twrap[16 + IDLETOKEN_NODECRYPT_OVERHEAD];
        idletoken_buf tb; idletoken_buf_init(&tb, tplain, sizeof(tplain));
        for (int i = 0; i < 4; i++) idletoken_buf_put_u32(&tb, toks[i]);
        size_t wl = 0;
        ST(idletoken_nodecrypt_wrap(&tx, tplain, tb.pos, twrap, sizeof(twrap), &wl)
               == IDLETOKEN_NC_OK, "infer_begin v7: token ids wrap");
        idletoken_buf_put_bytes(&wb, twrap, wl);
        ST(!wb.err, "infer_begin v7: encrypted payload packs");

        /* The whole point: the ids must not survive anywhere in the frame. */
        int leaked = 0;
        for (size_t i = 0; i + 4 <= wb.pos; i++)
            if (memcmp(wire + i, tplain, 4) == 0) leaked = 1;
        ST(!leaked, "infer_begin v7: no plaintext token id survives in the frame");

        /* --- the worker's receive path, verbatim --- */
        idletoken_buf rb;
        idletoken_buf_init(&rb, wire, wb.pos);
        uint8_t p1, p2, p3, p4; uint32_t pos0_x, ntok_x, rsv_x;
        idletoken_buf_get_u8(&rb, &p1); idletoken_buf_get_u8(&rb, &p2);
        idletoken_buf_get_u8(&rb, &p3); idletoken_buf_get_u8(&rb, &p4);
        idletoken_buf_get_u32(&rb, &pos0_x); idletoken_buf_get_u32(&rb, &ntok_x);
        idletoken_buf_get_u32(&rb, &rsv_x);
        uint8_t rwire[16 + IDLETOKEN_NODECRYPT_OVERHEAD], rplain[16];
        idletoken_buf_get_bytes(&rb, rwire, ntok_x * 4 + IDLETOKEN_NODECRYPT_OVERHEAD);
        size_t got = 0;
        ST(ntok_x == 4 && !rb.err, "infer_begin v7: header fields read back unshifted");
        ST(idletoken_nodecrypt_unwrap(&rx, rwire, ntok_x * 4 + IDLETOKEN_NODECRYPT_OVERHEAD,
                                      rplain, sizeof(rplain), &got) == IDLETOKEN_NC_OK,
           "infer_begin v7: the peer decrypts it");
        idletoken_buf ob; idletoken_buf_init(&ob, rplain, got);
        uint32_t back[4] = {0};
        for (int i = 0; i < 4; i++) idletoken_buf_get_u32(&ob, &back[i]);
        ST(memcmp(back, toks, sizeof(toks)) == 0, "infer_begin v7: token ids round trip exactly");

        idletoken_nodecrypt_clear(&tx); idletoken_nodecrypt_clear(&rx);
    }

#undef ST
    /* Node-crypto framing (docs/inter-node-encryption.md N1). Lives with the
     * coordinator's selftest because it needs no cluster, no weights and no
     * network -- which is the point: the security-critical part of the design
     * (nonce discipline, replay rejection, direction separation) is provable on
     * any machine, including ones where a real paired cluster cannot run. The
     * wire-level "no token ids in the captured stream" assertion still needs a
     * real Linux cluster; this covers the logic that assertion depends on. */
    fails += idletoken_nodecrypt_selftest();



    fprintf(stderr, "selftest: %s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}

/* Handle one HTTP request. v0.1 routes:
 *
 *   GET  /health                  → 200 {"status":"ok","cluster_size":N}
 *   GET  /idletoken/v1/stats                → 200 serving counters (dashboard activity)
 *   POST /v1/messages             → 200 {"id":..., "type":"message",
 *                                          "role":"assistant",
 *                                          "content":[{"type":"text",
 *                                                       "text":"<token id>"}],
 *                                          "stop_reason":"end_turn"}  (Anthropic shape)
 *   POST /v1/chat/completions     → 200 {OpenAI ChatCompletion shape}
 *   anything else                 → 404
 *
 * v0.1 body content is whatever the user sent — we don't tokenize yet. We
 * drive ONE mock INFER step (workers return zero logits → argmax=0 → token 0)
 * and report that token id in the response. Once ds4 + GGUF wire in,
 * tokenize→decode_loop→detokenize→content is the swap. */
/* Per-request generation ceiling, from --max-decode (0 = context-bound only).
 * File-scope because the HTTP handler runs far from main(); set once at startup
 * and read-only afterwards. */
static int g_max_decode = 4096;

/* Naive JSON int extractor for `"max_tokens": N`. Returns N if found, else
 * `dflt`. Not robust against trailing decimal or strings; v0.1 enough. */
static int extract_int_field(const char *json, size_t json_len,
                             const char *key, int dflt) {
    if (!json || !key) return dflt;
    size_t klen = strlen(key);
    if (klen + 2 > json_len) return dflt;
    for (size_t i = 0; i + klen + 2 <= json_len; i++) {
        if (json[i] != '"') continue;
        if (i + 1 + klen + 1 > json_len) break;
        if (memcmp(json + i + 1, key, klen) != 0) continue;
        if (json[i + 1 + klen] != '"') continue;
        size_t p = i + 1 + klen + 1;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json_len || json[p] != ':') continue;
        p++;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        int sign = 1;
        if (p < json_len && json[p] == '-') { sign = -1; p++; }
        int v = 0, digits = 0;
        while (p < json_len && json[p] >= '0' && json[p] <= '9') {
            v = v * 10 + (json[p] - '0');
            p++; digits++;
        }
        if (digits == 0) return dflt;
        return sign * v;
    }
    return dflt;
}

/* Naive JSON bool extractor for `"stream": true`. Same tolerance level as
 * extract_int_field: finds the first `"key"` then `:` then literal true/false.
 * Returns 1/0, or `dflt` when the key is absent. */
static int extract_bool_field(const char *json, size_t json_len,
                              const char *key, int dflt) {
    if (!json || !key) return dflt;
    size_t klen = strlen(key);
    if (klen + 2 > json_len) return dflt;
    for (size_t i = 0; i + klen + 2 <= json_len; i++) {
        if (json[i] != '"') continue;
        if (i + 1 + klen + 1 > json_len) break;
        if (memcmp(json + i + 1, key, klen) != 0) continue;
        if (json[i + 1 + klen] != '"') continue;
        size_t p = i + 1 + klen + 1;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json_len || json[p] != ':') continue;
        p++;
        while (p < json_len && (json[p] == ' ' || json[p] == '\t' ||
                                 json[p] == '\n' || json[p] == '\r')) p++;
        if (p + 4 <= json_len && memcmp(json + p, "true", 4) == 0) return 1;
        if (p + 5 <= json_len && memcmp(json + p, "false", 5) == 0) return 0;
        return dflt;
    }
    return dflt;
}

/* Escape `src[0..len)` for embedding inside a JSON string literal. Handles
 * \" \\ \n \r \t and \u00XX for other control bytes. NUL-terminates dst.
 * Returns the escaped length. (Shared by the streaming + non-streaming +
 * mock response paths — one escaping rule everywhere.) */
static size_t json_escape_text(char *dst, size_t cap,
                               const char *src, size_t len) {
    size_t je = 0;
    for (size_t i = 0; i < len && je + 7 < cap; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"' || c == '\\') { dst[je++] = '\\'; dst[je++] = (char)c; }
        else if (c == '\n')         { dst[je++] = '\\'; dst[je++] = 'n'; }
        else if (c == '\r')         { dst[je++] = '\\'; dst[je++] = 'r'; }
        else if (c == '\t')         { dst[je++] = '\\'; dst[je++] = 't'; }
        else if (c < 0x20)          { je += (size_t)snprintf(dst + je, 7, "\\u%04x", c); }
        else                        { dst[je++] = (char)c; }
    }
    dst[je] = 0;
    return je;
}

/* Length of the longest prefix of buf[0..len) that does not end in the middle
 * of a UTF-8 multi-byte sequence. BPE tokens can split a single Unicode char
 * (e.g. one Chinese char across two tokens); the streaming path holds the
 * incomplete tail back until the next token completes it, so every SSE frame
 * carries valid UTF-8. Invalid sequences are passed through unchanged. */
static size_t utf8_complete_len(const char *buf, size_t len) {
    size_t i = len, back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = (unsigned char)buf[i - 1];
        if ((c & 0xC0) != 0x80) {           /* ASCII or a lead byte */
            size_t need = 1;
            if      ((c & 0x80) == 0x00) need = 1;
            else if ((c & 0xE0) == 0xC0) need = 2;
            else if ((c & 0xF0) == 0xE0) need = 3;
            else if ((c & 0xF8) == 0xF0) need = 4;
            size_t have = len - (i - 1);
            return (have >= need) ? len : i - 1;
        }
        i--; back++;                        /* continuation byte: keep looking */
    }
    return len;  /* all continuation bytes / invalid — emit as-is */
}

/* --- SSE emitters ---------------------------------------------------------
 *
 * One emitter per wire shape, mirroring the platform gateway's controllers
 * (platform/.../gateway/openai.controller.ts + anthropic.controller.ts) so a
 * client that speaks to the platform speaks to a direct coord unchanged:
 *
 *   OpenAI   : data-only `chat.completion.chunk` frames, final frame carries
 *              finish_reason + usage, then `data: [DONE]`.
 *   Anthropic: message_start → content_block_start → content_block_delta* →
 *              content_block_stop → message_delta(stop_reason+usage) →
 *              message_stop.  (Claude Code consumes this sequence.)
 *
 * `failed` latches on the first write error (client hangup); every later call
 * becomes a no-op and the decode loop bails out to stop wasting the cluster. */
typedef struct {
    int  fd;
    int  anthropic;      /* 1 = /v1/messages event sequence */
    int  failed;
    char id[48];         /* response id suffix: "%llu" req_id, or "mock" */
    long long created;
} idletoken_sse;

static void sse_emitf(idletoken_sse *s, const char *event, const char *fmt, ...) {
    if (s->failed) return;
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(buf)) { s->failed = 1; return; }
    if (idletoken_http_sse_event(s->fd, event, buf, (size_t)n) != 0) s->failed = 1;
}

/* HTTP head + protocol preamble. `n_input` feeds Anthropic's message_start
 * usage.input_tokens (OpenAI reports usage only on the final frame). */
static void sse_begin(idletoken_sse *s, int n_input) {
    if (idletoken_http_send_sse_head(s->fd) != 0) { s->failed = 1; return; }
    if (s->anthropic) {
        sse_emitf(s, "message_start",
            "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_idletoken_%s\","
             "\"type\":\"message\",\"role\":\"assistant\","
             "\"model\":\"%s\",\"content\":[],"
             "\"stop_reason\":null,\"stop_sequence\":null,"
             "\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}",
            s->id, coord_model()->id, n_input);
        sse_emitf(s, "content_block_start",
            "{\"type\":\"content_block_start\",\"index\":0,"
             "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
    } else {
        sse_emitf(s, NULL,
            "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                           "\"finish_reason\":null}]}",
            s->id, s->created, coord_model()->id);
    }
}

/* One text delta. `esc` must already be JSON-escaped (json_escape_text). */
static void sse_delta(idletoken_sse *s, const char *esc) {
    if (!esc[0]) return;
    if (s->anthropic) {
        sse_emitf(s, "content_block_delta",
            "{\"type\":\"content_block_delta\",\"index\":0,"
             "\"delta\":{\"type\":\"text_delta\",\"text\":\"%s\"}}", esc);
    } else {
        sse_emitf(s, NULL,
            "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
                           "\"finish_reason\":null}]}",
            s->id, s->created, coord_model()->id, esc);
    }
}

/* Trailer: usage + stop reason + the protocol's end-of-stream marker. */
/* The SSE event for a failure partway through generation.
 *
 * On the streaming path the deltas have already gone out and the HTTP status can
 * no longer be changed, so the only place left to tell the truth is the stream
 * itself. Each protocol has its own convention: OpenAI clients look for an
 * `error` object inside `data:`, while Anthropic has a dedicated `error` event
 * type. */
static void sse_error(idletoken_sse *s, const char *msg) {
    if (!s || s->failed) return;
    if (s->anthropic)
        sse_emitf(s, "error",
                  "{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"%s\"}}", msg);
    else
        sse_emitf(s, NULL,
                  "{\"error\":{\"type\":\"api_error\",\"message\":\"%s\"}}", msg);
}

/* Keepalive + progress during prefill.
 *
 * Prefill on a LAN cluster is minutes, not milliseconds, and it used to be
 * minutes of ABSOLUTE SILENCE on the socket: sse_begin ran only after the last
 * chunk. Any client with a read timeout — ours has one, 300s — kills the
 * request and shows the raw OS error ("os error 10060" on Windows, which is
 * just WSAETIMEDOUT). Turn one was short enough to land inside the window and
 * turn two was not, so the client looked like it broke after a couple of
 * rounds. Now bytes flow the whole way through.
 *
 * The progress line is an SSE **comment** (`: prefill 128/512`). The spec says
 * a line starting with ':' is ignored, so Claude Code and every OpenAI client
 * skip it; our own client parses it to show "processing the prompt" instead of
 * a blank bubble. Anthropic additionally gets a real `ping` event, which is
 * what the upstream API sends for exactly this purpose. */
static void sse_prefill_tick(idletoken_sse *s, int done, int total, int reused) {
    if (!s || s->failed) return;
    /* `reused` is how many prompt tokens came from the KV cache. It is reported
     * separately and not merely implied by the starting `done`, because the two
     * cases have to be TELLABLE APART on screen: "120/135" (a hit, 15 tokens of
     * real work) and "0/135" (a miss, the whole prompt recomputed) look
     * identical while they scroll past, and a user watching that reasonably
     * concludes the cache is doing nothing. */
    char line[96];
    int n = snprintf(line, sizeof(line), ": prefill %d/%d reuse=%d\n\n", done, total, reused);
    if (n < 0 || (size_t)n >= sizeof(line)) return;
    if (idletoken_sendall(s->fd, line, (size_t)n) < 0) { s->failed = 1; return; }
    if (s->anthropic) sse_emitf(s, "ping", "{\"type\":\"ping\"}");
}

static void sse_finish(idletoken_sse *s, int n_input, int n_output, int eos_stop) {
    if (s->anthropic) {
        sse_emitf(s, "content_block_stop",
            "{\"type\":\"content_block_stop\",\"index\":0}");
        sse_emitf(s, "message_delta",
            "{\"type\":\"message_delta\","
             "\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
             "\"usage\":{\"output_tokens\":%d}}",
            eos_stop ? "end_turn" : "max_tokens", n_output);
        sse_emitf(s, "message_stop", "{\"type\":\"message_stop\"}");
    } else {
        sse_emitf(s, NULL,
            "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                         "\"total_tokens\":%d}}",
            s->id, s->created, coord_model()->id, eos_stop ? "stop" : "length",
            n_input, n_output, n_input + n_output);
        sse_emitf(s, NULL, "[DONE]");
    }
}

/* Stream a whole text as word-sized SSE deltas (mock path: real streaming has
 * no meaning without a real engine, but the wire behavior — several frames,
 * correct trailer — must be exercisable without the 80GB GGUF). `esc` is the
 * ALREADY-ESCAPED text; splitting on spaces never lands inside an escape
 * sequence because our escaper never emits a space inside one. */
static void sse_stream_words(idletoken_sse *s, const char *esc) {
    char frame[512];
    size_t fl = 0;
    for (size_t i = 0; ; i++) {
        char c = esc[i];
        if (fl < sizeof(frame) - 1 && c != 0) frame[fl++] = c;
        /* Flush after each space (word boundary) and at end-of-text. */
        if ((c == ' ' || c == 0 || fl == sizeof(frame) - 1) && fl > 0) {
            frame[fl] = 0;
            sse_delta(s, frame);
            fl = 0;
        }
        if (c == 0) break;
    }
}

/* ===== llama.cpp single-machine relay (v2 rebuild WS-B1+B3) ================
 *
 * When g_llama is set, the chat/tokenize/count_tokens routes below relay to a
 * local llama-server (loopback only, supervised by src/coord/llama_sidecar.c)
 * instead of driving a worker cluster. The coordinator's HTTP surface stays
 * byte-compatible: requests are translated INTO the sidecar's OpenAI endpoint,
 * and responses are re-emitted through the same emitters/assembly as the
 * cluster path (same ids, same "model" field from coord_model(), same SSE
 * event sequences).
 *
 * A deliberate trick runs through all of it: JSON string values are moved
 * around as RAW STILL-ESCAPED spans. The bytes inside a JSON string literal
 * are valid in any other JSON string literal, so the relay never unescapes and
 * re-escapes — which is both cheaper and immune to the escape-table drift that
 * corrupting a prompt would take weeks to notice. */

/* Position of the value of `"key":` (first unescaped-quote match), or NULL.
 * Same tolerance level as extract_int_field above: a first-occurrence scan,
 * guarded against key-lookalikes inside string values (their quote is \"). */
static const char *json_value_pos(const char *json, size_t len, const char *key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);
    if (klen + 2 > len) return NULL;
    for (size_t i = 0; i + klen + 2 <= len; i++) {
        if (json[i] != '"') continue;
        if (i > 0 && json[i - 1] == '\\') continue;   /* inside a string value */
        if (memcmp(json + i + 1, key, klen) != 0) continue;
        if (json[i + 1 + klen] != '"') continue;
        size_t p = i + klen + 2;
        while (p < len && (json[p] == ' ' || json[p] == '\t' ||
                           json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= len || json[p] != ':') continue;
        p++;
        while (p < len && (json[p] == ' ' || json[p] == '\t' ||
                           json[p] == '\n' || json[p] == '\r')) p++;
        return p < len ? json + p : NULL;
    }
    return NULL;
}

/* Raw span (still escaped, quotes excluded) of a string-valued key.
 * Returns 0/-1; -1 also when the value is not a string (null, array...). */
static int json_raw_str_span(const char *json, size_t len, const char *key,
                             const char **out, size_t *out_len) {
    const char *v = json_value_pos(json, len, key);
    if (!v || *v != '"') return -1;
    const char *end = json + len;
    const char *p = ++v;
    int esc = 0;
    while (p < end) {
        if (esc) esc = 0;
        else if (*p == '\\') esc = 1;
        else if (*p == '"') break;
        p++;
    }
    if (p >= end) return -1;
    *out = v;
    *out_len = (size_t)(p - v);
    return 0;
}

/* Length of a bracketed value `[...]` including both brackets (v points at
 * '['), string- and escape-aware. -1 when unterminated. */
static ssize_t json_bracket_len(const char *v, const char *end) {
    int depth = 0, in_str = 0, esc = 0;
    for (const char *p = v; p < end; p++) {
        char ch = *p;
        if (esc)            esc = 0;
        else if (in_str)    { if (ch == '\\') esc = 1; else if (ch == '"') in_str = 0; }
        else if (ch == '"') in_str = 1;
        else if (ch == '[') depth++;
        else if (ch == ']') { if (--depth == 0) return (ssize_t)(p - v + 1); }
    }
    return -1;
}

/* How many token ids are in the sidecar's {"tokens":[...]} reply. -1 when the
 * array is missing (the count 0 — an empty array — is a valid answer). */
static int llama_tokens_count(const char *json, size_t len) {
    const char *v = json_value_pos(json, len, "tokens");
    if (!v || *v != '[') return -1;
    ssize_t bl = json_bracket_len(v, json + len);
    if (bl < 0) return -1;
    int count = 0, in_num = 0;
    for (const char *p = v + 1; p < v + bl - 1; p++) {
        if ((*p >= '0' && *p <= '9') || *p == '-') {
            if (!in_num) { count++; in_num = 1; }
        } else {
            in_num = 0;
        }
    }
    return count;
}

/* Naive double extractor (for the engine's timings.predicted_per_second). */
static double json_double_field(const char *json, size_t len,
                                const char *key, double dflt) {
    const char *v = json_value_pos(json, len, key);
    if (!v) return dflt;
    const char *end = json + len;
    char tmp[40];
    size_t n = 0;
    while (v < end && n + 1 < sizeof(tmp) &&
           ((*v >= '0' && *v <= '9') || *v == '-' || *v == '+' ||
            *v == '.' || *v == 'e' || *v == 'E'))
        tmp[n++] = *v++;
    if (n == 0) return dflt;
    tmp[n] = '\0';
    return atof(tmp);
}

/* Tiny growing string builder for the translated upstream bodies. */
typedef struct { char *p; size_t len, cap; int oom; } llama_sb;

static void sb_put(llama_sb *b, const char *s, size_t n) {
    if (b->oom || n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 512;
        while (nc < b->len + n + 1) nc *= 2;
        char *np = realloc(b->p, nc);
        if (!np) { b->oom = 1; return; }
        b->p = np;
        b->cap = nc;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sb_cstr(llama_sb *b, const char *s) { sb_put(b, s, strlen(s)); }

/* The Anthropic-to-OpenAI request translation lives in src/common/apiconv.c
 * (idletoken_anthropic_to_openai): pure and unit-tested (make apitest). It
 * carries system promotion, non-leading-system demotion, tools / tool_use /
 * tool_result, sampling passthrough and the max_tokens default. */

/* Upstream body for the OpenAI face: the client's JSON passed through (so
 * sampling parameters survive), with two keys injected at the FRONT of the
 * object — a later duplicate key wins in the engine's parser, so the client's
 * own values still take precedence: stream_options.include_usage so the final
 * SSE chunk carries usage, and a default max_tokens when the client sent none
 * (--max-decode, the same default the cluster path applies).
 * `force_nonstream` appends "stream":false at the END of the object (the
 * later duplicate wins), for the tools one-shot path where the client asked
 * to stream but the upstream request must not. */
static char *llama_openai_upstream_body(const char *body, size_t len,
                                        int want_stream, int force_nonstream,
                                        size_t *out_len) {
    if (!body || len == 0) return NULL;
    size_t i = 0;
    while (i < len && (body[i] == ' ' || body[i] == '\t' ||
                       body[i] == '\n' || body[i] == '\r')) i++;
    if (i >= len || body[i] != '{') return NULL;
    size_t j = i + 1;
    while (j < len && (body[j] == ' ' || body[j] == '\t' ||
                       body[j] == '\n' || body[j] == '\r')) j++;
    const int empty_obj = (j < len && body[j] == '}');
    llama_sb b = {0};
    sb_cstr(&b, "{");
    if (!empty_obj) {   /* injecting into `{}` would leave a trailing comma */
        if (want_stream)
            sb_cstr(&b, "\"stream_options\":{\"include_usage\":true},");
        if (g_max_decode > 0 && extract_int_field(body, len, "max_tokens", -1) < 0) {
            char mt[48];
            snprintf(mt, sizeof(mt), "\"max_tokens\":%d,", g_max_decode);
            sb_cstr(&b, mt);
        }
    }
    if (force_nonstream) {
        size_t e = len;
        while (e > i + 1 && (body[e - 1] == ' ' || body[e - 1] == '\t' ||
                             body[e - 1] == '\n' || body[e - 1] == '\r')) e--;
        if (e <= i + 1 || body[e - 1] != '}') { free(b.p); return NULL; }
        sb_put(&b, body + i + 1, e - i - 2);   /* inner of the object */
        sb_cstr(&b, empty_obj ? "\"stream\":false}" : ",\"stream\":false}");
    } else {
        sb_put(&b, body + i + 1, len - i - 1);
    }
    if (b.oom) { free(b.p); return NULL; }
    if (out_len) *out_len = b.len;
    return b.p;
}

static void llama_error_json(int conn_fd, int status, const char *type,
                             const char *msg) {
    /* JSON, not send_error's text/plain: clients on both protocols read errors
     * out of {"error":{"message":...}}. `msg` is assembled from literals and
     * our own state names — nothing in it needs escaping. */
    char body[640];
    int bl = snprintf(body, sizeof(body),
                      "{\"error\":{\"type\":\"%s\",\"message\":\"%s\"}}", type, msg);
    if (bl > 0 && (size_t)bl < sizeof(body))
        idletoken_http_send_json(conn_fd, status, body, (size_t)bl);
    else
        idletoken_http_send_error(conn_fd, status, msg);
}

/* Inference routes may only proceed while the engine is READY. Anything else
 * answers 503 NAMING the engine state — never a mock, never ds4 (v2 hard
 * invariant #4: no silent fallback; a broken engine stays loudly broken). */
static int llama_gate_ready(int conn_fd) {
    idletoken_llama_state st = idletoken_llama_get_state(g_llama);
    if (st == IDLETOKEN_LLAMA_READY) return 0;
    char msg[560];
    if (st == IDLETOKEN_LLAMA_FAILED) {
        char why[256];
        idletoken_llama_fail_reason(g_llama, why, sizeof(why));
        snprintf(msg, sizeof(msg), "inference engine state is 'failed': %s", why);
    } else {
        snprintf(msg, sizeof(msg),
                 "inference engine state is '%s' (llama-server is not serving yet); retry shortly",
                 idletoken_llama_state_name(st));
    }
    fprintf(stderr, "coord: llama-relay: refusing request — engine state %s\n",
            idletoken_llama_state_name(st));
    llama_error_json(conn_fd, 503, "api_error", msg);
    return -1;
}

/* Count the prompt tokens a chat request will ACTUALLY cost, through the
 * sidecar's own template + tokenizer: POST /apply-template renders the chat
 * template, then POST /tokenize with add_special=true — mirroring what the
 * engine's own completion path does with the rendered prompt. Both the chat
 * relay and count_tokens call THIS function, so the two numbers agree by
 * construction (gate G_API_MODELS claim 2).
 * Returns the count, or -1 with err filled; *bad_request set = the engine
 * judged the body malformed (caller answers 400, not 503). */
static int llama_prompt_token_count(const char *oai_body, size_t len,
                                    int *bad_request, char *err, size_t err_cap) {
    *bad_request = 0;
    const int port = idletoken_llama_port_of(g_llama);
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(port, "POST", "/apply-template", oai_body, len,
                                  30000, &c) != 0) {
        snprintf(err, err_cap, "engine unreachable (apply-template)");
        return -1;
    }
    size_t rlen = 0;
    char *resp = idletoken_llama_http_read_all(&c, &rlen, 16u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    if (!resp) {
        snprintf(err, err_cap, "engine connection failed (apply-template)");
        return -1;
    }
    const char *pspan;
    size_t plen;
    if (status != 200 ||
        json_raw_str_span(resp, rlen, "prompt", &pspan, &plen) != 0) {
        *bad_request = (status >= 400 && status < 500);
        snprintf(err, err_cap, "chat template failed (engine HTTP %d)", status);
        free(resp);
        return -1;
    }
    llama_sb tb = {0};
    sb_cstr(&tb, "{\"content\":\"");
    sb_put(&tb, pspan, plen);
    sb_cstr(&tb, "\",\"add_special\":true,\"parse_special\":true}");
    free(resp);
    if (tb.oom) { free(tb.p); snprintf(err, err_cap, "out of memory"); return -1; }
    if (idletoken_llama_http_open(port, "POST", "/tokenize", tb.p, tb.len,
                                  30000, &c) != 0) {
        free(tb.p);
        snprintf(err, err_cap, "engine unreachable (tokenize)");
        return -1;
    }
    free(tb.p);
    resp = idletoken_llama_http_read_all(&c, &rlen, 16u << 20);
    status = c.status;
    idletoken_llama_http_close(&c);
    if (!resp || status != 200) {
        free(resp);
        snprintf(err, err_cap, "tokenize failed (engine HTTP %d)", status);
        return -1;
    }
    int n = llama_tokens_count(resp, rlen);
    free(resp);
    if (n < 0) {
        snprintf(err, err_cap, "tokenize returned no tokens array");
        return -1;
    }
    return n;
}

/* POST /idletoken/v1/tokenize — RAW text count for platform metering: no chat
 * template, add_special=false (the platform meters user-visible text, not our
 * prompt framing — same contract as the cluster path's ds4x count). */
static void llama_tokenize_route(int conn_fd, const idletoken_http_req *req) {
    if (llama_gate_ready(conn_fd) != 0) return;
    const char *tspan = NULL;
    size_t tlen = 0;
    if (req->body && req->body_len)
        json_raw_str_span((const char *)req->body, req->body_len, "text",
                          &tspan, &tlen);
    if (!tspan || tlen == 0) {
        idletoken_http_send_error(conn_fd, 400, "missing or empty 'text' field");
        return;
    }
    llama_sb tb = {0};
    sb_cstr(&tb, "{\"content\":\"");
    sb_put(&tb, tspan, tlen);
    sb_cstr(&tb, "\",\"add_special\":false,\"parse_special\":true}");
    if (tb.oom) {
        free(tb.p);
        idletoken_http_send_error(conn_fd, 500, "oom");
        return;
    }
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(idletoken_llama_port_of(g_llama), "POST",
                                  "/tokenize", tb.p, tb.len, 30000, &c) != 0) {
        free(tb.p);
        llama_error_json(conn_fd, 503, "api_error", "engine unreachable (tokenize)");
        return;
    }
    free(tb.p);
    size_t rlen = 0;
    char *resp = idletoken_llama_http_read_all(&c, &rlen, 16u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    int n = (resp && status == 200) ? llama_tokens_count(resp, rlen) : -1;
    free(resp);
    if (n < 0) {
        llama_error_json(conn_fd, 503, "api_error", "engine tokenize failed");
        return;
    }
    char body[128];
    int bl = snprintf(body, sizeof(body), "{\"tokens\":%d,\"model\":\"%s\"}",
                      n, coord_model()->id);
    idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
}

/* POST /v1/messages/count_tokens — the templated count, via the same helper
 * the chat relay uses (agreement by construction, see above). */
static void llama_count_tokens_route(int conn_fd, const idletoken_http_req *req) {
    if (llama_gate_ready(conn_fd) != 0) return;
    size_t uplen = 0;
    char *up = idletoken_anthropic_to_openai((const char *)req->body,
                                             req->body_len, 0, g_max_decode,
                                             &uplen);
    if (!up) {
        idletoken_http_send_error(conn_fd, 400, "missing or empty 'messages'/'content'");
        return;
    }
    char err[200];
    int bad = 0;
    int n = llama_prompt_token_count(up, uplen, &bad, err, sizeof(err));
    free(up);
    if (n < 0) {
        llama_error_json(conn_fd, bad ? 400 : 503, "api_error", err);
        return;
    }
    char body[64];
    int bl = snprintf(body, sizeof(body), "{\"input_tokens\":%d}", n);
    idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
}

/* Serving counters, shared with the cluster path's bookkeeping (same fields,
 * same EWMA half-life). tok_per_s comes from the engine's own timings when it
 * reports them. */
static void llama_account(int n_input, int n_output, double tok_per_s,
                          long long t0_ms) {
    g_stats.requests++;
    if (n_input  > 0) g_stats.in_tokens  += (uint64_t)n_input;
    if (n_output > 0) g_stats.out_tokens += (uint64_t)n_output;
    g_stats.last_request_at = (long long)time(NULL);
    if (tok_per_s > 0) g_stats.last_tok_per_s = tok_per_s;
    double dt = (double)(now_ms() - t0_ms);
    g_stats.service_ms_ewma = g_stats.service_ms_ewma > 0
        ? g_stats.service_ms_ewma * 0.875 + dt * 0.125 : dt;
}

/* Non-stream chat: one upstream JSON in, one of OUR response bodies out —
 * the exact shapes coord_req_finish builds for the cluster path, model field
 * from coord_model() (G_API_MODELS claim 1). */
static void llama_chat_nonstream(int conn_fd, int is_anthropic,
                                 const char *up, size_t uplen,
                                 int n_input, uint64_t req_id, long long t0) {
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(idletoken_llama_port_of(g_llama), "POST",
                                  IDLETOKEN_PATH_OPENAI, up, uplen, 0, &c) != 0) {
        llama_error_json(conn_fd, 503, "api_error", "inference engine connection failed");
        return;
    }
    size_t rlen = 0;
    char *resp = idletoken_llama_http_read_all(&c, &rlen, 64u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    if (!resp) {
        llama_error_json(conn_fd, 503, "api_error",
                         "inference engine connection lost mid-response");
        return;
    }
    if (status != 200) {
        /* the engine's own error body is already an {"error":...} JSON */
        idletoken_http_send_json(conn_fd, status, resp, rlen);
        free(resp);
        return;
    }
    const char *content = "";
    size_t clen = 0;
    json_raw_str_span(resp, rlen, "content", &content, &clen);
    const char *fr = NULL;
    size_t frlen = 0;
    int have_fr = json_raw_str_span(resp, rlen, "finish_reason", &fr, &frlen) == 0;
    int eos_stop = have_fr && frlen == 4 && !memcmp(fr, "stop", 4);
    int fr_tools = have_fr && frlen == 10 && !memcmp(fr, "tool_calls", 10);
    /* OpenAI face: the rebuilt message must carry the engine's tool_calls
     * through verbatim, not drop them (they are valid JSON from the engine). */
    const char *tcalls = NULL;
    long tcalls_len = 0;
    {
        const char *msg;
        size_t mlen;
        if (idletoken_oai_resp_message(resp, rlen, &msg, &mlen) == 0) {
            const char *tv = idletoken_json_obj_get(msg, mlen, "tool_calls");
            if (tv && *tv == '[') {
                long vl = idletoken_json_value_len(tv, msg + mlen);
                if (vl > 0) { tcalls = tv; tcalls_len = vl; }
            }
        }
    }
    int up_in = extract_int_field(resp, rlen, "prompt_tokens", n_input);
    int n_out = extract_int_field(resp, rlen, "completion_tokens", 0);
    double tps = json_double_field(resp, rlen, "predicted_per_second", 0.0);

    /* Anthropic face: content blocks + stop_reason through apiconv, so the
     * engine's tool_calls come back as tool_use blocks instead of being
     * silently dropped. Falls back to the plain text-block shape only when
     * the translation itself failed (OOM / no message object). */
    char *ablocks = NULL;
    size_t ablen = 0;
    char sreason[16] = "max_tokens";
    if (is_anthropic)
        ablocks = idletoken_oai_resp_to_anthropic_content(resp, rlen, sreason,
                                                          sizeof(sreason), &ablen);

    size_t body_cap = clen + ablen + (size_t)tcalls_len + 1024;
    char *body = malloc(body_cap);
    if (!body) {
        llama_error_json(conn_fd, 500, "api_error", "out of memory building the response");
        free(ablocks);
        free(resp);
        return;
    }
    int bl;
    if (is_anthropic && ablocks) {
        bl = snprintf(body, body_cap,
                      "{\"id\":\"msg_idletoken_%llu\","
                       "\"type\":\"message\","
                       "\"role\":\"assistant\","
                       "\"model\":\"%s\","
                       "\"content\":%s,"
                       "\"stop_reason\":\"%s\","
                       "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d},"
                       "\"cache_hit\":false,\"cached_tokens\":0}",
                      (unsigned long long)req_id, coord_model()->id,
                      ablocks, sreason, up_in, n_out);
    } else if (is_anthropic) {
        bl = snprintf(body, body_cap,
                      "{\"id\":\"msg_idletoken_%llu\","
                       "\"type\":\"message\","
                       "\"role\":\"assistant\","
                       "\"model\":\"%s\","
                       "\"content\":[{\"type\":\"text\",\"text\":\"%.*s\"}],"
                       "\"stop_reason\":\"%s\","
                       "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d},"
                       "\"cache_hit\":false,\"cached_tokens\":0}",
                      (unsigned long long)req_id, coord_model()->id,
                      (int)clen, content,
                      eos_stop ? "end_turn" : "max_tokens",
                      up_in, n_out);
    } else {
        char tc_field[32] = "";
        if (tcalls) snprintf(tc_field, sizeof(tc_field), ",\"tool_calls\":");
        bl = snprintf(body, body_cap,
                      "{\"id\":\"chatcmpl_idletoken_%llu\","
                       "\"object\":\"chat.completion\","
                       "\"created\":%lld,"
                       "\"model\":\"%s\","
                       "\"choices\":[{\"index\":0,"
                                      "\"message\":{\"role\":\"assistant\","
                                                    "\"content\":\"%.*s\"%s%.*s},"
                                      "\"finish_reason\":\"%s\"}],"
                       "\"usage\":{\"prompt_tokens\":%d,"
                                   "\"completion_tokens\":%d,"
                                   "\"total_tokens\":%d},"
                       "\"cache_hit\":false,\"cached_tokens\":0}",
                      (unsigned long long)req_id, (long long)time(NULL),
                      coord_model()->id, (int)clen, content,
                      tc_field, (int)tcalls_len, tcalls ? tcalls : "",
                      fr_tools ? "tool_calls" : (eos_stop ? "stop" : "length"),
                      up_in, n_out, up_in + n_out);
    }
    if (bl < 0 || (size_t)bl >= body_cap)
        idletoken_http_send_error(conn_fd, 500, "response too large");
    else
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
    free(body);
    free(ablocks);
    free(resp);
    fprintf(stderr, "coord: chat: generated %d tok (llama.cpp relay), stop=%s\n",
            n_out, eos_stop ? "EOS" : "max_tokens");
    llama_account(up_in, n_out, tps, t0);
}

/* Streaming chat: consume the engine's SSE incrementally and re-emit through
 * OUR emitters, so the wire shapes stay identical to the cluster path's
 * (OpenAI chunk frames + [DONE]; Anthropic message_start...message_stop). */
static void llama_chat_stream(int conn_fd, int is_anthropic,
                              const char *up, size_t uplen,
                              int n_input, uint64_t req_id, long long t0) {
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(idletoken_llama_port_of(g_llama), "POST",
                                  IDLETOKEN_PATH_OPENAI, up, uplen, 0, &c) != 0) {
        llama_error_json(conn_fd, 503, "api_error", "inference engine connection failed");
        return;
    }
    if (c.status != 200) {
        /* our stream has not started: a real HTTP status is still possible */
        size_t rlen = 0;
        char *resp = idletoken_llama_http_read_all(&c, &rlen, 1u << 20);
        int status = c.status;
        idletoken_llama_http_close(&c);
        if (resp && rlen)
            idletoken_http_send_json(conn_fd, status, resp, rlen);
        else
            llama_error_json(conn_fd, 503, "api_error", "inference engine refused the request");
        free(resp);
        return;
    }

    idletoken_sse s = (idletoken_sse){ conn_fd, is_anthropic, 0, "", 0 };
    snprintf(s.id, sizeof(s.id), "%llu", (unsigned long long)req_id);
    s.created = (long long)time(NULL);
    sse_begin(&s, n_input);

    /* Upstream SSE: split on '\n', handle `data: {...}` lines. Each delta's
     * content span is already-escaped JSON text — exactly what sse_delta
     * takes, so it goes out without an unescape/re-escape round trip. */
    char chunk[4096];
    char *line = NULL;
    size_t llen = 0, lcap = 0;
    int done = 0, eos_stop = 0, broke = 0;
    int n_deltas = 0, up_in = -1, up_out = -1;
    double tps = 0.0;
    for (;;) {
        ssize_t r = idletoken_llama_http_read(&c, chunk, sizeof(chunk));
        if (r < 0) { broke = !done; break; }
        if (r == 0) break;
        for (ssize_t i = 0; i < r; i++) {
            char ch = chunk[i];
            if (ch != '\n') {
                if (llen + 2 > lcap) {
                    size_t nc = lcap ? lcap * 2 : 1024;
                    if (nc > (1u << 20)) { broke = 1; goto stream_end; }
                    char *nl = realloc(line, nc);
                    if (!nl) { broke = 1; goto stream_end; }
                    line = nl;
                    lcap = nc;
                }
                line[llen++] = ch;
                continue;
            }
            while (llen > 0 && line[llen - 1] == '\r') llen--;
            if (line) line[llen] = '\0';
            if (llen >= 5 && !memcmp(line, "data:", 5)) {
                const char *d = line + 5;
                while (*d == ' ') d++;
                size_t dlen = llen - (size_t)(d - line);
                if (dlen == 6 && !memcmp(d, "[DONE]", 6)) {
                    done = 1;
                } else if (dlen > 0) {
                    const char *span;
                    size_t slen;
                    if (json_raw_str_span(d, dlen, "content", &span, &slen) == 0 &&
                        slen > 0) {
                        char *esc = malloc(slen + 1);
                        if (esc) {
                            memcpy(esc, span, slen);
                            esc[slen] = '\0';
                            sse_delta(&s, esc);
                            free(esc);
                        }
                        n_deltas++;
                    }
                    const char *fr;
                    size_t frlen;
                    if (json_raw_str_span(d, dlen, "finish_reason", &fr, &frlen) == 0)
                        eos_stop = (frlen == 4 && !memcmp(fr, "stop", 4));
                    if (json_value_pos(d, dlen, "usage")) {
                        up_in  = extract_int_field(d, dlen, "prompt_tokens", up_in);
                        up_out = extract_int_field(d, dlen, "completion_tokens", up_out);
                        double v = json_double_field(d, dlen, "predicted_per_second", 0.0);
                        if (v > 0) tps = v;
                    }
                }
            }
            llen = 0;
        }
        if (s.failed) break;   /* client hung up: stop draining the engine */
    }
stream_end:
    free(line);
    /* Closing the upstream connection is also how a hung-up client cancels
     * generation: llama-server aborts the slot when its client disconnects. */
    idletoken_llama_http_close(&c);
    int n_out = up_out >= 0 ? up_out : n_deltas;
    int n_in  = up_in  >= 0 ? up_in  : n_input;
    if (broke) sse_error(&s, "engine connection lost mid-generation");
    sse_finish(&s, n_in, n_out, eos_stop && !broke);
    fprintf(stderr, "coord: chat: generated %d tok (llama.cpp relay), stop=%s (streamed)\n",
            n_out, broke ? "decode_failed" : (eos_stop ? "EOS" : "max_tokens"));
    llama_account(n_in, n_out, tps, t0);
}

/* One escape-safe chunk of an already-escaped span: never cuts inside a
 * \x or \uXXXX escape sequence (sse_emitf frames cap at 4 KiB, so long
 * texts must go out in pieces). */
static size_t esc_chunk_len(const char *esc, size_t len, size_t max) {
    size_t j = 0;
    while (j < len) {
        size_t unit = 1;
        if (esc[j] == '\\' && j + 1 < len)
            unit = (esc[j + 1] == 'u' && j + 5 < len) ? 6 : 2;
        if (j + unit > max) break;
        j += unit;
    }
    return j ? j : (len < max ? len : max);
}

/* Streaming chat WITH tools declared. Streaming llama-server's OpenAI
 * tool_calls deltas through an incremental re-emitter would need a full
 * delta-merge state machine on both faces; instead the UPSTREAM request runs
 * non-stream and the complete result goes out as one legal SSE sequence
 * (Anthropic: message_start .. tool_use blocks with input_json_delta ..
 * message_stop; OpenAI: chunk frames + [DONE]). Correctness over streaming
 * latency — tools are never silently dropped. */
static void llama_chat_stream_tools(int conn_fd, int is_anthropic,
                                    const char *up, size_t uplen,
                                    int n_input, uint64_t req_id, long long t0) {
    idletoken_llama_conn c;
    if (idletoken_llama_http_open(idletoken_llama_port_of(g_llama), "POST",
                                  IDLETOKEN_PATH_OPENAI, up, uplen, 0, &c) != 0) {
        llama_error_json(conn_fd, 503, "api_error", "inference engine connection failed");
        return;
    }
    size_t rlen = 0;
    char *resp = idletoken_llama_http_read_all(&c, &rlen, 64u << 20);
    int status = c.status;
    idletoken_llama_http_close(&c);
    if (!resp) {
        llama_error_json(conn_fd, 503, "api_error",
                         "inference engine connection lost mid-response");
        return;
    }
    if (status != 200) {
        /* our stream has not started: a real HTTP status is still possible */
        idletoken_http_send_json(conn_fd, status, resp, rlen);
        free(resp);
        return;
    }

    const char *msg = NULL;
    size_t mlen = 0;
    if (idletoken_oai_resp_message(resp, rlen, &msg, &mlen) != 0) {
        llama_error_json(conn_fd, 502, "api_error",
                         "inference engine returned no message object");
        free(resp);
        return;
    }
    const char *text = "";
    size_t textl = 0;
    idletoken_json_obj_str(msg, mlen, "content", &text, &textl);
    int up_in  = extract_int_field(resp, rlen, "prompt_tokens", n_input);
    int n_out  = extract_int_field(resp, rlen, "completion_tokens", 0);
    double tps = json_double_field(resp, rlen, "predicted_per_second", 0.0);
    char sreason[16] = "max_tokens";
    {
        char *probe = idletoken_oai_resp_to_anthropic_content(resp, rlen, sreason,
                                                              sizeof(sreason), NULL);
        free(probe);   /* only wanted the finish_reason mapping */
    }

    idletoken_sse s = (idletoken_sse){ conn_fd, is_anthropic, 0, "", 0 };
    snprintf(s.id, sizeof(s.id), "%llu", (unsigned long long)req_id);
    s.created = (long long)time(NULL);

    if (is_anthropic) {
        if (idletoken_http_send_sse_head(s.fd) != 0) { free(resp); return; }
        sse_emitf(&s, "message_start",
            "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_idletoken_%s\","
             "\"type\":\"message\",\"role\":\"assistant\","
             "\"model\":\"%s\",\"content\":[],"
             "\"stop_reason\":null,\"stop_sequence\":null,"
             "\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}",
            s.id, coord_model()->id, up_in);
        int idx = 0;
        size_t probe = 0;
        idletoken_tool_call tc;
        int have_calls = idletoken_oai_next_tool_call(msg, mlen, &probe, &tc);
        if (textl > 0 || !have_calls) {
            sse_emitf(&s, "content_block_start",
                "{\"type\":\"content_block_start\",\"index\":%d,"
                 "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}", idx);
            for (size_t i = 0; i < textl && !s.failed; ) {
                size_t n = esc_chunk_len(text + i, textl - i, 2048);
                sse_emitf(&s, "content_block_delta",
                    "{\"type\":\"content_block_delta\",\"index\":%d,"
                     "\"delta\":{\"type\":\"text_delta\",\"text\":\"%.*s\"}}",
                    idx, (int)n, text + i);
                i += n;
            }
            sse_emitf(&s, "content_block_stop",
                "{\"type\":\"content_block_stop\",\"index\":%d}", idx);
            idx++;
        }
        size_t it = 0;
        while (idletoken_oai_next_tool_call(msg, mlen, &it, &tc)) {
            sse_emitf(&s, "content_block_start",
                "{\"type\":\"content_block_start\",\"index\":%d,"
                 "\"content_block\":{\"type\":\"tool_use\",\"id\":\"%.*s\","
                 "\"name\":\"%.*s\",\"input\":{}}}",
                idx, (int)tc.id_len, tc.id, (int)tc.name_len, tc.name);
            for (size_t i = 0; i < tc.args_len && !s.failed; ) {
                size_t n = esc_chunk_len(tc.args + i, tc.args_len - i, 2048);
                sse_emitf(&s, "content_block_delta",
                    "{\"type\":\"content_block_delta\",\"index\":%d,"
                     "\"delta\":{\"type\":\"input_json_delta\","
                     "\"partial_json\":\"%.*s\"}}",
                    idx, (int)n, tc.args + i);
                i += n;
            }
            sse_emitf(&s, "content_block_stop",
                "{\"type\":\"content_block_stop\",\"index\":%d}", idx);
            idx++;
        }
        sse_emitf(&s, "message_delta",
            "{\"type\":\"message_delta\","
             "\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
             "\"usage\":{\"output_tokens\":%d}}", sreason, n_out);
        sse_emitf(&s, "message_stop", "{\"type\":\"message_stop\"}");
    } else {
        sse_begin(&s, up_in);      /* role preamble frame */
        for (size_t i = 0; i < textl && !s.failed; ) {
            size_t n = esc_chunk_len(text + i, textl - i, 2048);
            char frame[2560];
            snprintf(frame, sizeof(frame), "%.*s", (int)n, text + i);
            sse_delta(&s, frame);
            i += n;
        }
        size_t it = 0;
        idletoken_tool_call tc;
        int call_idx = 0, have_calls = 0;
        while (idletoken_oai_next_tool_call(msg, mlen, &it, &tc)) {
            have_calls = 1;
            /* one delta names the call, follow-ups append arguments (OpenAI
             * clients concatenate tool_calls[i].function.arguments deltas) */
            sse_emitf(&s, NULL,
                "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
                 "\"index\":%d,\"id\":\"%.*s\",\"type\":\"function\","
                 "\"function\":{\"name\":\"%.*s\",\"arguments\":\"\"}}]},"
                 "\"finish_reason\":null}]}",
                s.id, s.created, coord_model()->id, call_idx,
                (int)tc.id_len, tc.id, (int)tc.name_len, tc.name);
            for (size_t i = 0; i < tc.args_len && !s.failed; ) {
                size_t n = esc_chunk_len(tc.args + i, tc.args_len - i, 2048);
                sse_emitf(&s, NULL,
                    "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
                     "\"created\":%lld,\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
                     "\"index\":%d,\"function\":{\"arguments\":\"%.*s\"}}]},"
                     "\"finish_reason\":null}]}",
                    s.id, s.created, coord_model()->id, call_idx,
                    (int)n, tc.args + i);
                i += n;
            }
            call_idx++;
        }
        const char *fr = have_calls ? "tool_calls"
                       : (!strcmp(sreason, "end_turn") ? "stop" : "length");
        sse_emitf(&s, NULL,
            "{\"id\":\"chatcmpl_idletoken_%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                         "\"total_tokens\":%d}}",
            s.id, s.created, coord_model()->id, fr,
            up_in, n_out, up_in + n_out);
        sse_emitf(&s, NULL, "[DONE]");
    }
    free(resp);
    fprintf(stderr, "coord: chat: generated %d tok (llama.cpp relay), stop=%s "
                    "(streamed, tools one-shot)\n", n_out, sreason);
    llama_account(up_in, n_out, tps, t0);
}

/* The chat entry point for llamacpp mode (both faces, both stream modes). */
static void llama_chat_route(int conn_fd, const idletoken_http_req *req,
                             int is_anthropic, int want_stream) {
    if (llama_gate_ready(conn_fd) != 0) return;
    /* Tools + stream => the one-shot path above; the upstream body must then
     * be built WITHOUT "stream":true. */
    const int tools_oneshot = want_stream &&
        idletoken_body_has_tools((const char *)req->body, req->body_len);
    size_t uplen = 0;
    char *up = is_anthropic
        ? idletoken_anthropic_to_openai((const char *)req->body, req->body_len,
                                        want_stream && !tools_oneshot,
                                        g_max_decode, &uplen)
        : llama_openai_upstream_body((const char *)req->body, req->body_len,
                                     want_stream && !tools_oneshot,
                                     tools_oneshot, &uplen);
    if (!up) {
        idletoken_http_send_error(conn_fd, 400,
                                  is_anthropic ? "missing or empty 'messages'/'content'"
                                               : "missing or malformed JSON body");
        return;
    }
    /* Prompt cost through the engine's own template + tokenizer. This line is
     * scraped by gate G_API_MODELS (claim 2); count_tokens answers from the
     * same helper, so the two numbers cannot drift apart. The prompt's TEXT is
     * deliberately not logged (same invariant as the cluster path). */
    char terr[200];
    int bad = 0;
    int n_input = llama_prompt_token_count(up, uplen, &bad, terr, sizeof(terr));
    if (n_input < 0) {
        llama_error_json(conn_fd, bad ? 400 : 503, "api_error", terr);
        free(up);
        return;
    }
    fprintf(stderr, "coord: chat: tokenized %d-tok prompt (llama.cpp relay)\n",
            n_input);
    const long long t0 = now_ms();
    const uint64_t req_id = ((uint64_t)time(NULL) << 16)
                          ^ (uint64_t)(g_stats.requests + 1);
    if (want_stream && tools_oneshot)
        llama_chat_stream_tools(conn_fd, is_anthropic, up, uplen, n_input,
                                req_id, t0);
    else if (want_stream)
        llama_chat_stream(conn_fd, is_anthropic, up, uplen, n_input, req_id, t0);
    else
        llama_chat_nonstream(conn_fd, is_anthropic, up, uplen, n_input, req_id, t0);
    free(up);
}

/* ===== end llama.cpp relay ================================================ */

/* --api-token check for the inference endpoints. Returns 1 when the request
 * may proceed. /health and /idletoken/v1/cluster/status are exempt by the caller —
 * the client and pairing peers poll them before any token reaches them. */
static int api_token_ok(const idletoken_http_req *req, const char *api_token) {
    if (!api_token || !api_token[0]) return 1;   /* no token configured */
    char hv[512];
    if (idletoken_http_header_get(req, "authorization", hv, sizeof hv) == 0 &&
        idletoken_http_auth_value_matches(hv, api_token))
        return 1;
    if (idletoken_http_header_get(req, "x-api-key", hv, sizeof hv) == 0 &&
        strcmp(hv, api_token) == 0)
        return 1;
    return 0;
}

/* Drive a tokenize→prefill→decode→detokenize round through the cluster.
 *
 * `coord_engine` is a CPU-mode ds4_engine opened by main() once at startup;
 * it owns the tokenizer + chat template + vocab.
 *
 * Flow:
 *   1) Encode the user prompt into token IDs with `ds4_encode_chat_prompt`.
 *   2) For each prompt token, drive one cluster decode step (slow per-token
 *      "prefill" — chunked prefill is a TODO).
 *   3) Decode loop: feed previous sampled token, sample new one, accumulate;
 *      stop at EOS or `max_tokens`.
 *   4) Detokenize the accumulated tokens, splat into the response JSON. */

/* Everything an inference request needs **from the decode phase onwards**.
 *
 * Extracting it is the first step of E3 (PP micro-batching). Today it is still
 * called inline and the execution order is identical to before the refactor;
 * what changed is that "the state of one request" is now **an object that can be
 * put down and picked back up**, instead of forty locals scattered through the
 * 670-line straight line that was handle_http_request. The E3.3 executor has to
 * hold K such objects at once and interleave sends and receives between them to
 * fill the PP pipeline bubbles.
 *
 * Lifetime: the caller constructs it on the stack and fills in the fields;
 * coord_req_decode_and_reply() owns **all** cleanup (freeing prompt, http_body
 * and generated, and sending the HTTP response). */
typedef struct {
    /* --- Connection and response shape --- */
    int        conn_fd;
    int        is_anthropic;
    int        want_stream;
    uint64_t   req_id;
    idletoken_sse sse;
    /* The stream was already opened by the prefill stage (head + preamble sent,
     * keepalives running), so coord_req_begin must not open it a second time —
     * a second HTTP head mid-stream is garbage on the wire. */
    int        sse_started;

    /* --- Sequence slot (the ledger behind KV prefix reuse) --- */
    int            sel;            /* index into g_slots */
    coord_kv_slot *hs;             /* = &g_slots[sel] */
    uint8_t        seq_id;
    int            cache_hit;
    uint32_t       cached_tokens;
    uint32_t       hist_base_this;

    /* --- Inputs (this struct owns and frees them all) --- */
    ds4_tokens prompt;
    uint8_t   *http_body;          /* the type of idletoken_http_req.body */

    /* --- Decode progress --- */
    int  max_tokens;
    int *generated;
    int  n_generated;
    int  n_fed;                    /* generated tokens actually fed back into the KV */
    int  decode_failed;
    int  client_gone;              /* peer hung up mid-generation (see below) */
    int  next_token;
    int  eos;
    int      await_logits;         /* INFER_BEGIN sent, logits not back yet */
    uint64_t step_req_id;          /* this step's request_id, used to claim LOGITS */

    /* --- Accumulated output ---
     * Heap, grown on demand. This used to be `char text_out[4096]` with the
     * append silently skipped once full: the non-stream body was cut off
     * mid-sentence while finish_reason still said "stop" and completion_tokens
     * still reported every token — a truncation the caller cannot detect, which
     * is precisely what the decode_failed path below exists to avoid. It was
     * always reachable (4096 BYTES is roughly 1-1.5k tokens, well under the old
     * 4096-token ceiling); with --max-decode 0 it is the common case. */
    char  *text_out;
    size_t text_len;
    size_t text_cap;
    char   carry[8];               /* incomplete trailing UTF-8 bytes held across frames */
    size_t carry_len;

    struct timespec decode_t0;
    long long       exec_start_ms;  /* when execution of this request began (intake queueing excluded) */
    long long       ttft_ms;        /* start of execution -> first token, see g_stats.ttft_ms_ewma */
    long long       admit_ms;      /* when it entered the executor, for the service-time EWMA */
    long long       queued_ms;     /* how long it waited in the intake queue */
} coord_req;

/* The cluster-side execution environment: shared by every request and passed to
 * the decoder alongside the request. */
typedef struct {
    idletoken_worker_info *ws;
    int                 n;
    uint8_t            *lbuf;
    size_t              lbuf_cap;
    uint32_t           *running_pos;
    ds4_engine         *engine;
    ds4x_tokenizer     *xtok;
    uint32_t            ctx_size;
} coord_exec;

/* Decode a request whose prefill is done all the way through, and reply. This is
 * equivalent to steps 3 through 6 of handle_http_request before the refactor,
 * structurally identical -- it just operates on r-> instead of locals. */
/* E3.2: the decode phase is split into four pump functions -- begin,
 * step_send, step_recv, finish. Today coord_req_decode_and_reply() drives them
 * inline in the order begin -> (send, recv)* -> finish, behaving exactly as E3.1
 * did; the E3.3 executor will instead pump send for K requests in turn, then
 * pump recv for them in turn.
 *
 * State machine (r->await_logits is the only phase bit):
 *   step_send: emit the current token's text -> check stop conditions -> send
 *              the next step's INFER_BEGIN, set await=1
 *   step_recv: receive logits -> take the argmax -> advance pos/n_fed/next_token,
 *              set await=0
 * step_send returns 0 = sent and awaiting, 1 = this request is done decoding
 * (nothing pending), -1 = error. */
static int coord_req_begin(coord_req *r, const coord_exec *x) {
    if (!r->sse_started) {
        r->sse = (idletoken_sse){ r->conn_fd, r->is_anthropic, 0, "", 0 };
        snprintf(r->sse.id, sizeof(r->sse.id), "%llu", (unsigned long long)r->req_id);
        r->sse.created = (long long)time(NULL);
        if (r->want_stream) sse_begin(&r->sse, r->prompt.len);
    }

    r->eos = coord_tok_eos(x->engine, x->xtok);
    /* Clamp against remaining context once more: the prompt already occupies up
     * to pos, and decode must not push the KV past ctx. */
    {
        int headroom = (int)x->ctx_size - (int)*x->running_pos - 4;  /* keep 4 in reserve */
        if (headroom < 1) headroom = 1;
        if (r->max_tokens > headroom) {
            fprintf(stderr, "coord: chat: max_tokens capped to ctx headroom %d (pos=%u ctx=%u)\n",
                    headroom, *x->running_pos, x->ctx_size);
            r->max_tokens = headroom;
        }
    }
    r->generated = malloc(sizeof(int) * (size_t)r->max_tokens);
    if (!r->generated) {
        /* This path **cleans up after itself** (reply, free, release the slot);
         * on a non-zero return the caller must not touch r again. */
        r->hs->in_flight = 0;
        /* Once the stream is open the HTTP status is already 200 and can no
         * longer be changed; the only place left to tell the truth is the
         * stream itself. */
        if (r->want_stream && r->sse_started) {
            sse_error(&r->sse, "out of memory");
            sse_finish(&r->sse, r->prompt.len, 0, 0);
        } else {
            idletoken_http_send_error(r->conn_fd, 500, "oom");
        }
        ds4_tokens_free(&r->prompt);
        free(r->http_body);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &r->decode_t0);
    /* By this point prefill has run and the first token is already in
     * r->next_token (the argmax of prefill's last chunk), so this moment is where
     * TTFT ends. */
    if (r->exec_start_ms > 0) r->ttft_ms = now_ms() - r->exec_start_ms;
    return 0;
}

/* Append decoded text to r->text_out, growing it. Keeps room for the NUL that
 * coord_req_finish writes. Returns 0, or -1 on OOM (caller must fail the
 * request rather than continue with a short buffer). */
static int coord_text_append(coord_req *r, const char *s, size_t n) {
    if (r->text_len + n + 1 > r->text_cap) {
        size_t cap = r->text_cap ? r->text_cap : 4096;
        while (cap < r->text_len + n + 1) cap *= 2;
        char *p = realloc(r->text_out, cap);
        if (!p) return -1;
        r->text_out = p;
        r->text_cap = cap;
    }
    memcpy(r->text_out + r->text_len, s, n);
    r->text_len += n;
    return 0;
}

/* Returns 0 = sent and awaiting, 1 = decoding finished, -1 = error (errors take
 * the same cleanup path as 1). */
static int coord_req_step_send(coord_req *r, const coord_exec *x) {
    if (r->n_generated >= r->max_tokens) return 1;
    {
        r->generated[r->n_generated++] = r->next_token;
        if (r->next_token != r->eos) {
            char tbuf[512];
            size_t tlen = coord_tok_text(x->engine, x->xtok, r->next_token, tbuf, sizeof(tbuf));
            char *t = tlen ? tbuf : NULL;
            if (t) {
                if (coord_text_append(r, t, tlen) != 0) {
                    /* Out of memory. Loud, not silent: the reply would be short
                     * by exactly this token and nothing downstream could tell. */
                    r->decode_failed = 1;
                    return 1;
                }
                if (r->want_stream && !r->sse.failed) {
                    char work[512];
                    if (r->carry_len + tlen <= sizeof(work)) {
                        memcpy(work, r->carry, r->carry_len);
                        memcpy(work + r->carry_len, t, tlen);
                        size_t wl = r->carry_len + tlen;
                        size_t comp = utf8_complete_len(work, wl);
                        /* 6x the 512-byte token cap: every byte can escape to
                         * \u00XX, and the escaper truncates rather than fails. */
                        char esc[3200];
                        json_escape_text(esc, sizeof(esc), work, comp);
                        sse_delta(&r->sse, esc);
                        r->carry_len = wl - comp;   /* ≤ 3 bytes by construction */
                        if (r->carry_len > sizeof(r->carry)) r->carry_len = 0;
                        memcpy(r->carry, work + comp, r->carry_len);
                    } else {
                        /* Pathologically long token text: flush carry, then
                         * the token as-is (escaper truncates at its cap). */
                        /* 6x the 512-byte token cap: every byte can escape to
                         * \u00XX, and the escaper truncates rather than fails. */
                        char esc[3200];
                        if (r->carry_len > 0) {
                            json_escape_text(esc, sizeof(esc), r->carry, r->carry_len);
                            sse_delta(&r->sse, esc);
                            r->carry_len = 0;
                        }
                        json_escape_text(esc, sizeof(esc), t, tlen);
                        sse_delta(&r->sse, esc);
                    }
                }
                /* no free: coord_tok_text copied into tbuf and already freed
                 * the backend's heap buffer. */
            }
        }
    }
    if (r->next_token == r->eos) return 1;
    if (r->n_generated >= r->max_tokens) return 1;
    if (r->want_stream && r->sse.failed) return 1;   /* client hung up: stop decoding */
    /* Same question for the non-stream path, which writes nothing until the end
     * and so cannot learn it from a failed send. Probed every 32 tokens rather
     * than every one: a select() per token is pure overhead next to a PP round
     * trip, and 32 tokens is a couple of seconds of waste at worst. */
    if (!r->want_stream && (r->n_generated & 31) == 0 && idletoken_peer_closed(r->conn_fd)) {
        r->client_gone = 1;
        return 1;
    }
    /* This step's req_id must be **unique**: the executor claims results by the
     * request_id echoed in the LOGITS header, and a collision would hand A's
     * token to B. */
    r->step_req_id = r->req_id ^ (uint64_t)(r->prompt.len + r->n_generated);
    if (coord_round_send(x->ws, x->n, r->step_req_id, r->hs->pos,
                         &(uint32_t){ (uint32_t)r->next_token }, 1,
                         0, r->seq_id) != 0) {
        r->decode_failed = 1;
        return 1;
    }
    r->await_logits = 1;
    return 0;
}

/* Apply already-received logits to a request (receiving them is not part of
 * this). The concurrent executor claims by request_id first and then calls this;
 * the serial driver goes through step_recv below. */
static void coord_req_apply_logits(coord_req *r, uint32_t out_tok) {
    r->await_logits = 0;
    r->hs->pos++;
    r->n_fed++;
    r->next_token = (int)out_tok;
}

/* Receive this request's logits for this step and advance. Returns 0 to
 * continue, -1 on error (the caller moves to cleanup). */
static int coord_req_step_recv(coord_req *r, const coord_exec *x) {
    uint32_t out_tok = 0;
    uint64_t got_req = 0;
    if (coord_round_recv(x->ws, x->n, x->lbuf, x->lbuf_cap, &out_tok, &got_req) != 0) {
        r->decode_failed = 1;
        r->await_logits = 0;
        return -1;
    }
    /* Under the serial driver this always holds -- there is only one request in
     * hand. */
    if (got_req != r->step_req_id) {
        fprintf(stderr, "coord: LOGITS req_id mismatch (expected %llu, got %llu)\n",
                (unsigned long long)r->step_req_id, (unsigned long long)got_req);
        r->decode_failed = 1;
        r->await_logits = 0;
        return -1;
    }
    coord_req_apply_logits(r, out_tok);
    return 0;
}

static void coord_req_finish(coord_req *r, const coord_exec *x) {
    /* Update the KV history (the ledger behind prefix reuse): it is only
     * trustworthy if everything succeeded. On a hit we keep the old prefix and
     * append this round's new tokens; on a miss we start a fresh ledger from this
     * prompt (base = where prefill started). */
    if (r->decode_failed) {
        kv_slot_reset(r->sel);
    } else {
        if (!r->cache_hit) {
            kv_slot_reset(r->sel);
            r->hs->base = r->hist_base_this;
            r->hs->valid = 1;
        }
        int hist_ok = 1;
        for (int hi2 = (int)r->cached_tokens; hi2 < r->prompt.len && hist_ok; hi2++)
            hist_ok = kv_hist_push_slot(r->sel, (uint32_t)r->prompt.v[hi2]) == 0;
        for (int gi = 0; gi < r->n_fed && hist_ok; gi++)
            hist_ok = kv_hist_push_slot(r->sel, (uint32_t)r->generated[gi]) == 0;
        if (!hist_ok) kv_slot_reset(r->sel);  /* OOM: rather not reuse at all */
    }
    /* Release the slot -- necessarily **after** the history is written back, or
     * another request could select this slot and start extending it before we
     * have finished writing the ledger. */
    r->hs->in_flight = 0;
    /* Legacy out-parameter: keeps the "cursor of the most recently used slot"
     * semantics, for the logs and warmup to hook into. */
    *x->running_pos = r->hs->pos;
    if (r->text_out) r->text_out[r->text_len] = 0;   /* NULL when nothing decoded */
    if (r->want_stream && r->carry_len > 0 && !r->sse.failed) {
        /* Generation ended mid-UTF-8-char: emit the tail bytes anyway for
         * parity with the non-stream body, which also carries them. */
        char esc[64];
        json_escape_text(esc, sizeof(esc), r->carry, r->carry_len);
        sse_delta(&r->sse, esc);
    }
    int stop_reason_eos = (r->n_generated > 0 && r->generated[r->n_generated - 1] == r->eos);

    /* Machine-readable id trace: the E4 oracle (scripts/run_single_infer.sh)
     * compares this greedy sequence against single-node ds4 --dump-logprobs. */
    fprintf(stderr, "coord: chat: generated_ids:");
    for (int i = 0; i < r->n_generated; i++) fprintf(stderr, " %d", r->generated[i]);
    fprintf(stderr, "\n");
    /* The stop reason has to be able to say "it broke". There used to be only
     * EOS and max_tokens, so **every failure path was reported as max_tokens**:
     * in a concurrency test one request generated 2 tokens and still returned
     * HTTP 200 with finish_reason=length, leaving the client unable to tell "the
     * model finished" from "it broke partway". For an OpenAI/Anthropic-compatible
     * API that is fatal -- downstreams like Claude Code would treat a truncated
     * answer as the final one. */
    const char *stop_why = r->decode_failed ? "decode_failed"
                         : r->client_gone   ? "client_gone"
                         : (stop_reason_eos ? "EOS" : "max_tokens");
    fprintf(stderr, "coord: chat: generated %d tok (%zu B text), stop=%s%s\n",
            r->n_generated, r->text_len, stop_why,
            r->want_stream ? " (streamed)" : "");

    int n_input = r->prompt.len;
    int n_output = stop_reason_eos ? r->n_generated - 1 : r->n_generated;

    /* Serving counters (GET /idletoken/v1/stats). Decode throughput excludes prefill:
     * the loop ran n_generated-1 decode steps after the prefill's sample. */
    {
        struct timespec decode_t1;
        clock_gettime(CLOCK_MONOTONIC, &decode_t1);
        double decode_s = (double)(decode_t1.tv_sec - r->decode_t0.tv_sec) +
                          (double)(decode_t1.tv_nsec - r->decode_t0.tv_nsec) / 1e9;
        g_stats.requests++;
        g_stats.in_tokens  += (uint64_t)(n_input > 0 ? n_input : 0);
        g_stats.out_tokens += (uint64_t)(n_output > 0 ? n_output : 0);
        if (r->cache_hit) { g_stats.cache_hits++; g_stats.cached_tokens += r->cached_tokens; }
        g_stats.last_request_at = (long long)time(NULL);
        if (decode_s > 0.001 && r->n_generated > 1)
            g_stats.last_tok_per_s = (double)(r->n_generated - 1) / decode_s;
        if (r->ttft_ms > 0) {
            const double tt = (double)r->ttft_ms;
            g_stats.ttft_ms_ewma = g_stats.ttft_ms_ewma > 0
                ? g_stats.ttft_ms_ewma * 0.875 + tt * 0.125 : tt;   /* half-life of 8 requests, same as service time */
        }
    }

    /* 4. Streaming: trailer frames (usage + stop + [DONE]/message_stop) and
     *    done — the deltas already went out token by token. */
    if (r->want_stream) {
        /* Streaming has already sent the deltas and the status code can no
         * longer change; at least make the trailer tell the truth, and leave a
         * trace in the server log (stop_why above). */
        if (r->decode_failed) sse_error(&r->sse, "decode failed mid-generation");
        sse_finish(&r->sse, n_input, n_output, stop_reason_eos);
        free(r->generated);
        free(r->text_out);
        ds4_tokens_free(&r->prompt);
        free(r->http_body);
        return;
    }

    /* A failed generation must not masquerade as a normal completion. Better to
     * give the caller an explicit error than to ship truncated content with
     * finish_reason=length -- the caller has **no way whatsoever** to detect the
     * latter. (The streaming path has already returned above: its deltas are out,
     * see the handling in sse_finish.) */
    if (r->decode_failed) {
        idletoken_http_send_error(r->conn_fd, 500, "decode failed mid-generation");
        free(r->generated);
        free(r->text_out);
        ds4_tokens_free(&r->prompt);
        free(r->http_body);
        return;
    }

    /* 5. Non-stream: escape the accumulated text for JSON embedding.
     * Heap-sized from the actual text, for the same reason text_out is: the
     * fixed 8 KiB buffer truncated silently (json_escape_text just stops), so a
     * long reply -- or a short one full of quotes and newlines, which escape to
     * 2 bytes each -- came back cut off with finish_reason=stop. Worst case is
     * 6 bytes out per byte in (\u00XX for control chars), plus the NUL. */
    size_t json_cap = r->text_len * 6 + 8;
    char *json_text = malloc(json_cap);
    if (!json_text) {
        idletoken_http_send_error(r->conn_fd, 500, "out of memory building the response");
        free(r->generated);
        free(r->text_out);
        ds4_tokens_free(&r->prompt);
        free(r->http_body);
        return;
    }
    json_escape_text(json_text, json_cap, r->text_out, r->text_len);

    /* 6. Build response JSON. The envelope (ids, model name, usage) is well
     * under 1 KiB; 4 KiB of headroom over the escaped text is generous. */
    size_t body_cap = json_cap + 4096;
    char *body = malloc(body_cap);
    if (!body) {
        idletoken_http_send_error(r->conn_fd, 500, "out of memory building the response");
        free(json_text);
        free(r->generated);
        free(r->text_out);
        ds4_tokens_free(&r->prompt);
        free(r->http_body);
        return;
    }
    int bl;
    if (r->is_anthropic) {
        bl = snprintf(body, body_cap,
                      "{\"id\":\"msg_idletoken_%llu\","
                       "\"type\":\"message\","
                       "\"role\":\"assistant\","
                       "\"model\":\"%s\","
                       "\"content\":[{\"type\":\"text\",\"text\":\"%s\"}],"
                       "\"stop_reason\":\"%s\","
                       "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d},"
                       "\"cache_hit\":%s,\"cached_tokens\":%u}",
                      (unsigned long long)r->req_id, coord_model()->id, json_text,
                      stop_reason_eos ? "end_turn" : "max_tokens",
                      n_input, n_output,
                      r->cache_hit ? "true" : "false", r->cached_tokens);
    } else {
        bl = snprintf(body, body_cap,
                      "{\"id\":\"chatcmpl_idletoken_%llu\","
                       "\"object\":\"chat.completion\","
                       "\"created\":%lld,"
                       "\"model\":\"%s\","
                       "\"choices\":[{\"index\":0,"
                                      "\"message\":{\"role\":\"assistant\","
                                                    "\"content\":\"%s\"},"
                                      "\"finish_reason\":\"%s\"}],"
                       "\"usage\":{\"prompt_tokens\":%d,"
                                   "\"completion_tokens\":%d,"
                                   "\"total_tokens\":%d},"
                       "\"cache_hit\":%s,\"cached_tokens\":%u}",
                      (unsigned long long)r->req_id, (long long)time(NULL),
                      coord_model()->id, json_text,
                      stop_reason_eos ? "stop" : "length",
                      n_input, n_output, n_input + n_output,
                      r->cache_hit ? "true" : "false", r->cached_tokens);
    }
    if (bl < 0 || (size_t)bl >= body_cap) {
        idletoken_http_send_error(r->conn_fd, 500, "response too large");
    } else {
        idletoken_http_send_json(r->conn_fd, 200, body, (size_t)bl);
    }

    free(body);
    free(json_text);
    free(r->generated);
    free(r->text_out);
    ds4_tokens_free(&r->prompt);
    free(r->http_body);
}

/* The serial driver: begin -> (send, recv)* -> finish. The E3.3 executor uses
 * the same pump functions, but pumps send and recv across K requests in
 * batches. */
static void coord_req_decode_and_reply(coord_req *r, const coord_exec *x) {
    if (coord_req_begin(r, x) != 0) return;   /* begin already cleaned up (the OOM path) */
    /* IDLETOKEN_DS4X_PROF=1: the coordinator-side per-token breakdown. A worker
     * only sees its own segment, and the compute of both stages together accounts
     * for merely half the measured s/token -- where the other half goes has to be
     * answered here, or optimization just spins on the compute side. */
    const int prof = getenv("IDLETOKEN_DS4X_PROF") != NULL;
    double send_s = 0.0, recv_s = 0.0;
    uint64_t steps = 0;
    for (;;) {
        double t0 = prof ? coord_prof_now_s() : 0.0;
        int rc = coord_req_step_send(r, x);
        if (prof) send_s += coord_prof_now_s() - t0;
        if (rc != 0) break;                   /* finished or failed */
        t0 = prof ? coord_prof_now_s() : 0.0;
        int rrc = coord_req_step_recv(r, x);
        if (prof) { recv_s += coord_prof_now_s() - t0; steps++; }
        if (rrc != 0) break;
    }
    if (prof && steps) {
        fprintf(stderr,
                "coord: PROF2 decode steps=%llu  send_avg=%.1fms (detokenize + SSE + send INFER_BEGIN)  "
                "recv_avg=%.1fms (block on LOGITS + argmax)  total %.1fms/token\n",
                (unsigned long long)steps,
                send_s * 1000.0 / (double)steps,
                recv_s * 1000.0 / (double)steps,
                (send_s + recv_s) * 1000.0 / (double)steps);
        fflush(stderr);
    }
    coord_req_finish(r, x);
}

/* A non-NULL `out_parked` enables **parking mode** (E3.3): a request that
 * reaches the decode phase does not run to completion here. Instead it is
 * heap-allocated as a coord_req, handed back to the executor after begin, and the
 * executor pumps send/recv for it interleaved with other requests. NULL keeps the
 * original behaviour -- run to the end, then return.
 * On a successful park, ownership of conn_fd moves to the coord_req (which closes
 * it during cleanup) and the caller must not close it. */
static void handle_http_request(int conn_fd,
                                idletoken_worker_info *ws, int n,
                                uint8_t *lbuf, size_t lbuf_cap,
                                uint32_t *running_pos,
                                ds4_engine *coord_engine,
                                ds4x_tokenizer *coord_xtok,
                                uint32_t ctx_size,
                                const char *api_token,
                                coord_req **out_parked) {
    /* Where TTFT starts: from here to "first token ready" (prefill done) is this
     * machine's time to first token. Intake queueing is deliberately **excluded**
     * -- the platform accounts for it separately as queue_depth x
     * avg_service_ms, so including it would double count. */
    const long long exec_start_ms = now_ms();
    /* Either a ds4 engine (DSv4) or a ds4x tokenizer (small models) provides
     * tokenize/detokenize; `tok_ready` is the "chat is servable" gate. */
    const int tok_ready = (coord_engine != NULL) || (coord_xtok != NULL);
    idletoken_http_req req;
    if (idletoken_http_read_request(conn_fd, &req) != 0) {
        fprintf(stderr, "coord: http: read_request: %s\n", strerror(errno));
        idletoken_http_send_error(conn_fd, 400, "bad request");
        return;
    }
    const int from_platform = request_from_platform(&req);
    fprintf(stderr, "coord: http: %s %s  body=%zuB  origin=%s\n",
            req.method, req.path, req.body_len, from_platform ? "platform" : "local");

    /* GET /health — quick liveness probe. In llamacpp mode two extra fields
     * name the engine and its state; the original fields are untouched so
     * existing probes keep parsing. */
    if (!strcmp(req.method, "GET") && !strcmp(req.path, "/health")) {
        char engine_extra[160] = "";
        if (g_llama)
            snprintf(engine_extra, sizeof(engine_extra),
                     ",\"engine\":\"llamacpp\",\"engine_state\":\"%s\"",
                     idletoken_llama_state_name(idletoken_llama_get_state(g_llama)));
        /* cluster_size = number of workers (legacy semantics). In llamacpp
         * mode the legacy roster `n` is always 0 — count the RPC peers, or
         * /health claims a 2-machine cluster is running solo. */
        char body[288];
        int bl = snprintf(body, sizeof(body),
                          "{\"status\":\"ok\",\"cluster_size\":%d,\"pos\":%u%s}",
                          g_llama ? g_n_rpc_peers : n, *running_pos,
                          engine_extra);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(req.body);
        return;
    }

    /* GET /idletoken/v1/stats — serving counters for the client dashboard.
     * Exempt from the API token gate like /health: numbers only, no content. */
    if (!strcmp(req.method, "GET") && !strcmp(req.path, IDLETOKEN_PATH_STATS)) {
        long long now = (long long)time(NULL);
        /* v4: report the sequence slots too. The platform's scheduler derives
         * its concurrency ceiling from seq_slots (scheduler-design §4.5b,
         * contract 1); if we do not report it, it assumes 1, i.e. v3 behaviour.
         * seq_slots_live = slots currently holding a reusable prefix, an
         * observable measure of cache locality. */
        int slots_live = 0;
        uint32_t slot_tokens = 0;
        for (int si = 0; si < g_n_slots; si++)
            if (g_slots[si].valid && g_slots[si].len > 0) { slots_live++; slot_tokens += g_slots[si].len; }
        /* What this cluster is ACTUALLY serving. The client used to show the
         * model from its own local settings, which is a claim it cannot back:
         * change the setting without restarting, or join a cluster someone
         * else coordinates, and the panel confidently names the wrong model.
         * The only authority on "what is loaded" is the process that loaded it. */
        /* llamacpp mode: append the sidecar's state + restart count so the
         * dashboard can show "engine restarting/failed" instead of a silent
         * flatline (WS-B1: the API surface must not fake green). */
        char engine_extra[224] = "";
        if (g_llama)
            snprintf(engine_extra, sizeof(engine_extra),
                     ",\"engine\":\"llamacpp\",\"engine_state\":\"%s\","
                     "\"engine_restarts\":%d",
                     idletoken_llama_state_name(idletoken_llama_get_state(g_llama)),
                     idletoken_llama_restart_count(g_llama));
        char body[896];   /* grew for the concurrency / avg_ttft_ms / model / engine fields */
        int bl = snprintf(body, sizeof(body),
            "{\"model\":\"%s\",\"model_label\":\"%s\",\"quant\":\"%s\","
             "\"requests\":%llu,\"input_tokens\":%llu,\"output_tokens\":%llu,"
             "\"cache_hits\":%llu,\"cached_tokens\":%llu,"
             "\"seq_slots\":%d,\"seq_slots_auto\":%d,\"seq_slots_live\":%d,\"slot_prefix_tokens\":%u,"
             "\"concurrency\":%d,"
             "\"queue_depth\":%d,\"queue_cap\":%d,\"avg_service_ms\":%.0f,"
             "\"avg_ttft_ms\":%.0f,"
             "\"ctx_size\":%u,"
             "\"uptime_s\":%lld,\"last_request_unix\":%lld,"
             "\"last_tok_per_s\":%.2f%s}",
            coord_model()->id, coord_model()->label, coord_quant(),
            (unsigned long long)g_stats.requests,
            (unsigned long long)g_stats.in_tokens,
            (unsigned long long)g_stats.out_tokens,
            (unsigned long long)g_stats.cache_hits,
            (unsigned long long)g_stats.cached_tokens,
            g_n_slots, g_n_slots_auto > 0 ? g_n_slots_auto : g_n_slots, slots_live, slot_tokens,
            /* Report the **effective concurrency** (>=1). It is not the same as
             * seq_slots: slots are "how many independent KV caches fit",
             * concurrency is "how many are really in flight at once". Serial
             * execution reports 1, or the platform would treat a serial machine
             * as N-way parallel and underestimate the wait N-fold. */
            g_concurrent_live > 0 ? g_concurrent_live : 1,
            intake_depth(), g_intake.cap, g_stats.service_ms_ewma,
            g_stats.ttft_ms_ewma, ctx_size,
            g_stats.started_at ? now - g_stats.started_at : 0,
            g_stats.last_request_at,
            g_stats.last_tok_per_s,
            engine_extra);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(req.body);
        return;
    }

    /* GET /idletoken/v1/cluster/status — membership + layer plan for the client's
     * pairing/orchestration UI (acceptance P3/P4). Served only once the
     * cluster is formed (this HTTP server starts after CLUSTER_READY), so
     * phase is always "ready" here; the client layers its own pre-ready
     * phases on top. */
    if (!strcmp(req.method, "GET") && !strcmp(req.path, IDLETOKEN_PATH_CLUSTER)) {
        if (g_llama) {
            const char *estate = idletoken_llama_state_name(
                idletoken_llama_get_state(g_llama));
            const char *phase = !strcmp(estate, "ready") ? "ready" : "starting";
            char local_host[64] = "coordinator";
            if (gethostname(local_host, sizeof(local_host) - 1) != 0)
                snprintf(local_host, sizeof(local_host), "coordinator");
            local_host[sizeof(local_host) - 1] = '\0';
            char body[4096];
            size_t off = 0;
            off += (size_t)snprintf(body + off, sizeof(body) - off,
                "{\"phase\":\"%s\",\"engine\":\"llamacpp\","
                "\"engine_state\":\"%s\",\"cluster_size\":%d,\"members\":["
                "{\"hostname\":\"%s\",\"role\":\"coordinator\","
                "\"state\":\"%s\"}",
                phase, estate, g_n_rpc_peers + 1, local_host, phase);
            for (int i = 0; i < g_n_rpc_peers && off < sizeof(body); i++) {
                off += (size_t)snprintf(body + off, sizeof(body) - off,
                    ",{\"hostname\":\"%s\",\"role\":\"worker\","
                    "\"rpc_endpoint\":\"%s\",\"state\":\"%s\"}",
                    g_rpc_peers[i].hostname, g_rpc_peers[i].endpoint, phase);
            }
            if (off < sizeof(body))
                off += (size_t)snprintf(body + off, sizeof(body) - off, "]}");
            idletoken_http_send_json(conn_fd, 200, body, off);
            free(req.body);
            return;
        }
        char body[4096];
        size_t off = 0;
        off += (size_t)snprintf(body + off, sizeof(body) - off,
                                "{\"phase\":\"ready\",\"cluster_size\":%d,\"members\":[", n);
        for (int i = 0; i < n && off < sizeof(body); i++) {
            off += (size_t)snprintf(body + off, sizeof(body) - off,
                "%s{\"hostname\":\"%s\",\"gpu\":\"%s\",\"stage\":%u,"
                "\"layer_lo\":%u,\"layer_hi\":%u,\"state\":\"ready\"}",
                i ? "," : "", ws[i].hostname, ws[i].gpu_name,
                (unsigned)ws[i].stage_id, (unsigned)ws[i].layer_lo,
                (unsigned)ws[i].layer_hi);
        }
        if (off < sizeof(body))
            off += (size_t)snprintf(body + off, sizeof(body) - off, "]}");
        idletoken_http_send_json(conn_fd, 200, body, off);
        free(req.body);
        return;
    }

    /* GET /idletoken/v1/capability — "what can THIS CLUSTER run?" for the client's model
     * picker and for a user who just wants to know what their machines are
     * good for. Same advisor as `idletoken-worker --advise`, but over the whole
     * roster, so the answer changes (correctly) as machines join or leave. */
    if (!strcmp(req.method, "GET") && !strcmp(req.path, IDLETOKEN_PATH_CAPABILITY)) {
        idletoken_node_mem nodes[IDLETOKEN_MAX_WORKERS];
        for (int i = 0; i < n && i < IDLETOKEN_MAX_WORKERS; i++) {
            nodes[i].vram_usable = ws[i].vram_usable;
            nodes[i].ram_usable  = ws[i].ram_usable;
            nodes[i].ram_pinnable= ws[i].ram_pinnable;
            nodes[i].unified     = ws[i].unified;
        }
        idletoken_advice_row rows[IDLETOKEN_ADVISE_MAX_ROWS];
        int nr = idletoken_advise(nodes, n > 0 ? n : 1, rows, IDLETOKEN_ADVISE_MAX_ROWS);
        if (nr < 0) {
            idletoken_http_send_error(conn_fd, 500, "capability advisor failed");
            free(req.body);
            return;
        }
        static char body[16384];
        int len = idletoken_advise_json(rows, nr, n > 0 ? n : 1, body, sizeof body);
        if (len < 0) {
            idletoken_http_send_error(conn_fd, 500, "capability report too large");
            free(req.body);
            return;
        }
        idletoken_http_send_json(conn_fd, 200, body, (size_t)len);
        free(req.body);
        return;
    }

    int is_anthropic = !strcmp(req.path, IDLETOKEN_PATH_ANTHROPIC);
    int is_openai    = !strcmp(req.path, IDLETOKEN_PATH_OPENAI);
    int is_tokenize  = !strcmp(req.path, IDLETOKEN_PATH_TOKENIZE);
    int is_count_tok = !strcmp(req.path, IDLETOKEN_PATH_COUNT_TOK);
    int is_models    = !strcmp(req.path, IDLETOKEN_PATH_MODELS);
    /* /v1/models is the one GET among the vendor-compatible routes. */
    int method_ok = is_models ? !strcmp(req.method, "GET")
                              : !strcmp(req.method, "POST");
    if (!method_ok ||
        (!is_anthropic && !is_openai && !is_tokenize && !is_count_tok && !is_models)) {
        idletoken_http_send_error(conn_fd, 404, "no such endpoint");
        free(req.body);
        return;
    }

    /* --api-token gate (client setting apiToken): inference endpoints only. */
    if (!api_token_ok(&req, api_token)) {
        static const char unauth[] =
            "{\"error\":{\"type\":\"authentication_error\","
            "\"message\":\"missing or invalid API token\"}}";
        idletoken_http_send_json(conn_fd, 401, unauth, sizeof(unauth) - 1);
        free(req.body);
        return;
    }

    /* GET /v1/models — OpenAI-compatible model list (docs/api-surface.md §6).
     *
     * Exactly ONE entry: the model this coordinator loaded. That is not a
     * simplification, it is the honest answer — the chat handler never reads
     * `body.model`, so every id we listed beyond the loaded one would accept a
     * request and answer with a different model's output. "What could this
     * hardware run" is a different question, and it already has its own
     * endpoint (/idletoken/v1/capability); this one means "what can I ask for
     * right now and get".
     *
     * Answered even before the engine is ready: readiness is /health's job, and
     * a client that lists, picks, and calls gets an honest 503 from the chat
     * route. Making the list itself 503 would leave a model picker empty with
     * no way to tell "still starting" from "nothing here". */
    if (is_models) {
        char body[512];
        int bl = snprintf(body, sizeof(body),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"created\":%lld,\"owned_by\":\"idletoken\"}]}",
            coord_model()->id, (long long)g_stats.started_at);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(req.body);
        return;
    }

    /* `"stream":true` → SSE (OpenAI chunk frames / Anthropic event sequence,
     * wire-compatible with the platform gateway's controllers and Claude
     * Code). Chat endpoints only; non-stream requests behave exactly as
     * before. */
    int want_stream = (is_openai || is_anthropic) &&
        extract_bool_field((const char *)req.body, req.body ? req.body_len : 0,
                           "stream", 0);

    /* llamacpp single-machine mode: relay the inference routes to the local
     * llama-server. Placed BEFORE the tokenizer/mock branch on purpose — an
     * engine that is not READY must answer 503 naming its state, and may never
     * fall through to the mock or the ds4 paths (v2 hard invariant #4). The
     * mock stays reachable only in the pre-existing no-engine configuration. */
    if (g_llama) {
        if (is_tokenize)       llama_tokenize_route(conn_fd, &req);
        else if (is_count_tok) llama_count_tokens_route(conn_fd, &req);
        else                   llama_chat_route(conn_fd, &req, is_anthropic, want_stream);
        free(req.body);
        return;
    }

    if (!tok_ready) {
        /* Transport-chain testing without the 80GB GGUF (G_PLAT mock branch):
         * with IDLETOKEN_MOCK_OK=1 serve a clearly-labeled mock completion so the
         * platform -> agent -> coord chain can be proven end to end. Off by
         * default — a production coord with a broken engine must keep failing
         * loudly (503), never quietly serve mock text. /idletoken/v1/tokenize stays 503
         * either way: without the vocab we cannot count tokens honestly. */
        if (getenv("IDLETOKEN_MOCK_OK") && (is_openai || is_anthropic)) {
            char mock_user[2048] = "";
            if (req.body && req.body_len > 0)
                idletoken_http_json_extract_str((const char *)req.body, req.body_len,
                                             "content", mock_user, sizeof(mock_user));
            char esc[4096];
            json_escape_text(esc, sizeof(esc), mock_user, strlen(mock_user));
            /* Mock streaming: same markers, split into word-sized frames so
             * the SSE wire shape (≥2 deltas + trailer) is testable without
             * the 80GB GGUF (scripts/sse_smoke.sh). */
            if (want_stream) {
                idletoken_sse sse = { conn_fd, is_anthropic, 0, "mock", 0 };
                sse.created = (long long)time(NULL);
                char full[4352];
                snprintf(full, sizeof(full), "[IDLETOKEN MOCK ENGINE] echo: %s", esc);
                sse_begin(&sse, 0);
                sse_stream_words(&sse, full);
                sse_finish(&sse, 0, 0, 1);
                fprintf(stderr, "coord: chat: MOCK stream reply "
                                "(engine absent, IDLETOKEN_MOCK_OK set)\n");
                free(req.body);
                return;
            }
            char body[8192];
            int bl;
            if (is_anthropic) {
                bl = snprintf(body, sizeof(body),
                    "{\"id\":\"msg_idletoken_mock\",\"type\":\"message\","
                     "\"role\":\"assistant\",\"model\":\"%s\","
                     "\"content\":[{\"type\":\"text\","
                                    "\"text\":\"[IDLETOKEN MOCK ENGINE] echo: %s\"}],"
                     "\"stop_reason\":\"end_turn\","
                     "\"usage\":{\"input_tokens\":0,\"output_tokens\":0},"
                     "\"cache_hit\":false,\"cached_tokens\":0}",
                    coord_model()->id, esc);
            } else {
                bl = snprintf(body, sizeof(body),
                    "{\"id\":\"chatcmpl_idletoken_mock\",\"object\":\"chat.completion\","
                     "\"created\":%lld,\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,"
                                    "\"message\":{\"role\":\"assistant\","
                                    "\"content\":\"[IDLETOKEN MOCK ENGINE] echo: %s\"},"
                                    "\"finish_reason\":\"stop\"}],"
                     "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,"
                                 "\"total_tokens\":0},"
                     "\"cache_hit\":false,\"cached_tokens\":0}",
                    (long long)time(NULL), coord_model()->id, esc);
            }
            fprintf(stderr, "coord: chat: MOCK reply (engine absent, IDLETOKEN_MOCK_OK set)\n");
            if (bl > 0 && (size_t)bl < sizeof(body))
                idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
            else
                idletoken_http_send_error(conn_fd, 500, "mock response too large");
            free(req.body);
            return;
        }
        idletoken_http_send_error(conn_fd, 503,
                               "coord tokenizer engine not loaded; cannot tokenize");
        free(req.body);
        return;
    }

    /* POST /idletoken/v1/tokenize {"text":"..."} -> {"tokens":N} — billing/metering
     * endpoint (integration-plan 2.3). Counts RAW text tokens through the
     * engine's own vocab so the platform's metering shares the engine's token
     * definition; chat-template overhead is deliberately excluded (the
     * platform meters user-visible text, not our prompt framing). */
    if (is_tokenize) {
        char *text = malloc(req.body_len + 1);
        if (!text) {
            idletoken_http_send_error(conn_fd, 500, "oom");
            free(req.body);
            return;
        }
        text[0] = '\0';
        if (req.body && req.body_len > 0)
            idletoken_http_json_extract_str((const char *)req.body, req.body_len,
                                         "text", text, req.body_len + 1);
        if (!text[0]) {
            idletoken_http_send_error(conn_fd, 400, "missing or empty 'text' field");
            free(text);
            free(req.body);
            return;
        }
        int n_tok;
        if (coord_xtok) {
            /* ds4x: count via the byte-BPE encoder (special-token aware). Two
             * passes — size then fill — so long inputs aren't truncated. */
            int64_t got = ds4x_tok_encode(coord_xtok, text, NULL, 0);
            n_tok = got < 0 ? 0 : (int)got;
        } else {
            ds4_tokens toks = {0};
            ds4_tokenize_text(coord_engine, text, &toks);
            n_tok = toks.len;
            ds4_tokens_free(&toks);
        }
        char body[128];
        int bl = snprintf(body, sizeof(body),
                          "{\"tokens\":%d,\"model\":\"%s\"}",
                          n_tok, coord_model()->id);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        free(text);
        free(req.body);
        return;
    }

    /* POST /v1/messages/count_tokens -> {"input_tokens":N} — Anthropic
     * compatible (docs/api-surface.md §6). Takes the same body as
     * /v1/messages.
     *
     * Counts the prompt the chat path would ACTUALLY prefill (same helper,
     * chat template and all), not the raw message text: its only use is
     * context budgeting, and a number that disagrees with what gets prefilled
     * would be worse than none. This is also why it does not reuse
     * /idletoken/v1/tokenize — that one deliberately excludes the template
     * framing, because the platform meters user-visible text.
     *
     * Needs no cluster: tokenizing is local to the coordinator, so a
     * --tokenizer-only instance answers this too. */
    if (is_count_tok) {
        ds4_tokens cprompt = {0};
        char cfirst[128] = "";
        if (coord_encode_request_prompt(coord_engine, coord_xtok, req.body, req.body_len,
                                        1, &cprompt,
                                        cfirst, sizeof(cfirst), NULL) != 0) {
            idletoken_http_send_error(conn_fd, 400, "missing or empty 'messages'/'content'");
            ds4_tokens_free(&cprompt);
            free(req.body);
            return;
        }
        char body[128];
        int bl = snprintf(body, sizeof(body), "{\"input_tokens\":%d}", cprompt.len);
        idletoken_http_send_json(conn_fd, 200, body, (size_t)bl);
        ds4_tokens_free(&cprompt);
        free(req.body);
        return;
    }

    /* Chat needs an attached cluster. A --tokenizer-only coordinator (the
     * platform's metering instance) has none — refuse honestly instead of
     * dereferencing a NULL worker table. */
    if (n <= 0 || !ws || !lbuf) {
        idletoken_http_send_error(conn_fd, 503,
                               "no cluster attached (tokenizer-only coordinator)");
        free(req.body);
        return;
    }

    /* Cap = min(hard cap, remaining context). The hard cap went from 256 to 4096
     * (256 is nowhere near enough for coding); at the same time prompt+decode
     * must not exceed ctx, which would overflow the KV. On a cache hit
     * running_pos already sits past the prefix, so headroom is computed from it.
     * IDLETOKEN_MAX_DECODE overrides. */
    /* The ceiling is configuration now, not a constant: --max-decode carries
     * the user's setting in from the client, and 0 means "only the context
     * bounds this". IDLETOKEN_MAX_DECODE still wins for ops on a running box. */
    int hard_cap = g_max_decode > 0 ? g_max_decode : INT_MAX;
    { const char *mc = getenv("IDLETOKEN_MAX_DECODE");
      if (mc && atoi(mc) > 0) hard_cap = atoi(mc); }

    /* No max_tokens in the request = "generate until EOS or the context runs
     * out" -- what OpenAI clients, llama.cpp (n_predict -1) and Ollama all
     * assume. It used to default to **16**, so every reply to a client that
     * omitted the field was chopped after a few words with finish_reason
     * "length". Anthropic makes the field mandatory, so Claude Code never hit
     * it; a plain OpenAI client hit it every single time -- on the one endpoint
     * this product advertises as OpenAI-compatible.
     * The ctx-headroom clamp further down is what keeps "until the context runs
     * out" from overflowing the KV. */
    int max_tokens = extract_int_field((const char *)req.body,
                                       req.body ? req.body_len : 0,
                                       "max_tokens", 0);
    if (max_tokens < 1) max_tokens = hard_cap;
    if (max_tokens > hard_cap) max_tokens = hard_cap;

    const uint64_t req_id = ((uint64_t)time(NULL) << 16) ^ (uint64_t)*running_pos;

    /* 1. Encode prompt with the chat template -- the full multi-turn messages
     *    array when present (the precondition for KV prefix reuse: each turn of a
     *    continuing conversation must render the same prefix). With no messages
     *    array we fall back to the old single-content path, for compatibility
     *    with older clients and scripts. */
    ds4_tokens prompt = {0};
    char first_text[128] = "";  /* for the log summary */
    int n_msgs = 0;
    if (coord_encode_request_prompt(coord_engine, coord_xtok, req.body, req.body_len,
                                    is_anthropic, &prompt,
                                    first_text, sizeof(first_text), &n_msgs) != 0) {
        idletoken_http_send_error(conn_fd, 400, "missing or empty 'content' field");
        ds4_tokens_free(&prompt);
        free(req.body);
        return;
    }
    if (prompt.len <= 0) {
        idletoken_http_send_error(conn_fd, 500, "tokenizer produced empty prompt");
        ds4_tokens_free(&prompt);
        free(req.body);
        return;
    }
    /* The prompt's TEXT is deliberately not logged.
     *
     * This line used to print the first 40 characters of every request. On a
     * machine that shares compute, that is somebody else's prompt written to
     * this operator's disk: the sealed envelope keeps it off the wire, the
     * agent hands it to us over loopback, and then the log copied it back out.
     * "The coordinator has a momentary plaintext window" was true; the window
     * just had a tape recorder pointed at it.
     *
     * The excerpt survives as an opt-in debugging aid, and ONLY for locally
     * originated requests: whatever the operator sets, a request the platform
     * dispatched is never quoted, because that content is not theirs to read.
     * Counts stay unconditional -- they are what operating the thing needs. */
    if (getenv("IDLETOKEN_LOG_PROMPTS") && !from_platform) {
        fprintf(stderr, "coord: chat: tokenized %d-tok prompt (%d msgs) text=%.40s%s\n",
                prompt.len, n_msgs > 0 ? n_msgs : 1, first_text,
                strlen(first_text) > 40 ? "..." : "");
    } else {
        fprintf(stderr, "coord: chat: tokenized %d-tok prompt (%d msgs)\n",
                prompt.len, n_msgs > 0 ? n_msgs : 1);
    }

    /* The prompt must FIT THE CONTEXT, with room left to actually answer.
     *
     * Nothing downstream checked this. Prefill would drive the KV cursor
     * straight past ctx_size, the workers write beyond the allocation they
     * were assigned, and the request dies as a bare "cluster prefill failed"
     * -- or, worse, comes back with quiet garbage. The normal way to reach it
     * is simply a conversation that got long: the client resends the whole
     * history every turn, so an old thread eventually exceeds the window while
     * a fresh one still works. That asymmetry ("old chats are broken, new
     * chats are fine") is impossible to interpret from the old message.
     *
     * Switching models makes it arrive sooner: ctx_size is clamped to the
     * model's own ceiling at startup (1M for DSv4, 262K for the Qwen3.5
     * family, 40K for qwen3-8b), so the same conversation that fitted before
     * can stop fitting after a switch, with nothing on screen connecting the
     * two events.
     *
     * Deliberately placed BEFORE the SSE head goes out, so this is still a
     * real HTTP status rather than an error smuggled through the stream. */
    {
        /* Room for a reply worth having; below this the answer is a stub and
         * "it fits" would be a technicality. */
        const int min_reply = 16;
        if (prompt.len + min_reply > (int)ctx_size) {
            /* JSON, not send_error's text/plain: every client on both wire
             * protocols reads errors out of {"error":{"message":...}}, and a
             * bare text body reaches the user as "unexpected response" with the
             * explanation buried inside it. No escaping needed — the message is
             * assembled here from a literal and three integers. */
            char body[420];
            int bl = snprintf(body, sizeof(body),
                "{\"error\":{\"type\":\"context_length_exceeded\",\"message\":"
                "\"This conversation needs %d tokens but the cluster is running a "
                "%u-token context. Start a new conversation, or raise the context "
                "tier in Settings and restart the cluster.\"},"
                "\"prompt_tokens\":%d,\"context_size\":%u}",
                prompt.len + min_reply, ctx_size, prompt.len, ctx_size);
            fprintf(stderr, "coord: chat: prompt %d tok + %d reserve > ctx %u -> 413\n",
                    prompt.len, min_reply, ctx_size);
            if (bl > 0 && bl < (int)sizeof(body))
                idletoken_http_send_json(conn_fd, 413, body, (size_t)bl);
            else
                idletoken_http_send_error(conn_fd, 413, "conversation too long for the context");
            ds4_tokens_free(&prompt);
            free(req.body);
            return;
        }
    }

    /* 1b. Select a sequence slot and decide on KV prefix reuse (see the
     *      coord_kv_slot comment at the top of this file). v4 multi-sequence:
     *      look for a slot that can **strictly extend** this prompt (evicting an
     *      LRU slot if none can), and on a hit prefill only the suffix. With
     *      g_n_slots==1 this is exactly equivalent to v3. */
    const int sel = kv_pick_slot(prompt.v, (uint32_t)prompt.len);
    if (sel < 0) {
        /* Every slot is held by an in-flight request. This is unreachable under
         * serial execution (nothing else is in flight when we arrive) and only
         * happens once E3.3 interleaves. Replying 429 honestly, so upstream
         * retries or switches machines, beats grabbing a slot that is being
         * written -- that would have both sides clobber each other's KV and
         * produce **silently** wrong output. */
        long long est = (long long)(g_stats.service_ms_ewma > 0
                                    ? g_stats.service_ms_ewma : 1000.0);
        /* Header and body written separately, Content-Length from strlen -- the
         * same shape as the 429 for a full intake queue. A hardcoded length that
         * does not match yields half a response, which the client sees only as a
         * truncated connection: harder to diagnose than the 429 itself. */
        static const char busy_body[] =
            "{\"error\":{\"message\":\"all sequence slots busy\"}}";
        char hdr[256];
        int hl = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n"
            "Retry-After: %lld\r\nX-IdleToken-Est-Wait-Ms: %lld\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n",
            (est + 999) / 1000, est, sizeof(busy_body) - 1);
        if (hl > 0 && hl < (int)sizeof(hdr)) {
            ssize_t w1 = write(conn_fd, hdr, (size_t)hl);              (void)w1;
            ssize_t w2 = write(conn_fd, busy_body, sizeof(busy_body) - 1); (void)w2;
        }
        fprintf(stderr, "coord: every sequence slot is in flight -> 429, est_wait=%lldms\n", est);
        ds4_tokens_free(&prompt);
        free(req.body);
        return;
    }
    coord_kv_slot *hs = &g_slots[sel];
    hs->used_at = ++g_slot_clock;
    hs->in_flight = 1;          /* cleared only in coord_req_finish */
    uint32_t cached_tokens = 0;
    int      cache_hit = 0;
    if (kv_slot_extends(sel, prompt.v, (uint32_t)prompt.len)) {
        cache_hit = 1;
        cached_tokens = hs->len;
        fprintf(stderr, "coord: chat: KV prefix hit on slot %d — reuse %u tok, prefill %d new\n",
                sel, cached_tokens, prompt.len - (int)cached_tokens);
    } else {
        /* A miss means this slot starts a new session: reset the KV cursor to
         * zero. The first prefill chunk carries is_first_chunk, and a REAL worker
         * that receives it calls ds4_session_rewind(0) on **that sequence's**
         * session to clear the compressed layers' rolling state. That also fixes
         * two hazards of the old build at once: cross-session KV contamination (a
         * new request blindly appending after the old session, with attention
         * seeing the old content) and infer_pos growing without ever resetting
         * until it filled ctx. */
        if (hs->pos != 0)
            fprintf(stderr, "coord: chat: KV miss on slot %d — rewind to pos 0 (was %u)\n",
                    sel, hs->pos);
        hs->pos = 0;
    }
    const int fresh_start = !cache_hit;
    const uint32_t hist_base_this = cache_hit ? hs->base : hs->pos;
    const uint8_t  seq_id = (uint8_t)sel;

    /* 2. Chunked prefill: same chunk boundaries as official single-machine
     *    ds4 (ds4_prefill_chunk_cap_for_ctx), so the batched kernels — and
     *    therefore the numerics — match the baseline exactly. The final
     *    chunk's argmax becomes the first sampled token. */
    uint32_t out_tok = 0;
    int prefill_rc = 0;
    /* Open the stream BEFORE prefill, not after it.
     *
     * Everything above this point can still fail with a real HTTP status (400
     * bad request, 429 no free slot, ...), which is why the head goes out here
     * and not earlier. From here on the status is committed to 200 and failures
     * have to be reported as SSE `error` events — see the prefill_rc branch.
     *
     * What this buys: the client gets bytes immediately and a progress tick per
     * chunk, instead of a silent socket for the entire prefill. See
     * sse_prefill_tick for the failure this fixes. */
    idletoken_sse pre_sse = (idletoken_sse){ conn_fd, is_anthropic, 0, "", 0 };
    snprintf(pre_sse.id, sizeof(pre_sse.id), "%llu", (unsigned long long)req_id);
    pre_sse.created = (long long)time(NULL);
    if (want_stream) {
        sse_begin(&pre_sse, prompt.len);
        sse_prefill_tick(&pre_sse, (int)cached_tokens, prompt.len, (int)cached_tokens);
    }
    /* Per-token prefill fallback: feed the prompt one token at a time through
     * the DECODE path (single-token INFER) instead of batched chunks. Slower
     * and numerically slightly different (per-token vs batched FP8 rounding),
     * but it avoids the batch-prefill kernels — useful for a heterogeneous
     * cluster where some GPU (e.g. an sm_75 Turing card) mishandles the batched
     * prefill path. Set IDLETOKEN_NO_CHUNKED_PREFILL=1 to force it. */
    uint32_t chunk_cap = ds4_prefill_chunk_cap_for_ctx((int)ctx_size);
    if (chunk_cap == 0) chunk_cap = 1;
    if (getenv("IDLETOKEN_NO_CHUNKED_PREFILL")) chunk_cap = 1;
    /* On a hit we start past the common prefix (KV [base, base+cached) is already
     * materialized, so we simply continue); for a new session (fresh_start) the
     * first chunk carries is_first_chunk, which makes the worker rewind. */
    for (int off = (int)cached_tokens; off < prompt.len; ) {
        uint32_t chunk = (uint32_t)(prompt.len - off);
        if (chunk > chunk_cap) chunk = chunk_cap;
        int first = fresh_start && off == (int)cached_tokens;
        int rc = coord_infer_round(ws, n, req_id ^ (uint64_t)off, hs->pos,
                                   (const uint32_t *)&prompt.v[off], chunk,
                                   first, seq_id, lbuf, lbuf_cap, &out_tok);
        if (rc != 0) { prefill_rc = -1; break; }
        hs->pos += chunk;
        off += (int)chunk;
        if (want_stream) {
            sse_prefill_tick(&pre_sse, off, prompt.len, (int)cached_tokens);
            /* The tick is also how we notice the client left. A client that
             * gave up (its own read timeout, or the user closing the window)
             * used to go unnoticed until the first decode write, so the cluster
             * ground through the rest of a prefill nobody would ever read —
             * and the user's NEXT request queued behind that ghost, which is
             * how one timeout turns into a cascade of them. */
            if (pre_sse.failed) {
                fprintf(stderr, "coord: chat: client hung up during prefill "
                                "(%d/%d tok) — abandoning the request\n", off, prompt.len);
                prefill_rc = -2;
                break;
            }
        }
    }
    if (prefill_rc != 0) {
        kv_slot_reset(sel);  /* how far the KV actually got is unknown: invalidate this slot's history conservatively */
        hs->in_flight = 0;   /* the slot MUST be released: miss one early-return
                              * path and a slot leaks permanently, until enough
                              * have leaked that everything 429s -- with no clue
                              * which path leaked them */
        *running_pos = hs->pos;
        /* prefill_rc == -2 is "the client is already gone": there is nobody to
         * tell, and writing more only produces another EPIPE. */
        if (prefill_rc != -2) {
            if (want_stream) {
                /* The 200 + head already went out, so the honest channel is the
                 * stream. A bare socket close here would look to the client
                 * like a successful empty reply. */
                sse_error(&pre_sse, "cluster prefill failed");
                sse_finish(&pre_sse, prompt.len, 0, 0);
            } else {
                idletoken_http_send_error(conn_fd, 503, "cluster prefill failed");
            }
        }
        ds4_tokens_free(&prompt);
        free(req.body);
        return;
    }

    /* 3. Decode loop, streaming-aware. The SSE head + preamble went out before
     *    prefill (see pre_sse above), so a prefill failure is an SSE `error`
     *    event rather than an HTTP 503. Each sampled token is detokenized immediately
     *    (ds4_token_text) and accumulated into text_out for the non-stream
     *    body/logs; with "stream":true it is ALSO pushed as one SSE frame,
     *    held back only while it ends mid-UTF-8-sequence (BPE tokens can
     *    split one Unicode char across frames otherwise). */
    /* 3. Decode and reply: handed to the decoder built on coord_req (defined
     *    above in this file). From here on, all of a request's state lives in `r`
     *    rather than in this function's locals -- the precondition for E3 (PP
     *    micro-batching): the executor can only hold K requests and interleave
     *    between them if each request's state is an object that can be put down
     *    and picked back up. Today the call is still inline and strictly one at a
     *    time, behaving exactly as before the refactor.
     *    Note: freeing prompt / req.body / generated and sending the HTTP
     *    response are entirely the decoder's responsibility; this function must
     *    not touch them afterwards. */
    coord_req r = (coord_req){0};
    r.conn_fd        = conn_fd;
    r.is_anthropic   = is_anthropic;
    r.want_stream    = want_stream;
    r.sse            = pre_sse;   /* head + preamble already on the wire */
    r.sse_started    = 1;
    r.req_id         = req_id;
    r.sel            = sel;
    r.hs             = hs;
    r.seq_id         = seq_id;
    r.cache_hit      = cache_hit;
    r.cached_tokens  = cached_tokens;
    r.hist_base_this = hist_base_this;
    r.prompt         = prompt;
    r.http_body      = req.body;
    r.max_tokens     = max_tokens;
    r.next_token     = (int)out_tok;
    r.exec_start_ms  = exec_start_ms;
    const coord_exec x = {
        ws, n, lbuf, lbuf_cap, running_pos, coord_engine, coord_xtok, ctx_size,
    };
    if (out_parked) {
        coord_req *hr = malloc(sizeof(*hr));
        if (hr) {
            *hr = r;
            if (coord_req_begin(hr, &x) == 0) { *out_parked = hr; return; }
            free(hr);          /* a failed begin already cleaned up (reply, free, slot release) */
            return;
        }
        /* malloc failed: fall back to inline execution rather than dropping the request. */
    }
    coord_req_decode_and_reply(&r, &x);
}

/* --- llamacpp single-machine mode (v2 rebuild WS-B1+B3) --------------------
 *
 * No workers, no cluster wait: spawn + supervise a local llama-server (the
 * sidecar module owns that lifecycle) and serve the coordinator's own HTTP
 * surface on top of it. The serve loop mirrors --tokenizer-only: synchronous,
 * one connection at a time — llama-server does its own request batching, so
 * the coordinator adds no interleaving of its own here.
 *
 * Returns the process exit code. */
/* SIGINT/SIGTERM in llamacpp mode: set a flag and let accept() return EINTR,
 * so the serve loop breaks and the sidecar shutdown actually runs. Without
 * this, killing the coordinator orphaned the llama-server child — found in
 * the very first smoke run of this mode. */
static volatile sig_atomic_t g_llama_stop_sig;
static void llama_stop_handler(int sig) { (void)sig; g_llama_stop_sig = 1; }

/* --- rpc worker health attribution (WS-C, cluster mode only) ---------------
 *
 * When the engine sidecar leaves READY in cluster mode, the one question the
 * operator has is "WHICH machine is the problem?" — and llama-server's own log
 * only says an RPC endpoint failed, with no machine name. This thread watches
 * the sidecar state and, on a transition into RESTARTING/FAILED, TCP-probes
 * every worker endpoint and prints a per-machine verdict. A plain TCP connect
 * cannot check TLS credentials (the PSK handshake happens deeper), so the
 * reachable-but-failing case names the credential as the next suspect.
 * Runs on Windows too (winpthread is linked; sleep() is shimmed) — a Windows
 * coordinator is a supported configuration since 2026-08-15 and its operator
 * needs the same per-machine verdict. */
static void *rpc_peer_monitor(void *arg) {
    (void)arg;
    idletoken_llama_state last = IDLETOKEN_LLAMA_OFF;
    for (;;) {
        sleep(2);
        idletoken_llama *lc = g_llama;
        if (!lc) return NULL;
        idletoken_llama_state st = idletoken_llama_get_state(lc);
        if (st != last &&
            (st == IDLETOKEN_LLAMA_RESTARTING || st == IDLETOKEN_LLAMA_FAILED)) {
            fprintf(stderr, "coord: engine is %s — probing the %d rpc worker(s):\n",
                    idletoken_llama_state_name(st), g_n_rpc_peers);
            for (int i = 0; i < g_n_rpc_peers; i++) {
                int fd = idletoken_connect_tcp(g_rpc_peers[i].endpoint);
                if (fd < 0) {
                    fprintf(stderr,
                            "coord:   worker %s (%s): UNREACHABLE — the machine is "
                            "down, its rpc-server died, or a firewall closed the "
                            "port\n",
                            g_rpc_peers[i].hostname, g_rpc_peers[i].endpoint);
                } else {
                    close(fd);
                    fprintf(stderr,
                            "coord:   worker %s (%s): TCP reachable — if the engine "
                            "keeps failing here, suspect its TLS credential (a "
                            "wrong GGML_RPC_PSK is refused at handshake); see the "
                            "engine log\n",
                            g_rpc_peers[i].hostname, g_rpc_peers[i].endpoint);
                }
            }
        }
        last = st;
    }
}

/* HEARTBEAT cadence on the worker control links. Workers give up after 60 s
 * of silence (four missed beats), so keep interval << timeout. */
#define IDLETOKEN_HB_INTERVAL_S 15

/* Send a HEARTBEAT frame down every live worker control link. Two jobs:
 * (a) a coordinator that hangs or gets SIGSTOPped keeps its sockets open —
 *     without traffic the workers cannot tell that from idling, and their
 *     rpc-server stays an open compute port with nobody accountable for it;
 * (b) a failed send is the earliest cross-check that a worker machine dropped
 *     — name the machine now, before the engine-level failure surfaces. */
static void llama_peers_heartbeat(time_t *last) {
    if (g_n_rpc_peers <= 0) return;
    time_t now = time(NULL);
    if (*last != 0 && now - *last < IDLETOKEN_HB_INTERVAL_S) return;
    *last = now;
    for (int i = 0; i < g_n_rpc_peers; i++) {
        if (g_rpc_peers[i].fd <= 0) continue;
        idletoken_msg_header h = {
            .magic = IDLETOKEN_PROTO_MAGIC,
            .version = IDLETOKEN_PROTO_VERSION,
            .msg_type = IDLETOKEN_MSG_HEARTBEAT,
            .payload_bytes = 0,
            .request_id = 0,
            .stage_id = IDLETOKEN_STAGE_COORD,
            .segment_id = IDLETOKEN_SEGMENT_NONE,
        };
        if (idletoken_send_msg(g_rpc_peers[i].fd, &h, NULL, 0) != 0) {
            fprintf(stderr, "coord: heartbeat to %s (%s) failed: %s — that "
                            "machine is unreachable or its worker exited; if "
                            "generation stalls, that is the machine to check\n",
                    g_rpc_peers[i].hostname, g_rpc_peers[i].endpoint,
                    strerror(errno));
            g_rpc_peers[i].fd = -1;
        }
    }
}

/* 1 s readability wait on the HTTP listener, so the accept loop has a pulse
 * for heartbeats and signal checks even with no traffic. Same dual-platform
 * shape as the worker's rpc_coord_readable. */
static int llama_lfd_readable(int fd, int timeout_ms) {
#ifdef _WIN32
    fd_set rd;
    struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    FD_ZERO(&rd);
    FD_SET((SOCKET)fd, &rd);
    return select(0, &rd, NULL, NULL, &tv) > 0;
#else
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int pr = poll(&pfd, 1, timeout_ms);
    return pr > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR));
#endif
}

static int run_llamacpp_mode(const char *llama_bin, const char *llama_gguf,
                             int llama_port, const char *api_bind,
                             const char *api_token, uint32_t ctx_size,
                             const char *cluster_args,
                             const idletoken_rpc_peer *peers, int n_peers) {
    printf("  mode        : llamacpp %s (engine = llama-server sidecar)\n",
           n_peers > 0 ? "cluster (ggml-RPC + TLS)" : "single-machine");
    printf("  engine bin  : %s\n", llama_bin);
    if (cluster_args && cluster_args[0])
        printf("  cluster args: %s\n", cluster_args);
    printf("  engine gguf : %s\n", llama_gguf);
    printf("  engine port : 127.0.0.1:%d (loopback only)\n", llama_port);
    printf("  api bind    : %s\n", api_bind);
    printf("  api token   : %s\n", (api_token && api_token[0]) ? "required" : "off");
    printf("  ctx size    : %u\n\n", ctx_size);

    /* Pre-flight the two paths so a typo fails with a sentence, not with a
     * respawn loop chewing through its backoff budget. */
    struct stat st;
    if (stat(llama_bin, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "idletoken-coord: --llama-server-bin is not a file: %s\n",
                llama_bin);
        return 2;
    }
    if (stat(llama_gguf, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "idletoken-coord: --llama-gguf is not a file: %s\n",
                llama_gguf);
        return 2;
    }

    /* Engine log: its own file (IDLETOKEN_LLAMA_LOG overrides), never our
     * stderr — part of the no-prompt-in-coord-logs invariant. */
    char log_path[512];
    const char *log_env = getenv("IDLETOKEN_LLAMA_LOG");
    if (log_env && log_env[0]) {
        snprintf(log_path, sizeof(log_path), "%s", log_env);
    } else {
        /* Absolute, under the user's state dir — NOT relative to cwd. A
         * bundled client launched from Finder runs with cwd "/", where the
         * relative path is unwritable; the sidecar then falls back to our own
         * stderr, which is exactly what this invariant forbids. */
        const char *home = getenv("HOME");
#ifdef _WIN32
        if (!home || !home[0]) home = getenv("USERPROFILE");
#endif
        if (home && home[0]) {
            char dir[400];
            snprintf(dir, sizeof(dir), "%s/.idletoken", home);
#ifdef _WIN32
            _mkdir(dir);
#else
            mkdir(dir, 0700);
#endif
            snprintf(log_path, sizeof(log_path), "%s/llama-server-%d.log",
                     dir, llama_port);
        } else {
            snprintf(log_path, sizeof(log_path), "idletoken-llama-server-%d.log",
                     llama_port);
        }
    }

    char err[256] = "";
    g_llama = idletoken_llama_start(llama_bin, llama_gguf, llama_port, ctx_size,
                                    cluster_args, log_path, err, sizeof(err));
    if (!g_llama) {
        fprintf(stderr, "idletoken-coord: could not start the llama.cpp sidecar: %s\n",
                err);
        return 1;
    }

    g_n_rpc_peers = (peers && n_peers > 0)
                        ? (n_peers > IDLETOKEN_LLPLAN_MAX_NODES
                               ? IDLETOKEN_LLPLAN_MAX_NODES : n_peers)
                        : 0;
    if (g_n_rpc_peers > 0)
        memcpy(g_rpc_peers, peers, (size_t)g_n_rpc_peers * sizeof(peers[0]));
    /* Cluster mode: per-worker health attribution when the engine degrades. */
    if (g_n_rpc_peers > 0) {
        pthread_t mt;
        if (pthread_create(&mt, NULL, rpc_peer_monitor, NULL) == 0)
            pthread_detach(mt);
    }

    ignore_sigpipe();
#ifndef _WIN32
    {
        /* Deliberately no SA_RESTART: accept() must come back with EINTR. */
        struct sigaction sa = {0};
        sa.sa_handler = llama_stop_handler;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }
#endif
    int lfd = idletoken_listen_tcp(api_bind);
    if (lfd < 0) {
        fprintf(stderr, "coord: http listen(%s): %s\n", api_bind, strerror(errno));
        idletoken_llama_shutdown(g_llama);
        g_llama = NULL;
        return 1;
    }
    g_stats.started_at = (long long)time(NULL);
    fprintf(stderr, "coord: llamacpp mode — HTTP API on %s (model loading in the "
                    "background; chat answers 503 until the engine is ready). "
                    "Ctrl-C to stop.\n", api_bind);
    uint32_t pos = 0;
    time_t last_hb = 0;
    for (;;) {
        int ready = llama_lfd_readable(lfd, 1000);
        if (g_llama_stop_sig) {
            fprintf(stderr, "coord: signal received — stopping the sidecar\n");
            break;
        }
        llama_peers_heartbeat(&last_hb);
        if (!ready) continue;
        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "coord: http accept: %s\n", strerror(errno));
            break;
        }
        handle_http_request(cfd, NULL, 0, NULL, 0, &pos,
                            NULL, NULL, ctx_size, api_token, NULL);
        close(cfd);
    }
    close(lfd);
    idletoken_llama_shutdown(g_llama);
    g_llama = NULL;
    fprintf(stderr, "\ncoord: shutting down.\n");
    return 0;
}

/* --- cluster RPC PSK persistence (WS-C2) -----------------------------------
 * One 32-byte PSK per cluster, minted on first use and persisted under
 * ~/.idletoken/ so restarts keep the same credential. It travels to workers
 * ONLY inside the pairing channel (idletoken_pair_wrap_secret), never in the
 * clear on the LAN. */
static void rpc_psk_path(char *out, size_t cap) {
    const char *env = getenv("IDLETOKEN_RPC_PSK_FILE");
    if (env && env[0]) { snprintf(out, cap, "%s", env); return; }
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    snprintf(out, cap, "%s\\IdleToken\\rpc_psk", base && base[0] ? base : ".");
#else
    const char *home = getenv("HOME");
    snprintf(out, cap, "%s/.idletoken/rpc_psk", home && home[0] ? home : ".");
#endif
}

/* Load the persisted PSK (64 hex chars) or mint + persist a fresh one.
 * Also fills `raw` with the 32 bytes. Returns 0 / -1. */
static int rpc_psk_load_or_mint(char hex[65], uint8_t raw[32]) {
    char path[512];
    rpc_psk_path(path, sizeof(path));

    FILE *f = fopen(path, "r");
    if (f) {
        char buf[80] = "";
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[n] = '\0';
        char *e = buf + strlen(buf);
        while (e > buf && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) *--e = '\0';
        if (idletoken_hex64_valid(buf)) {
            memcpy(hex, buf, 65);
            for (int i = 0; i < 32; i++) {
                unsigned v = 0;
                sscanf(buf + i * 2, "%2x", &v);
                raw[i] = (uint8_t)v;
            }
            fprintf(stderr, "coord: cluster RPC PSK loaded from %s\n", path);
            return 0;
        }
        fprintf(stderr, "coord: %s exists but is not 64 hex chars — minting a "
                        "fresh PSK over it\n", path);
    }

    if (idletoken_disc_random_bytes(raw, 32) != 0) {
        fprintf(stderr, "coord: could not mint an RPC PSK (no entropy)\n");
        return -1;
    }
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[i * 2]     = hx[raw[i] >> 4];
        hex[i * 2 + 1] = hx[raw[i] & 15];
    }
    hex[64] = '\0';

    /* Best-effort mkdir of the parent, then 0600 write. A failed persist is
     * loud but not fatal: the cluster still works, it just re-mints next run
     * (workers get the new PSK through pairing anyway). */
    char dir[512];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
#ifdef _WIN32
    { char *bs = strrchr(dir, '\\'); if (bs > slash) slash = bs; }
#endif
    if (slash) {
        *slash = '\0';
#ifdef _WIN32
        _mkdir(dir);
#else
        mkdir(dir, 0700);
#endif
    }
    f = fopen(path, "w");
    if (f) {
        fprintf(f, "%s\n", hex);
        fclose(f);
#ifndef _WIN32
        chmod(path, 0600);
#endif
        fprintf(stderr, "coord: cluster RPC PSK minted and persisted to %s\n", path);
    } else {
        fprintf(stderr, "coord: WARNING: could not persist the RPC PSK to %s "
                        "(%s) — a restart will mint a new one\n",
                path, strerror(errno));
    }
    return 0;
}

/* Tolerant IDLETOKEN_ALLOW_SMALL_CLUSTER parse (cmd.exe leaves a trailing
 * space inside the value; see the legacy path's comment). */
static int allow_small_cluster_env(void) {
    const char *ovr = getenv("IDLETOKEN_ALLOW_SMALL_CLUSTER");
    if (!ovr) return 0;
    while (*ovr == ' ' || *ovr == '\t') ovr++;
    if (*ovr != '1') return 0;
    const char *p = ovr + 1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return *p == '\0';
}

/* --- llamacpp cluster mode (v2 WS-C + the B2 plan consumer) ----------------
 *
 * Accept `n_remote` rpc-supervisor workers over the existing pairing/HELLO
 * channel, enforce the WS-C invariants (pairing mandatory, one llama.cpp
 * version, no overlay endpoints), hand each worker the cluster TLS PSK, run
 * the WS-B2 planner over coordinator + workers, and drive one local
 * llama-server with --rpc/--device/--tensor-split from the plan.
 *
 * Device order is load-bearing (G-PRIV-7 precondition): by default llama.cpp
 * puts RPC devices BEFORE local ones and assigns layer 0 to the first device,
 * which would hand the first layers to a remote machine. We therefore pass an
 * explicit --device list with the LOCAL device first, matching the plan's
 * coordinator-first tensor_split order. llama.cpp additionally pins the input
 * (token_embd) layer to the host CPU unconditionally, so embedding lookup and
 * layer 0 both stay on this machine. The packet-level G-PRIV-7 gate lands in
 * WS-F. */
static int run_llamacpp_cluster_mode(
        const char *llama_bin, const char *llama_gguf, int llama_port,
        const char *api_bind, const char *api_token, uint32_t ctx_size,
        const char *bind, int disc_port, int n_remote,
        const char *pair_code_in, int create,
        const char *pair_acct, const char *acct_token, const char *rendezvous,
        const idletoken_llm_model_size *msize, const idletoken_node_mem *me) {
    if (n_remote < 1 || n_remote > IDLETOKEN_LLPLAN_MAX_NODES - 1) {
        fprintf(stderr, "idletoken-coord: cluster mode supports 1..%d remote "
                        "workers (got %d)\n", IDLETOKEN_LLPLAN_MAX_NODES - 1, n_remote);
        return 2;
    }

    /* Engine version of THIS machine — the value every worker must match.
     * Unprovable = refuse: the invariant cannot be enforced by guessing. */
    char self_ver[IDLETOKEN_ENGINE_VERSION_MAX] = "";
    if (idletoken_engine_version(llama_bin, self_ver, sizeof(self_ver)) != 0) {
        fprintf(stderr, "idletoken-coord: cannot determine the engine version "
                        "(`%s --version` failed) — refusing to form a cluster "
                        "whose version invariant cannot be checked\n", llama_bin);
        return 2;
    }
    fprintf(stderr, "coord: engine version %s (cluster invariant: every node "
                    "must match)\n", self_ver);

    /* Pairing is MANDATORY here: the TLS PSK travels wrapped under the pairing
     * session key, and without pairing there is no key to wrap it with. */
    idletoken_pair_id pair_id;
    char minted[16] = "";
    if (pair_acct) {
        if (!acct_token || !rendezvous) {
            fprintf(stderr, "idletoken-coord: --pair-account needs "
                            "--account-token and --rendezvous\n");
            return 2;
        }
        if (idletoken_pair_id_from_account(&pair_id, pair_acct, acct_token,
                                           rendezvous) != 0) {
            fprintf(stderr, "idletoken-coord: bad account pairing spec\n");
            return 2;
        }
    } else {
        if (create && !pair_code_in) {
            if (idletoken_pair_code_mint(minted, sizeof(minted)) != 0) {
                fprintf(stderr, "idletoken-coord: could not mint a join code\n");
                return 1;
            }
            pair_code_in = minted;
        }
        if (!pair_code_in) {
            fprintf(stderr, "idletoken-coord: cluster mode requires pairing "
                            "(--create / --pair-code / --pair-account): the "
                            "RPC TLS credential is delivered through the "
                            "pairing channel and must never cross the LAN in "
                            "the clear\n");
            return 2;
        }
        if (!idletoken_pair_code_valid(pair_code_in)) {
            fprintf(stderr, "idletoken-coord: invalid join code '%s'\n", pair_code_in);
            return 2;
        }
        if (idletoken_pair_id_from_code(&pair_id, pair_code_in) != 0) {
            fprintf(stderr, "idletoken-coord: bad join code\n");
            return 2;
        }
    }

    /* The cluster TLS PSK (persisted; see rpc_psk_load_or_mint). */
    char psk_hex[65] = "";
    uint8_t psk_raw[32];
    if (rpc_psk_load_or_mint(psk_hex, psk_raw) != 0) return 1;

    int lfd = idletoken_listen_tcp(bind);
    if (lfd < 0) {
        fprintf(stderr, "idletoken-coord: listen(%s): %s\n", bind, strerror(errno));
        return 1;
    }

    /* Advertise on the LAN so workers join by code with no manual address. */
    idletoken_discovery *disc = NULL;
    {
        char lan_ip[64] = "127.0.0.1";
        idletoken_local_ipv4(lan_ip, sizeof(lan_ip));
        int coord_port = 14100;
        { const char *c = strrchr(bind, ':'); if (c) coord_port = atoi(c + 1); }
        char adv_addr[80];
        snprintf(adv_addr, sizeof(adv_addr), "%s:%d", lan_ip, coord_port);
        disc = idletoken_discovery_multi((uint16_t)disc_port, NULL);
        if (!disc || disc->advertise(disc, &pair_id, adv_addr) != 0) {
            fprintf(stderr, "idletoken-coord: failed to start LAN advertising\n");
            if (disc) disc->destroy(disc);
            close(lfd);
            return 1;
        }
        fprintf(stderr, "\ncoord: cluster pairing active (%s mode) — advertising "
                        "%s on udp/%d\n",
                pair_id.mode == IDLETOKEN_PAIR_MODE_ACCOUNT ? "account" : "code",
                adv_addr, disc_port);
        if (pair_acct == NULL && pair_code_in)
            fprintf(stderr, "\n  ================  JOIN CODE:  %s  ================\n\n",
                    pair_code_in);
    }

    static idletoken_worker_info ws[IDLETOKEN_LLPLAN_MAX_NODES - 1];
    memset(ws, 0, sizeof(ws));
    int n = 0;
    while (n < n_remote) {
        fprintf(stderr, "coord: waiting for rpc worker %d/%d on %s\n",
                n + 1, n_remote, bind);
        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) {
            fprintf(stderr, "coord: accept: %s\n", strerror(errno));
            if (disc) disc->destroy(disc);
            close(lfd);
            return 1;
        }
        ws[n].fd = cfd;

        if (idletoken_pair_server_auth(cfd, &pair_id, ws[n].session_key) != 0) {
            fprintf(stderr, "coord: rejected an unauthenticated join attempt (%s)\n",
                    strerror(errno));
            memset(ws[n].session_key, 0, sizeof(ws[n].session_key));
            close(cfd);
            continue;
        }
        ws[n].has_session_key = 1;
        {
            char fp[9];
            session_key_fp(ws[n].session_key, fp);
            fprintf(stderr, "coord: worker %d passed pairing auth (session=%s)\n",
                    n, fp);
        }

        uint64_t rid = 0;
        if (do_hello(cfd, &ws[n], &rid) != 0) { close(cfd); continue; }

        /* WS-C3: the ONE cluster invariant — same llama.cpp build everywhere.
         * (OS families may mix freely now; the old G-HOMO rule is a ds4-era
         * gate and does not apply to the llama.cpp line.) */
        if (ws[n].engine_version[0] == '\0') {
            char why[256];
            snprintf(why, sizeof(why),
                     "machine %s did not report its llama.cpp engine version — "
                     "its idletoken-worker build predates the version check. "
                     "Upgrade IdleToken on %s.",
                     ws[n].hostname, ws[n].hostname);
            fprintf(stderr, "coord: refused %s: %s\n", ws[n].hostname, why);
            send_hello_reject(cfd, rid, /*reasoncode=*/2, why);
            close(cfd);
            continue;
        }
        if (strcmp(ws[n].engine_version, self_ver) != 0) {
            char why[256];
            snprintf(why, sizeof(why),
                     "machine %s runs llama.cpp %s, this cluster runs %s — "
                     "upgrade %s so every node runs the same engine build.",
                     ws[n].hostname, ws[n].engine_version, self_ver,
                     ws[n].hostname);
            fprintf(stderr, "coord: refused %s: %s\n", ws[n].hostname, why);
            send_hello_reject(cfd, rid, /*reasoncode=*/2, why);
            close(cfd);
            continue;
        }

        /* Hard invariant #3: tensor traffic never crosses an overlay. */
        {
            char host[64] = "";
            const char *colon = strrchr(ws[n].bind_addr, ':');
            size_t hl = colon ? (size_t)(colon - ws[n].bind_addr)
                              : strlen(ws[n].bind_addr);
            if (hl < sizeof(host)) { memcpy(host, ws[n].bind_addr, hl); host[hl] = '\0'; }
            if (idletoken_ip_is_overlay(host)) {
                char why[256];
                snprintf(why, sizeof(why),
                         "machine %s offered rpc endpoint %s, which is on an "
                         "overlay network (Tailscale/CGNAT — 100.64.0.0/10 or "
                         "fd7a:115c:a1e0::/48). "
                         "Tensor traffic must use the real LAN; bind the "
                         "rpc-server to the machine's LAN interface.",
                         ws[n].hostname, ws[n].bind_addr);
                fprintf(stderr, "coord: refused %s: %s\n", ws[n].hostname, why);
                send_hello_reject(cfd, rid, /*reasoncode=*/3, why);
                close(cfd);
                continue;
            }
        }

        if (send_hello_ack(cfd, rid) != 0)          { close(cfd); continue; }
        if (recv_resource_report(cfd, &ws[n]) != 0) { close(cfd); continue; }

        /* Wildcard rewrite (same as the legacy path): a worker that could not
         * name its own LAN ip gets the peer ip of this very connection. */
        {
            char host[64] = "", pip[16] = "";
            const char *colon = strrchr(ws[n].bind_addr, ':');
            size_t hlen = colon ? (size_t)(colon - ws[n].bind_addr)
                                : strlen(ws[n].bind_addr);
            if (hlen < sizeof(host)) { memcpy(host, ws[n].bind_addr, hlen); host[hlen] = '\0'; }
            int wildcard = (host[0] == '\0' || !strcmp(host, "0.0.0.0") ||
                            !strcmp(host, "*"));
            if (wildcard && colon &&
                idletoken_peer_ip(cfd, pip, sizeof(pip)) == 0 && pip[0]) {
                int wport = atoi(colon + 1);
                char rewritten[24];
                snprintf(rewritten, sizeof(rewritten), "%s:%d", pip, wport);
                fprintf(stderr, "coord: worker %d rpc addr %s -> %s (peer ip)\n",
                        n, ws[n].bind_addr, rewritten);
                snprintf(ws[n].bind_addr, sizeof(ws[n].bind_addr), "%s", rewritten);
            }
        }

        /* WS-C2: hand over the cluster TLS PSK, wrapped under the pairing
         * session key. The plaintext PSK never touches this socket. */
        {
            uint8_t nonce[IDLETOKEN_PAIR_NONCE_BYTES];
            uint8_t ct[IDLETOKEN_SESSION_KEY_BYTES];
            uint8_t tag[IDLETOKEN_PAIR_TAG_BYTES];
            idletoken_disc_random_bytes(nonce, sizeof(nonce));
            idletoken_pair_wrap_secret(ws[n].session_key, nonce, psk_raw, ct, tag);

            uint8_t pay[4 + sizeof(nonce) + sizeof(ct) + sizeof(tag)];
            idletoken_buf b;
            idletoken_buf_init(&b, pay, sizeof(pay));
            idletoken_buf_put_u8(&b, 1);              /* payload version */
            uint8_t z3[3] = {0};
            idletoken_buf_put_bytes(&b, z3, 3);
            idletoken_buf_put_bytes(&b, nonce, sizeof(nonce));
            idletoken_buf_put_bytes(&b, ct, sizeof(ct));
            idletoken_buf_put_bytes(&b, tag, sizeof(tag));
            idletoken_msg_header h = {
                .magic = IDLETOKEN_PROTO_MAGIC,
                .version = IDLETOKEN_PROTO_VERSION,
                .msg_type = IDLETOKEN_MSG_RPC_ASSIGN,
                .payload_bytes = b.pos,
                .request_id = rid,
                .stage_id = IDLETOKEN_STAGE_COORD,
                .segment_id = IDLETOKEN_SEGMENT_NONE,
            };
            if (b.err || idletoken_send_msg(cfd, &h, pay, b.pos) != 0) {
                fprintf(stderr, "coord: send RPC_ASSIGN to %s failed: %s\n",
                        ws[n].hostname, strerror(errno));
                close(cfd);
                continue;
            }
        }

        /* Wait for the worker's rpc-server to come up (it spawns only after
         * it has the PSK). A worker that cannot start its rpc-server reports
         * an error instead — surface it and fail closed for that worker. */
        {
            uint8_t rp[512];
            idletoken_msg_header rh;
            if (idletoken_recv_msg(cfd, &rh, rp, sizeof(rp)) != 0 ||
                rh.msg_type != IDLETOKEN_MSG_RPC_READY) {
                fprintf(stderr, "coord: worker %s did not reach RPC_READY "
                                "(%s) — dropping it\n",
                        ws[n].hostname,
                        errno ? strerror(errno) : "unexpected message");
                close(cfd);
                continue;
            }
            idletoken_buf rb;
            idletoken_buf_init(&rb, rp, rh.payload_bytes);
            char ep[64] = "";
            idletoken_buf_get_str(&rb, ep, sizeof(ep));
            if (!rb.err && ep[0])
                snprintf(ws[n].bind_addr, sizeof(ws[n].bind_addr), "%s", ep);
        }

        /* "Bound" is not "reachable from here": Windows Firewall filters the
         * inbound port silently, so the worker honestly reports READY while
         * this machine cannot connect (measured, win_PC2 2026-08-15) — and the
         * engine's later failure would read as a coordinator problem. Probe
         * once now and, if blocked, name the machine that can fix it. The
         * rpc-server survives a bare connect+close (same probe the health
         * monitor uses). */
        {
            int pfd = idletoken_connect_tcp(ws[n].bind_addr);
            if (pfd < 0) {
                fprintf(stderr,
                        "coord: worker %s says its rpc-server is ready on %s, but "
                        "this machine CANNOT connect to it. The block is on the "
                        "WORKER side — usually its firewall filtering the inbound "
                        "port (on Windows the worker provisions the rule itself; "
                        "if it printed 'could not add firewall rule', run the "
                        "netsh command it showed, as admin, ON %s) — dropping "
                        "this worker\n",
                        ws[n].hostname, ws[n].bind_addr, ws[n].hostname);
                close(cfd);
                continue;
            }
            close(pfd);
        }

        fprintf(stderr,
                "coord: rpc worker %d ready: %-16s %s engine=%s os=%s "
                "vram_usable=%.1fGiB ram_usable=%.1fGiB\n",
                n, ws[n].hostname, ws[n].bind_addr, ws[n].engine_version,
                idletoken_os_family_name(ws[n].os_family),
                ws[n].vram_usable / 1073741824.0,
                ws[n].ram_usable / 1073741824.0);
        n++;
    }
    if (disc) { disc->destroy(disc); disc = NULL; }

    /* --- consume the WS-B2 plan ------------------------------------------ */
    idletoken_node_mem nodes[IDLETOKEN_LLPLAN_MAX_NODES];
    memset(nodes, 0, sizeof(nodes));
    nodes[0] = *me;   /* coordinator = node 0 (the planner pins layer 0 here) */
    for (int i = 0; i < n; i++) {
        nodes[i + 1].vram_usable = ws[i].vram_usable;
        nodes[i + 1].ram_usable  = ws[i].ram_usable;
        nodes[i + 1].unified     = ws[i].unified;
    }
    const int n_nodes = n + 1;
    const int allow_small = allow_small_cluster_env();
    if (allow_small)
        fprintf(stderr, "coord: WARNING: IDLETOKEN_ALLOW_SMALL_CLUSTER=1 — "
                        "forcing a cluster even if the model fits one machine "
                        "(test vehicle, not a supported configuration)\n");

    idletoken_llama_plan lplan;
    if (idletoken_plan_llamacpp(msize, nodes, n_nodes, 0, ctx_size,
                                allow_small, &lplan) != 0) {
        fprintf(stderr, "idletoken-coord: internal scheduler error\n");
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 1;
    }
    fprintf(stderr, "coord: scheduler: %s\n", lplan.why);
    if (lplan.kind == IDLETOKEN_LLPLAN_REFUSE) {
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 3;
    }
    if (lplan.kind == IDLETOKEN_LLPLAN_SINGLE) {
        fprintf(stderr, "coord: releasing %d paired worker(s) — the model fits "
                        "this machine and clustering would only add round-trip "
                        "cost\n", n);
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return run_llamacpp_mode(llama_bin, llama_gguf, llama_port, api_bind,
                                 api_token, ctx_size, NULL, NULL, 0);
    }

    /* CLUSTER. Build --rpc / --device / --tensor-split from the plan.
     *
     * The plan orders nodes coordinator-first. We keep exactly that order in
     * all three lists: the local device is named FIRST in --device (so it
     * receives the first tensor_split share, i.e. layer 0), and RPC devices
     * follow in --rpc order (llama.cpp numbers them RPC0, RPC1, ... in
     * registration order). --rpc must precede --device on the command line —
     * arg handlers run in argv order and RPC device names only exist after
     * --rpc registered the servers. */
    if (lplan.order[0] != 0 || lplan.tensor_split[0] <= 0.0) {
        fprintf(stderr, "idletoken-coord: plan violates the layer-0 pin "
                        "(order[0]=%d split[0]=%.4f) — refusing (privacy "
                        "invariant: layer 0 + embedding stay on the "
                        "coordinator)\n",
                lplan.order[0], lplan.tensor_split[0]);
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 1;
    }

    const char *local_dev = getenv("IDLETOKEN_LLAMA_DEVICE");
    if (!local_dev || !local_dev[0]) {
#ifdef __APPLE__
        local_dev = "MTL0";
#else
        local_dev = "CUDA0";
#endif
    }

    char rpc_list[576] = "", dev_list[320] = "", split_list[320] = "";
    snprintf(dev_list, sizeof(dev_list), "%s", local_dev);
    snprintf(split_list, sizeof(split_list), "%.4f", lplan.tensor_split[0]);
    static idletoken_rpc_peer peers[IDLETOKEN_LLPLAN_MAX_NODES];
    int n_peers = 0;
    for (int i = 1; i < lplan.n_nodes; i++) {
        const int widx = lplan.order[i] - 1;   /* nodes[k] -> ws[k-1] */
        if (widx < 0 || widx >= n) continue;   /* cannot happen; belt+braces */
        size_t rl = strlen(rpc_list), dl = strlen(dev_list), sl = strlen(split_list);
        snprintf(rpc_list + rl, sizeof(rpc_list) - rl, "%s%s",
                 rl ? "," : "", ws[widx].bind_addr);
        snprintf(dev_list + dl, sizeof(dev_list) - dl, ",RPC%d", i - 1);
        snprintf(split_list + sl, sizeof(split_list) - sl, ",%.4f",
                 lplan.tensor_split[i]);
        snprintf(peers[n_peers].endpoint, sizeof(peers[n_peers].endpoint), "%s",
                 ws[widx].bind_addr);
        snprintf(peers[n_peers].hostname, sizeof(peers[n_peers].hostname), "%s",
                 ws[widx].hostname);
        peers[n_peers].fd = ws[widx].fd;
        n_peers++;
    }

    char cluster_args[1024];
    snprintf(cluster_args, sizeof(cluster_args),
             "--rpc %s --device %s --tensor-split %s",
             rpc_list, dev_list, split_list);

    fprintf(stderr, "\ncoord: cluster topology (device order = tensor-split "
                    "order; layer 0 on the local device):\n");
    fprintf(stderr, "  node 0 -> %-24s share=%.4f  (coordinator, %s — holds "
                    "layer 0 + token_embd)\n", "local", lplan.tensor_split[0],
            local_dev);
    for (int i = 0; i < n_peers; i++)
        fprintf(stderr, "  node %d -> %-24s share=%.4f  (RPC%d, %s)\n",
                i + 1, peers[i].endpoint, lplan.tensor_split[i + 1], i,
                peers[i].hostname);
    fprintf(stderr, "coord: G-PRIV-7 precondition holds: the local device takes "
                    "the first nonzero tensor-split share, and llama.cpp pins "
                    "the input (token_embd) layer to the host CPU. The "
                    "packet-level G-PRIV-7 gate lands in WS-F.\n\n");

    /* The llama-server child (RPC client side of the TLS transport) reads the
     * PSK from its environment, inherited across fork — and, on Windows, across
     * CreateProcess, which passes lpEnvironment=NULL and so hands the child a
     * copy of ours.
     *
     * This was `#ifndef _WIN32` until 2026-08-15, i.e. a Windows coordinator
     * spawned llama-server with NO PSK. The engine then did the right thing —
     * "refusing plaintext RPC connect", per privacy invariant #10 — and exited 1
     * five times while the coordinator dutifully restarted it. Every layer
     * behaved correctly; the cluster simply could never form on Windows.
     * setenv() is MinGW-shimmed in src/platform/win/win_compat.c. */
    setenv("GGML_RPC_PSK", psk_hex, 1);

    close(lfd);   /* the worker control fds in ws[].fd stay open on purpose:
                   * they carry the 15 s HEARTBEAT (llama_peers_heartbeat), and
                   * workers shut their rpc-server down on EOF *or* 60 s of
                   * silence — so a coordinator that hangs without closing the
                   * socket no longer leaves remote compute ports open. */
    return run_llamacpp_mode(llama_bin, llama_gguf, llama_port, api_bind,
                             api_token, ctx_size, cluster_args, peers, n_peers);
}

int main(int argc, char **argv) {
    /* A supervisor's log must be tail-able while it runs. Redirected stdio is
     * FULLY buffered (MinGW CRT buffers even stderr), so a live coordinator's
     * log file stayed at 0 bytes and the cluster matrix lane read it as dead
     * (2026-08-15). Log volume here is human-scale; unbuffer both streams. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#ifdef __linux__
    /* Opt-in (set by the client supervisor): die with the launching client so
     * even a SIGKILLed client never orphans the coordinator. Not default —
     * scripted deploys launch us under nohup and outlive their shell. */
    if (getenv("IDLETOKEN_DIE_WITH_PARENT")) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() == 1) _exit(0);
    }
#elif defined(_WIN32)
    /* Windows side of parent-death: watch the client's process handle. */
    idletoken_die_with_parent();
#endif
    const char *bind       = "0.0.0.0:14100";
    const char *api_bind   = "0.0.0.0:8000";
    /* API access token (client setting apiToken). Empty/NULL = no auth (LAN
     * default). Env fallback so the Tauri sidecar can avoid arg quoting. */
    const char *api_token  = getenv("IDLETOKEN_API_TOKEN");
    const char *model_id   = NULL;   /* --model-id; default = registry default */
    const char *model_path = NULL;   /* --model-path; default = model's own gguf */
    const char *gguf_dir   = NULL;   /* --gguf-dir; where WE find our tokenizer copy */
    const char *quant      = NULL;   /* --quant; default = model's default variant */
    int num_workers        = 1;
    int num_workers_set    = 0;      /* explicit --num-workers: in llamacpp mode
                                      * this switches on the WS-C cluster path
                                      * (N remote rpc workers + local device) */
    int n_predict          = 1;
    int max_decode         = 4096;   /* per-request generation ceiling */
    int http_serve         = 0;
    int tokenizer_only     = 0;
    uint32_t ctx_size      = 0;   /* 0 = defaulted per mode below */
    /* Pairing / discovery: when a code (or account) is given, advertise this
     * coordinator over the LAN so workers self-assemble by code — no manual
     * --coordinator on the worker side. */
    const char *pair_code  = NULL;   /* --pair-code CODE (code mode) */
    int         create     = 0;      /* --create: mint a fresh code */
    const char *pair_acct  = NULL;   /* --pair-account CLUSTER (account mode) */
    const char *acct_token = NULL;   /* --account-token JWT */
    const char *rendezvous = NULL;   /* --rendezvous HOST:PORT */
    int         disc_port  = IDLETOKEN_DISCOVERY_PORT;
    /* llamacpp single-machine mode (v2 rebuild WS-B1+B3): BOTH bin and gguf
     * given → drive a local llama-server instead of a worker cluster. Env
     * fallbacks follow the api_token pattern (the Tauri sidecar avoids arg
     * quoting). */
    const char *llama_bin  = getenv("IDLETOKEN_LLAMA_SERVER_BIN");
    const char *llama_gguf = getenv("IDLETOKEN_LLAMA_GGUF");
    int         llama_port = 18099;
    /* Per-machine usage caps (the client's "this machine's usage" sliders,
     * wire-to-B2): cap what the probe reports before planning, same contract
     * as the worker's --max-vram-mb/--max-ram-mb. 0 = uncapped. */
    long        max_vram_mb = 0;
    long        max_ram_mb  = 0;
    {
        const char *lp = getenv("IDLETOKEN_LLAMA_PORT");
        if (lp && atoi(lp) > 0) llama_port = atoi(lp);
    }

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--bind")        && i + 1 < argc) bind        = argv[++i];
        else if (!strcmp(a, "--api-bind")    && i + 1 < argc) api_bind    = argv[++i];
        else if (!strcmp(a, "--api-token")   && i + 1 < argc) api_token   = argv[++i];
        else if (!strcmp(a, "--num-workers") && i + 1 < argc) { num_workers = atoi(argv[++i]); num_workers_set = 1; }
        else if (!strcmp(a, "--ctx-size")    && i + 1 < argc) ctx_size    = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(a, "--model-id")    && i + 1 < argc) model_id    = argv[++i];
        else if (!strcmp(a, "--model-path")  && i + 1 < argc) model_path  = argv[++i];
        else if (!strcmp(a, "--gguf-dir")    && i + 1 < argc) gguf_dir    = argv[++i];
        else if (!strcmp(a, "--quant")       && i + 1 < argc) quant       = argv[++i];
        else if (!strcmp(a, "--n-predict")   && i + 1 < argc) n_predict   = atoi(argv[++i]);
        else if (!strcmp(a, "--max-decode")  && i + 1 < argc) max_decode  = atoi(argv[++i]);
        else if (!strcmp(a, "--seq-slots")   && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) { g_n_slots = 0; }
            else {
                int nv = atoi(v);
                if (nv < 1 || nv > IDLETOKEN_MAX_SEQ_SLOTS) {
                    fprintf(stderr, "idletoken-coord: --seq-slots must be 'auto' or 1..%d\n",
                            IDLETOKEN_MAX_SEQ_SLOTS);
                    return 2;
                }
                g_n_slots = nv;
            }
        }
        else if (!strcmp(a, "--concurrent-decode") && i + 1 < argc) {
            const char *cv = argv[++i];
            if (!strcmp(cv, "auto")) { g_concurrent_decode = -1; continue; }
            int nv = atoi(cv);
            if (nv < 0 || nv > IDLETOKEN_MAX_SEQ_SLOTS) {
                fprintf(stderr, "idletoken-coord: --concurrent-decode must be 0..%d\n",
                        IDLETOKEN_MAX_SEQ_SLOTS);
                return 2;
            }
            g_concurrent_decode = nv;
        }
        else if (!strcmp(a, "--http"))                        http_serve  = 1;
        else if (!strcmp(a, "--selftest"))                    return coord_selftest();
        else if (!strcmp(a, "--tokenizer-only"))              tokenizer_only = 1;
        else if (!strcmp(a, "--pair-code")   && i + 1 < argc) pair_code   = argv[++i];
        else if (!strcmp(a, "--create"))                      create      = 1;
        else if (!strcmp(a, "--pair-account")&& i + 1 < argc) pair_acct   = argv[++i];
        else if (!strcmp(a, "--account-token")&& i + 1 < argc) acct_token = argv[++i];
        else if (!strcmp(a, "--rendezvous")  && i + 1 < argc) rendezvous  = argv[++i];
        else if (!strcmp(a, "--discovery-port") && i + 1 < argc) disc_port = atoi(argv[++i]);
        else if (!strcmp(a, "--llama-server-bin") && i + 1 < argc) llama_bin  = argv[++i];
        else if (!strcmp(a, "--llama-gguf")       && i + 1 < argc) llama_gguf = argv[++i];
        else if (!strcmp(a, "--llama-port")       && i + 1 < argc) llama_port = atoi(argv[++i]);
        else if (!strcmp(a, "--max-vram-mb")      && i + 1 < argc) max_vram_mb = atol(argv[++i]);
        else if (!strcmp(a, "--max-ram-mb")       && i + 1 < argc) max_ram_mb  = atol(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "idletoken-coord: unknown argument: %s\n\n", a); usage(stderr); return 2; }
    }

    /* Resolve the model before anything touches the network: unknown ids and
     * not-yet-runnable backends must fail loudly at startup, not mid-join. */
    g_model = model_id ? idletoken_model_get(model_id) : idletoken_model_default();
    if (!g_model) {
        fprintf(stderr, "idletoken-coord: unknown model id '%s'\n", model_id);
        return 2;
    }

    /* --- llamacpp single-machine mode decision (v2 rebuild WS-B1+B3) ------
     * Both engine flags present → serve through a local llama-server sidecar,
     * no worker cluster. One flag without the other is a mistake, not a mode.
     * IDLETOKEN_FORCE_BACKEND=ds4 is the explicit escape back to the frozen
     * legacy engine — loud on purpose, so a run on the wrong engine can never
     * pass for a normal one. */
    {
        const int have_bin  = llama_bin  && llama_bin[0];
        const int have_gguf = llama_gguf && llama_gguf[0];
        if (have_bin != have_gguf) {
            fprintf(stderr, "idletoken-coord: llamacpp mode needs BOTH "
                            "--llama-server-bin and --llama-gguf (got only %s)\n",
                    have_bin ? "the binary" : "the gguf");
            return 2;
        }
        int llamacpp_mode = have_bin && have_gguf;
        if (llamacpp_mode) {
            const char *fb = getenv("IDLETOKEN_FORCE_BACKEND");
            if (fb && !strcmp(fb, "ds4")) {
                fprintf(stderr,
                    "\n"
                    "*************************************************************************\n"
                    "**  FORCED LEGACY ds4 BACKEND  (IDLETOKEN_FORCE_BACKEND=ds4)           **\n"
                    "**                                                                     **\n"
                    "**  --llama-server-bin / --llama-gguf are set but IGNORED this run:    **\n"
                    "**  inference goes through the frozen vendor/ds4 engine path.          **\n"
                    "**  Unset IDLETOKEN_FORCE_BACKEND to use the llama.cpp engine.         **\n"
                    "*************************************************************************\n\n");
                llamacpp_mode = 0;
            }
        }
        if (llamacpp_mode) {
            if (max_decode < 0) {
                fprintf(stderr, "idletoken-coord: --max-decode must be >= 0 (0 = context-bound)\n");
                return 2;
            }
            if (llama_port < 1 || llama_port > 65535) {
                fprintf(stderr, "idletoken-coord: --llama-port must be 1..65535\n");
                return 2;
            }

            /* --- open model intake (v2 WS-B4) --------------------------------
             * Without an explicit --model-id, the GGUF header is the source of
             * truth: build a runtime spec from it, so /v1/models, stats and
             * every response's "model" field name what is ACTUALLY served —
             * not the registry default. (Before this, pointing --llama-gguf at
             * a Qwen file reported "deepseek-v4-flash" while serving Qwen.)
             * `static`: g_model keeps pointing into it for the process life. */
            static idletoken_auto_model auto_model;
            if (!model_id) {
                char aerr[256] = "";
                if (idletoken_model_from_gguf(llama_gguf, &auto_model,
                                              aerr, sizeof(aerr)) != 0) {
                    fprintf(stderr, "idletoken-coord: cannot serve %s: %s\n",
                            llama_gguf, aerr);
                    return 2;
                }
                g_model = &auto_model.spec;
                fprintf(stderr,
                        "coord: auto manifest: id=%s arch=%s layers=%u vocab=%u "
                        "ctx_max=%u file=%.2f GiB kv=%.1f KiB/token\n",
                        auto_model.id, auto_model.arch,
                        (unsigned)auto_model.spec.n_layers,
                        auto_model.spec.n_vocab, auto_model.spec.ctx_max,
                        (double)auto_model.file_bytes / 1073741824.0,
                        (double)auto_model.kv_bytes_per_token / 1024.0);
            }

            /* --- scheduler preflight (v2 WS-B2) ------------------------------
             * Measure THIS machine, then ask the planner whether the model +
             * KV + overhead fit. A machine that cannot hold the model gets a
             * sentence naming the numbers, not a crash-looping sidecar. */
            idletoken_llm_model_size msize;
            if (!model_id) {
                msize.total_bytes       = auto_model.file_bytes;
                msize.n_layers          = auto_model.spec.n_layers;
                msize.kv_bytes_per_token = auto_model.kv_bytes_per_token;
            } else {
                uint64_t lb = 0, sb = 0;
                idletoken_model_weight_bytes(g_model, quant, &lb, &sb);
                msize.total_bytes = lb + sb;
                msize.n_layers    = g_model->n_layers;
                msize.kv_bytes_per_token =
                    (uint64_t)g_model->kv_bytes_per_token_layer * g_model->n_layers;
            }

            idletoken_node_mem me;
            memset(&me, 0, sizeof(me));
            const char *fake_usable = getenv("IDLETOKEN_TEST_USABLE_BYTES");
            if (fake_usable && fake_usable[0]) {
                /* TEST ONLY: lets the refusal path be exercised on a machine
                 * that would otherwise fit the model (the honest alternative is
                 * an 80 GiB download). Loud on purpose — a production run with
                 * this set must be unmistakable in the log. */
                uint64_t v = strtoull(fake_usable, NULL, 10);
                fprintf(stderr,
                        "coord: *** TEST OVERRIDE *** IDLETOKEN_TEST_USABLE_BYTES=%llu "
                        "(%.2f GiB) replaces the measured memory of this machine. "
                        "Never set this outside a test harness.\n",
                        (unsigned long long)v, (double)v / 1073741824.0);
                me.vram_usable = v;
                me.ram_usable  = v;
                me.unified     = 1;   /* counted once by the usable metric */
            } else {
                idletoken_resource_report rep;
                if (idletoken_resource_probe(&rep, NULL) != 0) {
                    fprintf(stderr,
                            "idletoken-coord: could not measure this machine's "
                            "memory (hardware probe failed — see the messages "
                            "above); refusing to guess whether %s fits. Fix the "
                            "probe's complaint and start again.\n", g_model->id);
                    return 3;
                }
                if (max_vram_mb > 0 || max_ram_mb > 0) {
                    idletoken_resource_apply_caps(&rep,
                        (uint64_t)max_vram_mb * 1024 * 1024,
                        (uint64_t)max_ram_mb * 1024 * 1024);
                    fprintf(stderr,
                            "coord: usage caps applied (this machine's sliders): "
                            "vram_usable=%.2f GiB ram_usable=%.2f GiB\n",
                            (double)rep.vram_usable / 1073741824.0,
                            (double)rep.ram_usable / 1073741824.0);
                }
                me.vram_usable  = rep.vram_usable;
                me.ram_usable   = rep.ram_usable;
                me.ram_pinnable = 0;
                me.unified      = rep.unified_memory ? 1 : 0;
            }
            const uint64_t usable = idletoken_llama_node_usable(&me);

            /* --- WS-C cluster path: remote rpc workers requested ----------
             * BEFORE the single-machine ctx fitting below: a coordinator
             * that cannot hold the model alone is exactly the machine that
             * clusters, so the single-machine refusals must not run here.
             * Cluster ctx sizing stays simple (default ask, model clamp);
             * the planner's needed() accounts for the KV bytes. */
            if (num_workers_set && num_workers >= 1) {
                uint32_t cctx = ctx_size ? ctx_size : 32768;
                if (cctx > g_model->ctx_max) {
                    fprintf(stderr, "idletoken-coord: ctx-size %u clamped to %s "
                                    "max %u\n", cctx, g_model->id, g_model->ctx_max);
                    cctx = g_model->ctx_max;
                }
                g_max_decode = max_decode;
                printf("idletoken-coord v0.1.0-pre  (llamacpp cluster mode)\n");
                printf("  model id    : %s (%s)\n", g_model->id, g_model->label);
                return run_llamacpp_cluster_mode(llama_bin, llama_gguf,
                                                 llama_port, api_bind, api_token,
                                                 cctx, bind, disc_port,
                                                 num_workers, pair_code, create,
                                                 pair_acct, acct_token,
                                                 rendezvous, &msize, &me);
            }

            /* Context: an explicit --ctx-size is honored (and fit-checked by
             * the planner below); otherwise size it from memory. Default ask
             * stays 32K — the Anthropic face must fit a real Claude Code
             * session, whose system prompt alone is ~13K tokens (measured; 8K
             * failed on first contact). The 16K floor exists for the same
             * reason: silently granting less would fail mid-conversation
             * instead of here, so we refuse loudly instead. */
            const uint32_t ctx_ask = ctx_size ? ctx_size : 32768;
            uint32_t ctx_capped = ctx_ask;
            if (ctx_capped > g_model->ctx_max) {
                fprintf(stderr, "idletoken-coord: ctx-size %u clamped to %s max %u\n",
                        ctx_capped, g_model->id, g_model->ctx_max);
                ctx_capped = g_model->ctx_max;
            }
            if (ctx_size == 0) {
                const uint32_t granted =
                    idletoken_llama_fit_ctx(usable, &msize, ctx_capped, 16384);
                if (granted == 0) {
                    fprintf(stderr,
                            "idletoken-coord: refuse: %s (%.2f GiB weights) on this "
                            "machine's %.2f GiB usable memory leaves room for less "
                            "than a 16K-token context (%.1f KiB of KV cache per "
                            "token). A Claude Code session needs ~13K tokens of "
                            "input, so a smaller window would fail mid-conversation "
                            "instead of here. Free memory, pick a smaller "
                            "quantization, or pass an explicit --ctx-size to "
                            "accept a small window deliberately.\n",
                            g_model->id,
                            (double)msize.total_bytes / 1073741824.0,
                            (double)usable / 1073741824.0,
                            (double)msize.kv_bytes_per_token / 1024.0);
                    return 3;
                }
                if (granted < ctx_capped)
                    fprintf(stderr,
                            "coord: ctx sized to %u from available memory "
                            "(asked %u; %.2f GiB usable)\n",
                            granted, ctx_capped, (double)usable / 1073741824.0);
                ctx_size = granted;
            } else {
                ctx_size = ctx_capped;
            }

            idletoken_llama_plan lplan;
            if (idletoken_plan_llamacpp(&msize, &me, 1, 0, ctx_size, 0,
                                        &lplan) != 0) {
                fprintf(stderr, "idletoken-coord: internal scheduler error\n");
                return 1;
            }
            if (lplan.kind == IDLETOKEN_LLPLAN_REFUSE) {
                fprintf(stderr, "idletoken-coord: %s\n", lplan.why);
                return 3;
            }
            fprintf(stderr, "coord: scheduler: %s\n", lplan.why);

            g_max_decode = max_decode;
            printf("idletoken-coord v0.1.0-pre  (llamacpp single-machine mode)\n");
            printf("  model id    : %s (%s)\n", g_model->id, g_model->label);
            return run_llamacpp_mode(llama_bin, llama_gguf, llama_port,
                                     api_bind, api_token, ctx_size,
                                     NULL, NULL, 0);
        }
    }

    /* Legacy (non-llamacpp) paths keep their historical 8K default. */
    if (ctx_size == 0) ctx_size = 8192;

    if (g_model->backend != IDLETOKEN_BACKEND_DS4 &&
        g_model->backend != IDLETOKEN_BACKEND_DS4X) {
        fprintf(stderr, "idletoken-coord: model '%s' backend %u not implemented by this build\n",
                g_model->id, (unsigned)g_model->backend);
        return 2;
    }
    /* ds4x runs on the generic CPU backend (small models); the `available`
     * flag stays advisory (0 until real-GGUF validation) but no longer blocks
     * bring-up on a machine that has the weights. */
    const int coord_use_ds4x = (g_model->backend == IDLETOKEN_BACKEND_DS4X);
    if (!g_model->available && !coord_use_ds4x) {
        fprintf(stderr, "idletoken-coord: model '%s' is registered but not runnable yet\n",
                g_model->id);
        return 2;
    }
    /* Large models cluster; small models do not (CLAUDE.md hard constraint).
     * Checked HERE, before the listening socket opens, so the user learns it
     * instead of watching two machines download weights and then stall.
     *
     * The marker makes the client show this sentence in the UI rather than a
     * bare "crashed" — the same channel the mixed-OS refusal uses (G_HOMO).
     *
     * The escape hatch exists for the acceptance scripts only: DSv4 is the only
     * runnable cluster model and it needs ~81 GiB, so without a small model to
     * drive them the cross-machine gates (v5_pipeline_check, xmachine_*, the
     * multi-node half of the topology matrix) would have no vehicle at all. It
     * is deliberately an env var and not a flag: a test harness sets it, a user
     * following the UI never encounters it. */
    if (num_workers > 1) {
        char why[320];
        /* Whitespace-tolerant on purpose. Windows scripts set this through
         * `cmd /c "set VAR=1 & prog"`, and cmd puts the space BEFORE the
         * separator INSIDE the value — so the engine would see "1 " and, on a
         * strict compare, silently ignore the override. That failure only
         * happens on Windows, only in the test harness, and looks like the
         * policy check misfiring. The scripts use the no-space form; this
         * accepts either, because being strict here buys nothing. */
        const char *ovr = getenv("IDLETOKEN_ALLOW_SMALL_CLUSTER");
        int allow_override = 0;
        if (ovr) {
            while (*ovr == ' ' || *ovr == '\t') ovr++;
            if (*ovr == '1') {
                const char *p = ovr + 1;
                while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
                allow_override = (*p == '\0');
            }
        }
        if (!idletoken_model_may_cluster(g_model, why, sizeof why)) {
            if (!allow_override) {
                fprintf(stderr, "idletoken-coord: " IDLETOKEN_JOIN_REFUSED_MARK "%s\n", why);
                return 2;
            }
            fprintf(stderr, "idletoken-coord: WARNING: serving '%s' across %d nodes because "
                            "IDLETOKEN_ALLOW_SMALL_CLUSTER=1. This is a test vehicle, not a "
                            "supported configuration -- %s\n",
                    g_model->id, num_workers, why);
        }
    }
    if (ctx_size > g_model->ctx_max) {
        fprintf(stderr, "idletoken-coord: ctx-size %u clamped to %s max %u\n",
                ctx_size, g_model->id, g_model->ctx_max);
        ctx_size = g_model->ctx_max;
    }
    /* Resolve precision: pick the requested variant, or the model's default.
     * A model with no variant menu keeps quant = "" (single implicit precision). */
    const idletoken_model_variant *variant = idletoken_model_variant_get(g_model, quant);
    if (variant) {
        if (quant && strcmp(variant->quant, quant) != 0)
            fprintf(stderr, "idletoken-coord: unknown quant '%s' for %s — using default '%s'\n",
                    quant, g_model->id, variant->quant);
        g_quant = variant->quant;
    } else if (quant && quant[0]) {
        fprintf(stderr, "idletoken-coord: model '%s' has no selectable precision; "
                        "ignoring --quant %s\n", g_model->id, quant);
    }
    if (!model_path)
        model_path = variant && variant->gguf[0] ? variant->gguf : g_model->default_gguf;
    /* `model_path` goes on the wire as-is; each worker resolves it under its own
     * --gguf-dir. But the COORDINATOR also needs to open a copy for its
     * tokenizer, and a bare variant filename (which is what --quant leaves us
     * with) is not resolvable from our CWD. Without this, `--quant Q8_0` built a
     * cluster whose stages loaded fine while every chat request answered
     * "coord tokenizer engine not loaded" — the documented way to pick a
     * precision left the API dead. */
    static char tok_path[1024];
    const char *coord_tok_path = model_path;
    if (gguf_dir && model_path[0] != '/' && !(model_path[0] && model_path[1] == ':')) {
        const size_t dl = strlen(gguf_dir);
        snprintf(tok_path, sizeof tok_path, "%s%s%s", gguf_dir,
                 (dl && (gguf_dir[dl - 1] == '/' || gguf_dir[dl - 1] == '\\')) ? "" : "/",
                 model_path);
        coord_tok_path = tok_path;
    }
    fprintf(stderr, "coord: serving model %s (%s, %u layers)%s%s\n",
            g_model->id, g_model->label, (unsigned)g_model->n_layers,
            g_quant[0] ? " @ " : "", g_quant);

    /* Model identity for ASSIGN_PLAN. Cheap (metadata region only) and every
     * worker can check it — including one that fetched just its own layers.
     * If we cannot read our own copy we send zeros and say so: workers treat
     * all-zero as "coordinator could not vouch", which is exactly what it
     * means. Silently sending zeros while claiming verification is worse than
     * not verifying (this field WAS all-zero with a `TODO` for months). */
    {
        char ierr[256] = "";
        if (idletoken_gguf_identity(coord_tok_path, g_model_identity, ierr, sizeof ierr) == 0) {
            g_model_identity_ok = 1;
            fprintf(stderr, "coord: model identity ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%02x", g_model_identity[i]);
            fprintf(stderr, "… (sha256 of the GGUF metadata region)\n");
        } else {
            fprintf(stderr, "coord: model identity unavailable (%s) — workers "
                            "will not be able to verify their copy\n", ierr);
        }
    }

    /* Build the pairing identity (code or account), if any. */
    idletoken_pair_id pair_id;
    int pairing = 0;
    char minted[16] = "";
    if (pair_acct) {
        if (!acct_token || !rendezvous) {
            fprintf(stderr, "idletoken-coord: --pair-account needs --account-token and --rendezvous\n");
            return 2;
        }
        if (idletoken_pair_id_from_account(&pair_id, pair_acct, acct_token, rendezvous) != 0) {
            fprintf(stderr, "idletoken-coord: bad account pairing spec\n"); return 2;
        }
        pairing = 1;
    } else if (create || pair_code) {
        if (create && !pair_code) {
            if (idletoken_pair_code_mint(minted, sizeof(minted)) != 0) {
                fprintf(stderr, "idletoken-coord: could not mint a join code\n"); return 1;
            }
            pair_code = minted;
        }
        if (!idletoken_pair_code_valid(pair_code)) {
            fprintf(stderr, "idletoken-coord: invalid join code '%s'\n", pair_code); return 2;
        }
        if (idletoken_pair_id_from_code(&pair_id, pair_code) != 0) {
            fprintf(stderr, "idletoken-coord: bad join code\n"); return 2;
        }
        pairing = 1;
    }

    /* Mint the cluster salt and derive the token-encryption key once, right
     * after the pairing identity exists (proto v7). Every worker derives the
     * same key from the same psk plus the salt we ship in ASSIGN_PLAN, so
     * nothing has to be distributed and worker<->worker links are covered too.
     *
     * Without pairing there is no psk, so there is no key: the salt stays
     * all-zero and g_cluster_key_ok stays 0. That is the honest state for a
     * --coordinator cluster, not something to paper over. */
    if (pairing) {
        if (idletoken_disc_random_bytes(g_cluster_salt, sizeof(g_cluster_salt)) != 0) {
            fprintf(stderr, "idletoken-coord: could not mint a cluster salt\n");
            return 1;
        }
        idletoken_nodecrypt_cluster_key(pair_id.psk, sizeof(pair_id.psk),
                                        g_cluster_salt, sizeof(g_cluster_salt),
                                        g_cluster_key);
        g_cluster_key_ok = 1;
        char fp[9]; session_key_fp(g_cluster_key, fp);
        fprintf(stderr, "coord: cluster token key ready (cluster=%s)\n", fp);
    } else {
        fprintf(stderr, "coord: no pairing secret — node links cannot be encrypted (cluster=none)\n");
    }

    if (max_decode < 0) {
        fprintf(stderr, "idletoken-coord: --max-decode must be >= 0 (0 = context-bound)\n");
        return 2;
    }
    if (n_predict < 0 || n_predict > 4096) {
        fprintf(stderr, "idletoken-coord: --n-predict must be 0..4096 (0 = no warmup)\n");
        return 2;
    }
    if (num_workers < 1 || num_workers > IDLETOKEN_MAX_WORKERS) {
        fprintf(stderr, "idletoken-coord: --num-workers must be 1..%d\n", IDLETOKEN_MAX_WORKERS);
        return 2;
    }

    printf("idletoken-coord v0.1.0-pre  (proto v%u, header=%zuB)\n",
           (unsigned)IDLETOKEN_PROTO_VERSION, sizeof(idletoken_msg_header));
    printf("  worker bind : %s\n", bind);
    printf("  api bind    : %s\n", api_bind);
    printf("  api token   : %s\n", (api_token && api_token[0]) ? "required" : "off");
    printf("  num workers : %d\n", num_workers);
    printf("  ctx size    : %u\n", ctx_size);
    printf("  model path  : %s\n", model_path);
    if (coord_tok_path != model_path) printf("  tokenizer   : %s\n", coord_tok_path);
    g_max_decode = max_decode;
    printf("  max_decode  : %d%s\n", max_decode, max_decode ? "" : " (context-bound)");
    printf("  n_predict   : %d\n\n", n_predict);

    /* --- tokenizer-only mode (integration-plan: platform metering) --------
     * No cluster, no warmup: open the vocab (a layer-free SPARSE shard is
     * enough — ds4 only touches the metadata/vocab pages) and serve /health
     * + /idletoken/v1/tokenize. Chat routes 503 via the no-cluster guard. This is the
     * platform's own metering instance, so billing counts with the exact
     * engine vocab without shipping the 80GB weights to the cloud. */
    if (tokenizer_only) {
        char tk_lock[64];
        snprintf(tk_lock, sizeof(tk_lock), "/tmp/ds4-%ld.lock", (long)getpid());
        setenv("DS4_LOCK_FILE", tk_lock, 1);
        ds4_engine_options teo = {
            .model_path = model_path,
            .backend = DS4_BACKEND_CPU,
            .n_threads = 0,
            .warm_weights = false,
            .quality = false,
            .load_layer_lo = 0,
            .load_layer_hi = 0,
        };
        struct stat tst;
        if (stat(model_path, &tst) != 0 || !S_ISREG(tst.st_mode)) {
            fprintf(stderr, "idletoken-coord: --tokenizer-only needs a readable "
                            "model/vocab shard: %s\n", model_path);
            return 1;   /* nothing else to serve — fail loudly */
        }
        ds4_engine *tok_engine = NULL;
        if (ds4_engine_open(&tok_engine, &teo) != 0 || !tok_engine) {
            fprintf(stderr, "idletoken-coord: tokenizer engine open failed\n");
            return 1;
        }
        fprintf(stderr, "coord: tokenizer-only — vocab ready (eos_token=%d)\n",
                ds4_token_eos(tok_engine));
        ignore_sigpipe();
        int tk_lfd = idletoken_listen_tcp(api_bind);
        if (tk_lfd < 0) {
            fprintf(stderr, "coord: http listen(%s): %s\n", api_bind, strerror(errno));
            ds4_engine_close(tok_engine);
            return 1;
        }
        fprintf(stderr, "coord: tokenizer-only HTTP on %s. Ctrl-C to stop.\n", api_bind);
        uint32_t tk_pos = 0;
        for (;;) {
            int cfd = idletoken_accept_tcp(tk_lfd);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "coord: http accept: %s\n", strerror(errno));
                break;
            }
            handle_http_request(cfd, NULL, 0, NULL, 0, &tk_pos,   /* parking mode does not apply */
                                tok_engine, NULL, ctx_size, api_token, NULL);
            close(cfd);
        }
        ds4_engine_close(tok_engine);
        close(tk_lfd);
        return 0;
    }

#ifdef _WIN32
    /* Self-provision inbound firewall rules, exactly as the worker does.
     *
     * The worker has done this since the pairing bring-up; the COORDINATOR
     * never did, and that asymmetry is invisible until a Windows machine is
     * the coordinator: it listens fine, the weight server (a worker process)
     * is reachable because the worker opened its own port, and only the
     * coordinator's port is silently filtered. Every joiner then dies on
     * `connect:` -- which on Windows prints "No error", because errno is not
     * what winsock sets. Two machines, three ports, and the one that is closed
     * is the one nothing announces.
     *
     * Found 2026-08-12 on the real all-Windows pair: TCP 8001 (weights)
     * reachable, TCP 14310 (coordinator) refused.
     *
     * Both ports: the worker-facing one AND the API port, since the API is the
     * whole point of the product and is reached from other machines. */
    {
        const char *colon; char rule[64]; int port;
        colon = strrchr(bind, ':'); port = colon ? atoi(colon + 1) : 0;
        if (port > 0) {
            snprintf(rule, sizeof rule, "IdleToken coord TCP %d", port);
            idletoken_win_ensure_firewall_rule(rule, "TCP", port);
        }
        colon = strrchr(api_bind, ':'); port = colon ? atoi(colon + 1) : 0;
        if (http_serve && port > 0) {
            snprintf(rule, sizeof rule, "IdleToken API TCP %d", port);
            idletoken_win_ensure_firewall_rule(rule, "TCP", port);
        }
        if (pairing) {
            snprintf(rule, sizeof rule, "IdleToken discovery UDP %d", disc_port);
            idletoken_win_ensure_firewall_rule(rule, "UDP", disc_port);
        }
    }
#endif

    int lfd = idletoken_listen_tcp(bind);
    if (lfd < 0) {
        fprintf(stderr, "idletoken-coord: listen(%s): %s\n", bind, strerror(errno));
        return 1;
    }

    /* Start advertising this coordinator over the LAN so workers can join by
     * code/account with no manual --coordinator. The advertised address is
     * this host's LAN ip + the worker-facing TCP port. */
    idletoken_discovery *disc = NULL;
    if (pairing) {
        char lan_ip[64] = "127.0.0.1";
        idletoken_local_ipv4(lan_ip, sizeof(lan_ip));
        int coord_port = 14100;
        { const char *c = strrchr(bind, ':'); if (c) coord_port = atoi(c + 1); }
        char adv_addr[80];
        snprintf(adv_addr, sizeof(adv_addr), "%s:%d", lan_ip, coord_port);
        disc = idletoken_discovery_multi((uint16_t)disc_port,
                                      NULL /* coord doesn't need manual */);
        if (!disc || disc->advertise(disc, &pair_id, adv_addr) != 0) {
            fprintf(stderr, "idletoken-coord: failed to start LAN advertising\n");
            if (disc) disc->destroy(disc);
            close(lfd);
            return 1;
        }
        fprintf(stderr, "\ncoord: pairing active (%s mode) — advertising %s on udp/%d\n",
                pair_id.mode == IDLETOKEN_PAIR_MODE_ACCOUNT ? "account" : "code",
                adv_addr, disc_port);
        if (pair_code)
            fprintf(stderr, "\n  ================  JOIN CODE:  %s  ================\n\n",
                    pair_code);
    }

    idletoken_worker_info ws[IDLETOKEN_MAX_WORKERS];
    memset(ws, 0, sizeof(ws));
    int n = 0;

    /* Accept workers one by one (synchronous; workers serialize). */
    while (n < num_workers) {
        fprintf(stderr, "coord: waiting for worker %d/%d on %s\n", n + 1, num_workers, bind);
        int cfd = idletoken_accept_tcp(lfd);
        if (cfd < 0) { fprintf(stderr, "coord: accept: %s\n", strerror(errno)); close(lfd); return 1; }
        ws[n].fd = cfd;

        /* Pairing auth preamble: prove both sides know the code/account secret
         * before HELLO. A node with the wrong code is rejected here. */
        ws[n].has_session_key = 0;
        memset(ws[n].session_key, 0, sizeof(ws[n].session_key));
        memset(&ws[n].nc, 0, sizeof(ws[n].nc));
        if (pairing) {
            /* N0: keep the derived key on the worker record instead of dropping
             * it with the stack frame (docs/inter-node-encryption.md §3). */
            if (idletoken_pair_server_auth(cfd, &pair_id, ws[n].session_key) != 0) {
                fprintf(stderr, "coord: rejected an unauthenticated join attempt (%s)\n",
                        strerror(errno));
                memset(ws[n].session_key, 0, sizeof(ws[n].session_key));
                close(cfd); continue;
            }
            ws[n].has_session_key = 1;
            char fp[9]; session_key_fp(ws[n].session_key, fp);
            fprintf(stderr, "coord: worker %d passed pairing auth (session=%s)\n", n, fp);
        } else {
            /* No shared secret exists on this path at all — see the struct
             * comment; N2 makes this refuse platform traffic. */
            fprintf(stderr, "coord: worker %d joined without pairing (session=none)\n", n);
        }

        uint64_t rid = 0;
        if (do_hello(cfd, &ws[n], &rid) != 0)        { close(cfd); continue; }

        /* macOS compute nodes are sealed (2026-08-13). Checked BEFORE the
         * homogeneity rule below, and for every worker including the first: a
         * cluster of nothing but Macs is homogeneous and would sail through.
         * Rationale and the escape hatch live in idletoken_proto.h. */
        if (ws[n].os_family == IDLETOKEN_OS_MACOS && idletoken_macos_node_sealed()) {
            char why[256];
            snprintf(why, sizeof(why),
                     "macOS compute nodes are sealed in this build — the Mac line "
                     "is parked until it has a numerical baseline of its own. Use "
                     "this Mac as a control machine and run compute on Windows or "
                     "Linux nodes.");
            fprintf(stderr, "coord: refused %s (%s): %s\n",
                    ws[n].hostname, ws[n].bind_addr, why);
            send_hello_reject(cfd, rid, /*reasoncode=*/1, why);
            close(cfd); continue;
        }

        /* A cluster must be homogeneous: same OS family on every compute node
         * (CLAUDE.md hard constraint #2). The first worker to join sets the
         * family — deliberately NOT the coordinator's own OS, because the
         * coordinator may run on a control machine that computes nothing.
         *
         * This is refused rather than warned because a mixed cluster has no
         * oracle: the numeric gates compare token ids against a single-machine
         * ds4 baseline, and CUDA (--use_fast_math) vs Metal differ slightly per
         * layer, so greedy decoding eventually flips an argmax. See
         * docs/archive/macos-node.md §5. */
        if (n > 0 && ws[n].os_family != ws[0].os_family) {
            char why[256];
            snprintf(why, sizeof(why),
                     "cluster is %s; this node is %s. IdleToken does not support "
                     "mixed-OS clusters — run all compute nodes on one OS.",
                     idletoken_os_family_name(ws[0].os_family),
                     idletoken_os_family_name(ws[n].os_family));
            fprintf(stderr, "coord: refused %s (%s): %s\n",
                    ws[n].hostname, ws[n].bind_addr, why);
            send_hello_reject(cfd, rid, /*reasoncode=*/1, why);
            close(cfd); continue;
        }
        fprintf(stderr, "coord: worker %d is %s (%s)\n", n,
                idletoken_os_family_name(ws[n].os_family), ws[n].hostname);

        if (send_hello_ack(cfd, rid) != 0)           { close(cfd); continue; }
        if (recv_resource_report(cfd, &ws[n]) != 0)  { close(cfd); continue; }

        /* Zero-config addressing: a worker that binds a wildcard host
         * ("0.0.0.0"/""/"*") doesn't know its own LAN ip. Substitute the real
         * peer ip we see on this accepted connection so the inter-stage HC
         * address the coordinator hands to neighbors is actually dialable. */
        {
            char host[64] = "", pip[16] = "";   /* pip: INET_ADDRSTRLEN */
            const char *colon = strrchr(ws[n].bind_addr, ':');
            size_t hlen = colon ? (size_t)(colon - ws[n].bind_addr) : strlen(ws[n].bind_addr);
            if (hlen < sizeof(host)) { memcpy(host, ws[n].bind_addr, hlen); host[hlen] = '\0'; }
            int wildcard = (host[0] == '\0' || !strcmp(host, "0.0.0.0") || !strcmp(host, "*"));
            if (wildcard && colon && idletoken_peer_ip(cfd, pip, sizeof(pip)) == 0 && pip[0]) {
                int wport = atoi(colon + 1);
                char rewritten[24];
                snprintf(rewritten, sizeof(rewritten), "%s:%d", pip, wport);
                fprintf(stderr, "coord: worker %d HC addr %s -> %s (peer ip)\n",
                        n, ws[n].bind_addr, rewritten);
                snprintf(ws[n].bind_addr, sizeof(ws[n].bind_addr), "%s", rewritten);
            }
        }

        fprintf(stderr,
                "coord: worker %d: %-16s gpu=%s vram_usable=%.1fGiB ram_usable=%.1fGiB score=%llu\n",
                n,
                ws[n].hostname,
                ws[n].gpu_name,
                ws[n].vram_usable / (1024.0*1024.0*1024.0),
                ws[n].ram_usable  / (1024.0*1024.0*1024.0),
                (unsigned long long)ws[n].score);
        n++;
    }

    /* All workers joined — stop advertising (roster is frozen). */
    if (disc) { disc->destroy(disc); disc = NULL; }

    /* Rank by score, decide mode, split layers. */
    qsort(ws, (size_t)n, sizeof(ws[0]), cmp_worker_desc);

    idletoken_node_mem mode_nodes[IDLETOKEN_MAX_WORKERS];
    for (int i = 0; i < n; i++) {
        mode_nodes[i].vram_usable = ws[i].vram_usable;
        mode_nodes[i].ram_usable  = ws[i].ram_usable;
        mode_nodes[i].ram_pinnable= ws[i].ram_pinnable;
        mode_nodes[i].unified     = ws[i].unified;
    }
    char mode_why[256] = "";
    idletoken_mode mode = idletoken_mode_decide(coord_model(), mode_nodes, n, ctx_size,
                                          mode_why, sizeof(mode_why));
    fprintf(stderr, "coord: mode decision: %s\n", mode_why);
    if (mode == IDLETOKEN_MODE_REFUSE) {
        fprintf(stderr, "coord: refusing to start the cluster — %s\n", mode_why);
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 1;
    }
    if (mode == IDLETOKEN_MODE_HYBRID) {
        /* Layers are sized by VRAM+RAM below; each worker auto-offloads the
         * portion of its assigned layers that exceeds its VRAM to pinned host
         * RAM (cudaHostAlloc, PCIe-accessed). Overflow weights run slower. */
        fprintf(stderr, "coord: HYBRID — sizing layers by VRAM+RAM; workers "
                        "auto-offload VRAM overflow to host RAM\n");
    }

    plan_layers(ws, n, mode, ctx_size);

    fprintf(stderr, "\ncoord: planned topology (stage 0 = strongest)\n");
    for (int i = 0; i < n; i++) {
        fprintf(stderr,
                "  stage %d -> %s  layers [%u,%u)  (%u layers)  bind=%s\n",
                ws[i].stage_id, ws[i].hostname,
                ws[i].layer_lo, ws[i].layer_hi,
                (unsigned)(ws[i].layer_hi - ws[i].layer_lo),
                ws[i].bind_addr);
    }
    fprintf(stderr, "\n");

    /* Bind the token-field crypto now that every stage_id is final: the ids go
     * into every nonce, so this cannot be done at accept time when the roster is
     * still being ordered. Without a cluster key the state stays !ready and the
     * token fields go out in the clear (proto v7). */
    for (int i = 0; i < n; i++) {
        idletoken_nodecrypt_init(&ws[i].nc,
                                 g_cluster_key_ok ? g_cluster_key : NULL,
                                 IDLETOKEN_NC_ID_COORD, ws[i].stage_id);
    }

    /* Send ASSIGN_PLAN to each. */
    for (int i = 0; i < n; i++) {
        const idletoken_worker_info *prev = (i == 0)     ? NULL : &ws[i - 1];
        const idletoken_worker_info *next = (i == n - 1) ? NULL : &ws[i + 1];
        if (send_assign_plan(&ws[i], n, prev, next, ctx_size, model_path, bind,
                             (uint8_t)mode) != 0) {
            fprintf(stderr, "coord: failed to send ASSIGN_PLAN to stage %d\n", ws[i].stage_id);
        } else {
            fprintf(stderr, "coord: sent ASSIGN_PLAN to stage %d (%s)\n",
                    ws[i].stage_id, ws[i].hostname);
        }
    }

    /* Collect LOAD_MODEL_DONE from each worker (per wire-protocol.md). v0.1
     * each worker only does a GGUF preflight; real ds4_engine_open lands
     * with main #6. */
    int load_failures = 0;
    for (int i = 0; i < n; i++) {
        uint8_t dp[512];
        idletoken_msg_header dh;
        if (idletoken_recv_msg(ws[i].fd, &dh, dp, sizeof(dp)) != 0) {
            fprintf(stderr, "coord: recv LOAD_MODEL_DONE from stage %u: %s\n",
                    ws[i].stage_id, strerror(errno));
            load_failures++;
            continue;
        }
        if (dh.msg_type != IDLETOKEN_MSG_LOAD_MODEL_DONE) {
            fprintf(stderr, "coord: stage %u sent msg_type=0x%04x, expected LOAD_MODEL_DONE\n",
                    ws[i].stage_id, dh.msg_type);
            load_failures++;
            continue;
        }
        idletoken_buf db;
        idletoken_buf_init(&db, dp, dh.payload_bytes);
        uint8_t  ok = 0, pad7[7];
        uint64_t vram_used = 0, ram_used = 0;
        uint32_t raw_cap = 0, comp_cap = 0;
        char     err[256] = "";
        idletoken_buf_get_u8   (&db, &ok);
        idletoken_buf_get_bytes(&db, pad7, 7);
        idletoken_buf_get_u64  (&db, &vram_used);
        idletoken_buf_get_u64  (&db, &ram_used);
        idletoken_buf_get_u32  (&db, &raw_cap);
        idletoken_buf_get_u32  (&db, &comp_cap);
        ws[i].vram_used_after = vram_used;
        ws[i].ram_used_after  = ram_used;
        idletoken_buf_get_str  (&db, err, sizeof(err));
        if (db.err) {
            fprintf(stderr, "coord: stage %u LOAD_MODEL_DONE payload malformed\n",
                    ws[i].stage_id);
            load_failures++;
            continue;
        }
        fprintf(stderr,
                "coord: stage %u (%s) load %s  vram_used=%.2fGiB ram_used=%.2fGiB "
                "raw_cap=%u comp_cap=%u\n",
                ws[i].stage_id, ws[i].hostname,
                ok ? "OK" : "FAIL",
                vram_used / (1024.0*1024.0*1024.0),
                ram_used  / (1024.0*1024.0*1024.0),
                raw_cap, comp_cap);
        if (err[0]) {
            fprintf(stderr, "        message: %s\n", err);
        }
        if (!ok) load_failures++;
    }

    if (load_failures > 0) {
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        fprintf(stderr, "\ncoord: %d worker(s) failed to load model; aborting.\n",
                load_failures);
        return 1;
    }

    fprintf(stderr, "\ncoord: cluster ready (handshake + plan + load_done). "
                    "Warmup: %d decode step(s) (mock); HTTP: %s\n\n",
                    n_predict, http_serve ? api_bind : "off");

    /* Pre-allocate logits recv buffer once, sized from THIS model's vocab
     * (DSv4 129280 → ~505 KB; Qwen3 151936 → ~594 KB). Hard-coding the DSv4
     * vocab here made every larger-vocab model fail with "Message too long"
     * right after a successful load (real bug, 2026-07-27). */
    const size_t logits_payload = 8 + (size_t)coord_model()->n_vocab * sizeof(float);
    uint8_t *lbuf = malloc(logits_payload);
    if (!lbuf) {
        fprintf(stderr, "coord: malloc logits buf failed\n");
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 1;
    }

    int      infer_rc = 0;
    uint32_t infer_pos = 0;
    uint32_t next_token = 1;
    const uint64_t base_req_id = ((uint64_t)time(NULL) << 16) ^ (uint64_t)getpid();

    /* --- Warmup loop: drive --n-predict steps end-to-end ------------------ */
    for (int step = 0; step < n_predict; step++) {
        uint32_t out_tok = 0;
        if (coord_decode_step(ws, n, base_req_id ^ ((uint64_t)step << 4),
                              infer_pos, next_token, 0 /* warmup always uses slot 0 */,
                              lbuf, logits_payload, &out_tok) != 0) {
            infer_rc = 1; break;
        }
        fprintf(stderr, "coord: warmup step %d  pos=%u in_token=%u  -> sampled=%u\n",
                step, infer_pos, next_token, out_tok);
        infer_pos++;
        next_token = out_tok;
    }

    /* Warmup wrote slot 0's KV, so hand the cursor over to it -- otherwise the
     * first HTTP request would believe slot 0's cursor is at 0 while the worker's
     * sequence-0 session has already been written up to infer_pos. */
    g_slots[0].pos = infer_pos;
    /* D2: only after loading do we know how much memory is left, so the slot
     * count is computed here. An explicit user value is respected, but the
     * computed value is still reported (the client's settings panel shows
     * "auto: N (editable)"). */
    g_n_slots_auto = coord_auto_seq_slots(coord_model(), ws, n, ctx_size);
    if (g_n_slots == 0) {
        g_n_slots = g_n_slots_auto;
        fprintf(stderr, "coord: seq-slots auto -> %d (per-seq KV %llu B/layer x layers x %u ctx)\n",
                g_n_slots, (unsigned long long)coord_model()->kv_bytes_per_token_layer, ctx_size);
    } else if (g_n_slots != g_n_slots_auto) {
        fprintf(stderr, "coord: seq-slots %d (user override; auto would be %d)\n",
                g_n_slots, g_n_slots_auto);
    }
    if (g_n_slots > 1)
        fprintf(stderr, "coord: %d persistent sequence slots (proto v4 multi-sequence)\n", g_n_slots);

    /* The effective interleaving value: auto decides from the topology, and an
     * explicit value is still clamped by the slot count (two requests cannot
     * share a slot). */
    {
        const int c_auto = coord_auto_concurrent_decode(ws, n);
        if (g_concurrent_decode < 0) {
            g_concurrent_live = c_auto;
            fprintf(stderr, "coord: concurrent-decode auto -> %d (%d stage%s, %s)\n",
                    g_concurrent_live, n, n == 1 ? "" : "s",
                    c_auto > 0 ? "across machines -> there are pipeline bubbles to fill"
                               : (n < 2 ? "one stage -> no bubbles" : "several stages on one machine -> same GPU, interleaving is slower"));
        } else {
            g_concurrent_live = g_concurrent_decode;
            const int slots = g_n_slots > 0 ? g_n_slots : 1;
            if (g_concurrent_live > slots) {
                fprintf(stderr, "coord: --concurrent-decode %d exceeds the %d sequence slots; running %d\n",
                        g_concurrent_live, slots, slots);
                g_concurrent_live = slots;
            }
            if (g_concurrent_live != c_auto)
                fprintf(stderr, "coord: concurrent-decode %d (user-specified; auto would be %d)\n",
                        g_concurrent_live, c_auto);
        }
    }

    if (infer_rc != 0) {
        fprintf(stderr, "\ncoord: warmup INFER loop failed (rc=%d)\n", infer_rc);
        free(lbuf);
        for (int i = 0; i < n; i++) close(ws[i].fd);
        close(lfd);
        return 1;
    }

    /* --- HTTP serve loop (optional) --------------------------------------- */
    ds4_engine *coord_engine = NULL;
    ds4x_tokenizer *coord_xtok = NULL;   /* ds4x models: GGUF byte-BPE tokenizer */
    if (http_serve) {
        /* Open a CPU-mode ds4_engine on coord for tokenizer + chat template
         * + detokenize. Backend=CPU means no CUDA prefetch (coord is built
         * with -DDS4_NO_GPU anyway). The GGUF gets mmap'd — virtually 80 GB
         * but physically just the vocab pages we actually touch. */
        /* Same platform split as the worker (worker_main.c): there is no /tmp
         * on Windows, and ds4_acquire_instance_lock() exit(2)s the whole
         * process when the lock file cannot be created. The worker got this
         * right and the coordinator did not — which stayed invisible for as
         * long as every coordinator ran on Linux. On an all-Windows cluster it
         * kills the coord the instant it opens its tokenizer engine, i.e.
         * immediately AFTER "cluster ready" — both stages loaded, then the
         * whole cluster tore down with the workers reporting nothing worse
         * than "prev stage closed HC". */
        char ds4_lock_path[512];
#ifdef _WIN32
        const char *tmpdir = getenv("TEMP");
        if (!tmpdir || !tmpdir[0]) tmpdir = ".";
        snprintf(ds4_lock_path, sizeof(ds4_lock_path),
                 "%s\\ds4-%ld.lock", tmpdir, (long)getpid());
#else
        snprintf(ds4_lock_path, sizeof(ds4_lock_path),
                 "/tmp/ds4-%ld.lock", (long)getpid());
#endif
        setenv("DS4_LOCK_FILE", ds4_lock_path, 1);

        /* Pre-check the file: upstream ds4 exits the whole process on a
         * missing model (ds4_die_errno), which would take the coordinator —
         * and with it the cluster-management API — down with it. No model =
         * chat endpoints 503, management endpoints stay up. */
        struct stat mst;
        if (stat(coord_tok_path, &mst) != 0 || !S_ISREG(mst.st_mode)) {
            fprintf(stderr, "coord: model %s not readable; HTTP chat endpoints will be unavailable"
                            " (pass --gguf-dir if the model is not in the working directory)\n",
                    coord_tok_path);
        } else if (coord_use_ds4x) {
            /* ds4x models: no ds4_engine — just the GGUF byte-BPE tokenizer for
             * prompt encode + detokenize (embed/lm_head live on the workers). */
            char xerr[256] = "";
            coord_xtok = ds4x_tok_load(coord_tok_path, xerr, sizeof(xerr));
            if (!coord_xtok)
                fprintf(stderr, "coord: ds4x tokenizer load failed: %.180s; chat endpoints unavailable\n", xerr);
            else
                fprintf(stderr, "coord: ds4x tokenizer ready (eos_token=%d)\n", (int)ds4x_tok_eos(coord_xtok));
        } else {
            ds4_engine_options ceo = {
                .model_path = model_path,
                .backend = DS4_BACKEND_CPU,
                .n_threads = 0,
                .warm_weights = false,
                .quality = false,
                .load_layer_lo = 0,
                .load_layer_hi = 0,  /* full-load default; CPU backend skips prefetch anyway */
            };
            fprintf(stderr, "coord: opening CPU-mode ds4 engine for tokenizer at %s...\n",
                    model_path);
            if (ds4_engine_open(&coord_engine, &ceo) != 0 || !coord_engine) {
                fprintf(stderr, "coord: ds4_engine_open (CPU) failed; HTTP chat endpoints will be unavailable\n");
                coord_engine = NULL;
            } else {
                fprintf(stderr, "coord: ds4 tokenizer engine ready (eos_token=%d)\n",
                        ds4_token_eos(coord_engine));
            }
        }

        ignore_sigpipe();
        int api_lfd = idletoken_listen_tcp(api_bind);
        if (api_lfd < 0) {
            fprintf(stderr, "coord: http listen(%s): %s\n", api_bind, strerror(errno));
            free(lbuf);
            if (coord_engine) ds4_engine_close(coord_engine);
            if (coord_xtok) ds4x_tok_free(coord_xtok);
            for (int i = 0; i < n; i++) close(ws[i].fd);
            close(lfd);
            return 1;
        }
        /* E1: intake is separated from execution. The cap is min(queue capacity,
         * seq_slots*2) -- purely a memory backstop, since queueing too much on a
         * home machine means OOM; the latency-budget decision lives on the
         * platform side (§4.2b). */
        intake_init(g_n_slots * 2);
        g_intake_lfd = api_lfd;
        pthread_t acc_tid;
        int acc_ok = (pthread_create(&acc_tid, NULL, intake_accept_thread, NULL) == 0);
        if (!acc_ok) {
            fprintf(stderr, "coord: pthread_create(accept) failed: %s — falling back to serial accept\n",
                    strerror(errno));
        }
        fprintf(stderr, "\ncoord: HTTP API listening on %s (intake queue cap %d). Ctrl-C to stop.\n",
                api_bind, g_intake.cap);
        g_stats.started_at = (long long)time(NULL);
        /* The service-time EWMA (half-life of 8 requests) and cumulative
         * queueing time both feed /idletoken/v1/stats for the platform's cost function.
         * Updated only by the executor thread, so no lock is needed. */
        #define COORD_ACCOUNT(t0_ms, q_ms) do {                                   \
            double dt_ = (double)(now_ms() - (t0_ms));                             \
            g_stats.service_ms_ewma = g_stats.service_ms_ewma > 0                  \
                ? g_stats.service_ms_ewma * 0.875 + dt_ * 0.125 : dt_;             \
            g_stats.queued_total_ms += (uint64_t)((q_ms) > 0 ? (q_ms) : 0);        \
        } while (0)

        const coord_exec xctx = {
            ws, n, lbuf, logits_payload, &infer_pos, coord_engine, coord_xtok, ctx_size,
        };
        /* The concurrency cap is clamped by the slot count as well: two requests
         * cannot share one KV slot (see E3.3a). */
        int conc = g_concurrent_live;
        if (conc > 0 && !acc_ok) {
            fprintf(stderr, "coord: intake thread not running; interleaved execution falls back to serial\n");
            conc = 0;
        }
        coord_req *active[IDLETOKEN_MAX_SEQ_SLOTS];
        int n_active = 0;
        for (;;) {
            /* --- Intake ----------------------------------------------------
             * With nothing in flight we block (saving CPU); with requests in
             * flight we may only poll, or the in-flight ones would starve
             * whenever no new request arrives. */
            while (conc == 0 ? n_active == 0 : n_active < conc) {
                long long queued_ms = 0;
                int cfd;
                if (!acc_ok) {
                    cfd = idletoken_accept_tcp(api_lfd);   /* degraded path: as it was before E1 */
                    if (cfd < 0) {
                        if (errno == EINTR) continue;
                        fprintf(stderr, "coord: http accept: %s\n", strerror(errno));
                        goto serve_done;
                    }
                } else if (n_active == 0) {
                    cfd = intake_pop(&queued_ms);
                    if (cfd < 0) goto serve_done;       /* stop */
                } else {
                    cfd = intake_try_pop(&queued_ms);
                    if (cfd < 0) break;                 /* queue empty: go pump what is in flight */
                }
                long long t0 = now_ms();
                coord_req *parked = NULL;
                handle_http_request(cfd, ws, n, lbuf, logits_payload, &infer_pos,
                                    coord_engine, coord_xtok, ctx_size, api_token,
                                    conc > 0 ? &parked : NULL);
                if (parked) {
                    parked->admit_ms  = t0;
                    parked->queued_ms = queued_ms;
                    active[n_active++] = parked;
                } else {
                    close(cfd);                          /* non-decode paths have already replied */
                    COORD_ACCOUNT(t0, queued_ms);
                }
            }
            if (n_active == 0) continue;

            /* --- Event-driven pumping: advance whoever's logits arrive first --
             * This must **not** be written as round synchronization ("send them
             * all, then receive them all"). That way A's logits are back long
             * before B's, yet A cannot send its next round, and every round pays
             * an extra stage-1 drain. Measured, round synchronization reached only
             * 1.16x where the bound is 1.41x -- the difference is exactly that
             * barrier.
             * Instead: send one round for every idle request, then receive
             * **exactly one** set of logits, claim it by request_id, advance that
             * request, and it gets to send again on the very next pass. In steady
             * state stage 0 never idles. */
            for (int i = 0; i < n_active; i++) {
                if (active[i]->await_logits != 0) continue;   /* in flight, or awaiting cleanup */
                if (coord_req_step_send(active[i], &xctx) != 0)
                    active[i]->await_logits = -1;             /* -1 = time to clean up */
            }
            int n_await = 0;
            for (int i = 0; i < n_active; i++) if (active[i]->await_logits == 1) n_await++;
            if (n_await > 0) {
                uint32_t out_tok = 0;
                uint64_t got_req = 0;
                if (coord_round_recv(ws, n, lbuf, logits_payload, &out_tok, &got_req) != 0) {
                    /* With nothing received we cannot tell whose it was, so every
                     * in-flight request has to be abandoned. */
                    for (int i = 0; i < n_active; i++)
                        if (active[i]->await_logits == 1) {
                            active[i]->decode_failed = 1;
                            active[i]->await_logits = -1;
                        }
                } else {
                    int hit = -1;
                    for (int i = 0; i < n_active; i++)
                        if (active[i]->await_logits == 1 && active[i]->step_req_id == got_req) {
                            hit = i; break;
                        }
                    if (hit < 0) {
                        /* Unclaimed means we do not know whose logits these are.
                         * Guessing is out of the question: guess wrong and A's
                         * token is appended to B's sequence, producing
                         * **silently** wrong output. */
                        fprintf(stderr, "coord: LOGITS req_id %llu unclaimed; abandoning in-flight requests\n",
                                (unsigned long long)got_req);
                        for (int i = 0; i < n_active; i++)
                            if (active[i]->await_logits == 1) {
                                active[i]->decode_failed = 1;
                                active[i]->await_logits = -1;
                            }
                    } else {
                        coord_req_apply_logits(active[hit], out_tok);
                    }
                }
            }
            /* --- Clean up and compact in place (preserving relative order) --- */
            int w = 0;
            for (int i = 0; i < n_active; i++) {
                coord_req *rq = active[i];
                if (rq->await_logits == -1) {
                    coord_req_finish(rq, &xctx);
                    close(rq->conn_fd);
                    COORD_ACCOUNT(rq->admit_ms, rq->queued_ms);
                    free(rq);
                } else {
                    active[w++] = rq;
                }
            }
            n_active = w;
        }
serve_done:
        while (n_active > 0) {                 /* shutting down: drain what is in flight */
            coord_req *rq = active[--n_active];
            coord_req_finish(rq, &xctx);
            close(rq->conn_fd);
            free(rq);
        }
        #undef COORD_ACCOUNT
        if (acc_ok) {
            pthread_mutex_lock(&g_intake.mu);
            g_intake.stop = 1;
            pthread_cond_broadcast(&g_intake.cv);
            pthread_mutex_unlock(&g_intake.mu);
        }
        close(api_lfd);
    }

    free(lbuf);
    if (coord_engine) ds4_engine_close(coord_engine);
    if (coord_xtok) ds4x_tok_free(coord_xtok);
    for (int i = 0; i < n; i++) close(ws[i].fd);
    close(lfd);
    fprintf(stderr, "\ncoord: shutting down.\n");
    return 0;
}
