#!/usr/bin/env python3
"""Mock worker: a minimal stub that speaks the IdleToken wire protocol, so the
coordinator can form a 1-stage cluster and serve HTTP on a development machine
without CUDA (a Mac, say) -- a local red/green loop for coordinator-side changes.
Inference semantics match the worker's MOCK mode: single-peak logits, so the
argmax is always the chosen token.

Usage:
  ./build/idletoken-coord --bind 127.0.0.1:14100 --num-workers 1 --n-predict 0 \
      --http --api-bind 127.0.0.1:18300 &
  python3 scripts/mock_worker.py 127.0.0.1:14100 [n_vocab]

n_vocab must equal the n_vocab in models/<id>.json for whichever model the
coordinator serves (the default is DSv4's 129280), or the coordinator declares
INFER_LOGITS malformed.

The authority for the protocol layout is the send/receive code in
src/coord/coord_main.c; the ASSIGN_PLAN section of docs/wire-protocol.md lags the
v2 code, so the code is the source of truth.
"""
import os
import platform   # os_family in HELLO; without it the mock NameErrors at connect
import socket
import struct
import sys
import uuid

MAGIC = 0x31494148
# Must match IDLETOKEN_PROTO_VERSION in include/idletoken_proto.h exactly -- the
# coordinator rejects any worker whose version differs at HELLO. Every protocol
# bump has to be mirrored here.
# v4 (multi-sequence, 2026-07-30): reserved byte [3] of INFER_BEGIN became seq_id.
# v6 (argmax at the last stage, 2026-08-11): INFER_LOGITS gained a short form
# (n_vocab == 0, then a u32 token id). This mock keeps sending the LONG form on
# purpose -- the coordinator must go on accepting it, both because that is what
# IDLETOKEN_FULL_LOGITS=1 produces and because non-greedy sampling will need it
# again. So this file is also the regression test for that acceptance path.
# v7 (cluster salt, 2026-08-13): ASSIGN_PLAN grew a trailing 16-byte salt for
# node-link token encryption. This mock only reads the first two payload bytes,
# so the field costs it nothing -- but the VERSION below is NOT optional: the
# coordinator rejects a mismatched worker at HELLO, and four gates went red the
# moment the engine moved to v7 while this file still said 6. The wire protocol
# has a fourth implementation and it lives here.
VERSION = 7
MSG_HELLO = 0x0001
MSG_HELLO_ACK = 0x0002
MSG_RESOURCE_REPORT = 0x0010
MSG_ASSIGN_PLAN = 0x0020
MSG_LOAD_MODEL_DONE = 0x0022
MSG_INFER_BEGIN = 0x0040
MSG_INFER_LOGITS = 0x0042
# 0x0043 = INFER_TOKEN_ACK, retired in v5 (it was a pure pipeline barrier). The
# number stays reserved and must not be reused.
MSG_HEARTBEAT = 0x0080
# DSv4's vocabulary size by default; override it with argv[2] when serving another
# model. The coordinator validates INFER_LOGITS against the n_vocab in **that
# model's registry entry** and declares a mismatch malformed -- which is the stub
# being wrong, not the coordinator.
N_VOCAB_DEFAULT = 129280


def header(msg_type: int, payload: bytes, request_id: int = 0, stage_id: int = 0) -> bytes:
    return struct.pack("<IHHQQII16x", MAGIC, VERSION, msg_type, len(payload), request_id, stage_id, 0)


def send_msg(sock: socket.socket, msg_type: int, payload: bytes, request_id: int = 0) -> None:
    sock.sendall(header(msg_type, payload, request_id) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket):
    h = recv_exact(sock, 48)
    magic, ver, msg_type, plen, req_id, stage, seg = struct.unpack("<IHHQQII16x", h)
    assert magic == MAGIC and ver == VERSION, f"bad frame magic={magic:#x} ver={ver}"
    payload = recv_exact(sock, plen) if plen else b""
    return msg_type, req_id, payload


def pstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def main() -> None:
    addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:14100"
    n_vocab = int(sys.argv[2]) if len(sys.argv) > 2 else N_VOCAB_DEFAULT
    # argv[3]: make the argmax land on a chosen token (0 by default). Testing
    # prefix reuse across requests requires a token whose **text round-trips
    # exactly**: when the client appends the answer back, it must re-tokenize to
    # the same ids, or the strict-extension match diverges on the assistant turn
    # (the coordinator is conservative about re-tokenization drift, so that is a
    # miss).
    argmax_token = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    host, port = addr.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # HELLO: uuid[16] + hostname + version + bind_addr + os_family + pad3
    # os_family must be this host's, not a hardcoded Linux: the coordinator
    # refuses a cluster that mixes OS families, and a mock is routinely run
    # alongside a real worker on the same machine.
    # IDLETOKEN_MOCK_OS_FAMILY forces a value: that is how G_HOMO stages a
    # mixed-OS join on one machine, where every real process reports the same OS.
    os_family = int(os.environ.get("IDLETOKEN_MOCK_OS_FAMILY") or
                    {"Linux": 1, "Windows": 2, "Darwin": 3}.get(platform.system(), 0))
    hello = uuid.uuid4().bytes + pstr("mock-worker-py") + pstr("idletoken-mock-worker py1") \
        + pstr("127.0.0.1:19999") + struct.pack("<B3x", os_family)
    send_msg(sock, MSG_HELLO, hello, request_id=1)
    t, _, ack = recv_msg(sock)
    assert t == MSG_HELLO_ACK, f"expected HELLO_ACK got {t:#x}"
    # Empty payload = accepted (all a coordinator sent before 2026-08-12).
    if ack and ack[0] == 0:
        # accepted(u8) reasoncode(u8) rsvd[2] proto(u16) rsvd(u16) hb(u32)
        # then str coord_version, str reject_message.
        off = 12
        (n,) = struct.unpack_from("<I", ack, off); off += 4 + n      # coord_version
        (n,) = struct.unpack_from("<I", ack, off); off += 4
        print("mock-worker: REFUSED: " + ack[off:off + n].decode(), flush=True)
        sys.exit(3)

    # RESOURCE_REPORT: report a fake machine large enough for the plan and mode
    # decision to pass (GPU_ONLY).
    gib = 1024 ** 3
    rr = pstr("Mock GPU 96GB") + struct.pack("<BBBB", 12, 1, 0, 0) \
        + struct.pack("<QQQ", 96 * gib, 2 * gib, 90 * gib) \
        + struct.pack("<QQQ", 128 * gib, 8 * gib, 100 * gib) \
        + struct.pack("<II", 20, 0) + struct.pack("<Q", 500 * gib) \
        + struct.pack("<II", 1000, 0) + struct.pack("<B7x", 1)
    send_msg(sock, MSG_RESOURCE_REPORT, rr)

    t, _, payload = recv_msg(sock)
    assert t == MSG_ASSIGN_PLAN, f"expected ASSIGN_PLAN got {t:#x}"
    cluster_size, stage_id = payload[0], payload[1]
    print(f"mock-worker: plan received (cluster_size={cluster_size} stage={stage_id})", flush=True)

    # LOAD_MODEL_DONE: ok=1 + pad7 + vram/ram used + raw/comp cap + err("")
    done = struct.pack("<B7xQQII", 1, 1 * gib, 1 * gib, 8192, 2048) + pstr("")
    send_msg(sock, MSG_LOAD_MODEL_DONE, done)
    print("mock-worker: ready (serving zero logits)", flush=True)

    # All zeros except argmax_token, which gets 1.0f so the coordinator's argmax
    # selects it.
    logits = bytearray(n_vocab * 4)
    struct.pack_into("<f", logits, argmax_token * 4, 1.0)
    logits = bytes(logits)
    while True:
        t, req_id, payload = recv_msg(sock)
        if t == MSG_INFER_BEGIN:
            _phase, first, _l, seq_id, pos0, n_tokens = struct.unpack("<BBBBII", payload[:12])
            if first:
                # is_first_chunk: the coordinator saw a KV miss and restarts from
                # zero (a real worker calls rewind(0) on **that sequence's**
                # session).
                print(f"mock-worker: first_chunk -> would rewind(0) (seq={seq_id} pos0={pos0})",
                      flush=True)
            else:
                print(f"mock-worker: continue seq={seq_id} pos0={pos0} n={n_tokens}", flush=True)
            pos = pos0 + n_tokens - 1
            send_msg(sock, MSG_INFER_LOGITS, struct.pack("<II", pos, n_vocab) + logits, req_id)
        elif t == MSG_HEARTBEAT:
            pass
        else:
            print(f"mock-worker: unexpected msg {t:#x}, ignoring", flush=True)


if __name__ == "__main__":
    main()
