#!/usr/bin/env python3
"""TCP tap proxy: listen on --listen, forward to --target, and append every
byte of both directions to --out.

A userspace substitute for loopback packet capture where BPF privileges are not
available (e.g. an unprivileged macOS session where /dev/bpf* is root-only). The
RPC client (llama-server) is pointed at the tap's listen port instead of the
rpc-server directly, so the recorded stream is exactly the bytes that cross the
wire between the two RPC endpoints. Used by scripts/gpriv7_embedding_check.sh.
"""
import argparse
import socket
import threading


def pump(src: socket.socket, dst: socket.socket, out, lock) -> None:
    while True:
        try:
            data = src.recv(65536)
        except OSError:
            break
        if not data:
            break
        with lock:
            out.write(data)
            out.flush()
        try:
            dst.sendall(data)
        except OSError:
            break
    for s in (src, dst):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.listen))
    srv.listen(8)
    out = open(args.out, "ab")
    lock = threading.Lock()
    print("tap: 127.0.0.1:%d -> 127.0.0.1:%d -> %s" %
          (args.listen, args.target, args.out), flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        up = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        up.connect(("127.0.0.1", args.target))
        threading.Thread(target=pump, args=(conn, up, out, lock), daemon=True).start()
        threading.Thread(target=pump, args=(up, conn, out, lock), daemon=True).start()


if __name__ == "__main__":
    main()
