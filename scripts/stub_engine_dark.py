#!/usr/bin/env python3
"""A stand-in idletoken-server that goes dark exactly the way the real one did.

WHY (2026-08-17). The coordinator was found wedged on an engine that was still
a live process but had stopped answering, with the connection never closed —
see results/coord-wedge-20260817.md. Reproducing that with the real engine is
awkward: SIGSTOP freezes it, but a stopped child cannot act on SIGTERM either,
so the sidecar monitor wedges too and the test ends up measuring a DIFFERENT
hang. SIGKILL is no good either — it closes the socket, which is the one thing
the field failure never did.

So: an engine that behaves normally, then goes dark on command, and stays
killable throughout.

  GET  /health                  -> {"status":"ok"}  (until dark, then silence)
  POST /v1/chat/completions     -> a few SSE frames, then silence FOREVER with
                                   the socket held open — and the server goes
                                   dark, so nothing answers any more.
  POST /apply-template,/tokenize-> minimal plausible answers

--dark-mode one (default: all) darkens only the FIRST chat connection and keeps
serving everything else. That is the multi-slot question (2026-08-18): with the
coordinator serving N requests at once, does one dead relay fail ONLY itself, or
does it take the other slots with it? "all" cannot answer that — every stream
dies at the same instant either way, so a passing run would prove nothing.

The coordinator spawns this through --llama-server-bin, so it accepts (and
ignores) llama-server's arguments and only reads --port.
"""
import json
import socket
import sys
import threading
import time

port = 8080
dark_mode = "all"
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
    if a == "--dark-mode" and i + 1 < len(sys.argv):
        dark_mode = sys.argv[i + 1]

DARK = threading.Event()
HELD = []          # sockets deliberately kept open and silent
DARKENED = threading.Lock()
darkened_once = [False]


def send(sock, status, body, ctype="application/json"):
    b = body.encode()
    sock.sendall(
        f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
        f"Content-Length: {len(b)}\r\nConnection: close\r\n\r\n".encode() + b
    )


def handle(sock):
    sock.settimeout(30)
    try:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                return
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        line = head.split(b"\r\n")[0].decode()
        path = line.split(" ")[1] if " " in line else "/"

        # Once dark, every connection is accepted and then ignored — held open,
        # never answered, never closed. That is the state that used to park the
        # coordinator's only executor thread forever.
        if DARK.is_set():
            HELD.append(sock)
            return

        if path.startswith("/health"):
            send(sock, "200 OK", '{"status":"ok"}')
        elif path.startswith("/apply-template"):
            send(sock, "200 OK", json.dumps({"prompt": "hello"}))
        elif path.startswith("/tokenize"):
            send(sock, "200 OK", json.dumps({"tokens": list(range(27))}))
        elif path.startswith("/props"):
            send(sock, "200 OK", json.dumps({"default_generation_settings": {}}))
        elif "chat/completions" in path or "completion" in path:
            sock.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            for i in range(3):
                frame = json.dumps({
                    "choices": [{"index": 0, "delta": {"content": f"tok{i} "},
                                 "finish_reason": None}],
                })
                payload = f"data: {frame}\n\n".encode()
                sock.sendall(b"%x\r\n" % len(payload) + payload + b"\r\n")
                time.sleep(0.2)
            # Go dark mid-stream: no terminating chunk, no close, no further
            # bytes, ever. Hold a reference so Python cannot garbage-collect the
            # socket and close it behind our back — a closed socket would send a
            # FIN and let the coordinator off the hook, which is precisely the
            # case this stub exists NOT to test.
            # "one": darken this connection only, and go on serving the rest.
            # The socket is still held open and silent, so the coordinator's
            # relay for THIS request sees exactly the field failure.
            with DARKENED:
                first = not darkened_once[0]
                darkened_once[0] = True
            if dark_mode == "all":
                DARK.set()
            if dark_mode == "all" or first:
                HELD.append(sock)
                print("stub: went dark mid-stream", flush=True)
                return
            # Later connections in "one" mode finish normally.
            tail = b'data: [DONE]\n\n'
            sock.sendall(b"%x\r\n" % len(tail) + tail + b"\r\n0\r\n\r\n")
            return
        else:
            send(sock, "404 Not Found", "{}")
    except Exception:
        pass
    finally:
        if sock not in HELD:
            try:
                sock.close()
            except Exception:
                pass


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(64)
print(f"stub engine listening on 127.0.0.1:{port}", flush=True)
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
