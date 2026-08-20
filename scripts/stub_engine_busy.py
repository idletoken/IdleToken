#!/usr/bin/env python3
"""A stand-in idletoken-server that works, slowly.

Sibling of stub_engine_dark.py, for the opposite question. That one asks what a
DEAD engine does to the coordinator; this one asks what a BUSY one does — it
answers everything correctly, but a chat takes --hold-s seconds, so the
coordinator's sequence slots really do fill up and the admission gate really
does refuse. That refusal is where overflow routing decides whether to borrow
another machine (G_OVERFLOW), and it cannot be reached on an idle engine.

Deliberately not the real engine: the gate is about the coordinator's decision,
and a real model would make the fixture depend on the machine's GPU, its memory
and how long a 0.8B takes today.

  GET  /health              -> {"status":"ok"}
  POST /apply-template      -> a plausible prompt
  POST /tokenize            -> a fixed token list
  POST /v1/chat/completions -> after --hold-s seconds, one non-streaming
                               completion in llama-server's shape

The coordinator spawns this through --llama-server-bin, so it accepts (and
ignores) llama-server's arguments and reads only --port.
"""
import json
import socket
import sys
import threading
import time

port = 8080
hold_s = 8.0
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
    if a == "--hold-s" and i + 1 < len(sys.argv):
        hold_s = float(sys.argv[i + 1])


def send(sock, status, body, ctype="application/json"):
    b = body.encode()
    sock.sendall(
        f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
        f"Content-Length: {len(b)}\r\nConnection: close\r\n\r\n".encode() + b
    )


def handle(sock):
    sock.settimeout(120)
    try:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                return
            data += chunk
        head, _, _rest = data.partition(b"\r\n\r\n")
        line = head.split(b"\r\n")[0].decode()
        path = line.split(" ")[1] if " " in line else "/"

        if path.startswith("/health"):
            send(sock, "200 OK", '{"status":"ok"}')
        elif path.startswith("/apply-template"):
            send(sock, "200 OK", json.dumps({"prompt": "hello"}))
        elif path.startswith("/tokenize"):
            send(sock, "200 OK", json.dumps({"tokens": list(range(27))}))
        elif path.startswith("/props"):
            send(sock, "200 OK", json.dumps({"default_generation_settings": {}}))
        elif "chat/completions" in path or "completion" in path:
            # The whole point: hold the slot. /health keeps answering on other
            # connections while this one is parked, which is what lets the gate
            # tell "the machine is full" from "the machine is wedged".
            time.sleep(hold_s)
            send(sock, "200 OK", json.dumps({
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "local-answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 27, "completion_tokens": 3},
            }))
        else:
            send(sock, "404 Not Found", "{}")
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(64)
print(f"stub engine (busy, hold={hold_s}s) listening on 127.0.0.1:{port}", flush=True)
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
