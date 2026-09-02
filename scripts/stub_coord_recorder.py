#!/usr/bin/env python3
"""A coordinator-shaped recorder, for judging what the platform agent SENDS.

scripts/admission_origin_gate.sh needs to see the exact headers and the exact
bytes the real `idletoken-platform-agent` binary posts into the coordinator
after it opens a sealed job.  A real coordinator cannot be used for that on
every host -- it needs a GPU budget and a model -- and, more importantly, a real
coordinator would answer with its own verdict, which is the thing under test.
So this stands in for it: it records and it agrees.

What it records, one file each, per request:
    <rec>/headers.txt   the raw request headers, verbatim
    <rec>/body.bin      the raw request body, byte for byte

`body.bin` is written unmodified on purpose.  The gate hashes it with the
system `shasum`, an implementation this project did not write, and compares that
digest to the body-hash field inside the capability the agent minted.  If the
two agree, the binding is real; if the gate hashed with our own SHA-256 instead,
it would only be proving that our hasher agrees with itself.

Usage: stub_coord_recorder.py --port N --record DIR [--status 200]
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):    # keep the gate's output readable
        pass

    def _record(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with open(os.path.join(ARGS.record, "headers.txt"), "w") as f:
            for k, v in self.headers.items():
                f.write("%s: %s\n" % (k, v))
        with open(os.path.join(ARGS.record, "body.bin"), "wb") as f:
            f.write(body)
        return body

    def do_GET(self):
        # The agent probes /idletoken/v1/stats before listing. Answer something
        # well-formed so a failure there cannot be mistaken for the assertions
        # this fixture exists for.
        payload = json.dumps({
            "status": "ok", "ctx_size": 8192, "seq_slots": 1,
            "concurrency": 1, "queue_depth": 0, "queue_cap": 0,
            "avg_service_ms": 100, "avg_ttft_ms": 50,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self._record()
        payload = json.dumps({
            "id": "stub", "object": "chat.completion", "model": "stub",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "stub-coord-answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3,
                      "total_tokens": 6},
        }).encode()
        self.send_response(ARGS.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--status", type=int, default=200)
    ARGS = p.parse_args()
    os.makedirs(ARGS.record, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    sys.stderr.write("stub-coord: listening on 127.0.0.1:%d\n" % ARGS.port)
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
