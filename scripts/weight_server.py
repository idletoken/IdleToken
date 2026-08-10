#!/usr/bin/env python3
"""Range-capable static file server for the IdleToken weight repo.

Serves a directory (the master GGUF + its .idx manifest) over HTTP with
byte-range support, which the stdlib http.server lacks. This is the interim
repo host: run it on the coordinator/DGX now; the same layout (master.gguf +
master.gguf.idx) later moves to the NAS or HuggingFace unchanged — the worker
only needs a range-capable GET.

    python3 weight_server.py <dir> [--port 8001] [--bind 0.0.0.0]
"""
import argparse
import http.server
import os
import re


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "Not found")
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        ctype = self.guess_type(path)
        if not rng:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._stream(path, 0, size - 1)
            return
        m = re.match(r"bytes=(\d+)-(\d*)", rng.strip())
        if not m:
            self.send_error(416, "Bad Range")
            return
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._stream(path, start, end)

    def _stream(self, path, start, end):
        remaining = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--bind", default="0.0.0.0")
    a = ap.parse_args()
    os.chdir(a.dir)
    httpd = http.server.ThreadingHTTPServer((a.bind, a.port), RangeHandler)
    print(f"weight_server: serving {a.dir} on {a.bind}:{a.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
