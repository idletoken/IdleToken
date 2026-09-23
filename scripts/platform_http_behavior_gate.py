#!/usr/bin/env python3
"""Real helper traffic through an authenticated proxy, resets and concurrent UI calls."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time


class Proxy(ThreadingHTTPServer):
    daemon_threads = True
    # Sixteen authenticated requests can open thirty-two connections. The
    # Python default backlog of five rejects SYNs on Windows before our
    # application can inspect them; that is fixture capacity, not this oracle.
    request_queue_size = 128
    def handle_error(self, *args):
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    executions = []
    challenges = 0
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        expected = 'Basic ' + base64.b64encode(b'fixture:p@ss:word').decode()
        if self.headers.get('Proxy-Authorization') != expected:
            with self.lock:
                type(self).challenges += 1
            self.send_response(407)
            self.send_header('Proxy-Authenticate', 'Basic realm="test proxy"')
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            return
        left = int(self.headers.get('Content-Length', 0))
        digest = hashlib.sha256()
        while left:
            chunk = self.rfile.read(min(left, 32768))
            if not chunk:
                return
            left -= len(chunk)
            digest.update(chunk)
            if self.path.endswith('/slow-upload'):
                time.sleep(0.006)
        with self.lock:
            self.executions.append(self.path)
        if self.path.endswith('/ambiguous-post'):
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.close_connection = True
            return
        if self.path.endswith('/silent-generation'):
            time.sleep(1.2)
        body = json.dumps({'path': self.path, 'sha256': digest.hexdigest()}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)


def main():
    binary = str(Path(sys.argv[1]).resolve())
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    server = Proxy(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env.update(NO_PROXY='', HTTP_PROXY=f'http://fixture:p%40ss%3Aword@127.0.0.1:{server.server_port}')

    def call(path, body='{}', timeout=4000, overrides=None):
        request = json.dumps({'url': 'http://behavior.invalid/' + path, 'method': 'POST',
                              'body': body, 'timeout_ms': timeout}).encode()
        start = time.monotonic()
        p = subprocess.run([binary, '--platform-http-stdio'], input=struct.pack('!I', len(request)) + request,
                           env={**env, **(overrides or {})}, capture_output=True, timeout=8)
        header, _, data = p.stdout.partition(b'\n')
        return p.returncode, json.loads(header), data, int((time.monotonic() - start) * 1000)

    def emit(name, elapsed, **values):
        print(json.dumps({'case': name, 'elapsed_ms': elapsed, **values}), flush=True)

    try:
        rc, head, data, elapsed = call('authenticated')
        assert rc == 0 and head['status'] == 200 and Handler.challenges > 0
        assert json.loads(data)['path'].endswith('/authenticated')
        emit('proxy-authentication-with-escaped-password', elapsed, challenges=Handler.challenges)

        count = len(Handler.executions)
        rc, head, _, elapsed = call('denied', overrides={'HTTP_PROXY': f'http://fixture:wrong@127.0.0.1:{server.server_port}'})
        assert rc == 0 and head['status'] == 407 and len(Handler.executions) == count
        emit('wrong-proxy-password-never-reaches-service', elapsed, status=407)

        body = ('x' * (3 * 1024 * 1024)) + '\nUnicode: \u4e2d\u6587 \\ "'
        rc, head, data, elapsed = call('slow-upload', body)
        assert rc == 0 and head['status'] == 200
        assert json.loads(data)['sha256'] == hashlib.sha256(body.encode()).hexdigest()
        emit('three-MiB-slow-upload-through-authenticated-proxy', elapsed, bytes=len(body.encode()))

        count = len(Handler.executions)
        rc, head, _, elapsed = call('ambiguous-post')
        assert rc != 0 and 'error' in head and len(Handler.executions) == count + 1
        emit('disconnect-after-POST-is-not-automatically-replayed', elapsed, executions=1)
        rc, head, _, elapsed = call('after-reset')
        assert rc == 0 and head['status'] == 200
        emit('same-proxy-recovers-after-reset', elapsed)

        rc, head, _, elapsed = call('silent-generation', timeout=200)
        assert rc != 0 and 'error' in head and elapsed < 1000
        emit('deadline-during-silent-generation', elapsed)

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=16) as pool:
            values = list(pool.map(lambda n: call('concurrent-' + str(n)), range(16)))
        for n, (rc, head, data, elapsed) in enumerate(values):
            assert rc == 0 and head.get('status') == 200, (n, rc, head, elapsed)
            assert json.loads(data)['path'].endswith('/concurrent-' + str(n))
        emit('sixteen-account-control-requests-through-one-proxy', int((time.monotonic() - start) * 1000), succeeded=16)
        print('PLATFORM_HTTP_BEHAVIOR_OK')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
