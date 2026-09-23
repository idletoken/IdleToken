#!/usr/bin/env python3
"""Portable wire-level regression: real TLS, proxies, IPv6 and interrupted HTTP.

The only certificates generated are temporary test CA/server fixtures. The
production client must reject them unless the test explicitly trusts that CA.
"""
import argparse
import contextlib
import http.server
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import ssl
import struct
import subprocess
import tempfile
import threading
import time


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    def handle_error(self, *args):
        pass  # connection resets are deliberately injected below


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path in ('/slow-upload', '/upload-stall'):
            # Pace the client's upload from this side: 32 MiB cannot sit in
            # loopback socket buffers, so draining 1 MiB per 100 ms makes the
            # upload take ~3 s of real progress. /upload-stall stops draining
            # after the first MiB so the upload makes no progress at all.
            length = int(self.headers.get('Content-Length', 0))
            got = 0
            while got < length:
                chunk = self.rfile.read(min(1 << 20, length - got))
                if not chunk:
                    return
                got += len(chunk)
                if self.path == '/upload-stall':
                    time.sleep(2)
                    self.close_connection = True
                    return
                time.sleep(0.1)
            data = b'{"ok":true}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data); self.wfile.flush()
            return
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.calls.append((self.path, body))
        if self.path == '/headers':
            time.sleep(3)
            return
        if self.path == '/redirect':
            self.send_response(307)
            self.send_header('Location', '/must-not-replay')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path in ('/receipt-longpoll', '/receipt-longpoll-stall', '/receipt-legacy-longpoll'):
            time.sleep(1.2)
        if self.path.startswith('/receipt'):
            value = {'job_id': 'clock-independent', 'remaining_ms': 550,
                     'expires_at_ms': 1 if self.path.endswith('past') else 9999999999999,
                     'sealed_request': 'x' * 10000}
            if self.path in ('/receipt-legacy', '/receipt-legacy-longpoll'):
                del value['remaining_ms']
                value['expires_at_ms'] = 1893456001549
            data = json.dumps(value).encode()
        elif self.path == '/large':
            data = b'x' * (3 * 1024 * 1024)
        else:
            data = b'{"ok":true}'
        if self.path in ('/receipt-legacy', '/receipt-legacy-longpoll'):
            self.send_response_only(200)
            self.send_header('Date', 'Tue, 01 Jan 2030 00:00:00 GMT')
        else:
            self.send_response(200)
        if self.path == '/chunked':
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            for chunk in [data[:4], data[4:]]:
                self.wfile.write(('%x\r\n' % len(chunk)).encode() + chunk + b'\r\n')
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            return
        self.send_header('Content-Length', str(len(data) if self.path != '/short' else 10000))
        self.end_headers()
        if self.path == '/stall':
            self.wfile.write(data[:1]); self.wfile.flush(); time.sleep(3)
        elif self.path.startswith('/receipt') and self.path != '/receipt-longpoll':
            self.wfile.write(data[:1024]); self.wfile.flush(); time.sleep(3)
        else:
            self.wfile.write(data); self.wfile.flush()
            if self.path == '/short':
                self.close_connection = True


class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    def handle_error(self, *args):
        pass


def relay(a, b):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        ready, _, _ = select.select([a, b], [], [], 0.2)
        for source in ready:
            data = source.recv(65536)
            if not data:
                return
            (b if source is a else a).sendall(data)


class HttpProxy(socketserver.StreamRequestHandler):
    calls = []
    target = None
    def handle(self):
        line = self.rfile.readline().decode().strip()
        self.calls.append(line)
        while self.rfile.readline().strip():
            pass
        if line.startswith('CONNECT '):
            with socket.create_connection(self.target) as upstream:
                self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n'); self.wfile.flush()
                relay(self.connection, upstream)
        else:
            self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')


class SocksProxy(socketserver.StreamRequestHandler):
    calls = []
    def handle(self):
        version, count = self.rfile.read(2)
        assert version == 5
        self.rfile.read(count)
        self.wfile.write(b'\x05\x00'); self.wfile.flush()
        version, command, _, kind = self.rfile.read(4)
        assert version == 5 and command == 1
        if kind == 3:
            host = self.rfile.read(self.rfile.read(1)[0]).decode()
        elif kind == 1:
            host = socket.inet_ntop(socket.AF_INET, self.rfile.read(4))
        else:
            host = socket.inet_ntop(socket.AF_INET6, self.rfile.read(16))
        port = struct.unpack('!H', self.rfile.read(2))[0]
        self.calls.append((host, port))
        self.wfile.write(b'\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00'); self.wfile.flush()
        line = self.rfile.readline()
        while self.rfile.readline().strip():
            pass
        self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')


@contextlib.contextmanager
def serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=1)


def local_ipv4_candidates():
    """Return assigned non-loopback addresses before route-selected tunnels.

    A transparent proxy can install a default route through a synthetic
    198.18/15 interface.  Binding that point-to-point address succeeds, but a
    connection to it is routed into the tunnel instead of back to the local
    test server.  Hostname resolution normally exposes the machine's assigned
    LAN addresses, so prefer those and keep the route lookup as a fallback for
    hosts whose name resolves only to loopback.
    """
    candidates = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET, socket.SOCK_STREAM):
            address = info[4][0]
            if not address.startswith('127.') and address != '0.0.0.0' and address not in candidates:
                candidates.append(address)
    except socket.gaierror:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(('192.0.2.1', 9))
            address = route.getsockname()[0]
        if not address.startswith('127.') and address != '0.0.0.0' and address not in candidates:
            candidates.append(address)
    except OSError:
        pass
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('probe')
    parser.add_argument('--cert-dir', type=Path, help='existing temporary TLS fixtures (for hosts without openssl)')
    args = parser.parse_args()
    probe = str(Path(args.probe).resolve())
    base_env = {k: v for k, v in os.environ.items() if k.lower() not in
                ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy', 'curl_ca_bundle')}
    base_env['NO_PROXY'] = '*'
    results = []

    def run(name, url, success=True, env=None, timeout=1200, receipt=0, cancel=0,
            upload=0, body_idle=None, body_total=None):
        argv = [probe, url, str(timeout), str(receipt), str(cancel)]
        if upload:
            # Sealed-result shape: `timeout` bounds only the wait for the first
            # response header; the upload itself is bounded by the body budgets.
            argv += [str(upload), str(timeout if body_idle is None else body_idle),
                     str(timeout if body_total is None else body_total)]
        process = subprocess.run(argv, env={**base_env, **(env or {})}, capture_output=True, timeout=8)
        assert process.returncode == 0, (name, process.stderr.decode())
        first, _, body = process.stdout.partition(b'\n')
        value = json.loads(first)
        assert (value['result'] == 0) == success, (name, value, process.stderr.decode())
        if not success:
            assert value['elapsed_ms'] < timeout + 650, (name, value)
        results.append({'case': name, **value})
        print(json.dumps(results[-1]), flush=True)
        return value, body

    with serve(Server(('127.0.0.1', 0), Handler)) as port:
        base = 'http://127.0.0.1:%d' % port
        run('direct-ipv4', base + '/ok')
        _, data = run('multi-megabyte-body', base + '/large', timeout=4000)
        assert data == b'x' * (3 * 1024 * 1024)
        _, data = run('chunked-framing', base + '/chunked')
        assert json.loads(data) == {'ok': True}
        run('truncated-response', base + '/short', False)
        run('body-stall', base + '/stall', False, timeout=500)
        run('headers-stall', base + '/headers', False, timeout=500)
        # A sealed reply's upload must be bounded by the body budgets, never by
        # the "start answering" timeout. The server paces 32 MiB at ~1 MiB per
        # 100 ms (~3 s of genuine progress) while the answer timeout is 400 ms
        # (a large body also gets the 3 s idle budget to drain out of socket
        # buffers before an answer is expected); the transport that derived a
        # whole-transfer clock from that timeout cut this at 0.4 s, retried
        # from byte 0 and never landed an image reply on a home uplink. The
        # positive control that follows stops draining after the first MiB:
        # no progress must be caught, and fast.
        r, data = run('slow-upload-outlives-answer-timeout', base + '/slow-upload', timeout=400,
                      upload=32 * 1024 * 1024, body_idle=3000, body_total=20000)
        assert r['elapsed_ms'] >= 2000 and json.loads(data) == {'ok': True}, r
        r, _ = run('stalled-upload-is-caught-by-idle-budget', base + '/upload-stall', False, timeout=400,
                   upload=32 * 1024 * 1024, body_idle=500, body_total=20000)
        assert r['truncated'] == 1 and r['elapsed_ms'] >= 500, r
        r, _ = run('caller-cancellation', base + '/headers', False, timeout=1500, cancel=200)
        assert r['cancelled'] == 1 and r['elapsed_ms'] < 700
        before = len(Handler.calls)
        r, _ = run('already-cancelled-sends-nothing', base + '/ok', False, cancel=-1)
        assert r['cancelled'] == 1 and len(Handler.calls) == before
        r, _ = run('redirect-does-not-replay-post', base + '/redirect')
        assert r['status'] == 307 and not any(p == '/must-not-replay' for p, _ in Handler.calls)
        for when in ('past', 'future'):
            r, _ = run('receipt-ignores-wall-clock-' + when, base + '/receipt-' + when,
                       False, timeout=1800, receipt=1)
            assert 400 <= r['elapsed_ms'] < 1100
        r, _ = run('legacy-receipt-uses-server-date', base + '/receipt-legacy', False, timeout=1800, receipt=1)
        assert 400 <= r['elapsed_ms'] < 1100
        r, _ = run('longpoll-does-not-expire-new-job', base + '/receipt-longpoll', timeout=2800, receipt=1)
        assert r['elapsed_ms'] >= 1200
        for suffix in ('longpoll-stall', 'legacy-longpoll'):
            r, _ = run('longpoll-bounds-body-' + suffix, base + '/receipt-' + suffix,
                       False, timeout=2800, receipt=1)
            assert 1600 <= r['elapsed_ms'] < 2400
        with serve(Proxy(('127.0.0.1', 0), HttpProxy)) as proxy_port:
            proxy = 'http://127.0.0.1:%d' % proxy_port
            env = {'NO_PROXY': '', 'HTTP_PROXY': proxy}
            run('http-proxy', 'http://unresolvable.invalid:8080/ok', env=env)
            assert HttpProxy.calls[-1].startswith('POST http://unresolvable.invalid:8080/ok ')
            before = len(HttpProxy.calls)
            run('loopback-never-proxied', base + '/ok', env=env)
            assert len(HttpProxy.calls) == before
            run('no-proxy-port-rule', 'http://unresolvable.invalid:8080/ok', False,
                env={**env, 'NO_PROXY': 'unresolvable.invalid:8080'})
            assert len(HttpProxy.calls) == before
        with serve(Proxy(('127.0.0.1', 0), SocksProxy)) as proxy_port:
            run('socks5-remote-dns', 'http://unresolvable.invalid:8080/ok',
                env={'NO_PROXY': '', 'ALL_PROXY': 'socks5h://127.0.0.1:%d' % proxy_port})
            assert SocksProxy.calls[-1] == ('unresolvable.invalid', 8080)
    # First prove a local non-loopback address is reachable directly; otherwise
    # a failed proxy test proves nothing. Try every assigned address because a
    # transparent proxy may make its point-to-point address look like the
    # default route while connections to that address never loop back locally.
    local_error = None
    for local_ip in local_ipv4_candidates():
        try:
            server = Server((local_ip, 0), Handler)
        except OSError as error:
            local_error = error
            continue
        with serve(server) as port, socket.socket() as refused:
            refused.bind(('127.0.0.1', 0))  # reserve a port without accepting connections
            url = 'http://%s:%d/ok' % (local_ip, port)
            try:
                run('direct-positive-control-for-proxy-failure', url)
            except AssertionError as error:
                local_error = error
                continue
            before = len(Handler.calls)
            run('dead-proxy-no-direct-fallback', url, False,
                env={'NO_PROXY': '', 'HTTP_PROXY': 'http://127.0.0.1:%d' % refused.getsockname()[1]})
            assert len(Handler.calls) == before, 'failed proxy leaked a direct request'
            break
    else:
        raise AssertionError(('no directly reachable non-loopback IPv4 address', local_error))
    class V6Server(Server):
        address_family = socket.AF_INET6
    with serve(V6Server(('::1', 0), Handler)) as port:
        run('direct-ipv6', 'http://[::1]:%d/ok' % port)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = args.cert_dir or Path(temporary)
        config = tmp / 'openssl.cnf'
        if not args.cert_dir:
            config.write_text('[req]\ndistinguished_name=dn\nx509_extensions=v3\nprompt=no\n'
                          '[dn]\nCN=localhost\n[v3]\nsubjectAltName=DNS:localhost,DNS:proxy-target.test,IP:127.0.0.1\n'
                          'basicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n')
        cert, key = tmp / 'cert.pem', tmp / 'key.pem'
        if not args.cert_dir:
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                        '-config', str(config), '-keyout', str(key), '-out', str(cert)],
                       check=True, capture_output=True)
        server = Server(('127.0.0.1', 0), Handler)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert, key)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        with serve(server) as port:
            url = 'https://localhost:%d/ok' % port
            run('untrusted-certificate-rejected', url, False)
            run('trusted-tls-and-dns-address-fallback', url, env={'CURL_CA_BUNDLE': str(cert)})
            HttpProxy.target = ('127.0.0.1', port)
            with serve(Proxy(('127.0.0.1', 0), HttpProxy)) as proxy_port:
                env = {'NO_PROXY': '', 'HTTPS_PROXY': 'http://127.0.0.1:%d' % proxy_port,
                       'CURL_CA_BUNDLE': str(cert)}
                run('https-connect-valid-tunnel', 'https://proxy-target.test/ok', env=env)
                assert HttpProxy.calls[-1].startswith('CONNECT proxy-target.test:443 ')
                # A non-loopback authority forces CONNECT. The certificate
                # must still reject its mismatching hostname through the tunnel.
                run('https-connect-preserves-hostname-verification', 'https://mismatch.invalid/ok', False, env=env)
                assert HttpProxy.calls[-1].startswith('CONNECT mismatch.invalid:443 ')
    print('PLATFORM_HTTP_OK %d' % len(results))


if __name__ == '__main__':
    main()
