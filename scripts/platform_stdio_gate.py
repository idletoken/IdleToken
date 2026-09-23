#!/usr/bin/env python3
"""Real desktop-helper framing, credentials, HTTP status and proxy integration."""
import http.server
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from platform_http_gate import Server, Proxy, HttpProxy, serve


class Control(http.server.BaseHTTPRequestHandler):
    seen = []
    reads = {}
    def log_message(self, *args):
        pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.seen.append((dict(self.headers), body))
        if self.path == '/reset-post':
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if self.path == '/wait':
            time.sleep(2)
        self.send_response(401 if self.path == '/login' else 200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reads[self.path] = self.reads.get(self.path, 0) + 1
        count = self.reads[self.path]
        if self.path == '/reset-always' or (self.path == '/reset-once' and count == 1):
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if self.path == '/shared-deadline':
            time.sleep(0.30 if count == 1 else 1)
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.send_response(503 if self.path == '/busy' else 200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'OK')


def main():
    binary = str(Path(sys.argv[1]).resolve())
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    version = json.loads((Path(__file__).resolve().parents[1] / 'client/package.json').read_text())['version']
    def call(url, extra=None, timeout=2000, request_body=None, method='POST'):
        value = {'method':method, 'url':url, 'bearer':'stdin-only-test-token',
                 'body':json.dumps({'text':'line 1\nline 2', 'unicode':'\u4e2d\u6587'}, ensure_ascii=False), 'timeout_ms':timeout}
        if request_body is not None: value['body'] = request_body
        if method == 'GET': value.pop('body')
        payload = json.dumps(value).encode()
        before = time.monotonic()
        result = subprocess.run([binary, '--platform-http-stdio'], input=struct.pack('!I', len(payload)) + payload,
                                env={**env, **(extra or {})}, capture_output=True, timeout=5)
        header, _, body = result.stdout.partition(b'\n')
        meta = json.loads(header)
        assert b'stdin-only-test-token' not in result.stdout + result.stderr
        return meta, body, value.get('body', '').encode(), time.monotonic() - before
    with serve(Server(('127.0.0.1', 0), Control)) as port:
        base = 'http://127.0.0.1:%d' % port
        meta, body, wanted, _ = call(base + '/login')
        assert meta['status'] == 401 and body == wanted
        headers, incoming = Control.seen[-1]
        assert incoming == wanted
        assert headers['Authorization'] == 'Bearer stdin-only-test-token'
        assert headers['X-IdleToken-Version'] == version
        print('STDIO_OK framing, unicode, body, version, bearer and HTTP 401 preserved')
        meta, body, wanted, _ = call(base + '/large', timeout=4000, request_body=json.dumps({'content':'x' * (10 * 1024 * 1024)}))
        assert meta['status'] == 200 and body == wanted
        print('STDIO_OK 10 MiB body is preserved without truncation')
        meta, _, _, elapsed = call(base + '/wait', timeout=200)
        assert 'error' in meta and elapsed < 1
        print('STDIO_OK caller timeout bounds response-header wait')
        meta, body, _, _ = call(base + '/reset-once', method='GET')
        assert meta.get('status') == 200 and body == b'OK' and Control.reads['/reset-once'] == 2
        print('STDIO_OK transient GET reset recovers with one retry')
        meta, _, _, _ = call(base + '/reset-always', method='GET')
        assert 'error' in meta and Control.reads['/reset-always'] == 2
        meta, _, _, _ = call(base + '/busy', method='GET')
        assert meta.get('status') == 503 and Control.reads['/busy'] == 1
        before = len(Control.seen)
        meta, _, _, _ = call(base + '/reset-post')
        assert 'error' in meta and len(Control.seen) == before + 1
        print('STDIO_OK retry is bounded; HTTP errors and ambiguous POST are not replayed')
        meta, _, _, elapsed = call(base + '/shared-deadline', method='GET', timeout=500)
        assert 'error' in meta and Control.reads['/shared-deadline'] == 2 and elapsed < 0.95
        print('STDIO_OK both GET attempts share the original total deadline')
        with serve(Proxy(('127.0.0.1', 0), HttpProxy)) as proxy:
            proxy_env = {'HTTP_PROXY':'http://127.0.0.1:%d' % proxy}
            meta, _, _, _ = call('http://unresolvable.invalid/control', proxy_env)
            assert meta['status'] == 200 and HttpProxy.calls[-1].startswith('POST http://unresolvable.invalid/control ')
            calls = len(HttpProxy.calls)
            meta, _, _, _ = call(base + '/ok', proxy_env)
            assert meta['status'] == 200 and len(HttpProxy.calls) == calls
            print('STDIO_OK explicit proxy and mandatory loopback bypass')
        if sys.platform.startswith('linux') and os.environ.get('IDLETOKEN_TEST_PAC_HOST'):
            # Resolve the machine's own hostname to the existing local server,
            # while exercising PAC (numeric loopback URLs bypass it by design).
            # The caller supplies a verified local DNS alias; changing hosts
            # files or the desktop's proxy settings is outside this test.
            host = os.environ['IDLETOKEN_TEST_PAC_HOST']
            assert host.lower() != 'localhost' and socket.gethostbyname(host) == '127.0.0.1', 'PAC fixture needs a local hostname resolving to 127.0.0.1'
            class Pac(http.server.BaseHTTPRequestHandler):
                script = b''
                def log_message(self, *args):
                    pass
                def do_GET(self):
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(self.script)))
                    self.end_headers(); self.wfile.write(self.script)
            with tempfile.TemporaryDirectory(prefix='idletoken-stdio-pac-') as settings_dir, \
                    serve(Server(('127.0.0.1', 0), Pac)) as pac_port, socket.socket() as dead:
                dead.bind(('127.0.0.1', 0))
                settings_env = {**env, 'XDG_CONFIG_HOME': settings_dir, 'GSETTINGS_BACKEND': 'keyfile',
                                'XDG_CURRENT_DESKTOP': 'GNOME', 'DESKTOP_SESSION': 'gnome', 'NO_PROXY': ''}
                def setting(key, value):
                    subprocess.run(['gsettings', 'set', 'org.gnome.system.proxy', key, value],
                                   env=settings_env, check=True, timeout=3)
                setting('mode', 'auto')
                proxy = 'PROXY 127.0.0.1:%d' % dead.getsockname()[1]
                for index, (route, success) in enumerate(((proxy + '; DIRECT', True),
                                                          ('DIRECT; ' + proxy, True), (proxy, False), (None, False))):
                    Pac.script = ('function FindProxyForURL(url, host) { return "%s"; }' % route).encode() if route else b'invalid JavaScript !!!'
                    setting('autoconfig-url', 'http://127.0.0.1:%d/%d.pac' % (pac_port, index))
                    seen = len(Control.seen)
                    meta, body, wanted, _ = call('http://%s:%d/pac-check' % (host, port), settings_env, timeout=4000)
                    if success:
                        assert meta.get('status') == 200 and body == wanted and len(Control.seen) == seen + 1
                    else:
                        assert 'error' in meta and len(Control.seen) == seen
                    print('STDIO_PAC_OK case=%d allowed=%s executions=%d' % (index, success, len(Control.seen) - seen))
    print('PLATFORM_STDIO_OK')


if __name__ == '__main__':
    main()
