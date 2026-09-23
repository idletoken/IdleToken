#!/usr/bin/env python3
"""Exercise OS PAC engines and Linux desktop refresh without changing user settings."""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


class Pac(http.server.BaseHTTPRequestHandler):
    delayed_requests = 0
    stall_ended = threading.Event()
    def do_GET(self):
        if self.path == '/recover.pac':
            type(self).delayed_requests += 1
            if type(self).delayed_requests == 1:
                time.sleep(6)
                type(self).stall_ended.set()
        data = {
            '/ok.pac': b'function FindProxyForURL(url, host) { return "PROXY 127.0.0.1:18741; DIRECT"; }',
            '/direct-first.pac': b'function FindProxyForURL(url, host) { return "DIRECT; PROXY 127.0.0.1:18741"; }',
            '/chain.pac': b'function FindProxyForURL(url, host) { return "PROXY 127.0.0.1:18741; PROXY 127.0.0.1:18742; DIRECT"; }',
            '/direct.pac': b'function FindProxyForURL(url, host) { return "DIRECT"; }',
            '/recover.pac': b'function FindProxyForURL(url, host) { return "DIRECT"; }',
        }.get(self.path, b'this is not valid JavaScript !!!')
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ns-proxy-autoconfig')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The stalled lookup is deliberately cancelled by the client.

    def log_message(self, *args):
        pass


def main():
    binary = str(Path(sys.argv[1]).resolve())
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    with tempfile.TemporaryDirectory() as tmp:
        if sys.platform.startswith('linux'):
            env.update(XDG_CONFIG_HOME=tmp, GSETTINGS_BACKEND='keyfile',
                       XDG_CURRENT_DESKTOP='GNOME', DESKTOP_SESSION='gnome')
            def settings(schema, key, value):
                subprocess.run(['gsettings', 'set', schema, key, value], env=env, check=True, timeout=3)
            settings('org.gnome.system.proxy', 'mode', 'manual')
            settings('org.gnome.system.proxy.http', 'host', '127.0.0.1')
            settings('org.gnome.system.proxy.http', 'port', '18741')
            process = subprocess.Popen([binary, 'system', 'http://example.test/'], env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                for index, port in enumerate((18741, 18742)):
                    if index:
                        settings('org.gnome.system.proxy.http', 'port', str(port))
                        process.stdin.write('\n'); process.stdin.flush()
                    code, count = map(int, process.stdout.readline().split())
                    routes = [process.stdout.readline().strip() for _ in range(count)]
                    print(json.dumps({'case': 'live-manual-proxy', 'port': port, 'code': code, 'routes': routes}), flush=True)
                    assert code == 0 and routes[0] == 'http://127.0.0.1:%d' % port
            finally:
                process.stdin.close(); process.wait(timeout=5)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Pac)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for case in ('ok', 'chain', 'direct-first', 'direct', 'bad', 'recover'):
                pac = 'http://127.0.0.1:%d/%s.pac' % (server.server_port, case)
                if sys.platform.startswith('linux'):
                    settings('org.gnome.system.proxy', 'mode', 'auto')
                    settings('org.gnome.system.proxy', 'autoconfig-url', pac)
                    pac = 'system'
                started = time.monotonic()
                line_times = []
                if case == 'recover':
                    process = subprocess.Popen([binary, pac, 'http://example.test/'], env=env, text=True,
                                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    lines = []
                    first_result = threading.Event()
                    def collect():
                        for line in process.stdout:
                            lines.append(line.rstrip('\n')); line_times.append(time.monotonic() - started)
                            first_result.set()
                    reader = threading.Thread(target=collect, daemon=True); reader.start()
                    try:
                        assert first_result.wait(6), 'the first lookup did not end'
                        assert Pac.stall_ended.wait(8), 'the injected network stall did not end'
                        retry_started = time.monotonic() - started
                        process.stdin.write('\n'); process.stdin.close()
                        code = process.wait(timeout=12)
                        reader.join(timeout=1)
                        stderr = process.stderr.read()
                    finally:
                        if process.poll() is None: process.kill(); process.wait()
                        process.stdout.close(); process.stderr.close()
                else:
                    result = subprocess.run([binary, pac, 'http://example.test/'], input='',
                                            env=env, text=True, capture_output=True, timeout=12)
                    lines, code, stderr = result.stdout.splitlines(), result.returncode, result.stderr
                elapsed = time.monotonic() - started
                print(json.dumps({'case': 'pac-' + case, 'stdout': lines, 'stderr': stderr,
                                  'elapsed_s': elapsed, 'line_times_s': line_times}), flush=True)
                assert code == 0 and lines
                if case == 'ok':
                    assert lines == ['0 2', 'http://127.0.0.1:18741', 'DIRECT'], 'PAC fallback order must survive resolution'
                elif case == 'direct-first':
                    # WinHTTP treats a leading DIRECT as terminal. Every OS
                    # must still preserve that decision instead of proxying.
                    assert lines[0].startswith('0 ') and lines[1] == 'DIRECT', 'an explicit first DIRECT must not be discarded'
                elif case == 'chain':
                    assert lines == ['0 3', 'http://127.0.0.1:18741', 'http://127.0.0.1:18742', 'DIRECT']
                elif case == 'recover':
                    # OS services may share/cache the pending PAC download.
                    # Bound each lookup, not the sum of two independent calls.
                    assert Pac.delayed_requests >= 1, 'the stalled request must reach the fixture'
                    assert lines == ['-1 0', '0 1', 'DIRECT'], 'the next lookup must recover'
                    assert line_times[0] < 5.5 and line_times[1] - retry_started < 5.5, 'each PAC lookup must be bounded'
                elif case == 'direct':
                    assert lines == ['0 1', 'DIRECT'], 'valid DIRECT must remain usable'
                else:
                    assert lines[0] == '-1 0', 'failed PAC must not become a direct connection'
        finally:
            server.shutdown(); server.server_close()
        # Fail-closed regression (Linux): a PAC that cannot be DOWNLOADED (server
        # unreachable) must block, not just a malformed one; and a benign
        # non-resolver warning (a garbage GSETTINGS_BACKEND, emitted under
        # GLib-GIO) must NOT block. Together these pin the inverse-allowlist
        # domain classifier against both a fail-open and a fail-closed
        # regression -- the 2026-09-20 audit's core finding.
        if sys.platform.startswith('linux'):
            settings('org.gnome.system.proxy', 'mode', 'auto')
            settings('org.gnome.system.proxy', 'autoconfig-url', 'http://127.0.0.1:1/dead.pac')
            r = subprocess.run([binary, 'system', 'http://example.test/'], input='',
                               env=env, text=True, capture_output=True, timeout=8)
            print(json.dumps({'case': 'pac-unreachable', 'stdout': r.stdout.splitlines()}), flush=True)
            assert r.stdout.splitlines()[:1] == ['-1 0'], 'an undownloadable PAC must fail closed, not go direct'
            benign = dict(env, GSETTINGS_BACKEND='nonsense')
            r = subprocess.run([binary, 'system', 'http://example.test/'], input='',
                               env=benign, text=True, capture_output=True, timeout=8)
            print(json.dumps({'case': 'benign-backend-warning', 'stdout': r.stdout.splitlines()}), flush=True)
            assert r.stdout.splitlines() and r.stdout.splitlines()[0].startswith('0 '), \
                'a benign GLib warning must not block resolution'
    print('PLATFORM_PROXY_OK')


if __name__ == '__main__':
    main()
