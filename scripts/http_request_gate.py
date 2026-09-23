#!/usr/bin/env python3
"""Exercise production HTTP request framing across arbitrary TCP packet splits."""
import argparse
import json
import socket
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
args = parser.parse_args()
with socket.socket() as reserved:
    reserved.bind(('127.0.0.1', 0))
    port = reserved.getsockname()[1]
child = subprocess.Popen([args.binary, f'127.0.0.1:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert child.stdout.readline().strip() == b'READY'
failures = 0


def exchange(headers, body=b'', fragmented=False, expect_continue=False):
    with socket.create_connection(('127.0.0.1', port), timeout=3) as connection:
        connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n' + headers + b'\r\n')
        if expect_continue:
            interim = connection.recv(4096)
            assert interim == b'HTTP/1.1 100 Continue\r\n\r\n', repr(interim)
        try:
            if fragmented:
                for offset in range(0, len(body), 17):
                    connection.sendall(body[offset:offset + 17])
                    time.sleep(0.001)
            else:
                connection.sendall(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        response = b''
        while True:
            try:
                part = connection.recv(65536)
            except ConnectionResetError:
                break
            if not part:
                break
            response += part
        head, payload = response.split(b'\r\n\r\n', 1)
        return int(head.split(b' ')[1]), payload


def fingerprint(body):
    hash_ = 2166136261
    for byte in body:
        hash_ = ((hash_ ^ byte) * 16777619) & 0xffffffff
    return f'{hash_:08x}'


def check(name, fn):
    global failures
    try:
        fn()
        print(json.dumps({'case': name, 'pass': True}), flush=True)
    except Exception as error:
        failures += 1
        print(json.dumps({'case': name, 'pass': False, 'error': str(error)}), flush=True)


def accepted(headers, wire, actual=None, origin=False, **options):
    actual = wire if actual is None else actual
    status, payload = exchange(headers, wire, **options)
    assert status == 200, (status, payload[:200])
    result = json.loads(payload)
    assert result == {'bytes': len(actual), 'hash': fingerprint(actual), 'origin_present': origin}, result


def rejected(headers, body=b'', status=400):
    observed, payload = exchange(headers, body)
    assert observed == status, (observed, payload[:200])


try:
    body = b'{"messages":[{"role":"user","content":"hello"}]}'
    length = b'Content-Length: ' + str(len(body)).encode() + b'\r\n'
    check('fixed-length', lambda: accepted(length, body))
    check('fragmented-body', lambda: accepted(length, body, fragmented=True))
    check('header-value-is-not-content-length', lambda: accepted(b'X-Note: content-length: 0\r\n' + length, body))
    check('origin-after-5k-header', lambda: accepted(b'X-Padding: ' + b'a' * 5000 + b'\r\nOrigin: https://untrusted.test\r\n' + length, body, origin=True))
    large = b'abcdef 123456789\n' * (5 * 1024 * 1024 // 16)
    check('five-mib-context-body', lambda: accepted(b'Content-Length: ' + str(len(large)).encode() + b'\r\n', large))
    wire = b'%x;note=yes\r\n' % 13 + body[:13] + b'\r\n%x\r\n' % (len(body)-13) + body[13:] + b'\r\n0\r\nX-Digest: checked\r\n\r\n'
    check('chunked-with-trailer', lambda: accepted(b'Transfer-Encoding: chunked\r\n', wire, actual=body, fragmented=True))
    check('expect-continue', lambda: accepted(b'Expect: 100-continue\r\n' + length, body, expect_continue=True))
    check('negative-length', lambda: rejected(b'Content-Length: -1\r\n', body))
    check('length-trailing-junk', lambda: rejected(b'Content-Length: 1x\r\n', body))
    check('conflicting-lengths', lambda: rejected(b'Content-Length: 1\r\n' + length, body))
    check('length-and-transfer-encoding', lambda: rejected(b'Transfer-Encoding: chunked\r\n' + length, body))
    check('oversize-before-upload', lambda: rejected(b'Content-Length: 33554433\r\n', status=413))
    check('malformed-chunk', lambda: rejected(b'Transfer-Encoding: chunked\r\n', b'zz\r\ninvalid\r\n0\r\n\r\n'))
    check('forbidden-origin-trailer', lambda: rejected(b'Transfer-Encoding: chunked\r\n', b'0\r\nOrigin: https://untrusted.test\r\n\r\n'))
    check('forbidden-admission-trailer', lambda: rejected(b'Transfer-Encoding: chunked\r\n', b'0\r\nX-IdleToken-Origin: platform\r\n\r\n'))
    check('oversize-chunk-before-upload', lambda: rejected(b'Transfer-Encoding: chunked\r\n', b'2000001\r\n', status=413))
    check('missing-chunk-ending', lambda: rejected(b'Transfer-Encoding: chunked\r\n', b'3\r\nabcXX0\r\n\r\n'))
    check('truncated-fixed-body', lambda: rejected(b'Content-Length: 100\r\n', body))
    check('valid-request-after-failures', lambda: accepted(length, body))
finally:
    child.terminate()
    try:
        child.wait(3)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait()
raise SystemExit(1 if failures else 0)
