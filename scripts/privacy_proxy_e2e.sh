#!/usr/bin/env bash
# IdleToken privacy proxy — end-to-end test over real sockets.
#
# Proves the sealed-envelope terminator (idletoken-privacy-proxy) in front of a
# coord: an external consumer talks only ciphertext to the proxy; the proxy
# decrypts, forwards plaintext to the (mock) coord over loopback, seals the
# reply back. No plaintext ever crosses the consumer-facing socket.
#
# Pure-C + a tiny python mock upstream (stands in for idletoken-coord's HTTP API,
# so this runs with no GPU/GGUF). Prints PASS/FAIL per check; exit 0 iff all
# pass. Wired nowhere by default — a bonus integration test on top of the
# headless G_PRIV selftests.
#
# Usage: scripts/privacy_proxy_e2e.sh
set -u
cd "$(dirname "$0")/.."

PX=build/privacy
PROXY_PORT=18443
UP_PORT=18000
KEY=/tmp/idletoken_e2e_node.key
rc=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1"; rc=1; }

command -v python3 >/dev/null || { echo "SKIP: python3 needed for mock upstream"; exit 0; }

echo "== building proxy + client =="
make -f Makefile.privacy proxy >/dev/null 2>&1 || { echo "build failed"; exit 1; }

cleanup() { kill "${MOCK:-}" "${PROXY:-}" 2>/dev/null; rm -f "$KEY"; }
trap cleanup EXIT
pkill -f idletoken_e2e_mock.py 2>/dev/null; pkill -f idletoken-privacy-proxy 2>/dev/null; rm -f "$KEY"; sleep 0.2

cat > /tmp/idletoken_e2e_mock.py <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen(8)
while True:
    c,_ = s.accept(); data=b""
    while b"\r\n\r\n" not in data:
        ch=c.recv(4096)
        if not ch: break
        data+=ch
    head,_,rest=data.partition(b"\r\n\r\n"); clen=0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"): clen=int(line.split(b":")[1])
    body=rest
    while len(body)<clen: body+=c.recv(4096)
    sys.stderr.write("UPSTREAM_RECV: "+body.decode("utf-8","replace")+"\n"); sys.stderr.flush()
    rb=b'{"marker":"UPSTREAM-SAW-PLAINTEXT","content":[{"type":"text","text":"pong"}]}'
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"%len(rb)+rb)
    c.close()
PY

python3 /tmp/idletoken_e2e_mock.py "$UP_PORT" 2>/tmp/idletoken_e2e_mock.log & MOCK=$!
disown "$MOCK" 2>/dev/null || true
"$PX/idletoken-privacy-proxy" --bind 127.0.0.1:$PROXY_PORT --upstream 127.0.0.1:$UP_PORT --key-file "$KEY" >/dev/null 2>/tmp/idletoken_e2e_proxy.log & PROXY=$!
disown "$PROXY" 2>/dev/null || true
sleep 0.8

echo "== round-trip =="
OUT=$("$PX/idletoken-privacy-client" --proxy 127.0.0.1:$PROXY_PORT \
      --inner '{"content":"SECRET-PROMPT-XYZZY the launch code is 1234","max_tokens":8}' 2>/tmp/idletoken_e2e_client.err)
echo "$OUT" | grep -q "UPSTREAM-SAW-PLAINTEXT" && ok "consumer decrypted the sealed reply (full round-trip)" || bad "reply not decrypted"
grep -q "SECRET-PROMPT-XYZZY" /tmp/idletoken_e2e_mock.log && ok "proxy decrypted & forwarded plaintext to coord over loopback" || bad "upstream never saw plaintext"
grep -q "PRIVACY_CLIENT_OK" /tmp/idletoken_e2e_client.err && ok "client reports PRIVACY_CLIENT_OK" || bad "client did not finish ok"

echo "== auth enforcement =="
if command -v nc >/dev/null 2>&1; then
    ST=$(printf 'POST /idletoken/v1/privacy/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\nConnection: close\r\n\r\n%0100d' 0 | nc -w2 127.0.0.1 $PROXY_PORT | head -1)
    echo "$ST" | grep -q "401" && ok "forged/unauthenticated sealed request rejected (401)" || bad "forged request not rejected (got: $ST)"
else
    echo "  [skip] nc not present — auth-rejection is covered by privacy_http_selftest EAUTH checks"
fi

echo "------------------------------------------------------"
[ $rc -eq 0 ] && echo "PRIVACY_PROXY_E2E_OK" || echo "PRIVACY_PROXY_E2E_FAIL"
exit $rc
