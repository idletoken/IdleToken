#!/usr/bin/env python3
"""Receipt-robustness gate: the REAL platform agent against a stub gateway that
returns crafted delivery receipts.

Every case asserts what the agent must do when the platform's receipt is odd
(a future or buggy gateway): skip cleanly, never crash, never run inference
without a valid, acknowledged receipt, and never blame the seller's uplink
for a job the platform delivered with no time budget left.

Pure Python on purpose: no node, no real gateway, no Redis, so this runs on
every builder as part of `make -f Makefile.platform httpcheck`.

Usage: agent_receipt_gate.py <idletoken-platform-agent binary>
Prints one JSON line per case and AGENT_RECEIPT_OK <n>; exits 1 on any miss.
"""
import http.server, json, os, signal, socketserver, subprocess, sys, tempfile, threading, time

WAIT_S = 5.0

def make_cases():
    now_ms = lambda: int(time.time() * 1000)
    job = lambda **k: dict({'job_id': 'j1', 'delivery_token': 't1', 'sealed_request': 'AAAA'}, **k)
    ok_lease = {'ok': True, 'provider_lease_ms': 60000}
    return {
        # A valid receipt: acknowledged, then processed (the garbage seal fails
        # locally and a failure result is posted -- the point is it got there).
        'valid-receipt':      dict(poll=lambda: job(remaining_ms=120000, expires_at_ms=now_ms() + 120000), ack=ok_lease,
                                   expect=dict(ack=True, result=True, skip=False)),
        # A token with no parseable deadline: skip, never ack, never run.
        'missing-deadline':   dict(poll=lambda: job(), ack=ok_lease,
                                   expect=dict(ack=False, result=False, skip=True)),
        # Budget already spent on arrival: not a link fault, nothing to hand back.
        'remaining-zero':     dict(poll=lambda: job(remaining_ms=0), ack=ok_lease,
                                   expect=dict(ack=False, result=False, skip=False,
                                               msg='deadline was already spent', forbid='too slow')),
        # A malformed lease from the platform cancels rather than crashes.
        'lease-string':       dict(poll=lambda: job(remaining_ms=120000), ack={'ok': True, 'provider_lease_ms': '60000'},
                                   expect=dict(ack=True, result=False, skip=True)),
        'lease-zero':         dict(poll=lambda: job(remaining_ms=120000), ack={'ok': True, 'provider_lease_ms': 0},
                                   expect=dict(ack=True, result=False, skip=True)),
        # A remaining_ms beyond 24 h is capped, not rejected.
        'remaining-huge':     dict(poll=lambda: job(remaining_ms=999999999999), ack=ok_lease,
                                   expect=dict(ack=True, result=True, skip=False)),
        # A gateway that pretty-prints JSON must still deliver a readable receipt.
        'pretty-printed':     dict(poll=lambda: None, ack=ok_lease,
                                   raw='{\n  "job_id": "j1",\n  "delivery_token": "t1",\n  "remaining_ms": 120000 ,\n  "sealed_request": "AAAA"\n}',
                                   expect=dict(ack=True, result=True, skip=False)),
    }

class StubCoord(http.server.BaseHTTPRequestHandler):
    """The coordinator the agent reads its identity from. ctx_size must be at
    least the platform's minimum tier or the agent refuses to list."""
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == '/idletoken/v1/stats':
            return self._json(200, {'model': 'dsv4-flash', 'quant': 'IQ2_XXS', 'ctx_size': 8192, 'queue_depth': 0,
                                    'concurrency': 1, 'avg_service_ms': 100, 'shared_mode': True,
                                    'engine_verified': True, 'engine_link': 'unix'})
        return self._json(404, {'error': 'not stubbed'})
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        return self._json(503, {'error': 'stub coordinator does not run inference'})

class StubGateway(http.server.BaseHTTPRequestHandler):
    cfg = None; log = []; delivered = False
    def log_message(self, *a): pass
    def _json(self, code, obj=None, raw=None):
        b = raw.encode() if raw is not None else json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = self.rfile.read(n).decode(errors='replace')
        p = self.path; StubGateway.log.append((p, body[:160]))
        if p == '/providers':
            return self._json(201, {'id': 'stubprov'})
        if p.endswith('/relay/poll'):
            if not StubGateway.delivered:
                StubGateway.delivered = True
                return self._json(200, StubGateway.cfg['poll'](), raw=StubGateway.cfg.get('raw'))
            time.sleep(0.3)
            return self._json(200, {'job_id': None})
        if p.endswith('/relay/ack'):
            return self._json(200, StubGateway.cfg['ack'])
        return self._json(200, {'ok': True})
    def do_GET(self): return self._json(200, {'ok': True})
    def do_PUT(self): return self.do_POST()

def serve(handler):
    srv = socketserver.ThreadingTCPServer(('127.0.0.1', 0), handler); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]

def run_case(agent, name, cfg, coord_port):
    StubGateway.cfg = cfg; StubGateway.log = []; StubGateway.delivered = False
    gw, port = serve(StubGateway)
    state = tempfile.mkdtemp(prefix='receipt-gate-')
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(NO_PROXY='127.0.0.1,localhost', IDLETOKEN_STATE_DIR=state)
    proc = subprocess.Popen([agent, '--relay', '--platform', f'http://127.0.0.1:{port}',
                             '--coord', f'http://127.0.0.1:{coord_port}', '--name', 'receipt-gate',
                             '--jwt', 'stub', '--key-file', os.path.join(state, 'agent.key')],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    time.sleep(WAIT_S); alive = proc.poll() is None
    proc.send_signal(signal.SIGTERM)
    try: err = proc.communicate(timeout=3)[1]
    except subprocess.TimeoutExpired: proc.kill(); err = proc.communicate()[1]
    gw.shutdown()
    paths = [p for p, _ in StubGateway.log]
    got = dict(ack=any(p.endswith('/relay/ack') for p in paths),
               result=any(p.endswith('/relay/result') for p in paths),
               skip='skipping inference' in err,
               crash=(not alive and proc.returncode not in (0, -15, 143)) or 'Segmentation' in err or 'Abort' in err)
    e = cfg['expect']
    ok = got['ack'] == e['ack'] and got['result'] == e['result'] and got['skip'] == e['skip'] and not got['crash']
    if e.get('msg') and e['msg'] not in err: ok = False
    if e.get('forbid') and e['forbid'] in err: ok = False
    print(json.dumps({'case': name, **got, 'ok': ok}), flush=True)
    if not ok:
        for line in err.strip().splitlines()[-5:]: print('   agent: ' + line[:160], file=sys.stderr)
        print('   gateway saw: ' + ' '.join(p.rsplit('/', 1)[-1] for p in paths[:10]), file=sys.stderr)
    return ok

def main():
    if len(sys.argv) != 2: print(__doc__); return 2
    agent = os.path.abspath(sys.argv[1])
    if not os.access(agent, os.X_OK): print(f'AGENT_RECEIPT_FAIL: {agent} is not executable'); return 1
    coord, coord_port = serve(StubCoord)
    try:
        cases = make_cases()
        results = [run_case(agent, n, c, coord_port) for n, c in cases.items()]
    finally:
        coord.shutdown()
    if all(results):
        print(f'AGENT_RECEIPT_OK {len(results)}'); return 0
    print(f'AGENT_RECEIPT_FAIL {results.count(False)} of {len(results)} cases'); return 1

if __name__ == '__main__':
    sys.exit(main())
