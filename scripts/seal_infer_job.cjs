// Post one sealed job to a running idletoken-platform-agent, the way the real
// platform does (POST /infer, direct transport).
//
// This exists so G_OVERFLOW can assert its ironclad rule against the REAL AGENT
// BINARY rather than against the coordinator's own predicate. The whole rule
// -- platform-dispatched work is never forwarded on -- rests on the agent
// actually setting X-IdleToken-Origin. If that header is ever dropped, the
// coordinator will not error, will not crash, and will not log anything unusual:
// it will simply start forwarding other people's jobs. Only an assertion driven
// through the agent's own product can catch that (overflow-routing-design §3).
//
// Prints "HTTP <status>" and, when the envelope came back, "SEALED_OK".
//
// Usage: node scripts/seal_infer_job.cjs --agent 127.0.0.1:PORT
//            --pubkey <b64 agent X25519 pk> --prompt TEXT [--sodium-from DIR]
const http = require('node:http');
const path = require('node:path');

const arg = (n, d) => {
  const i = process.argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : d;
};
const gatewayDir = arg('sodium-from',
  path.join(__dirname, '..', 'platform', 'packages', 'gateway'));
const _sodium = require(require.resolve('libsodium-wrappers', { paths: [gatewayDir] }));

const agent = arg('agent', '127.0.0.1:9700');
const pubkeyB64 = arg('pubkey', '');
const prompt = arg('prompt', 'hello');

_sodium.ready.then(() => {
  const V = _sodium.base64_variants.ORIGINAL;
  const agentPk = _sodium.from_base64(pubkeyB64, V);
  // A reply key pair per job, exactly as the gateway does it.
  const reply = _sodium.crypto_box_keypair();
  const inner = JSON.stringify({
    model: 'qwen3.5-0.8b',
    messages: [{ role: 'user', content: prompt }],
    maxTokens: 8,
  });
  const body = JSON.stringify({
    sealed_request: _sodium.to_base64(
      _sodium.crypto_box_seal(Buffer.from(inner), agentPk), V),
    reply_to: _sodium.to_base64(reply.publicKey, V),
  });

  const [host, port] = agent.split(':');
  const req = http.request(
    { host, port: Number(port), path: '/infer', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } },
    (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        console.log(`HTTP ${res.statusCode}`);
        const text = Buffer.concat(chunks).toString('utf8');
        if (text.includes('sealed_response')) console.log('SEALED_OK');
        else console.log(text.slice(0, 300));
      });
    });
  req.on('error', (e) => { console.log(`HTTP 0`); console.log(String(e.message)); });
  req.end(body);
});
