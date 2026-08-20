// A stand-in IdleToken platform for the overflow gate (G_OVERFLOW).
//
// It speaks the two routes the coordinator's overflow path talks to, using the
// SAME libsodium the real gateway uses (platform/packages/gateway's
// libsodium-wrappers), so a passing gate means the coordinator's hand-written
// TweetNaCl+BLAKE2b construction really is wire-compatible with the platform —
// not merely self-consistent.
//
//   GET  /idletoken/v1/platform-key  -> the signed attestation (§5.1c)
//   POST /idletoken/v1/sealed/chat   -> open the envelope, seal an answer back
//
// It also records what the gate needs to judge:
//   conn.log   one line per ACCEPTED connection. "the coordinator opened no
//              outbound connection" is asserted against this file, and the
//              gate arms it with a positive control before trusting it.
//   wire.log   every request's raw bytes as they arrived. "the prompt did not
//              go out in the clear" is a search over this file — and the gate
//              proves the search works by planting a plaintext request first.
//
// --mode picks which of the bad samples to serve, so the gate can assert that
// each is refused:
//   good      a correct attestation
//   swapped   correctly signed, by a signing key the coordinator does not trust
//   expired   correctly signed, not_after in the past
//   nodomain  signed over "<pubkey>\n<not_after>" with no domain prefix
//   unsigned  no sig field at all
//
// Usage: node scripts/stub_platform.cjs --port N --record DIR [--mode good]
//                                        [--charge-milli N] [--sodium-from DIR]
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

// libsodium comes out of the REAL gateway's node_modules, resolved by path
// rather than by name: CommonJS resolves relative to this file, and pnpm's
// layout puts nothing next to scripts/. Borrowing the platform's own copy is
// also the point -- the gate should hold the coordinator against the library
// the platform actually runs, not a second copy that could drift a version.
const gatewayDir = process.argv.includes('--sodium-from')
  ? process.argv[process.argv.indexOf('--sodium-from') + 1]
  : path.join(__dirname, '..', 'platform', 'packages', 'gateway');
const _sodium = require(require.resolve('libsodium-wrappers', { paths: [gatewayDir] }));

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : dflt;
};

const port = Number(arg('port', '18790'));
const mode = arg('mode', 'good');
const recordDir = arg('record', '/tmp/idletoken-stub-platform');
// What the platform claims it charged. The gate sets this above the
// coordinator's daily cap so one borrowed request is enough to reach it.
const chargeMilli = Number(arg('charge-milli', '1'));

fs.mkdirSync(recordDir, { recursive: true });
const connLog = path.join(recordDir, 'conn.log');
const wireLog = path.join(recordDir, 'wire.log');
for (const f of [connLog, wireLog]) fs.writeFileSync(f, '');

// Everything below needs the WASM module up, so the whole server is started
// from inside the ready handler rather than at top level (no top-level await in
// CommonJS).
_sodium.ready.then(() => main());

function main() {
// The encryption key pair (what the coordinator seals to) and the signing key
// pair (what vouches for it). In `swapped` mode the coordinator is given a
// DIFFERENT verify key than the one that signs, which is the substituted-key
// attack the pin exists to stop.
const enc = _sodium.crypto_box_keypair();
const signer = _sodium.crypto_sign_keypair();
const trusted = mode === 'swapped' ? _sodium.crypto_sign_keypair() : signer;

const b64 = (u8) => _sodium.to_base64(u8, _sodium.base64_variants.ORIGINAL);

// The verify key the coordinator should be started with. Written to a file
// rather than printed, so the gate does not have to parse stdout.
fs.writeFileSync(path.join(recordDir, 'verify_key.txt'), b64(trusted.publicKey));

function attestation() {
  const pubkey = b64(enc.publicKey);
  const notAfter = mode === 'expired'
    ? new Date(Date.now() - 86_400_000).toISOString()
    : new Date(Date.now() + 30 * 86_400_000).toISOString();
  // Byte for byte the same canonical string as KeyAttestationService.
  const prefix = mode === 'nodomain' ? '' : 'idletoken-platform-key-v1\n';
  const msg = new TextEncoder().encode(`${prefix}${pubkey}\n${notAfter}`);
  const body = {
    alg: 'x25519-sealedbox',
    sign_alg: 'ed25519',
    pubkey,
    not_after: notAfter,
    key_id: _sodium.to_hex(_sodium.crypto_generichash(8, _sodium.from_string(pubkey))),
  };
  if (mode !== 'unsigned') {
    body.sig = b64(_sodium.crypto_sign_detached(msg, signer.privateKey));
  }
  return body;
}

const srv = http.createServer((req, res) => {
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    const raw = Buffer.concat(chunks);
    // The whole request as it arrived, headers included: a gate looking for
    // plaintext must be able to see everything that crossed the wire, not only
    // the part we chose to decode.
    fs.appendFileSync(wireLog, `${req.method} ${req.url}\n${raw.toString('utf8')}\n---\n`);

    if (req.method === 'GET' && req.url.startsWith('/idletoken/v1/platform-key')) {
      const body = JSON.stringify(attestation());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(body);
      return;
    }
    if (req.method === 'POST' && req.url.startsWith('/idletoken/v1/sealed/chat')) {
      let inner, replyTo;
      try {
        const env = JSON.parse(raw.toString('utf8'));
        replyTo = _sodium.from_base64(env.reply_to, _sodium.base64_variants.ORIGINAL);
        const opened = _sodium.crypto_box_seal_open(
          _sodium.from_base64(env.sealed_request, _sodium.base64_variants.ORIGINAL),
          enc.publicKey, enc.privateKey);
        inner = JSON.parse(Buffer.from(opened).toString('utf8'));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: String(e?.message ?? e) }));
        return;
      }
      // What we could read out of the envelope, so the gate can assert that the
      // prompt really did arrive (sealed) and that the api_key came inside it
      // rather than in a header.
      fs.appendFileSync(path.join(recordDir, 'opened.log'),
                        JSON.stringify(inner) + '\n');
      const answer = {
        text: `borrowed-answer for ${inner.messages?.[inner.messages.length - 1]?.content ?? '?'}`,
        model: inner.model,
        usage: { input_tokens: 11, output_tokens: 7 },
        credits: { charged_milli: chargeMilli, balance_after_milli: 1_000_000 },
      };
      const sealed = b64(_sodium.crypto_box_seal(Buffer.from(JSON.stringify(answer)), replyTo));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ sealed_response: sealed }));
      return;
    }
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end('{}');
  });
});

// Counted at the CONNECTION level, not per request: the assertion the gate
// rests on is "the coordinator did not dial out at all", and a connection that
// was opened and then abandoned still contradicts it.
srv.on('connection', (s) => {
  fs.appendFileSync(connLog, `conn ${s.remoteAddress}:${s.remotePort}\n`);
});

srv.listen(port, '127.0.0.1', () => {
  console.log(`stub platform on 127.0.0.1:${port} mode=${mode} record=${recordDir}`);
});
}
