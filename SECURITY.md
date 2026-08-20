# Security

## Reporting

Email **security@idletoken.ai**, or open a GitHub security advisory. Please
don't file a public issue for anything exploitable.

Include what you did, what happened, and the version. We'll acknowledge within
a few days.

## What this software already assumes

- **The inference API is bound to 127.0.0.1 and always requires a token.** It
  is not reachable from another machine by design, and the coordinator rewrites
  a non-loopback bind address rather than honouring it. The client generates
  the API key and shows it to you; a request without it gets 401. To reach your
  own cluster from elsewhere, go through the platform relay rather than
  exposing the port.
- **The LAN is trusted for discovery and pairing.** Nodes find each other over
  UDP broadcast on the local network. Cluster links between machines carry TLS
  with a pre-shared key minted by the coordinator, and a worker refuses to start
  its rpc-server without one.
- **Prompt privacy on an untrusted cluster is best-effort.** Layer 0 and the
  embedding table are pinned to the coordinator, so a worker never sees token
  ids it could look up in a public GGUF; the cross-machine link is encrypted;
  requests that leave the machine are envelope-encrypted. Together these raise
  the cost of casual snooping — sniffing the wire, reading logs, dumping files.
  They do **not** defend against someone with root on a worker reading process
  memory, nor against learning-based inversion of hidden states. That boundary
  is stated, not hidden.
- **Installers are unsigned.** The Windows installer is not Authenticode-signed
  and the macOS dmg is not notarized. Verify the SHA256 published on the release
  page.
- **Model weights are fetched over the network.** Downloads come from Hugging
  Face, falling back to the `hf-mirror.com` mirror when the origin is
  unreachable, and every finished file is checked against the SHA-256 recorded
  in the model manifest before it is used.

## Out of scope

Denial of service against your own cluster, and anything requiring physical
access to a machine that is already running the software.
