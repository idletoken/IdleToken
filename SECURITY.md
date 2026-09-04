# Security

## Reporting

Email **security@idletoken.ai**, or open a GitHub security advisory. Please
don't file a public issue for anything exploitable.

Include what you did, what happened, and the version. We'll acknowledge within
a few days.

## What this software already assumes

- **The inference API is bound to 127.0.0.1.** It is not reachable from another
  machine by design, and the coordinator rewrites a non-loopback bind address
  rather than honouring it. Browser requests that carry an `Origin` header are
  refused as a CSRF control; ordinary local API clients do not carry one. This
  is not authentication against another program running on the same computer,
  so platform consumption is bounded by the account's Spark balance. A daily
  limit applies only when a user or operator explicitly configures one. If an
  operator explicitly configures the legacy API-token option, that token is
  still enforced. To reach your own cluster from elsewhere, use the platform
  relay rather than exposing the port.
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
- **Installers carry no OS code-signing certificate.** The Windows installer is
  not Authenticode-signed and the macOS dmg is not notarized, so the official
  installer produces a SmartScreen box on Windows and a Gatekeeper refusal on
  macOS. That is said plainly because the alternative trains you to click
  through those warnings for someone else's installer too. Verify the download
  instead — see "Verifying your download" below.
- **Model weights are fetched over the network.** Downloads come from Hugging
  Face, falling back to the `hf-mirror.com` mirror when the origin is
  unreachable, and every finished file is checked against the SHA-256 recorded
  in the model manifest before it is used.

## Verifying your download

Current releases contain native installers only. There is no in-app updater,
update feed, updater ZIP, detached signature or provenance sidecar. Download
the installer from the official GitHub release and compare its SHA-256 with the
digest GitHub displays for that asset before running it.

Official downloads come only from
`https://github.com/idletoken/IdleToken/releases`. Software portals, cloud-drive
links, chat attachments and "green"/repacked builds are not ours whatever they
are called. `scripts/release-channels.json` is the machine-readable list of
official origins and the things support will never ask you to do — notably: we
never ask anyone to run a downloaded "repair tool" or
"accelerator pack", and we never ask for a key, token, pairing code or PSK.

## Out of scope

Denial of service against your own cluster, and anything requiring physical
access to a machine that is already running the software.
