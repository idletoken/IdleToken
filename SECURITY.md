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
  so local use is also bounded by the daily consumption limit. If an operator
  explicitly configures the legacy API-token option, that token is still
  enforced. To reach your own cluster from elsewhere, use the platform relay
  rather than exposing the port.
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

The in-app updater installs nothing whose signature does not verify against the
key compiled into the client (`28F23C3CE24BFDE9`). That protects upgrades of an
already-official install. It does **not** help with the first install: an
attacker's build simply carries the attacker's own key and updates itself
happily.

So every release also publishes a signed provenance record listing the SHA-256
of each artifact, and a verifier that needs neither IdleToken nor any extra
tool:

```
scripts/verify_release.sh --provenance idletoken-release-<version>-<platform>.provenance.json <the-installer>
```

It refuses unless the record is signed by `28F23C3CE24BFDE9` **and** your file's
digest is the one the record lists. `scripts/minisign_verify.py` does the
signature check on its own — pure Python, no dependencies, deliberately a second
implementation rather than the one the client uses, so a bug in one does not
silently excuse the other.

Official downloads come only from
`https://github.com/idletoken/IdleToken/releases`. Software portals, cloud-drive
links, chat attachments and "green"/repacked builds are not ours whatever they
are called. `scripts/release-channels.json` is the machine-readable list of
official origins, the key fingerprint, and the things support will never ask you
to do — notably: we never ask anyone to run a downloaded "repair tool" or
"accelerator pack", and we never ask for a key, token, pairing code or PSK.

Every release is also appended to a hash-chained log in `releases/`. That does
not prevent a release signed with a stolen key, but it means such a release is
either absent from the log or permanently recorded in it, and rewriting the log
breaks the chain against any older copy.

## Out of scope

Denial of service against your own cluster, and anything requiring physical
access to a machine that is already running the software.
