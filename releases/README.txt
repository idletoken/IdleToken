IdleToken release transparency log
==================================

`transparency-log.jsonl` is an append-only, hash-chained record of every
official release. One line per platform per version. Each line's `hash` covers
the whole entry including the previous line's hash, so the file is a chain.

Why it exists
-------------

Every other release defence assumes the signing key is ours. If it is not — a
stolen key, a compromised build machine, an insider — the attacker produces
artifacts that pass every signature check, and no user can tell. A signature
answers "was the key used?". It cannot answer "did the project mean to publish
this?".

A public chain can:

  * A build published without a log entry here is visibly unlogged.
  * A build published WITH an entry leaves permanent, timestamped evidence.
  * Removing that evidence later breaks the chain against any older copy of
    this file — ours, a mirror's, or one a user saved.

That does not prevent a key-compromise release. It removes the attacker's
ability to do it invisibly, which is what turns a permanent supply-chain
compromise into an incident with a discovery date.

How to check a release yourself
-------------------------------

    scripts/release_transparency.sh verify
    scripts/verify_release.sh --provenance <the .provenance.json> <the installer>

Neither needs IdleToken installed. `verify_release.sh` needs python3 and
shasum, nothing else — deliberately, because the person who most needs to check
is the person deciding whether to run an installer for the first time.

What this log does NOT claim
----------------------------

It is a single-operator log with no witness network and no cosigners. Somebody
who controls both the signing key and this repository can fork the chain from
any point; what they cannot do is make an older copy of the file agree with the
fork. Anyone comparing a saved copy — including the chain head printed at each
release — sees the divergence.

Empty until the first release is logged. `scripts/release_manifest.sh --sign`
produces the provenance record; `scripts/release_transparency.sh append` adds
the entry, and refuses to log a record that is not signed by the pinned release
key.
