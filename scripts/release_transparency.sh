#!/usr/bin/env bash
# Append-only, hash-chained release log — the control for "the signing key
# itself was used against us" (DIST-08, DIST-09, CHAIN-07).
#
# WHY A SIGNATURE IS NOT ENOUGH
#
# Every other defence in this workstream assumes the release key is ours. The
# moment it is not — a stolen key, a compromised build host, an insider — the
# attacker produces artifacts that pass every signature check we have, and the
# user has no way to tell. Signatures answer "was the key used?"; they cannot
# answer "did the project mean to publish this?".
#
# What CAN answer that is a public, append-only record. Every release appends
# one line whose hash covers the previous line, so the log is a chain. Then:
#
#   * A build published without a log entry is visibly unlogged. An attacker who
#     stays out of the log is detectable by anyone who checks; an attacker who
#     writes to the log leaves permanent evidence with a timestamp.
#   * Rewriting history to remove that evidence breaks the chain against ANY
#     older copy — the maintainers', a mirror's, a user's. Deletion stops being
#     silent.
#
# This is deliberately the cheap version of a transparency log: no witness
# network, no cosigning, one repository. It does not PREVENT a key-compromise
# release; it removes the attacker's ability to do it invisibly, which is the
# property that turns a supply-chain compromise from permanent into an incident
# with a discovery date.
#
# Usage:
#   scripts/release_transparency.sh append <provenance.json> [--sig FILE]
#   scripts/release_transparency.sh verify [--log FILE]
#   scripts/release_transparency.sh self-test
#
# Contract: last line is RELEASE_LOG_OK[: detail] or RELEASE_LOG_FAIL: <reason>.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
LOG_DEFAULT="$ROOT/releases/transparency-log.jsonl"

fail() { echo "RELEASE_LOG_FAIL: $*"; exit 1; }

CMD="${1:-}"; shift 2>/dev/null || true
LOG="$LOG_DEFAULT"
PROV="" SIGFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="${2:-}"; shift 2 ;;
        --sig) SIGFILE="${2:-}"; shift 2 ;;
        --*)   fail "unknown option $1" ;;
        *)     PROV="$1"; shift ;;
    esac
done

# One implementation of the chain rule, used by both append and verify. Writing
# it twice is how the two drift and the log silently stops meaning anything.
chain_py() {
python3 - "$@" <<'PY'
import hashlib, json, os, sys

MODE = sys.argv[1]
LOG = sys.argv[2]


def canonical(entry):
    """The bytes the chain hash covers: the entry without its own `hash`,
    serialised deterministically. Sorting keys matters — a re-serialisation
    that reorders fields must not change the hash, or verification of a log
    written by another tool fails for no reason."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def read_log(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append((n, json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"FAIL\tline {n} is not valid JSON: {exc}")
                raise SystemExit(0)
    return out


def verify(path):
    entries = read_log(path)
    if not entries:
        print("EMPTY\t")
        return
    prev = "0" * 64
    for n, e in entries:
        if e.get("prev") != prev:
            print(f"FAIL\tline {n} ({e.get('version')}/{e.get('platform')}) claims prev "
                  f"{str(e.get('prev'))[:16]}…, the chain says {prev[:16]}… — an entry "
                  f"was removed, reordered or rewritten")
            return
        want = hashlib.sha256(canonical(e)).hexdigest()
        if e.get("hash") != want:
            print(f"FAIL\tline {n} ({e.get('version')}/{e.get('platform')}) was edited: "
                  f"its contents hash to {want[:16]}…, the entry records {str(e.get('hash'))[:16]}…")
            return
        prev = e["hash"]
    print(f"OK\t{len(entries)} entries, chain head {prev}")


if MODE == "verify":
    verify(LOG)
elif MODE == "append":
    prov_path, sig_path, keyid = sys.argv[3], sys.argv[4], sys.argv[5]
    doc = json.load(open(prov_path))
    entries = read_log(LOG)
    prev = entries[-1][1]["hash"] if entries else "0" * 64
    h = hashlib.sha256()
    with open(prov_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    entry = {
        "schema": "idletoken-release-log/1",
        "version": doc.get("version"),
        "platform": doc.get("platform"),
        "commit": doc.get("source", {}).get("commit"),
        "dirty": bool(doc.get("source", {}).get("dirty")),
        "builtAt": doc.get("builtAt"),
        "builder": doc.get("builder"),
        "enginePin": doc.get("engine", {}).get("llamacppPin"),
        "provenanceSha256": h.hexdigest(),
        "provenanceSigned": bool(sig_path) and os.path.exists(sig_path),
        "signingKeyId": keyid,
        "artifacts": [
            {"name": a["name"], "sha256": a["sha256"]}
            for a in sorted(doc.get("artifacts", []), key=lambda x: x["name"])
        ],
        "prev": prev,
    }
    entry["hash"] = hashlib.sha256(canonical(entry)).hexdigest()
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    # Append, never rewrite. Opening in "a" is the whole append-only property at
    # this layer; the git history behind the file is the second layer.
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"OK\tappended {entry['version']}/{entry['platform']} hash {entry['hash']}")
PY
}

case "$CMD" in
    append)
        [ -n "$PROV" ] || fail "append needs a provenance json"
        [ -f "$PROV" ] || fail "no such provenance file: $PROV"
        [ -n "$SIGFILE" ] || SIGFILE="$PROV.sig"
        # Refuse to log an unsigned record. A log entry for something nobody
        # signed would let an attacker manufacture a plausible history entry
        # without holding any key at all.
        if [ ! -f "$SIGFILE" ]; then
            fail "no signature at $SIGFILE — sign the provenance record before logging it (release_manifest.sh --sign)"
        fi
        PINNED=$(python3 -c "import json;print(json.load(open('$ROOT/client/src-tauri/tauri.conf.json'))['plugins']['updater']['pubkey'])" 2>/dev/null)
        [ -n "$PINNED" ] || fail "could not read the pinned updater key"
        KEYID=$(python3 "$ROOT/scripts/minisign_verify.py" --pubkey "$PINNED" --sig "$SIGFILE" "$PROV" --json 2>/dev/null \
                | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['ok'] else 1) or print(d['keyId'])" 2>/dev/null) || \
            fail "the provenance record is not signed by the pinned release key — refusing to log it"
        out=$(chain_py append "$LOG" "$PROV" "$SIGFILE" "$KEYID") || fail "could not append"
        case "$out" in
            OK*) echo "  ${out#OK$'\t'}" ;;
            *)   fail "${out#*$'\t'}" ;;
        esac
        # Immediately re-verify the whole chain: an append that broke it should
        # be found now, by the person who did it, not months later by a user.
        v=$(chain_py verify "$LOG")
        case "$v" in
            OK*) echo "RELEASE_LOG_OK: ${v#OK$'\t'}" ;;
            *)   fail "the log no longer verifies after this append: ${v#*$'\t'}" ;;
        esac
        ;;
    verify)
        v=$(chain_py verify "$LOG")
        case "$v" in
            OK*)    echo "RELEASE_LOG_OK: ${v#OK$'\t'}" ;;
            EMPTY*) echo "RELEASE_LOG_OK: no entries yet ($LOG)" ;;
            *)      fail "${v#*$'\t'}" ;;
        esac
        ;;
    self-test)
        # The positive control. A chain checker that cannot go red is the exact
        # failure mode this repo has shipped twice in other checkers, and here
        # it would mean the log quietly stopped detecting rewritten history.
        T=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-relog.XXXXXX") || exit 1
        trap 'rm -rf "$T"' EXIT
        bad=0
        mk() {  # $1 = version -> a signed-looking provenance + a fake sig
            printf '%s' "x" > "$T/$1.bin"
            python3 - "$T/$1.json" "$1" <<'PY'
import json, sys
json.dump({
    "schema": "idletoken-release-provenance/1", "version": sys.argv[2],
    "platform": "selftest", "builtAt": "2026-01-01T00:00:00Z", "builder": "selftest",
    "source": {"commit": "0" * 40, "dirty": False},
    "engine": {"llamacppPin": "deadbee", "patches": []},
    "artifacts": [{"name": f"a-{sys.argv[2]}.bin", "sha256": "ab" * 32, "bytes": 1}],
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
        }
        L="$T/log.jsonl"
        for v in 1.0.0 1.0.1 1.0.2; do
            mk "$v"
            # append without the signature gate (this control is about the
            # chain, not the signature; the signature path has its own test)
            chain_py append "$L" "$T/$v.json" "" "SELFTEST" >/dev/null || { echo "  [BAD] append failed"; bad=1; }
        done
        r=$(chain_py verify "$L")
        case "$r" in OK*) echo "  [ok] a 3-entry chain verifies (${r#OK$'\t'})" ;; *) echo "  [BAD] a clean chain did not verify: $r"; bad=1 ;; esac

        # Control A: edit an entry in place (the "we never published that"
        # rewrite an attacker would attempt).
        cp "$L" "$T/edited.jsonl"
        python3 - "$T/edited.jsonl" <<'PY'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
lines[1]["artifacts"][0]["sha256"] = "cd" * 32
open(sys.argv[1], "w").write(
    "\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in lines) + "\n")
PY
        r=$(chain_py verify "$T/edited.jsonl")
        case "$r" in FAIL*) echo "  [ok] control: an edited entry is detected (${r#FAIL$'\t'})" ;; *) echo "  [BAD] CONTROL: an edited entry verified"; bad=1 ;; esac

        # Control B: delete an entry from the middle (the "make the malicious
        # release disappear" rewrite).
        python3 - "$L" "$T/deleted.jsonl" <<'PY'
import sys
lines = [l for l in open(sys.argv[1]) if l.strip()]
open(sys.argv[2], "w").writelines(lines[:1] + lines[2:])
PY
        r=$(chain_py verify "$T/deleted.jsonl")
        case "$r" in FAIL*) echo "  [ok] control: a removed entry is detected (${r#FAIL$'\t'})" ;; *) echo "  [BAD] CONTROL: a removed entry verified"; bad=1 ;; esac

        # Control C: truncate the tail and re-append something else. The chain
        # still verifies (that is honest — a chain alone cannot detect this),
        # so the property claimed is the one that holds: ANY older copy of the
        # log disagrees at that point. Assert exactly that, so nobody later
        # believes the log promises more than it does.
        head -2 "$L" > "$T/forked.jsonl"
        mk 9.9.9
        chain_py append "$T/forked.jsonl" "$T/9.9.9.json" "" "SELFTEST" >/dev/null
        r=$(chain_py verify "$T/forked.jsonl")
        head_old=$(chain_py verify "$L" | cut -f2)
        head_new=$(printf '%s' "$r" | cut -f2)
        if [ "$head_old" != "$head_new" ]; then
            echo "  [ok] control: a truncate-and-refork produces a DIFFERENT chain head than the original"
        else
            echo "  [BAD] CONTROL: a forked log produced the same head as the original — the chain is not covering entries"; bad=1
        fi
        [ "$bad" = 0 ] || fail "self-test controls did not all fire"
        echo "RELEASE_LOG_OK: self-test (append, edit, delete and fork controls all fire)"
        ;;
    *)
        fail "usage: release_transparency.sh append <provenance.json> [--sig F] | verify [--log F] | self-test"
        ;;
esac
