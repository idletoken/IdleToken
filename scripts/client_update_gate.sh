#!/usr/bin/env bash
# G-UPDATE — the client's in-app updater, end to end against a feed this script
# signs itself.
#
# Why a local feed rather than the real one: the production feed announces
# whatever the last release was, so a gate pointed at it can only ever assert
# "found nothing" — which is also what a completely broken updater reports.
# Here the manifest is written per case, so all five answers are reachable and
# each one is checked:
#
#   1. same version announced        -> "you are up to date"      (found=false)
#   2. newer version announced       -> the prompt would appear   (found=true)
#   3. download of a signed artifact -> succeeds, bytes returned
#   4. download of a TAMPERED one    -> refused with a signature error
#   5. feed unreachable              -> an ERROR, never "up to date"
#
# Case 4 is the reason this gate exists. An updater downloads code and runs it;
# if a bad signature installed anyway, the update channel would be a remote code
# execution channel for whoever can answer for the release host. Case 5 is the
# second reason: reporting an unreachable feed as "current" is how machines sit
# months behind believing they are on the latest build.
#
# WHERE IT RUNS. `IDLETOKEN_CLIENT_NODE=<ssh alias>` runs the client on that
# machine (Windows compute node) with the feed served from here; unset runs it
# locally, which is only a smoke test — the installer handover this feature ends
# in is NSIS, and a macOS run says nothing about it.
#
# What is NOT covered, stated plainly: the final install + relaunch step. It
# replaces the running application, which under a harness is the harness's own
# binary. Everything up to and including signature verification runs through the
# production path (src-tauri/src/update.rs, driven by the same UI-test channel a
# user's click goes through); the last step is exercised by hand on a real
# release.
#
# Last-line contract (used by acceptance.sh):
#   UPDATE_GATE_OK
#   UPDATE_GATE_FAIL: <reason>
#   UPDATE_GATE_SKIP: <reason>
set -u
cd "$(dirname "$0")/.." || exit 1
TMP="${TMPDIR:-/tmp}/idletoken-update-gate.$$"
PORT="${IDLETOKEN_UPDATE_GATE_PORT:-14631}"
SERVER_PID=""
TUNNEL_PID=""

# Quiet on purpose: the last line of this script is a contract, and bash's
# "Terminated" notice for the background server would otherwise be printed
# after it.
cleanup() {
    for pid in "$TUNNEL_PID" "$SERVER_PID"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done
    rm -rf "$TMP"
}
trap cleanup EXIT

fail() { echo "UPDATE_GATE_FAIL: $*"; exit 1; }
skip() { echo "UPDATE_GATE_SKIP: $*"; exit 0; }

command -v python3 >/dev/null 2>&1 || skip "python3 not available (needed to serve the test feed)"
command -v pnpm    >/dev/null 2>&1 || skip "pnpm not available (needed to sign the test artifact)"
[ -d client ] || skip "client/ not present"

# shellcheck disable=SC1091
. "$(dirname "$0")/client-gate-lib.sh"
export PATH="$HOME/.cargo/bin:$PATH"
cg_init || skip "$CG_ERR"
if [ "$CG_KIND" = local ] && [ "$(uname)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    skip "no display (the client is a GUI app)"
fi
mkdir -p "$TMP" || fail "cannot create $TMP"
echo "client under test: $(cg_where)"
cg_build || skip "$CG_ERR"

CURRENT_VERSION=$(grep -m1 '^version = ' client/src-tauri/Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
[ -n "$CURRENT_VERSION" ] || fail "could not read the client version from Cargo.toml"

# ---- a throwaway signing key ----------------------------------------------
# Its own keypair, never the production one: a gate that needs the release
# private key is a gate that only runs on the machine holding it.
(cd client && pnpm tauri signer generate -w "$TMP/test.key" --password "" -f) \
    >"$TMP/keygen.log" 2>&1 || fail "could not generate a test keypair (see $TMP/keygen.log)"
PUBKEY=$(cat "$TMP/test.key.pub") || fail "no public key produced"

# ---- the artifact + its signature ------------------------------------------
# Content does not matter: nothing installs it here. What matters is that the
# signature is over exactly these bytes.
head -c 262144 /dev/urandom > "$TMP/update-artifact.tar.gz" 2>/dev/null \
    || fail "could not create the test artifact"
(cd client && pnpm tauri signer sign -f "$TMP/test.key" -p "" "$TMP/update-artifact.tar.gz") \
    >"$TMP/sign.log" 2>&1 || fail "could not sign the test artifact (see $TMP/sign.log)"
SIG=$(cat "$TMP/update-artifact.tar.gz.sig") || fail "no signature produced"

write_manifest() {  # $1 = version, $2 = artifact file name
    # 127.0.0.1 for the artifact too, not just the manifest: the node reaches
    # both through the same reverse tunnel.
    python3 - "$TMP/latest.json" "$1" "$2" "$SIG" 127.0.0.1 "$PORT" <<'PY'
import json, sys
out, version, artifact, sig, host, port = sys.argv[1:7]
url = f"http://{host}:{port}/{artifact}"
# Every target points at the same file so this runs unchanged whichever of the
# three platforms the client under test happens to be.
targets = ["darwin-aarch64", "darwin-x86_64", "linux-x86_64", "linux-aarch64",
           "windows-x86_64", "windows-i686"]
json.dump({
    "version": version,
    "notes": "Test manifest written by scripts/client_update_gate.sh",
    "pub_date": "2026-08-13T00:00:00Z",
    "platforms": {t: {"signature": sig, "url": url} for t in targets},
}, open(out, "w"))
PY
}

# Loopback only, and reached from the node under test through an ssh reverse
# tunnel rather than over the LAN.
#
# The LAN route was tried first and is a trap: the address the node sees us on
# (%SSH_CLIENT%) is not necessarily an address we are listening on — on this
# testbed the two differ — and even when they agree, a host firewall or a
# guest-isolating access point turns the whole gate red for reasons that have
# nothing to do with the updater. The tunnel makes the feed 127.0.0.1 on the
# NODE, so it works over VPN, NAT and hostile Wi-Fi alike.
(cd "$TMP" && exec python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null
sleep 1
kill -0 "$SERVER_PID" 2>/dev/null || fail "the test feed server did not start on port $PORT"

if [ "$CG_KIND" != local ]; then
    ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -R "$PORT:127.0.0.1:$PORT" "$CG_NODE" >"$TMP/tunnel.log" 2>&1 &
    TUNNEL_PID=$!
    disown "$TUNNEL_PID" 2>/dev/null
    sleep 3
    kill -0 "$TUNNEL_PID" 2>/dev/null || fail "could not open the reverse tunnel to $CG_NODE (see $TMP/tunnel.log)"
fi

FEED="http://127.0.0.1:$PORT/latest.json"
echo "test feed: $FEED (served from this machine)"

run_case() {  # $1 = directives, $2 = timeout, $3 = log, $4 = feed url
    cg_run "$1" "$2" "$3" "IDLETOKEN_UPDATE_URL=$4" "IDLETOKEN_UPDATE_PUBKEY=$PUBKEY"
}

# --- case 1: the feed announces the version we are running -------------------
write_manifest "$CURRENT_VERSION" update-artifact.tar.gz
run_case "update-check,quit:9000" 60 "$TMP/case1.log" "$FEED" >/dev/null
R=$(cg_report "$TMP/case1.log" update)
[ -n "$R" ] || fail "case 1: the client reported nothing (see $TMP/case1.log)"
[ "$(cg_field "$R" found)" = "False" ] || fail "case 1: same version announced but the client reported found=$(cg_field "$R" found) — $R"
[ "$(cg_field "$R" current)" = "$CURRENT_VERSION" ] || fail "case 1: the client reports version $(cg_field "$R" current), Cargo.toml says $CURRENT_VERSION"
[ "$(cg_field "$R" feedOverridden)" = "True" ] || fail "case 1: the client did not use the test feed — $R"
echo "case 1 ok: same version -> up to date"

# --- case 2: a newer version ------------------------------------------------
write_manifest "99.0.0" update-artifact.tar.gz
run_case "update-check,quit:9000" 60 "$TMP/case2.log" "$FEED" >/dev/null
R=$(cg_report "$TMP/case2.log" update)
[ -n "$R" ] || fail "case 2: the client reported nothing (see $TMP/case2.log)"
[ "$(cg_field "$R" found)" = "True" ] || fail "case 2: a newer version was announced but the client did not offer it — $R"
[ "$(cg_field "$R" version)" = "99.0.0" ] || fail "case 2: offered version $(cg_field "$R" version), expected 99.0.0"
echo "case 2 ok: newer version -> offered"

# --- case 3: download a correctly signed artifact ---------------------------
run_case "update-download,quit:20000" 80 "$TMP/case3.log" "$FEED" >/dev/null
R=$(cg_report "$TMP/case3.log" update-download)
[ -n "$R" ] || fail "case 3: the client reported nothing (see $TMP/case3.log)"
[ "$(cg_field "$R" ok)" = "True" ] || fail "case 3: a correctly signed update failed to download — $R"
BYTES=$(cg_field "$R" bytes)
[ "${BYTES:-0}" -gt 0 ] 2>/dev/null || fail "case 3: download reported $BYTES bytes"
echo "case 3 ok: signed artifact downloaded and verified ($BYTES bytes)"

# --- case 4: the artifact is tampered with AFTER signing --------------------
# The fail-closed case. The manifest still carries the valid signature, so the
# only thing between these bytes and the user's machine is the check.
cp "$TMP/update-artifact.tar.gz" "$TMP/tampered.tar.gz"
printf 'evil' | dd of="$TMP/tampered.tar.gz" bs=1 seek=1024 conv=notrunc 2>/dev/null
write_manifest "99.0.0" tampered.tar.gz
run_case "update-download,quit:20000" 80 "$TMP/case4.log" "$FEED" >/dev/null
R=$(cg_report "$TMP/case4.log" update-download)
[ -n "$R" ] || fail "case 4: the client reported nothing (see $TMP/case4.log)"
[ "$(cg_field "$R" ok)" = "False" ] || fail "case 4: A TAMPERED UPDATE WAS ACCEPTED — $R"
case "$(cg_field "$R" reason)" in
    *signature*|*Signature*|*verif*|*Verif*) : ;;
    *) fail "case 4: refused, but not for the signature — $(cg_field "$R" reason)" ;;
esac
echo "case 4 ok: tampered artifact refused ($(cg_field "$R" reason))"

# --- case 5: the feed cannot be reached -------------------------------------
# Port 1 is reserved and nothing listens there.
run_case "update-check,quit:20000" 80 "$TMP/case5.log" "http://127.0.0.1:1/latest.json" >/dev/null
R=$(cg_report "$TMP/case5.log" update)
[ -n "$R" ] || fail "case 5: the client reported nothing (see $TMP/case5.log)"
[ "$(cg_field "$R" found)" != "False" ] || fail "case 5: an unreachable feed was reported as 'up to date' — $R"
[ -n "$(cg_field "$R" error)" ] || fail "case 5: no error reported for an unreachable feed — $R"
echo "case 5 ok: unreachable feed -> error, not a false all-clear"

echo "UPDATE_GATE_OK"
