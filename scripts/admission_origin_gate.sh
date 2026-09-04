#!/usr/bin/env bash
# G_ADMISSION — where a loopback request came from, and what that decides.
#
# Threat register: PROV-28, PRIV-05, PRIV-09, CHAIN-05 (docs/modified-client-
# threat-register-2026-08.md). Design: include/idletoken_admission.h.
#
# THE ATTACK THIS IS ABOUT. Overflow routing rests on one rule: a job the
# PLATFORM dispatched is finished here or refused here, and is never forwarded
# back out. The coordinator used to learn which requests those were from
# `X-IdleToken-Origin: platform` — a header the platform agent sets on ITSELF,
# on the provider's own machine. Delete that one line and every dispatched job
# reads as local work: free to be forwarded to a third machine, charged a second
# time, and shown to one more stranger, with nothing anywhere reporting an
# error.
#
# WHY THIS GATE EXISTS SEPARATELY FROM G_OVERFLOW. scripts/overflow_gate.sh
# judges the FORWARDING DECISION, and to do that it needs a coordinator with a
# live engine — which needs a GPU budget, and is a SKIP on any machine that has
# no memory to spare. This one judges the PROVENANCE MACHINERY, and every
# assertion below is reachable without an engine:
#
#   1  the coordinator publishes a per-process channel key and a persistent
#      local-origin marker, both 0600
#   2  restarting rolls the channel key and KEEPS the local marker (both halves
#      asserted: either one alone is a bug that looks like the other's fix)
#   3  private and --shared modes both select capability so ordinary compatible
#      API clients may borrow; strict/legacy overrides announce their trade-offs
#   4  a sharing coordinator refuses IDLETOKEN_LOG_PROMPTS out loud, and a
#      non-sharing one still honours it (PRIV-09 with its positive control)
#   5  THE REAL platform-agent binary, given a real sealed job, mints a real
#      capability — and its body binding is checked with the SYSTEM's sha256,
#      not ours, so a green here is not our hasher agreeing with itself
#   6  the coordinator's own verifier spends that capability exactly once
#   7  with no channel key to attach to, the agent still marks the job platform
#      work (fail-safe), and sends no capability at all
#   8  a capability minted under a different channel is refused as a bad
#      signature — the verifier discriminates rather than accepting everything
#
# Contract: last line ADMISSION_GATE_OK, ADMISSION_GATE_FAIL: <why>, or
# ADMISSION_GATE_SKIP: <why>.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
# shellcheck disable=SC1091
[ -f scripts/testbed.env ] && . ./scripts/testbed.env

REC=/tmp/idletoken-admission-gate
AGENT_PORT="${IDLETOKEN_ADMISSION_GATE_PORT:-18960}"
COORD_PORT=$((AGENT_PORT + 1))
MARKER="ADMISSIONGATEPROMPT5517"

skip() { echo "ADMISSION_GATE_SKIP: $*"; cleanup; exit 0; }
fail() { echo "ADMISSION_GATE_FAIL: $*"; cleanup; exit 1; }
note() { echo "  $*"; }

cleanup() {
    pkill -9 -f "[i]dletoken-platform-agent --port $AGENT_PORT" 2>/dev/null
    pkill -9 -f "[s]tub_coord_recorder.py --port $COORD_PORT" 2>/dev/null
    return 0
}

command -v python3 >/dev/null 2>&1 || skip "no python3 — scripts/stub_coord_recorder.py is the fixture"
command -v node    >/dev/null 2>&1 || skip "no node — scripts/seal_infer_job.cjs is the fixture"
command -v curl    >/dev/null 2>&1 || skip "no curl"

# sha256sum on Linux, shasum on macOS. One definition, because a claim that
# silently produced an empty digest would compare "" to "" and pass.
sha256_hex() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1
    else shasum -a 256 | cut -d' ' -f1; fi
}
printf 'abc' | sha256_hex | grep -q \
  '^ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad$' \
  || skip "the system sha256 tool does not reproduce the FIPS 180-4 vector for \"abc\" — this gate's independent oracle is not available here"
note "fixture: the system sha256 tool matches the published FIPS 180-4 vector for \"abc\""

[ -x ./idletoken-coord ] || { make coord >/dev/null 2>&1 || skip "no ./idletoken-coord and it would not build"; }
[ -x ./build/admission_verify ] || {
    make admissionverify >/dev/null 2>&1 || skip "could not build build/admission_verify (claim 6 spends through it)"
}
[ -x ./build/idletoken-platform-agent ] || {
    make -f Makefile.platform >/dev/null 2>&1 || skip "could not build the platform agent (claims 5-7 assert against it)"
}

SODIUM_DIR="${IDLETOKEN_SODIUM_DIR:-$ROOT/platform/packages/gateway}"
node -e "require.resolve('libsodium-wrappers', { paths: ['$SODIUM_DIR'] })" >/dev/null 2>&1 \
    || skip "libsodium-wrappers is not installed under $SODIUM_DIR — set IDLETOKEN_SODIUM_DIR"

cleanup; sleep 0.5
rm -rf "$REC"; mkdir -p "$REC/state" "$REC/recorded"

echo "======================================================"
echo " G_ADMISSION — request provenance (PROV-28 / CHAIN-05)"
echo "======================================================"

# Arm a coordinator far enough to publish its two files and print its policy,
# then let it exit. Giving ONE engine flag makes it refuse in argument
# validation — after the admission block, before anything that needs a GPU — so
# this runs on a control machine with no memory to spare. Everything this gate
# asserts about the coordinator is printed or written by then.
arm_coord() {   # arm_coord <state-dir> [extra env assignments...] ; echoes output
    local state="$1"; shift
    mkdir -p "$state"
    env "$@" IDLETOKEN_STATE_DIR="$state" ./idletoken-coord \
        --model-id qwen3.5-0.8b --llama-server-bin /bin/echo 2>&1
}

# ===================================================================
# Claim 1 — the two files exist, are 0600, and are 32 hex-encoded bytes.
# ===================================================================
out=$(arm_coord "$REC/state")
CHAN="$REC/state/coord-admission.key"
LOCAL="$REC/state/coord-local-origin.key"
[ -f "$CHAN" ]  || fail "claim 1: the coordinator published no admission channel key: $out"
[ -f "$LOCAL" ] || fail "claim 1: the coordinator published no local-origin marker: $out"
for f in "$CHAN" "$LOCAL"; do
    mode=$(ls -l "$f" | cut -c1-10)
    case "$mode" in
        -rw-------) ;;
        *) fail "claim 1: $f is $mode, not 0600 — a key any local user can read is not a key" ;;
    esac
    v=$(tr -d ' \r\n' < "$f")
    case "$v" in
        [0-9a-f][0-9a-f]*) [ "${#v}" = 64 ] || fail "claim 1: $f holds ${#v} characters, expected 64 hex" ;;
        *) fail "claim 1: $f does not hold lower-case hex: $v" ;;
    esac
done
note "claim 1: channel key and local-origin marker published, 0600, 64 hex each"

# ===================================================================
# Claim 2 — a restart rolls the channel key and KEEPS the local marker.
#
# Both halves, because each is the other's failure mode. A channel key that
# survived a restart would let a capability minted before the restart be spent
# after it; a local marker that did NOT survive would silently demote every
# client that cached it to "unattributed", which on a sharing machine looks
# exactly like the attack this whole mechanism exists to stop.
# ===================================================================
chan1=$(tr -d ' \r\n' < "$CHAN"); local1=$(tr -d ' \r\n' < "$LOCAL")
arm_coord "$REC/state" >/dev/null
chan2=$(tr -d ' \r\n' < "$CHAN"); local2=$(tr -d ' \r\n' < "$LOCAL")
[ "$chan1" != "$chan2" ] \
    || fail "claim 2: the channel key survived a restart — a capability minted by the previous process would still be spendable"
[ "$local1" = "$local2" ] \
    || fail "claim 2: the local-origin marker changed on restart — every client holding the old one just became unattributed"
note "claim 2: restart rolled the channel key and kept the local-origin marker"

# ===================================================================
# Claim 3 — the product origin policy, and its explicit overrides.
# ===================================================================
out=$(arm_coord "$REC/state")
printf '%s' "$out" | grep -q "origin policy 'capability'" \
    || fail "claim 3: a machine that does not share did not select the capability policy: $(printf '%s' "$out" | head -3)"
out=$(arm_coord "$REC/state" IDLETOKEN_SHARED=1)
printf '%s' "$out" | grep -q "origin policy 'capability'" \
    || fail "claim 3: a SHARING machine did not select capability — ordinary OpenAI/Anthropic clients would get 429 instead of borrowing: $(printf '%s' "$out" | head -3)"
printf '%s' "$out" | grep -q "loopback API clients may borrow" \
    || fail "claim 3: the sharing default does not say that compatible loopback clients may borrow"
note "claim 3: private and --shared modes default to capability for compatible loopback clients"

out=$(arm_coord "$REC/state" IDLETOKEN_SHARED=1 IDLETOKEN_OVERFLOW_ORIGIN_POLICY=strict)
printf '%s' "$out" | grep -q "OVERRIDDEN to 'strict'" \
    || fail "claim 3: the strict override did not announce itself"
printf '%s' "$out" | grep -q "Nimbalyst" \
    || fail "claim 3: the strict override did not warn that ordinary third-party API clients lose overflow"
note "claim 3: strict is explicit and warns that unmarked third-party clients cannot borrow"

out=$(arm_coord "$REC/state" IDLETOKEN_OVERFLOW_ORIGIN_POLICY=legacy)
printf '%s' "$out" | grep -q "OVERRIDDEN to 'legacy'" \
    || fail "claim 3: the legacy override did not announce itself — an override nobody can see in the log is one nobody remembers setting"
printf '%s' "$out" | grep -q "PROV-28" \
    || fail "claim 3: the legacy override announcement does not name what it re-opens"
out=$(arm_coord "$REC/state" IDLETOKEN_OVERFLOW_ORIGIN_POLICY=banana)
printf '%s' "$out" | grep -q "refuse:" \
    || fail "claim 3: a nonsense origin policy was accepted instead of failing the start"
note "claim 3: the legacy override is loud and names PROV-28; a nonsense value refuses the start"

# ===================================================================
# Claim 4 — PRIV-09: a sharing machine may not quote prompts, and says so.
# The second half is the positive control: without --shared the same variable
# is still honoured, so the refusal above is the rule and not a dead switch.
# ===================================================================
out=$(arm_coord "$REC/state" IDLETOKEN_SHARED=1 IDLETOKEN_LOG_PROMPTS=1)
printf '%s' "$out" | grep -q "IDLETOKEN_LOG_PROMPTS is set and is being" \
    || fail "claim 4: a sharing coordinator did not announce that it is ignoring IDLETOKEN_LOG_PROMPTS (PRIV-09)"
out=$(arm_coord "$REC/state" IDLETOKEN_LOG_PROMPTS=1)
printf '%s' "$out" | grep -q "IDLETOKEN_LOG_PROMPTS is set and is being" \
    && fail "claim 4 control: a NON-sharing coordinator also refused the operator's own debug switch — the rule has degenerated into refusing everything"
note "claim 4: shared refuses prompt excerpts out loud; non-shared still honours the switch"

# ===================================================================
# Claims 5-7 — the REAL platform agent binary.
#
# The agent is started with the same IDLETOKEN_STATE_DIR the coordinator armed,
# so it attaches to that coordinator's channel. Its forward goes to a recorder
# rather than to a coordinator: what is being judged here is what the agent
# SENDS, and a real coordinator would answer with its own verdict, which is a
# different claim (scripts/overflow_gate.sh claims 7-11).
# ===================================================================
start_agent() {   # start_agent <state-dir-or-empty>
    rm -f /tmp/idletoken-adm-agent.key
    : > "$REC/agent.log"
    if [ -n "$1" ]; then
        IDLETOKEN_STATE_DIR="$1" ./build/idletoken-platform-agent --port "$AGENT_PORT" \
            --coord "http://127.0.0.1:$COORD_PORT" \
            --key-file /tmp/idletoken-adm-agent.key >"$REC/agent.log" 2>&1 &
    else
        IDLETOKEN_STATE_DIR="$REC/empty" ./build/idletoken-platform-agent --port "$AGENT_PORT" \
            --coord "http://127.0.0.1:$COORD_PORT" \
            --key-file /tmp/idletoken-adm-agent.key >"$REC/agent.log" 2>&1 &
    fi
    disown 2>/dev/null
    for _ in $(seq 1 40); do grep -q "pubkey (b64)" "$REC/agent.log" && break; sleep 0.25; done
    AGENT_PK=$(grep "pubkey (b64)" "$REC/agent.log" | sed 's/.*: *//' | tr -d ' \r')
    [ -n "$AGENT_PK" ]
}

send_job() {   # send_job <prompt>; leaves headers.txt/body.bin in $REC/recorded
    rm -f "$REC/recorded/headers.txt" "$REC/recorded/body.bin"
    node scripts/seal_infer_job.cjs --agent "127.0.0.1:$AGENT_PORT" \
        --pubkey "$AGENT_PK" --prompt "$1" --sodium-from "$SODIUM_DIR" 2>&1
}

# The recorder stands in for the coordinator on loopback.
python3 scripts/stub_coord_recorder.py --port "$COORD_PORT" --record "$REC/recorded" \
    >"$REC/stub.log" 2>&1 &
disown 2>/dev/null
for _ in $(seq 1 40); do
    curl -s -m 1 "http://127.0.0.1:$COORD_PORT/idletoken/v1/stats" >/dev/null 2>&1 && break
    sleep 0.25
done
curl -s -m 2 "http://127.0.0.1:$COORD_PORT/idletoken/v1/stats" | grep -q '"ctx_size"' \
    || fail "fixture: the coordinator recorder did not come up on $COORD_PORT (see $REC/stub.log)"

# Re-arm so the channel on disk is the one the agent will attach to.
arm_coord "$REC/state" >/dev/null
start_agent "$REC/state" || fail "claim 5: the platform agent did not report a public key (see $REC/agent.log)"
out=$(send_job "$MARKER")
printf '%s' "$out" | grep -q "HTTP " \
    || fail "claim 5 fixture: seal_infer_job.cjs never posted a job, so nothing below is an assertion about the agent: $out"
[ -s "$REC/recorded/headers.txt" ] \
    || fail "claim 5: the agent never forwarded anything into the coordinator (see $REC/agent.log)"

grep -qi "^X-IdleToken-Origin: platform" "$REC/recorded/headers.txt" \
    || fail "claim 5: the agent's forward carries no platform-origin header — the legacy half of the rule is gone"
grep -qi "^X-IdleToken-Hops: 0" "$REC/recorded/headers.txt" \
    || fail "claim 5: the agent's forward declares no hop count, so a chain would have no ceiling (CHAIN-05)"
CAP=$(grep -i "^X-IdleToken-Job-Admission:" "$REC/recorded/headers.txt" | sed 's/^[^:]*: *//' | tr -d ' \r')
[ -n "$CAP" ] \
    || fail "claim 5: the agent minted no admission capability — the only part of the origin claim a modified sender cannot fake (see $REC/agent.log)"
note "claim 5: the real agent forwarded with origin, hops and a minted capability"

# The body binding, judged by an implementation this project did not write.
# The capability's fourth field is the SHA-256 the coordinator will compare
# against the bytes that arrive; if it did not match the bytes that were
# actually posted, one ticket plus a rewritten body would be a free pass.
CAP_BODY_HASH=$(printf '%s' "$CAP" | awk -F. '{print $5}')
WIRE_BODY_HASH=$(sha256_hex < "$REC/recorded/body.bin")
[ -n "$CAP_BODY_HASH" ] && [ ${#CAP_BODY_HASH} = 64 ] \
    || fail "claim 5: the capability has no 64-hex body-hash field: $CAP"
[ "$CAP_BODY_HASH" = "$WIRE_BODY_HASH" ] \
    || fail "claim 5: the capability is bound to $CAP_BODY_HASH but the agent posted bytes hashing to $WIRE_BODY_HASH — the binding names a body that was never sent"
note "claim 5: the capability's body hash equals the system sha256 of the exact bytes posted"

# ===================================================================
# Claim 6 — the coordinator's own verifier spends it exactly once.
# Two processes: the agent minted, this spends, and the only thing they share
# is the 0600 file. --twice requires the replay to be refused as already spent.
# ===================================================================
out=$(./build/admission_verify "$CHAN" "$CAP" "$REC/recorded/body.bin" --twice 2>&1)
rc=$?
[ "$rc" = 0 ] \
    || fail "claim 6: the coordinator's verifier refused a capability the real agent minted: $out"
printf '%s' "$out" | grep -q "^ADMIT_OK" \
    || fail "claim 6: unexpected verifier output: $out"
note "claim 6: spent once by the coordinator's own verifier, and the replay was refused ($out)"

# Bound to the body it names, judged from the other side too: the same
# capability against DIFFERENT bytes must be refused, and for that reason.
printf 'not the body that was posted' > "$REC/other-body.bin"
out=$(./build/admission_verify "$CHAN" "$CAP" "$REC/other-body.bin" 2>&1)
printf '%s' "$out" | grep -q "different request body" \
    || fail "claim 6: the same capability admitted a DIFFERENT body, or was refused for the wrong reason: $out"
note "claim 6: the same capability against other bytes is refused as a body mismatch"

# ===================================================================
# Claim 8 — the verifier discriminates. A capability minted under someone
# else's channel must be refused as a bad signature, not merely "refused":
# a verifier that had collapsed into refusing everything would pass claim 6's
# negative and take this machine off the air for the platform.
# ===================================================================
mkdir -p "$REC/other-state"
arm_coord "$REC/other-state" >/dev/null
out=$(./build/admission_verify "$REC/other-state/coord-admission.key" "$CAP" \
        "$REC/recorded/body.bin" 2>&1)
printf '%s' "$out" | grep -q "signature does not verify" \
    || fail "claim 8: a capability minted under another coordinator's channel was not refused as a bad signature: $out"
note "claim 8: a capability from another channel is refused as a bad signature"

# ===================================================================
# Claim 7 — fail-safe. With no channel key to attach to, the agent must still
# mark the job as platform work and must send no capability at all. This is the
# configuration a coordinator that could not write its state directory produces,
# and it must degrade to the OLD behaviour rather than to no behaviour.
# ===================================================================
cleanup; sleep 0.5
rm -rf "$REC/empty"; mkdir -p "$REC/empty"
python3 scripts/stub_coord_recorder.py --port "$COORD_PORT" --record "$REC/recorded" \
    >"$REC/stub2.log" 2>&1 &
disown 2>/dev/null
for _ in $(seq 1 40); do
    curl -s -m 1 "http://127.0.0.1:$COORD_PORT/idletoken/v1/stats" >/dev/null 2>&1 && break
    sleep 0.25
done
start_agent "" || fail "claim 7: the platform agent did not start without a channel key"
out=$(send_job "$MARKER-nokey")
printf '%s' "$out" | grep -q "HTTP " \
    || fail "claim 7 fixture: seal_infer_job.cjs never posted a job: $out"
[ -s "$REC/recorded/headers.txt" ] || fail "claim 7: the agent forwarded nothing"
grep -qi "^X-IdleToken-Origin: platform" "$REC/recorded/headers.txt" \
    || fail "claim 7: with no channel key the agent ALSO dropped the legacy origin header — a machine that cannot mint would start forwarding buyers' jobs"
grep -qi "^X-IdleToken-Job-Admission:" "$REC/recorded/headers.txt" \
    && fail "claim 7: the agent sent a capability it could not have minted"
grep -q "cannot attach the coordinator's admission channel" "$REC/agent.log" \
    || fail "claim 7: the agent could not mint and did not say so — the failure would present as 'the platform stopped sending me work'"
note "claim 7: with no channel key the agent still marks platform work, sends no capability, and says why"

cleanup
echo "ADMISSION_GATE_OK: the coordinator publishes and rolls its channel correctly; the real agent mints a capability bound (by the system sha256) to the exact bytes it posts; the coordinator's verifier spends it exactly once and refuses replay, body mismatch and a foreign channel; and an agent that cannot mint degrades to the legacy marker out loud"
