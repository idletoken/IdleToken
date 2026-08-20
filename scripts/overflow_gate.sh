#!/usr/bin/env bash
# G_OVERFLOW — overflow routing: when this machine is full, may it borrow
# another one, and under what conditions must it refuse?
#
# Design: docs/overflow-routing-design.md, docs/api-surface.md §5.
# Plan:   docs/overflow-b2b-plan-2026-08.md §2 (the six claims below are its
#         O5 list, in order).
#
# SIX CLAIMS
#   1  platform-dispatched work is NEVER forwarded, and the assertion is driven
#      through the REAL platform agent binary
#   2  a local request on a full machine IS forwarded, sealed, and no plaintext
#      crosses the wire
#   3  overflow without a local API token refuses the START
#   4  stream:true never forwards
#   5  each of the four bad platform keys refuses to enable
#   6  the daily spend cap stops forwarding once it is reached
#
# EVERY CLAIM CARRIES ITS OWN CONTROL, because most of them are of the form
# "nothing happened", and nothing happens by itself very reliably:
#   - "no outbound connection" is judged from the stub platform's connection
#     log, and the log is first PROVEN to record one (claim 2 runs before
#     claims 1, 4 and 6 and leaves a connection behind);
#   - "no plaintext on the wire" is a search over the recorded bytes, and the
#     searcher is first proven to find a marker that IS there;
#   - "the machine is full" is proven by a plain 429 before overflow is
#     switched on at all — a gate that mistakes an idle machine for a full one
#     would pass while testing nothing.
#
# THE FIXTURES are stubs on purpose. scripts/stub_engine_busy.py answers
# correctly but slowly, so the coordinator's slots genuinely fill; a real model
# would make the timings depend on today's GPU. scripts/stub_platform.cjs
# speaks the two platform routes using the REAL gateway's libsodium, so a green
# run means the coordinator's hand-written TweetNaCl + BLAKE2b construction is
# wire-compatible with the platform rather than merely self-consistent.
#
# Contract: last line OVERFLOW_GATE_OK, OVERFLOW_GATE_FAIL: <why>, or
# OVERFLOW_GATE_SKIP: <why>. A SKIP is for a machine that cannot run the
# fixture at all — never for a fixture that armed and then did not hold.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
# shellcheck disable=SC1091
[ -f scripts/testbed.env ] && . ./scripts/testbed.env

API_PORT="${IDLETOKEN_OVERFLOW_GATE_PORT:-18894}"
PLAT_PORT=$((API_PORT + 1))
AGENT_PORT=$((API_PORT + 2))
REC=/tmp/idletoken-overflow-gate
GGUF="${IDLETOKEN_SMOKE_GGUF:-}"
MARKER="OVERFLOWGATEPROMPT7391"

# This gate is about FORWARDING BEHAVIOUR, not about resource planning. The
# engine here is a stub that never loads a weight, but the coordinator still
# reads the GGUF header and budgets for it, so on a small host (met on a 16 GiB
# Mac) the slow-oversubscribe check refuses the start and every claim below goes
# red for a reason that has nothing to do with overflow. The escape hatch exists
# for acceptance scripts exactly like this one; it is set for every coordinator
# this gate starts, including the ones in claims 3 and 5.
export IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE=1

skip() { echo "OVERFLOW_GATE_SKIP: $*"; cleanup; exit 0; }
fail() { echo "OVERFLOW_GATE_FAIL: $*"; cleanup; exit 1; }
note() { echo "  $*"; }

cleanup() {
    pkill -9 -f '[i]dletoken-coord --llama-server-bin /tmp/idletoken-ovf-engine' 2>/dev/null
    pkill -9 -f '[s]tub_engine_busy' 2>/dev/null
    pkill -9 -f '[s]tub_platform.cjs' 2>/dev/null
    pkill -9 -f '[i]dletoken-platform-agent --port '"$AGENT_PORT" 2>/dev/null
}

# Missing prerequisites say nothing about whether overflow obeys its rules.
command -v node >/dev/null 2>&1 || skip "no node — scripts/stub_platform.cjs is the fixture"
command -v python3 >/dev/null 2>&1 || skip "no python3 — scripts/stub_engine_busy.py is the fixture"
command -v curl >/dev/null 2>&1 || skip "no curl"
[ -x ./idletoken-coord ] || skip "no ./idletoken-coord — run 'make coord' first"
# The stubs borrow libsodium from wherever a copy is installed. The default is
# the gateway's own node_modules, which is deliberate -- holding the
# coordinator's hand-written TweetNaCl + BLAKE2b against the library the
# PLATFORM really runs is most of what a green run here means. Point
# IDLETOKEN_SODIUM_DIR at any node package tree that has it.
SODIUM_DIR="${IDLETOKEN_SODIUM_DIR:-$ROOT/platform/packages/gateway}"
node -e "require.resolve('libsodium-wrappers', { paths: ['$SODIUM_DIR'] })" >/dev/null 2>&1 \
    || skip "libsodium-wrappers is not installed under $SODIUM_DIR — install it there (pnpm i in platform/) or set IDLETOKEN_SODIUM_DIR"
[ -n "$GGUF" ] && [ -f "$GGUF" ] \
    || skip "no smoke GGUF: set IDLETOKEN_SMOKE_GGUF in scripts/testbed.env (the coordinator reads the header even with a stub engine)"
[ -x ./build/idletoken-platform-agent ] || {
    make -f Makefile.platform >/dev/null 2>&1 || skip "could not build the platform agent (claim 1 asserts against it)"
}

cleanup; sleep 1
rm -rf "$REC"; mkdir -p "$REC"

cat > /tmp/idletoken-ovf-engine.sh <<EOF
#!/bin/sh
exec python3 $ROOT/scripts/stub_engine_busy.py "\$@" --hold-s 12
EOF
chmod +x /tmp/idletoken-ovf-engine.sh

BODY="{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"$MARKER\"}]}"
STREAM_BODY="{\"model\":\"m\",\"stream\":true,\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"$MARKER\"}]}"

# Claim 6 needs ONE borrowed request to reach the daily ceiling, so the stub
# platform must claim a charge above the coordinator's DEFAULT cap. That number
# used to be written here as a literal 6000 next to a comment naming the 5000 it
# was derived from; when the default was raised to 50000 (pricing decision D2)
# the gate went red for the right reason but for a wrong cause -- the ceiling was
# fine, the fixture had simply stopped reaching it. Read the default from the
# header instead, so raising it again cannot silently disarm this claim.
CAP_MILLI=$(awk '$1=="#define" && $2=="IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI" {print $3}' \
            include/idletoken_overflow.h)
case "$CAP_MILLI" in
    ''|*[!0-9]*) fail "could not read IDLETOKEN_OVF_DEFAULT_DAILY_CAP_MILLI from include/idletoken_overflow.h (got \"$CAP_MILLI\") — claim 6 cannot be armed against a cap it does not know" ;;
esac
CHARGE_MILLI=$((CAP_MILLI + 1000))

# The stale-key trap, met once and worth naming: the verify key file is written
# by the stub at boot, so a start_platform that only waits for the file to EXIST
# returns instantly on the PREVIOUS run's file. The coordinator then pins one
# signer while the stub signs with another, every borrow fails verification, and
# the gate reports a broken invariant when the truth is a broken fixture. Delete
# it first, and wait for the port to answer as well as for the file to appear.
start_platform() {  # start_platform <mode> [charge_milli]
    rm -f "$REC/verify_key.txt"
    node scripts/stub_platform.cjs --port "$PLAT_PORT" --record "$REC" \
        --mode "$1" --charge-milli "${2:-3}" --sodium-from "$SODIUM_DIR" \
        >"$REC/platform.log" 2>&1 &
    disown 2>/dev/null
    for _ in $(seq 1 40); do
        if [ -s "$REC/verify_key.txt" ] &&
           curl -s -m 2 "http://127.0.0.1:$PLAT_PORT/idletoken/v1/platform-key" \
                | grep -q '"pubkey"'; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

start_coord() {  # start_coord <extra args...>; overflow flags come from the caller
    IDLETOKEN_LLAMA_SLOTS=1 IDLETOKEN_PLATFORM_VERIFY_KEY="$(cat "$REC/verify_key.txt")" \
    ./idletoken-coord --llama-server-bin /tmp/idletoken-ovf-engine.sh \
        --llama-gguf "$GGUF" --http --api-bind "127.0.0.1:$API_PORT" \
        --model-id qwen3.5-0.8b --ctx-size 4096 --api-token gatetok \
        "$@" >"$REC/coord.log" 2>&1 &
    disown 2>/dev/null
    for _ in $(seq 1 60); do
        curl -s -m 3 "http://127.0.0.1:$API_PORT/health" 2>/dev/null \
            | grep -q '"engine_state":"ready"' && return 0
        # The other way a start ends: the scheduler refuses the host outright
        # and the process is gone. The oversubscribe branch is disarmed above,
        # but the hard-need branch (KV + engine overhead do not fit) has no
        # escape hatch and should not have one: a host with less free memory
        # than one KV cache cannot run the fixture, which is the SKIP case, not
        # a broken invariant. Met on a busy 16 GiB laptop, where usable memory
        # had fallen to ~1 GiB.
        #
        # Judged from the log line, not from `kill -0` on the child: the start
        # is disowned, so bash never reaps it and a dead coordinator lingers as
        # a zombie whose pid still answers kill -0.
        grep -q "^idletoken-coord: refuse:" "$REC/coord.log" 2>/dev/null && return 2
        sleep 1
    done
    return 1
}

# Wraps start_coord so a resource refusal reports itself as a SKIP, quoting the
# coordinator's own sentence — never a bare "did not come up".
start_coord_or_skip() {
    start_coord "$@"
    case $? in
        0) return 0 ;;
        2) skip "this host cannot run the fixture: $(grep -m1 '^idletoken-coord: refuse:' "$REC/coord.log" | sed 's/^idletoken-coord: //')" ;;
        *) return 1 ;;
    esac
}

# Occupy every slot and the queue behind it, then leave them occupied. Returns
# once the machine really is full — proven by a request being refused, not by a
# sleep.
fill_machine() {
    for _ in 1 2; do
        curl -s -m 40 -H 'Authorization: Bearer gatetok' -H 'Content-Type: application/json' \
             -d "$BODY" "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1 &
    done
    disown -a 2>/dev/null
    sleep 2
}

post_chat() {  # post_chat <body> [extra curl args...]; echoes "<code> <body>"
    local body="$1"; shift
    curl -s -m 25 -w '\n%{http_code}' -H 'Authorization: Bearer gatetok' \
         -H 'Content-Type: application/json' "$@" -d "$body" \
         "http://127.0.0.1:$API_PORT/v1/chat/completions" 2>/dev/null
}

conns() { wc -l < "$REC/conn.log" 2>/dev/null | tr -d ' '; }

echo "======================================================"
echo " G_OVERFLOW — overflow routing"
echo "======================================================"

# ===================================================================
# Claim 3 — no local API token, no overflow. Stands alone: it never gets
# as far as needing an engine.
# ===================================================================
start_platform good || fail "the stub platform did not come up"
VK=$(cat "$REC/verify_key.txt")

out=$(IDLETOKEN_PLATFORM_VERIFY_KEY="$VK" ./idletoken-coord \
        --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate \
        --num-workers 0 --n-predict 0 2>&1)
rc=$?
case "$rc:$out" in
    2:*"local API token"*) note "claim 3: no --api-token -> refuse to start (exit 2)" ;;
    *) fail "claim 3: overflow started without an --api-token (exit $rc): $(printf '%s' "$out" | tail -1)" ;;
esac
# Control: the SAME command line with a token must get past this check, or the
# assertion above would also pass on a coordinator that refuses everything.
out=$(IDLETOKEN_PLATFORM_VERIFY_KEY="$VK" ./idletoken-coord \
        --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate \
        --api-token gatetok --num-workers 0 --n-predict 0 2>&1)
printf '%s' "$out" | grep -q "overflow: on" \
    || fail "claim 3 control: overflow would not switch on even WITH a token, so the refusal above proves nothing"
note "claim 3 control: with a token it switches on"

# ===================================================================
# Claim 5 — the four bad platform keys. Each must refuse to enable, and each
# for its own reason: "refused" alone would still hold on a verifier that had
# degenerated into refusing everything, which takes overflow off the air just
# as silently as accepting everything lets prompts out.
# ===================================================================
for mode in swapped expired nodomain unsigned; do
    cleanup; sleep 0.5
    start_platform "$mode" || fail "the stub platform ($mode) did not come up"
    out=$(IDLETOKEN_PLATFORM_VERIFY_KEY="$(cat "$REC/verify_key.txt")" ./idletoken-coord \
            --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate \
            --api-token gatetok --num-workers 0 --n-predict 0 2>&1)
    rc=$?
    printf '%s' "$out" | grep -q "overflow: on" \
        && fail "claim 5 ($mode): overflow switched on against a bad platform key"
    [ "$rc" = 2 ] \
        || fail "claim 5 ($mode): a bad platform key did not fail the start (exit $rc)"
    case "$mode" in
        swapped|nodomain) want="does not verify" ;;
        expired)          want="expired" ;;
        unsigned)         want="no signature" ;;
    esac
    printf '%s' "$out" | grep -q "$want" \
        || fail "claim 5 ($mode): refused, but not for the right reason (wanted \"$want\"): $(printf '%s' "$out" | tail -1)"
    note "claim 5: $mode -> refused ($want)"
done

# ===================================================================
# From here on: one full machine, one good platform.
# ===================================================================
cleanup; sleep 1
rm -rf "$REC"; mkdir -p "$REC"
start_platform good "$CHARGE_MILLI" || fail "the stub platform did not come up"
# charge-milli is one credit above the coordinator's default cap (read from the
# header above), so ONE borrowed request is enough to reach the ceiling — which
# is what claim 6 needs.

# --- control: the machine really does fill up, before overflow is involved ---
start_coord_or_skip || fail "the coordinator never became ready (see $REC/coord.log)"
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "control: a full machine answered $code, not 429 — the fixture never filled, so nothing below would be tested"
note "control: a full machine refuses with 429 (overflow off)"

# ===================================================================
# Claim 2 — a local request on a full machine is forwarded, sealed.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"; : > "$REC/wire.log"; rm -f "$REC/opened.log"
start_platform good "$CHARGE_MILLI" || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready with overflow on (see $REC/coord.log)"
base_conns=$(conns)   # the platform-key fetch at start-up
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] \
    || fail "claim 2: a local request on a full machine got $code, expected a borrowed 200"
printf '%s' "$got" | grep -q "borrowed-answer" \
    || fail "claim 2: the 200 did not carry the platform's answer"
[ -s "$REC/opened.log" ] \
    || fail "claim 2: the platform never opened a sealed request"
grep -q "\"api_key\":\"sk-gate-key\"" "$REC/opened.log" \
    || fail "claim 2: the account key did not arrive inside the envelope"
grep -q "$MARKER" "$REC/opened.log" \
    || fail "claim 2: the prompt did not arrive inside the envelope"
note "claim 2: forwarded, sealed, opened by the platform, answered 200"

# The connection log has now recorded real connections. Everything below that
# says "no outbound connection" is judged against this same file, so this is
# where it is proven to work at all.
after_conns=$(conns)
[ "$after_conns" -gt "$base_conns" ] \
    || fail "claim 2 control: the connection log did not record the connection that just happened — it cannot prove an absence either"
note "claim 2 control: the connection log records connections ($base_conns -> $after_conns)"

# ...and the plaintext search, proven on a marker that IS present.
grep -q "$MARKER" "$REC/wire.log" \
    && fail "claim 2: the prompt crossed the wire in the clear"
curl -s -m 5 -X POST -H 'Content-Type: application/json' \
     -d "{\"plaintext_control\":\"$MARKER\"}" \
     "http://127.0.0.1:$PLAT_PORT/idletoken/v1/sealed/chat" >/dev/null 2>&1
grep -q "$MARKER" "$REC/wire.log" \
    || fail "claim 2 control: a deliberately plaintext body was NOT found in the wire log — the search proves nothing"
# Remove the control's own plaintext so later greps judge the coordinator only.
grep -v "plaintext_control" "$REC/wire.log" > "$REC/wire.clean" && mv "$REC/wire.clean" "$REC/wire.log"
note "claim 2 control: the wire search finds a planted marker, and found none from the coordinator"

# ===================================================================
# Claim 6 — the daily cap. The borrow above charged $CHARGE_MILLI milli-credits
# against the coordinator's default ceiling, so the next one must not go out.
# ===================================================================
before=$(conns)
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 6: past the daily cap the coordinator answered $code, not 429"
[ "$(conns)" = "$before" ] \
    || fail "claim 6: past the daily cap the coordinator still dialled the platform"
grep -q "daily spend cap" "$REC/coord.log" \
    || fail "claim 6: refused, but the log does not say it was the cap"
note "claim 6: past the cap -> 429, and no outbound connection"

# ===================================================================
# Claim 4 — stream:true never forwards. Restarted so the daily cap is not
# the reason it refuses; that would make this claim pass for claim 6's
# reason and stay green if streaming forwarding were added tomorrow.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready (see $REC/coord.log)"
before=$(conns)
fill_machine
got=$(post_chat "$STREAM_BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 4: a streaming request on a full machine got $code, not 429"
[ "$(conns)" = "$before" ] \
    || fail "claim 4: a streaming request caused an outbound connection"
grep -q "stream:true is refused locally" "$REC/coord.log" \
    || fail "claim 4: refused, but not for being a stream"
# Control: the same coordinator, same moment, non-streaming -> it DOES borrow.
# Without this, claim 4 would also pass on a machine where overflow was simply
# not working.
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] \
    || fail "claim 4 control: the same coordinator would not borrow for a NON-streaming request either ($code) — the refusal above proves nothing about streaming"
note "claim 4: stream:true -> 429 with no outbound; the same coordinator borrows for a non-stream"

# ===================================================================
# Claim 1 — the ironclad rule, asserted through the REAL agent binary.
#
# The agent opens a sealed job and forwards it to the coordinator over
# loopback with X-IdleToken-Origin: platform. If that header is ever dropped
# the coordinator starts forwarding other people's work, silently. So the
# assertion is driven through the agent's own product, not through the
# coordinator's predicate.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready (see $REC/coord.log)"

rm -f /tmp/idletoken-ovf-agent.key
# --coord-token because RULE 2 forces this coordinator to have one, and an
# agent that cannot present it gets 401 before the admission gate is ever
# reached -- which would make claim 1 pass for entirely the wrong reason.
./build/idletoken-platform-agent --port "$AGENT_PORT" \
    --coord "http://127.0.0.1:$API_PORT" --coord-token gatetok \
    --key-file /tmp/idletoken-ovf-agent.key \
    >"$REC/agent.log" 2>&1 &
for _ in $(seq 1 30); do grep -q "pubkey (b64)" "$REC/agent.log" && break; sleep 0.5; done
AGENT_PK=$(grep "pubkey (b64)" "$REC/agent.log" | sed 's/.*: *//' | tr -d ' \r')
[ -n "$AGENT_PK" ] || fail "claim 1: the platform agent did not report a public key"

before=$(conns)
fill_machine
# --sodium-from, like every other node fixture here: the helper's own default is
# this repo's gateway package, which has no node_modules in a fresh worktree. It
# then dies on require() before sending anything, and claim 1's two log
# assertions below fail exactly as they would if the agent had dropped the origin
# header — a fixture that never armed, reported as the one rule that may not break.
agent_out=$(node scripts/seal_infer_job.cjs --agent "127.0.0.1:$AGENT_PORT" \
                --pubkey "$AGENT_PK" --prompt "$MARKER-platform" \
                --sodium-from "$SODIUM_DIR" 2>&1)
printf '%s' "$agent_out" | grep -q "HTTP 0" \
    && fail "claim 1: could not reach the platform agent at all: $agent_out"
# And prove the job was really posted, before reading anything into its absence.
printf '%s' "$agent_out" | grep -q "HTTP " \
    || fail "claim 1 fixture: seal_infer_job.cjs never posted a job, so nothing below is an assertion about the agent: $agent_out"
# The agent turns the coordinator's 429 into a sealed error, so what matters is
# not its status code but that the coordinator did NOT dial out.
[ "$(conns)" = "$before" ] \
    || fail "claim 1: a PLATFORM-DISPATCHED job was forwarded out of this machine — the one rule that may not break"
grep -q "platform work is never forwarded" "$REC/coord.log" \
    || fail "claim 1: the coordinator did not record refusing to forward platform work (did the agent send the origin header?)"
grep -q "origin=platform" "$REC/coord.log" \
    || fail "claim 1: the coordinator never saw a platform-origin request — the agent's header is missing, and the whole rule rests on it"
note "claim 1: the real agent's job was recognised as platform work and never forwarded"

# Control: through the SAME full coordinator, a LOCAL request does go out. If
# it did not, claim 1 would be passing because overflow was dead, not because
# the rule held.
before=$(conns)
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 1 control: a local request on the same full coordinator did not borrow ($code) — the absence above proves nothing"
note "claim 1 control: a local request through the same coordinator DID go out"

cleanup
echo "OVERFLOW_GATE_OK: platform work never forwarded (via the real agent); local overflow sealed with no plaintext; fail-closed on token and on four bad keys; streams and the daily cap refuse without dialling out"
