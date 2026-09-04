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
#   3  overflow without a local API token remains supported on loopback
#   4  stream:true forwards only after a complete sealed reply and is re-emitted
#      as a complete local SSE sequence
#   5  each of the four bad platform keys refuses to enable
#   6  an explicitly configured daily spend cap stops forwarding once reached
#
# Claims 7-15 were added after that list and are documented where they run:
# 7 origin policy, 8 admission capabilities, 9 the hop budget, 10 an in-flight
# platform job, 11 the posture endpoint, 12 which success codes count as a
# borrow, 13 unmodified third-party API traffic while --shared is on, 14
# tool-calling request/response fidelity, and 15 downstream cancellation.
#
# EVERY CLAIM CARRIES ITS OWN CONTROL, because most of them are of the form
# "nothing happened", and nothing happens by itself very reliably:
#   - "no outbound connection" is judged from the stub platform's connection
#     log, and the log is first PROVEN to record one (claim 2 runs before
#     claims 1 and 6 and leaves a connection behind);
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
# engine here is a stub that never allocates GPU memory, but the coordinator
# still reads the GGUF header and performs the production GPU_ONLY hard gate.
# Testing against today's free Metal memory made the fixture fail or pass based
# on unrelated desktop load. Use the coordinator's loud TEST-ONLY budget input
# for every process this gate starts; the gate is about HTTP admission and
# forwarding, not whether this laptop currently has a model-sized free pool.
export IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE=1
export IDLETOKEN_TEST_USABLE_BYTES=17179869184

skip() { echo "OVERFLOW_GATE_SKIP: $*"; cleanup; exit 0; }
fail() { echo "OVERFLOW_GATE_FAIL: $*"; cleanup; exit 1; }
note() { echo "  $*"; }

cleanup() {
    pkill -9 -f '[i]dletoken-coord --llama-server-bin /tmp/idletoken-ovf-engine' 2>/dev/null
    pkill -9 -f '[s]tub_engine_busy' 2>/dev/null
    pkill -9 -f '[s]tub_platform.cjs' 2>/dev/null
    pkill -9 -f '[i]dletoken-platform-agent --port '"$AGENT_PORT" 2>/dev/null
    FILL_PID=""
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
ANTH_STREAM_BODY="{\"model\":\"m\",\"stream\":true,\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"$MARKER\"}]}"
TOOL_BODY="{\"model\":\"m\",\"stream\":true,\"max_tokens\":64,\"messages\":[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_old\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_old\",\"content\":\"old result\"},{\"role\":\"user\",\"content\":\"$MARKER-tool\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"description\":\"read a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}],\"tool_choice\":\"required\"}"

# Claim 6 verifies the optional operator stop-loss, not the product default.
# The default is deliberately uncapped: the account balance is the spend gate.
CAP_MILLI=50000
CHARGE_MILLI=$((CAP_MILLI + 1000))

# The stale-key trap, met once and worth naming: the verify key file is written
# by the stub at boot, so a start_platform that only waits for the file to EXIST
# returns instantly on the PREVIOUS run's file. The coordinator then pins one
# signer while the stub signs with another, every borrow fails verification, and
# the gate reports a broken invariant when the truth is a broken fixture. Delete
# it first, and wait for the port to answer as well as for the file to appear.
start_platform() {  # start_platform <mode> [charge_milli] [chat_status]
    rm -f "$REC/verify_key.txt"
    node scripts/stub_platform.cjs --port "$PLAT_PORT" --record "$REC" \
        --mode "$1" --charge-milli "${2:-3}" --chat-status "${3:-200}" \
        --sodium-from "$SODIUM_DIR" \
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
    # The admission channel and the local-origin marker are written into the
    # gate's own recording directory, never into the operator's ~/.idletoken:
    # this script starts and kills a dozen coordinators, and one of them
    # clobbering the key a REAL coordinator on this machine published would
    # break that machine's agent for reasons nobody would connect to running a
    # gate. IDLETOKEN_OVERFLOW_ORIGIN_POLICY is passed through from the caller's
    # environment so a claim can pick the policy it is judging.
    mkdir -p "$REC/state"
    IDLETOKEN_STATE_DIR="$REC/state" \
    IDLETOKEN_PLATFORM_VERIFY_KEY="$(cat "$REC/verify_key.txt")" \
    ./idletoken-coord --llama-server-bin /tmp/idletoken-ovf-engine.sh \
        --llama-gguf "$GGUF" --http --api-bind "127.0.0.1:$API_PORT" \
        --model-id qwen3.5-0.8b --ctx-size 131072 --api-token gatetok \
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

# Occupy the one local slot and leave it occupied. The local queue is
# deliberately zero-deep now: the next request is the overflow candidate, so a
# second filler would borrow and contaminate the outbound-connection oracle.
fill_machine() {
    if [ -n "${FILL_PID:-}" ] && kill -0 "$FILL_PID" 2>/dev/null; then
        wait "$FILL_PID" 2>/dev/null || true
    fi
    curl -s -m 40 -H 'Authorization: Bearer gatetok' -H 'Content-Type: application/json' \
         -d "$BODY" "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1 &
    FILL_PID=$!
    sleep 2
}

post_chat() {  # post_chat <body> [extra curl args...]; echoes "<code> <body>"
    local body="$1"; shift
    curl -s -m 25 -w '\n%{http_code}' -H 'Authorization: Bearer gatetok' \
         -H 'Content-Type: application/json' "$@" -d "$body" \
         "http://127.0.0.1:$API_PORT/v1/chat/completions" 2>/dev/null
}

post_anthropic() {
    local body="$1"
    curl -s -m 25 -w '\n%{http_code}' -H 'Authorization: Bearer gatetok' \
         -H 'Content-Type: application/json' -d "$body" \
         "http://127.0.0.1:$API_PORT/v1/messages" 2>/dev/null
}

conns() { wc -l < "$REC/conn.log" 2>/dev/null | tr -d ' '; }

# --- helpers for the admission-capability claims (7-10) ---------------------
# sha256sum on Linux, shasum on macOS. Written once: a claim that silently got
# an empty hash would mint a capability bound to nothing and then "pass".
sha256_hex() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1
    else shasum -a 256 | cut -d' ' -f1; fi
}

channel_key() { cat "$REC/state/coord-admission.key" 2>/dev/null | tr -d ' \r\n'; }
local_marker() { cat "$REC/state/coord-local-origin.key" 2>/dev/null | tr -d ' \r\n'; }

# mint_cap <job_id> <body> -> the capability, or "" on refusal.
mint_cap() {
    local job="$1" body="$2" h
    h=$(printf '%s' "$body" | sha256_hex)
    curl -s -m 5 -H "Authorization: Bearer $(channel_key)" \
         -H 'Content-Type: application/json' \
         -d "{\"job_id\":\"$job\",\"body_sha256\":\"$h\"}" \
         "http://127.0.0.1:$API_PORT$ADMIT_PATH" 2>/dev/null |
        sed -n 's/.*"capability":"\([^"]*\)".*/\1/p'
}
ADMIT_PATH=/idletoken/v1/platform/admit

echo "======================================================"
echo " G_OVERFLOW — overflow routing"
echo "======================================================"

# ===================================================================
# Claim 3 — a machine with NO local API token can still switch sharing on.
#
# ⚠ INVERTED 2026-08-21 (e78269d). This claim used to be the opposite: "no
# local API token, no overflow". The token was retired because it protected
# nothing — the API has answered 127.0.0.1 only since 2026-08-16, and a program
# on this machine reads the token out of the same settings file as the overflow
# key it would rather take. Refusing without one also meant every install that
# predated token minting could not switch sharing on AT ALL.
#
# So the claim is now positive, matching the RULE 2 assertion in
# src/coord/overflow.c that was inverted the same day: "no token" is a
# SUPPORTED configuration, and a check that says so is what stops it being
# re-forbidden by accident. Keeping the old spelling here is how the ladder
# ended up red against a requirement that no longer existed (found 2026-08-23,
# cutting 0.1.19 — the code and its unit test had been inverted for two days
# while this file and G_LOCAL_TOKEN still asserted the retired contract).
#
# Stands alone: it never gets as far as needing an engine.
# ===================================================================
start_platform good || fail "the stub platform did not come up"
VK=$(cat "$REC/verify_key.txt")

# How these runs terminate. `--model <id>` with no engine flags makes the
# coordinator configure overflow, say so, and then exit on the missing
# --llama-server-bin/--llama-gguf — 12ms, no engine, no join wait. The old
# spelling used `--num-workers 0` for the same purpose, but the coordinator now
# rejects 0 during argument validation, BEFORE overflow is configured: claim 3
# was failing on "--num-workers must be 1..16" and its control had the identical
# bug, so neither half ever reached the logic it was written to judge. Do not
# "fix" this by passing --num-workers 1 instead — that gets past validation and
# then blocks for the 180s join wait, three times over.
# This is the same probe shape scripts/build_client_release.bat uses to check
# the pinned verify key; one idiom, already proven on three platforms.
ovf_probe() {   # ovf_probe <extra args...>; echoes the coordinator's output
    IDLETOKEN_PLATFORM_VERIFY_KEY="$VK" ./idletoken-coord \
        --overflow-key sk-gate --model probe-sentinel "$@" 2>&1
}

out=$(ovf_probe --overflow-url "http://127.0.0.1:$PLAT_PORT")
printf '%s' "$out" | grep -q "overflow: on" \
    || fail "claim 3: overflow refused to switch on without an --api-token, but no token is a supported configuration since 2026-08-21: $(printf '%s' "$out" | tail -1)"
printf '%s' "$out" | grep -q "local API token" \
    && fail "claim 3: the coordinator still cites a 'local API token' as a reason — the retired requirement is back"
note "claim 3: no --api-token -> overflow still switches on"

# Control. "No refusal was seen" is worthless unless this harness can see a
# refusal at all, so make it refuse for a reason that IS still live (an empty
# platform URL) on an otherwise identical command line.
out=$(ovf_probe --overflow-url "")
printf '%s' "$out" | grep -q "platform URL" \
    || fail "claim 3 control: an empty --overflow-url did not produce the 'platform URL' refusal, so the assertion above cannot tell 'allowed' from 'never evaluated': $(printf '%s' "$out" | tail -1)"
note "claim 3 control: a still-live refusal (empty platform URL) is visible to this harness"

# The token is not gone as a FEATURE, only as a requirement: set one and the
# coordinator must still switch on and enforce it.
out=$(ovf_probe --overflow-url "http://127.0.0.1:$PLAT_PORT" --api-token gatetok)
printf '%s' "$out" | grep -q "overflow: on" \
    || fail "claim 3: overflow would not switch on WITH a token either — an operator who sets one must still be served"
note "claim 3: with a token it still switches on"

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
# charge-milli is one credit above the explicit test cap, so ONE borrowed
# request is enough to reach the ceiling — which is what claim 6 needs.

# --- control: the machine really does fill up, before overflow is involved ---
start_coord_or_skip || fail "the coordinator never became ready (see $REC/coord.log)"
stats=$(curl -s -m 3 "http://127.0.0.1:$API_PORT/idletoken/v1/stats")
printf '%s' "$stats" | grep -q '"seq_slots":1' \
    || fail "control: default local engine is not single-slot: $stats"
printf '%s' "$stats" | grep -q '"concurrency":1' \
    || fail "control: default local concurrency is not 1: $stats"
printf '%s' "$stats" | grep -q '"queue_cap":0' \
    || fail "control: local inference queue is not zero-deep: $stats"
grep -q 'resources could hold 4' "$REC/coord.log" \
    || fail "control: the fixture did not prove that product policy overrode a resource-derived multi-slot result"
note "control: resources permit 4 slots, product policy starts 1 with queue 0"
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "control: a full machine answered $code, not 429 — the fixture never filled, so nothing below would be tested"
note "control: a full machine refuses with 429 (overflow off)"

# ===================================================================
# Claim 15 — a departed local API client releases both the coordinator slot
# and the engine request. Non-stream llama-server sends no response head until
# generation is complete, so observing disconnects only after open() returns
# leaves the entire expensive interval uncovered.
#
# The control immediately above proved this harness sees a genuinely occupied
# slot as 429. Here curl deliberately leaves after one second; a second request
# must then be ADMITTED (it times out waiting for the slow fixture, HTTP 000),
# not immediately refused as 429. The fixture itself watches its upstream EOF,
# so this also exercises coordinator-close -> engine-cancel rather than merely
# releasing an accounting counter.
# ===================================================================
cleanup; sleep 1
rm -rf "$REC"; mkdir -p "$REC"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip || fail "the coordinator never became ready for cancellation"

first_code=$(curl -s -m 1 -o /dev/null -w '%{http_code}' \
    -H 'Authorization: Bearer gatetok' -H 'Content-Type: application/json' \
    -d "$BODY" "http://127.0.0.1:$API_PORT/v1/chat/completions" 2>/dev/null)
first_rc=$?
[ "$first_rc" = "28" ] && [ "$first_code" = "000" ] \
    || fail "claim 15 setup: the first client did not leave during generation (curl=$first_rc HTTP=$first_code)"
sleep 1

second_code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' \
    -H 'Authorization: Bearer gatetok' -H 'Content-Type: application/json' \
    -d "$BODY" "http://127.0.0.1:$API_PORT/v1/chat/completions" 2>/dev/null)
second_rc=$?
[ "$second_rc" = "28" ] && [ "$second_code" = "000" ] \
    || fail "claim 15: the departed caller still held the only slot (curl=$second_rc HTTP=$second_code)"
sleep 1
cancel_count=$(grep -c 'downstream client disconnected' "$REC/coord.log" 2>/dev/null)
[ "$cancel_count" -ge 2 ] \
    || fail "claim 15: admitted requests did not close their engine links on disconnect (saw $cancel_count cancellation logs)"
note "claim 15: client disconnect -> engine link closed and the single slot is reusable"

# ===================================================================
# Claim 2 — a local request on a full machine is forwarded, sealed.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"; : > "$REC/wire.log"; rm -f "$REC/opened.log"
start_platform good "$CHARGE_MILLI" || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    --overflow-daily-cap "$CAP_MILLI" \
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
grep -q '"quant":"Q4_K_M"' "$REC/opened.log" \
    || fail "claim 2: the coordinator's selected precision did not arrive inside the envelope"
grep -q "$MARKER" "$REC/opened.log" \
    || fail "claim 2: the prompt did not arrive inside the envelope"
note "claim 2: forwarded with its exact precision, sealed, opened by the platform, answered 200"

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
# Claim 6 — an explicitly configured daily cap. The borrow above charged
# $CHARGE_MILLI milli-credits against the test ceiling, so the next one must not
# go out. The shipped default has no additional cap.
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
# Claim 4 — stream:true borrows safely. The cloud exchange itself is
# non-streaming and sealed; only after the complete reply arrives may the
# coordinator open a local SSE response. That removes the old half-stream
# failure ambiguity while letting the desktop chat use overflow.
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
[ "$code" = "200" ] \
    || fail "claim 4: a streaming request on a full machine got $code, not 200"
[ "$(conns)" -gt "$before" ] \
    || fail "claim 4: a streaming request did not reach overflow"
printf '%s' "$got" | grep -q 'chat.completion.chunk' \
    || fail "claim 4: borrowed stream is not an OpenAI chunk sequence"
printf '%s' "$got" | grep -q '\[DONE\]' \
    || fail "claim 4: borrowed stream has no complete [DONE] trailer"
note "claim 4: OpenAI stream:true borrowed after a complete reply and returned a complete SSE sequence"

# The same rule must cover the Anthropic face used by Claude Code. It has a
# different event grammar, so an OpenAI-only assertion cannot stand in for it.
fill_machine
before=$(conns)
got=$(post_anthropic "$ANTH_STREAM_BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] \
    || fail "claim 4: an Anthropic streaming request on a full machine got $code, not 200"
[ "$(conns)" -gt "$before" ] \
    || fail "claim 4: an Anthropic streaming request did not reach overflow"
printf '%s' "$got" | grep -q 'event: message_start' \
    || fail "claim 4: borrowed Anthropic stream has no message_start"
printf '%s' "$got" | grep -q 'event: message_stop' \
    || fail "claim 4: borrowed Anthropic stream has no message_stop"
note "claim 4: Anthropic stream:true returned a complete message_start…message_stop sequence"

# ===================================================================
# Claim 14 — tool-bearing overflow is the exact OpenAI contract, not a
# text-only approximation. OpenCode/Nimbalyst rely on all four pieces here:
# tools, tool_choice, prior assistant/tool messages, and structured tool_calls
# in the streamed answer. Dropping any one produces raw markup or a stuck turn.
# ===================================================================
fill_machine
got=$(post_chat "$TOOL_BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] \
    || fail "claim 14: a tool-bearing request on a full machine got $code, not 200"
printf '%s' "$got" | grep -q '\"tool_calls\"' \
    || fail "claim 14: borrowed OpenAI stream lost structured tool_calls"
printf '%s' "$got" | grep -q '\"name\":\"read_file\"' \
    || fail "claim 14: borrowed tool call lost its function name"
printf '%s' "$got" | grep -q '\"finish_reason\":\"tool_calls\"' \
    || fail "claim 14: borrowed tool stream did not finish as tool_calls"
tail -1 "$REC/opened.log" | grep -q '\"tool_choice\":\"required\"' \
    || fail "claim 14: tool_choice did not reach the sealed platform request"
tail -1 "$REC/opened.log" | grep -q '\"tool_call_id\":\"call_old\"' \
    || fail "claim 14: prior tool-call history was flattened during overflow"
note "claim 14: tools, required choice, call history and structured tool response all survived overflow"

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
IDLETOKEN_STATE_DIR="$REC/state" ./build/idletoken-platform-agent --port "$AGENT_PORT" \
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

# And the capability the agent minted was really SPENT, not merely sent. Claim 1
# above would hold on the legacy header alone; this is what says the mechanism
# that a modified agent cannot fake actually ran.
grep -q "capability spent" "$REC/coord.log" \
    || fail "claim 1: the coordinator never spent an admission capability — the agent fell back to the legacy header, so PROV-28's proven path is untested (see $REC/agent.log)"
note "claim 1: ...and the capability the real agent minted was spent by the coordinator"

# ===================================================================
# Claim 7 — the STRICT operator opt-in. On a machine that serves the platform,
# strict makes a request with NO origin marker non-forwardable, because that is
# exactly the shape a dispatched job takes once its agent deletes every
# provenance signal. This is deliberately NOT the product default: claim 13
# proves the compatible-API default separately.
#
# THREE PARTS, and the middle one is the whole reason this claim is credible:
#
#   7a  the strict opt-in:  unmarked + strict     -> 429, nothing dialled out
#   7b  the attack oracle:  unmarked + legacy     -> 200, dialled out
#   7c  the non-regression: marked   + strict     -> 200, dialled out
#
# Without 7b, 7a proves only that something refused; a coordinator with overflow
# quietly broken would pass it. Without 7c, the "fix" could be "never forward
# anything", which switches the feature off for the user who paid for it. The
# attack is demonstrated on the SAME BINARY that stops it, via the documented
# escape hatch, because an oracle that needs an older build is one nobody runs.
#
# --shared is deliberately NOT used to make this machine a provider. Spending
# one real capability does it, which also exercises the rule that a machine
# serving platform work is a provider whether or not the flag was typed.
# ===================================================================
ovf_become_provider() {   # spend one capability so "no marker" stops being unambiguous
    local cap
    cap=$(mint_cap "gate-provider" "$BODY")
    [ -n "$cap" ] || return 1
    post_chat "$BODY" -H "$ADMIT_HDR: $cap" >/dev/null 2>&1
    grep -q "capability spent" "$REC/coord.log"
}
ADMIT_HDR=X-IdleToken-Job-Admission
LOCAL_HDR=X-IdleToken-Local-Origin
HOPS_HDR=X-IdleToken-Hops

cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
IDLETOKEN_OVERFLOW_ORIGIN_POLICY=strict \
    start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready under the strict origin policy (see $REC/coord.log)"
grep -q "origin policy OVERRIDDEN to 'strict'" "$REC/coord.log" \
    || fail "claim 7: the coordinator did not adopt the explicit strict policy, so 7a would be testing the compatible-client default"
ovf_become_provider \
    || fail "claim 7 fixture: could not spend a capability, so this coordinator never became a provider and an unmarked request is legitimately local — nothing below would be an assertion"
note "claim 7 fixture: one capability spent; this coordinator now serves platform work"

# --- 7a: the fix ---------------------------------------------------------
before=$(conns)
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 7a: an UNATTRIBUTED request on a machine that serves the platform was answered $code, not 429 — this is the PROV-28 shape and it must not be forwarded"
[ "$(conns)" = "$before" ] \
    || fail "claim 7a: an UNATTRIBUTED request was forwarded off this machine — a stripped-header platform job would be charged twice and shown to one more stranger"
grep -q "unattributed request on a machine that serves the platform" "$REC/coord.log" \
    || fail "claim 7a: refused, but not for the origin reason — the log must name why, or the next person will read it as an ordinary busy 429"
note "claim 7a: explicit strict opt-in + unattributed -> 429, nothing dialled out, reason named"

# --- 7c: the non-regression control --------------------------------------
# The SAME coordinator, the SAME body, one extra header: the local-origin
# marker it published. If this does not go out, 7a is passing because overflow
# is dead rather than because the origin rule held.
before=$(conns)
fill_machine
got=$(post_chat "$BODY" -H "$LOCAL_HDR: $(local_marker)")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 7c: an ATTRIBUTED local request on the same strict coordinator did not borrow ($code) — the refusal in 7a is a dead feature, not an origin check"
note "claim 7c: the same request WITH the local-origin marker still borrows under strict"

# --- 7b: the attack oracle ------------------------------------------------
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
IDLETOKEN_OVERFLOW_ORIGIN_POLICY=legacy \
    start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready under the legacy origin policy (see $REC/coord.log)"
ovf_become_provider \
    || fail "claim 7b fixture: could not spend a capability under the legacy policy"
before=$(conns)
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 7b: the legacy policy did NOT reproduce the pre-hardening behaviour ($code) — 7a's refusal cannot be attributed to the fix, because this harness cannot show the machine ever forwarded an unmarked request"
note "claim 7b attack oracle: under the legacy policy the same unmarked request IS forwarded — the hole is real and 7a closes it"

# ===================================================================
# Claim 8 — the capability's lifecycle over real HTTP. Each refusal must carry
# its OWN reason: a verifier that answered "no" to everything would pass a test
# that only asserted "rejected", and would silently stop this machine serving
# the platform at all.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready (see $REC/coord.log)"

CAP=$(mint_cap "gate-job-8" "$BODY")
[ -n "$CAP" ] || fail "claim 8: the coordinator would not mint a capability against its own channel key"

# Minting requires the channel key and nothing else stands in for it.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H 'Authorization: Bearer not-the-key' \
        -H 'Content-Type: application/json' -d '{"job_id":"x","body_sha256":"00"}' \
        "http://127.0.0.1:$API_PORT$ADMIT_PATH")
[ "$code" = "401" ] \
    || fail "claim 8: minting without the channel key answered $code, not 401 — anything on this machine could mint proofs of platform origin"
note "claim 8: minting needs the channel key (401 without it)"

# The capability admits its body once, and the coordinator says so.
before=$(conns)
fill_machine
got=$(post_chat "$BODY" -H "$ADMIT_HDR: $CAP")
[ "$(conns)" = "$before" ] \
    || fail "claim 8: a request admitted by a capability was forwarded — proven platform work must never leave this machine"
grep -q "platform job gate-job-8 admitted" "$REC/coord.log" \
    || fail "claim 8: the coordinator did not record spending the capability, so the assertion above may be about an ordinary refusal"
note "claim 8: a capability admits its request once, and that request is not forwarded"

# Replay, body mismatch and garbage: each refused, each named, each a 403 that
# stops the request rather than demoting it to an ordinary one. If a bad
# capability quietly became "local", corrupting one byte of a ticket would be an
# easier way to strip it than deleting the header.
for probe in replay mismatch garbage; do
    case "$probe" in
        replay)   hdr="$CAP";              body="$BODY";        want="already spent" ;;
        mismatch) hdr=$(mint_cap "gate-job-8b" "$BODY"); body="$STREAM_BODY"
                  want="different request body" ;;
        garbage)  hdr="itadm1.x.y.z.w";    body="$BODY";        want="not a v1 admission capability" ;;
    esac
    [ -n "$hdr" ] || fail "claim 8 ($probe): could not build the probe"
    out=$(post_chat "$body" -H "$ADMIT_HDR: $hdr")
    code=$(printf '%s' "$out" | tail -1)
    [ "$code" = "403" ] \
        || fail "claim 8 ($probe): answered $code, not 403 — a capability that does not verify must refuse the request, never be demoted to an unmarked one"
    printf '%s' "$out" | grep -q "$want" \
        || fail "claim 8 ($probe): refused, but not for its own reason (wanted \"$want\"): $(printf '%s' "$out" | head -1)"
    note "claim 8: $probe -> 403 ($want)"
done

# ===================================================================
# Claim 9 — the hop budget (CHAIN-05). RULE 1 ends the loop that goes out
# through the platform and back, but it cannot see a chain that grows one
# machine at a time: from each coordinator's own side, every hop looks like a
# first one. A counter that travels with the request does not depend on anyone's
# honesty about origin.
# ===================================================================
before=$(conns)
fill_machine
got=$(post_chat "$BODY" -H "$HOPS_HDR: $(awk '$1=="#define" && $2=="IDLETOKEN_OVF_MAX_HOPS" {print $3}' include/idletoken_overflow.h)")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 9: a request that had already been forwarded was forwarded again ($code)"
[ "$(conns)" = "$before" ] \
    || fail "claim 9: a request at the hop budget still dialled out — the exposure set of one prompt has no ceiling"
grep -q "already been forwarded once" "$REC/coord.log" \
    || fail "claim 9: refused, but the log does not say it was the hop budget"
note "claim 9: a request at the hop budget is refused, and nothing is dialled out"

# Control: one hop below the budget still goes out, so claim 9 is a ceiling and
# not an off-switch.
before=$(conns)
fill_machine
got=$(post_chat "$BODY" -H "$HOPS_HDR: 0")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 9 control: a request BELOW the hop budget did not borrow ($code) — the budget is an off-switch, not a ceiling"
note "claim 9 control: below the budget the same request still borrows"

# ===================================================================
# Claim 10 — fee expansion (CHAIN-05). A machine in the middle of somebody
# else's paid job must not pay a third machine at the same time. Unlike the
# origin question this needs no honesty from anybody: the count comes from
# capabilities this coordinator minted and spent itself.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready (see $REC/coord.log)"
CAP=$(mint_cap "gate-job-10" "$BODY")
[ -n "$CAP" ] || fail "claim 10: could not mint a capability"
# Occupy the single local slot with a PLATFORM job and leave it running.
curl -s -m 40 -H 'Authorization: Bearer gatetok' -H 'Content-Type: application/json' \
     -H "$ADMIT_HDR: $CAP" -d "$BODY" \
     "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1 &
PLAT_JOB_PID=$!
sleep 2
grep -q "platform job gate-job-10 admitted" "$REC/coord.log" \
    || fail "claim 10 fixture: the platform job was not admitted, so no platform work is in flight and nothing below is an assertion"
before=$(conns)
got=$(post_chat "$BODY" -H "$LOCAL_HDR: $(local_marker)")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 10: while a platform job was in flight, a local request was answered $code instead of 429 — this machine would have paid a third machine while already being paid for the slot"
[ "$(conns)" = "$before" ] \
    || fail "claim 10: while a platform job was in flight, the coordinator dialled out — the same prompt can now reach a third party and be charged twice"
grep -q "platform-dispatched job is in flight" "$REC/coord.log" \
    || fail "claim 10: refused, but the log does not say it was the in-flight platform job"
wait "$PLAT_JOB_PID" 2>/dev/null || true
note "claim 10: while a platform job is in flight, an attributed local request is not forwarded"

# Control: once the platform job has finished, the very same attributed request
# borrows again — the rule releases, it does not latch.
sleep 1
before=$(conns)
fill_machine
got=$(post_chat "$BODY" -H "$LOCAL_HDR: $(local_marker)")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 10 control: after the platform job finished the same request still would not borrow ($code) — the in-flight mark latched, and this machine can never borrow again"
note "claim 10 control: the in-flight mark releases when the platform job ends"

# ===================================================================
# Claim 11 — the disclosure endpoint reports what is ENFORCED (PRIV-01/02,
# HOST-06). Judged by making the state change and requiring the answer to
# change with it: a constant string would otherwise pass.
# ===================================================================
posture() { curl -s -m 5 "http://127.0.0.1:$API_PORT$PRIVACY_PATH" 2>/dev/null; }
PRIVACY_PATH=/idletoken/v1/privacy/posture
p_on=$(posture)
printf '%s' "$p_on" | grep -q '"overflow_enabled":true' \
    || fail "claim 11: overflow is on but the posture endpoint does not say so: $p_on"
printf '%s' "$p_on" | grep -q 'the provider a borrowed request lands on' \
    || fail "claim 11: with borrowing on, the posture does not disclose that another provider will see the prompt (PRIV-01/PRIV-04)"
printf '%s' "$p_on" | grep -q '"admission":{"armed":true' \
    || fail "claim 11: the posture does not report the admission channel as armed"

cleanup; sleep 1
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip || fail "the coordinator never became ready with overflow OFF"
p_off=$(posture)
printf '%s' "$p_off" | grep -q '"overflow_enabled":false' \
    || fail "claim 11 control: with overflow off the posture still says it is on — the endpoint reports a constant, not the enforced state: $p_off"
printf '%s' "$p_off" | grep -q 'the provider a borrowed request lands on' \
    && fail "claim 11 control: with overflow off the posture still names a borrowed provider — the disclosure is boilerplate, not a reading of the state"
note "claim 11: the posture endpoint tracks the enforced state in both directions"

# ===================================================================
# Claim 12 — a borrowed answer is judged by the ENVELOPE, not by which 2xx the
# platform chose. This claim exists because its absence cost the feature its
# whole production life: the gateway answered NestJS's default 201 for a @Post,
# the coordinator tested `status != 200`, and so every successful borrow was
# discarded by the borrower — routed, generated, CHARGED, and then handed back
# to the caller as an ordinary "busy" 429 (measured between two Windows
# providers). The gate could not see it because this fixture answered 200
# while the thing it stood in for answered 201.
#
# Its control is the other half: a NON-2xx must still be refused. Without that,
# "accepts 201" would also pass on a coordinator that had stopped looking at the
# status at all, which is a different bug wearing this one's clothes.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 201 || fail "the stub platform did not come up answering 201"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready with overflow on (see $REC/coord.log)"
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] \
    || fail "claim 12: the platform answered 201 (NestJS's default for a POST) and the borrow was thrown away — the caller got $code. This is the 2026-09-03 production bug: the job is still dispatched and still billed."
printf '%s' "$got" | grep -q "borrowed-answer" \
    || fail "claim 12: the 201 was accepted but its answer did not reach the caller"
note "claim 12: a 201 from the platform is a successful borrow"

cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 418 || fail "the stub platform did not come up answering 418"
start_coord_or_skip --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the coordinator never became ready with overflow on (see $REC/coord.log)"
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "429" ] \
    || fail "claim 12 control: the platform answered 418 and the coordinator relayed it as $code — it is no longer reading the status at all, so 'accepts 201' proves nothing"
grep -q "platform answered 418" "$REC/coord.log" \
    || fail "claim 12 control: refused, but the log does not name the status that caused it"
note "claim 12 control: a non-2xx platform answer is still refused, and named"

# ===================================================================
# Claim 13 — the actual compatible-client product path. A provider machine runs
# --shared, its one local slot is occupied, and the second request carries ONLY
# ordinary OpenAI/Anthropic headers. Nimbalyst, Claude Code, Codex and curl do
# not know about X-IdleToken-Local-Origin; requiring it makes "OpenAI-compatible"
# false exactly when overflow is needed.
#
# Claim 1 already proves that a real platform-agent request, with its admission
# capability and legacy marker, never forwards under this same CAPABILITY
# policy. Claim 10 independently proves the in-flight interlock. This claim is
# their product-side counterpart: absence of a private header on a loopback API
# request must not turn borrowing off for the user.
# ===================================================================
cleanup; sleep 1
: > "$REC/conn.log"
start_platform good 1 || fail "the stub platform did not come up"
start_coord_or_skip --shared \
    --overflow-url "http://127.0.0.1:$PLAT_PORT" --overflow-key sk-gate-key \
    || fail "the shared coordinator never became ready with overflow on (see $REC/coord.log)"
grep -q "origin policy 'capability'" "$REC/coord.log" \
    || fail "claim 13: --shared did not use the compatible-client capability policy"
grep -q "loopback API clients may borrow" "$REC/coord.log" \
    || fail "claim 13: the startup log does not state the compatible-client contract"

before=$(conns)
fill_machine
got=$(post_chat "$BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 13: an unmodified OpenAI request on a busy --shared machine did not borrow ($code) — this reproduces the Nimbalyst multi-session failure"
printf '%s' "$got" | grep -q "borrowed-answer" \
    || fail "claim 13: the unmodified OpenAI request got 200 without the platform's borrowed answer"
note "claim 13: unmodified OpenAI traffic borrows while --shared is busy"

before=$(conns)
fill_machine
got=$(post_anthropic "$ANTH_STREAM_BODY")
code=$(printf '%s' "$got" | tail -1)
[ "$code" = "200" ] && [ "$(conns)" -gt "$before" ] \
    || fail "claim 13: an unmodified Anthropic request on a busy --shared machine did not borrow ($code) — Claude Code would still fail"
printf '%s' "$got" | grep -q 'event: message_stop' \
    || fail "claim 13: the borrowed Anthropic response was not a complete stream"
note "claim 13: unmodified Anthropic traffic borrows while --shared is busy"

cleanup
echo "OVERFLOW_GATE_OK:platform work never forwarded (via the real agent, and its capability really spent); strict opt-in refuses an unattributed request while a marked request still borrows, and legacy reproduces the wider pre-hardening hole; the product default lets unmodified OpenAI and Anthropic clients borrow on a busy --shared machine; a disconnected local client cancels its engine request and releases the single slot; capability replay/mismatch/garbage each 403 for their own reason; the hop budget and an in-flight platform job both stop a forward and both release; local overflow sealed with no plaintext; tokenless loopback is supported and four bad keys fail closed; stream and non-stream local work may borrow; tool definitions, forced choice, call history and structured tool responses survive a borrow; an explicit operator cap refuses without dialling out; the posture endpoint tracks what is enforced; a 201 borrow succeeds and a non-2xx is still refused"
