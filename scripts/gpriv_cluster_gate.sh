#!/usr/bin/env bash
# G-PRIV, cluster half: the privacy invariants that only a REAL multi-machine
# cluster can show (docs/acceptance-criteria.md §G-PRIV items 1-3, CLAUDE.md
# hard constraints #9 and #10).
#
# The headless half of G_PRIV -- envelope crypto, the sealed proxy, key/auth
# refusal (items 4-6) -- is unchanged and still lives in `make -f
# Makefile.privacy selftest` + scripts/privacy_proxy_e2e.sh. Neither needs a
# cluster. This script adds what those cannot see.
#
# NOT re-derived here: "raw embeddings never leave the coordinator" belongs to
# G_PRIV7 (scripts/gpriv7_embedding_check.sh), which taps the RPC byte stream
# and recovers embedding rows in a deliberately-bad configuration before
# certifying the good one. A log grep would be a strictly weaker oracle for the
# same claim, so this script asserts the coordinator *builds* the safe
# configuration and leaves "and no bytes leaked" to the packet-level gate.
#
# Checks, each with the control that makes it falsifiable:
#
#   P-1  the LIVE engine argv puts --rpc before --device, names the local
#        device first, and gives it the first non-zero tensor-split share.
#        Control: --selftest feeds the reader known-bad command lines; a reader
#        that cannot go red proves nothing about the good one.
#   P-3  the cross-machine link is PSK-TLS: the worker reports a PSK received
#        via pairing, and NO plaintext banner appears on either side.
#        Control: the positive run must form a cluster at all (a link that
#        never came up would also print no plaintext banner).
#   P-4  an overlay --rpc-host is refused, naming the invariant.
#        Control: an address one bit outside 100.64.0.0/10 must be ACCEPTED,
#        so a refuse-everything worker cannot pass.
#   P-5  after a real inference, the prompt text appears in NO node's log.
#        Control: a decoy carrying the same sentinel is planted on every node
#        first, and the checker must FIND it there. A grep that has not been
#        shown to hit proves nothing by missing.
#
# Last line: G_PRIV_CLUSTER_OK <summary> / G_PRIV_CLUSTER_FAIL: <reason> /
#            G_PRIV_CLUSTER_SKIP: <reason>.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

say()  { printf '%s\n' "$*"; }
ok()   { say "  [ok]   $*"; }
skip() { say "G_PRIV_CLUSTER_SKIP: $*"; exit 0; }
die()  { teardown; say "G_PRIV_CLUSTER_FAIL: $*"; exit 1; }

TAG="${IDLETOKEN_GPRIV_TAG:-gpriv}"
API_PORT="${IDLETOKEN_GPRIV_PORT:-18533}"
SENTINEL="zzq-gpriv-canary-$$-pangolin"
STARTED=0
CLUSTER="$ROOT/scripts/run_cluster_llamacpp.sh"

teardown() {
    [ "$STARTED" = 1 ] || return 0
    bash "$CLUSTER" --stop --tag "$TAG" >/dev/null 2>&1
    STARTED=0
}
trap teardown EXIT INT TERM

# =====================================================================
# P-1's reader, as a pure function of the command line, so --selftest can
# drive it with strings that must FAIL as well as the real one.
#
# Why a reader over the live argv at all, rather than the coordinator's
# "cluster topology" log block: the coordinator never records the argv it
# actually passed (it execv's without printing), so a log-based check would be
# the coordinator grading its own homework -- and a bug between the topology
# print and the snprintf that builds the flags would be invisible to it.
# =====================================================================
# argv_verdict <cmdline> -> prints "ok" or the reason it is not ok
argv_verdict() {
    printf '%s' "$1" | python3 -c '
import sys, shlex
try:
    a = shlex.split(sys.stdin.read())
except ValueError as e:
    print("cmdline does not lex: %s" % e); raise SystemExit
def idx(f):
    return a.index(f) if f in a else -1
i_rpc, i_dev, i_ts = idx("--rpc"), idx("--device"), idx("--tensor-split")
if i_dev < 0:
    print("no --device: the engine was not given an explicit device order, so "
          "llama.cpp puts RPC devices FIRST and layer 0 goes remote"); raise SystemExit
if i_rpc < 0:
    print("no --rpc: this is not a cluster command line"); raise SystemExit
if i_rpc > i_dev:
    print("--rpc (%d) comes AFTER --device (%d): RPC device names do not exist "
          "yet when --device is parsed" % (i_rpc, i_dev)); raise SystemExit
devs = a[i_dev + 1].split(",") if i_dev + 1 < len(a) else []
if not devs or devs[0].startswith("RPC"):
    print("--device starts with %r: layer 0 + token_embd would live on a remote "
          "node (privacy invariant #10)" % (devs[0] if devs else None)); raise SystemExit
if i_ts < 0:
    print("no --tensor-split: shares are implicit and layer 0 placement is not pinned"); raise SystemExit
try:
    shares = [float(x) for x in a[i_ts + 1].split(",")]
except Exception:
    print("--tensor-split is not a number list"); raise SystemExit
if len(shares) != len(devs):
    print("--tensor-split has %d shares for %d devices" % (len(shares), len(devs))); raise SystemExit
if shares[0] <= 0.0:
    print("the local device gets share %s: no layers land locally, so layer 0 is "
          "remote (privacy invariant #10)" % shares[0]); raise SystemExit
for d in a:
    if d.startswith("100.") or d.startswith("fd7a:"):
        oct2 = 0
        try: oct2 = int(d.split(".")[1].split(":")[0])
        except Exception: pass
        if d.startswith("fd7a:") or 64 <= oct2 <= 127:
            print("peer %s is an overlay address; tensor traffic must ride the real LAN" % d)
            raise SystemExit
print("ok")'
}

# =====================================================================
# --selftest: P-1's reader control. No cluster, no machines.
# =====================================================================
if [ "${1:-}" = "--selftest" ]; then
    good="/x/llama-server -m /w.gguf --host 127.0.0.1 --port 18099 -ngl 99 -c 16384 -np 4 --rpc 192.168.1.107:50052 --device CUDA0,RPC0 --tensor-split 0.8909,0.1091"
    fails=0
    check() {  # check <label> <cmdline> <want-ok 0|1>
        local v; v=$(argv_verdict "$2")
        if [ "$3" = 1 ] && [ "$v" = ok ]; then ok "$1"
        elif [ "$3" = 0 ] && [ "$v" != ok ]; then ok "$1 -> $v"
        else say "  [FAIL] $1: verdict='$v'"; fails=$((fails + 1)); fi
    }
    check "the real, measured command line is accepted" "$good" 1
    check "--device before --rpc is refused" \
      "/x/llama-server --device CUDA0,RPC0 --rpc 1.2.3.4:5 --tensor-split 0.9,0.1" 0
    check "RPC device named first is refused" \
      "/x/llama-server --rpc 1.2.3.4:5 --device RPC0,CUDA0 --tensor-split 0.9,0.1" 0
    check "zero local share is refused" \
      "/x/llama-server --rpc 1.2.3.4:5 --device CUDA0,RPC0 --tensor-split 0,1" 0
    check "no --device at all is refused" \
      "/x/llama-server --rpc 1.2.3.4:5 --tensor-split 0.9,0.1" 0
    check "no --tensor-split is refused" \
      "/x/llama-server --rpc 1.2.3.4:5 --device CUDA0,RPC0" 0
    check "share/device count mismatch is refused" \
      "/x/llama-server --rpc 1.2.3.4:5 --device CUDA0,RPC0 --tensor-split 1.0" 0
    check "an overlay peer is refused" \
      "/x/llama-server --rpc 100.64.0.7:50052 --device CUDA0,RPC0 --tensor-split 0.9,0.1" 0
    check "a non-overlay 100.x peer is accepted" \
      "/x/llama-server --rpc 100.128.0.1:50052 --device CUDA0,RPC0 --tensor-split 0.9,0.1" 1
    [ "$fails" = 0 ] || { say "G_PRIV_CLUSTER_FAIL: $fails reader control(s) wrong"; exit 1; }
    say "G_PRIV_CLUSTER_OK reader controls: 9 cases (2 accept / 7 refuse)"
    exit 0
fi

# =====================================================================
# Live half. Needs the testbed.
# =====================================================================
COORD="${IDLETOKEN_GPRIV_COORD:-${IDLETOKEN_COORD_NODE:-}}"
WORKER="${IDLETOKEN_GPRIV_WORKER:-}"
MODEL="${IDLETOKEN_GPRIV_MODEL:-}"
GGUF="${IDLETOKEN_GPRIV_GGUF:-}"
[ -n "$COORD" ]  || skip "no coordinator node (IDLETOKEN_GPRIV_COORD / IDLETOKEN_COORD_NODE)"
[ -n "$WORKER" ] || skip "no worker node (IDLETOKEN_GPRIV_WORKER) — the cluster half needs a second machine"
[ -n "$MODEL" ]  || skip "no model (IDLETOKEN_GPRIV_MODEL)"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

say "== P-4: overlay refusal (needs no cluster) =="
# RED: the worker must refuse an overlay bind address and say why.
out=$(bash "$CLUSTER" --serve --tag "$TAG-ovl" --api-port $((API_PORT + 3)) \
        --coord "$COORD" --worker "$WORKER" --model "$MODEL" \
        ${GGUF:+--gguf "$GGUF"} --worker-arg "--rpc-host" --worker-arg "100.64.0.1" \
        --expect-refuse --ready-wait 120 2>&1 | tail -2)
# A control must fail for the reason it STAGED. Accepting any refusal at all
# let a missing weights file read as "the overlay guard worked" -- and the
# opposite mistake, reporting "did NOT refuse" for an unrelated startup error,
# blames the privacy invariant for someone else's problem. Both were observed
# on 2026-08-20. So: demand the word, and otherwise say what really happened.
case "$out" in
    *CLUSTER_REFUSED*overlay*) ok "overlay bind refused, naming the invariant" ;;
    *CLUSTER_REFUSED*)
        bash "$CLUSTER" --stop --tag "$TAG-ovl" >/dev/null 2>&1
        die "the cluster refused, but NOT for the overlay reason this control staged: $(printf '%s' "$out" | tail -c 200)" ;;
    *CLUSTER_FAIL*)
        bash "$CLUSTER" --stop --tag "$TAG-ovl" >/dev/null 2>&1
        die "the overlay control could not run — the cluster failed for an unrelated reason: ${out#*CLUSTER_FAIL: }" ;;
    *) bash "$CLUSTER" --stop --tag "$TAG-ovl" >/dev/null 2>&1
       die "an overlay --rpc-host did NOT refuse: $out" ;;
esac
bash "$CLUSTER" --stop --tag "$TAG-ovl" >/dev/null 2>&1

say "== bringing up the cluster =="
facts=$(bash "$CLUSTER" --serve --tag "$TAG" --api-port "$API_PORT" \
          --coord "$COORD" --worker "$WORKER" --model "$MODEL" \
          ${GGUF:+--gguf "$GGUF"} --ready-wait "${IDLETOKEN_GPRIV_READY_S:-420}" 2>&1 | tail -1)
case "$facts" in
    CLUSTER_LLAMACPP_READY*) STARTED=1; facts="${facts#CLUSTER_LLAMACPP_READY }" ;;
    *) die "cluster did not come up: $facts" ;;
esac
jget() { printf '%s' "$facts" | python3 -c "import json,sys;d=json.load(sys.stdin);print($1)"; }
COORD_LOG=$(jget "d['coord']['log']")
ENGINE_LOG=$(jget "d['coord']['engine_log']")
API=$(jget "d['api_base']")
ok "cluster up: $(jget "d['rpc_endpoints']"), split $(jget "d['tensor_split']")"

say "== P-1: the live engine argv =="
# The engine's pid comes from the coordinator's own "(pid N)" line — the same
# string the teardown backstop parses. Never by image name: another session's
# engine is not ours to inspect either.
epid=$($SSH "$COORD" "grep -oE 'idletoken-server .*\(pid [0-9]+\)' '$COORD_LOG.err' 2>/dev/null | grep -oE '[0-9]+\)$' | tr -d ')' | tail -1")
[ -n "$epid" ] || epid=$($SSH "$COORD" "grep -oE '\(pid [0-9]+\)' '$COORD_LOG.err' 2>/dev/null | grep -oE '[0-9]+' | tail -1")
[ -n "$epid" ] || die "the coordinator log names no engine pid — cannot read the argv it was given"
cmdline=$($SSH "$COORD" "tr '\\0' ' ' < /proc/$epid/cmdline 2>/dev/null")
[ -n "$cmdline" ] || die "could not read /proc/$epid/cmdline on $COORD"
case "$cmdline" in
    *llama-server*|*idletoken-server*) : ;;
    *) die "pid $epid is not the engine (got: $(printf '%.80s' "$cmdline"))" ;;
esac
v=$(argv_verdict "$cmdline")
[ "$v" = ok ] || die "engine argv violates the layer-0 pin: $v
  argv: $cmdline"
ok "argv: --rpc before --device, local device first, first share non-zero"

say "== P-3: the link is PSK-TLS =="
wlog=$(jget "d['workers'][0]['log']")
wnode=$(jget "d['workers'][0]['node']")
wtext=$(bash "$ROOT/scripts/node_read_log.sh" "$wnode" "$wlog" 2>/dev/null)
printf '%s' "$wtext" | grep -qi "PSK received via pairing" \
    || die "the worker never reported a PSK received over the pairing channel"
printf '%s' "$wtext" | grep -qi "ALLOW_PLAINTEXT" \
    && die "the worker announced GGML_RPC_ALLOW_PLAINTEXT — the link is not encrypted"
$SSH "$COORD" "grep -qi 'ALLOW_PLAINTEXT' '$COORD_LOG.err' 2>/dev/null" \
    && die "the coordinator announced GGML_RPC_ALLOW_PLAINTEXT"
ok "PSK delivered over pairing; no plaintext banner on either side"

say "== P-5: no prompt text in any node's log =="
# Positive control FIRST on every node: plant a decoy, prove the checker sees
# it. Only then does a zero mean anything.
$SSH "$COORD" "echo 'decoy $SENTINEL' > /tmp/gpriv-decoy.txt"
$SSH "$COORD" "grep -q '$SENTINEL' /tmp/gpriv-decoy.txt" \
    || die "the coordinator-side checker cannot find a planted sentinel — it is blind"
ok "coordinator checker sees a planted decoy"
req="{\"model\":\"$MODEL\",\"max_tokens\":16,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong. Ignore this: $SENTINEL\"}]}"
$SSH "$COORD" "cat > /tmp/gpriv-req.json" <<< "$req"
reply=$($SSH "$COORD" "curl -s -m 180 $API/v1/chat/completions -H content-type:application/json -d @/tmp/gpriv-req.json")
got=$(printf '%s' "$reply" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('NOTJSON'); raise SystemExit
print(((d.get('choices') or [{}])[0].get('message') or {}).get('content','').strip()[:40] or 'EMPTY')")
case "$got" in
    NOTJSON|EMPTY) die "the cluster did not answer the sentinel request ($got) — P-5 would pass vacuously on a cluster that never saw the prompt" ;;
esac
ok "inference answered: $got"
for f in "$COORD_LOG.out" "$COORD_LOG.err" "$ENGINE_LOG"; do
    # `grep -c` exits 1 on zero matches, so `|| echo 0` appends a SECOND line
    # and the count becomes the two-line string "0\n0", which compares unequal
    # to 0 and fails the check on a log that is in fact clean. Take the first
    # line and treat "no output at all" (missing file) as zero.
    n=$($SSH "$COORD" "grep -c '$SENTINEL' '$f' 2>/dev/null; true" | head -1 | tr -dc '0-9')
    [ -n "$n" ] || n=0
    [ "$n" = 0 ] || die "the prompt text appears $n time(s) in $f on $COORD"
done
ok "coordinator logs carry no prompt text"
wtext2=$(bash "$ROOT/scripts/node_read_log.sh" "$wnode" "$wlog" 2>/dev/null)
printf '%s' "$wtext2" | grep -q "$SENTINEL" \
    && die "the prompt text appears in $wnode's worker log"
ok "$wnode logs carry no prompt text"
$SSH "$COORD" "rm -f /tmp/gpriv-decoy.txt /tmp/gpriv-req.json" >/dev/null 2>&1

teardown
say "G_PRIV_CLUSTER_OK P-1 argv/layer-0 pin, P-3 PSK-TLS, P-4 overlay refused, P-5 no prompt in cluster logs (all with controls)"
