#!/usr/bin/env bash
# N0 oracle — the CPU layer-range refactor (patch 0004) is token-identical.
#
# docs/cpu-hybrid-design.md gate N0: parameterizing the two CPU forward passes
# must not change what the FULL-range path computes. That is true by
# construction (the original signatures survive as wrappers passing
# 0..DS4_N_LAYER), and "by construction" is exactly the reasoning this project
# has been burned by — so we check it against a real model instead.
#
# Method: build the vendored ds4 CPU CLI twice from the SAME tree, once with
# patch 0004 reverse-applied, and greedy-decode the same prompt. The selected
# token ids must match exactly.
#
# Runs on a Linux box that has the GGUF (DGX). Last line: N0_OK or N0_FAIL.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

GGUF="${IDLETOKEN_GGUF:-$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
PATCHFILE="$ROOT/scripts/ds4-patches/0004-cpu-pp-range.patch"
PROMPT="${IDLETOKEN_N0_PROMPT:-What is the capital of France?}"
NTOK="${IDLETOKEN_N0_TOKENS:-8}"
CTX="${IDLETOKEN_N0_CTX:-2048}"
WORK=/tmp/n0-cpu-refactor
LOG=$WORK/logs

fail() { echo "N0: $*" >&2; echo N0_FAIL; exit 1; }

[ -f "$GGUF" ]      || fail "no GGUF at $GGUF (set IDLETOKEN_GGUF)"
[ -f "$PATCHFILE" ] || fail "no patch at $PATCHFILE"
command -v patch >/dev/null || fail "need patch(1)"

rm -rf "$WORK"; mkdir -p "$LOG"

# Work on a COPY of vendor/ds4 — never mutate the checked-out tree, or a failed
# run leaves the repo holding a reverse-patched engine.
cp -r "$ROOT/vendor/ds4" "$WORK/tree" || fail "could not copy vendor/ds4"

# --- the two variants -------------------------------------------------------
cp "$WORK/tree/ds4.c" "$WORK/ds4_post.c"
cp "$WORK/tree/ds4.c" "$WORK/ds4_pre.c"
patch --quiet -R "$WORK/ds4_pre.c" < "$PATCHFILE" \
    || fail "could not reverse-apply 0004 — is vendor/ds4/ds4.c still the patched tree?"
cmp -s "$WORK/ds4_pre.c" "$WORK/ds4_post.c" \
    && fail "pre and post are identical — patch 0004 does not apply to this tree"

run_variant() {   # $1 = pre|post
    local tag=$1
    cp "$WORK/ds4_${tag}.c" "$WORK/tree/ds4.c" || fail "could not stage $tag source"
    ( cd "$WORK/tree" && make clean >/dev/null 2>&1; make cpu -j"$(nproc)" ) \
        > "$LOG/build_$tag.log" 2>&1 || { sed -n '1,20p' "$LOG/build_$tag.log" >&2; fail "$tag build failed"; }
    [ -x "$WORK/tree/ds4" ] || fail "$tag build produced no ./ds4"
    # stderr, NOT stdout: this function's stdout IS the token-id list that the
    # caller captures. Echoing progress here made the first run compare
    # "running pre …\n<ids>" against "running post …\n<ids>" and report a
    # divergence at the word `pre` vs `post` while the ids were identical —
    # a gate whose FAIL needs a human to reinterpret is not a gate.
    echo "N0: running $tag (CPU backend, greedy, $NTOK tokens — the 80G model on CPU is slow)" >&2
    "$WORK/tree/ds4" -m "$GGUF" --backend cpu -c "$CTX" -sys "" --nothink --temp 0 \
        -n "$NTOK" -p "$PROMPT" --dump-logprobs "$WORK/$tag.json" \
        > "$LOG/run_$tag.log" 2>&1 || { tail -20 "$LOG/run_$tag.log" >&2; fail "$tag run failed"; }
    python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(" ".join(str(s["selected"]["id"]) for s in d["steps"]))' "$WORK/$tag.json"
}

PRE_IDS=$(run_variant pre)   || exit 1
POST_IDS=$(run_variant post) || exit 1

echo "N0: pre  ids: $PRE_IDS"
echo "N0: post ids: $POST_IDS"
[ -n "$PRE_IDS" ] || fail "pre produced no tokens"

if [ "$PRE_IDS" = "$POST_IDS" ]; then
    echo "N0: $NTOK tokens identical — the refactor did not change full-range numerics"
    echo N0_OK
else
    # Say WHERE they diverge; "they differ" alone has cost this project hours.
    python3 - "$PRE_IDS" "$POST_IDS" <<'PY'
import sys
a=sys.argv[1].split(); b=sys.argv[2].split()
for i,(x,y) in enumerate(zip(a,b)):
    if x!=y:
        print(f"N0: first divergence at step {i}: pre={x} post={y}", file=sys.stderr); break
else:
    print(f"N0: same prefix, different length: pre={len(a)} post={len(b)}", file=sys.stderr)
PY
    echo N0_FAIL
    exit 1
fi
