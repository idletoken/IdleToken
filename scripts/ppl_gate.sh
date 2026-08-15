#!/usr/bin/env bash
# G-PPL: distribution-level numeric gate (v2 rebuild plan §1.10 / WS-F3).
#
# Replaces the exact-token-match oracles that decision #10 retired: two correct
# engines diverge at the 4th greedy token even on one machine, so token-for-token
# equality cannot certify correctness across engine changes. What CAN is a
# distribution-level signal -- here the perplexity of the pinned engine over a
# fixed, committed corpus, checked against a recorded band.
#
# The band lives in test-assets/ppl/<model>.<quant>.json (mean +/- tolerance),
# measured once on the pinned build. This gate recomputes with llama-perplexity
# (already a build artifact) and asserts the result is inside the band. A result
# outside means a real regression: a broken quant, a wrong chat/tokenizer
# setup, or an engine change that shifted the distribution.
#
# HF cross-validation (a second, independent reference) is documented as PENDING
# in the reference file and lands with larger models -- for a 0.8B model the
# self-consistent band against the SAME pinned engine is the regression guard,
# and the absolute number is only comparable against that same build (hence the
# engine pin is recorded alongside the band).
#
# Last line contract: G_PPL_OK <summary> / G_PPL_FAIL: <reason> /
# G_PPL_SKIP: <reason>.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="${IDLETOKEN_ENGINE_DIR:-$REPO/vendor/llama.cpp/build/bin}"
PPL_BIN="$ENGINE_DIR/llama-perplexity"
GGUF="${IDLETOKEN_SMOKE_GGUF:-}"
MODEL_ID="${IDLETOKEN_SMOKE_MODEL_ID:-}"
QUANT="${IDLETOKEN_PPL_QUANT:-Q4_K_M}"
CORPUS="$REPO/test-assets/ppl/corpus.txt"

say()  { printf '%s\n' "$*"; }
skip() { say "G_PPL_SKIP: $*"; exit 0; }
die()  { say "G_PPL_FAIL: $*"; exit 1; }

command -v python3 >/dev/null 2>&1 || skip "python3 not available"
[ -x "$PPL_BIN" ] || skip "no llama-perplexity in $ENGINE_DIR (scripts/build_llamacpp.sh)"
[ -r "$CORPUS" ]  || skip "committed corpus missing: $CORPUS"
[ -n "$GGUF" ] || skip "set IDLETOKEN_SMOKE_GGUF to the small model's GGUF"
[ -r "$GGUF" ] || skip "IDLETOKEN_SMOKE_GGUF unreadable: $GGUF"
[ -n "$MODEL_ID" ] || skip "set IDLETOKEN_SMOKE_MODEL_ID (selects the reference band)"

REF="$REPO/test-assets/ppl/${MODEL_ID}.${QUANT}.json"
[ -r "$REF" ] || skip "no reference band for ${MODEL_ID}.${QUANT} at $REF (record one on the pinned build before gating)"

read -r PPL_MEAN TOL < <(python3 -c "
import json
d=json.load(open('$REF'))
print(d['ppl_mean'], d['tolerance'])
" 2>/dev/null)
[ -n "${PPL_MEAN:-}" ] || die "could not read ppl_mean/tolerance from $REF"

say "ppl: recomputing perplexity over $(basename "$CORPUS") for ${MODEL_ID}.${QUANT}"
OUT="$($PPL_BIN -m "$GGUF" -f "$CORPUS" -ngl 99 2>&1)"
LINE="$(printf '%s\n' "$OUT" | grep -E 'Final estimate: PPL' | tail -1)"
[ -n "$LINE" ] || die "llama-perplexity produced no Final estimate (see engine output: $(printf '%s' "$OUT" | tail -3 | tr '\n' ' '))"
PPL="$(printf '%s' "$LINE" | sed -nE 's/.*PPL = ([0-9.]+).*/\1/p')"
[ -n "$PPL" ] || die "could not parse PPL from: $LINE"

VERDICT="$(python3 -c "
ppl=float('$PPL'); mean=float('$PPL_MEAN'); tol=float('$TOL')
lo=mean-tol; hi=mean+tol
if lo <= ppl <= hi: print('OK %.4f in [%.4f, %.4f]' % (ppl, lo, hi))
else: print('OUT %.4f outside [%.4f, %.4f] (band = %.4f +/- %.4f)' % (ppl, lo, hi, mean, tol))
" 2>/dev/null)"
case "$VERDICT" in
    OK\ *) say "G_PPL_OK ${MODEL_ID}.${QUANT} PPL ${VERDICT#OK }" ; exit 0 ;;
    OUT\ *) die "perplexity regression: ${MODEL_ID}.${QUANT} PPL ${VERDICT#OUT }" ;;
    *) die "could not compare PPL $PPL against the band in $REF" ;;
esac
