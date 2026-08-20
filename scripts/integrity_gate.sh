#!/usr/bin/env bash
# G-INTEGRITY — the model-integrity and probe-honesty gate.
#
# Two properties, one gate (docs/threat-model.md §2):
#
#   A. A file that is not the curated model never reaches the engine. Every
#      model offered to users pins a SHA-256, and every download path verifies
#      it before use.
#   B. A provider cannot pass the canary by recognising it. Probes are
#      indistinguishable from real traffic, and the verdict is a
#      distribution-level judgement, not an echo check.
#
# **Every check here carries its own positive control** — it first proves the
# check can go red, then proves it is green. This repo has twice shipped a
# check that could not fail (a grep pattern that never matched, a `grep -P` on
# BSD grep counting nothing); a gate without a control is decoration.
#
# Contract: prints exactly one G_INTEGRITY_(OK|FAIL|SKIP) line last.
# Local gate: no cluster nodes, no GPU, no weights.
set -u

cd "$(dirname "$0")/.." || exit 1

ok()   { echo "  [ok] $*"; }
bad()  { echo "  [BAD] $*"; FAILED="${FAILED}
  - $*"; }
FAILED=""
SKIPPED=""

# --- A1. Manifest coverage ------------------------------------------------
# Every model users can pick must pin a hash for every file it can download.
#
# Judged HERE rather than by shelling out to model_manifest_check.py: that
# script also compares manifests against the compiled engine registry, so a
# model mid-curation (variants added, registry not yet regenerated) would turn
# this gate red for a reason that has nothing to do with integrity. A gate whose
# red light points at the wrong thing gets ignored. The registry comparison has
# its own home; this one owns the hash rule, control included.
coverage() {   # prints offending "<id>[<quant>]" lines; silence = covered
    python3 - <<'PY'
import glob, json
for p in sorted(glob.glob("models/*.json")):
    m = json.load(open(p))
    if not m.get("available"):
        continue          # still under curation — not offered to users
    vs = m.get("variants") or []
    if vs:
        for v in vs:
            if not v.get("sha256"):
                print(f"{m.get('id')}[{v.get('quant')}]")
    elif not m.get("sha256"):
        print(f"{m.get('id')}[default_gguf]")
PY
}
# Positive control FIRST: strip a hash and require the checker to notice.
victim=$(python3 - <<'PY'
import glob, json
for p in sorted(glob.glob("models/*.json")):
    m = json.load(open(p))
    if m.get("available") and ((m.get("variants") or [{}])[0].get("sha256") or m.get("sha256")):
        print(p); break
PY
)
if [ -z "$victim" ]; then
    bad "no available model with a pinned hash — nothing for the control to strip"
else
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    cp "$victim" "$tmp/victim.json"
    python3 - "$victim" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
(m["variants"][0] if m.get("variants") else m).pop("sha256", None)
json.dump(m, open(sys.argv[1], "w"), indent=2, ensure_ascii=False)
open(sys.argv[1], "a").write("\n")
PY
    if [ -z "$(coverage)" ]; then
        bad "CONTROL: coverage check saw nothing wrong with a hash removed — it does not check"
    else
        ok "control: removing a pinned hash is detected"
    fi
    cp "$tmp/victim.json" "$victim"
fi
missing=$(coverage)
if [ -z "$missing" ]; then
    ok "every available model pins a SHA-256 for every downloadable file"
else
    bad "available models missing a pinned hash: $(echo "$missing" | tr '\n' ' ')"
fi

# --- A2. Client download gate --------------------------------------------
# The Rust integrity tests include the control that matters: a file with one
# byte changed must be rejected AND deleted.
# A non-interactive shell does not read the profile that puts cargo on PATH,
# so look where rustup installs it before concluding it is absent — otherwise
# this gate skips its most important control on a machine that can run it.
[ -x "$HOME/.cargo/bin/cargo" ] && PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
    SKIPPED="${SKIPPED:+$SKIPPED; }cargo not on PATH (client integrity tests not run)"
else
    # Not --quiet: the per-test lines ARE the evidence that the cases ran. A
    # quiet run that executed nothing exits 0 and looks identical to a pass.
    if (cd client/src-tauri && cargo test integrity_tests 2>&1) >/tmp/idletoken-integrity-rust.log 2>&1; then
        n=$(grep -cE '^test weights::integrity_tests' /tmp/idletoken-integrity-rust.log || true)
        # A suite that silently stopped running its tests reports "0 passed"
        # and exits 0 — the same shape as a real pass.
        if [ "${n:-0}" -ge 6 ]; then
            ok "client hash gate: $n integrity tests pass (incl. corrupted-file control)"
        else
            bad "client integrity tests ran only ${n:-0} cases — expected the full set"
        fi
    else
        bad "client integrity tests failed (see /tmp/idletoken-integrity-rust.log)"
    fi
fi

# --- A3. The script path is not a back door ------------------------------
# model_fetch.sh must verify the same hash; a download path that skips the gate
# is the gate's biggest hole, and it is one grep away from silently returning.
if grep -q "sha256sum\|shasum -a 256" scripts/model_fetch.sh; then
    ok "scripts/model_fetch.sh verifies the pinned hash"
else
    bad "scripts/model_fetch.sh no longer verifies downloads against the manifest hash"
fi

# --- B. Canary honesty ----------------------------------------------------
# The probe bank's specs contain the three fake providers (garbage / echo /
# selective performer). The selective one is the whole point: it passes the OLD
# marked probe and must fail the new one.
GW=platform/packages/gateway
if [ ! -d "$GW/node_modules" ] && [ ! -d node_modules ]; then
    SKIPPED="${SKIPPED:+$SKIPPED; }gateway dependencies not installed (canary honesty not run)"
elif [ ! -f "$GW/src/canary/probe-bank.spec.ts" ]; then
    SKIPPED="${SKIPPED:+$SKIPPED; }platform layer not present in this checkout"
else
    if (cd "$GW" && npx jest src/canary --silent 2>&1) >/tmp/idletoken-integrity-canary.log 2>&1; then
        passed=$(grep -oE 'Tests:[^,]*[0-9]+ passed' /tmp/idletoken-integrity-canary.log | tail -1)
        ok "canary honesty: ${passed:-tests pass} (fake providers incl. the selective performer are caught)"
    else
        bad "canary tests failed (see /tmp/idletoken-integrity-canary.log)"
    fi
    # The marker removal itself: a probe payload that says "canary" is
    # recognisable, and no test can catch it if someone puts it back.
    #
    # Scoped to the PAYLOAD EXPRESSION, not the file: the comments there
    # necessarily quote the old marker to explain why it is gone, and a
    # file-wide grep matches that prose and cries wolf (it did, first run).
    payload_src=$(sed -n '/const payload = new TextEncoder/,/^    );$/p' \
                      "$GW/src/canary/canary.service.ts" | grep -v '^\s*[/*]')
    if [ -z "$payload_src" ]; then
        bad "could not locate the canary payload expression (did canary.service.ts change shape?)"
    # Judge what goes ON THE WIRE, not our own identifiers: `probe.content` is a
    # local variable name the provider never sees, and matching on it made this
    # check cry wolf on its first run. What must not appear is a marker word in
    # the serialised request, or a field beyond the three a real request has.
    elif printf '%s' "$payload_src" | grep -qiE "canary|nonce"; then
        bad "the canary payload names itself (canary/nonce) — providers can detect probes"
    elif printf '%s' "$payload_src" | grep -qE "^\s+[a-zA-Z_]+:" \
         && [ "$(printf '%s' "$payload_src" | grep -cE "^\s+(model|messages|maxTokens|quant):")" \
              -ne "$(printf '%s' "$payload_src" | grep -cE "^\s+[a-zA-Z_]+:")" ]; then
        bad "the canary payload carries a field a real request does not have — that is a marker"
    else
        ok "probe payload carries no identifying marker (fields match a real request)"
    fi
fi

# --- verdict --------------------------------------------------------------
if [ -n "$FAILED" ]; then
    echo "G_INTEGRITY_FAIL:$(echo "$FAILED" | tr '\n' ' ')"
    exit 1
fi
if [ -n "$SKIPPED" ]; then
    # Partial coverage is NOT a pass: say so, and let the ladder record a SKIP.
    echo "G_INTEGRITY_SKIP: $SKIPPED"
    exit 0
fi
echo "G_INTEGRITY_OK: manifest coverage, client hash gate (with control), script path, canary honesty"
