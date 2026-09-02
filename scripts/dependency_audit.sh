#!/usr/bin/env bash
# Dependency and engine-pin integrity — DIST-10 (poisoned dependency reaches an
# official build) and the mechanical half of OPS-09 (a pinned component grows a
# CVE).
#
# The threat is not "a dependency has a bug". It is that the official installer
# is the ONE artifact users are told to trust, so anything that slips into it
# inherits that trust — and lockfiles, a pinned upstream commit and a patch
# series all look identical whether or not someone changed them.
#
# Five checks, in order of how quietly they can go wrong:
#
#   D1  the vendored llama.cpp is the commit UPSTREAM claims
#   D2  the patch series has not been edited (digests, recorded once)
#   D3  the Rust lockfile is intact and free of known advisories (cargo audit)
#   D4  the JS lockfile is intact and free of known advisories (pnpm audit)
#   D5  the human review recorded in dependency-review.json is not stale
#
# D2 is the one with no other owner: a patch file is a diff we apply to a third
# -party tree at build time, it is not covered by any lockfile, and an edit to
# it changes the shipped engine with nothing in `git log` looking unusual to a
# reviewer skimming a rebuild.
#
# Contract: prints exactly one DEP_AUDIT_(OK|FAIL|SKIP) line last.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
FAILED="" SKIPPED=""
ok()   { echo "  [ok] $*"; }
bad()  { echo "  [BAD] $*"; FAILED="${FAILED}
  - $*"; }
skip() { echo "  [skip] $*"; SKIPPED="${SKIPPED:+$SKIPPED; }$*"; }

PDIR="$ROOT/scripts/llamacpp-patches"
DIGESTS="$PDIR/PATCH-DIGESTS"
WRITE_DIGESTS=0
[ "${1:-}" = "--record-digests" ] && WRITE_DIGESTS=1

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else echo ""; fi
}

# --- D1: the vendored engine is the pinned commit ---------------------------
echo "  ---- D1 engine pin"
UP="$ROOT/scripts/llamacpp-patches/UPSTREAM"
if [ ! -f "$UP" ]; then
    bad "no scripts/llamacpp-patches/UPSTREAM — nothing states which llama.cpp this tree builds"
else
    PIN=$(awk 'NR==1{print $2}' "$UP")
    [ -n "$PIN" ] || bad "UPSTREAM does not name a commit"
    if [ -d "$ROOT/vendor/llama.cpp/.git" ]; then
        HAVE=$(git -C "$ROOT/vendor/llama.cpp" rev-parse HEAD 2>/dev/null)
        if [ "$HAVE" = "$PIN" ]; then
            ok "vendor/llama.cpp is at the pinned ${PIN:0:12}"
        else
            bad "vendor/llama.cpp is at ${HAVE:0:12}, UPSTREAM pins ${PIN:0:12} — this tree would build an engine nobody decided to ship"
        fi
        # A patched vendor tree is dirty BY DESIGN — the patch series is applied
        # in place, so "clean" would mean the patches were never applied. What
        # must hold is narrower and is the thing that actually matters: every
        # modified file is one the patch series claims to touch. A file changed
        # outside the series is an edit to third-party engine source that
        # appears in NO diff we review — our git history does not track
        # vendor/llama.cpp's contents, and the patch files would not mention it.
        touched=$(cd "$PDIR" 2>/dev/null && grep -h '^+++ b/' ./*.patch 2>/dev/null \
                  | sed 's|^+++ b/||' | sort -u)
        modified=$(git -C "$ROOT/vendor/llama.cpp" status --porcelain 2>/dev/null \
                   | awk '{print $NF}' | sort -u)
        if [ -z "$modified" ]; then
            ok "vendor/llama.cpp is unmodified (patch series not applied in this checkout)"
        else
            stray=""
            while IFS= read -r f; do
                [ -n "$f" ] || continue
                printf '%s\n' "$touched" | grep -qx "$f" || stray="$stray $f"
            done <<EOF
$modified
EOF
            if [ -n "$stray" ]; then
                bad "vendor/llama.cpp has modifications OUTSIDE the patch series:$stray — an edit to engine source that no patch file records and no review would see"
            else
                ok "every modified file in vendor/llama.cpp ($(printf '%s\n' "$modified" | grep -c .)) is one the patch series declares"
            fi
            # Positive control: a file the series does not touch must be
            # reported. Uses the same comparison, not a second one.
            if printf '%s\n' "$touched" | grep -qx "README.md"; then
                : # improbable, but do not build a control on a file the series owns
            else
                probe=$(printf '%s\nREADME.md\n' "$modified" | sort -u)
                sctl=""
                while IFS= read -r f; do
                    [ -n "$f" ] || continue
                    printf '%s\n' "$touched" | grep -qx "$f" || sctl="$sctl $f"
                done <<EOF
$probe
EOF
                case "$sctl" in
                    *README.md*) ok "control: a modified file outside the patch series is reported" ;;
                    *) bad "CONTROL: a planted out-of-series modification was not reported — the vendor check is not checking" ;;
                esac
            fi
        fi
    else
        skip "vendor/llama.cpp is not a git checkout here (engine pin not verified against the source tree)"
    fi
fi

# --- D2: the patch series is what it was ------------------------------------
echo "  ---- D2 patch series integrity"
if [ ! -d "$PDIR" ]; then
    skip "no patch directory"
else
    current=$(cd "$PDIR" && for p in *.patch; do
        [ -f "$p" ] || continue
        printf '%s  %s\n' "$(sha256_of "$p")" "$p"
    done | sort -k2)
    if [ "$WRITE_DIGESTS" = 1 ]; then
        {
            echo "# SHA-256 of each llama.cpp patch, recorded by scripts/dependency_audit.sh --record-digests."
            echo "# A patch file is a diff applied to third-party source at build time. No lockfile"
            echo "# covers it, so an edit here changes the shipped engine while looking like an"
            echo "# ordinary file in git. Regenerate ONLY when deliberately changing the series,"
            echo "# and say which patch changed and why in the same commit."
            printf '%s\n' "$current"
        } > "$DIGESTS"
        ok "recorded $(printf '%s\n' "$current" | grep -c . ) patch digests in $DIGESTS"
    elif [ ! -f "$DIGESTS" ]; then
        bad "no scripts/llamacpp-patches/PATCH-DIGESTS — run scripts/dependency_audit.sh --record-digests once, then commit it"
    else
        recorded=$(grep -v '^#' "$DIGESTS" | grep . | sort -k2)
        if [ "$current" = "$recorded" ]; then
            ok "all $(printf '%s\n' "$current" | grep -c .) engine patches match their recorded digests"
        else
            diffout=$(diff <(printf '%s\n' "$recorded") <(printf '%s\n' "$current") | head -12)
            bad "the llama.cpp patch series does not match PATCH-DIGESTS: $(printf '%s' "$diffout" | tr '\n' ' ')"
        fi
        # Positive control: a patch whose bytes changed must be reported.
        ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-patchctl.XXXXXX")
        first=$(printf '%s\n' "$current" | head -1 | awk '{print $2}')
        if [ -n "$first" ]; then
            cp "$PDIR/$first" "$ctl/x.patch"
            echo "+ /* injected */" >> "$ctl/x.patch"
            if [ "$(sha256_of "$ctl/x.patch")" = "$(sha256_of "$PDIR/$first")" ]; then
                bad "CONTROL: appending a line to a patch did not change its digest — the digest check is not checking"
            else
                ok "control: a one-line edit to a patch changes its digest"
            fi
        fi
        rm -rf "$ctl"
    fi
fi

# --- D3: Rust dependencies ---------------------------------------------------
echo "  ---- D3 rust dependencies"
[ -x "$HOME/.cargo/bin/cargo" ] && PATH="$HOME/.cargo/bin:$PATH"
LOCK="$ROOT/client/src-tauri/Cargo.lock"
if [ ! -f "$LOCK" ]; then
    bad "client/src-tauri/Cargo.lock is missing — builds would resolve versions freely, which is the DIST-10 hole itself"
elif ! command -v cargo >/dev/null 2>&1; then
    skip "cargo not on PATH (advisory scan not run)"
elif ! command -v cargo-audit >/dev/null 2>&1 && ! cargo audit --version >/dev/null 2>&1; then
    skip "cargo-audit not installed (cargo install cargo-audit) — Rust advisories not scanned"
else
    log=/tmp/idletoken-cargo-audit.log
    if (cd client/src-tauri && cargo audit --quiet) >"$log" 2>&1; then
        ok "cargo audit reports no known advisories for the pinned Rust dependency set"
    else
        bad "cargo audit reports advisories (see $log): $(grep -cE '^(ID|Crate):' "$log" 2>/dev/null || echo '?') findings"
    fi
fi

# --- D4: JavaScript dependencies --------------------------------------------
echo "  ---- D4 javascript dependencies"
JSLOCK="$ROOT/client/pnpm-lock.yaml"
if [ ! -f "$JSLOCK" ]; then
    bad "client/pnpm-lock.yaml is missing — the frontend would resolve versions freely"
elif ! command -v pnpm >/dev/null 2>&1; then
    skip "pnpm not on PATH (npm advisories not scanned)"
else
    log=/tmp/idletoken-pnpm-audit.log
    # --prod: a devDependency advisory does not reach the shipped bundle, and a
    # gate that is permanently red for build-time tooling gets ignored, which
    # costs us the production findings too.
    if (cd client && pnpm audit --prod --audit-level high) >"$log" 2>&1; then
        ok "pnpm audit reports no high/critical advisories in the shipped dependency set"
    else
        bad "pnpm audit reports high/critical advisories in shipped dependencies (see $log)"
    fi
fi

# --- D5: the human review is not stale --------------------------------------
echo "  ---- D5 review window"
REVIEW="$ROOT/scripts/dependency-review.json"
if [ ! -f "$REVIEW" ]; then
    bad "no scripts/dependency-review.json"
else
    out=$(python3 - "$ROOT" <<'PY'
import datetime, json, os, sys
d = json.load(open(os.path.join(sys.argv[1], "scripts/dependency-review.json")))
w = int(d.get("reviewWindowDays", 45))
today = datetime.date.today()
for c in d.get("components", []):
    age = (today - datetime.date.fromisoformat(c["lastReviewed"])).days
    print(("P" if age > w else "N") + "\t" + f"{c['name']}: reviewed {age} days ago (window {w})")
PY
)
    while IFS=$'\t' read -r kind msg; do
        [ -n "$msg" ] || continue
        case "$kind" in P) bad "$msg — triage the advisory sources and update the record" ;; N) ok "$msg" ;; esac
    done <<EOF
$out
EOF
fi

if [ -n "$FAILED" ]; then
    echo "DEP_AUDIT_FAIL:$(echo "$FAILED" | tr '\n' ' ')"
    exit 1
fi
if [ -n "$SKIPPED" ]; then
    echo "DEP_AUDIT_SKIP: $SKIPPED"
    exit 0
fi
echo "DEP_AUDIT_OK: engine pin, patch digests, Rust + JS advisories, review window"
