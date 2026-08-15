#!/usr/bin/env bash
# G-TRAY — background residency: the client keeps serving after its window is
# closed, and can always be got back.
#
# A node in this product is a background service. Closing its window must not
# stop it, and hiding that window must never leave a process the user can see in
# Task Manager and cannot reach. Four cases, in that order of importance:
#
#   1. the shell reports whether a tray icon actually exists, and never claims
#      hiding is allowed without one (the fail-safe, checked both ways)
#   2. closing the window hides it AND the process survives
#   3a. window geometry is written on the way out
#   3b. and used on the way back in
#   4. "start in the tray" launches with no window, but with an icon
#
# Case 1 matters most on Linux, where tray support depends on an AppIndicator
# implementation being installed. When there is none this gate does not fail —
# it asserts the fallback instead: hiding must be refused, so the X button quits
# like an ordinary app.
#
# WHERE IT RUNS. `IDLETOKEN_CLIENT_NODE=<ssh alias>` runs the client on that
# machine; unset runs it locally. Windows is the case that matters — "the tray"
# there is the notification area, and close-to-tray is how most people put an
# app away on that platform.
#
# Cases 3 and 4 write the shell's own preferences file (window.json), which is
# what the front end writes when the user flips those switches. The user's copy
# is saved and put back afterwards.
#
# Last-line contract (used by acceptance.sh):
#   TRAY_GATE_OK
#   TRAY_GATE_FAIL: <reason>
#   TRAY_GATE_SKIP: <reason>
set -u
cd "$(dirname "$0")/.." || exit 1
TMP="${TMPDIR:-/tmp}/idletoken-tray-gate.$$"
PREFS_EXISTED=""

fail() { echo "TRAY_GATE_FAIL: $*"; exit 1; }
skip() { echo "TRAY_GATE_SKIP: $*"; exit 0; }

cleanup() {
    # Put the user's own preferences back, whatever happened.
    if [ -n "$PREFS_EXISTED" ] && [ -f "$TMP/window.json.user" ]; then
        cg_prefs_push "$TMP/window.json.user" 2>/dev/null
    elif [ -n "${CG_PREFS:-}" ] && [ -z "$PREFS_EXISTED" ]; then
        cg_prefs_rm 2>/dev/null
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT

command -v python3 >/dev/null 2>&1 || skip "python3 not available (needed to read the reports)"
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

cg_prefs_pull "$TMP/window.json.user" && PREFS_EXISTED=1

write_prefs() {  # $1 = start_minimized (true/false)
    python3 - "$TMP/window.json.new" "$1" "$TMP/window.json.user" <<'PY'
import json, os, sys
out, minimized, existing = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
prefs = {}
if os.path.exists(existing):
    try:
        prefs = json.load(open(existing))
    except Exception:
        prefs = {}
prefs.update({"tray_icon": True, "close_to_tray": True,
              "start_minimized": minimized, "remember_window": True})
json.dump(prefs, open(out, "w"))
PY
    cg_prefs_push "$TMP/window.json.new" || fail "could not write window.json on the machine under test"
}

# ---- case 1: what the shell says about its own tray ------------------------
write_prefs false
cg_run "report-shell-window,quit:6000" 60 "$TMP/case1.log" >/dev/null
R=$(cg_report "$TMP/case1.log" shell-window)
[ -n "$R" ] || fail "case 1: the client reported nothing (see $TMP/case1.log)"
TRAY_ALIVE=$(cg_field "$R" trayAlive)
HIDE_ALLOWED=$(cg_field "$R" hideAllowed)
[ "$(cg_field "$R" visible)" = "True" ] || fail "case 1: the window is not visible on a normal launch — $R"
if [ "$TRAY_ALIVE" = "True" ]; then
    [ "$HIDE_ALLOWED" = "True" ] || fail "case 1: a tray icon exists but hiding is refused — $R"
    echo "case 1 ok: tray icon present, hiding allowed"
else
    # The fail-safe. Not a pass-by-default: this is the assertion that the app
    # cannot be hidden into a tray that does not exist.
    [ "$HIDE_ALLOWED" = "False" ] || fail "case 1: NO TRAY ICON but hiding is allowed — the window could be hidden with no way back ($R)"
    echo "case 1 ok: no tray on this desktop, and hiding is correctly refused"
fi

# ---- case 2: closing the window ------------------------------------------
cg_run "close-to-tray:2500,quit:12000" 60 "$TMP/case2.log" >/dev/null
R=$(cg_report "$TMP/case2.log" close-to-tray)
[ -n "$R" ] || fail "case 2: nothing reported after closing the window — the process most likely died with it (see $TMP/case2.log)"
[ "$(cg_field "$R" stillRunning)" = "True" ] || fail "case 2: the shell stopped answering after the window closed — $R"
if [ "$TRAY_ALIVE" = "True" ]; then
    [ "$(cg_field "$R" visibleBefore)" = "True" ] || fail "case 2: the window was not visible before the close — $R"
    [ "$(cg_field "$R" visibleAfter)" = "False" ] || fail "case 2: closing the window did not hide it — $R"
    echo "case 2 ok: the window hid, the process kept running"
else
    echo "case 2 ok (no tray): the close was not intercepted"
fi

# ---- case 3a: the geometry gets written on the way out --------------------
# Starting from a file with no geometry at all, so "it was already there" is
# not what the next case reads back.
cg_prefs_rm
write_prefs false
cg_run "report-shell-window,quit:6000" 60 "$TMP/case3a.log" >/dev/null
cg_prefs_pull "$TMP/window.json.after" || fail "case 3a: no window.json on the machine after the run"
SAVED=$(python3 - "$TMP/window.json.after" <<'PY'
import json, sys
g = (json.load(open(sys.argv[1])) or {}).get("geometry")
print("" if not g else "%dx%d" % (g["width"], g["height"]))
PY
)
[ -n "$SAVED" ] || fail "case 3a: nothing was written to window.json when the app quit"
echo "case 3a ok: geometry saved on exit ($SAVED)"

# ---- case 3b: and it is used on the way back in ---------------------------
# Physical pixels, deliberately not the default size, and comfortably above the
# window's minimum so nothing clamps it.
python3 - "$TMP/window.json.after" "$TMP/window.json.geom" <<'PY'
import json, sys
prefs = json.load(open(sys.argv[1]))
prefs["geometry"] = {"x": 140, "y": 120, "width": 1600, "height": 1200, "maximized": False}
json.dump(prefs, open(sys.argv[2], "w"))
PY
cg_prefs_push "$TMP/window.json.geom" || fail "case 3b: could not write the test geometry"
cg_run "report-shell-window,quit:6000" 60 "$TMP/case3b.log" >/dev/null
R=$(cg_report "$TMP/case3b.log" shell-window)
[ -n "$R" ] || fail "case 3b: the client reported nothing on the second launch (see $TMP/case3b.log)"
printf '%s' "$R" | python3 -c '
import json, sys
g = json.load(sys.stdin).get("geometry")
if not g:
    print("no geometry restored"); sys.exit(1)
# A window manager may adjust by a pixel or two (frame insets, snapping); what
# this rejects is a restore that landed on a different size entirely.
if abs(g["width"] - 1600) > 8 or abs(g["height"] - 1200) > 8:
    print("restored %dx%d, expected 1600x1200" % (g["width"], g["height"])); sys.exit(1)
' || fail "case 3b: the saved window size was not restored — $R"
echo "case 3b ok: saved geometry restored on the next launch"

# ---- case 4: start in the tray -------------------------------------------
if [ "$TRAY_ALIVE" = "True" ]; then
    write_prefs true
    cg_run "report-shell-window,quit:6000" 60 "$TMP/case4.log" >/dev/null
    R=$(cg_report "$TMP/case4.log" shell-window)
    [ -n "$R" ] || fail "case 4: the client reported nothing (see $TMP/case4.log)"
    [ "$(cg_field "$R" visible)" = "False" ] || fail "case 4: 'start in the tray' still opened a window — $R"
    echo "case 4 ok: started with no window, tray present"
else
    echo "case 4 skipped: no tray on this desktop, so start-minimized does not apply"
fi

echo "TRAY_GATE_OK"
