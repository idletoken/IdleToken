#!/usr/bin/env bash
# Build the installable desktop client on a Linux compute node (.deb + .rpm).
#
# The engine and the client are two processes (design philosophy 17), but the
# *installer* has to carry both: Tauri ships the engine binaries as sidecars.
# This script stages those sidecars under the target-triple names Tauri expects
# and then runs the real bundler. It is the E3-for-the-product half: after this,
# a user installs one package and has a working node — no repo, no toolchain.
#
# Usage:  scripts/build_client_release.sh            # bundle everything
#         scripts/build_client_release.sh --no-bundle  # just compile the shell
#
# Contract: last line is CLIENT_RELEASE_OK (artifacts listed above it) or
# CLIENT_RELEASE_FAIL: <reason>.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
export PATH="$HOME/.cargo/bin:$PATH"
export CI=true          # pnpm refuses to touch node_modules without a TTY otherwise

BUNDLE=1
[ "${1:-}" = "--no-bundle" ] && BUNDLE=0

fail() { echo "CLIENT_RELEASE_FAIL: $1"; exit 1; }

command -v cargo >/dev/null 2>&1 || fail "no cargo on PATH (need the Rust toolchain)"
command -v pnpm  >/dev/null 2>&1 || fail "no pnpm on PATH (need node + pnpm for the frontend)"

TRIPLE=$(rustc -vV | awk '/^host:/{print $2}')
[ -n "$TRIPLE" ] || fail "could not read the rustc host triple"
echo "target triple: $TRIPLE"

# --- rebuild our own binaries against a portable CPU baseline -------------
# The Makefile defaults to -march/-mcpu=native, which is right for a dev build
# on the machine that will run it and wrong for every installer. Staging
# whatever `make` last left behind is how an AVX-512 coordinator reached the
# 0.1.63/0.1.64 x86 packages: this builder is a Xeon Gold 6530, so `native`
# emitted 284 zmm references into idletoken-coord, and every user CPU without
# AVX-512 — all AMD before Zen 4, every Intel consumer part since AVX-512 was
# fused off — would take SIGILL on launch. Nothing downstream caught it: the
# portability check below only reads the llama.cpp build, not our own binaries.
#
# package_client_mac.sh has always done this (`make -B
# NATIVE_CPU_FLAG=-mcpu=apple-m1`); this path simply never grew the same step.
#
# -B on purpose: make cannot see that the flags changed, so without it a warm
# tree keeps yesterday's native objects. Same reasoning as the verify-key pin
# in the mac script, which shipped unpinned once for exactly this reason.
# A caller that already chose a baseline wins: package_client_linux.sh sets one
# per architecture before it ever gets here, and silently widening its choice
# would be worse than doing nothing. The values below match that script's, so
# the direct path and the driven path ship the same bytes.
if [ -n "${NATIVE_CPU_FLAG:-}" ]; then
    PORTABLE_CPU=$NATIVE_CPU_FLAG
else
    case "$(uname -m)" in
        x86_64)  PORTABLE_CPU=-march=x86-64 ;;    # the original x86-64 baseline
        aarch64) PORTABLE_CPU=-march=armv8-a ;;   # the floor every 64-bit ARM has
        *)       fail "no portable CPU baseline defined for $(uname -m) — add one rather than shipping a native build" ;;
    esac
fi
echo "  CPU baseline for our binaries: $PORTABLE_CPU"
# Pin the platform verify key here too. -B discards whatever the tree had, so a
# rebuild that does not re-pin produces a coordinator that refuses to enable
# sharing on every machine that installs it (overflow.c RULE 3). The gate below
# catches that, and did on the first run of this step — but the right place to
# get it right is where the rebuild happens, exactly as package_client_mac.sh
# does it. The environment still wins, for dev-gateway builds.
if [ -z "${IDLETOKEN_PLATFORM_VERIFY_KEY_B64:-}" ] && [ -f scripts/platform-verify-key.b64 ]; then
    IDLETOKEN_PLATFORM_VERIFY_KEY_B64=$(cat scripts/platform-verify-key.b64)
fi
IDLETOKEN_PLATFORM_VERIFY_KEY_B64="${IDLETOKEN_PLATFORM_VERIFY_KEY_B64:-}" \
make -B NATIVE_CPU_FLAG="$PORTABLE_CPU" coord worker >/dev/null \
    || fail "make coord worker failed at $PORTABLE_CPU"
make -f Makefile.platform >/dev/null || fail "make -f Makefile.platform failed"
# Ask the binaries, not the flags. A baseline that was passed but did not reach
# the object file looks identical from here otherwise.
if command -v objdump >/dev/null 2>&1; then
    for b in idletoken-coord idletoken-worker build/idletoken-platform-agent; do
        [ -f "$b" ] || continue
        case "$(uname -m)" in
            x86_64)
                bad=$(objdump -d "$b" 2>/dev/null \
                    | grep -cE '%zmm|vmovdqu64|vpternlog' || true) ;;
            *)  bad=0 ;;
        esac
        [ "${bad:-0}" -eq 0 ] \
            || fail "$b still carries builder-only CPU instructions ($bad hits) — it would SIGILL on a user machine"
    done
    echo "  our binaries carry no builder-only CPU instructions"
fi

# --- stage engine sidecars ------------------------------------------------
# scripts/stage_sidecars.sh is the single source of truth for what a client
# ships (coord, worker, platform-agent, and — since the v2 llama.cpp pivot —
# the pinned idletoken-server + idletoken-rpc-server). This script used to carry its own copy of the
# staging and it drifted: it kept shipping bundles without idletoken-server after
# stage_sidecars.sh already required it.
out=$(scripts/stage_sidecars.sh) || { echo "$out"; fail "sidecar staging failed"; }
echo "$out" | sed 's/^/  /'

# --- the coordinator must carry the pinned platform verify key --------------
# Without it, nobody who installs this package can ever switch sharing on
# (overflow.c RULE 3). Every release up to 0.1.5 shipped unpinned because
# nothing set the variable; 0.1.19's first macOS bundle shipped unpinned even
# though something did, because make had no reason to recompile overflow.c.
# So ask the binary rather than the build environment. Its only network target
# refuses instantly: a PINNED coordinator prints "overflow: on", an unpinned one
# refuses before any network I/O. Demanding the positive line means a probe that
# stops producing overflow output at all fails the build instead of passing it.
if [ -x ./idletoken-coord ]; then
    if ! ./idletoken-coord --overflow-url http://127.0.0.1:1 --overflow-key pin-probe \
           --model pin-probe-sentinel 2>&1 | grep -q "overflow: on"; then
        fail "idletoken-coord has no pinned platform verify key — rebuild with IDLETOKEN_PLATFORM_VERIFY_KEY_B64 set (scripts/platform-verify-key.b64 is the default)"
    fi
fi

# --- stage licences -----------------------------------------------------------
# The sidecars above carry vendored third-party code (ds4 = MIT, rax = BSD
# 3-Clause), and both licences require the notice to travel with a BINARY
# distribution; Apache-2.0 section 4(d) says the same about our own NOTICE. The
# installer is a binary distribution, so the texts have to be inside it —
# tauri.conf.json `bundle.resources` picks up this directory.
#
# Staged (copied) rather than referenced across the tree: Tauri resolves
# resources relative to src-tauri, and a `../../LICENSE` that silently resolves
# to nothing would ship an installer with no licences and no error. Copying
# fails loudly here instead.
LIC=client/src-tauri/licenses
mkdir -p "$LIC"
cp -f "$ROOT/LICENSE" "$LIC/LICENSE.txt"       || fail "could not stage LICENSE"
cp -f "$ROOT/NOTICE"  "$LIC/NOTICE.txt"        || fail "could not stage NOTICE"
cp -f "$ROOT/vendor/ds4/LICENSE" "$LIC/ds4-MIT.txt" || fail "could not stage the ds4 licence"
# llama.cpp is MIT and is shipped as the idletoken-server sidecar (v2 pivot).
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"
echo "  staged licences -> $LIC"

# --- stage the redistributable Linux CUDA runtime ---------------------------
# A release package must run on a machine that has an NVIDIA driver but no
# CUDA Toolkit. The build nodes both have complete toolkits, so merely running
# the staged engine here is a false oracle: the loader quietly borrows
# /usr/local/cuda. Derive the exact SONAMEs from the built engine, copy the
# matching runtime + cuBLAS files into a private package directory, and rewrite
# only the staged engine copies to resolve that directory via $ORIGIN.
command -v readelf >/dev/null 2>&1 || fail "readelf is required to audit the Linux engine"
command -v ldd >/dev/null 2>&1 || fail "ldd is required to resolve the pinned CUDA runtime"
command -v readlink >/dev/null 2>&1 || fail "readlink is required to resolve CUDA library symlinks"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required to record engine digests"

case "$TRIPLE" in
    x86_64-unknown-linux-gnu)  EXPECTED_CUDA_MAJOR=12 ;;
    aarch64-unknown-linux-gnu) EXPECTED_CUDA_MAJOR=13 ;;
    *) fail "unsupported Linux release target $TRIPLE" ;;
esac

ENGINE_CACHE="$ROOT/vendor/llama.cpp/build/CMakeCache.txt"
[ -f "$ENGINE_CACHE" ] || fail "missing llama.cpp CMakeCache.txt; rebuild the release engine"
grep -Fxq 'GGML_NATIVE:BOOL=OFF' "$ENGINE_CACHE" \
    || fail "llama.cpp was built with builder-native CPU instructions; rebuild with GGML_NATIVE=OFF"
grep -Fxq 'LLAMA_OPENSSL:BOOL=OFF' "$ENGINE_CACHE" \
    || fail "llama.cpp was built with an external OpenSSL dependency; rebuild with LLAMA_OPENSSL=OFF"
native_flags=$(find "$ROOT/vendor/llama.cpp/build" -type f -name flags.make \
    -exec grep -H -E -- '(-march|-mcpu)=native' {} + 2>/dev/null || true)
[ -z "$native_flags" ] \
    || fail "llama.cpp build flags still contain a builder-native CPU target: $native_flags"
if [ "$TRIPLE" = x86_64-unknown-linux-gnu ]; then
    for feature in GGML_SSE42 GGML_AVX GGML_AVX2 GGML_BMI2 GGML_FMA GGML_F16C; do
        grep -Fxq "$feature:BOOL=OFF" "$ENGINE_CACHE" \
            || fail "x86 release engine does not pin $feature=OFF"
    done
fi
echo "  engine CPU baseline: portable (GGML_NATIVE=OFF)"

ENGINE_STAGE="$ROOT/client/src-tauri/binaries"
STAGED_SERVER="$ENGINE_STAGE/idletoken-server-$TRIPLE"
STAGED_RPC="$ENGINE_STAGE/idletoken-rpc-server-$TRIPLE"
[ -x "$STAGED_SERVER" ] || fail "missing staged idletoken-server for $TRIPLE"
[ -x "$STAGED_RPC" ] || fail "missing staged idletoken-rpc-server for $TRIPLE"

needed_sonames() {
    readelf -d "$1" | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p'
}
resolved_needed() {
    ldd "$1" | awk -v wanted="$2" '$1 == wanted && $2 == "=>" { print $3; exit }'
}

CUDART_SONAME=$(needed_sonames "$STAGED_SERVER" | grep -E '^libcudart\.so\.[0-9]+$' | head -1)
CUBLAS_SONAME=$(needed_sonames "$STAGED_SERVER" | grep -E '^libcublas\.so\.[0-9]+$' | head -1)
[ "$CUDART_SONAME" = "libcudart.so.$EXPECTED_CUDA_MAJOR" ] \
    || fail "engine needs ${CUDART_SONAME:-no libcudart}, expected libcudart.so.$EXPECTED_CUDA_MAJOR for $TRIPLE"
[ "$CUBLAS_SONAME" = "libcublas.so.$EXPECTED_CUDA_MAJOR" ] \
    || fail "engine needs ${CUBLAS_SONAME:-no libcublas}, expected libcublas.so.$EXPECTED_CUDA_MAJOR for $TRIPLE"
for engine in "$STAGED_SERVER" "$STAGED_RPC"; do
    needed=$(needed_sonames "$engine")
    printf '%s\n' "$needed" | grep -Fxq "$CUDART_SONAME" \
        || fail "$(basename "$engine") does not link $CUDART_SONAME"
    printf '%s\n' "$needed" | grep -Fxq "$CUBLAS_SONAME" \
        || fail "$(basename "$engine") does not link $CUBLAS_SONAME"
    printf '%s\n' "$needed" | grep -Fxq 'libcuda.so.1' \
        || fail "$(basename "$engine") does not link the NVIDIA driver ABI libcuda.so.1"
done

CUDART_SRC=$(resolved_needed "$STAGED_SERVER" "$CUDART_SONAME")
CUBLAS_SRC=$(resolved_needed "$STAGED_SERVER" "$CUBLAS_SONAME")
[ -f "${CUDART_SRC:-}" ] || fail "could not resolve $CUDART_SONAME from the built engine"
[ -f "${CUBLAS_SRC:-}" ] || fail "could not resolve $CUBLAS_SONAME from the built engine"
CUBLASLT_SONAME=$(needed_sonames "$CUBLAS_SRC" | grep -E '^libcublasLt\.so\.[0-9]+$' | head -1)
[ "$CUBLASLT_SONAME" = "libcublasLt.so.$EXPECTED_CUDA_MAJOR" ] \
    || fail "$CUBLAS_SONAME needs ${CUBLASLT_SONAME:-no libcublasLt}, expected libcublasLt.so.$EXPECTED_CUDA_MAJOR"
CUBLASLT_SRC=$(resolved_needed "$CUBLAS_SRC" "$CUBLASLT_SONAME")
[ -f "${CUBLASLT_SRC:-}" ] || fail "could not resolve $CUBLASLT_SONAME from $CUBLAS_SONAME"

RUNTIME_ROOT="$ROOT/client/src-tauri/runtime/linux"
CUDA_STAGE="$RUNTIME_ROOT/cuda"
DIGEST_STAGE="$RUNTIME_ROOT/engine-digests"
reset_runtime_stage() {
    case "$1" in
        "$RUNTIME_ROOT"/*) ;;
        *) fail "refusing unsafe runtime staging path $1" ;;
    esac
    if [ -L "$1" ] || { [ -e "$1" ] && [ ! -d "$1" ]; }; then
        fail "unexpected non-directory runtime staging path $1"
    fi
    if [ -d "$1" ]; then
        rm -rf -- "$1"
    fi
    mkdir -p "$1" || fail "could not create runtime staging path $1"
}
reset_runtime_stage "$CUDA_STAGE"
reset_runtime_stage "$DIGEST_STAGE"

for pair in \
    "$CUDART_SONAME:$CUDART_SRC" \
    "$CUBLAS_SONAME:$CUBLAS_SRC" \
    "$CUBLASLT_SONAME:$CUBLASLT_SRC"; do
    soname=${pair%%:*}
    src=${pair#*:}
    cp -Lf "$src" "$CUDA_STAGE/$soname" \
        || fail "could not stage $soname from $src"
    printf '  CUDA %-20s %12s bytes  %s…\n' "$soname" \
        "$(wc -c < "$CUDA_STAGE/$soname" | tr -d ' ')" \
        "$(sha256sum "$CUDA_STAGE/$soname" | cut -c1-16)"
done

# NVIDIA lists the CUDA runtime and BLAS dynamic libraries as redistributable,
# but the Toolkit terms must travel with them. Locate the EULA belonging to the
# exact Toolkit tree that supplied libcudart; never copy a similarly named file
# from whichever /usr/local/cuda symlink happens to be current.
CUDA_LIB_DIR=$(dirname "$(readlink -f "$CUDART_SRC")")
CUDA_ROOT=$(cd "$CUDA_LIB_DIR/../../.." 2>/dev/null && pwd)
CUDA_EULA="$CUDA_ROOT/EULA.txt"
[ -f "$CUDA_EULA" ] || fail "CUDA EULA missing beside the selected runtime ($CUDA_EULA)"
grep -q 'libcudart\.so' "$CUDA_EULA" \
    || fail "CUDA EULA does not identify libcudart.so as redistributable"
grep -q 'libcublasLt\.so' "$CUDA_EULA" \
    || fail "CUDA EULA does not identify libcublasLt.so as redistributable"
cp -f "$CUDA_EULA" "$LIC/NVIDIA-CUDA-EULA.txt" \
    || fail "could not stage the NVIDIA CUDA EULA"

BUNDLED_CUDA_RPATH='$ORIGIN/../lib/IdleToken/cuda'
if command -v patchelf >/dev/null 2>&1; then
    RPATH_TOOL=patchelf
elif command -v chrpath >/dev/null 2>&1; then
    RPATH_TOOL=chrpath
else
    fail "need patchelf or chrpath to make the Linux package use its bundled CUDA runtime"
fi
for final_name in idletoken-server idletoken-rpc-server; do
    staged="$ENGINE_STAGE/$final_name-$TRIPLE"
    if [ "$RPATH_TOOL" = patchelf ]; then
        patchelf --set-rpath "$BUNDLED_CUDA_RPATH" "$staged" \
            || fail "patchelf could not set the bundled CUDA RUNPATH on $final_name"
    else
        chrpath -r "$BUNDLED_CUDA_RPATH" "$staged" >/dev/null \
            || fail "chrpath could not set the bundled CUDA RUNPATH on $final_name"
    fi
    got_rpath=$(readelf -d "$staged" | sed -n 's/.*Library \(run\|r\)path: \[\([^]]*\)\].*/\2/p' | head -1)
    [ "$got_rpath" = "$BUNDLED_CUDA_RPATH" ] \
        || fail "$final_name RUNPATH is '$got_rpath', expected '$BUNDLED_CUDA_RPATH'"
    hash=$(sha256sum "$staged" | awk '{print $1}') \
        || fail "could not hash patched $final_name"
    printf '%s  %s-%s\n' "$hash" "$final_name" "$TRIPLE" > "$staged.sha256" \
        || fail "could not refresh the staged digest for $final_name"
    printf '%s  %s\n' "$hash" "$final_name" > "$DIGEST_STAGE/$final_name.sha256" \
        || fail "could not stage the installed digest for $final_name"
done
echo "  staged CUDA $EXPECTED_CUDA_MAJOR runtime + engine RUNPATH + NVIDIA EULA"

# --- build ----------------------------------------------------------------
cd client || fail "no client/ directory"
pnpm install >/tmp/client-release-install.log 2>&1 || fail "pnpm install failed (see /tmp/client-release-install.log)"

if [ "$BUNDLE" = 0 ]; then
    (cd src-tauri && cargo build --release) || fail "cargo build --release failed"
    echo "shell binary: $ROOT/client/src-tauri/target/release/idletoken-client"
    echo CLIENT_RELEASE_OK
    exit 0
fi

# `tauri build` runs beforeBuildCommand (pnpm build:release) itself — that is
# what injects the production platform URL into the shipped frontend.
#
# Linux releases are native package-manager installers only. Keep this list
# fail closed so a stale environment cannot quietly resurrect AppImage or an
# updater artifact in a public release.
BUNDLES="${IDLETOKEN_BUNDLES:-deb,rpm}"
case ",$BUNDLES," in
    ,deb,|,rpm,|,deb,rpm,|,rpm,deb,) ;;
    *) fail "IDLETOKEN_BUNDLES must contain only deb and/or rpm (got '$BUNDLES')" ;;
esac
# ⚠ not `| tail`: the pipe exit code is tail's, and a bundler that failed
# AFTER producing the .deb sailed through as CLIENT_RELEASE_OK (hit 2026-08-15;
# same trap as the repo-wide "never read an exit code through a pipe" rule).
BUILD_LOG=$(mktemp /tmp/idletoken-tauri-build.XXXXXX)
if ! pnpm tauri build --bundles "$BUNDLES" > "$BUILD_LOG" 2>&1; then
    tail -30 "$BUILD_LOG"; rm -f "$BUILD_LOG"
    fail "tauri build failed"
fi
tail -30 "$BUILD_LOG"; rm -f "$BUILD_LOG"

# --- report artifacts -----------------------------------------------------
BDIR="src-tauri/target/release/bundle"
[ -d "$BDIR" ] || fail "no bundle directory at $BDIR"
found=0
echo "--- artifacts ---"
while IFS= read -r f; do
    found=1
    printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done < <(find "$BDIR" -type f \( -name '*.deb' -o -name '*.rpm' \) | sort)
[ "$found" = 1 ] || fail "bundler produced no Linux package"

# No automatic provenance, signature, updater archive or feed is emitted here.
# The release contract is the native installer files listed above, and only
# those files are uploaded to GitHub.

echo CLIENT_RELEASE_OK
