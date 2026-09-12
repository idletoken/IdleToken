#!/usr/bin/env bash
# Fetch, patch and build the pinned llama.cpp engine.
#
# Products (static except for operating-system APIs and Linux's redistributable
# CUDA user-space runtime, which the client release builder bundles privately):
#   vendor/llama.cpp/build/bin/llama-server       inference + OpenAI API
#   vendor/llama.cpp/build/bin/ggml-rpc-server    worker-side RPC backend
#
# Platform backends: macOS = Metal, Linux = CUDA. Linux arm64 defaults to the
# DGX's sm_121; Linux x86_64 defaults to every CUDA 12.8 architecture at or
# above the product floor. Windows builds via its own batch script.
#
# No silent fallback (v2 hard invariant #4): every failure here is fatal.
# If CUDA/Metal/cmake is missing, this script exits red — it never downgrades
# to a CPU build to "keep things green".
#
# Usage:
#   scripts/build_llamacpp.sh              fetch + patch + build + verify
#   scripts/build_llamacpp.sh --fetch-only clean checkout at the pinned SHA
#                                          (for regenerating patches)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH_DIR="$ROOT/scripts/llamacpp-patches"
SRC_DIR="${IDLETOKEN_LLAMACPP_SRC:-$ROOT/vendor/llama.cpp}"
BUILD_DIR="$SRC_DIR/build"

read -r REPO_URL PIN_SHA PIN_TAG _rest < "$PATCH_DIR/UPSTREAM"
[ -n "$PIN_SHA" ] || { echo "FATAL: cannot parse $PATCH_DIR/UPSTREAM" >&2; exit 1; }
# The third field is the upstream release tag, and it is required: it is what
# makes the engine's own version string a function of the pin instead of a
# function of how this machine happened to clone.
#
# llama.cpp derives "build N" from `git rev-list --count HEAD` and the commit
# from `git rev-parse --short HEAD`. Both depend on the checkout: the shallow
# fetch this script performs yields `build 1` and a 7-char hash, while a full
# clone of the same commit yields `build 10502` and a 9-char hash. Measured
# 2026-08-19 with DGX and the control Mac at the identical pin:
#     DGX : version: 0.1.2-dev (build 1, commit 0adcc3b)
#     Mac : version: 0.1.2-dev (build 10502, commit 0adcc3bb5)
# G-ENGINE-VER compares that string for equality across the cluster, so two
# nodes running byte-identical engines refuse to form a cluster and the refusal
# names a machine that has nothing to upgrade. Forcing both values makes the
# string depend only on UPSTREAM. 7 chars because that is the width both build
# scripts already assert the pin against.
case "$PIN_TAG" in
    b[0-9]*) ;;
    *) echo "FATAL: $PATCH_DIR/UPSTREAM has no release tag in field 3 (got '${PIN_TAG:-}')" >&2
       echo "       Expected: <url> <full-sha> b<build-number>" >&2; exit 1 ;;
esac
PIN_BUILD="${PIN_TAG#b}"

# --- fetch ------------------------------------------------------------------
if [ ! -d "$SRC_DIR/.git" ]; then
    mkdir -p "$SRC_DIR"
    git -C "$SRC_DIR" init -q
    git -C "$SRC_DIR" remote add origin "$REPO_URL"
fi
if ! git -C "$SRC_DIR" cat-file -e "$PIN_SHA" 2>/dev/null; then
    echo "== fetching $PIN_SHA from $REPO_URL"
    # A mirror, tried only after the real upstream fails. Machines behind a
    # restrictive network cannot reach github.com at all (a rented GPU box in
    # China times out on the TLS handshake), and the pin is a content hash --
    # a mirror either serves those exact bytes or `checkout` below fails, so
    # this cannot quietly substitute a different tree. Same escape hatch the
    # Windows script has had as IDLETOKEN_LLAMACPP_GIT_URL.
    if ! git -C "$SRC_DIR" fetch --depth 1 origin "$PIN_SHA"; then
        [ -n "${IDLETOKEN_LLAMACPP_GIT_URL:-}" ] || {
            echo "FATAL: cannot fetch $PIN_SHA from $REPO_URL." >&2
            echo "       Set IDLETOKEN_LLAMACPP_GIT_URL to a reachable mirror," >&2
            echo "       or copy a checkout that already has the pin into $SRC_DIR." >&2
            exit 1
        }
        echo "== upstream unreachable; retrying from $IDLETOKEN_LLAMACPP_GIT_URL"
        git -C "$SRC_DIR" fetch --depth 1 "$IDLETOKEN_LLAMACPP_GIT_URL" "$PIN_SHA"
    fi
fi
# Reset tracked files to the pinned commit; patches are reapplied below.
# (Local edits in vendor/llama.cpp/ are lost here — capture them as a patch
# first. build/ is untracked and survives.)
git -C "$SRC_DIR" checkout -qf "$PIN_SHA"

# --- patch ------------------------------------------------------------------
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
shopt -u nullglob
# `checkout -f` resets tracked paths but deliberately leaves untracked files.
# Once a patch adds a new source/header, the next build would therefore stop at
# "already exists in working directory". Remove only paths that this exact
# patch series declares as additions; never run `git clean`, because build/
# and unrelated local diagnostics also live in this ignored checkout.
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
    while IFS= read -r rel; do
        case "$rel" in
            ""|..|/*|../*|*/..|*/../*)
                echo "FATAL: unsafe added path '$rel' in $(basename "$p")" >&2
                exit 1
                ;;
        esac
        if git -C "$SRC_DIR" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
            echo "FATAL: $(basename "$p") declares tracked upstream path '$rel' as new" >&2
            exit 1
        fi
        if [ -d "$SRC_DIR/$rel" ] && [ ! -L "$SRC_DIR/$rel" ]; then
            echo "FATAL: patch-owned added path is unexpectedly a directory: $rel" >&2
            exit 1
        fi
        rm -f -- "$SRC_DIR/$rel"
    done < <(awk '
        previous == "--- /dev/null" && /^\+\+\+ b\// {
            sub(/^\+\+\+ b\//, ""); print
        }
        { previous = $0 }
    ' "$p")
done
# ${arr[@]+...} form: macOS ships bash 3.2, where expanding an empty array
# under `set -u` is fatal.
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
    echo "== applying $(basename "$p")"
    git -C "$SRC_DIR" apply --verbose "$p"
done

# Upstream's UI provisioner treats an existing dist/ as a higher-priority
# input even when both UI build/download switches are OFF. Without clearing
# these exact ignored cache directories, a repeat build silently embeds the UI
# from an older run. The product never serves this UI; fail rather than follow
# an unexpected symlink or remove anything outside the two known cache paths.
for ui_cache in "$SRC_DIR/tools/ui/dist" "$BUILD_DIR/tools/ui/dist"; do
    if [ -L "$ui_cache" ] || { [ -e "$ui_cache" ] && [ ! -d "$ui_cache" ]; }; then
        echo "FATAL: unexpected non-directory UI cache path: $ui_cache" >&2
        exit 1
    fi
    if [ -d "$ui_cache" ]; then
        echo "== removing stale embedded UI cache: $ui_cache"
        rm -rf -- "$ui_cache"
    fi
done

if [ "${1:-}" = "--fetch-only" ]; then
    echo "== checkout ready at $PIN_SHA with ${#PATCHES[@]} patch(es): $SRC_DIR"
    exit 0
fi

# --- configure --------------------------------------------------------------
COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    # Deterministic engine version string — see the PIN_TAG note above.
    -DLLAMA_BUILD_NUMBER="$PIN_BUILD"
    -DLLAMA_BUILD_COMMIT="${PIN_SHA:0:7}"
    -DGGML_RPC=ON
    -DGGML_RPC_TLS=ON         # PSK-TLS inside the RPC transport (patch 0001)
    -DBUILD_SHARED_LIBS=OFF
    # Installers run on other CPUs. A release built on an AVX-512 Xeon or a
    # recent Apple Silicon generation must not execute builder-specific
    # instructions before it ever reaches the GPU/Metal backend.
    -DGGML_NATIVE=OFF
    -DLLAMA_CURL=OFF          # downloads are the coordinator's job
    # llama-server is reachable only through the coordinator's loopback HTTP
    # sidecar. Public HTTPS is not a product path, while enabling it links the
    # macOS binary to a maintainer's Homebrew OpenSSL and breaks clean Macs.
    # RPC transport TLS is independent and remains pinned to mbedTLS above.
    -DLLAMA_OPENSSL=OFF
    # The embedded web UI is fetched from a Hugging Face bucket AT BUILD TIME,
    # and when the pinned tag's asset is unreachable upstream silently falls
    # back to `.../resolve/latest/dist.tar.gz` — unpinned bytes inside a pinned
    # engine. On DGX (2026-08-19) the pinned fetch timed out and the fallback
    # then hung the build outright. We never serve that UI anyway: llama-server
    # is loopback-only behind the coordinator, and the product UI is the Tauri
    # client. Both switches are required: LLAMA_BUILD_UI=OFF only disables the
    # npm build, while LLAMA_USE_PREBUILT_UI otherwise still downloads a bucket
    # artifact and can fall back from the pinned tag to `latest`.
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
)
# The TLS transport patch pulls mbedTLS v3.6.7 through CMake FetchContent,
# i.e. a second trip to github.com -- so a machine that needed the mirror
# above fails again here, several minutes later, with a completely different
# error message. Point this at a local mbedTLS source tree to skip the
# download. The Windows script already accepted this variable; this side
# silently ignored it, which meant "seed the tree" advice only worked on one
# of the two platforms.
#
# Written as `if`, not `[ -n ... ] && COMMON_FLAGS+=(...)`: under `set -e` the
# latter exits the whole script when the variable is unset, i.e. on every
# ordinary build (an explicit if keeps the unset case valid under set -e).
if [ -n "${IDLETOKEN_MBEDTLS_SRC:-}" ]; then
    COMMON_FLAGS+=(-DIDLETOKEN_MBEDTLS_SRC="$IDLETOKEN_MBEDTLS_SRC")
fi
case "$(uname -s)" in
    Darwin)
        PLATFORM_FLAGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
        NPROC=$(sysctl -n hw.ncpu)
        ;;
    Linux)
        # Non-interactive ssh sessions miss the login-shell PATH; find nvcc in
        # the standard install location before failing.
        if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
            export CUDACXX=/usr/local/cuda/bin/nvcc
            export PATH="/usr/local/cuda/bin:$PATH"
        fi
        case "$(uname -m)" in
            x86_64|amd64)
                DEFAULT_CUDA_ARCHS="75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;120"
                # Keep the x86 control/CPU path at the architectural baseline.
                # These are explicit because CMake's defaults differ between
                # a cold GGML_NATIVE=OFF tree and a warm formerly-native tree.
                PLATFORM_FLAGS=(-DGGML_CUDA=ON
                                -DGGML_SSE42=OFF -DGGML_AVX=OFF
                                -DGGML_AVX2=OFF -DGGML_BMI2=OFF
                                -DGGML_FMA=OFF -DGGML_F16C=OFF)
                ;;
            aarch64|arm64)
                DEFAULT_CUDA_ARCHS="75-real;80-real;86-real;87-real;88-real;89-real;90-real;100-real;103-real;110-real;120-real;121"
                PLATFORM_FLAGS=(-DGGML_CUDA=ON)
                ;;
            *)
                echo "FATAL: unsupported Linux architecture $(uname -m)" >&2
                exit 1
                ;;
        esac
        PLATFORM_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="${IDLETOKEN_CUDA_ARCHS:-$DEFAULT_CUDA_ARCHS}")
        NPROC=$(nproc)
        ;;
    *)
        echo "FATAL: unsupported platform $(uname -s) (Windows uses its own batch script)" >&2
        exit 1
        ;;
esac

cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${COMMON_FLAGS[@]}" "${PLATFORM_FLAGS[@]}"
cmake --build "$BUILD_DIR" -j "$NPROC" \
      --target llama-server ggml-rpc-server

# --- verify -----------------------------------------------------------------
# Stale artifacts from earlier shared-lib builds caused a broken-but-present
# llama-server once (dyld missing-symbol at startup). Actually execute each
# product; existence alone proves nothing.
for bin in llama-server ggml-rpc-server; do
    [ -x "$BUILD_DIR/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
done
VERSION_LINE=$("$BUILD_DIR/bin/llama-server" --version 2>&1 | grep -m1 'version:') \
    || { echo "FATAL: llama-server does not run" >&2; exit 1; }
case "$VERSION_LINE" in
    *"${PIN_SHA:0:7}"*) ;;
    *) echo "FATAL: built version '$VERSION_LINE' does not match pin $PIN_SHA" >&2; exit 1 ;;
esac

# A release executable may use OS libraries, but never a package manager from
# the maintainer's machine. This exact bug produced a small DMG whose server
# worked here only because /opt/homebrew/opt/openssl@3 happened to exist.
if [ "$(uname -s)" = "Darwin" ]; then
    command -v otool >/dev/null 2>&1 \
        || { echo "FATAL: otool is required to audit macOS engine dependencies" >&2; exit 1; }
    for bin in llama-server ggml-rpc-server; do
        BAD_DYLIBS=$(otool -L "$BUILD_DIR/bin/$bin" | awk 'NR > 1 { print $1 }' \
            | grep -Ev '^(/System/Library/|/usr/lib/)' || true)
        if [ -n "$BAD_DYLIBS" ]; then
            echo "FATAL: $bin depends on non-system macOS libraries:" >&2
            printf '%s\n' "$BAD_DYLIBS" >&2
            exit 1
        fi
    done
    echo "== macOS dependency audit: system frameworks/libraries only"
fi

{
    echo "upstream $REPO_URL $PIN_SHA"
    echo "version $VERSION_LINE"
    for arg in "${COMMON_FLAGS[@]}" "${PLATFORM_FLAGS[@]}"; do
        echo "cmake-arg $arg"
    done
    for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
        echo "patch $(basename "$p") $(shasum -a 256 "$p" | awk '{print $1}')"
    done
} > "$BUILD_DIR/IDLETOKEN_ENGINE_STAMP"

echo "== OK: $VERSION_LINE"
echo "== products in $BUILD_DIR/bin (stamp: IDLETOKEN_ENGINE_STAMP)"
