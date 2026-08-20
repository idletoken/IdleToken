#!/usr/bin/env bash
# Fetch, patch and build the pinned llama.cpp engine.
#
# Products (static, self-contained — no dylib/DLL soup next to them):
#   vendor/llama.cpp/build/bin/llama-server       inference + OpenAI API
#   vendor/llama.cpp/build/bin/ggml-rpc-server    worker-side RPC backend
#   vendor/llama.cpp/build/bin/llama-perplexity   distribution-level numeric gate
#
# Platform backends: macOS = Metal, Linux = CUDA. Windows builds via its own
# batch script (MSVC + CUDA), not this file.
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
# ${arr[@]+...} form: macOS ships bash 3.2, where expanding an empty array
# under `set -u` is fatal.
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
    echo "== applying $(basename "$p")"
    git -C "$SRC_DIR" apply --verbose "$p"
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
    -DLLAMA_CURL=OFF          # downloads are the coordinator's job
    # The embedded web UI is fetched from a Hugging Face bucket AT BUILD TIME,
    # and when the pinned tag's asset is unreachable upstream silently falls
    # back to `.../resolve/latest/dist.tar.gz` — unpinned bytes inside a pinned
    # engine. On DGX (2026-08-19) the pinned fetch timed out and the fallback
    # then hung the build outright. We never serve that UI anyway: llama-server
    # is loopback-only behind the coordinator, and the product UI is the Tauri
    # client. Off makes the build hermetic after the git fetch.
    -DLLAMA_BUILD_UI=OFF
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
# ordinary build (the trap already documented in scripts/testbed-lib.sh).
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
        PLATFORM_FLAGS=(-DGGML_CUDA=ON
                        -DCMAKE_CUDA_ARCHITECTURES="${IDLETOKEN_CUDA_ARCHS:-121}")
        NPROC=$(nproc)
        ;;
    *)
        echo "FATAL: unsupported platform $(uname -s) (Windows uses its own batch script)" >&2
        exit 1
        ;;
esac

cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${COMMON_FLAGS[@]}" "${PLATFORM_FLAGS[@]}"
cmake --build "$BUILD_DIR" -j "$NPROC" \
      --target llama-server ggml-rpc-server llama-perplexity

# --- verify -----------------------------------------------------------------
# Stale artifacts from earlier shared-lib builds caused a broken-but-present
# llama-server once (dyld missing-symbol at startup). Actually execute each
# product; existence alone proves nothing.
for bin in llama-server ggml-rpc-server llama-perplexity; do
    [ -x "$BUILD_DIR/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
done
VERSION_LINE=$("$BUILD_DIR/bin/llama-server" --version 2>&1 | grep -m1 'version:') \
    || { echo "FATAL: llama-server does not run" >&2; exit 1; }
case "$VERSION_LINE" in
    *"${PIN_SHA:0:7}"*) ;;
    *) echo "FATAL: built version '$VERSION_LINE' does not match pin $PIN_SHA" >&2; exit 1 ;;
esac

{
    echo "upstream $REPO_URL $PIN_SHA"
    echo "version $VERSION_LINE"
    for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
        echo "patch $(basename "$p") $(shasum -a 256 "$p" | awk '{print $1}')"
    done
} > "$BUILD_DIR/IDLETOKEN_ENGINE_STAMP"

echo "== OK: $VERSION_LINE"
echo "== products in $BUILD_DIR/bin (stamp: IDLETOKEN_ENGINE_STAMP)"
