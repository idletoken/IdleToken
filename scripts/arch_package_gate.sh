#!/usr/bin/env bash
# Install an IdleToken .pkg.tar.zst in a clean official Arch Linux userspace,
# validate ALPM ownership and runtime linkage, smoke the desktop shell under
# Xvfb, then uninstall it and prove the product files are gone.
#
# Usage: scripts/arch_package_gate.sh <idletoken-bin-VERSION-1-x86_64.pkg.tar.zst>
# Set IDLETOKEN_ARCH_GATE_MIRROR to an HTTPS Arch mirror base when the
# official Fastly/Geo mirrors are unavailable from the validation host.
# Last line: ARCH_GATE_OK, or ARCH_GATE_FAIL: reason.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

fail() {
    echo "ARCH_GATE_FAIL: $1" >&2
    exit 1
}

[ "$#" -eq 1 ] || fail "usage: scripts/arch_package_gate.sh <package.pkg.tar.zst>"
PKG_INPUT=$1
[ -f "$PKG_INPUT" ] || fail "package not found: $PKG_INPUT"
PKG_DIR=$(cd "$(dirname "$PKG_INPUT")" && pwd)
PKG="$PKG_DIR/$(basename "$PKG_INPUT")"

RUNTIME=${IDLETOKEN_ARCH_GATE_RUNTIME:-auto}
if [ "$RUNTIME" = auto ]; then
    if command -v docker >/dev/null 2>&1; then
        RUNTIME=docker
    elif command -v podman >/dev/null 2>&1; then
        RUNTIME=podman
    else
        fail "Docker or Podman is required for the clean Arch install gate"
    fi
fi
case "$RUNTIME" in
    docker|podman) command -v "$RUNTIME" >/dev/null 2>&1 || fail "$RUNTIME is unavailable" ;;
    *) fail "IDLETOKEN_ARCH_GATE_RUNTIME must be auto, docker, or podman" ;;
esac

if [ -n "${IDLETOKEN_ARCH_EXPECT_ENGINE_PIN:-}" ]; then
    PIN_SHORT=${IDLETOKEN_ARCH_EXPECT_ENGINE_PIN:0:7}
else
    PIN_SHORT=$(awk 'NR == 1 { print substr($2, 1, 7) }' scripts/llamacpp-patches/UPSTREAM)
fi
[ -n "$PIN_SHORT" ] || fail "could not determine the expected engine pin"

ARCH_IMAGE=${IDLETOKEN_ARCH_GATE_IMAGE:-docker.io/library/archlinux@sha256:f3691b4dde62ba4c4b6f0ae2c1fbf28e8c0c8c4b9a35c7e06dc1f70e21aa29f6}
ARCH_MIRROR=${IDLETOKEN_ARCH_GATE_MIRROR:-}
container_args=(run --rm --platform linux/amd64
    -e "PIN_SHORT=$PIN_SHORT"
    -v "$PKG:/packages/idletoken.pkg.tar.zst:ro")
if [ -n "$ARCH_MIRROR" ]; then
    case "$ARCH_MIRROR" in
        https://*) container_args+=(-e "ARCH_MIRROR=${ARCH_MIRROR%/}") ;;
        *) fail "IDLETOKEN_ARCH_GATE_MIRROR must be an HTTPS URL" ;;
    esac
fi
if [ "$(uname -m)" != x86_64 ]; then
    container_args+=(--security-opt seccomp=unconfined)
fi

"$RUNTIME" "${container_args[@]}" "$ARCH_IMAGE" bash -lc '
set -euo pipefail

if [[ -n ${ARCH_MIRROR:-} ]]; then
    printf "Server = %s/\$repo/os/\$arch\n" "$ARCH_MIRROR" >/etc/pacman.d/mirrorlist
fi

# Positive control: prove pacman rejects something that is not an ALPM package
# before trusting it to certify the real artifact.
if pacman -Qip /packages/not-a-package.pkg.tar.zst >/dev/null 2>&1; then
    echo "pacman positive control accepted a nonexistent package" >&2
    exit 1
fi

pacman -Sy --disable-sandbox --noconfirm >/tmp/pacman-sync.log
pacman -U --disable-sandbox --noconfirm /packages/idletoken.pkg.tar.zst >/tmp/pacman-install.log
pacman -S --disable-sandbox --noconfirm namcap xorg-server-xvfb >/tmp/pacman-desktop.log

namcap /packages/idletoken.pkg.tar.zst | tee /tmp/namcap.log
if grep -Fq " E: " /tmp/namcap.log; then
    echo "namcap reported package errors" >&2
    exit 1
fi

pacman -Q idletoken-bin
pacman -Qkk idletoken-bin | tee /tmp/pacman-integrity.log
grep -Fq "0 altered files" /tmp/pacman-integrity.log

cd /usr/bin
sha256sum -c idletoken-server.sha256
sha256sum -c idletoken-rpc-server.sha256
set +e
timeout 15s dbus-run-session -- xvfb-run -a env \
    WEBKIT_DISABLE_DMABUF_RENDERER=1 \
    NO_PROXY=127.0.0.1,localhost \
    /usr/bin/idletoken-client >/tmp/client-smoke.log 2>&1
ui_rc=$?
set -e
if [[ $ui_rc -ne 124 && $ui_rc -ne 143 ]]; then
    tail -40 /tmp/client-smoke.log >&2
    printf "desktop client exited during the 15-second Arch smoke (rc=%s)\n" "$ui_rc" >&2
    exit 1
fi
if grep -q "error while loading shared libraries" /tmp/client-smoke.log; then
    tail -40 /tmp/client-smoke.log >&2
    exit 1
fi
echo "desktop shell remained alive for the 15-second Arch smoke"

# A GPU-less container has no host driver. Install Arch user-space driver files
# only after the Xvfb smoke (the NVIDIA GLX module has no device in this
# container), then use them for final loader and --version checks. A real GPU
# probe remains a separate hardware gate and may not be inferred from this test.
pacman -S --disable-sandbox --noconfirm nvidia-utils >/tmp/pacman-nvidia.log
for binary in idletoken-client idletoken-coord idletoken-worker \
              idletoken-platform-agent idletoken-server idletoken-rpc-server; do
    unresolved=$(ldd "/usr/bin/$binary" 2>&1 | grep "not found" || true)
    if [[ -n $unresolved ]]; then
        printf "%s has unresolved libraries:\n%s\n" "$binary" "$unresolved" >&2
        exit 1
    fi
done

for engine in idletoken-server idletoken-rpc-server; do
    runpath=$(readelf -d "/usr/bin/$engine" | sed -n "s/.*Library runpath: \[\([^]]*\)\].*/\1/p")
    [[ $runpath == \$ORIGIN/../lib/IdleToken/cuda ]]
    loader=$(ldd "/usr/bin/$engine")
    for library in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
        resolved=$(awk -v wanted="$library" "\$1 == wanted && \$2 == \"=>\" {print \$3; exit}" <<<"$loader")
        [[ $resolved == /usr/bin/../lib/IdleToken/cuda/* || $resolved == /usr/lib/IdleToken/cuda/* ]]
    done
done

version_line=$(/usr/bin/idletoken-server --version 2>&1 | grep -m1 "version:")
[[ $version_line == *"$PIN_SHORT"* ]] || {
    printf "engine version mismatch: expected %s in %s\n" "$PIN_SHORT" "$version_line" >&2
    exit 1
}
printf "engine: %s\n" "$version_line"

pacman -R --noconfirm idletoken-bin >/tmp/pacman-remove.log
for path in /usr/bin/idletoken-client /usr/bin/idletoken-server /usr/lib/IdleToken; do
    [[ ! -e $path ]] || { echo "uninstall left $path" >&2; exit 1; }
done
' || fail "clean Arch install gate failed"

echo ARCH_GATE_OK
