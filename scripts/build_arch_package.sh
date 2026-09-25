#!/usr/bin/env bash
# Unsupported packaging experiment: build an Arch Linux package from an
# already verified x86_64 Debian payload. This helper is retained for community
# adaptation, but is not called by the official release build, does not produce
# a supported release asset, and carries no project support commitment. The
# conversion deliberately preserves every payload byte.
#
# Usage: scripts/build_arch_package.sh <IdleToken_VERSION_amd64.deb> [output-dir]
#
# The builder uses a local non-root makepkg when available. Otherwise it uses a
# pinned official Arch base-devel container through Docker or Podman. The last
# line is ARCH_PACKAGE_OK <absolute-package-path>, or ARCH_PACKAGE_FAIL: reason.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

fail() {
    echo "ARCH_PACKAGE_FAIL: $1" >&2
    exit 1
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] \
    || fail "usage: scripts/build_arch_package.sh <amd64.deb> [output-dir]"

DEB_INPUT=$1
[ -f "$DEB_INPUT" ] || fail "Debian package not found: $DEB_INPUT"
DEB_DIR=$(cd "$(dirname "$DEB_INPUT")" && pwd)
DEB="$DEB_DIR/$(basename "$DEB_INPUT")"

OUT_INPUT=${2:-$ROOT/client/src-tauri/target/release/bundle/arch}
mkdir -p "$OUT_INPUT" || fail "could not create output directory: $OUT_INPUT"
OUT_DIR=$(cd "$OUT_INPUT" && pwd)

for tool in ar awk bsdtar diff grep gzip sed; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

CONTROL_MEMBER=$(ar t "$DEB" | awk '/^control\.tar/{print; exit}')
DATA_MEMBER=$(ar t "$DEB" | awk '/^data\.tar/{print; exit}')
[ -n "$CONTROL_MEMBER" ] || fail "Debian package has no control archive"
[ -n "$DATA_MEMBER" ] || fail "Debian package has no data archive"

CONTROL=$(ar p "$DEB" "$CONTROL_MEMBER" | bsdtar -xOf - control) \
    || fail "could not read Debian control metadata"
control_value() {
    printf '%s\n' "$CONTROL" | awk -v key="$1" '$1 == key ":" {print $2; exit}'
}

DEB_PACKAGE=$(control_value Package)
VERSION=$(control_value Version)
DEB_ARCH=$(control_value Architecture)
[ "$DEB_PACKAGE" = "idle-token" ] \
    || fail "expected Debian package idle-token, got '${DEB_PACKAGE:-missing}'"
[ "$DEB_ARCH" = "amd64" ] \
    || fail "experimental Arch packaging accepts x86_64 only; Debian payload is '${DEB_ARCH:-missing}'"
case "$VERSION" in
    ''|*[!0-9A-Za-z.+_]*) fail "Debian version '$VERSION' is not a safe Arch pkgver" ;;
esac

DEB_NAME="IdleToken_${VERSION}_amd64.deb"
DEB_SHA256=$(sha256_file "$DEB")
TEMPLATE="$ROOT/packaging/arch/PKGBUILD.in"
[ -f "$TEMPLATE" ] || fail "missing $TEMPLATE"

WORK=$(mktemp -d "$OUT_DIR/.idletoken-arch-build.XXXXXX") \
    || fail "could not create an Arch package work directory"
cleanup() {
    case "$WORK" in
        "$OUT_DIR"/.idletoken-arch-build.*)
            chmod -R u+rwX "$WORK" 2>/dev/null || true
            rm -rf -- "$WORK"
            ;;
        *) echo "ARCH_PACKAGE_FAIL: refusing unsafe cleanup path $WORK" >&2 ;;
    esac
}
trap cleanup EXIT

sed \
    -e "s/@VERSION@/$VERSION/g" \
    -e "s/@DEB_NAME@/$DEB_NAME/g" \
    -e "s/@DEB_SHA256@/$DEB_SHA256/g" \
    "$TEMPLATE" > "$WORK/PKGBUILD" \
    || fail "could not render PKGBUILD"

BUILDER=${IDLETOKEN_ARCH_BUILDER:-auto}
ARCH_PACKAGER=${IDLETOKEN_ARCH_PACKAGER:-IdleToken Packaging Experiment <support@idletoken.ai>}
if [ "$BUILDER" = auto ]; then
    if command -v makepkg >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
        BUILDER=native
    elif command -v docker >/dev/null 2>&1; then
        BUILDER=docker
    elif command -v podman >/dev/null 2>&1; then
        BUILDER=podman
    else
        fail "need non-root makepkg, Docker, or Podman to build the Arch package"
    fi
fi

case "$BUILDER" in
    native)
        command -v makepkg >/dev/null 2>&1 || fail "IDLETOKEN_ARCH_BUILDER=native but makepkg is unavailable"
        [ "$(id -u)" -ne 0 ] || fail "makepkg refuses to run as root; use the container builder"
        cp -f "$DEB" "$WORK/$DEB_NAME" \
            || fail "could not stage the Debian payload for native makepkg"
        (cd "$WORK" && PACKAGER="$ARCH_PACKAGER" makepkg --force --nodeps --noconfirm) \
            || fail "makepkg failed"
        ;;
    docker|podman)
        command -v "$BUILDER" >/dev/null 2>&1 \
            || fail "IDLETOKEN_ARCH_BUILDER=$BUILDER but $BUILDER is unavailable"
        # Pinned consciously: changing the Arch build image changes the
        # experiment's toolchain and must be reviewed independently.
        ARCH_IMAGE=${IDLETOKEN_ARCH_IMAGE:-docker.io/library/archlinux@sha256:8745817f349ed24373341ddb92776209eeec3f0364ea48f7f645ac5800d30a50}
        container_args=(run --rm --platform linux/amd64
            -e "HOST_UID=$(id -u)"
            -e "HOST_GID=$(id -g)"
            -e "ARCH_PACKAGER=$ARCH_PACKAGER"
            -e "DEB_NAME=$DEB_NAME"
            -v "$DEB:/input/$DEB_NAME:ro"
            -v "$WORK:/work"
            -w /work)
        if [ "$(uname -m)" != x86_64 ]; then
            # QEMU user-mode emulation needs these syscalls. Native x86_64
            # Keep the container's default seccomp profile.
            container_args+=(--security-opt seccomp=unconfined)
        fi
        "$BUILDER" "${container_args[@]}" "$ARCH_IMAGE" bash -lc \
            'set -euo pipefail
             useradd --create-home builder
             mkdir -p /build
             cp /work/PKGBUILD /build/
             ln -s "/input/$DEB_NAME" "/build/$DEB_NAME"
             chown -R builder:builder /build
             runuser -u builder -- env PACKAGER="$ARCH_PACKAGER" bash -lc "cd /build && makepkg --force --nodeps --noconfirm"
             pkg=$(printf "%s\n" /build/idletoken-bin-*.pkg.tar.zst | tail -1)
             pacman -Qip "$pkg" >/work/pacman-package-info.txt
             pacman -Qlp "$pkg" >/work/pacman-package-files.txt
             cp "$pkg" /work/
             chown "$HOST_UID:$HOST_GID" /work/pacman-package-info.txt /work/pacman-package-files.txt "/work/$(basename "$pkg")"' \
            || fail "$BUILDER Arch package build or pacman metadata validation failed"
        ;;
    *) fail "IDLETOKEN_ARCH_BUILDER must be auto, native, docker, or podman (got '$BUILDER')" ;;
esac

BUILT="$WORK/idletoken-bin-${VERSION}-1-x86_64.pkg.tar.zst"
[ -f "$BUILT" ] || fail "makepkg produced no expected package for $VERSION"

PKGINFO=$(bsdtar -xOf "$BUILT" .PKGINFO) || fail "Arch package has no readable .PKGINFO"
for required in \
    "pkgname = idletoken-bin" \
    "pkgver = $VERSION-1" \
    "arch = x86_64" \
    "xdata = pkgtype=pkg" \
    "depend = dbus" \
    "depend = glibc>=2.35" \
    "depend = hicolor-icon-theme" \
    "depend = webkit2gtk-4.1"; do
    printf '%s\n' "$PKGINFO" | grep -Fxq "$required" \
        || fail "Arch metadata is missing '$required'"
done

LISTING=$(bsdtar -tf "$BUILT") || fail "could not list the Arch package"
for required in .PKGINFO .BUILDINFO .MTREE \
    usr/bin/idletoken-client usr/bin/idletoken-coord usr/bin/idletoken-worker \
    usr/bin/idletoken-platform-agent usr/bin/idletoken-server \
    usr/bin/idletoken-rpc-server usr/bin/idletoken-server.sha256 \
    usr/bin/idletoken-rpc-server.sha256 usr/lib/IdleToken/licenses/NVIDIA-CUDA-EULA.txt; do
    printf '%s\n' "$LISTING" | grep -Fxq "$required" \
        || fail "Arch package is missing $required"
done
printf '%s\n' "$LISTING" | grep -Eq '(^|/)libcuda\.so\.1$' \
    && fail "Arch package illegally bundles the NVIDIA driver library libcuda.so.1"
bsdtar -xOf "$BUILT" .MTREE | gzip -t \
    || fail "Arch package contains an invalid .MTREE"

# Prove makepkg did not strip, rewrite, omit, or add product files. Generate
# content-addressed mtree manifests directly from both archives rather than
# extracting two multi-gigabyte trees. Paths, object types, modes, ownership,
# symlink targets, sizes, and regular-file bytes all participate in the diff.
# This is the bridge that lets the existing Debian payload gates certify the
# Arch payload without quietly requiring many extra gigabytes of scratch disk.
MTREE_OPTIONS='!all,type,mode,uid,gid,link,size,sha256'
DEB_MANIFEST="$WORK/deb-payload.mtree"
ARCH_MANIFEST="$WORK/arch-payload.mtree"
ar p "$DEB" "$DATA_MEMBER" \
    | bsdtar --format=mtree --options="$MTREE_OPTIONS" -cf - @/dev/stdin \
    | awk '$1 == "./usr" || index($1, "./usr/") == 1' \
    | LC_ALL=C sort > "$DEB_MANIFEST" \
    || fail "could not inventory the Debian payload"
bsdtar --format=mtree --options="$MTREE_OPTIONS" -cf - @"$BUILT" \
    | awk '$1 == "./usr" || index($1, "./usr/") == 1' \
    | LC_ALL=C sort > "$ARCH_MANIFEST" \
    || fail "could not inventory the Arch payload"
[ -s "$DEB_MANIFEST" ] || fail "Debian payload inventory is empty"
[ -s "$ARCH_MANIFEST" ] || fail "Arch payload inventory is empty"
if ! diff -u "$DEB_MANIFEST" "$ARCH_MANIFEST" > "$WORK/payload-diff.txt"; then
    sed -n '1,40p' "$WORK/payload-diff.txt" >&2
    fail "Arch product payload differs from the verified Debian payload"
fi

FINAL="$OUT_DIR/$(basename "$BUILT")"
cp -f "$BUILT" "$FINAL" || fail "could not copy the Arch package to $OUT_DIR"
FINAL_SHA=$(sha256_file "$FINAL")
echo "Arch payload matches Debian byte-for-byte"
echo "artifact: $FINAL"
echo "sha256: $FINAL_SHA"
echo "ARCH_PACKAGE_OK $FINAL"
