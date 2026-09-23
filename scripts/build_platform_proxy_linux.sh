#!/usr/bin/env bash
# A private, reproducible resolver avoids depending on libproxy 0.5 in the
# distribution. Ubuntu 22.04 supplies 0.4; its PAC failures can become DIRECT.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ "$(uname -s)" = Linux ] || { echo "Linux build required" >&2; exit 1; }
VERSION=0.5.4
SHA=a6e2220349b2025de9b6d9d7f8bb347bf0c728f02a921761ad5f9f66c7436de9
CACHE="$ROOT/tools/libproxy-linux"
SOURCE="$CACHE/source"
BUILD="$CACHE/build"
STAGE="$ROOT/client/src-tauri/runtime/linux/network"
LIC="$ROOT/client/src-tauri/licenses"
for tool in curl sha256sum tar gzip patch meson ninja pkg-config patchelf; do
    command -v "$tool" >/dev/null || { echo "Missing build tool: $tool" >&2; exit 1; }
done
pkg-config --atleast-version=2.72 glib-2.0 || { echo 'GLib >= 2.72 development headers required' >&2; exit 1; }
for dep in gio-2.0 libcurl duktape gsettings-desktop-schemas; do
    pkg-config --exists "$dep" || { echo "Missing development dependency: $dep" >&2; exit 1; }
done
mkdir -p "$CACHE" "$STAGE" "$LIC"
ARCHIVE="$CACHE/libproxy-$VERSION.tar.gz"
if [ ! -f "$ARCHIVE" ]; then
    curl --fail --location --proto '=https' --connect-timeout 15 --max-time 180 \
        "https://codeload.github.com/libproxy/libproxy/tar.gz/refs/tags/$VERSION" -o "$ARCHIVE"
fi
printf '%s  %s\n' "$SHA" "$ARCHIVE" | sha256sum --check --status
if [ ! -f "$SOURCE/meson.build" ]; then
    mkdir -p "$SOURCE"
    tar xzf "$ARCHIVE" -C "$SOURCE" --strip-components=1
fi
for fix in "$ROOT"/scripts/libproxy-patches/*.patch; do
    if patch --dry-run --forward --batch -p1 -d "$SOURCE" < "$fix" >/dev/null 2>&1; then
        patch --forward --batch -p1 -d "$SOURCE" < "$fix"
    elif ! patch --dry-run --reverse --batch -p1 -d "$SOURCE" < "$fix" >/dev/null 2>&1; then
        echo "Resolver patch does not match the pinned source: $(basename "$fix")" >&2
        exit 1
    fi
done
# The package's oldest GLib baseline is 2.72 (Ubuntu 22.04), even when the
# builder has newer headers. New APIs must fail compilation rather than ship.
case "$(uname -m)" in
    aarch64) baseline=-march=armv8-a ;;
    x86_64) baseline=-march=x86-64 ;;
    *) echo 'Unsupported Linux architecture' >&2; exit 1 ;;
esac
export CFLAGS="-O2 $baseline -DGLIB_VERSION_MIN_REQUIRED=GLIB_VERSION_2_72 -DGLIB_VERSION_MAX_ALLOWED=GLIB_VERSION_2_72 -Werror=deprecated-declarations -Werror=implicit-function-declaration"
if [ ! -f "$BUILD/build.ninja" ]; then
    meson setup "$BUILD" "$SOURCE" --buildtype=release --prefix=/usr --libdir=lib \
        -Ddocs=false -Dtests=false -Dvapi=false -Dintrospection=false \
        -Dconfig-gnome=true -Dconfig-kde=true -Dpacrunner-duktape=true
fi
meson compile -C "$BUILD"
# Keep the two upstream shared libraries replaceable and use relative lookup.
cp -Lf "$BUILD/src/libproxy/libproxy.so.1" "$STAGE/libproxy.so.1"
cp -Lf "$BUILD/src/backend/libpxbackend-1.0.so" "$STAGE/libpxbackend-1.0.so"
for lib in "$STAGE"/*.so*; do patchelf --set-rpath '$ORIGIN' "$lib"; done
cp "$SOURCE/COPYING" "$LIC/libproxy-LGPL-2.1.txt"
# Ship the patched source used by this binary, with a deterministic archive.
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf - -C "$SOURCE" . \
    | gzip -n > "$STAGE/libproxy-$VERSION-source.tar.gz"
sha256sum "$STAGE/libproxy-$VERSION-source.tar.gz" | awk '{print $1}' > "$STAGE/libproxy-source.sha256"
printf '%s\n' "$SHA" > "$STAGE/libproxy-upstream.sha256"
echo PLATFORM_PROXY_LINUX_OK
