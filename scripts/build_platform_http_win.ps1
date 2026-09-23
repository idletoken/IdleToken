param([string]$Root = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Parent batch scripts restrict PATH to the compiler and Windows directories.
# Resolve the standard CMake installation too, rather than relying on a build
# machine that happens to keep cmake.exe beside gcc.exe.
$cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
$cmake = if ($env:IDLETOKEN_CMAKE) { $env:IDLETOKEN_CMAKE } elseif ($cmakeCommand) { $cmakeCommand.Source } else { Join-Path $env:ProgramFiles 'CMake/bin/cmake.exe' }
if (!(Test-Path $cmake)) { throw 'CMake is required; install CMake or set IDLETOKEN_CMAKE to cmake.exe' }
# Native Windows TLS and certificate store; no OpenSSL or libcurl DLL to ship.
$version = '8.22.0'
$digest = 'd54dd598bf05927a726deb38df31c6a255ba83ff1de57c5d1464dac3ed8f44a1'
$tools = Join-Path $Root 'tools'
$prefix = Join-Path $tools 'curl-win'
$archive = Join-Path $tools "curl-$version.tar.gz"
$source = Join-Path $tools "curl-$version"
$build = Join-Path $tools 'curl-win-build'
$stamp = "$version $digest schannel-static-threaded-dns-v1"
$marker = Join-Path $prefix 'idletoken-build-stamp'
New-Item -ItemType Directory -Force $tools | Out-Null
# Cached libraries may outlive the source/build directories. Restore the
# pinned source for its required license independently of a library rebuild.
if (!(Test-Path (Join-Path $source 'COPYING')) -or !(Test-Path (Join-Path $source 'CMakeLists.txt'))) {
    if (!(Test-Path $archive)) {
        Invoke-WebRequest -UseBasicParsing "https://curl.se/download/curl-$version.tar.gz" -OutFile $archive -TimeoutSec 120
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        throw 'Pinned curl source digest mismatch'
    }
    & tar.exe -xzf $archive -C $tools
    if ($LASTEXITCODE) { throw 'curl source extraction failed' }
}
if (!(Test-Path $marker) -or (Get-Content -Raw $marker).Trim() -ne $stamp -or
    !(Test-Path (Join-Path $prefix 'lib/libcurl.a')) -or
    !(Test-Path (Join-Path $prefix 'include/curl/curl.h'))) {
    if (!(Test-Path $archive)) {
        Invoke-WebRequest -UseBasicParsing "https://curl.se/download/curl-$version.tar.gz" -OutFile $archive -TimeoutSec 120
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        throw 'Pinned curl source digest mismatch'
    }
    & $cmake -S $source -B $build -G 'MinGW Makefiles' "-DCMAKE_INSTALL_PREFIX=$prefix" `
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON `
        -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF -DCURL_USE_SCHANNEL=ON -DCURL_USE_OPENSSL=OFF `
        -DENABLE_THREADED_RESOLVER=ON -DENABLE_IPV6=ON -DCURL_USE_LIBPSL=OFF `
        -DCURL_ZLIB=OFF -DCURL_BROTLI=OFF -DCURL_ZSTD=OFF -DUSE_NGHTTP2=OFF `
        -DCURL_USE_LIBSSH2=OFF -DCURL_USE_LIBSSH=OFF -DCURL_USE_GSSAPI=OFF `
        -DCURL_USE_LIBIDN2=OFF -DCURL_USE_WIN32_IDN=ON -DCURL_ENABLE_NTLM=ON `
        -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON
    if ($LASTEXITCODE) { throw 'curl configure failed' }
    & $cmake --build $build --parallel 4
    if ($LASTEXITCODE) { throw 'curl build failed' }
    & $cmake --install $build
    if ($LASTEXITCODE) { throw 'curl install failed' }
    if (!(Test-Path (Join-Path $prefix 'lib/libcurl.a'))) { throw 'Static libcurl missing' }
    Set-Content -Encoding ASCII $marker $stamp
}
$licenses = Join-Path $Root 'client/src-tauri/licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
Copy-Item (Join-Path $source 'COPYING') (Join-Path $licenses 'curl.txt') -Force
Write-Output 'PLATFORM_HTTP_WIN_OK'
