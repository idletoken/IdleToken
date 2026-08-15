@echo off
REM Fetch, patch and build the pinned llama.cpp engine on Windows (MSVC + CUDA).
REM Mirrors scripts/build_llamacpp.sh (which covers macOS/Linux and refuses to
REM run on Windows). Same pin, same patch dir, same targets, same stamp format.
REM
REM Products (static CRT + static cudart -- no DLL soup next to them):
REM   vendor\llama.cpp\build\bin\Release\llama-server.exe      inference + OpenAI API
REM   vendor\llama.cpp\build\bin\Release\ggml-rpc-server.exe   worker-side RPC backend
REM   vendor\llama.cpp\build\bin\Release\llama-perplexity.exe  numeric gate
REM
REM Runtime DLL note for packaging: cudart is linked statically (GGML_STATIC),
REM but the CUDA Toolkit for Windows has no static cuBLAS, so the exes still
REM import cublas64_12.dll + cublasLt64_12.dll (Toolkit DLLs, NOT shipped with
REM the driver). Verify with dumpbin /dependents after any flag change.
REM
REM No silent fallback (v2 hard invariant): if CUDA or MSVC is missing this
REM script exits red. It never downgrades to a CPU build to "keep things green".
REM
REM Toolchain discovery (why it looks the way it does):
REM   - git: build machines here do not have Git for Windows installed and
REM     github.com:443 is not reachable from every LAN. We look for git on PATH,
REM     then Program Files, then a portable MinGit at <repo>\tools\mingit.
REM     If the pinned SHA is already present in the checkout (seeded via scp'd
REM     git objects or a bundle), no network is touched at all.
REM   - cmake: a MinGW cmake sits on PATH on the build box; letting it pick a
REM     default generator selects MinGW Makefiles + gcc. We prefer the cmake
REM     bundled with VS Build Tools and ALWAYS force the Visual Studio
REM     generator, so even a MinGW cmake binary produces an MSVC build.
REM   - CUDA: the Visual Studio generator needs the toolkit's MSBuild
REM     integration (BuildCustomizations\CUDA <ver>.props). Default 12.8: its
REM     runtime works on any r525+ driver via minor-version compatibility
REM     (same reasoning as scripts/build_ds4cuda.ps1), and it is the only
REM     version with VS integration installed on the build box.
REM
REM Usage:  scripts\build_llamacpp_win.bat              fetch + patch + build + verify
REM         scripts\build_llamacpp_win.bat --fetch-only clean checkout at the pin
REM Env:    IDLETOKEN_LLAMACPP_SRC      checkout dir  (default <repo>\vendor\llama.cpp)
REM         IDLETOKEN_CUDA_VER          toolkit ver   (default 12.8)
REM         IDLETOKEN_CUDA_ARCHS        CUDA archs    (default 75-real;120 = RTX 2070
REM                                     SASS + RTX 5060 Ti SASS + compute_120 PTX;
REM                                     covers the whole Windows testbed fleet)
REM         IDLETOKEN_LLAMACPP_GIT_URL  mirror URL tried when the UPSTREAM url fails
REM         IDLETOKEN_MBEDTLS_SRC       local mbedTLS v3.6.7 source tree for the TLS
REM                                     transport patch (default <repo>\tools\mbedtls
REM                                     when present). Without it, cmake FetchContent
REM                                     clones github.com/Mbed-TLS/mbedtls -- which
REM                                     this LAN cannot reach, so seed the tree.
setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
set "PATCH_DIR=%ROOT%\scripts\llamacpp-patches"
if defined IDLETOKEN_LLAMACPP_SRC (set "SRC_DIR=%IDLETOKEN_LLAMACPP_SRC%") else (set "SRC_DIR=%ROOT%\vendor\llama.cpp")
set "BUILD_DIR=%SRC_DIR%\build"
if not defined IDLETOKEN_CUDA_VER set "IDLETOKEN_CUDA_VER=12.8"
if not defined IDLETOKEN_CUDA_ARCHS set "IDLETOKEN_CUDA_ARCHS=75-real;120"

REM --- pin ---------------------------------------------------------------------
set "REPO_URL="
set "PIN_SHA="
for /f "usebackq tokens=1,2" %%A in ("%PATCH_DIR%\UPSTREAM") do (
    if not defined PIN_SHA (
        set "REPO_URL=%%A"
        set "PIN_SHA=%%B"
    )
)
if not defined PIN_SHA (
    echo FATAL: cannot parse %PATCH_DIR%\UPSTREAM
    exit /b 1
)
set "PIN7=%PIN_SHA:~0,7%"

REM --- git discovery -----------------------------------------------------------
set "GIT="
set "GIT_BINDIR="
where git >nul 2>&1 && set "GIT=git"
if not defined GIT if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "GIT=%ProgramFiles%\Git\cmd\git.exe"
    set "GIT_BINDIR=%ProgramFiles%\Git\cmd"
)
if not defined GIT if exist "%ROOT%\tools\mingit\cmd\git.exe" (
    set "GIT=%ROOT%\tools\mingit\cmd\git.exe"
    set "GIT_BINDIR=%ROOT%\tools\mingit\cmd"
)
REM cmake runs its own find_package(Git) for build info; a git that only this
REM script knows about must also go on PATH, or llama-server reports version
REM "unknown" and the pin check below can never pass.
if defined GIT_BINDIR set "PATH=%GIT_BINDIR%;%PATH%"
if not defined GIT (
    echo FATAL: git not found. Install Git for Windows, or unpack portable MinGit
    echo        into %ROOT%\tools\mingit ^(so that tools\mingit\cmd\git.exe exists^).
    exit /b 1
)

REM --- fetch -------------------------------------------------------------------
if not exist "%SRC_DIR%\.git" (
    mkdir "%SRC_DIR%" 2>nul
    "%GIT%" -C "%SRC_DIR%" init -q
    if errorlevel 1 exit /b 1
    "%GIT%" -C "%SRC_DIR%" remote add origin %REPO_URL%
)
REM Deterministic LF checkout: the *.patch files are generated on POSIX; a
REM CRLF-normalized working tree makes `git apply` reject every hunk.
"%GIT%" -C "%SRC_DIR%" config core.autocrlf false
"%GIT%" -C "%SRC_DIR%" config core.longpaths true
REM filemode: a checkout seeded from a POSIX machine keeps core.filemode=true,
REM and Windows cannot represent the executable bit -- every script then shows
REM as "modified" and the build stamps itself <sha>-dirty for no real change.
"%GIT%" -C "%SRC_DIR%" config core.filemode false

"%GIT%" -C "%SRC_DIR%" cat-file -e %PIN_SHA% 2>nul
if not errorlevel 1 goto :have_pin
echo == fetching %PIN_SHA% from %REPO_URL%
"%GIT%" -C "%SRC_DIR%" fetch --depth 1 origin %PIN_SHA%
if not errorlevel 1 goto :have_pin
if not defined IDLETOKEN_LLAMACPP_GIT_URL (
    echo FATAL: cannot fetch %PIN_SHA% from %REPO_URL%.
    echo        If this LAN cannot reach github.com, either set
    echo        IDLETOKEN_LLAMACPP_GIT_URL to a reachable mirror of the repo
    echo        ^(the pinned SHA is the integrity check, not the mirror^), or
    echo        seed the git objects into %SRC_DIR%\.git from another machine.
    exit /b 1
)
echo == upstream unreachable, trying mirror %IDLETOKEN_LLAMACPP_GIT_URL%
"%GIT%" -C "%SRC_DIR%" fetch --depth 1 %IDLETOKEN_LLAMACPP_GIT_URL% %PIN_SHA%
if errorlevel 1 (
    echo FATAL: mirror fetch failed too.
    exit /b 1
)
:have_pin
REM Reset tracked files to the pinned commit; patches are reapplied below.
REM (Local edits in the checkout are lost here -- capture them as a patch
REM first. build\ is untracked and survives.)
"%GIT%" -C "%SRC_DIR%" checkout -qf %PIN_SHA%
if errorlevel 1 (
    echo FATAL: git checkout %PIN_SHA% failed
    exit /b 1
)

REM --- patch -------------------------------------------------------------------
REM Lexical order via `dir /on`; there may be zero patches today (the TLS
REM transport patch lands with WS-A A2) and that must not be an error.
set "PATCH_COUNT=0"
for /f "delims=" %%P in ('dir /b /a-d /on "%PATCH_DIR%\*.patch" 2^>nul') do (
    echo == applying %%P
    "%GIT%" -C "%SRC_DIR%" apply --verbose "%PATCH_DIR%\%%P"
    if errorlevel 1 (
        echo FATAL: patch %%P does not apply
        exit /b 1
    )
    set /a PATCH_COUNT+=1
)

if "%~1"=="--fetch-only" (
    echo == checkout ready at %PIN_SHA% with %PATCH_COUNT% patch^(es^): %SRC_DIR%
    exit /b 0
)

REM --- toolchain checks (fail closed) ------------------------------------------
set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%IDLETOKEN_CUDA_VER%"
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo FATAL: CUDA %IDLETOKEN_CUDA_VER% not found at "%CUDA_HOME%".
    echo        This build is CUDA-only by design -- no CPU fallback. Install the
    echo        toolkit or set IDLETOKEN_CUDA_VER to an installed version.
    exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
)
if not defined VSROOT (
    echo FATAL: no Visual Studio 2022 with the C++ toolset found ^(vswhere^).
    echo        Install VS 2022 Build Tools with the "Desktop development with C++"
    echo        workload -- MinGW cannot build the CUDA backend.
    exit /b 1
)
REM The VS generator drives nvcc through the toolkit's MSBuild integration;
REM without these .props the configure error is cryptic, so check up front.
if not exist "%VSROOT%\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA %IDLETOKEN_CUDA_VER%.props" (
    echo FATAL: CUDA %IDLETOKEN_CUDA_VER% MSBuild integration is missing under
    echo        "%VSROOT%\MSBuild\Microsoft\VC\v170\BuildCustomizations".
    echo        Re-run the CUDA installer and enable "Visual Studio Integration".
    exit /b 1
)

REM Prefer the cmake bundled with VS; any cmake works because the generator is
REM forced, but the VS one is guaranteed present and MSVC-aware.
set "CMAKE=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"

REM --- configure ---------------------------------------------------------------
REM Offline mbedTLS for the TLS transport patch: FetchContent needs github.com,
REM which not every build LAN can reach. An empty cache var is falsy for the
REM patch's `if (IDLETOKEN_MBEDTLS_SRC)`, so passing it unconditionally is safe.
if not defined IDLETOKEN_MBEDTLS_SRC if exist "%ROOT%\tools\mbedtls\CMakeLists.txt" set "IDLETOKEN_MBEDTLS_SRC=%ROOT%\tools\mbedtls"

REM Flag-for-flag the COMMON_FLAGS of build_llamacpp.sh, plus:
REM   GGML_STATIC=ON                        static cudart (cuBLAS stays a DLL,
REM                                         see header note)
REM   CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded   /MT -- no VC redist needed on
REM                                         a machine with only the NVIDIA driver
echo == configuring ^(MSVC + CUDA %IDLETOKEN_CUDA_VER%, archs %IDLETOKEN_CUDA_ARCHS%^)
"%CMAKE%" -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -T cuda=%IDLETOKEN_CUDA_VER% ^
    -DGGML_CUDA=ON ^
    "-DCMAKE_CUDA_ARCHITECTURES=%IDLETOKEN_CUDA_ARCHS%" ^
    -DGGML_RPC=ON ^
    -DGGML_RPC_TLS=ON ^
    "-DIDLETOKEN_MBEDTLS_SRC=%IDLETOKEN_MBEDTLS_SRC%" ^
    -DGGML_STATIC=ON ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
if errorlevel 1 (
    echo FATAL: cmake configure failed
    exit /b 1
)

echo == building ^(this takes 30-60+ min for the CUDA kernels^)
"%CMAKE%" --build "%BUILD_DIR%" --config Release -j %NUMBER_OF_PROCESSORS% ^
    --target llama-server ggml-rpc-server llama-perplexity
if errorlevel 1 (
    echo FATAL: build failed
    exit /b 1
)

REM --- verify ------------------------------------------------------------------
REM Actually execute the product; existence alone proves nothing (a stale or
REM half-linked exe sits there looking green). Multi-config generator puts the
REM binaries under bin\Release, not bin\ -- consumers take note.
set "BIN_DIR=%BUILD_DIR%\bin\Release"
for %%B in (llama-server.exe ggml-rpc-server.exe llama-perplexity.exe) do (
    if not exist "%BIN_DIR%\%%B" (
        echo FATAL: %%B not built
        exit /b 1
    )
)
REM Two-step on purpose: a `for /f ('...')` whose command starts with a quoted
REM path trips cmd's quote stripping ("The filename, directory name, or volume
REM label syntax is incorrect"). Redirect to a file, then filter the file.
set "VERSION_LINE="
"%BIN_DIR%\llama-server.exe" --version > "%BUILD_DIR%\version_probe.txt" 2>&1
for /f "usebackq delims=" %%L in (`findstr /c:"version:" "%BUILD_DIR%\version_probe.txt"`) do set "VERSION_LINE=%%L"
if not defined VERSION_LINE (
    echo FATAL: llama-server.exe does not run ^(no version output^)
    exit /b 1
)
echo %VERSION_LINE% | findstr /c:"%PIN7%" >nul
if errorlevel 1 (
    echo FATAL: built version "%VERSION_LINE%" does not match pin %PIN_SHA%
    exit /b 1
)

REM --- stamp (same format as build_llamacpp.sh) --------------------------------
REM Written line by line, NOT as a `> file ( ... )` block: inside such a block
REM cmd stops the line at the first unescaped ")" -- which silently truncated
REM "version: 1 (0a50d99)" to "version: 1 (0a50d99".
echo upstream %REPO_URL% %PIN_SHA%> "%BUILD_DIR%\IDLETOKEN_ENGINE_STAMP"
echo version %VERSION_LINE%>> "%BUILD_DIR%\IDLETOKEN_ENGINE_STAMP"
for /f "delims=" %%P in ('dir /b /a-d /on "%PATCH_DIR%\*.patch" 2^>nul') do (
    for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 '%PATCH_DIR%\%%P').Hash.ToLower()"`) do (
        echo patch %%P %%H>> "%BUILD_DIR%\IDLETOKEN_ENGINE_STAMP"
    )
)

echo == OK: %VERSION_LINE%
echo == products in %BIN_DIR% ^(stamp: %BUILD_DIR%\IDLETOKEN_ENGINE_STAMP^)
exit /b 0
