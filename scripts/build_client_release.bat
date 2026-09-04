@echo off
REM Build the installable Windows client (NSIS .exe setup). Run on the Windows
REM build node from the repo root:
REM     scripts\build_client_release.bat
REM
REM Two things this does that a plain `tauri build` does not:
REM   1) stages the engine binaries under the sidecar names Tauri expects, and
REM   2) stages the CUDA runtime DLLs into runtime\windows\ so the installer
REM      drops them next to the sidecar exe. Without them the worker dies with
REM      0xC0000135 (DLL_NOT_FOUND) on any machine that has only the driver —
REM      which is exactly the machine this product targets.
REM
REM The build node has no node/pnpm, so the frontend must already be built into
REM client\dist (sync it from a machine that has node). beforeBuildCommand is
REM therefore overridden to nothing.
REM
REM Contract: prints CLIENT_RELEASE_OK or CLIENT_RELEASE_FAIL: <reason>.
REM
REM There is no updater artifact and no signing since 2026-09-02 (user ruling):
REM the product has no in-app updater, so a release is the installer alone.
REM IDLETOKEN_DEFER_UPDATER_SIGNING and the deferred-signing dance it named are
REM gone with it.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set ROOT=%CD%
set TRIPLE=x86_64-pc-windows-msvc
set ALT_TRIPLE=x86_64-pc-windows-gnu
set "RUST_TOOLCHAIN="

REM A normal MSVC Rust host still cannot link when this shell has no Visual
REM Studio/Windows SDK `link.exe` (the Windows test-bed build node intentionally
REM has only WinLibs). In that case use the installed Rust GNU host: it shares
REM the same MinGW runtime as the native coord/worker build and needs no SDK.
REM The sidecar suffix MUST follow the selected Rust target or Tauri refuses the
REM build before compiling the client.
where link.exe >nul 2>&1
if errorlevel 1 (
    rustup run stable-x86_64-pc-windows-gnu rustc -vV >nul 2>&1
    if errorlevel 1 (
        echo CLIENT_RELEASE_FAIL: neither MSVC link.exe nor stable-x86_64-pc-windows-gnu is available
        exit /b 1
    )
    set TRIPLE=x86_64-pc-windows-gnu
    set ALT_TRIPLE=x86_64-pc-windows-msvc
    set "RUST_TOOLCHAIN=+stable-x86_64-pc-windows-gnu"
    echo   using Rust GNU toolchain ^(Visual Studio linker is unavailable^)
)

REM --- build cache from a different repo path -----------------------------
REM Cargo and Tauri bake ABSOLUTE paths into target\ (the generated plugin
REM permission .toml files among them). Move or rename the repo — as the
REM 2026-08-04 IdleToken rename did (...\HomeAI -> ...\IdleToken)
REM — and the next release build dies with
REM   failed to read plugin permissions: ...\HomeAI\...\app_hide.toml
REM   The system cannot find the path specified
REM naming a directory that no longer exists. Nothing in that message mentions
REM the rename, so it costs a debugging round every time; worse, the ladder
REM cannot see it at all (E1 checks that an exe EXISTS, and the P gates drive a
REM debug client on the coord node — neither builds the package). Stamp the
REM root we built under and wipe the cache when it changes.
set "STAMP=%ROOT%\client\src-tauri\target\.idletoken-build-root"
set "OLDROOT="
if exist "%STAMP%" set /p OLDROOT=<"%STAMP%"
if defined OLDROOT if /i not "!OLDROOT!"=="%ROOT%" (
    echo   build cache was produced under "!OLDROOT!" — wiping target build dirs
    if exist "%ROOT%\client\src-tauri\target\release\build" rmdir /s /q "%ROOT%\client\src-tauri\target\release\build"
    if exist "%ROOT%\client\src-tauri\target\debug\build" rmdir /s /q "%ROOT%\client\src-tauri\target\debug\build"
)
if not exist "%ROOT%\client\src-tauri\target" mkdir "%ROOT%\client\src-tauri\target"
> "%STAMP%" echo %ROOT%

REM --- native sidecars ---------------------------------------------------
REM Release packaging must compile these from the sources that were just
REM synced. Merely checking that an exe exists is not provenance: on 2026-09-02
REM an old worker survived several source syncs and made a freshly built client
REM report 13.1 GiB available while NVML reported 15.7 GiB free. The source had
REM already removed that discount; the stale binary silently put it back into
REM the installer. Fail closed if any current-source build fails.
call "%ROOT%\build_worker_win.bat"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: current-source worker build failed
    exit /b 1
)
call "%ROOT%\scripts\build_coord_win.bat"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: current-source coordinator build failed
    exit /b 1
)
call "%ROOT%\scripts\build_agent_win.bat"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: current-source platform agent build failed
    exit /b 1
)

REM --- engine binaries ---------------------------------------------------
if not exist "%ROOT%\idletoken-worker.exe" (
    echo CLIENT_RELEASE_FAIL: no idletoken-worker.exe after current-source build
    exit /b 1
)
if not exist "%ROOT%\idletoken-coord.exe" (
    echo CLIENT_RELEASE_FAIL: no idletoken-coord.exe after current-source build
    exit /b 1
)
REM The coord must carry the pinned platform verify key, or sharing can never
REM be switched on by anyone who installs this package (overflow.c RULE 3 —
REM every release up to 0.1.5 shipped unpinned, found 2026-08-21). String-
REM grepping the exe is a documented dead end (Rust/C literals get merged and
REM reordered), so ask the binary itself: an offline probe whose only network
REM target refuses instantly. A PINNED coord prints "overflow: on" and exits on
REM the sentinel model id; an UNPINNED one refuses before any network I/O. The
REM check demands the positive line, so a probe that stops producing overflow
REM output at all fails the build instead of silently passing it.
"%ROOT%\idletoken-coord.exe" --overflow-url http://127.0.0.1:1 --overflow-key pin-probe --model pin-probe-sentinel > "%TEMP%\idletoken_pin_probe.txt" 2>&1
findstr /c:"overflow: on" "%TEMP%\idletoken_pin_probe.txt" >nul
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: idletoken-coord.exe has no pinned platform verify key ^(rebuild with scripts\build_coord_win.bat — it pins from scripts\platform-verify-key.b64^)
    type "%TEMP%\idletoken_pin_probe.txt"
    exit /b 1
)
if not exist "%ROOT%\client\src-tauri\binaries" mkdir "%ROOT%\client\src-tauri\binaries"
copy /y "%ROOT%\idletoken-worker.exe" "%ROOT%\client\src-tauri\binaries\idletoken-worker-%TRIPLE%.exe" >nul
copy /y "%ROOT%\idletoken-coord.exe"  "%ROOT%\client\src-tauri\binaries\idletoken-coord-%TRIPLE%.exe" >nul
REM cargo-tauri itself may have been installed for the other Windows ABI and
REM chooses externalBin names from its own host triple. The sidecars are
REM standalone processes, not linked libraries, so keep BOTH suffixes current
REM instead of letting an old alternate-suffix copy enter the installer.
copy /y "%ROOT%\idletoken-worker.exe" "%ROOT%\client\src-tauri\binaries\idletoken-worker-%ALT_TRIPLE%.exe" >nul
copy /y "%ROOT%\idletoken-coord.exe"  "%ROOT%\client\src-tauri\binaries\idletoken-coord-%ALT_TRIPLE%.exe" >nul
if not exist "%ROOT%\idletoken-platform-agent.exe" (
    echo CLIENT_RELEASE_FAIL: no idletoken-platform-agent.exe after current-source build
    exit /b 1
)
copy /y "%ROOT%\idletoken-platform-agent.exe" "%ROOT%\client\src-tauri\binaries\idletoken-platform-agent-%TRIPLE%.exe" >nul
copy /y "%ROOT%\idletoken-platform-agent.exe" "%ROOT%\client\src-tauri\binaries\idletoken-platform-agent-%ALT_TRIPLE%.exe" >nul

REM llama.cpp sidecars are the v2 compute engine: idletoken-server serves local
REM models and drives clusters; idletoken-rpc-server is supervised on worker nodes.
REM Both must come from the same pinned checkout/build or an installed client
REM can offer cluster mode while carrying only half of the engine.
REM The BUILD produces upstream's names (llama-server.exe, ggml-rpc-server.exe —
REM see scripts\build_llamacpp_win.bat, which cmake-builds those two targets);
REM only the STAGED copy carries our name, exactly as stage_sidecars.sh and
REM build_client_win.sh do it. This loop used to look for idletoken-server.exe
REM *in the llama.cpp build directory*, where nothing ever writes that name, so
REM it failed every time with "run scripts\build_llamacpp_win.bat first" —
REM advice that cannot fix it, because that script had already run.
set "LLAMA_BIN=%ROOT%\vendor\llama.cpp\build\bin\Release"
if not exist "%LLAMA_BIN%\llama-server.exe" set "LLAMA_BIN=%ROOT%\vendor\llama.cpp\build\bin"
if not exist "%ROOT%\client\src-tauri\runtime\windows" mkdir "%ROOT%\client\src-tauri\runtime\windows"
call :stage_engine llama-server    idletoken-server     || exit /b 1
call :stage_engine ggml-rpc-server idletoken-rpc-server || exit /b 1
goto :engines_staged

:stage_engine
if not exist "%LLAMA_BIN%\%1.exe" (
    echo CLIENT_RELEASE_FAIL: no pinned %1.exe under "%LLAMA_BIN%" ^(run scripts\build_llamacpp_win.bat first^)
    exit /b 1
)
copy /y "%LLAMA_BIN%\%1.exe" "%ROOT%\client\src-tauri\binaries\%2-%TRIPLE%.exe" >nul || (
    echo CLIENT_RELEASE_FAIL: could not stage %1.exe as %2 & exit /b 1)
copy /y "%LLAMA_BIN%\%1.exe" "%ROOT%\client\src-tauri\binaries\%2-%ALT_TRIPLE%.exe" >nul || (
    echo CLIENT_RELEASE_FAIL: could not stage %1.exe as %2 for alternate target & exit /b 1)
REM externalBin carries only the executable. Shared-mode integrity checks the
REM installed engine against a digest beside it, so bundle that digest as an
REM explicit root resource under the FINAL installed sidecar name.
powershell -NoProfile -Command ^
  "$h=(Get-FileHash '%ROOT%\client\src-tauri\binaries\%2-%TRIPLE%.exe' -Algorithm SHA256).Hash.ToLower(); Set-Content -Encoding ascii '%ROOT%\client\src-tauri\runtime\windows\%2.exe.sha256' ($h + '  %2.exe')"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: could not record the %2.exe engine digest
    exit /b 1
)
exit /b 0

:engines_staged

REM --- licences ----------------------------------------------------------
REM Apache-2.0 section 4(d) requires our NOTICE to travel with a BINARY
REM distribution. The retained ds4 source is not compiled into this package,
REM but its MIT licence deliberately remains in every distribution under the
REM project's archival/public-mirror contract.
REM An installer IS a binary distribution. Staged into src-tauri\licenses so
REM tauri.conf.json `bundle.resources` puts them inside the installer.
if not exist "%ROOT%\client\src-tauri\licenses" mkdir "%ROOT%\client\src-tauri\licenses"
copy /y "%ROOT%\LICENSE" "%ROOT%\client\src-tauri\licenses\LICENSE.txt" >nul || (
    echo CLIENT_RELEASE_FAIL: cannot stage LICENSE & exit /b 1)
copy /y "%ROOT%\NOTICE" "%ROOT%\client\src-tauri\licenses\NOTICE.txt" >nul || (
    echo CLIENT_RELEASE_FAIL: cannot stage NOTICE & exit /b 1)
copy /y "%ROOT%\vendor\ds4\LICENSE" "%ROOT%\client\src-tauri\licenses\ds4-MIT.txt" >nul || (
    echo CLIENT_RELEASE_FAIL: cannot stage the ds4 licence & exit /b 1)
copy /y "%ROOT%\vendor\llama.cpp\LICENSE" "%ROOT%\client\src-tauri\licenses\llamacpp-MIT.txt" >nul || (
    echo CLIENT_RELEASE_FAIL: cannot stage the llama.cpp licence & exit /b 1)

REM --- CUDA runtime -----------------------------------------------------
REM ds4/ds4x are not a backend: delete their stale DLLs. The llama.cpp CUDA
REM sidecars are different. DSv4 on the pinned Windows build was measured to
REM fail at process load with 0xC0000135 when cudart/cuBLAS were absent, so the
REM installer must carry the exact CUDA 12 runtime beside the sidecars. This is
REM what lets a compute node need only the NVIDIA driver, not the Toolkit.
if not exist "%ROOT%\client\src-tauri\runtime\windows" mkdir "%ROOT%\client\src-tauri\runtime\windows"
for %%D in (ds4cuda.dll ds4xcuda.dll) do (
    if exist "%ROOT%\client\src-tauri\runtime\windows\%%D" del /q "%ROOT%\client\src-tauri\runtime\windows\%%D"
)

REM The GNU WebView2 binding links its loader dynamically, but Tauri does not
REM add that DLL to the bundle by itself. Stage the x64 loader shipped by the
REM exact webview2-com-sys version locked in Cargo.lock BEFORE running cargo:
REM bundle resource paths are validated before compilation starts, so copying
REM target\release\WebView2Loader.dll after a prebuild creates a clean-build
REM deadlock (the prebuild itself refuses the missing resource).
set "WEBVIEW2_LOADER="
for /d %%D in ("%USERPROFILE%\.cargo\registry\src\*") do (
    if exist "%%~fD\webview2-com-sys-0.38.2\x64\WebView2Loader.dll" set "WEBVIEW2_LOADER=%%~fD\webview2-com-sys-0.38.2\x64\WebView2Loader.dll"
)
if not defined WEBVIEW2_LOADER (
    echo CLIENT_RELEASE_FAIL: missing webview2-com-sys 0.38.2 x64 WebView2Loader.dll in the Cargo registry ^(must match Cargo.lock^)
    exit /b 1
)
copy /y "%WEBVIEW2_LOADER%" "%ROOT%\client\src-tauri\runtime\windows\WebView2Loader.dll" >nul || exit /b 1

set "CUDA_RUNTIME_DIR=%IDLETOKEN_CUDA_RUNTIME_DIR%"
if not defined CUDA_RUNTIME_DIR if defined CUDA_PATH set "CUDA_RUNTIME_DIR=%CUDA_PATH%\bin"
REM Last resort: borrow the DLLs from an installed IdleToken. Program Files is
REM where the installer has put them since 2026-09-02; %LOCALAPPDATA% is where
REM a pre-migration install still has them.
if not defined CUDA_RUNTIME_DIR if exist "%ProgramFiles%\IdleToken\cudart64_12.dll" set "CUDA_RUNTIME_DIR=%ProgramFiles%\IdleToken"
if not defined CUDA_RUNTIME_DIR if exist "%LOCALAPPDATA%\IdleToken\cudart64_12.dll" set "CUDA_RUNTIME_DIR=%LOCALAPPDATA%\IdleToken"
for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do (
    if not exist "%CUDA_RUNTIME_DIR%\%%D" (
        echo CLIENT_RELEASE_FAIL: missing %%D ^(set IDLETOKEN_CUDA_RUNTIME_DIR to the pinned CUDA 12 runtime directory^)
        exit /b 1
    )
    copy /y "%CUDA_RUNTIME_DIR%\%%D" "%ROOT%\client\src-tauri\runtime\windows\%%D" >nul || exit /b 1
    for %%T in ("%CUDA_RUNTIME_DIR%\%%D") do echo   CUDA %%D  %%~zT bytes
)
if not exist "%SystemRoot%\System32\vcomp140.dll" (
    echo CLIENT_RELEASE_FAIL: missing %SystemRoot%\System32\vcomp140.dll ^(required by the pinned llama.cpp engine^)
    exit /b 1
)
copy /y "%SystemRoot%\System32\vcomp140.dll" "%ROOT%\client\src-tauri\runtime\windows\vcomp140.dll" >nul || exit /b 1

REM Chinese Windows profile/model/cache/key/socket paths are a release gate,
REM not an opt-in smoke. It inspects every native manifest and performs real
REM file, AF_UNIX, cache and persistent-key operations below a Chinese path.
REM Run it only after staging the exact CUDA DLLs the installer carries: the
REM source-tree llama-server otherwise exits at process load on machines where
REM the Toolkit is not globally visible, while an old runtime directory could
REM make a wrongly ordered gate pass by accident.
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\windows_unicode_path_gate.ps1" -RepoRoot "%ROOT%"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: Windows Unicode path gate failed
    exit /b 1
)

REM --- frontend ----------------------------------------------------------
if not exist "%ROOT%\client\dist\index.html" (
    echo CLIENT_RELEASE_FAIL: no client\dist ^(build the frontend on a node machine and copy it here^)
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\client\scripts\verify_frontend_provenance.ps1" -ClientRoot "%ROOT%\client"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: client\dist is stale or does not match the synced frontend sources
    exit /b 1
)

REM --- bundle ------------------------------------------------------------
REM NSIS only: the MSI target needs WiX, and both toolchains are fetched from
REM GitHub release assets on first use.
REM
REM The default is the official source, github.com, and it must stay that way.
REM What comes down here is the NSIS toolchain that builds the installer the
REM user double-clicks, and Tauri's downloader verifies no hash on it: whoever
REM serves those bytes can put anything into the installer. A third-party
REM GitHub proxy is therefore a build-chain trust root, and it is not one this
REM project can vouch for.
REM
REM On a network where the direct download fails (Tauri's downloader has been
REM seen dying with `timeout: global` on a URL that PowerShell fetches in 4s),
REM set TAURI_BUNDLER_TOOLS_GITHUB_MIRROR yourself before running this script.
REM Do that only for a mirror you trust, and never for a build that will be
REM published: an installer whose toolchain came from an unverifiable mirror
REM should not be signed and shipped.
if not defined TAURI_BUNDLER_TOOLS_GITHUB_MIRROR set "TAURI_BUNDLER_TOOLS_GITHUB_MIRROR=https://github.com"
set "BUNDLE=%ROOT%\client\src-tauri\target\release\bundle\nsis"
cd /d "%ROOT%\client"
cargo %RUST_TOOLCHAIN% tauri build --bundles nsis --config "{\"build\":{\"beforeBuildCommand\":\"\"}}"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: tauri build failed
    exit /b 1
)

echo --- artifacts ---
dir /b /s "%ROOT%\client\src-tauri\target\release\bundle\*.exe" 2>nul

REM GitHub computes and displays the asset digest. Do not emit a detached
REM checksum, signature, updater archive or feed: the release contract is the
REM native installer alone.
echo CLIENT_RELEASE_OK
