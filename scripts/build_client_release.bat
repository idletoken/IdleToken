@echo off
REM Build the installable Windows client (NSIS .exe setup + MSI when WiX is
REM present). Run on the Windows build node (win-a) from the repo root:
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
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set ROOT=%CD%
set TRIPLE=x86_64-pc-windows-msvc

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

REM --- engine binaries ---------------------------------------------------
if not exist "%ROOT%\idletoken-worker.exe" (
    REM build_ds4x_win.bat lives at the REPO ROOT and is the only copy —
    REM scripts\build_ds4x.bat was a drifted second copy, deleted 2026-08-03.
    echo CLIENT_RELEASE_FAIL: no idletoken-worker.exe ^(run build_ds4x_win.bat first^)
    exit /b 1
)
if not exist "%ROOT%\idletoken-coord.exe" (
    echo CLIENT_RELEASE_FAIL: no idletoken-coord.exe ^(run scripts\build_coord_win.bat first^)
    exit /b 1
)
if not exist "%ROOT%\client\src-tauri\binaries" mkdir "%ROOT%\client\src-tauri\binaries"
copy /y "%ROOT%\idletoken-worker.exe" "%ROOT%\client\src-tauri\binaries\idletoken-worker-%TRIPLE%.exe" >nul
copy /y "%ROOT%\idletoken-coord.exe"  "%ROOT%\client\src-tauri\binaries\idletoken-coord-%TRIPLE%.exe" >nul
if exist "%ROOT%\idletoken-platform-agent.exe" (
    copy /y "%ROOT%\idletoken-platform-agent.exe" "%ROOT%\client\src-tauri\binaries\idletoken-platform-agent-%TRIPLE%.exe" >nul
) else (
    echo WARN: no idletoken-platform-agent.exe — the Platform panel will fail to start the agent
    REM Tauri fails the build on a missing externalBin, so keep a truthful stub
    REM out of it: copy the coord and let the agent's own --selftest report.
    copy /y "%ROOT%\idletoken-coord.exe" "%ROOT%\client\src-tauri\binaries\idletoken-platform-agent-%TRIPLE%.exe" >nul
)

REM --- CUDA runtime DLLs -------------------------------------------------
if not exist "%ROOT%\client\src-tauri\runtime\windows" mkdir "%ROOT%\client\src-tauri\runtime\windows"
REM Repo root FIRST, dist\ only as a fallback: the build scripts write their
REM output to the root, while dist\ is a hand-assembled copy that goes stale.
REM Shipping a stale ds4cuda.dll cost a full cross-machine debug round — the
REM 07-16 copy in dist\ predated the WDDM DMA fix, so the packaged worker died
REM in cudaMemcpy and fell back to MOCK, i.e. the cluster came up "ready" and
REM answered with garbage. The timestamps are echoed for exactly that reason.
REM 2026-08-04: cudart/cublas/cublasLt are NO LONGER SHIPPED. cuBLAS alone was
REM 769 MB of a 790 MB payload and is Toolkit-only, not driver (philosophy 12).
REM `objdump -p` confirms both our DLLs import only KERNEL32 at load time, and
REM cuBLAS is delay-loaded with an actionable message when absent.
for %%D in (ds4cuda.dll ds4xcuda.dll) do (
    if exist "%ROOT%\%%D" (
        copy /y "%ROOT%\%%D" "%ROOT%\client\src-tauri\runtime\windows\%%D" >nul
        for %%T in ("%ROOT%\%%D") do echo   DLL %%D  ^<- repo root  %%~tT
    ) else if exist "%ROOT%\dist\%%D" (
        copy /y "%ROOT%\dist\%%D" "%ROOT%\client\src-tauri\runtime\windows\%%D" >nul
        for %%T in ("%ROOT%\dist\%%D") do echo   DLL %%D  ^<- dist\  %%~tT
    ) else (
        echo CLIENT_RELEASE_FAIL: missing runtime DLL %%D ^(looked in repo root and dist\^)
        exit /b 1
    )
)

REM tauri.windows.conf.json lists ONLY ds4cuda.dll + ds4xcuda.dll as resources.
REM cudart/cublas/cublasLt were removed on 2026-08-04: cuBLAS alone was 769 MB of
REM a 790 MB payload (97%), and it comes from the CUDA Toolkit, not the display
REM driver, so the user installs it (acceptance-criteria philosophy 12, E3).
REM Keep that list EXPLICIT per file — a directory glob would silently re-ship
REM whatever an older build left in runtime\windows\.
REM (The note lives here, not in the .json: Tauri's schema rejects unknown keys,
REM so a comment inside the config fails the build with
REM  "Additional properties are not allowed".)
REM
REM Purge CUDA runtime DLLs an older build staged here. tauri.windows.conf.json
REM no longer lists them, but leaving 800 MB of files in the staging directory
REM invites the next person to "fix" the config by adding them back — and it is
REM how dist\ silently stayed 447 MB for weeks.
for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do (
    if exist "%ROOT%\client\src-tauri\runtime\windows\%%D" (
        del /q "%ROOT%\client\src-tauri\runtime\windows\%%D"
        echo   removed stale staged %%D ^(user supplies it via CUDA Toolkit^)
    )
)

REM --- frontend ----------------------------------------------------------
if not exist "%ROOT%\client\dist\index.html" (
    echo CLIENT_RELEASE_FAIL: no client\dist ^(build the frontend on a node machine and copy it here^)
    exit /b 1
)

REM --- bundle ------------------------------------------------------------
REM NSIS only: the MSI target needs WiX, and both toolchains are fetched from
REM GitHub release assets on first use. Tauri's downloader dies with
REM `timeout: global` on this network even though the same URL downloads in 4s
REM from PowerShell — so route it through a mirror unless the caller set one.
REM Override with your own mirror (or point it at https://github.com to try
REM direct) by setting TAURI_BUNDLER_TOOLS_GITHUB_MIRROR before running.
if not defined TAURI_BUNDLER_TOOLS_GITHUB_MIRROR set "TAURI_BUNDLER_TOOLS_GITHUB_MIRROR=https://ghfast.top/https://github.com"
cd /d "%ROOT%\client"
cargo tauri build --bundles nsis --config "{\"build\":{\"beforeBuildCommand\":\"\"}}"
if errorlevel 1 (
    echo CLIENT_RELEASE_FAIL: tauri build failed
    exit /b 1
)

echo --- artifacts ---
dir /b /s "%ROOT%\client\src-tauri\target\release\bundle\*.exe" 2>nul

REM --- checksums ---------------------------------------------------------
REM We ship UNSIGNED (product decision 2026-08-04: no code-signing certificate
REM yet), so Windows greets every user with a SmartScreen block. The only thing
REM a user can actually verify in that situation is the file hash — so it has to
REM exist, and it has to be produced by the build, not by hand at upload time.
REM Signing proves WHO shipped it; a checksum proves the file was not swapped.
REM Publish SHA256SUMS.txt next to the installer on the release page, and see
REM docs/user-guide.md §2.1 for the Get-FileHash line users are told to run.
set "BUNDLE=%ROOT%\client\src-tauri\target\release\bundle\nsis"
powershell -NoProfile -Command ^
  "Get-ChildItem '%BUNDLE%' -Filter *.exe | ForEach-Object { ($_ | Get-FileHash -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name } | Set-Content -Encoding ascii '%BUNDLE%\SHA256SUMS.txt'"
if exist "%BUNDLE%\SHA256SUMS.txt" (
    echo --- sha256 ---
    type "%BUNDLE%\SHA256SUMS.txt"
) else (
    echo CLIENT_RELEASE_FAIL: could not write SHA256SUMS.txt
    exit /b 1
)
echo CLIENT_RELEASE_OK
