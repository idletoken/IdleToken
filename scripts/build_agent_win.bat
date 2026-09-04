@echo off
REM Native Windows build of the marketplace agent (idletoken-platform-agent.exe).
REM
REM Mirrors Makefile.platform, which already declares Windows support (bcrypt
REM for BCryptGenRandom) but needs a POSIX shell to run. The agent is pure
REM portable C — vendored TweetNaCl + BLAKE2b, sockets, HTTP — no CUDA, no
REM threads, so this is a straight compile.
REM
REM Without it the Windows installer had to ship *something* under the agent's
REM name (Tauri fails the bundle on a missing externalBin) and the Platform
REM panel would launch the wrong binary.
REM
REM Run from the repo root:  scripts\build_agent_win.bat
REM Contract: prints AGENT_WIN_OK or AGENT_WIN_FAIL (details in agent_build.log).
setlocal
cd /d "%~dp0.."
REM MinGW location: defaults to the per-user copy winget installs; override with
REM the IDLETOKEN_MINGW_BIN environment variable. This used to hardcode one build
REM machine's absolute path (username included) -- useless on any other machine,
REM and that username belongs to someone else and must not ship with an
REM open-source repository.
if defined IDLETOKEN_MINGW_BIN (set "M=%IDLETOKEN_MINGW_BIN%") else (set "M=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin")
set PATH=%M%;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\OpenSSH\
del agent_build.log 2>nul
where windres >nul 2>&1
if errorlevel 1 goto :fail

REM Generate the compile-time compatibility version from the same package
REM metadata used by the browser client. Missing metadata is a hard failure;
REM never add a handwritten fallback here.
set "VERSION_HEADER=a_client_version.h"
del "%VERSION_HEADER%" 2>nul
set "IDLETOKEN_CLIENT_VERSION="
set "POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL%" goto :fail
REM PATH above is intentionally reduced to the toolchain and core Windows
REM directories. Windows PowerShell lives in a subdirectory of System32, so a
REM bare `powershell` is not resolvable here on a clean build node. Use the
REM operating-system path explicitly; never reuse a stale agent just because
REM version extraction did not start.
for /f "usebackq delims=" %%V in (`%POWERSHELL% -NoProfile -Command "$p = ConvertFrom-Json -InputObject (Get-Content -Raw 'client/package.json'); if (-not $p.version) { exit 2 }; $p.version"`) do set "IDLETOKEN_CLIENT_VERSION=%%V"
if not defined IDLETOKEN_CLIENT_VERSION goto :fail
>"%VERSION_HEADER%" echo #define IDLETOKEN_CLIENT_VERSION "%IDLETOKEN_CLIENT_VERSION%"

set CF=-O2 -std=gnu11 -Wall -Wextra -D_GNU_SOURCE -Iinclude -Ivendor/tweetnacl -Ivendor/blake2 -Isrc/platform/win -include src/platform/win/win_compat.h -include %VERSION_HEADER%

echo === crypto ===> agent_build.log
gcc -c vendor/tweetnacl/tweetnacl.c %CF% -o a_tweetnacl.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c vendor/blake2/blake2b.c %CF% -o a_blake2b.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/common/privacy.c %CF% -o a_privacy.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
REM src/common/sodium_seal.c, NOT src/tools/ -- it moved on 2026-08-19 (504e968)
REM when the coordinator started sealing overflow requests. This script kept
REM naming the old path and kept building, because sync-to-win.sh merges a
REM tarball and never deletes: the pre-move copy is still lying on the build
REM node, byte-identical today, so the drift was invisible. The next edit to
REM sodium_seal.c would simply not have reached the Windows agent.
gcc -c src/common/sodium_seal.c %CF% -o a_sodium_seal.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/platform/win/win_compat.c -O2 -std=gnu11 -Isrc/platform/win -o a_win_compat.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
windres -I src/platform/win src/platform/win/idletoken_utf8.rc -O coff -o a_utf8_manifest.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === common ===>> agent_build.log
gcc -c src/common/net.c %CF% -o a_net.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/common/http.c %CF% -o a_http.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
REM b64.c is on Makefile.platform's object list; this copy of the list did not
REM have it, so the Windows agent stopped linking (undefined idletoken_b64_*)
REM as soon as platform_agent.c started base64-ing. Two lists of the same
REM objects is the drift; keep them in step until one of them goes away.
gcc -c src/common/b64.c %CF% -o a_b64.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
REM Top-level JSON spans keep tool_choice intact when relaying agent work.
gcc -c src/common/apiconv.c %CF% -o a_apiconv.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
REM Local admission capabilities (PROV-28): the agent mints under the
REM coordinator's channel key so the coordinator can recognise its jobs.
gcc -c src/common/admission.c %CF% -o a_admission.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === agent ===>> agent_build.log
gcc -c src/tools/platform_agent.c %CF% -o a_platform_agent.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === link ===>> agent_build.log
gcc -static-libgcc -o idletoken-platform-agent.exe a_platform_agent.o a_sodium_seal.o a_privacy.o a_tweetnacl.o a_blake2b.o a_net.o a_http.o a_b64.o a_apiconv.o a_admission.o a_win_compat.o a_utf8_manifest.o -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic -lws2_32 -lbcrypt >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo AGENT_WIN_OK
del "%VERSION_HEADER%" 2>nul
exit /b 0

:fail
del "%VERSION_HEADER%" 2>nul
echo AGENT_WIN_FAIL
exit /b 1
