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

set CF=-O2 -std=gnu11 -Wall -Wextra -D_GNU_SOURCE -Iinclude -Ivendor/tweetnacl -Ivendor/blake2 -Isrc/platform/win -include src/platform/win/win_compat.h

echo === crypto ===> agent_build.log
gcc -c vendor/tweetnacl/tweetnacl.c %CF% -o a_tweetnacl.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c vendor/blake2/blake2b.c %CF% -o a_blake2b.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/common/privacy.c %CF% -o a_privacy.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/tools/sodium_seal.c %CF% -o a_sodium_seal.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/platform/win/win_compat.c -O2 -std=gnu11 -Isrc/platform/win -o a_win_compat.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === common ===>> agent_build.log
gcc -c src/common/net.c %CF% -o a_net.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/common/http.c %CF% -o a_http.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === agent ===>> agent_build.log
gcc -c src/tools/platform_agent.c %CF% -o a_platform_agent.o >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo === link ===>> agent_build.log
gcc -static-libgcc -o idletoken-platform-agent.exe a_platform_agent.o a_sodium_seal.o a_privacy.o a_tweetnacl.o a_blake2b.o a_net.o a_http.o a_win_compat.o -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic -lws2_32 -lbcrypt >> agent_build.log 2>&1
if errorlevel 1 goto :fail

echo AGENT_WIN_OK
exit /b 0

:fail
echo AGENT_WIN_FAIL
exit /b 1
