@echo off
REM Native Windows build of the COORDINATOR (idletoken-coord.exe).
REM
REM Why this exists: the product promise is "any machine can be the
REM coordinator" (acceptance §1 — the user picks the role in the client). Until
REM now only the worker had a Windows build, so an all-Windows household had no
REM machine that could run the cluster — the client shipped a copy of the worker
REM under the coord's name, which fails at spawn time.
REM
REM The coord is the easy half of the port: no CUDA, no pthreads, no mmap of the
REM 80GB model — just sockets, the planner, the HTTP server and the GGUF
REM tokenizer. Built with the same MinGW toolchain as the worker and with
REM -DDS4_NO_GPU, mirroring the Makefile's CFLAGS_COORD.
REM
REM Run from the repo root:  scripts\build_coord_win.bat
REM Contract: prints COORD_WIN_OK or COORD_WIN_FAIL (details in coord_build.log).
setlocal
cd /d "%~dp0.."
REM MinGW location: defaults to the per-user copy winget installs; override with
REM the IDLETOKEN_MINGW_BIN environment variable. This used to hardcode one build
REM machine's absolute path (username included) -- useless on any other machine,
REM and that username belongs to someone else and must not ship with an
REM open-source repository.
if defined IDLETOKEN_MINGW_BIN (set "M=%IDLETOKEN_MINGW_BIN%") else (set "M=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin")
set PATH=%M%;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\OpenSSH\
del coord_build.log 2>nul

REM Same flags as CFLAGS_COORD in the Makefile, plus the Windows shim.
set CF=-D_GNU_SOURCE -DDS4_NO_GPU -Isrc/platform/win -Ivendor/ds4 -Iinclude -std=gnu11 -O2 -include src/platform/win/win_compat.h

echo === vendor ===> coord_build.log
gcc -c vendor/ds4/ds4.c %CF% -o c_ds4.o >> coord_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c vendor/ds4/rax.c -Ivendor/ds4 -std=gnu11 -O2 -o c_rax.o >> coord_build.log 2>&1
if errorlevel 1 goto :fail
gcc -c src/platform/win/win_compat.c -Isrc/platform/win -std=gnu11 -O2 -o c_win_compat.o >> coord_build.log 2>&1
if errorlevel 1 goto :fail

echo === common ===>> coord_build.log
REM advise.c = the capability table served by GET /v1/capability.
for %%F in (net discovery model http plan gguf advise) do (
    gcc -c src/common/%%F.c %CF% -o c_%%F.o >> coord_build.log 2>&1
    if errorlevel 1 goto :fail
)

echo === ds4x tokenizer ===>> coord_build.log
gcc -c src/ds4x/ds4x_tokenizer.c %CF% -o c_ds4x_tokenizer.o >> coord_build.log 2>&1
if errorlevel 1 goto :fail

echo === coord ===>> coord_build.log
gcc -c src/coord/coord_main.c %CF% -o c_coord_main.o >> coord_build.log 2>&1
if errorlevel 1 goto :fail

echo === link ===>> coord_build.log
REM -static-libgcc + static winpthread: the exe must run on a machine with only
REM the NVIDIA driver and no MinGW on PATH (same reason as the worker link).
gcc -static-libgcc -o idletoken-coord.exe c_coord_main.o c_net.o c_discovery.o c_model.o c_http.o c_plan.o c_gguf.o c_advise.o c_ds4x_tokenizer.o c_ds4.o c_rax.o c_win_compat.o -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic -lws2_32 -lbcrypt >> coord_build.log 2>&1
if errorlevel 1 goto :fail

echo COORD_WIN_OK
exit /b 0

:fail
echo COORD_WIN_FAIL
exit /b 1
