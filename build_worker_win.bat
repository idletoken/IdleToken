@echo off
REM Build idletoken-worker.exe on Windows (MinGW / WinLibs gcc). NO ds4/ds4x.
REM
REM WHY THIS EXISTS (2026-08-16). The only Windows worker build was
REM build_ds4x_win.bat, which compiles the ds4x kernels and links ds4cuda /
REM ds4xcuda. ds4/ds4x are shelved (hard constraint #1): not a backend, not
REM tested, not published. Building them cost a CUDA Toolkit dependency — a
REM machine with only the NVIDIA driver could not build the worker at all —
REM and about ten minutes of nvcc for code that never executes.
REM
REM This script builds the same worker against src/common/ds4_stub.c: the call
REM sites in worker_main.c are untouched (frozen code), the real ds4 objects
REM are simply not compiled. Needs gcc only; no CUDA Toolkit, no nvcc.
REM
REM Usage:  build_worker_win.bat            (from the repo root)
REM Prints WORKER_BUILD_OK or WORKER_BUILD_FAIL: <reason>. Both go to stdout —
REM the ds4x script printed LINK_DONE unconditionally, so a build that failed
REM at the first missing compiler still reported success, and two separate
REM "successful" builds turned out to have produced nothing.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Same repo-local toolchain probe as build_coord_win.bat: a WinLibs archive
REM unzipped into <repo>\mingw64 is enough, and nodes without winget (Windows 11
REM Home) then need no package manager and leave no system state behind.
if "%IDLETOKEN_MINGW_BIN%"=="" if exist "%~dp0mingw64\bin\gcc.exe" set "IDLETOKEN_MINGW_BIN=%~dp0mingw64\bin"
if "%IDLETOKEN_MINGW_BIN%"=="" set "IDLETOKEN_MINGW_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
set "PATH=%IDLETOKEN_MINGW_BIN%;%PATH%"
where gcc >NUL 2>&1 || (echo WORKER_BUILD_FAIL: no gcc on PATH ^(set IDLETOKEN_MINGW_BIN^) & exit /b 1)

set "LOG=worker_build.log"
del "%LOG%" 2>nul
set "CF=-D_GNU_SOURCE -Isrc/platform/win -Ivendor/ds4 -Iinclude -std=gnu11 -O2"
set "OBJ=build\win-worker"
if not exist "%OBJ%" mkdir "%OBJ%"

REM Every compile appends to the log and its exit code is checked immediately:
REM a failure four steps back is invisible by link time, which is exactly how
REM the old script kept "succeeding".
call :cc "vendor/ds4/rax.c"          "-Ivendor/ds4 -std=gnu11 -O2"                          rax          || goto :fail
call :cc "src/platform/win/win_compat.c" "-Isrc/platform/win -std=gnu11 -O2"                win_compat   || goto :fail
call :cc "src/common/net.c"          "-Iinclude -std=gnu11 -O2"                             net          || goto :fail
call :cc "src/common/resource.c"     "-Iinclude -std=gnu11 -O2"                             resource     || goto :fail
call :cc "src/common/weights.c"      "-Isrc/platform/win -Iinclude -std=gnu11 -O2"          weights      || goto :fail
call :cc "src/common/model.c"        "-Iinclude -std=gnu11 -O2"                             model        || goto :fail
call :cc "src/common/modelsize.c"    "-Isrc/platform/win -Iinclude -std=gnu11 -include src/platform/win/win_compat.h -O2" modelsize || goto :fail
call :cc "src/common/gguf.c"         "-Iinclude -std=gnu11 -O2"                             gguf         || goto :fail
call :cc "src/common/discovery.c"    "%CF% -include src/platform/win/win_compat.h"          discovery    || goto :fail
call :cc "src/common/nodecrypt.c"    "-Iinclude -Ivendor/tweetnacl -std=gnu11 -O2"          nodecrypt    || goto :fail
call :cc "src/common/privacy.c"      "-Isrc/platform/win -Iinclude -Ivendor/tweetnacl -include src/platform/win/win_compat.h -std=gnu11 -O2" privacy || goto :fail
call :cc "vendor/tweetnacl/tweetnacl.c" "-Ivendor/tweetnacl -std=gnu11 -O2 -w"              tweetnacl    || goto :fail
call :cc "src/common/plan.c"         "-Iinclude -std=gnu11 -O2"                             plan         || goto :fail
call :cc "src/common/enginever.c"    "-Iinclude -std=gnu11 -O2"                             enginever    || goto :fail
call :cc "src/common/advise.c"       "-Iinclude -std=gnu11 -O2"                             advise       || goto :fail

REM The stub stands in for the whole ds4/ds4x API. It includes the real headers,
REM so a signature drifting from its declaration is a compile error here.
call :cc "src/common/ds4_stub.c"     "%CF%"                                                 ds4_stub     || goto :fail

REM No -DIDLETOKEN_DS4X_CUDA: that define exists to wire worker_main.c into the
REM ds4x CUDA kernels, which this build does not have. Leaving it out is what
REM keeps ds4x_cuda_* out of the link.
call :cc "src/worker/worker_main.c"  "%CF% -include src/platform/win/win_compat.h"          worker_main  || goto :fail

echo === link ===>> "%LOG%"
REM -static-libgcc + static winpthread: the exe gets copied to machines with no
REM MinGW on PATH (win_PC2 has no toolchain at all — it receives binaries from
REM the build node). Without it the process dies before main() with NO output.
REM Link EVERY object this script compiled, by glob -- the same shape
REM build_coord_win.bat uses. It used to enumerate them by hand, and the list
REM drifted: modelsize.c was compiled but never linked, so once advise.c started
REM calling idletoken_model_size_resolve (commit 6f0d92c) the Windows worker
REM stopped linking entirely -- `undefined reference to idletoken_model_size_resolve`
REM -- while Linux and macOS, which build from the Makefile's object list, were
REM fine. Nobody noticed because both Windows boxes still had a worker exe from
REM before that commit. A hand-maintained parallel list of the files right above
REM it is a drift generator; the glob cannot miss one.
REM (Nothing unwanted lands in %OBJ%: ds4x is not compiled here at all, the stub
REM stands in for it.)
gcc -static-libgcc -o idletoken-worker.exe "%OBJ%\*.o" ^
    -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic -lws2_32 -lbcrypt >> "%LOG%" 2>&1
if errorlevel 1 (echo WORKER_BUILD_FAIL: link failed ^(see %LOG%^) & exit /b 1)
if not exist idletoken-worker.exe (echo WORKER_BUILD_FAIL: no idletoken-worker.exe produced & exit /b 1)

REM Prove the binary actually starts. A link can succeed and the exe still die
REM before main() on a missing runtime DLL — measured on win_PC2, where
REM llama-server.exe exited 0 with zero output until the CUDA runtime DLLs sat
REM next to it. "It linked" is not "it runs".
idletoken-worker.exe --help >NUL 2>&1
if errorlevel 1 (echo WORKER_BUILD_FAIL: idletoken-worker.exe does not start & exit /b 1)

echo WORKER_BUILD_OK
exit /b 0

:cc
REM :cc <source> <flags> <objname>
echo === %~3 ===>> "%LOG%"
gcc -c %~1 %~2 -o "%OBJ%\%~3.o" >> "%LOG%" 2>&1
if errorlevel 1 (echo WORKER_BUILD_FAIL: compiling %~1 ^(see %LOG%^) & exit /b 1)
exit /b 0

:fail
exit /b 1
