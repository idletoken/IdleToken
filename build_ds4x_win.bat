@echo off
REM Windows worker build WITH the ds4x backend (small models: Qwen3 GQA).
REM ds4x runs on the GPU here too: ds4x_cuda.cu is built by nvcc into its own
REM ds4xcuda.dll and reached through an import lib, so no MSVC object ever lands
REM in the MinGW link. The GPU is used automatically whenever the DLL loads and
REM finds a device -- since 2026-08-03 no runtime env var is needed (set
REM IDLETOKEN_DS4X_CPU=1 to force the CPU). -DIDLETOKEN_DS4X_CUDA below is still
REM REQUIRED at COMPILE time on every .c that touches the GPU path.
cd /d "%~dp0"
REM MinGW location: defaults to the per-user copy winget installs; override with
REM the IDLETOKEN_MINGW_BIN environment variable. This used to hardcode one
REM build machine's absolute path (username included) -- useless on any other
REM machine, and that username belongs to someone else and must not ship with an
REM open-source repository.
if defined IDLETOKEN_MINGW_BIN (set "M=%IDLETOKEN_MINGW_BIN%") else (set "M=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin")
set PATH=%M%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\OpenSSH\
del ds4x_build.log 2>nul
set CF=-D_GNU_SOURCE -Isrc/platform/win -Ivendor/ds4 -Iinclude -std=gnu11 -O2

REM ds4xcuda.dll is built separately by src\platform\win\build_ds4xcuda.bat
REM (it needs the MSVC env from vcvars64). Run that first when the .cu changes.
echo === dlltool ===> ds4x_build.log
dlltool -d src/platform/win/ds4cuda.def -l libds4cuda.a -D ds4cuda.dll >> ds4x_build.log 2>&1
dlltool -d src/platform/win/ds4xcuda.def -l libds4xcuda.a -D ds4xcuda.dll >> ds4x_build.log 2>&1

echo === vendor ===>> ds4x_build.log
gcc -c vendor/ds4/ds4.c %CF% -include src/platform/win/win_compat.h -o ds4.o >> ds4x_build.log 2>&1
gcc -c vendor/ds4/rax.c -Ivendor/ds4 -std=gnu11 -O2 -o rax.o >> ds4x_build.log 2>&1
gcc -c src/platform/win/win_compat.c -Isrc/platform/win -std=gnu11 -O2 -o win_compat.o >> ds4x_build.log 2>&1

echo === common ===>> ds4x_build.log
gcc -c src/common/net.c -Iinclude -std=gnu11 -O2 -o net.o >> ds4x_build.log 2>&1
gcc -c src/common/resource.c -Iinclude -std=gnu11 -O2 -o resource.o >> ds4x_build.log 2>&1
gcc -c src/common/weights.c -Isrc/platform/win -Iinclude -std=gnu11 -O2 -o weights.o >> ds4x_build.log 2>&1
gcc -c src/common/model.c -Iinclude -std=gnu11 -O2 -o model.o >> ds4x_build.log 2>&1
gcc -c src/common/gguf.c -Iinclude -std=gnu11 -O2 -o gguf.o >> ds4x_build.log 2>&1
gcc -c src/common/discovery.c -D_GNU_SOURCE -Isrc/platform/win -Iinclude -include src/platform/win/win_compat.h -std=gnu11 -O2 -o discovery.o >> ds4x_build.log 2>&1
REM node-crypto for the coord<->worker link (docs/inter-node-encryption.md):
REM nodecrypt.c holds the counter-nonce framing, privacy.c the XSalsa20-Poly1305
REM primitive, tweetnacl.c the vendored maths. Adding these to the Makefile is
REM NOT enough — this script has its own list, and a file missing here shows up
REM only as a Windows link error, on a machine nobody builds on daily.
gcc -c src/common/nodecrypt.c -Iinclude -Ivendor/tweetnacl -std=gnu11 -O2 -o nodecrypt.o >> ds4x_build.log 2>&1
gcc -c src/common/privacy.c -Isrc/platform/win -Iinclude -Ivendor/tweetnacl -include src/platform/win/win_compat.h -std=gnu11 -O2 -o privacy.o >> ds4x_build.log 2>&1
REM TweetNaCl is third-party: -w for the same reason the Makefile gives.
gcc -c vendor/tweetnacl/tweetnacl.c -Ivendor/tweetnacl -std=gnu11 -O2 -w -o tweetnacl.o >> ds4x_build.log 2>&1
REM plan.c + advise.c: the worker's --advise / --advise-json capability report
REM (and plan.c, which advise.c builds on). The Linux Makefile has always linked
REM these into the worker; this script did not, so the moment worker_main.c
REM actually called idletoken_advise* the Windows link died on three undefined
REM references. Keep the two build paths in step.
gcc -c src/common/plan.c -Iinclude -std=gnu11 -O2 -o plan.o >> ds4x_build.log 2>&1
gcc -c src/common/enginever.c -Iinclude -std=gnu11 -O2 -o enginever.o >> ds4x_build.log 2>&1
gcc -c src/common/advise.c -Iinclude -std=gnu11 -O2 -o advise.o >> ds4x_build.log 2>&1

echo === ds4x ===>> ds4x_build.log
gcc -c src/ds4x/ds4x_config.c  -Iinclude -std=gnu11 -O2 -o ds4x_config.o  >> ds4x_build.log 2>&1
gcc -c src/ds4x/ds4x_quant.c   -Iinclude -std=gnu11 -O2 -o ds4x_quant.o   >> ds4x_build.log 2>&1
gcc -c src/ds4x/ds4x_forward.c -Iinclude -DIDLETOKEN_DS4X_CUDA -std=gnu11 -O2 -o ds4x_forward.o >> ds4x_build.log 2>&1
gcc -c src/ds4x/ds4x_model.c   -Iinclude -DIDLETOKEN_DS4X_CUDA -std=gnu11 -O2 -o ds4x_model.o   >> ds4x_build.log 2>&1
REM ds4x_runner.c NEEDS -DIDLETOKEN_DS4X_CUDA too: that is where the linear-attention
REM recurrent state gets its VRAM handle (ds4x_cuda_gdn_create). Without the define
REM the layer silently keeps the recurrence on the CPU -- correct, just slower, and
REM invisible unless you diff IDLETOKEN_DS4X_PROF output against Linux.
REM (This fix lived only on the win-a box for a while; the repo copy had drifted
REM back to the broken form and overwriting the box with it cost ~6x per layer.)
gcc -c src/ds4x/ds4x_runner.c  -Iinclude -DIDLETOKEN_DS4X_CUDA -std=gnu11 -O2 -o ds4x_runner.o  >> ds4x_build.log 2>&1

echo === worker ===>> ds4x_build.log
REM worker_main.c ALSO needs -DIDLETOKEN_DS4X_CUDA: that is where the ds4x VRAM
REM budget is set from the machine's usable VRAM (ds4x_cuda_set_budget). Without
REM the define the call vanishes with the #ifdef and the budget stays 0 =
REM unlimited -- i.e. the user's "max VRAM" setting silently stops binding on
REM exactly the machines that have a small discrete card. Same trap as
REM ds4x_runner.c above; check with `objdump -p idletoken-worker.exe`.
gcc -c src/worker/worker_main.c %CF% -DIDLETOKEN_DS4X_CUDA -include src/platform/win/win_compat.h -o worker_main.o >> ds4x_build.log 2>&1

echo === link ===>> ds4x_build.log
REM -static-libgcc + static winpthread: so the exe can be copied to another
REM Windows box that has no MinGW on PATH (the 2070 laptop). Without it the
REM process dies before main() with NO output at all — not even --help.
gcc -static-libgcc -o idletoken-worker.exe worker_main.o ds4.o rax.o win_compat.o net.o resource.o weights.o model.o gguf.o discovery.o nodecrypt.o privacy.o tweetnacl.o plan.o advise.o enginever.o ds4x_config.o ds4x_quant.o ds4x_forward.o ds4x_model.o ds4x_runner.o -L. -lds4cuda -lds4xcuda -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic -lws2_32 -lbcrypt >> ds4x_build.log 2>&1
echo LINK_DONE >> ds4x_build.log
