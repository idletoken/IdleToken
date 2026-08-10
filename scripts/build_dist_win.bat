@echo off
REM Assemble the self-contained Windows dist\ folder (E3 / gate G3, Windows half).
REM
REM WHY THIS EXISTS (2026-08-04). dist\ used to be assembled BY HAND. On
REM 2026-08-04 its worker turned out to be from 07-16: no ds4x at all, so the
REM shipped bundle could not run a single small model, had no --advise and no
REM --model-id. G3 stayed green the whole time because its Windows half only
REM asks "does it start driver-only" — a three-week-old binary answers that
REM just fine. Hand-copied folders drift; scripts do not.
REM
REM Sources: the binaries built by build_ds4x_win.bat / build_coord_win.bat /
REM build_agent_win.bat in the repo root, plus the CUDA RUNTIME dlls.
REM
REM CUDA version must MATCH what ds4cuda.dll/ds4xcuda.dll were built against
REM (v12.8 — see scripts/build_ds4cuda.ps1 for why: a 12.8 build runs on any
REM r525+ driver, a 13.x build rejects the 555.85 laptop). Shipping 13.x
REM runtime dlls beside a 12.8-built kernel dll is a silent CPU fallback.
setlocal
set "CUDAVER=%~1"
if "%CUDAVER%"=="" set "CUDAVER=v12.8"
set "CUDABIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDAVER%\bin"
cd /d "%USERPROFILE%\IdleToken"

if not exist idletoken-worker.exe (
  echo DIST_FAIL: no idletoken-worker.exe in the repo root — run build_ds4x_win.bat first
  exit /b 1
)

if not exist dist mkdir dist

echo === engine binaries ===
for %%F in (idletoken-worker.exe idletoken-coord.exe idletoken-platform-agent.exe ds4cuda.dll ds4xcuda.dll) do (
  if exist %%F (
    copy /y %%F dist\%%F >nul
  ) else (
    echo   WARNING: %%F missing in repo root — dist\ will be incomplete
  )
)

REM === NO CUDA RUNTIME DLLs (2026-08-04 product decision) ===
REM We used to copy cudart + cublas + cublasLt in here. Measured on an actual
REM install: cublasLt64_12.dll 660.4 MB + cublas64_12.dll 108.4 MB = 97% of the
REM whole payload (everything else together is ~21 MB). They are NOT part of the
REM display driver — they come from the CUDA Toolkit — so the user installs the
REM Toolkit now (docs/acceptance-criteria.md philosophy 12, E3).
REM
REM Nothing needs staging: `objdump -p` shows ds4cuda.dll and ds4xcuda.dll import
REM only KERNEL32 at load time (nvcc links cudart statically on Windows), and
REM cuBLAS is DELAY-loaded with a LoadLibrary probe + actionable message in
REM cuda_init. So a machine without the Toolkit still starts and still answers —
REM just with the built-in prefill kernels (measured 9.1x slower).

REM Remove CUDA runtime DLLs left by earlier builds. "Stop copying them" is not
REM the same as "they are gone": dist\ is a directory, not a build product, so a
REM 660 MB file copied in three weeks ago keeps shipping until something deletes
REM it — and the package would silently stay 447 MB.
REM Old-name binaries from before the 2026-08-04 IdleToken rename. Same reason
REM as the CUDA dlls below: dist\ is a directory, so anything an earlier build
REM copied in stays until something deletes it — and the bundle would ship TWO
REM engines under two names.
for %%F in (homeai-worker.exe homeai-coord.exe homeai-platform-agent.exe) do (
  if exist dist\%%F (
    del /q dist\%%F
    echo   removed stale dist\%%F  ^(pre-rename name^)
  )
)

for %%F in (cublas64_12.dll cublasLt64_12.dll cudart64_12.dll) do (
  if exist dist\%%F (
    del /q dist\%%F
    echo   removed stale dist\%%F  ^(now supplied by the user's CUDA Toolkit^)
  )
)

REM Print every file WITH ITS TIMESTAMP. On 2026-07-29 a stale ds4cuda.dll went
REM into a release and the installed worker fell back to MOCK after a
REM cudaMemcpy error — the bundle looked fine, the answers were garbage. Dates
REM are the cheapest way to see that before shipping.
echo === dist\ contents ===
dir dist

echo === self-check (this is what gate G3 asserts) ===
dist\idletoken-worker.exe --help
echo DIST_WIN_DONE
endlocal
