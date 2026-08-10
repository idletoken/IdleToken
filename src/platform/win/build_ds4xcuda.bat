@echo off
REM Build ds4xcuda.dll — ds4x GPU matvec as a C-ABI DLL (nvcc + MSVC host),
REM mirroring build_ds4cuda.bat. MinGW's idletoken-worker.exe imports it via an
REM import lib that dlltool makes from ds4xcuda.def.
REM
REM Cross-CRT safety: the boundary passes only read-only host pointers + sizes,
REM and handles from ds4x_cuda_upload are freed by ds4x_cuda_free INSIDE the
REM DLL. No fd / FILE* / malloc-here-free-there (the trap ds4 hit).
REM
REM NOTE: each line re-expands %PATH%, which is why vcvars must be `call`ed on
REM its own line — doing it all on one command line expands %PATH% before
REM vcvars runs and nvcc then can't find cl.exe.
REM Optional arg 1 = CUDA toolkit version dir. DEFAULT IS v12.8, deliberately:
REM build against the OLDEST toolkit your fleet's driver supports. A DLL built
REM with CUDA 13.x will NOT initialise on a 12.x driver, and the 2070 laptop
REM runs driver 555.85 = CUDA 12.5. CUDA minor-version compatibility makes a
REM 12.8 build run fine on 12.5+ drivers, so v12.8 works on the WHOLE fleet
REM while v13.3 silently loses the Turing node to a CPU fallback.
REM (2026-07-28: the default used to be v13.3, and a v13.3 DLL sitting in the
REM build dir was mistaken for a 12.8 one — cost a wrong "can't be done"
REM conclusion. Pass v13.3 explicitly if you ever need it.)
setlocal
set "CUDAVER=%~1"
if "%CUDAVER%"=="" set "CUDAVER=v12.8"
cd /d "%USERPROFILE%\IdleToken"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDAVER%\bin;%PATH%"
nvcc -O3 --use_fast_math -shared -o ds4xcuda.dll src\ds4x\ds4x_cuda.cu ^
  -I include ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_120,code=sm_120 ^
  -gencode arch=compute_120,code=compute_120 ^
  -Xlinker /DEF:src\platform\win\ds4xcuda.def ^
  -cudart static
echo NVCC_EXIT=%ERRORLEVEL%
endlocal
