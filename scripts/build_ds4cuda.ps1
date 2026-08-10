# Build ds4cuda.dll via nvcc + MSVC host. Imports vcvars env into PowerShell and
# calls nvcc directly (nested .bat hangs under Start-Process). Exports the
# extern "C" ds4_gpu_* API via ds4cuda.def so the MinGW idletoken-worker.exe can
# import it. Multi-arch sm_75 (RTX 2070) + sm_120 (RTX 5060 Ti) + PTX.
#
# CUDA version: default 12.8 — its runtime runs on any r525+ driver (Jan 2023)
# via CUDA-12 minor-version compatibility, honoring the driver-only principle
# on machines with older drivers. (13.3 demanded r580+, which rejected the
# 555.85-driver laptop.) Override: -CudaVer 13.3
param([string]$CudaVer = "12.8", [switch]$FastLocal)
# -FastLocal: generate code for this machine's architecture only, skipping sm_75
#   and PTX. A release must use the full set (it has to cover every card from
#   Turing onwards, plus JIT forward compatibility); but in a debugging loop on
#   real hardware, compiling three code generations every time means two of them
#   are useless to the machines at hand (both measured here are sm_120).
Set-Location "$env:USERPROFILE\IdleToken"
Remove-Item cuda_build.log,ds4cuda.dll,ds4cuda.lib,ds4cuda.exp -EA SilentlyContinue

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaVer\bin;$env:PATH"
# cl.exe reads %CL% and applies it to EVERY cl invocation — including nvcc's
# internal passes that -Xcompiler/-D don't reach. CUDA 13.x CCCL requires the
# conforming preprocessor; harmless on 12.x.
$env:CL = "/Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"

$nvccArgs = @(
    '-O3','--use_fast_math','-shared','-std=c++17','-o','ds4cuda.dll','vendor/ds4/ds4_cuda.cu',
    '-I','vendor/ds4',
    '-I','src/platform/win-msvc',                        # POSIX shims (unistd.h) for the MSVC compile
    # The architecture list below depends on -FastLocal
    '-Xcompiler','/Zc:preprocessor',
    '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING',
    '-Xlinker','/DEF:src/platform/win/ds4cuda.def',
    # cuBLAS is DELAY-LOADED: we stopped shipping cublas64_12.dll (660 MB) +
    # cublasLt64_12.dll (108 MB) — 97% of the payload, and Toolkit-only, not
    # driver (product decision 2026-08-04). With a normal import a machine
    # without the Toolkit dies at 0xC0000135 before main() and cannot print the
    # "go install it" message the decision depends on. cuda_init probes with
    # LoadLibrary first; every cuBLAS call site is guarded by g_cublas_ready.
    '-Xlinker','/DELAYLOAD:cublas64_12.dll',
    '-Xlinker','delayimp.lib',
    '-lcublas'
)

if ($FastLocal) {
    $nvccArgs += @('-gencode','arch=compute_120,code=sm_120')
} else {
    $nvccArgs += @('-gencode','arch=compute_75,code=sm_75',
                   '-gencode','arch=compute_120,code=sm_120',
                   '-gencode','arch=compute_120,code=compute_120')
}
# Use cmd redirection (PowerShell mangles native stderr into NativeCommandError).
$argStr = $nvccArgs -join ' '
cmd /c "nvcc $argStr > cuda_build.log 2>&1"
"NVCC_EXIT=$LASTEXITCODE" | Add-Content cuda_build.log
