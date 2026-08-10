@echo off
REM Windows launcher for an ds4x-backed worker (small models: Qwen3 / Qwen3.5 ...).
REM
REM The GPU is now the DEFAULT (2026-08-03): a CUDA-built worker uses it whenever
REM a usable device is present, so this launcher no longer has to set anything.
REM It used to be a RUNTIME opt-in via IDLETOKEN_DS4X_CUDA, and forgetting it meant
REM ds4x ran on the CPU: correct output, ~28x slower, nothing in the log saying
REM so. Measured on the 2-GPU cross-machine cluster (qwen3-8b Q4_K_M): 2.931
REM s/token without it, 0.106 with. That variable is still accepted (harmless).
REM Note -DIDLETOKEN_DS4X_CUDA remains a COMPILE-time requirement -- see
REM build_ds4x_win.bat. Set IDLETOKEN_DS4X_CPU=1 to force the CPU path.
REM
REM IDLETOKEN_DS4X_PROF=1 makes the worker print its per-stage compute time on exit,
REM which is how you tell a CPU fallback from a slow GPU in the first place.
cd /d "%~dp0"
set IDLETOKEN_DS4X_CUDA=1
if "%IDLETOKEN_PROF%"=="1" set IDLETOKEN_DS4X_PROF=1
if "%~1"=="" (
  echo usage: runworker-ds4x-win.bat ^<coord-host:port^> [bind-host:port] [gguf-dir]
  echo   e.g. runworker-ds4x-win.bat 192.168.1.50:14100 192.168.1.51:14101 D:\gguf
  exit /b 2
)
set COORD=%~1
set BIND=%~2
set GGUF=%~3
if "%BIND%"=="" set BIND=0.0.0.0:14101
if "%GGUF%"=="" set GGUF=D:\gguf
idletoken-worker.exe --coordinator %COORD% --bind %BIND% --gguf-dir %GGUF%
