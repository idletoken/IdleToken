# IdleToken NSIS lifecycle hooks.
#
# An install-over can start after the Tauri shell has closed while one of its
# inference sidecars is still alive. On Windows, a process that loaded a CUDA
# DLL keeps the file locked, so NSIS used to stop at cublasLt64_12.dll and ask
# the user to Retry. Stop every product-owned process before NSIS copies or
# removes files, then wait for Windows to release their module handles.

!macro IDLETOKEN_STOP_RUNNING_PROCESSES HOOK_NAME
  DetailPrint "Stopping running IdleToken processes..."
  nsExec::ExecToLog `powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$names=@('idletoken-client','idletoken-coord','idletoken-worker','idletoken-platform-agent','idletoken-server','idletoken-rpc-server'); Get-Process -Name 'idletoken-client' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 250; Get-Process -Name $$names -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; $$deadline=(Get-Date).AddSeconds(20); do { $$remaining=Get-Process -Name $$names -ErrorAction SilentlyContinue; if (-not $$remaining) { Start-Sleep -Milliseconds 500; exit 0 }; Start-Sleep -Milliseconds 250 } while ((Get-Date) -lt $$deadline); exit 1"`
  Pop $0
  StrCmp $0 "0" idletoken_processes_stopped_${HOOK_NAME}
  MessageBox MB_OK|MB_ICONSTOP "IdleToken could not stop its running inference processes. Close IdleToken and try the installation again." /SD IDOK
  Abort
idletoken_processes_stopped_${HOOK_NAME}:
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro IDLETOKEN_STOP_RUNNING_PROCESSES PREINSTALL
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro IDLETOKEN_STOP_RUNNING_PROCESSES PREUNINSTALL
!macroend
