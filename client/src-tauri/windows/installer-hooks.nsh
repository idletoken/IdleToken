# IdleToken NSIS lifecycle hooks.
#
# Three jobs, all of which exist because of one 2026-09-02 change: the installer
# went from currentUser (%LOCALAPPDATA%\IdleToken) to perMachine
# (C:\Program Files\IdleToken).
#
#   1. Stop running processes before files move (pre-existing).
#   2. Remove a previous per-user installation, so the machine does not end up
#      with two IdleTokens.
#   3. Provision the inbound firewall rules here, where the installer is
#      elevated and where firewall changes are ordinary — instead of at
#      runtime, where they were neither.
#
# WHY THE INSTALL LOCATION MOVED
#
# Running from %LOCALAPPDATA% is one of the strongest "this is malware"
# heuristics Windows has, because that is where software that cannot ask for
# admin puts itself. We were an unsigned binary in AppData that spawns child
# processes, listens on the LAN and calls netsh. Defender's behaviour engine
# flagged the coordinator as Behavior:Win32/DefenseEvasion.A!ml on a real
# cluster run (2026-09-02) and blocked it. This does not fix that on its own —
# only a code-signing certificate does — but it removes two of the inputs.

!macro IDLETOKEN_STOP_RUNNING_PROCESSES HOOK_NAME
  DetailPrint "Stopping running IdleToken processes..."
  nsExec::ExecToLog `powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$names=@('idletoken-client','idletoken-coord','idletoken-worker','idletoken-platform-agent','idletoken-server','idletoken-rpc-server'); Get-Process -Name 'idletoken-client' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 250; Get-Process -Name $$names -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; $$deadline=(Get-Date).AddSeconds(20); do { $$remaining=Get-Process -Name $$names -ErrorAction SilentlyContinue; if (-not $$remaining) { Start-Sleep -Milliseconds 500; exit 0 }; Start-Sleep -Milliseconds 250 } while ((Get-Date) -lt $$deadline); exit 1"`
  Pop $0
  StrCmp $0 "0" idletoken_processes_stopped_${HOOK_NAME}
  MessageBox MB_OK|MB_ICONSTOP "IdleToken could not stop its running inference processes. Close IdleToken and try the installation again." /SD IDOK
  Abort
idletoken_processes_stopped_${HOOK_NAME}:
!macroend

# Retire a pre-2026-09-02 per-user installation.
#
# Without this the machine keeps two copies: the old one still owns its Start
# Menu shortcut and its autostart entry, so the binary a user actually launches
# would be whichever shortcut they happen to click. The uninstaller is run
# silently and its result is deliberately NOT fatal — a failed cleanup is worth
# a warning, not a refusal to install the version that fixes things.
#
# NOTE: the old uninstaller removes %LOCALAPPDATA%\IdleToken, which is also
# where the coordinator keeps `rpc_psk` and the `model-views` cache. Both are
# regenerable (the PSK is minted per cluster and redistributed by pairing; the
# views are a cache), and after this migration that directory is data-only, so
# the collision is a one-time cost of the move rather than a standing one.
!macro IDLETOKEN_REMOVE_PER_USER_INSTALL
  ReadRegStr $0 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\IdleToken" "UninstallString"
  StrCmp $0 "" idletoken_no_per_user_install 0
  DetailPrint "Removing the previous per-user installation..."
  # The string is stored quoted; ExecWait wants it that way too.
  ExecWait '$0 /S' $1
  StrCmp $1 "0" idletoken_per_user_removed 0
  DetailPrint "Warning: the previous per-user installation did not uninstall cleanly (code $1). Remove it from Settings > Apps if it still appears."
idletoken_per_user_removed:
  # Whatever the uninstaller did, make sure no autostart entry still points at
  # the old location. A stale Run key would launch a binary this installer has
  # just replaced somewhere else.
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "IdleToken"
idletoken_no_per_user_install:
!macroend

# Inbound rules for the executables that actually accept LAN connections.
#
# Program rules, not port rules: the ports are chosen at runtime (the API port
# is a setting, worker and RPC ports are assigned per cluster), so a port list
# is a list that goes stale. A program rule covers whatever that program binds.
#
# profile=private,domain and NOT public. The old runtime code used profile=any,
# which opens the machine on untrusted networks too — a home cluster is a
# private network by definition, and "any" was a convenience nobody asked for.
#
# idletoken-server.exe is absent on purpose: it binds loopback only (hard
# constraint 5). So is the platform agent, which only makes outbound calls.
!macro IDLETOKEN_ADD_FIREWALL_RULE RULE_LABEL EXE_NAME
  # Chinese Windows emits localized netsh output in CP936. ExecToLog treated
  # those bytes as another code page and painted mojibake into the installer's
  # details pane. We only consume the exit code, so keep localized subprocess
  # output out of the Unicode NSIS UI entirely.
  nsExec::Exec `netsh advfirewall firewall delete rule name="IdleToken ${RULE_LABEL}"`
  Pop $0
  nsExec::Exec `netsh advfirewall firewall add rule name="IdleToken ${RULE_LABEL}" dir=in action=allow program="$INSTDIR\${EXE_NAME}" enable=yes profile=private,domain`
  Pop $0
  StrCmp $0 "0" +2 0
  DetailPrint "Warning: could not add the firewall rule for ${EXE_NAME}. Cluster mode may need it added by hand."
!macroend

!macro IDLETOKEN_DELETE_FIREWALL_RULE RULE_LABEL
  nsExec::Exec `netsh advfirewall firewall delete rule name="IdleToken ${RULE_LABEL}"`
  Pop $0
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro IDLETOKEN_STOP_RUNNING_PROCESSES PREINSTALL
  !insertmacro IDLETOKEN_REMOVE_PER_USER_INSTALL
!macroend

!macro NSIS_HOOK_POSTINSTALL
  DetailPrint "Allowing IdleToken through Windows Firewall on private networks..."
  !insertmacro IDLETOKEN_ADD_FIREWALL_RULE "coordinator" "idletoken-coord.exe"
  !insertmacro IDLETOKEN_ADD_FIREWALL_RULE "worker" "idletoken-worker.exe"
  !insertmacro IDLETOKEN_ADD_FIREWALL_RULE "compute node" "idletoken-rpc-server.exe"
  !insertmacro IDLETOKEN_ADD_FIREWALL_RULE "app" "idletoken-client.exe"
  # Give the all-users shortcut an explicit icon source. Depending on an empty
  # IconLocation made Explorer keep the generic white icon after an in-place
  # upgrade even though the executable's embedded icon was valid. Recreating
  # the same shortcut is idempotent, Unicode-safe, and refreshes the shell cache
  # without changing where the shortcut lives.
  SetShellVarContext all
  CreateShortCut "$SMPROGRAMS\IdleToken.lnk" "$INSTDIR\idletoken-client.exe" "" "$INSTDIR\idletoken-client.exe" 0 SW_SHOWNORMAL "" "IdleToken"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro IDLETOKEN_STOP_RUNNING_PROCESSES PREUNINSTALL
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  !insertmacro IDLETOKEN_DELETE_FIREWALL_RULE "coordinator"
  !insertmacro IDLETOKEN_DELETE_FIREWALL_RULE "worker"
  !insertmacro IDLETOKEN_DELETE_FIREWALL_RULE "compute node"
  !insertmacro IDLETOKEN_DELETE_FIREWALL_RULE "app"
!macroend
