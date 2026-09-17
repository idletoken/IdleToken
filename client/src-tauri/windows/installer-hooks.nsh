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
# Installation operations use Windows Restart Manager and Firewall COM APIs.
# The small helper is built, versioned and signed with the other first-party
# executables. It is extracted to the installer's private temporary directory;
# it never registers a service or runs after the installer exits.
!define IDLETOKEN_HELPER_SOURCE "${__FILEDIR__}\..\runtime\windows\idletoken-installer-helper.exe"

!macro IDLETOKEN_EXTRACT_HELPER
  InitPluginsDir
  Push $R9
  StrCpy $R9 $OUTDIR
  SetOutPath "$PLUGINSDIR"
  File /oname=idletoken-installer-helper.exe "${IDLETOKEN_HELPER_SOURCE}"
  SetOutPath $R9
  Pop $R9
!macroend

!macro IDLETOKEN_STOP_RUNNING_PROCESSES HOOK_NAME
  !insertmacro IDLETOKEN_EXTRACT_HELPER
  DetailPrint "Closing running IdleToken applications..."
  Push $R9
  SetShellVarContext current
  StrCpy $R9 "$LOCALAPPDATA\IdleToken"
  SetShellVarContext all
  nsExec::ExecToLog '"$PLUGINSDIR\idletoken-installer-helper.exe" --stop "$INSTDIR" "$R9"'
  Pop $0
  Pop $R9
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

# Compute rules are restricted to TCP/UDP as required and local subnets.
# Pairing retains its private-network route options. Public-network profiles
# are never enabled. Uninstallation removes only this installation's rules.
!macro IDLETOKEN_FIREWALL OPERATION
  !insertmacro IDLETOKEN_EXTRACT_HELPER
  nsExec::ExecToLog '"$PLUGINSDIR\idletoken-installer-helper.exe" --firewall-${OPERATION} "$INSTDIR"'
  Pop $0
  StrCmp $0 "0" +2 0
  DetailPrint "Warning: IdleToken firewall configuration could not be updated. Cluster connections may require administrator attention."
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro IDLETOKEN_STOP_RUNNING_PROCESSES PREINSTALL
  !insertmacro IDLETOKEN_REMOVE_PER_USER_INSTALL
!macroend

!macro NSIS_HOOK_POSTINSTALL
  DetailPrint "Allowing IdleToken through Windows Firewall on private networks..."
  !insertmacro IDLETOKEN_FIREWALL install
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
  !insertmacro IDLETOKEN_FIREWALL remove
!macroend
