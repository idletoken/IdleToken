param([string]$RepoRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:TEMP ('IdleToken-installer-test-' + [guid]::NewGuid().ToString('N'))
$prefix = 'IdleToken test ' + [guid]::NewGuid().ToString('N')
$children = @()
try {
    $compiler = (Get-Command g++.exe -ErrorAction Stop).Source
    # Use real Unicode paths without depending on PowerShell 5.1's ANSI decoder.
    $target = Join-Path $root ('installed ' + [char]0x6D4B + [char]0x8BD5)
    $other = Join-Path $root 'unrelated installation'
    New-Item -ItemType Directory -Force -Path $target,$other | Out-Null
    $source = Join-Path $root 'fixture.cpp'
    [IO.File]::WriteAllText($source, '#include <windows.h>
int main() { Sleep(120000); return 0; }', [Text.Encoding]::ASCII)
    & $compiler -O2 -static $source -o "$root\fixture.exe"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot build process fixture.' }
    Copy-Item -LiteralPath "$root\fixture.exe" -Destination "$target\idletoken-worker.exe"
    Copy-Item -LiteralPath "$target\idletoken-worker.exe" -Destination "$other\idletoken-worker.exe"
    $helper = "$root\helper.exe"
    # Firewall tests get a unique group/name, never touching real product rules.
    $header = "$root\test-prefix.h"
    [IO.File]::WriteAllText($header, ('#define IDLETOKEN_FIREWALL_PREFIX L"' + $prefix + '"'), [Text.Encoding]::ASCII)
    & $compiler -std=c++17 -O2 -static -municode -include $header "$RepoRoot\src\platform\win\installer_helper.cpp" `
        -lrstrtmgr -lole32 -loleaut32 -luuid -o $helper
    if ($LASTEXITCODE -ne 0) { throw 'Cannot build isolated installer helper.' }
    $inside = Start-Process -FilePath "$target\idletoken-worker.exe" -PassThru -WindowStyle Hidden
    $outside = Start-Process -FilePath "$other\idletoken-worker.exe" -PassThru -WindowStyle Hidden
    $children = @($inside,$outside)
    & $helper --stop $target
    if ($LASTEXITCODE -ne 0) { throw 'Scoped process shutdown failed.' }
    $inside.Refresh(); $outside.Refresh()
    if (-not $inside.HasExited -or $outside.HasExited) { throw 'Process scope violation: expected only the target installation to stop.' }
    Write-Output 'PASS process shutdown: target stopped, unrelated same-name executable survived'

    foreach ($name in @('idletoken-coord','idletoken-rpc-server','idletoken-client')) {
        Copy-Item -LiteralPath "$target\idletoken-worker.exe" -Destination "$target\$name.exe"
    }
    & $helper --firewall-install $target
    if ($LASTEXITCODE -ne 0) { throw 'Firewall installation failed (run this gate elevated).' }
    # Independent readback uses the Windows NetSecurity module, not helper code.
    $installed = @(Get-NetFirewallRule -Group $prefix -ErrorAction Stop)
    if ($installed.Count -ne 6) { throw 'Expected exactly six firewall rules.' }
    foreach ($rule in $installed) {
        if ([int]$rule.Profile -ne 3 -or "$($rule.Direction)" -ne 'Inbound' -or "$($rule.Action)" -ne 'Allow') { throw 'Firewall profile/direction/action mismatch.' }
        $app = $rule | Get-NetFirewallApplicationFilter
        if (-not $app.Program.StartsWith($target, [StringComparison]::OrdinalIgnoreCase)) { throw 'Firewall application path mismatch.' }
        $port = $rule | Get-NetFirewallPortFilter
        $address = $rule | Get-NetFirewallAddressFilter
        $expectedProtocol = if ($rule.DisplayName -match 'discovery$|worker$') { 'UDP' } else { 'TCP' }
        if ("$($port.Protocol)" -ne $expectedProtocol) { throw 'Unexpected firewall protocol.' }
        $expectedRemote = if ($rule.DisplayName -match ' app( discovery)?$') { 'Any' } else { 'LocalSubnet' }
        if (@($address.RemoteAddress).Count -ne 1 -or "$($address.RemoteAddress)" -ne $expectedRemote) { throw 'Unexpected firewall network scope.' }
    }
    & $helper --firewall-install $target
    if ($LASTEXITCODE -ne 0 -or @(Get-NetFirewallRule -Group $prefix).Count -ne 6) { throw 'Reinstallation created duplicate rules.' }
    & $helper --firewall-remove $other
    if ($LASTEXITCODE -ne 0 -or @(Get-NetFirewallRule -Group $prefix).Count -ne 6) { throw 'Uninstall removed another installation''s rules.' }
    & $helper --firewall-remove $target
    if ($LASTEXITCODE -ne 0 -or @(Get-NetFirewallRule -Group $prefix -ErrorAction SilentlyContinue).Count -ne 0) { throw 'Firewall rules survived removal.' }
    Write-Output 'PASS firewall: private/domain only, exact protocols, scoped peers, idempotence and directory ownership'
    Write-Output 'WINDOWS_INSTALLER_GATE_OK'
} finally {
    foreach ($child in $children) { if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit() } }
    # Cleanup is also namespace-scoped, including failed test runs.
    Get-NetFirewallRule -Group $prefix -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
