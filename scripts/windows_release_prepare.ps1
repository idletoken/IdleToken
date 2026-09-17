param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory=$true)][ValidateSet('x86_64-pc-windows-msvc','x86_64-pc-windows-gnu')][string]$TargetTriple
)
$ErrorActionPreference = 'Stop'
try {
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
    $version = (Get-Content -LiteralPath "$RepoRoot\client\package.json" -Raw | ConvertFrom-Json).version
    $runtime = "$RepoRoot\client\src-tauri\runtime\windows"
    $bin = "$RepoRoot\client\src-tauri\binaries"
    $build = "$RepoRoot\build\windows-release"
    $licenses = "$RepoRoot\client\src-tauri\licenses"
    New-Item -ItemType Directory -Force -Path $runtime,$build,$licenses | Out-Null
    Copy-Item -Path "$RepoRoot\src\platform\win\licenses\*.txt" -Destination $licenses -Force
    $compiler = Get-Command g++.exe -ErrorAction SilentlyContinue
    if ($compiler) { $compiler = $compiler.Source }
    else {
        foreach ($directory in @($env:IDLETOKEN_MINGW_BIN, "$RepoRoot\mingw64\bin")) {
            if ($directory -and (Test-Path -LiteralPath "$directory\g++.exe")) { $compiler = "$directory\g++.exe"; break }
        }
    }
    if (-not $compiler) { throw 'MinGW g++ is required to build the native installer helper.' }
    $windres = Join-Path (Split-Path -Parent $compiler) 'windres.exe'
    Push-Location $RepoRoot
    try {
        & $windres -I src/platform/win src/platform/win/idletoken_utf8.rc -O coff -o "$build\helper-manifest.o"
        if ($LASTEXITCODE -ne 0) { throw 'Installer helper manifest compilation failed.' }
        & $compiler -std=c++17 -O2 -march=x86-64 -municode -static -fstack-protector-strong `
            src/platform/win/installer_helper.cpp "$build\helper-manifest.o" `
            '-Wl,--dynamicbase,--nxcompat,--high-entropy-va' -lrstrtmgr -lole32 -loleaut32 -luuid `
            -o "$runtime\idletoken-installer-helper.exe"
        if ($LASTEXITCODE -ne 0) { throw 'Native installer helper compilation failed.' }
    } finally { Pop-Location }
    $components = [ordered]@{
        'idletoken-coord' = 'IdleToken Inference Coordinator'
        'idletoken-worker' = 'IdleToken LAN Compute Supervisor'
        'idletoken-platform-agent' = 'IdleToken Service Sharing Agent'
        'idletoken-server' = 'IdleToken Local Inference Engine (llama.cpp)'
        'idletoken-rpc-server' = 'IdleToken LAN Compute Service (llama.cpp)'
        'idletoken-installer-helper' = 'IdleToken Installation Maintenance'
    }
    $provider = $env:IDLETOKEN_WINDOWS_SIGNING
    if ($provider -and $provider -notin @('none','artifact','certificate')) { throw 'Unknown Windows signing provider.' }
    foreach ($name in $components.Keys) {
        $source = if ($name -eq 'idletoken-installer-helper') { "$runtime\$name.exe" } else { "$bin\$name-$TargetTriple.exe" }
        if (-not (Test-Path -LiteralPath $source)) { throw "Missing staged component: $name" }
        & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PSScriptRoot\windows_release_identity.ps1" `
            -Path $source -Version $version -Description $components[$name] -OriginalFilename "$name.exe"
        if ($LASTEXITCODE -ne 0) { throw "Version resource preparation failed: $name" }
        if ($provider -in @('artifact','certificate')) {
            & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PSScriptRoot\windows_sign.ps1" -Path $source
            if ($LASTEXITCODE -ne 0) { throw "Signing failed: $name" }
        }
        if ($name -ne 'idletoken-installer-helper') {
            foreach ($triple in @('x86_64-pc-windows-msvc','x86_64-pc-windows-gnu')) {
                $destination = "$bin\$name-$triple.exe"
                if ($destination -ne $source) { Copy-Item -LiteralPath $source -Destination $destination -Force }
            }
        }
    }
    # Hash the final resource-stamped, optionally signed bytes, not build inputs.
    foreach ($name in @('idletoken-server','idletoken-rpc-server')) {
        $hash = (Get-FileHash -LiteralPath "$bin\$name-$TargetTriple.exe" -Algorithm SHA256).Hash.ToLower()
        [IO.File]::WriteAllText("$runtime\$name.exe.sha256", "$hash  $name.exe`n", [Text.Encoding]::ASCII)
    }
    $config = @{build=@{beforeBuildCommand=''}}
    if ($provider -in @('artifact','certificate')) {
        $config.bundle = @{windows=@{signCommand=@{cmd='powershell.exe';args=@(
            '-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',"$PSScriptRoot\windows_sign.ps1",'%1'
        )}}}
    } else { Write-Output 'WINDOWS_SIGNING: unsigned build; release signing status must remain false.' }
    if ($env:IDLETOKEN_WINDOWS_FAST_PACKAGE -eq '1') {
        if (-not $config.bundle) { $config.bundle = @{windows=@{}} }
        $config.bundle.windows.nsis = @{compression='zlib'}
        Write-Output 'WINDOWS_PACKAGE: fast validation package, not a final LZMA release.'
    }
    [IO.File]::WriteAllText("$build\tauri-config.json", ($config | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
    Write-Output 'WINDOWS_RELEASE_PREPARE_OK'
} catch {
    Write-Error $_
    exit 1
}
