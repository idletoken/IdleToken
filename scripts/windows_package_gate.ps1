param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReportPath = '',
    [switch]$RequireSignature
)
$ErrorActionPreference = 'Stop'
$payload = Join-Path $env:TEMP ('IdleToken-package-test-' + [guid]::NewGuid().ToString('N'))
$oldPath = $env:PATH
try {
    $Installer = (Get-Item -LiteralPath $Installer).FullName
    $version = (Get-Content "$RepoRoot\client\package.json" -Raw | ConvertFrom-Json).version
    $signed = $RequireSignature -or $env:IDLETOKEN_WINDOWS_SIGNING -in @('artifact','certificate')
    if ($signed -and -not $env:IDLETOKEN_WINDOWS_SIGNER_SUBJECT) { throw 'The expected signing subject is required.' }
    if ($signed) {
        $signature = Get-AuthenticodeSignature -LiteralPath $Installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $env:IDLETOKEN_WINDOWS_SIGNER_SUBJECT -or
            -not $signature.TimeStamperCertificate) { throw 'The installer is not signed by the expected publisher with a valid timestamp.' }
    }
    $seven = Get-Command 7z.exe -ErrorAction SilentlyContinue
    $seven = if ($seven) { $seven.Source } else { "$env:ProgramFiles\7-Zip\7z.exe" }
    if (-not (Test-Path -LiteralPath $seven)) { throw '7-Zip is required to inspect the final NSIS installer.' }
    New-Item -ItemType Directory -Path $payload | Out-Null
    & $seven x -y "-o$payload" $Installer | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Final installer extraction failed.' }
    $apps = @(Get-ChildItem -LiteralPath $payload -Filter idletoken-client.exe -Recurse)
    if ($apps.Count -ne 1) { throw 'Expected exactly one packaged desktop client.' }
    $app = $apps[0].Directory.FullName
    $names = @('idletoken-client','idletoken-coord','idletoken-worker',
        'idletoken-platform-agent','idletoken-server','idletoken-rpc-server','idletoken-installer-helper')
    $files = @()
    foreach ($name in $names) {
        $file = Get-Item -LiteralPath "$app\$name.exe"
        $info = $file.VersionInfo
        if ($info.ProductName -ne 'IdleToken' -or $info.CompanyName -ne 'IdleToken' -or
            -not $info.FileDescription -or $info.FileVersion -notmatch ('^' + [regex]::Escape($version) + '(\.0)?$')) {
            throw "Missing or stale executable identity: $name"
        }
        $signature = Get-AuthenticodeSignature -LiteralPath $file.FullName
        if ($signed -and ($signature.Status -ne 'Valid' -or
            $signature.SignerCertificate.Subject -cne $env:IDLETOKEN_WINDOWS_SIGNER_SUBJECT -or
            -not $signature.TimeStamperCertificate)) { throw "Missing or unexpected Authenticode signature: $name" }
        $files += [ordered]@{name=$file.Name; version=$info.FileVersion; description=$info.FileDescription;
            signature="$($signature.Status)"; sha256=(Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLower()}
    }
    # The outer installer and generated uninstaller belong to the same publisher.
    if ($signed) {
        $uninstallers = @(Get-ChildItem $payload -Recurse -Filter '*uninstall*.exe' | Select-Object -ExpandProperty FullName)
        if ($uninstallers.Count -eq 0) { throw 'The signed uninstaller could not be extracted for verification.' }
        $extra = @($Installer) + $uninstallers
        foreach ($file in $extra) {
            $signature = Get-AuthenticodeSignature -LiteralPath $file
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $env:IDLETOKEN_WINDOWS_SIGNER_SUBJECT -or
                -not $signature.TimeStamperCertificate) { throw 'Installer/uninstaller signature verification failed.' }
        }
    }
    foreach ($name in @('idletoken-server','idletoken-rpc-server')) {
        $expected = (Get-Content -LiteralPath "$app\$name.exe.sha256" -Raw).Trim().Split(' ')[0]
        if ($expected -notmatch '^[0-9a-f]{64}$' -or (Get-FileHash "$app\$name.exe" -Algorithm SHA256).Hash -ne $expected) {
            throw "Packaged engine digest mismatch: $name"
        }
    }
    foreach ($name in @('cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll','WebView2Loader.dll','vcomp140.dll')) {
        $actual = Get-FileHash -LiteralPath "$app\$name" -Algorithm SHA256
        $expected = Get-FileHash -LiteralPath "$RepoRoot\client\src-tauri\runtime\windows\$name" -Algorithm SHA256
        if ($actual.Hash -ne $expected.Hash) { throw "Packaged runtime differs from staging: $name" }
    }
    foreach ($name in @('LICENSE.txt','NOTICE.txt','ds4-MIT.txt','rax-BSD-3-Clause.txt','llamacpp-MIT.txt','NVIDIA-CUDA-EULA.txt',
        'gcc-14.2.0-GPL-3.0.txt','gcc-runtime-exception-3.1.txt','mingw-w64-runtime.txt','mingw-w64-headers.txt','winpthreads.txt')) {
        if ((Get-Item -LiteralPath "$app\licenses\$name").Length -eq 0) { throw "Missing license: $name" }
    }
    if (@(Get-ChildItem $payload -Recurse -File | Where-Object { $_.Name -match '(\.sig$|\.nsis\.zip$|latest\.json$|^ds4.*\.dll$)' }).Count) {
        throw 'A retired updater or ds4 artifact is in the installer.'
    }
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $mt = Get-ChildItem "$kits\*\x64\mt.exe" | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
    if (-not $mt) { throw 'Windows SDK mt.exe is required to verify preserved manifests.' }
    foreach ($name in $names | Where-Object { $_ -ne 'idletoken-client' }) {
        & $mt -nologo "-inputresource:$app\$name.exe;#1" "-out:$payload\manifest.xml" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Manifest extraction failed: $name" }
        $xml = Get-Content "$payload\manifest.xml" -Raw
        if ($xml -notmatch '<activeCodePage[^>]*>UTF-8</activeCodePage>' -or $xml -notmatch '<longPathAware[^>]*>true</longPathAware>') {
            throw "Native UTF-8 manifest was lost: $name"
        }
    }
    # Execute only version/help probes with packaged DLLs and system DLLs. No
    # Toolkit, build directory, model, port listener or serving process is used.
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($probe in @(@('idletoken-server','--version'),@('idletoken-rpc-server','--help'))) {
        # Start-Process -PassThru can lose ExitCode for a short-lived process
        # under Windows PowerShell 5.1. Own the Process handle from creation.
        $process = New-Object Diagnostics.Process
        $process.StartInfo.FileName = "$app\$($probe[0]).exe"
        $process.StartInfo.Arguments = $probe[1]
        $process.StartInfo.WorkingDirectory = $app
        $process.StartInfo.UseShellExecute = $false
        $process.StartInfo.CreateNoWindow = $true
        $process.StartInfo.RedirectStandardOutput = $true
        $process.StartInfo.RedirectStandardError = $true
        try {
            if (-not $process.Start()) { throw 'Could not start the packaged engine probe.' }
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(30000)) { $process.Kill(); $process.WaitForExit(); throw 'Packaged engine probe timed out.' }
            if ($process.ExitCode -ne 0) { throw "Packaged engine probe failed: $($probe[0]) ($($process.ExitCode))" }
            $body = $stdout.Result + $stderr.Result
            $expected = if ($probe[0] -eq 'idletoken-server') { 'version:|build:' } else { 'usage:' }
            if ($body -notmatch $expected) { throw 'The engine probe did not identify itself.' }
        } finally { $process.Dispose() }
    }
    $report = [ordered]@{version=$version; installerSha256=(Get-FileHash $Installer -Algorithm SHA256).Hash.ToLower();
        signaturesRequired=[bool]$signed; files=$files; engineDigests='pass'; runtimes='pass'; licenses='pass';
        nativeManifests='pass'; packagedEngineExecution='pass'}
    if ($ReportPath) { [IO.File]::WriteAllText($ReportPath, ($report | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false))) }
    Write-Output 'WINDOWS_PACKAGE_GATE_OK'
} finally {
    $env:PATH = $oldPath
    Remove-Item -LiteralPath $payload -Recurse -Force -ErrorAction SilentlyContinue
}
