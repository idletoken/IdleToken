param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference = 'Stop'
try {
    $file = (Get-Item -LiteralPath $Path).FullName
    $provider = $env:IDLETOKEN_WINDOWS_SIGNING
    if ($provider -notin @('artifact', 'certificate')) {
        throw 'Signing requires IDLETOKEN_WINDOWS_SIGNING=artifact or certificate. Refusing an unsigned fallback.'
    }
    $subject = $env:IDLETOKEN_WINDOWS_SIGNER_SUBJECT
    if ([string]::IsNullOrWhiteSpace($subject)) { throw 'Set IDLETOKEN_WINDOWS_SIGNER_SUBJECT to the exact validated certificate subject.' }
    $signtool = $env:IDLETOKEN_SIGNTOOL
    if (-not $signtool) {
        $tool = Get-Command signtool.exe -ErrorAction SilentlyContinue
        if ($tool) { $signtool = $tool.Source }
        else {
            $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
            $signtool = Get-ChildItem "$kits\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
        }
    }
    if (-not $signtool -or -not (Test-Path -LiteralPath $signtool)) { throw 'Windows SDK SignTool was not found.' }
    $existing = Get-AuthenticodeSignature -LiteralPath $file
    if ($existing.Status -eq 'Valid' -and $existing.SignerCertificate.Subject -ceq $subject -and $existing.TimeStamperCertificate) {
        if ($provider -eq 'certificate' -and $existing.SignerCertificate.Thumbprint -ne $env:IDLETOKEN_WINDOWS_CERT_THUMBPRINT) {
            throw 'Existing signature does not match the configured signing certificate.'
        }
        # Tauri also visits staged sidecars. Do not re-sign them after hashing.
        & $signtool verify /pa /all /tw $file
        if ($LASTEXITCODE -ne 0) { throw 'Existing signature failed SignTool verification.' }
        Write-Output "WINDOWS_SIGN_OK: $([IO.Path]::GetFileName($file)) (already signed)"
        exit 0
    }
    $arguments = @('sign', '/fd', 'SHA256', '/td', 'SHA256', '/d', 'IdleToken', '/du', 'https://idletoken.ai')
    if ($provider -eq 'artifact') {
        $metadata = $env:IDLETOKEN_ARTIFACT_SIGNING_METADATA
        $dlib = $env:IDLETOKEN_ARTIFACT_SIGNING_DLIB
        if (-not $metadata -or -not (Test-Path -LiteralPath $metadata) -or
            -not $dlib -or -not (Test-Path -LiteralPath $dlib)) {
            throw 'Artifact Signing requires IDLETOKEN_ARTIFACT_SIGNING_METADATA and IDLETOKEN_ARTIFACT_SIGNING_DLIB.'
        }
        $account = Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json
        if ($account.Endpoint -notmatch '^https://[a-z0-9]+\.codesigning\.azure\.net/?$' -or
            -not $account.CodeSigningAccountName -or -not $account.CertificateProfileName) {
            throw 'Invalid Artifact Signing metadata; use a Public Trust certificate profile.'
        }
        $arguments += @('/tr', 'http://timestamp.acs.microsoft.com', '/dlib', $dlib, '/dmdf', $metadata)
    } else {
        $thumbprint = $env:IDLETOKEN_WINDOWS_CERT_THUMBPRINT
        if ($thumbprint -notmatch '^[0-9a-fA-F]{40}$') { throw 'Set IDLETOKEN_WINDOWS_CERT_THUMBPRINT to a certificate in the Windows My store.' }
        $arguments += @('/tr', 'http://timestamp.digicert.com', '/sha1', $thumbprint, '/s', 'My')
        if ($env:IDLETOKEN_WINDOWS_CERT_STORE -eq 'LocalMachine') { $arguments += '/sm' }
        elseif ($env:IDLETOKEN_WINDOWS_CERT_STORE -and $env:IDLETOKEN_WINDOWS_CERT_STORE -ne 'CurrentUser') { throw 'Certificate store must be CurrentUser or LocalMachine.' }
    }
    & $signtool @arguments $file
    if ($LASTEXITCODE -ne 0) { throw 'SignTool signing failed; unsigned output must not be published as signed.' }
    & $signtool verify /pa /all /tw $file
    if ($LASTEXITCODE -ne 0) { throw 'SignTool signature verification failed.' }
    $signed = Get-AuthenticodeSignature -LiteralPath $file
    if ($signed.Status -ne 'Valid' -or $signed.SignerCertificate.Subject -cne $subject -or -not $signed.TimeStamperCertificate) {
        throw 'The signature is not valid, timestamped, and from the expected publisher.'
    }
    Write-Output "WINDOWS_SIGN_OK: $([IO.Path]::GetFileName($file))"
} catch {
    Write-Error $_
    exit 1
}
