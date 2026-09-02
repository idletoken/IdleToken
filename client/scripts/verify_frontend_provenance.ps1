param(
    [Parameter(Mandatory = $true)]
    [string]$ClientRoot
)

$ErrorActionPreference = "Stop"
$client = [IO.Path]::GetFullPath($ClientRoot)
$clientUri = [Uri]($client.TrimEnd("\") + "\")
function Get-ClientRelativePath([string]$Path) {
    $uri = [Uri]([IO.Path]::GetFullPath($Path))
    return [Uri]::UnescapeDataString($clientUri.MakeRelativeUri($uri).ToString())
}
$manifestPath = Join-Path $client "dist\frontend-provenance.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing dist/frontend-provenance.json; run pnpm build:release and sync client/dist"
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.schema -ne 1 -or $manifest.mode -ne "release") {
    throw "Frontend provenance is not a release build"
}
$package = Get-Content -Raw -LiteralPath (Join-Path $client "package.json") | ConvertFrom-Json
if ($manifest.packageVersion -ne $package.version) {
    throw "Frontend version $($manifest.packageVersion) does not match package version $($package.version)"
}

$expected = @($manifest.files.PSObject.Properties.Name | Sort-Object)
$actual = @()
foreach ($dir in @("src", "public", "..\packages\shared-ui\src")) {
    $root = [IO.Path]::GetFullPath((Join-Path $client $dir))
    if (Test-Path -LiteralPath $root -PathType Container) {
        $actual += Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
            Get-ClientRelativePath $_.FullName
        }
    }
}
foreach ($file in @(
    "index.html", "package.json", "pnpm-lock.yaml", "tsconfig.json",
    "tsconfig.node.json", "vite.config.ts", ".env.release",
    "..\packages\shared-ui\package.json"
)) {
    $path = [IO.Path]::GetFullPath((Join-Path $client $file))
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $actual += Get-ClientRelativePath $path
    }
}
$actual = @($actual | Sort-Object -Unique)
$fileListDiff = @(Compare-Object $expected $actual)
if ($fileListDiff.Count -ne 0) {
    foreach ($difference in $fileListDiff) {
        Write-Output "Frontend file-list mismatch ($($difference.SideIndicator)): $($difference.InputObject)"
    }
    throw "Frontend source file list changed after client/dist was built"
}

foreach ($entry in $manifest.files.PSObject.Properties) {
    $path = [IO.Path]::GetFullPath((Join-Path $client $entry.Name))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Frontend source is missing: $($entry.Name)"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actualHash -ne [string]$entry.Value) {
        throw "Stale frontend dist: $($entry.Name) changed after pnpm build:release"
    }
}

Write-Output "FRONTEND_DIST_PROVENANCE_OK $($expected.Count) files"
