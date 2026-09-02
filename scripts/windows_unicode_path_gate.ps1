param(
    [string]$RepoRoot = "",
    [string]$SmokeGguf = ""
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$SmokeGguf = if ($SmokeGguf) { $SmokeGguf } else { $env:IDLETOKEN_UNICODE_SMOKE_GGUF }
$RepoRoot = (Resolve-Path $RepoRoot).Path

function Fail([string]$Message) {
    throw "WINDOWS_UNICODE_PATH_FAIL: $Message"
}

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "$Label is missing: $Path"
    }
}

function Find-Mt {
    $cmd = Get-Command mt.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (Test-Path -LiteralPath $kits) {
        $hit = Get-ChildItem -LiteralPath $kits -Filter mt.exe -Recurse -File |
            Where-Object { $_.FullName -match '\\x64\\mt\.exe$' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    Fail "Windows SDK mt.exe is required to inspect embedded manifests"
}

function Assert-Utf8Manifest([string]$Mt, [string]$Exe, [string]$TempRoot) {
    Require-File $Exe "native executable"
    $dump = Join-Path $TempRoot ((Split-Path $Exe -Leaf) + ".manifest.xml")
    & $Mt -nologo "-inputresource:$Exe;#1" "-out:$dump" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $dump)) {
        Fail "cannot extract the application manifest from $Exe"
    }
    $xml = Get-Content -LiteralPath $dump -Raw
    if ($xml -notmatch '<activeCodePage[^>]*>UTF-8</activeCodePage>') {
        Fail "UTF-8 activeCodePage is missing from $Exe"
    }
    if ($xml -notmatch '<longPathAware[^>]*>true</longPathAware>') {
        Fail "longPathAware is missing from $Exe"
    }
}

function Free-TcpPort {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()
    return $port
}

function Unicode-Name([int[]]$CodeUnits) {
    $builder = New-Object System.Text.StringBuilder
    foreach ($unit in $CodeUnits) {
        [void]$builder.Append([char]$unit)
    }
    return $builder.ToString()
}

$coord = Join-Path $RepoRoot "idletoken-coord.exe"
$worker = Join-Path $RepoRoot "idletoken-worker.exe"
$agent = Join-Path $RepoRoot "idletoken-platform-agent.exe"
$llamaBin = Join-Path $RepoRoot "vendor\llama.cpp\build\bin\Release"
if (-not (Test-Path -LiteralPath (Join-Path $llamaBin "llama-server.exe"))) {
    $llamaBin = Join-Path $RepoRoot "vendor\llama.cpp\build\bin"
}
$server = Join-Path $llamaBin "llama-server.exe"
$rpc = Join-Path $llamaBin "ggml-rpc-server.exe"
$perplexity = Join-Path $llamaBin "llama-perplexity.exe"
$runtimeDir = Join-Path $RepoRoot "client\src-tauri\runtime\windows"

foreach ($pair in @(
    @($coord, "coordinator"),
    @($worker, "worker"),
    @($agent, "platform agent"),
    @($server, "llama-server"),
    @($rpc, "ggml-rpc-server"),
    @($perplexity, "llama-perplexity")
)) {
    Require-File $pair[0] $pair[1]
}
foreach ($dll in @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "vcomp140.dll")) {
    Require-File (Join-Path $runtimeDir $dll) "packaged CUDA runtime"
}

# Keep this name short enough for Windows AF_UNIX's sockaddr path limit while
# still testing both a non-ASCII profile component and non-ASCII leaf names.
# Windows PowerShell 5.1 decodes a BOM-less .ps1 using the system ANSI code
# page. Keep this release gate byte-for-byte ASCII and construct the CJK test
# names from UTF-16 code units, otherwise the gate itself fails to parse on the
# exact non-English systems it is meant to certify.
$testRoot = Join-Path $env:TEMP ((Unicode-Name @(0x95F2)) + "-" + $PID)
$modelDir = Join-Path $testRoot (Unicode-Name @(0x6A21, 0x578B))
$cacheDir = Join-Path $testRoot (Unicode-Name @(0x7F13, 0x5B58))
$keyDir = Join-Path $testRoot (Unicode-Name @(0x5BC6, 0x94A5))
$agentProcess = $null
$oldUtf8Dir = $env:IDLETOKEN_UTF8_TEST_DIR
$oldProcessPath = $env:PATH

try {
    New-Item -ItemType Directory -Force -Path $modelDir, $cacheDir, $keyDir | Out-Null

    # The installer places these DLL resources beside the installed sidecars.
    # The source-tree smoke runs the freshly built engine in its build directory,
    # so prepend the exact packaged runtime rather than relying on a developer's
    # globally installed CUDA Toolkit.
    $env:PATH = $runtimeDir + ";" + $env:PATH

    $mt = Find-Mt
    foreach ($exe in @($coord, $worker, $agent, $server, $rpc, $perplexity)) {
        Assert-Utf8Manifest $mt $exe $testRoot
    }

    # This is the original failure path plus model-file APIs: the coordinator
    # creates, sizes and renames a Chinese GGUF leaf, then binds/connects an
    # AF_UNIX coord-api.sock below a Chinese directory.
    $env:IDLETOKEN_UTF8_TEST_DIR = $testRoot
    # These C binaries intentionally write diagnostics to stderr. Under
    # ErrorActionPreference=Stop, Windows PowerShell 5.1 wraps any native
    # stderr line as a terminating NativeCommandError even when the process
    # exits zero. Capture both streams, then judge the real exit code.
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $coordOut = (& $coord --selftest 2>&1 | Out-String)
    $coordExit = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($coordExit -ne 0) {
        Fail "coordinator selftest failed under a Chinese directory: $coordOut"
    }
    if ($coordOut -notmatch 'PASS Windows UTF-8 file \+ AF_UNIX path') {
        Fail "coordinator did not execute the real UTF-8 file/socket probe"
    }

    # Exercise the cache implementation itself (FindFirstFileA + DeleteFileA),
    # including Chinese directory and leaf names. A non-cache neighbour must
    # survive so a false 'delete everything' implementation cannot pass.
    $kv = Join-Path $cacheDir ((Unicode-Name @(0x6696, 0x542F, 0x52A8)) + ".kv")
    $kvTmp = Join-Path $cacheDir ((Unicode-Name @(0x672A, 0x5B8C, 0x6210)) + ".kv.tmp")
    $keep = Join-Path $cacheDir ((Unicode-Name @(0x4FDD, 0x7559)) + ".txt")
    [IO.File]::WriteAllText($kv, "cache")
    [IO.File]::WriteAllText($kvTmp, "partial")
    [IO.File]::WriteAllText($keep, "keep")
    $ErrorActionPreference = "Continue"
    $workerOut = (& $worker --kv-clear --kv-dir $cacheDir 2>&1 | Out-String)
    $workerExit = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($workerExit -ne 0 -or (Test-Path -LiteralPath $kv) -or
        (Test-Path -LiteralPath $kvTmp) -or -not (Test-Path -LiteralPath $keep)) {
        Fail "worker cache operations corrupted a Chinese path: $workerOut"
    }

    # The platform agent writes its persistent identity before entering the
    # listener. Starting it briefly therefore tests Rust -> native argv and the
    # agent's fopen/rename path without needing a platform account.
    $keyPath = Join-Path $keyDir ((Unicode-Name @(0x5E73, 0x53F0, 0x4EE3, 0x7406)) + ".key")
    $agentStdout = Join-Path $testRoot "agent.stdout.log"
    $agentStderr = Join-Path $testRoot "agent.stderr.log"
    $port = Free-TcpPort
    $agentProcess = Start-Process -FilePath $agent -ArgumentList @(
        "--port", "$port", "--key-file", ('"' + $keyPath + '"')
    ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $agentStdout `
      -RedirectStandardError $agentStderr
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    while ([DateTime]::UtcNow -lt $deadline -and
           -not (Test-Path -LiteralPath $keyPath) -and
           -not $agentProcess.HasExited) {
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $keyPath)) {
        $log = if (Test-Path -LiteralPath $agentStderr) {
            Get-Content -LiteralPath $agentStderr -Raw
        } else { "" }
        Fail "platform agent could not persist a key under a Chinese path: $log"
    }

    # Optional but mandatory for release certification when supplied: hardlink
    # a real smoke GGUF under a Chinese name and wait for llama-server health.
    # This verifies the upstream MSVC process opens the exact path, not merely
    # that its manifest looks right.
    if ($SmokeGguf) {
        $SmokeGguf = (Resolve-Path -LiteralPath $SmokeGguf).Path
        $unicodeGguf = Join-Path $modelDir ((Unicode-Name @(0x771F, 0x5B9E, 0x6A21, 0x578B)) + ".gguf")
        try {
            New-Item -ItemType HardLink -Path $unicodeGguf -Target $SmokeGguf | Out-Null
        } catch {
            Copy-Item -LiteralPath $SmokeGguf -Destination $unicodeGguf
        }
        $serverStdout = Join-Path $testRoot "server.stdout.log"
        $serverStderr = Join-Path $testRoot "server.stderr.log"
        $serverPort = Free-TcpPort
        $serverProcess = Start-Process -FilePath $server -ArgumentList @(
            "-m", ('"' + $unicodeGguf + '"'), "--host", "127.0.0.1",
            "--port", "$serverPort", "-ngl", "99"
        ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $serverStdout `
          -RedirectStandardError $serverStderr
        try {
            $ready = $false
            $deadline = [DateTime]::UtcNow.AddSeconds(120)
            while ([DateTime]::UtcNow -lt $deadline -and -not $serverProcess.HasExited) {
                try {
                    $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
                        -Uri "http://127.0.0.1:$serverPort/health"
                    if ($health.Content -match '"status"\s*:\s*"ok"') {
                        $ready = $true
                        break
                    }
                } catch { }
                Start-Sleep -Milliseconds 500
            }
            if (-not $ready) {
                $processState = if ($serverProcess.HasExited) {
                    "exit code=" + $serverProcess.ExitCode
                } else {
                    "still running after health deadline"
                }
                $log = "process " + $processState + "`n"
                if (Test-Path -LiteralPath $serverStderr) {
                    $log += Get-Content -LiteralPath $serverStderr -Raw
                }
                if (Test-Path -LiteralPath $serverStdout) {
                    $log += Get-Content -LiteralPath $serverStdout -Raw
                }
                Fail "llama-server did not load the real GGUF through its Chinese path: $log"
            }
        } finally {
            if (-not $serverProcess.HasExited) { Stop-Process -Id $serverProcess.Id -Force }
            $serverProcess.WaitForExit()
        }
    }

    Write-Output "WINDOWS_UNICODE_PATH_OK $testRoot"
} finally {
    if ($agentProcess -and -not $agentProcess.HasExited) {
        Stop-Process -Id $agentProcess.Id -Force -ErrorAction SilentlyContinue
        $agentProcess.WaitForExit()
    }
    if ($null -eq $oldUtf8Dir) {
        Remove-Item Env:IDLETOKEN_UTF8_TEST_DIR -ErrorAction SilentlyContinue
    } else {
        $env:IDLETOKEN_UTF8_TEST_DIR = $oldUtf8Dir
    }
    $env:PATH = $oldProcessPath
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
