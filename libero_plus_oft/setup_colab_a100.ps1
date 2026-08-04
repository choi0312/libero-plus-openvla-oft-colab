$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$workspaceWsl = "/mnt/$drive" + $workspace.Substring(2).Replace('\', '/')
$wslHome = (& wsl.exe -d Ubuntu-22.04 -- bash -lc 'printf %s "$HOME"').Trim()
if ([string]::IsNullOrWhiteSpace($wslHome)) { throw "Could not resolve the WSL home directory." }
$colab = "$wslHome/.local/bin/colab"
$session = "openvla"
$gpuProbe = "$workspaceWsl/colab/probe_gpu.py"
$modelProbe = "$workspaceWsl/colab/probe_openvla_oft.py"
$installer = "$workspaceWsl/colab/install_openvla_oft.py"
$initializer = "$workspaceWsl/colab/init_openvla_oft.py"

function Invoke-ColabCommand {
    param([Parameter(Mandatory = $true)][string[]]$ColabArguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& wsl.exe -d Ubuntu-22.04 -- $colab @ColabArguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object { [string]$_ })
    }
}

function Write-ColabOutput {
    param([Parameter(Mandatory = $true)]$Result)
    foreach ($line in $Result.Output) { Write-Host $line }
}

function Invoke-ColabMarkerScript {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][string]$Marker,
        [Parameter(Mandatory = $true)][string]$ReadyProperty,
        [Parameter(Mandatory = $true)][string]$ExpectedSchema,
        [Parameter(Mandatory = $true)][string]$SchemaProperty
    )

    $result = Invoke-ColabCommand -ColabArguments @(
        "exec", "-s", $session, "--timeout", "$TimeoutSeconds", "-f", $Script
    )
    Write-ColabOutput -Result $result
    if ($result.ExitCode -ne 0) { return $false }

    $markerLines = @($result.Output | Where-Object { $_.Trim().StartsWith($Marker) })
    if ($markerLines.Count -ne 1) {
        Write-Host "Expected exactly one '$Marker' marker; received $($markerLines.Count)." -ForegroundColor Red
        return $false
    }

    try {
        $markerLine = $markerLines[0].Trim()
        $payload = $markerLine.Substring($Marker.Length) | ConvertFrom-Json
        $readyValue = $payload.$ReadyProperty
        $schemaValue = $payload.$SchemaProperty
        return (
            $readyValue -is [bool] -and
            $readyValue -eq $true -and
            $schemaValue -is [string] -and
            $schemaValue -eq $ExpectedSchema
        )
    }
    catch {
        Write-Host "Could not parse Colab marker '$Marker': $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-A100Kernel {
    $ready = Invoke-ColabMarkerScript `
        -Script $gpuProbe `
        -TimeoutSeconds 60 `
        -Marker "__VLA_GPU__" `
        -ReadyProperty "a100" `
        -ExpectedSchema "vla-gpu-probe/v1" `
        -SchemaProperty "schema"
    return $ready
}

function Test-OfficialModel {
    $ready = Invoke-ColabMarkerScript `
        -Script $modelProbe `
        -TimeoutSeconds 60 `
        -Marker "__OFT_PROBE__" `
        -ReadyProperty "probe_ready" `
        -ExpectedSchema "openvla-oft-probe/v1" `
        -SchemaProperty "probe_schema"
    return $ready
}

$sessionMutex = [Threading.Mutex]::new($false, "Local\LIBEROPlusA100Episode")
$ownsMutex = $false
$setupSucceeded = $false
try {
    $ownsMutex = $sessionMutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw "An episode or another A100 setup operation is already running."
    }

    Write-Host "[1/5] Check the Colab session, A100 kernel, and resident verified model" -ForegroundColor Cyan
    $status = Invoke-ColabCommand -ColabArguments @("status", "-s", $session)
    Write-ColabOutput -Result $status
    $gpuReady = $false
    $modelReady = $false
    if ($status.ExitCode -eq 0) {
        $gpuReady = Test-A100Kernel
        if ($gpuReady) { $modelReady = Test-OfficialModel }
    }

    if ($gpuReady -and $modelReady) {
        Write-Host "The NVIDIA A100 and verified official OpenVLA-OFT BF16 policy are already ready." -ForegroundColor Green
        $setupSucceeded = $true
    }
    else {
        Write-Host "[2/5] Recreate a clean Colab A100 kernel" -ForegroundColor Cyan
        $stopResult = Invoke-ColabCommand -ColabArguments @("stop", "-s", $session)
        Write-ColabOutput -Result $stopResult
        $newResult = Invoke-ColabCommand -ColabArguments @("new", "-s", $session, "--gpu", "A100")
        Write-ColabOutput -Result $newResult
        if ($newResult.ExitCode -ne 0) { throw "Could not allocate the Colab A100 session." }

        Write-Host "[3/5] Actively verify the new NVIDIA A100 kernel" -ForegroundColor Cyan
        $gpuReady = $false
        for ($attempt = 1; $attempt -le 5 -and -not $gpuReady; $attempt++) {
            $gpuReady = Test-A100Kernel
            if (-not $gpuReady -and $attempt -lt 5) { Start-Sleep -Seconds 5 }
        }
        if (-not $gpuReady) { throw "The new Colab runtime did not pass the NVIDIA A100 probe." }

        Write-Host "[4/5] Install and verify the pinned OpenVLA-OFT runtime" -ForegroundColor Cyan
        $installReady = Invoke-ColabMarkerScript `
            -Script $installer `
            -TimeoutSeconds 1500 `
            -Marker "__OFT_INSTALL_READY__" `
            -ReadyProperty "ready" `
            -ExpectedSchema "openvla-oft-install/v1" `
            -SchemaProperty "schema"
        if (-not $installReady) { throw "The pinned OpenVLA-OFT package installation did not verify." }

        Write-Host "[5/5] Load and independently verify the official Spatial BF16 checkpoint" -ForegroundColor Cyan
        $modelLoaded = Invoke-ColabMarkerScript `
            -Script $initializer `
            -TimeoutSeconds 1500 `
            -Marker "__OFT_READY__" `
            -ReadyProperty "ready" `
            -ExpectedSchema "openvla-oft-runtime/v1" `
            -SchemaProperty "schema"
        if (-not $modelLoaded) { throw "The official Spatial BF16 checkpoint did not verify." }
        if (-not (Test-OfficialModel)) { throw "The independent post-load OpenVLA-OFT probe failed." }
        $setupSucceeded = $true
    }
}
catch {
    Write-Host "A100 setup failed: $($_.Exception.Message)" -ForegroundColor Red
    $setupSucceeded = $false
}
finally {
    if ($ownsMutex) { $sessionMutex.ReleaseMutex() }
    $sessionMutex.Dispose()
}

if ($setupSucceeded) { exit 0 }
exit 1
