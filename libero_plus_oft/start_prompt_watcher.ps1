param(
    [switch]$OpenFiles,
    [switch]$Monitor
)

$ErrorActionPreference = "Stop"
$pidPath = Join-Path $PSScriptRoot "watcher.pid"
$watcher = Join-Path $PSScriptRoot "prompt_watcher.ps1"

$watcherRunning = $false
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int]([IO.File]::ReadAllText($pidPath).Trim())
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        Write-Host "A100 prompt watcher is already running (PID $existingPid)." -ForegroundColor Yellow
        $watcherRunning = $true
    }
    else {
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not $watcherRunning) {
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $watcher
    ) -WindowStyle Hidden -PassThru
    Write-Host "A100 prompt watcher started (PID $($process.Id))." -ForegroundColor Cyan
    Write-Host "Edit and save libero_plus_oft/prompt.txt in VS Code." -ForegroundColor DarkGray
}

if ($OpenFiles) {
    $code = Get-Command code.cmd -ErrorAction SilentlyContinue
    if (-not $code) { $code = Get-Command code.exe -ErrorAction SilentlyContinue }
    if ($code) {
        & $code.Source --reuse-window `
            (Join-Path $PSScriptRoot "prompt.txt") `
            (Join-Path $PSScriptRoot "watcher_status.json") `
            (Join-Path $PSScriptRoot "status.json")
    }
    else {
        Write-Host "VS Code CLI was not found; open prompt.txt and status.json manually." -ForegroundColor Yellow
    }
}

if ($Monitor) {
    & (Join-Path $PSScriptRoot "status.ps1")
}
