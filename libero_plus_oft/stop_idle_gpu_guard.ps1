$ErrorActionPreference = "Stop"
$pidPath = Join-Path $PSScriptRoot "idle_gpu_guard.pid"

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host "A100 idle guard is not running." -ForegroundColor Yellow
    exit 0
}

$guardPid = [int]([IO.File]::ReadAllText($pidPath).Trim())
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $guardPid"
if ($process -and $process.CommandLine -notmatch 'idle_gpu_guard\.ps1') {
    throw "PID $guardPid is not the A100 idle guard; refusing to stop it."
}
if ($process) {
    Stop-Process -Id $guardPid -Force
    Wait-Process -Id $guardPid -Timeout 5 -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
Write-Host "A100 idle guard stopped. The Colab session was left unchanged." -ForegroundColor Cyan
