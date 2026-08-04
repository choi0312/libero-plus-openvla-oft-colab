param([int]$IdleMinutes = 30)

$ErrorActionPreference = "Stop"
$pidPath = Join-Path $PSScriptRoot "idle_gpu_guard.pid"
$guard = Join-Path $PSScriptRoot "idle_gpu_guard.ps1"
$statusPath = Join-Path $PSScriptRoot "idle_gpu_guard_status.json"

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int]([IO.File]::ReadAllText($pidPath).Trim())
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid"
    if ($existing -and $existing.CommandLine -match 'idle_gpu_guard\.ps1') {
        Write-Host "A100 idle guard is already running (PID $existingPid)." -ForegroundColor Yellow
        if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath }
        exit 0
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}

$process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $guard,
    "-IdleMinutes", "$IdleMinutes"
) -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2
if ($process.HasExited) { throw "A100 idle guard exited during startup." }
Write-Host "A100 idle guard started (PID $($process.Id)); auto-stop after $IdleMinutes idle minutes." -ForegroundColor Cyan
if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath }
