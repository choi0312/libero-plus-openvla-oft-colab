$ErrorActionPreference = "Stop"
$pidPath = Join-Path $PSScriptRoot "watcher.pid"
if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host "A100 prompt watcher is not running." -ForegroundColor Yellow
    exit 0
}
$watcherPid = [int]([IO.File]::ReadAllText($pidPath).Trim())
$process = Get-Process -Id $watcherPid -ErrorAction SilentlyContinue
if ($null -ne $process) {
    Stop-Process -Id $watcherPid
    Wait-Process -Id $watcherPid -Timeout 5 -ErrorAction SilentlyContinue
    Write-Host "A100 prompt watcher stopped (PID $watcherPid)." -ForegroundColor Cyan
}
else {
    Write-Host "The recorded watcher process no longer exists." -ForegroundColor Yellow
}
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
$statusPath = Join-Path $PSScriptRoot "watcher_status.json"
$status = [ordered]@{
    state = "stopped"
    message = "The browser dashboard owns prompt execution."
    prompt = ""
    process_id = $null
    runner_exit_code = $null
    updated_at = [DateTimeOffset]::Now.ToString("o")
}
[IO.File]::WriteAllText($statusPath, ($status | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
