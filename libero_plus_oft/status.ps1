$statusPath = Join-Path $PSScriptRoot "status.json"
$watcherStatusPath = Join-Path $PSScriptRoot "watcher_status.json"

while ($true) {
    Clear-Host
    Write-Host "LIBERO-Plus / OpenVLA-OFT / Colab A100" -ForegroundColor Cyan
    Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -ForegroundColor DarkGray
    Write-Host

    if (Test-Path -LiteralPath $watcherStatusPath) {
        try {
            $watcher = [IO.File]::ReadAllText($watcherStatusPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
            Write-Host "PROMPT WATCHER" -ForegroundColor Green
            [pscustomobject]@{
                State = $watcher.state
                Message = $watcher.message
                Prompt = $watcher.prompt
                RunnerExitCode = $watcher.runner_exit_code
                Updated = $watcher.updated_at
            } | Format-List
        }
        catch {
            Write-Host "Watcher status is being updated..." -ForegroundColor Yellow
        }
    }

    Write-Host "CURRENT / LAST EPISODE" -ForegroundColor Green
    if (Test-Path -LiteralPath $statusPath) {
        try {
            $s = [IO.File]::ReadAllText($statusPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
            [pscustomobject]@{
                State = $s.state
                Instruction = $s.instruction
                Step = $s.step
                Action = if ($null -eq $s.action) { "-" } else { "[" + (($s.action | ForEach-Object { "{0:N3}" -f $_ }) -join ", ") + "]" }
                Success = $s.success
                Backend = $s.backend
                RemoteGPU = $s.remote_device
                Precision = $s.runtime_precision
                SourceCommit = $s.remote_source_commit
                RemoteVRAM = if ($null -eq $s.remote_allocated_memory_gib) { "-" } else { "$($s.remote_allocated_memory_gib) / $($s.remote_memory_gib) GiB allocated" }
                A100Latency = if ($null -eq $s.remote_roundtrip_ms) { "-" } else { "{0:N1} ms" -f $s.remote_roundtrip_ms }
                InputOrder = ($s.input_order -join " -> ")
                Updated = $s.updated_at
                Output = $s.output_dir
            } | Format-List
        }
        catch {
            Write-Host "Episode status is being updated..." -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Waiting for the first episode..." -ForegroundColor Yellow
    }

    Write-Host "Ctrl+C: close this monitor (the hidden watcher keeps running)." -ForegroundColor DarkGray
    Start-Sleep -Milliseconds 750
}
