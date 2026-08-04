param(
    [int]$IdleMinutes = 30,
    [int]$PollSeconds = 60,
    [string]$Session = "openvla"
)

$ErrorActionPreference = "Stop"
if ($IdleMinutes -lt 1) { throw "IdleMinutes must be at least 1." }
if ($PollSeconds -lt 10) { throw "PollSeconds must be at least 10." }

$pidPath = Join-Path $PSScriptRoot "idle_gpu_guard.pid"
$statusPath = Join-Path $PSScriptRoot "idle_gpu_guard_status.json"
$wslHome = (& wsl.exe -d Ubuntu-22.04 -- bash -lc 'printf %s "$HOME"').Trim()
if ([string]::IsNullOrWhiteSpace($wslHome)) { throw "Could not resolve the WSL home directory." }
$colab = "$wslHome/.local/bin/colab"
$idleSince = $null

function Write-GuardStatus {
    param(
        [string]$State,
        [string]$SessionState,
        [Nullable[datetime]]$IdleStartedAt = $null,
        [string]$Message = ""
    )
    $payload = [ordered]@{
        state = $State
        session = $Session
        session_state = $SessionState
        idle_limit_minutes = $IdleMinutes
        poll_seconds = $PollSeconds
        idle_started_at = if ($null -ne $IdleStartedAt) { ([datetime]$IdleStartedAt).ToString("o") } else { $null }
        updated_at = [DateTimeOffset]::Now.ToString("o")
        pid = $PID
        message = $Message
    }
    $temporary = "$statusPath.tmp"
    [IO.File]::WriteAllText($temporary, ($payload | ConvertTo-Json -Depth 3), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

[IO.File]::WriteAllText($pidPath, "$PID", [Text.UTF8Encoding]::new($false))
Write-GuardStatus -State "starting" -SessionState "unknown"

try {
    while ($true) {
        $statusOutput = @(& wsl.exe -d Ubuntu-22.04 -- $colab status -s $Session 2>&1)
        $statusExitCode = $LASTEXITCODE
        $statusText = $statusOutput -join "`n"
        if ($statusExitCode -ne 0) {
            Write-GuardStatus -State "session_not_running" -SessionState "absent" -Message $statusText
            break
        }

        $sessionBusy = $statusText -match 'Status:\s*(BUSY|STARTING|CONNECTING)'
        $activeLocal = @(Get-CimInstance Win32_Process | Where-Object {
            $_.ProcessId -ne $PID -and $_.CommandLine -and
            $_.CommandLine -match 'run_one_episode\.py|run_colab_a100\.ps1|setup_colab_a100\.ps1'
        }).Count -gt 0

        if ($sessionBusy -or $activeLocal) {
            $idleSince = $null
            Write-GuardStatus -State "active" -SessionState "busy" -Message "Idle timer reset because evaluation/setup activity was detected."
        }
        else {
            if ($null -eq $idleSince) { $idleSince = Get-Date }
            $idleElapsed = (Get-Date) - $idleSince
            if ($idleElapsed.TotalMinutes -ge $IdleMinutes) {
                Write-GuardStatus -State "stopping_idle_session" -SessionState "idle" -IdleStartedAt $idleSince
                $stopOutput = @(& wsl.exe -d Ubuntu-22.04 -- $colab stop -s $Session 2>&1)
                if ($LASTEXITCODE -ne 0) {
                    throw "Colab stop failed: $($stopOutput -join ' ')"
                }
                Write-GuardStatus -State "stopped_after_idle" -SessionState "stopped" -IdleStartedAt $idleSince -Message ($stopOutput -join "`n")
                break
            }
            Write-GuardStatus -State "monitoring_idle" -SessionState "idle" -IdleStartedAt $idleSince -Message ("Auto-stop in {0:N1} minutes if no activity starts." -f ($IdleMinutes - $idleElapsed.TotalMinutes))
        }
        Start-Sleep -Seconds $PollSeconds
    }
}
catch {
    Write-GuardStatus -State "guard_error" -SessionState "unknown" -IdleStartedAt $idleSince -Message $_.Exception.Message
    throw
}
finally {
    if (Test-Path -LiteralPath $pidPath) {
        $recordedPid = [IO.File]::ReadAllText($pidPath).Trim()
        if ($recordedPid -eq "$PID") { Remove-Item -LiteralPath $pidPath -Force }
    }
}
