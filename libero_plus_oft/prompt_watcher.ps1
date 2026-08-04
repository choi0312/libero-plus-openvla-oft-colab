param(
    [string]$PromptFile = (Join-Path $PSScriptRoot "prompt.txt")
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_colab_a100.ps1"
$watcherStatus = Join-Path $PSScriptRoot "watcher_status.json"
$watcherPid = Join-Path $PSScriptRoot "watcher.pid"
$runnerLog = Join-Path $PSScriptRoot "a100_runner.log"

function Write-WatcherStatus {
    param([string]$State, [string]$Message, [string]$Prompt = "", [Nullable[int]]$ExitCode = $null)
    $payload = [ordered]@{
        state = $State
        message = $Message
        prompt = $Prompt
        process_id = $PID
        runner_exit_code = $ExitCode
        updated_at = [DateTimeOffset]::Now.ToString("o")
    }
    $temporary = "$watcherStatus.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 3),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $watcherStatus -Force
}

function Get-PromptSignature {
    if (-not (Test-Path -LiteralPath $PromptFile)) { return "missing" }
    $item = Get-Item -LiteralPath $PromptFile
    return "$($item.LastWriteTimeUtc.Ticks):$($item.Length)"
}

[IO.File]::WriteAllText($watcherPid, "$PID", [Text.Encoding]::ASCII)
$lastSignature = Get-PromptSignature
Write-WatcherStatus "idle" "Ready. Save prompt.txt in VS Code to run exactly one A100 episode."

try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        $signature = Get-PromptSignature
        if ($signature -eq "missing" -or $signature -eq $lastSignature) { continue }
        $lastSignature = $signature

        $prompt = [IO.File]::ReadAllText((Resolve-Path $PromptFile), [Text.Encoding]::UTF8).Trim()
        if ([string]::IsNullOrWhiteSpace($prompt)) {
            Write-WatcherStatus "invalid_prompt" "prompt.txt is empty; no episode was started."
            continue
        }

        Write-WatcherStatus "running" "One Colab A100 episode is running." $prompt
        # WSL and LIBERO may write harmless warnings to stderr even when the episode
        # succeeds. PowerShell converts native stderr into ErrorRecord objects, so
        # temporarily keep them non-terminating and use the real process exit code.
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -PromptFile $PromptFile *>> $runnerLog
            $code = $LASTEXITCODE
        }
        catch {
            $code = 1
            $_ | Out-String | Add-Content -LiteralPath $runnerLog -Encoding UTF8
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($code -eq 0) {
            Write-WatcherStatus "idle" "Episode finished. Save prompt.txt again to run another single episode." $prompt $code
        }
        else {
            Write-WatcherStatus "runner_error" "Episode runner exited with code $code." $prompt $code
        }
    }
}
finally {
    if (Test-Path -LiteralPath $watcherPid) {
        Remove-Item -LiteralPath $watcherPid -Force
    }
}
