param(
    [string]$Instruction = "",
    [string]$PromptFile = (Join-Path $PSScriptRoot "prompt.txt"),
    [string]$RunId = "",
    [string]$LiveDir = "",
    [string]$AbortFile = "",
    [int]$TaskId = 988,
    [int]$MaxSteps = 220,
    [switch]$UseBenchmarkInstruction,
    [switch]$NoLive
)

$ErrorActionPreference = "Stop"
$Instruction = $Instruction.Trim()
if (-not $UseBenchmarkInstruction -and [string]::IsNullOrWhiteSpace($Instruction)) {
    if (-not (Test-Path -LiteralPath $PromptFile)) {
        throw "Prompt file was not found: $PromptFile"
    }
    $Instruction = [IO.File]::ReadAllText((Resolve-Path $PromptFile), [Text.Encoding]::UTF8).Trim()
}
if (-not $UseBenchmarkInstruction -and [string]::IsNullOrWhiteSpace($Instruction)) {
    throw "Prompt must not be empty."
}
if (-not $UseBenchmarkInstruction -and $TaskId -ne 988) {
    throw "The verified LIBERO-Plus A100 prompt runner is fixed to task ID 988."
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$workspaceWsl = "/mnt/$drive" + $workspace.Substring(2).Replace('\', '/')

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $resolved = [IO.Path]::GetFullPath($WindowsPath)
    $pathDrive = $resolved.Substring(0, 1).ToLowerInvariant()
    return "/mnt/$pathDrive" + $resolved.Substring(2).Replace('\', '/')
}

$episodeMutex = [Threading.Mutex]::new($false, "Local\LIBEROPlusA100Episode")
$ownsMutex = $false
try {
    $ownsMutex = $episodeMutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw "Another LIBERO-Plus episode is already running."
    }

    if (-not [string]::IsNullOrWhiteSpace($AbortFile) -and (Test-Path -LiteralPath $AbortFile)) {
        Remove-Item -LiteralPath $AbortFile -Force
    }

$wslHome = (& wsl.exe -d Ubuntu-22.04 -- bash -lc 'printf %s "$HOME"').Trim()
if ([string]::IsNullOrWhiteSpace($wslHome)) { throw "Could not resolve the WSL home directory." }
$python = "$wslHome/.venvs/libero-oft/bin/python"
$runner = "$workspaceWsl/libero_plus_oft/run_one_episode.py"
$instructionBase64 = if ($UseBenchmarkInstruction) {
    ""
} else {
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Instruction))
}

$command = [ordered]@{
    backend = "colab_a100_bf16"
    checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    suite = "libero_spatial"
    task_id = $TaskId
    episode_id = 0
    episodes = 1
    max_control_steps = $MaxSteps
    prompt = if ($UseBenchmarkInstruction) { "<official task language>" } else { $Instruction }
    run_id = $RunId
    live_dir = $LiveDir
    input_order = @("external_rgb", "wrist_rgb", "proprioception_8d")
    submitted_at = [DateTimeOffset]::Now.ToString("o")
}
$commandPath = Join-Path $PSScriptRoot "command.json"
$temporaryPath = "$commandPath.tmp"
$json = $command | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText($temporaryPath, $json, [Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporaryPath -Destination $commandPath -Force

$arguments = @(
    "-d", "Ubuntu-22.04", "--", "env",
    "MUJOCO_GL=egl",
    "HF_HOME=$workspaceWsl/.runtime/huggingface",
    $python,
    $runner,
    "--backend", "colab",
    "--colab-session", "openvla",
    "--task-id", "$TaskId",
    "--episode-id", "0",
    "--max-control-steps", "$MaxSteps"
)
if (-not $UseBenchmarkInstruction) { $arguments += @("--instruction-base64", $instructionBase64) }
if ($NoLive) { $arguments += "--no-live-window" }
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $arguments += @("--run-id", $RunId) }
if (-not [string]::IsNullOrWhiteSpace($LiveDir)) { $arguments += @("--live-dir", (Convert-ToWslPath $LiveDir)) }
if (-not [string]::IsNullOrWhiteSpace($AbortFile)) { $arguments += @("--abort-file", (Convert-ToWslPath $AbortFile)) }

Write-Host "LIBERO-Plus / official OpenVLA-OFT / Colab A100" -ForegroundColor Cyan
if ($UseBenchmarkInstruction) {
    Write-Host "Prompt: official benchmark task language (loaded from task definition)"
} else {
    Write-Host "Prompt: $Instruction"
}
Write-Host "Exactly one episode; task ID $TaskId; local GPU fallback disabled." -ForegroundColor DarkGray
& wsl.exe @arguments
$runnerExitCode = $LASTEXITCODE
}
finally {
    if ($ownsMutex) {
        $episodeMutex.ReleaseMutex()
    }
    $episodeMutex.Dispose()
}
exit $runnerExitCode
