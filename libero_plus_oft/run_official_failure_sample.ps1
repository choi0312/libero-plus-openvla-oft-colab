param(
    [int]$TaskId = 269,
    [int]$MaxSteps = 220
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_colab_a100.ps1"
$runId = "official-init-d5-$TaskId-" + (Get-Date -Format "yyyyMMdd-HHmmss")

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
    -UseBenchmarkInstruction `
    -TaskId $TaskId `
    -MaxSteps $MaxSteps `
    -RunId $runId

exit $LASTEXITCODE
