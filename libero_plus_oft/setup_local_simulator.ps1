$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$workspaceWsl = "/mnt/$drive" + $workspace.Substring(2).Replace('\', '/')

Write-Host "[1/3] WSL simulation libraries" -ForegroundColor Cyan
& wsl.exe -d Ubuntu-22.04 -u root -- bash -lc "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y curl ffmpeg git libegl1 libexpat1 libfontconfig1-dev libgl1-mesa-dri libglib2.0-0 libmagickwand-dev libosmesa6 python3 python3-venv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] LIBERO / MuJoCo Python environment" -ForegroundColor Cyan
& wsl.exe -d Ubuntu-22.04 -- bash -lc "chmod +x '$workspaceWsl/libero_plus_oft/setup_wsl.sh' '$workspaceWsl/libero_plus_oft/setup_assets.sh' && '$workspaceWsl/libero_plus_oft/setup_wsl.sh'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/3] Official LIBERO-Plus assets (no local model download)" -ForegroundColor Cyan
& wsl.exe -d Ubuntu-22.04 -- bash -lc "'$workspaceWsl/libero_plus_oft/setup_assets.sh'"
exit $LASTEXITCODE
