$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$workspaceWsl = "/mnt/$drive" + $workspace.Substring(2).Replace('\', '/')
& wsl.exe -d Ubuntu-22.04 -- bash -lc "chmod +x '$workspaceWsl/libero_plus_oft/setup_assets.sh' && '$workspaceWsl/libero_plus_oft/setup_assets.sh'"
exit $LASTEXITCODE
