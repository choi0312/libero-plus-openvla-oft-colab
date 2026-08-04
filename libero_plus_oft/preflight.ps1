$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$workspaceWsl = "/mnt/$drive" + $workspace.Substring(2).Replace('\', '/')
& wsl.exe -d Ubuntu-22.04 -- bash -lc "cd '$workspaceWsl/third_party/openvla-oft' && MUJOCO_GL=egl \"`$HOME/.venvs/libero-oft/bin/python\" '$workspaceWsl/libero_plus_oft/verify_install.py'"
exit $LASTEXITCODE
