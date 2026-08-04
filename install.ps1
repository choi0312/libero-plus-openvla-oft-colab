$ErrorActionPreference = "Stop"

$distro = "Ubuntu-22.04"
$distros = @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", "").Trim() }
if ($distro -notin $distros) {
    throw "WSL distro '$distro' is missing. Run 'wsl --install -d Ubuntu-22.04', reboot, and retry."
}

Write-Host "Installing pinned LIBERO-Plus/OpenVLA-OFT simulator environment..." -ForegroundColor Cyan
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    (Join-Path $PSScriptRoot "libero_plus_oft\setup_local_simulator.ps1")
exit $LASTEXITCODE
