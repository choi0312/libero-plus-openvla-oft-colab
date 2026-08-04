$ErrorActionPreference = "Stop"
$wslHome = (& wsl.exe -d Ubuntu-22.04 -- bash -lc 'printf %s "$HOME"').Trim()
if ([string]::IsNullOrWhiteSpace($wslHome)) { throw "Could not resolve the WSL home directory." }
$colab = "$wslHome/.local/bin/colab"
& wsl.exe -d Ubuntu-22.04 -- $colab stop -s openvla
exit $LASTEXITCODE
