#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${HOME}/.venvs/libero-oft"
UV="${HOME}/.local/bin/uv"

OFT_COMMIT="e4287e94541f459edc4feabc4e181f537cd569a8"
PLUS_COMMIT="4976dc30028e805ff8094b55501d532c48fec182"

if [[ ! -x "${UV}" ]]; then
  echo "Installing uv into ${HOME}/.local/bin..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -x "${UV}" ]]; then
  echo "ERROR: uv was not found at ${UV}." >&2
  exit 1
fi

mkdir -p "${WORKSPACE}/third_party"

clone_pinned() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

clone_pinned "https://github.com/moojink/openvla-oft.git" "${WORKSPACE}/third_party/openvla-oft" "${OFT_COMMIT}"
clone_pinned "https://github.com/sylvestf/LIBERO-plus.git" "${WORKSPACE}/third_party/LIBERO-plus" "${PLUS_COMMIT}"

if ! ldconfig -p 2>/dev/null | grep -qi "libMagickWand"; then
  echo "NOTICE: WSL system libraries are missing. Run setup.ps1 from Windows so it can install them as WSL root."
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV}" venv --python 3.10 "${VENV}"
else
  echo "Reusing existing virtual environment: ${VENV}"
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

"${UV}" pip install --python "${VENV}/bin/python" \
  -r "${SCRIPT_DIR}/requirements-runtime.txt"

# pynput is only needed for optional physical keyboard/SpaceMouse devices. Its
# evdev build needs WSL kernel headers, so install robosuite without that unused extra.
"${UV}" pip install --python "${VENV}/bin/python" --no-deps robosuite==1.4.1
ROBOSUITE_DIR="${VENV}/lib/python3.10/site-packages/robosuite"
if [[ ! -f "${ROBOSUITE_DIR}/macros_private.py" ]]; then
  "${VENV}/bin/python" "${ROBOSUITE_DIR}/scripts/setup_macros.py"
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  "git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab"

"${UV}" tool install --force google-colab-cli==0.6.0

"${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${WORKSPACE}/third_party/openvla-oft"
"${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${WORKSPACE}/third_party/LIBERO-plus"

"${VENV}/bin/python" "${SCRIPT_DIR}/verify_install.py" --skip-environment

echo
echo "Python environment is ready: ${VENV}"
echo "LIBERO assets are checked separately because assets.zip is about 6.4 GB."
