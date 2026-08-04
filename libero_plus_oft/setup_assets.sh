#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUS_ROOT="${WORKSPACE}/third_party/LIBERO-plus"
LIBERO_ROOT="${PLUS_ROOT}/libero/libero"
ZIP_PATH="${PLUS_ROOT}/assets.zip"
ASSET_URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
EXPECTED_SIZE=6395849578

mkdir -p "${PLUS_ROOT}"
CURRENT_SIZE=0
if [[ -f "${ZIP_PATH}" ]]; then
  CURRENT_SIZE="$(stat -c '%s' "${ZIP_PATH}")"
fi

if [[ "${CURRENT_SIZE}" -ne "${EXPECTED_SIZE}" ]]; then
  echo "Downloading/resuming official LIBERO-Plus assets (${CURRENT_SIZE}/${EXPECTED_SIZE} bytes)..."
  curl -L --retry 5 --retry-delay 3 -C - -o "${ZIP_PATH}" "${ASSET_URL}"
fi

ACTUAL_SIZE="$(stat -c '%s' "${ZIP_PATH}")"
if [[ "${ACTUAL_SIZE}" -ne "${EXPECTED_SIZE}" ]]; then
  echo "ERROR: assets.zip size is ${ACTUAL_SIZE}; expected ${EXPECTED_SIZE}." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/extract_assets.py" "${ZIP_PATH}" "${PLUS_ROOT}"
