#!/usr/bin/env bash
set -euo pipefail

TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl}"

PYTHON_BIN="${PYTHON_BIN:-python}"

cusparselt_found=0
for candidate in \
  "${CONDA_PREFIX:-}/lib/libcusparseLt.so.0" \
  /usr/local/cuda/targets/aarch64-linux/lib/libcusparseLt.so.0 \
  /usr/local/cuda/lib64/libcusparseLt.so.0 \
  /usr/lib/aarch64-linux-gnu/libcusparseLt.so.0
do
  if [[ -f "$candidate" ]]; then
    cusparselt_found=1
    break
  fi
done

if [[ "$cusparselt_found" -ne 1 ]]; then
  echo "libcusparseLt.so.0 was not found." >&2
  echo "Run ./scripts/install_jetson_cusparselt.sh first, then rerun this installer." >&2
  exit 1
fi

python_minor="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$python_minor" != "3.10" ]]; then
  echo "This Jetson torch wheel expects Python 3.10, but found $python_minor." >&2
  echo "Set PYTHON_BIN to a Python 3.10 interpreter or recreate the environment with Python 3.10." >&2
  exit 1
fi

echo "Installing NVIDIA Jetson GPU torch wheel:"
echo "  $TORCH_WHEEL_URL"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --no-cache-dir "$TORCH_WHEEL_URL"

echo
echo "Verifying torch + CUDA..."
"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY
