#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH="${1:-$PROJECT_ROOT/models/config.yaml}"
WEIGHTS_PATH="${2:-$PROJECT_ROOT/models/best.pt}"
ONNX_PATH="${3:-/tmp/ctrgcn_fp16.onnx}"
ENGINE_PATH="${4:-$PROJECT_ROOT/models/ctrgcn_fp16.engine}"

python - <<'PY' "$CONFIG_PATH" "$WEIGHTS_PATH" "$ONNX_PATH"
from __future__ import annotations

import sys
import torch
import onnx

from adventure_game_jetson.inference.backends import load_model_from_config

config_path, weights_path, onnx_path = sys.argv[1], sys.argv[2], sys.argv[3]
model = load_model_from_config(config_path, weights_path, torch.device("cpu"))
model.eval()
dummy = torch.randn(1, 3, 30, 33, 1)
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"Exported ONNX to {onnx_path}")
PY

/usr/src/tensorrt/bin/trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --skipInference

echo "Built TensorRT engine at $ENGINE_PATH"
