from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _import_tensorrt():
    dist_path = "/usr/lib/python3.10/dist-packages"
    if dist_path not in sys.path:
        sys.path.append(dist_path)
    import tensorrt as trt

    return trt


class TensorRTCTRGCNBackend:
    name = "tensorrt"

    def __init__(self, engine_path: str = "") -> None:
        self.trt = _import_tensorrt()
        self.device_label = "cuda:0 (TensorRT)"
        if not engine_path:
            raise ValueError(
                "TensorRT backend requires an engine path. Pass --action-engine /path/to/model.engine."
            )
        self.engine_path = str(Path(engine_path).expanduser())
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            self.runtime = self.trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.input_name = ""
        self.output_name = ""
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == self.trt.TensorIOMode.OUTPUT:
                self.output_name = name
        if not self.input_name or not self.output_name:
            raise RuntimeError("TensorRT engine must contain one input tensor and one output tensor.")

        self.input_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.output_name))
        self.input_tensor = torch.empty(self.input_shape, device="cuda", dtype=torch.float32)
        self.output_tensor = torch.empty(self.output_shape, device="cuda", dtype=torch.float32)
        self.stream = torch.cuda.Stream()

    def infer(self, window: np.ndarray) -> np.ndarray:
        arr = np.asarray(window, dtype=np.float32)
        arr = arr.transpose(2, 0, 1)
        arr = arr[None, :, :, :, None]
        if arr.shape != self.input_shape:
            raise ValueError(
                f"TensorRT engine expects input shape {self.input_shape}, got {arr.shape}."
            )

        host_tensor = torch.from_numpy(arr)
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            self.input_tensor.copy_(host_tensor)
            self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        current_stream.wait_stream(self.stream)
        return self.output_tensor[0].detach().cpu().numpy().copy()

    def close(self) -> None:
        self.context = None
        self.engine = None
        self.runtime = None
