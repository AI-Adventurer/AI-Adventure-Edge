from __future__ import annotations

from pathlib import Path

from .base import ActionModelBackend
from .pytorch_ctrgcn import PyTorchCTRGCNBackend, load_model_from_config
from .tensorrt_ctrgcn import TensorRTCTRGCNBackend

__all__ = [
    "ActionModelBackend",
    "PyTorchCTRGCNBackend",
    "TensorRTCTRGCNBackend",
    "create_action_backend",
    "load_model_from_config",
]


def create_action_backend(
    backend_name: str,
    config_path: str,
    weights_path: str,
    device: str,
    engine_path: str = "",
) -> ActionModelBackend:
    name = backend_name.strip().lower()
    weights_path_obj = Path(weights_path).expanduser().resolve()
    default_engine_candidates = [
        weights_path_obj.with_name("ctrgcn_fp16.engine"),
        weights_path_obj.with_name("ctrgcn_test_fp16.engine"),
    ]
    discovered_engine = ""
    for candidate in default_engine_candidates:
        if candidate.exists():
            discovered_engine = str(candidate)
            break

    if engine_path:
        engine_candidate = Path(engine_path).expanduser().resolve()
        engine_path = str(engine_candidate) if engine_candidate.exists() else ""
    else:
        engine_path = discovered_engine

    if name in {"auto", "default"}:
        if device != "cpu" and engine_path:
            return TensorRTCTRGCNBackend(engine_path=engine_path)
        return PyTorchCTRGCNBackend(
            config_path=config_path,
            weights_path=weights_path,
            device=device,
        )
    if name == "pytorch":
        return PyTorchCTRGCNBackend(
            config_path=config_path,
            weights_path=weights_path,
            device=device,
        )
    if name == "tensorrt":
        return TensorRTCTRGCNBackend(engine_path=engine_path)
    raise ValueError(f"Unsupported action backend: {backend_name}")
