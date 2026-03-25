from __future__ import annotations

from collections import OrderedDict
import sys

import numpy as np
import torch
import yaml


def import_class(import_str: str):
    mod_str, _, class_str = import_str.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def load_model_from_config(config_path: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cls = import_class(cfg["model"])
    model = model_cls(**cfg["model_args"])

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    new_state = OrderedDict()
    for key, value in state_dict.items():
        new_state[key.split("module.")[-1]] = value
    model.load_state_dict(new_state)

    model.to(device)
    model.eval()
    return model


class PyTorchCTRGCNBackend:
    name = "pytorch"

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.device_label = str(self.device)
        self.model = load_model_from_config(config_path, weights_path, self.device)

    def infer(self, window: np.ndarray) -> np.ndarray:
        arr = np.asarray(window, dtype=np.float32)
        arr = arr.transpose(2, 0, 1)
        arr = arr[None, :, :, :, None]
        inp = torch.from_numpy(arr).to(self.device)
        with torch.inference_mode():
            logits = self.model(inp)[0].detach().float().cpu().numpy()
        return logits

    def close(self) -> None:
        self.model = None

