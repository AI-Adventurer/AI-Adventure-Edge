from __future__ import annotations

from typing import Protocol

import numpy as np


class ActionModelBackend(Protocol):
    name: str
    device_label: str

    def infer(self, window: np.ndarray) -> np.ndarray:
        """Run inference on a (T, V, C) float32 skeleton window."""

    def close(self) -> None:
        """Release backend resources."""
