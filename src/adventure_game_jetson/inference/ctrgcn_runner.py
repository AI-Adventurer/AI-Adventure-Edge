from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from .backends import ActionModelBackend


def buffer_to_window(
    buffer: Deque[np.ndarray],
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not buffer:
        return np.empty((0, 0, 0), dtype=np.float32)

    first_frame = np.asarray(buffer[0], dtype=np.float32)
    window_shape = (len(buffer),) + first_frame.shape
    if out is None or out.shape != window_shape:
        out = np.empty(window_shape, dtype=np.float32)
    for idx, frame in enumerate(buffer):
        out[idx] = frame
    return out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 1e-12:
        return np.zeros_like(logits)
    return exp / denom


@dataclass(slots=True)
class InferenceResult:
    ready: bool
    action: str = "Warming up..."
    score: float = 0.0
    pred_idx: int = -1
    vote_idx: int = -1
    probabilities: np.ndarray | None = None


class CTRGCNRunner:
    def __init__(
        self,
        action_labels: list[str],
        backend: ActionModelBackend,
        window_size: int = 90,
        stride: int = 2,
        smooth_k: int = 5,
    ) -> None:
        self.action_labels = action_labels
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.smooth_k = max(1, int(smooth_k))
        self.backend = backend

        self.buf: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.pred_hist: Deque[int] = deque(maxlen=self.smooth_k)
        self._window_cache: np.ndarray | None = None

        self.frame_idx = 0
        self.last_action = "Warming up..."
        self.last_score = 0.0

    def reset(self) -> None:
        self.buf.clear()
        self.pred_hist.clear()
        self.frame_idx = 0
        self._window_cache = None
        self.last_action = "Warming up..."
        self.last_score = 0.0

    def close(self) -> None:
        self.backend.close()

    def step(self, skel_v3: np.ndarray) -> InferenceResult:
        self.frame_idx += 1
        self.buf.append(skel_v3)

        if len(self.buf) < self.window_size:
            return InferenceResult(ready=False)

        if (self.frame_idx % self.stride) != 0:
            return InferenceResult(ready=False)

        window = buffer_to_window(self.buf, self._window_cache)
        self._window_cache = window
        logits = np.asarray(self.backend.infer(window), dtype=np.float32).reshape(-1)
        probs = softmax_np(logits)
        pred_idx = int(np.argmax(probs))
        score = float(probs[pred_idx])

        self.pred_hist.append(pred_idx)
        vote_idx = Counter(self.pred_hist).most_common(1)[0][0]
        action = (
            self.action_labels[vote_idx]
            if 0 <= vote_idx < len(self.action_labels)
            else f"cls_{vote_idx}"
        )

        self.last_action = action
        self.last_score = score

        return InferenceResult(
            ready=True,
            action=action,
            score=score,
            pred_idx=pred_idx,
            vote_idx=vote_idx,
            probabilities=probs.copy(),
        )
