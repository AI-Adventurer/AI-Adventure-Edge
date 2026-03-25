from __future__ import annotations

from dataclasses import dataclass
import time
from time import perf_counter

import numpy as np

from .backends import create_action_backend
from .ctrgcn_runner import CTRGCNRunner, InferenceResult
from .pose_extractor import MediaPipePoseExtractor
from .profiling import RecognizerTimings


@dataclass(slots=True)
class ActionPrediction:
    action: str
    confidence: float
    produced_at: float


class ActionRecognizer:
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cpu",
        pose_backend: str = "mediapipe",
        action_backend: str = "pytorch",
        action_engine: str = "",
        window_size: int = 30,
        stride: int = 1,
        smooth_k: int = 5,
        pose_every_n_frames: int = 1,
        scale_w: float = 1.0,
        scale_h: float = 1.0,
        centralize: bool = True,
        interpolate_60fps: bool = True,
        mp_model_complexity: int = 1,
        mp_min_detection_confidence: float = 0.5,
        mp_min_tracking_confidence: float = 0.5,
        mp_input_width: int = 0,
        mp_input_height: int = 0,
    ) -> None:
        self.pose_backend = pose_backend.strip().lower()
        self.action_backend = action_backend.strip().lower()
        self.pose_every_n_frames = max(1, int(pose_every_n_frames))
        self.scale_w = float(scale_w)
        self.scale_h = float(scale_h)
        self.centralize = bool(centralize)
        self.interpolate_60fps = bool(interpolate_60fps)

        self.action_labels = ["stand", "jump", "crouch", "push", "run_forward"]
        backend = create_action_backend(
            backend_name=self.action_backend,
            config_path=config_path,
            weights_path=weights_path,
            device=device,
            engine_path=action_engine,
        )
        self.action_backend = backend.name
        self.action_device = backend.device_label
        self.runner = CTRGCNRunner(
            action_labels=self.action_labels,
            backend=backend,
            window_size=window_size,
            stride=stride,
            smooth_k=smooth_k,
        )
        if self.pose_backend != "mediapipe":
            raise ValueError(f"Unsupported pose backend: {pose_backend}")
        self.pose_extractor = MediaPipePoseExtractor(
            num_joints=33,
            model_complexity=mp_model_complexity,
            min_detection_confidence=mp_min_detection_confidence,
            min_tracking_confidence=mp_min_tracking_confidence,
            input_width=mp_input_width,
            input_height=mp_input_height,
        )
        self._prev_skeleton: np.ndarray | None = None
        self._last_skeleton = np.zeros((33, 3), dtype=np.float32)
        self._frame_counter = 0

    def reset(self) -> None:
        self.runner.reset()
        self._prev_skeleton = None
        self._last_skeleton = np.zeros((33, 3), dtype=np.float32)
        self._frame_counter = 0

    def close(self) -> None:
        self.runner.close()
        self.pose_extractor.close()

    def _preprocess_skeleton(self, skeleton: np.ndarray) -> np.ndarray | None:
        if skeleton is None:
            return None

        norm = np.array(skeleton, dtype=np.float32, copy=True)
        np.clip(norm, 0.0, 1.0, out=norm)
        norm[:, 0] *= self.scale_w
        norm[:, 1] *= self.scale_h
        norm[:, 2] *= self.scale_w

        if self.centralize and len(norm) >= 25:
            cx = (norm[23, 0] + norm[24, 0]) / 2.0
            cy = (norm[23, 1] + norm[24, 1]) / 2.0
            cz = (norm[23, 2] + norm[24, 2]) / 2.0
            norm[:, 0] -= cx
            norm[:, 1] -= cy
            norm[:, 2] -= cz

        return norm

    def _run_sequence(self, skeleton: np.ndarray) -> InferenceResult | None:
        final_res: InferenceResult | None = None
        if self.interpolate_60fps and self._prev_skeleton is not None:
            mid_skeleton = (self._prev_skeleton + skeleton) * 0.5
            final_res = self.runner.step(mid_skeleton)
        final_res = self.runner.step(skeleton)
        self._prev_skeleton = skeleton
        if final_res.ready:
            return final_res
        return None

    def predict(
        self,
        skeleton: np.ndarray,
        produced_at: float | None = None,
    ) -> tuple[ActionPrediction | None, float, float]:
        preprocess_started = perf_counter()
        processed = self._preprocess_skeleton(skeleton)
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0
        if processed is None or not np.any(processed):
            self._prev_skeleton = None
            return None, preprocess_ms, 0.0

        action_started = perf_counter()
        runner_result = self._run_sequence(processed)
        action_ms = (perf_counter() - action_started) * 1000.0
        if runner_result is None or not runner_result.ready:
            return None, preprocess_ms, action_ms

        ts = float(produced_at) if produced_at is not None else time.time()
        return (
            ActionPrediction(
                action=runner_result.action,
                confidence=float(runner_result.score),
                produced_at=ts,
            ),
            preprocess_ms,
            action_ms,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        produced_at: float | None = None,
    ) -> tuple[np.ndarray, ActionPrediction | None, RecognizerTimings]:
        total_started = perf_counter()
        self._frame_counter += 1
        refresh_pose = ((self._frame_counter - 1) % self.pose_every_n_frames) == 0

        if refresh_pose:
            pose_started = perf_counter()
            skeleton = self.pose_extractor.extract(frame_bgr)
            pose_ms = (perf_counter() - pose_started) * 1000.0
            self._last_skeleton = skeleton
            prediction, preprocess_ms, action_ms = self.predict(skeleton, produced_at=produced_at)
        else:
            skeleton = self._last_skeleton
            pose_ms = 0.0
            preprocess_ms = 0.0
            action_ms = 0.0
            prediction = None

        timings = RecognizerTimings(
            pose_ms=pose_ms,
            preprocess_ms=preprocess_ms,
            action_ms=action_ms,
            total_ms=(perf_counter() - total_started) * 1000.0,
            pose_backend=self.pose_backend,
            action_backend=self.action_backend,
        )
        return skeleton, prediction, timings
