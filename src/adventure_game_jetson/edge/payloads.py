from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from adventure_game_jetson.inference.profiling import RecognizerTimings
from adventure_game_jetson.inference.runtime import ActionPrediction

_SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
]


def _rounded_array(array: np.ndarray, decimals: int = 6) -> list[Any]:
    if array.size == 0:
        return []
    return np.round(array.astype(np.float32, copy=False), decimals=decimals).tolist()


def _serialize_scores(labels: list[str], prediction: ActionPrediction | None) -> dict[str, float]:
    scores = {label: 0.0 for label in labels}
    if prediction is None:
        return scores
    for label, score in prediction.scores.items():
        scores[label] = round(float(score), 6)
    return scores


def _resolve_preview_size(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    frame_h, frame_w = frame.shape[:2]
    if target_width > 0 and target_height > 0:
        return target_width, target_height
    if target_width > 0:
        scale = target_width / max(1, frame_w)
        return target_width, max(1, int(round(frame_h * scale)))
    if target_height > 0:
        scale = target_height / max(1, frame_h)
        return max(1, int(round(frame_w * scale))), target_height
    return frame_w, frame_h


def _draw_skeleton_overlay(image: np.ndarray, skeleton: np.ndarray | None) -> np.ndarray:
    if skeleton is None or skeleton.size == 0:
        return image

    canvas = np.array(image, copy=True)
    height, width = canvas.shape[:2]
    radius = max(2, min(width, height) // 160)
    thickness = max(2, min(width, height) // 220)
    points: list[tuple[int, int] | None] = []

    for x, y, *_rest in skeleton:
        if float(x) == 0.0 and float(y) == 0.0:
            points.append(None)
            continue
        px = int(round(float(np.clip(x, 0.0, 1.0)) * max(0, width - 1)))
        py = int(round(float(np.clip(y, 0.0, 1.0)) * max(0, height - 1)))
        points.append((px, py))

    for start, end in _SKELETON_CONNECTIONS:
        if start < len(points) and end < len(points) and points[start] and points[end]:
            cv2.line(canvas, points[start], points[end], (0, 255, 0), thickness)
    for point in points:
        if point is not None:
            cv2.circle(canvas, point, radius, (0, 255, 255), -1)
    return canvas


@dataclass(slots=True)
class EdgePacketBuilder:
    source_id: str
    action_labels: list[str]
    include_preview: bool = False
    preview_width: int = 0
    preview_height: int = 0
    preview_quality: int = 80
    preview_every_n_frames: int = 3
    preview_overlay: bool = False
    layout: str = "mediapipe_pose_33"

    def _build_preview_packet(
        self,
        frame_id: int,
        frame: np.ndarray | None,
        skeleton: np.ndarray | None,
    ) -> dict[str, Any] | None:
        if (
            not self.include_preview
            or frame is None
            or (frame_id % max(1, self.preview_every_n_frames)) != 0
        ):
            return None

        image = frame
        if self.preview_overlay:
            image = _draw_skeleton_overlay(image, skeleton)

        target_width, target_height = _resolve_preview_size(
            image,
            int(self.preview_width),
            int(self.preview_height),
        )
        if (target_width, target_height) != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, (target_width, target_height))

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, int(self.preview_quality)))],
        )
        if not ok:
            return None

        return {
            "encoding": "jpeg_base64",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "overlay": "skeleton" if self.preview_overlay else "none",
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    def build_packet(
        self,
        *,
        frame_id: int,
        timestamp: float,
        frame: np.ndarray | None,
        skeleton: np.ndarray,
        skeleton_sequence: np.ndarray | None,
        serialized_pose: list[list[float]] | None = None,
        serialized_skeleton_sequence: list[list[list[float]]] | None = None,
        prediction: ActionPrediction | None,
        timings: RecognizerTimings,
        capture_ms: float,
        pose_backend: str,
        action_backend: str,
        action_device: str,
    ) -> dict[str, Any]:
        preview_packet = self._build_preview_packet(frame_id, frame, skeleton)
        stable_action = prediction.action if prediction is not None else ""
        confidence = float(prediction.confidence) if prediction is not None else 0.0
        sequence_frames = (
            serialized_skeleton_sequence
            if serialized_skeleton_sequence is not None
            else _rounded_array(skeleton_sequence if skeleton_sequence is not None else np.zeros((0, 0, 0)))
        )
        if sequence_frames:
            sequence_shape = [
                len(sequence_frames),
                len(sequence_frames[0]),
                len(sequence_frames[0][0]) if sequence_frames[0] else 0,
            ]
        elif skeleton_sequence is not None and skeleton_sequence.ndim == 3:
            sequence_shape = [
                int(skeleton_sequence.shape[0]),
                int(skeleton_sequence.shape[1]),
                int(skeleton_sequence.shape[2]),
            ]
        else:
            sequence_shape = [0, int(skeleton.shape[0]), int(skeleton.shape[1])]

        packet = {
            "timestamp": round(float(timestamp), 6),
            "source": self.source_id,
            "frame_id": int(frame_id),
            "prediction_ready": bool(prediction is not None and prediction.action),
            "action_scores": _serialize_scores(self.action_labels, prediction),
            "stable_action": stable_action,
            "confidence": round(confidence, 6),
            "frame": {
                "width": int(frame.shape[1]) if frame is not None else 0,
                "height": int(frame.shape[0]) if frame is not None else 0,
            },
            "pose": {
                "layout": self.layout,
                "shape": [int(skeleton.shape[0]), int(skeleton.shape[1])],
                "points": (
                    serialized_pose
                    if serialized_pose is not None
                    else _rounded_array(skeleton)
                ),
            },
            "skeleton_sequence": {
                "layout": self.layout,
                "shape": sequence_shape,
                "frames": sequence_frames,
            },
            "timings_ms": {
                "capture": round(float(capture_ms), 3),
                "pose": round(float(timings.pose_ms), 3),
                "preprocess": round(float(timings.preprocess_ms), 3),
                "action": round(float(timings.action_ms), 3),
                "total": round(float(capture_ms + timings.total_ms), 3),
            },
            "runtime": {
                "pose_backend": pose_backend,
                "action_backend": action_backend,
                "action_device": action_device,
            },
        }
        if preview_packet is not None:
            packet["preview_frame"] = preview_packet
        return packet
