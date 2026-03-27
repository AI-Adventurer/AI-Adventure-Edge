from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


def _letterbox_resize(
    frame_bgr: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, float, float, int, int]:
    source_height, source_width = frame_bgr.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return frame_bgr, 1.0, 1.0, 0, 0

    scale = min(target_width / float(source_width), target_height / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    offset_x = max(0, (target_width - resized_width) // 2)
    offset_y = max(0, (target_height - resized_height) // 2)

    resized = cv2.resize(frame_bgr, (resized_width, resized_height))
    canvas = np.zeros((target_height, target_width, 3), dtype=frame_bgr.dtype)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, scale, scale, offset_x, offset_y


class MediaPipePoseExtractor:
    """Extract a 33x3 MediaPipe pose skeleton from a BGR frame."""

    def __init__(
        self,
        num_joints: int = 33,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        input_width: int = 0,
        input_height: int = 0,
    ) -> None:
        self.num_joints = int(num_joints)
        self.input_width = max(0, int(input_width))
        self.input_height = max(0, int(input_height))
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._pose.close()

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        source_height, source_width = frame_bgr.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        offset_x = 0
        offset_y = 0

        if self.input_width > 0 and self.input_height > 0:
            frame_bgr, scale_x, scale_y, offset_x, offset_y = _letterbox_resize(
                frame_bgr,
                self.input_width,
                self.input_height,
            )
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return np.zeros((self.num_joints, 3), dtype=np.float32)

        landmarks = results.pose_landmarks.landmark
        out = np.zeros((self.num_joints, 3), dtype=np.float32)
        count = min(self.num_joints, len(landmarks))
        for i in range(count):
            lm = landmarks[i]
            if self.input_width > 0 and self.input_height > 0:
                px = (lm.x * self.input_width) - float(offset_x)
                py = (lm.y * self.input_height) - float(offset_y)
                out[i, 0] = float(np.clip(px / max(1.0, source_width * scale_x), 0.0, 1.0))
                out[i, 1] = float(np.clip(py / max(1.0, source_height * scale_y), 0.0, 1.0))
            else:
                out[i, 0] = lm.x
                out[i, 1] = lm.y
            out[i, 2] = lm.z
        return out
