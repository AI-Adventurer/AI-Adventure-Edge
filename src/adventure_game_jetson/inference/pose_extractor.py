from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


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
        if self.input_width > 0 and self.input_height > 0:
            frame_bgr = cv2.resize(frame_bgr, (self.input_width, self.input_height))
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return np.zeros((self.num_joints, 3), dtype=np.float32)

        landmarks = results.pose_landmarks.landmark
        out = np.zeros((self.num_joints, 3), dtype=np.float32)
        count = min(self.num_joints, len(landmarks))
        for i in range(count):
            lm = landmarks[i]
            out[i, 0] = lm.x
            out[i, 1] = lm.y
            out[i, 2] = lm.z
        return out
