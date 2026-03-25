from __future__ import annotations

import cv2


class VideoSource:
    def __init__(
        self,
        camera_index: int = 0,
        camera_pipeline: str = "",
        video_path: str = "",
        loop: bool = True,
        mirror: bool = True,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        self.camera_index = int(camera_index)
        self.camera_pipeline = camera_pipeline
        self.video_path = video_path
        self.loop = bool(loop)
        self.mirror = bool(mirror)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.use_webcam = not bool(video_path)
        self.cap: cv2.VideoCapture | None = None
        self._open_capture()

    def _open_capture(self) -> None:
        if self.use_webcam and self.camera_pipeline:
            source = self.camera_pipeline
            self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            source = self.camera_index if self.use_webcam else self.video_path
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened() and self.use_webcam:
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        if self.use_webcam:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> cv2.typing.MatLike | None:
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            if not self.use_webcam and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
