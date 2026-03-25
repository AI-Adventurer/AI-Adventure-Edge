from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RecognizerTimings:
    pose_ms: float = 0.0
    preprocess_ms: float = 0.0
    action_ms: float = 0.0
    total_ms: float = 0.0
    pose_backend: str = ""
    action_backend: str = ""


@dataclass(slots=True)
class FrameTimings:
    capture_ms: float = 0.0
    pose_ms: float = 0.0
    preprocess_ms: float = 0.0
    action_ms: float = 0.0
    tick_ms: float = 0.0
    render_ms: float = 0.0
    show_ms: float = 0.0
    total_ms: float = 0.0


class RollingProfiler:
    def __init__(self, report_every: int = 60) -> None:
        self.report_every = max(1, int(report_every))
        self.frame_count = 0
        self.capture_sum = 0.0
        self.pose_sum = 0.0
        self.preprocess_sum = 0.0
        self.action_sum = 0.0
        self.tick_sum = 0.0
        self.render_sum = 0.0
        self.show_sum = 0.0
        self.total_sum = 0.0

    def update(self, timings: FrameTimings) -> None:
        self.frame_count += 1
        self.capture_sum += timings.capture_ms
        self.pose_sum += timings.pose_ms
        self.preprocess_sum += timings.preprocess_ms
        self.action_sum += timings.action_ms
        self.tick_sum += timings.tick_ms
        self.render_sum += timings.render_ms
        self.show_sum += timings.show_ms
        self.total_sum += timings.total_ms

        if (self.frame_count % self.report_every) == 0:
            scale = float(self.report_every)
            avg_total = self.total_sum / scale
            fps = 1000.0 / avg_total if avg_total > 1e-6 else 0.0
            print(
                "[profile] "
                f"frames={self.frame_count} "
                f"capture={self.capture_sum / scale:.1f}ms "
                f"pose={self.pose_sum / scale:.1f}ms "
                f"prep={self.preprocess_sum / scale:.1f}ms "
                f"action={self.action_sum / scale:.1f}ms "
                f"tick={self.tick_sum / scale:.1f}ms "
                f"render={self.render_sum / scale:.1f}ms "
                f"show={self.show_sum / scale:.1f}ms "
                f"total={avg_total:.1f}ms "
                f"fps={fps:.2f}"
            )
            self.capture_sum = 0.0
            self.pose_sum = 0.0
            self.preprocess_sum = 0.0
            self.action_sum = 0.0
            self.tick_sum = 0.0
            self.render_sum = 0.0
            self.show_sum = 0.0
            self.total_sum = 0.0

