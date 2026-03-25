from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from adventure_game_jetson.capture import VideoSource
from adventure_game_jetson.core import ActionPrediction, GameEngine
from adventure_game_jetson.inference import FrameTimings, RollingProfiler
from adventure_game_jetson.ui import GameRenderer

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "models" / "config.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_TRT_ENGINE = PROJECT_ROOT / "models" / "ctrgcn_fp16.engine"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone motion adventure game for Jetson.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-pipeline", default="")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pose-backend", default="mediapipe")
    parser.add_argument("--action-backend", default="auto")
    parser.add_argument("--action-engine", default=str(DEFAULT_TRT_ENGINE))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--pose-every-n-frames", type=int, default=2)
    parser.add_argument("--scale-w", type=float, default=1.0)
    parser.add_argument("--scale-h", type=float, default=1.0)
    parser.add_argument("--centralize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpolate-60fps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mp-model-complexity", type=int, default=0)
    parser.add_argument("--mp-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mp-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mp-input-width", type=int, default=256)
    parser.add_argument("--mp-input-height", type=int, default=256)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-every", type=int, default=60)
    parser.add_argument("--min-conf", type=float, default=0.6)
    parser.add_argument("--hp-init", type=int, default=10)
    parser.add_argument("--story-duration", type=float, default=5.0)
    parser.add_argument("--prep-duration", type=float, default=2.0)
    parser.add_argument("--event-duration", type=float, default=15.0)
    parser.add_argument("--result-duration", type=float, default=3.0)
    parser.add_argument("--run-duration", type=float, default=15.0)
    parser.add_argument("--ending-duration", type=float, default=15.0)
    parser.add_argument("--font-path", default="")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from adventure_game_jetson.inference.runtime import ActionRecognizer
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    video_source = VideoSource(
        camera_index=args.camera_index,
        camera_pipeline=args.camera_pipeline,
        video_path=args.video_path,
        loop=args.loop,
        mirror=args.mirror,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    recognizer = ActionRecognizer(
        config_path=args.config,
        weights_path=args.weights,
        device=args.device,
        pose_backend=args.pose_backend,
        action_backend=args.action_backend,
        action_engine=args.action_engine,
        window_size=args.window_size,
        stride=args.stride,
        smooth_k=args.smooth_k,
        pose_every_n_frames=args.pose_every_n_frames,
        scale_w=args.scale_w,
        scale_h=args.scale_h,
        centralize=args.centralize,
        interpolate_60fps=args.interpolate_60fps,
        mp_model_complexity=args.mp_model_complexity,
        mp_min_detection_confidence=args.mp_min_detection_confidence,
        mp_min_tracking_confidence=args.mp_min_tracking_confidence,
        mp_input_width=args.mp_input_width,
        mp_input_height=args.mp_input_height,
    )
    print(
        "[adventure-game] "
        f"pose_backend={args.pose_backend} "
        f"action_backend={recognizer.action_backend} "
        f"action_device={recognizer.action_device} "
        f"device={args.device}"
    )
    engine = GameEngine(
        min_conf=args.min_conf,
        hp_init=args.hp_init,
        story_duration=args.story_duration,
        prep_duration=args.prep_duration,
        event_duration=args.event_duration,
        result_duration=args.result_duration,
        run_duration=args.run_duration,
        ending_duration=args.ending_duration,
        openai_api_key=args.openai_api_key,
    )
    renderer = GameRenderer(
        hp_max=args.hp_init,
        font_path=args.font_path,
        width=args.window_width,
        height=args.window_height,
    )

    last_prediction = ActionPrediction()
    profiler = RollingProfiler(report_every=args.profile_every) if args.profile else None
    frame_period = 1.0 / max(1e-6, args.fps)
    next_frame_at = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
            next_frame_at = max(next_frame_at + frame_period, time.monotonic())

            frame_started = time.perf_counter()
            capture_started = time.perf_counter()
            frame = video_source.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            if frame is None:
                break

            produced_at = time.monotonic()
            skeleton, prediction, recognizer_timings = recognizer.process_frame(
                frame,
                produced_at=produced_at,
            )
            if prediction is not None:
                last_prediction = ActionPrediction(
                    action=prediction.action,
                    confidence=prediction.confidence,
                    produced_at=prediction.produced_at,
                )
                engine.submit_action(last_prediction)

            tick_started = time.perf_counter()
            snapshot = engine.tick(produced_at)
            tick_ms = (time.perf_counter() - tick_started) * 1000.0
            render_started = time.perf_counter()
            canvas = renderer.render(frame, skeleton, snapshot, last_prediction)
            render_ms = (time.perf_counter() - render_started) * 1000.0
            show_started = time.perf_counter()
            key = renderer.show(canvas)
            show_ms = (time.perf_counter() - show_started) * 1000.0

            if profiler is not None:
                profiler.update(
                    FrameTimings(
                        capture_ms=capture_ms,
                        pose_ms=recognizer_timings.pose_ms,
                        preprocess_ms=recognizer_timings.preprocess_ms,
                        action_ms=recognizer_timings.action_ms,
                        tick_ms=tick_ms,
                        render_ms=render_ms,
                        show_ms=show_ms,
                        total_ms=(time.perf_counter() - frame_started) * 1000.0,
                    )
                )
            if key in {27, ord("q")}:
                break
    finally:
        video_source.close()
        try:
            recognizer.close()
        except Exception:
            pass
        renderer.close()


if __name__ == "__main__":
    main()
