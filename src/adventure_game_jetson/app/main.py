from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

from adventure_game_jetson.inference import FrameTimings, RollingProfiler

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
DEFAULT_SOURCE_ID = socket.gethostname()
DEFAULT_EDGE_SERVER_URL = os.environ.get("ADVENTURE_GAME_EDGE_SERVER_URL", "http://192.168.50.174:8000")


def _iter_model_dir_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_model_dir = (
        os.environ.get("ADVENTURE_GAME_JETSON_MODEL_DIR", "")
        or os.environ.get("AI_ADVENTURE_EDGE_MODEL_DIR", "")
    )
    if env_model_dir:
        candidates.append(Path(env_model_dir).expanduser())

    env_home = (
        os.environ.get("ADVENTURE_GAME_JETSON_HOME", "")
        or os.environ.get("AI_ADVENTURE_EDGE_HOME", "")
    )
    if env_home:
        candidates.append(Path(env_home).expanduser() / "models")

    starts = [Path.cwd(), Path(__file__).resolve().parent]
    for start in starts:
        for candidate_root in [start, *start.parents]:
            candidates.append(candidate_root / "models")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _find_default_model_file(filename: str) -> Path:
    for model_dir in _iter_model_dir_candidates():
        candidate = model_dir / filename
        if candidate.exists():
            return candidate
    return Path.cwd() / "models" / filename


DEFAULT_CONFIG = _find_default_model_file("config.yaml")
DEFAULT_WEIGHTS = _find_default_model_file("best.pt")
DEFAULT_TRT_ENGINE = _find_default_model_file("ctrgcn_fp16.engine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone adventure game or headless Jetson edge inference node."
    )
    parser.add_argument("--mode", choices=("standalone", "edge"), default="standalone")
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
    parser.add_argument("--pose-every-n-frames", type=int, default=1)
    parser.add_argument("--scale-w", type=float, default=1.0)
    parser.add_argument("--scale-h", type=float, default=1.0)
    parser.add_argument("--centralize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpolate-60fps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mp-model-complexity", type=int, default=0)
    parser.add_argument("--mp-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mp-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mp-input-width", type=int, default=320)
    parser.add_argument("--mp-input-height", type=int, default=240)
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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--edge-output-path", default="")
    parser.add_argument("--edge-sio-url", default=DEFAULT_EDGE_SERVER_URL)
    parser.add_argument("--edge-sio-event", default="frame")
    parser.add_argument("--edge-sio-namespace", default="/edge/frames")
    parser.add_argument("--edge-sio-path", default="socket.io")
    parser.add_argument("--edge-sio-transports", default="polling,websocket")
    parser.add_argument("--edge-video-url", default=DEFAULT_EDGE_SERVER_URL)
    parser.add_argument("--edge-video-namespace", default="/edge/video")
    parser.add_argument("--edge-video-path", default="socket.io")
    parser.add_argument("--edge-video-transports", default="polling,websocket")
    parser.add_argument("--edge-video-offer-event", default="offer")
    parser.add_argument("--edge-video-answer-event", default="answer")
    parser.add_argument("--edge-video-candidate-event", default="candidate")
    parser.add_argument("--edge-video-response-event", default="response")
    parser.add_argument("--edge-video-fps", type=float, default=15.0)
    parser.add_argument("--edge-video-width", type=int, default=640)
    parser.add_argument("--edge-video-height", type=int, default=480)
    parser.add_argument("--edge-video-ice-servers", default="stun:stun.l.google.com:19302")
    parser.add_argument("--edge-include-preview", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-preview-width", type=int, default=480)
    parser.add_argument("--edge-preview-height", type=int, default=360)
    parser.add_argument("--edge-preview-quality", type=int, default=80)
    parser.add_argument("--edge-preview-every-n-frames", type=int, default=3)
    parser.add_argument("--edge-preview-overlay", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def _build_runtime(args: argparse.Namespace):
    from adventure_game_jetson.capture import VideoSource
    from adventure_game_jetson.inference.runtime import ActionRecognizer

    args.device = _resolve_device(args.device)
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
    return video_source, recognizer


def run_standalone(args: argparse.Namespace) -> None:
    from adventure_game_jetson.core import ActionPrediction, GameEngine
    from adventure_game_jetson.ui import GameRenderer

    video_source, recognizer = _build_runtime(args)
    print(
        "[adventure-game] "
        f"mode={args.mode} "
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
    frame_id = 0

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
            frame_id += 1

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
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        video_source.close()
        try:
            recognizer.close()
        except Exception:
            pass
        renderer.close()


def run_edge(args: argparse.Namespace) -> None:
    from adventure_game_jetson.edge import (
        EdgePacketBuilder,
        build_edge_publisher,
        build_edge_video_streamer,
    )

    sio_transports = [
        item.strip()
        for item in args.edge_sio_transports.split(",")
        if item.strip()
    ]
    video_transports = [
        item.strip()
        for item in args.edge_video_transports.split(",")
        if item.strip()
    ]
    video_ice_servers = [
        item.strip()
        for item in args.edge_video_ice_servers.split(",")
        if item.strip()
    ]
    video_source, recognizer = _build_runtime(args)
    publisher = build_edge_publisher(
        output_path=args.edge_output_path,
        sio_url=args.edge_sio_url,
        sio_event=args.edge_sio_event,
        sio_namespace=args.edge_sio_namespace,
        sio_path=args.edge_sio_path,
        sio_transports=sio_transports,
    )
    packet_builder = EdgePacketBuilder(
        source_id=args.source_id,
        action_labels=recognizer.action_labels,
        include_preview=args.edge_include_preview,
        preview_width=args.edge_preview_width,
        preview_height=args.edge_preview_height,
        preview_quality=args.edge_preview_quality,
        preview_every_n_frames=args.edge_preview_every_n_frames,
        preview_overlay=args.edge_preview_overlay,
        layout=recognizer.sequence_layout,
    )
    video_streamer = None
    if args.edge_video_url:
        video_streamer = build_edge_video_streamer(
            url=args.edge_video_url,
            source_id=args.source_id,
            namespace=args.edge_video_namespace,
            socketio_path=args.edge_video_path,
            transports=video_transports,
            offer_event=args.edge_video_offer_event,
            answer_event=args.edge_video_answer_event,
            candidate_event=args.edge_video_candidate_event,
            response_event=args.edge_video_response_event,
            fps=args.edge_video_fps if args.edge_video_fps > 0.0 else args.fps,
            width=args.edge_video_width,
            height=args.edge_video_height,
            ice_servers=video_ice_servers,
        )
        video_streamer.start()
    print(
        "[adventure-game] "
        f"mode={args.mode} "
        f"pose_backend={args.pose_backend} "
        f"action_backend={recognizer.action_backend} "
        f"action_device={recognizer.action_device} "
        f"device={args.device} "
        f"source_id={args.source_id} "
        f"sio_url={args.edge_sio_url or '-'} "
        f"sio_namespace={args.edge_sio_namespace} "
        f"sio_event={args.edge_sio_event} "
        f"sio_transports={','.join(sio_transports) or '-'} "
        f"video_url={args.edge_video_url or '-'} "
        f"video_namespace={args.edge_video_namespace if args.edge_video_url else '-'}",
        file=sys.stderr,
    )

    frame_period = 1.0 / max(1e-6, args.fps)
    next_frame_at = time.monotonic()
    frame_id = 0
    last_prediction = recognizer.latest_prediction()

    try:
        while True:
            now = time.monotonic()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
            next_frame_at = max(next_frame_at + frame_period, time.monotonic())

            capture_started = time.perf_counter()
            frame = video_source.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            if frame is None:
                break

            frame_id += 1
            if video_streamer is not None:
                video_streamer.submit_frame(frame)
            packet_timestamp = time.time()
            skeleton, prediction, recognizer_timings = recognizer.process_frame(
                frame,
                produced_at=packet_timestamp,
            )
            if prediction is not None:
                last_prediction = prediction

            packet = packet_builder.build_packet(
                frame_id=frame_id,
                timestamp=packet_timestamp,
                frame=frame,
                skeleton=skeleton,
                skeleton_sequence=recognizer.current_skeleton_sequence(),
                prediction=last_prediction,
                timings=recognizer_timings,
                capture_ms=capture_ms,
                pose_backend=recognizer.pose_backend,
                action_backend=recognizer.action_backend,
                action_device=recognizer.action_device,
            )
            publisher.publish(packet)

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        video_source.close()
        try:
            recognizer.close()
        except Exception:
            pass
        if video_streamer is not None:
            video_streamer.close()
        publisher.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "edge":
        run_edge(args)
        return
    run_standalone(args)


if __name__ == "__main__":
    main()
