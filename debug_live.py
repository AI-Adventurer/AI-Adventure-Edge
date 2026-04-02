"""
Live diagnostic v4 — CTR-GCN v3 model (mixed video + live data).
Usage: conda activate adventure_game_jetson && python debug_live.py
"""
import sys
import time
import numpy as np

from adventure_game_jetson.capture import VideoSource
from adventure_game_jetson.inference.runtime import ActionRecognizer

labels = ["stand", "jump", "crouch", "push", "run_forward"]

# Schedule: (start_frame, action_name, duration_sec)
# 30fps, so frame = sec * 30
SCHEDULE = [
    (0,   "STAND (站著不動)", 4),
    (120, "JUMP (跳躍)", 3),
    (210, "CROUCH (蹲下)", 3),
    (300, "PUSH (推)", 3),
    (390, "RUN (跑步)", 3),
]
TOTAL_FRAMES = 480  # 16 seconds

print("=== Opening camera ===")
video = VideoSource(camera_index=0, width=640, height=480, fps=30.0, mirror=True)

print("=== Setting up ActionRecognizer (v3 model) ===")
recognizer = ActionRecognizer(
    config_path="models/config.yaml",
    weights_path="models/best.pt",
    device="cuda",
    window_size=30,
    stride=6,
    smooth_k=2,
    pose_every_n_frames=2,
    interpolate_60fps=False,
    centralize=True,
    mp_model_complexity=1,
    mp_input_width=0,
    mp_input_height=0,
    min_conf=0.40,
)
print(f"Backend: {recognizer.action_backend} ({recognizer.action_device})")

# Countdown
print("\n準備好！3 秒後開始...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    # keep reading frames to warm up
    frame = video.read()
    if frame is not None:
        recognizer.process_frame(frame, produced_at=time.time())
    time.sleep(1.0)

print(f"\n{'=' * 60}")
print("  開始測試！請跟著提示做動作")
print(f"{'=' * 60}\n")

frame_count = 0
pred_count = 0
current_action_idx = 0

for i in range(TOTAL_FRAMES):
    frame = video.read()
    if frame is None:
        break

    frame_count += 1

    # Print action prompt at the right time
    for idx, (start_frame, action_name, dur) in enumerate(SCHEDULE):
        if frame_count == start_frame + 1:
            sec = dur
            print(f"\n>>> 現在請做: {action_name} ({sec}秒) <<<\n")

    produced_at = time.time()
    skeleton, prediction, timings = recognizer.process_frame(frame, produced_at=produced_at)

    if prediction is not None:
        pred_count += 1
        scores = prediction.scores
        top = " | ".join(
            f"{k}:{v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])
        )
        still = " [STILL]" if recognizer._is_still else ""
        print(
            f"  #{pred_count:2d} f={frame_count:3d} → {prediction.action:12s}"
            f"({prediction.confidence:.3f}) | {top}{still}"
        )

    # First 3 frames: show raw skeleton stats
    if frame_count <= 3 and skeleton is not None and np.any(skeleton):
        print(
            f"  [skel {frame_count}] "
            f"x:[{skeleton[:,0].min():.3f}..{skeleton[:,0].max():.3f}] "
            f"y:[{skeleton[:,1].min():.3f}..{skeleton[:,1].max():.3f}] "
            f"z:[{skeleton[:,2].min():.3f}..{skeleton[:,2].max():.3f}]"
        )

    time.sleep(1.0 / 30)

video.close()
recognizer.close()
print(f"\n{'=' * 60}")
print(f"  測試完成！{frame_count} frames, {pred_count} predictions")
print(f"{'=' * 60}")
