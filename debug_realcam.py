"""Diagnostic: capture real camera frames and test full pipeline."""
import sys
import time
import cv2
import numpy as np
import torch

config_path = "models/config.yaml"
weights_path = "models/best.pt"

from adventure_game_jetson.inference.pose_extractor import MediaPipePoseExtractor
from adventure_game_jetson.inference.backends.pytorch_ctrgcn import PyTorchCTRGCNBackend
from adventure_game_jetson.inference.ctrgcn_runner import softmax_np, CTRGCNRunner, buffer_to_window
from adventure_game_jetson.inference.backends import create_action_backend

labels = ["stand", "jump", "crouch", "push", "run_forward"]

# ── Check MediaPipe version ───────────────────────────────────
import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version:    {cv2.__version__}")
print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")

# ── Setup exactly matching original row2_ws ───────────────────
print("\n=== Setup (matching original row2_ws) ===")
pose = MediaPipePoseExtractor(
    num_joints=33,
    model_complexity=1,       # original default
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    input_width=0,            # no resize
    input_height=0,
)

backend = create_action_backend(
    backend_name="pytorch",
    config_path=config_path,
    weights_path=weights_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
runner = CTRGCNRunner(
    action_labels=labels,
    backend=backend,
    window_size=30,   # v3: match training window
    stride=6,         # v3: more responsive transitions
    smooth_k=2,       # v3: minimal smoothing
)
print(f"Backend: {backend.name} ({backend.device_label})")

# ── Open camera ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera: {w}x{h} @ {fps:.0f}fps")

# ── Replicate original inference_game_node.py pipeline ────────
print("\n=== Capturing 120 frames (~4 sec) - STAND STILL ===")
print("(Replicating original row2_ws pipeline exactly)")

prev_skel = None
frame_count = 0
prediction_count = 0

for i in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror, same as original

    # 1. Extract skeleton (same as original video_publisher)
    skeleton = pose.extract(frame)
    if np.all(skeleton == 0):
        continue

    frame_count += 1

    # 2. Preprocess (same as original inference_game_node._preprocess_skeleton)
    norm = skeleton.copy()
    np.clip(norm, 0.0, 1.0, out=norm)
    # centralize
    cx = (norm[23, 0] + norm[24, 0]) / 2.0
    cy = (norm[23, 1] + norm[24, 1]) / 2.0
    cz = (norm[23, 2] + norm[24, 2]) / 2.0
    norm[:, 0] -= cx
    norm[:, 1] -= cy
    norm[:, 2] -= cz

    # 3. Interpolate 30fps → 60fps (same as original)
    skeletons_to_feed = []
    if prev_skel is not None:
        mid = (prev_skel + norm) * 0.5
        skeletons_to_feed.append(mid)
    skeletons_to_feed.append(norm)
    prev_skel = norm

    # 4. Feed to runner (same as original)
    final_res = None
    for sk in skeletons_to_feed:
        final_res = runner.step(sk)

    # 5. Check result (same as original)
    if final_res and final_res.ready:
        prediction_count += 1
        action = final_res.action
        score = final_res.score
        # original confidence filter
        if score < 0.60:
            action = "stand"
        probs = softmax_np(np.zeros(5)) if final_res.probabilities is None else final_res.probabilities
        prob_str = " | ".join(f"{l}:{p:.3f}" for l, p in zip(labels, probs))
        print(f"  Frame {frame_count:3d} → {action:12s} ({score:.3f}) | {prob_str}")

    # Print raw skeleton for first few frames
    if frame_count <= 3:
        print(f"  [raw skel frame {frame_count}] nose=({skeleton[0,0]:.3f},{skeleton[0,1]:.3f}) "
              f"lhip=({skeleton[23,0]:.3f},{skeleton[23,1]:.3f}) "
              f"rhip=({skeleton[24,0]:.3f},{skeleton[24,1]:.3f})")

cap.release()
pose.close()
runner.close()

print(f"\nTotal: {frame_count} frames processed, {prediction_count} predictions made")
print(f"Runner frame_idx at end: {runner.frame_idx}")
print(f"Runner buf length: {len(runner.buf)}")
