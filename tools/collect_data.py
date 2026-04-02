"""
Data collection tool for CTR-GCN action recognition.
Records skeleton sequences from camera for each action class.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson
    python tools/collect_data.py --output data/collected --seconds 20 --rounds 3

The tool will guide you through recording each action with countdowns.
Each recording is saved as a .npy file: <action>_r<round>.npy  shape (T, 33, 3)
"""
import argparse
import os
import sys
import time

import numpy as np

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adventure_game_jetson.capture import VideoSource
from adventure_game_jetson.inference.pose_extractor import MediaPipePoseExtractor

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

# Tips displayed before each action recording
ACTION_TIPS = {
    "stand": "Stand naturally. Stay relaxed. Small shifts are OK.",
    "jump": "Jump repeatedly. Vary height and speed. Arms up when jumping.",
    "crouch": "Crouch down and stand up repeatedly. Vary speed and depth.",
    "push": "Push forward with both hands repeatedly. Step forward if you want.",
    "run_forward": "Run in place. Pump your arms. Vary speed.",
}


def collect(args):
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  CTR-GCN Data Collection Tool")
    print(f"  Actions: {', '.join(ACTIONS)}")
    print(f"  Rounds: {args.rounds}  |  Seconds/action: {args.seconds}")
    print(f"  Output: {args.output}/")
    print("=" * 60)

    video = VideoSource(
        camera_index=args.camera,
        width=640,
        height=480,
        fps=30.0,
        mirror=True,
    )
    pose = MediaPipePoseExtractor(
        num_joints=33,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        input_width=0,
        input_height=0,
    )

    # Warm up camera + MediaPipe (first few frames are often bad)
    print("\nWarming up camera and MediaPipe...")
    for _ in range(30):
        frame = video.read()
        if frame is not None:
            pose.extract(frame)
        time.sleep(1.0 / 30)
    print("Ready!\n")

    total_saved = 0

    for round_idx in range(args.rounds):
        print(f"\n{'#' * 60}")
        print(f"  ROUND {round_idx + 1} / {args.rounds}")
        print(f"{'#' * 60}")

        for action in ACTIONS:
            filename = f"{action}_r{round_idx:02d}.npy"
            filepath = os.path.join(args.output, filename)

            # Check if already exists (resume support)
            if os.path.exists(filepath) and not args.overwrite:
                print(f"\n  [SKIP] {filename} already exists (use --overwrite to redo)")
                continue

            print(f"\n{'=' * 50}")
            print(f"  Action: {action.upper()}")
            print(f"  Tip: {ACTION_TIPS.get(action, '')}")
            print(f"{'=' * 50}")

            # Countdown
            for i in range(args.countdown, 0, -1):
                print(f"  Starting in {i}...")
                # Keep reading frames during countdown to keep tracking warm
                frame = video.read()
                if frame is not None:
                    pose.extract(frame)
                time.sleep(1.0)

            print(f"  >>> GO! Perform '{action.upper()}' now! <<<\n")

            frames = []
            target_frames = int(args.seconds * 30)
            missed = 0

            for f in range(target_frames):
                frame = video.read()
                if frame is None:
                    missed += 1
                    continue

                skeleton = pose.extract(frame)
                if np.any(skeleton):
                    frames.append(skeleton.copy())
                else:
                    missed += 1

                # Progress every second
                if (f + 1) % 30 == 0:
                    elapsed = (f + 1) / 30
                    print(
                        f"  Recording... {elapsed:.0f}s/{args.seconds}s "
                        f"({len(frames)} valid, {missed} missed)"
                    )

                time.sleep(1.0 / 30)

            if len(frames) >= 10:
                arr = np.stack(frames)  # (T, 33, 3)
                np.save(filepath, arr)
                total_saved += 1
                print(f"  Saved: {filename} — {arr.shape[0]} frames, shape {arr.shape}")
            else:
                print(f"  WARNING: Only {len(frames)} valid frames, skipping save!")

            # Rest between actions
            if action != ACTIONS[-1] or round_idx < args.rounds - 1:
                print(f"  Rest for {args.rest}s...")
                time.sleep(args.rest)

    video.close()
    pose.close()

    # Write labels file
    labels_path = os.path.join(args.output, "labels.txt")
    with open(labels_path, "w") as f:
        for i, action in enumerate(ACTIONS):
            f.write(f"{i} {action}\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE! Saved {total_saved} recordings to {args.output}/")
    print(f"  Labels: {labels_path}")
    npy_files = sorted(f for f in os.listdir(args.output) if f.endswith(".npy"))
    print(f"  Files ({len(npy_files)}):")
    for f in npy_files:
        arr = np.load(os.path.join(args.output, f))
        print(f"    {f:30s} {arr.shape[0]:5d} frames  shape={arr.shape}")
    print(f"\nNext step: python tools/train_model.py --data {args.output}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Collect skeleton data for CTR-GCN training")
    parser.add_argument("--output", "-o", default="data/collected", help="Output directory")
    parser.add_argument("--seconds", "-s", type=int, default=20, help="Seconds per action per round")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Number of recording rounds")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown seconds before each action")
    parser.add_argument("--rest", type=int, default=3, help="Rest seconds between actions")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing recordings")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
