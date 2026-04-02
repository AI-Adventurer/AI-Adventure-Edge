"""
Extract MediaPipe skeletons from video files.
Processes all videos in a directory and saves skeleton sequences as .npy files.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson

    # Extract from a single directory of videos:
    python tools/extract_skeletons.py --input videos/jump/ --output data/extracted/jump --label jump

    # Extract from HMDB51 structure (auto-maps action names):
    python tools/extract_skeletons.py --input data/hmdb51/ --output data/extracted --hmdb51

    # Extract from UCF101 structure:
    python tools/extract_skeletons.py --input data/UCF101/ --output data/extracted --ucf101
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adventure_game_jetson.inference.pose_extractor import MediaPipePoseExtractor

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

# HMDB51 class name → our action mapping
HMDB51_MAP = {
    "stand": "stand",
    "jump": "jump",
    "push": "push",
    "run": "run_forward",
    "walk": "stand",       # walking can supplement stand data
    "sit": "crouch",       # sitting down is similar to crouching
    "situp": "crouch",     # sit-ups involve crouching motion
    "squat": "crouch",
    "climb_stairs": "run_forward",  # leg motion similar to running
    "kick": "push",        # similar body extension
    "punch": "push",       # similar upper body push motion
}

# UCF101 class name → our action mapping
UCF101_MAP = {
    "JumpingJack": "jump",
    "JumpRope": "jump",
    "BodyWeightSquats": "crouch",
    "Lunges": "crouch",
    "PushUps": "push",
    "BoxingPunchingBag": "push",
    "BoxingSpeedBag": "push",
    "Running": "run_forward",
    "Jogging": "run_forward",
    "WalkingWithDog": "stand",
    "TaiChi": "stand",
    "YogaPoses": "stand",
}


def extract_skeletons_from_video(
    video_path: str,
    pose: MediaPipePoseExtractor,
    max_frames: int = 0,
) -> np.ndarray | None:
    """Extract skeleton sequence from a video file. Returns (T, 33, 3) or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if max_frames > 0 and count > max_frames:
            break

        skeleton = pose.extract(frame)
        if np.any(skeleton):
            frames.append(skeleton.copy())

    cap.release()

    if len(frames) < 10:
        return None

    return np.stack(frames).astype(np.float32)


def process_directory(
    input_dir: str,
    output_dir: str,
    label: str,
    pose: MediaPipePoseExtractor,
    max_frames: int = 0,
    prefix: str = "",
) -> int:
    """Process all videos in a directory and save as .npy files. Returns count saved."""
    os.makedirs(output_dir, exist_ok=True)
    video_exts = {".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpg", ".mpeg"}
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in video_exts
    )

    if not files:
        return 0

    saved = 0
    for i, filename in enumerate(files):
        video_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]
        out_name = f"{label}_{prefix}{base}.npy"
        out_path = os.path.join(output_dir, out_name)

        # Skip if already extracted
        if os.path.exists(out_path):
            saved += 1
            continue

        skeleton_seq = extract_skeletons_from_video(video_path, pose, max_frames)

        if skeleton_seq is not None:
            np.save(out_path, skeleton_seq)
            saved += 1
            print(f"  [{saved}/{len(files)}] {out_name}: {skeleton_seq.shape[0]} frames")
        else:
            print(f"  [{i+1}/{len(files)}] {filename}: FAILED (no skeleton detected)")

    return saved


def process_hmdb51(input_dir: str, output_dir: str, pose: MediaPipePoseExtractor, max_frames: int):
    """Process HMDB51 dataset structure: input_dir/<action_class>/videos.avi"""
    print(f"\nProcessing HMDB51 from: {input_dir}")

    total = 0
    for class_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        our_label = HMDB51_MAP.get(class_name)
        if our_label is None:
            continue

        print(f"\n--- {class_name} → {our_label} ---")
        n = process_directory(
            class_dir, output_dir, our_label, pose, max_frames,
            prefix=f"hmdb_{class_name}_",
        )
        total += n
        print(f"  Saved: {n} sequences")

    print(f"\nHMDB51 total: {total} sequences")
    return total


def process_ucf101(input_dir: str, output_dir: str, pose: MediaPipePoseExtractor, max_frames: int):
    """Process UCF101 dataset structure: input_dir/<ActionClass>/videos.avi"""
    print(f"\nProcessing UCF101 from: {input_dir}")

    total = 0
    for class_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        our_label = UCF101_MAP.get(class_name)
        if our_label is None:
            continue

        print(f"\n--- {class_name} → {our_label} ---")
        n = process_directory(
            class_dir, output_dir, our_label, pose, max_frames,
            prefix=f"ucf_{class_name}_",
        )
        total += n
        print(f"  Saved: {n} sequences")

    print(f"\nUCF101 total: {total} sequences")
    return total


def print_summary(output_dir: str):
    """Print summary of extracted data."""
    if not os.path.isdir(output_dir):
        return

    print(f"\n{'=' * 60}")
    print(f"  Extraction Summary: {output_dir}")
    print(f"{'=' * 60}")

    by_label = {}
    total_frames = 0
    for f in sorted(os.listdir(output_dir)):
        if not f.endswith(".npy"):
            continue
        # Label is everything before the first _
        parts = f.split("_", 1)
        label = parts[0]
        arr = np.load(os.path.join(output_dir, f))
        by_label.setdefault(label, []).append(arr.shape[0])
        total_frames += arr.shape[0]

    for label in ACTIONS:
        counts = by_label.get(label, [])
        if counts:
            print(f"  {label:12s}: {len(counts):4d} sequences, {sum(counts):6d} total frames")
        else:
            print(f"  {label:12s}:    0 sequences (MISSING!)")

    total_seqs = sum(len(v) for v in by_label.values())
    print(f"  {'TOTAL':12s}: {total_seqs:4d} sequences, {total_frames:6d} total frames")

    # Write labels file
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as fp:
        for i, action in enumerate(ACTIONS):
            fp.write(f"{i} {action}\n")
    print(f"\n  Labels written to: {labels_path}")
    print(f"  Next step: python tools/train_model.py --data {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe skeletons from video files")
    parser.add_argument("--input", "-i", required=True, help="Input directory with videos")
    parser.add_argument("--output", "-o", default="data/extracted", help="Output directory for .npy files")
    parser.add_argument("--label", "-l", default="", help="Action label (for single directory mode)")
    parser.add_argument("--hmdb51", action="store_true", help="Process as HMDB51 dataset structure")
    parser.add_argument("--ucf101", action="store_true", help="Process as UCF101 dataset structure")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per video (0=unlimited)")
    parser.add_argument("--complexity", type=int, default=1, help="MediaPipe model complexity (0/1/2)")
    args = parser.parse_args()

    pose = MediaPipePoseExtractor(
        num_joints=33,
        model_complexity=args.complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        input_width=0,
        input_height=0,
    )

    os.makedirs(args.output, exist_ok=True)

    if args.hmdb51:
        process_hmdb51(args.input, args.output, pose, args.max_frames)
    elif args.ucf101:
        process_ucf101(args.input, args.output, pose, args.max_frames)
    elif args.label:
        if args.label not in ACTIONS:
            print(f"WARNING: label '{args.label}' not in known actions {ACTIONS}")
        n = process_directory(args.input, args.output, args.label, pose, args.max_frames)
        print(f"\nSaved: {n} sequences")
    else:
        print("ERROR: Specify --label, --hmdb51, or --ucf101")
        sys.exit(1)

    pose.close()
    print_summary(args.output)


if __name__ == "__main__":
    main()
