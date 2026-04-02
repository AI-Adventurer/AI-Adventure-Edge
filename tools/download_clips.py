"""
Download short video clips for each action class using yt-dlp,
then extract MediaPipe skeletons directly.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson
    python tools/download_clips.py --output data/extracted
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adventure_game_jetson.inference.pose_extractor import MediaPipePoseExtractor

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

# YouTube search queries for each action.
# We search and download multiple short clips per action.
SEARCH_QUERIES = {
    "stand": [
        "person standing still front camera",
        "standing idle pose full body",
        "person standing straight exercise start position",
    ],
    "jump": [
        "person jumping in place exercise",
        "jump exercise fitness full body",
        "vertical jump training",
        "jumping jacks exercise",
    ],
    "crouch": [
        "bodyweight squat exercise form",
        "squat exercise tutorial full body",
        "deep squat exercise fitness",
        "crouch down stand up exercise",
    ],
    "push": [
        "standing push exercise arms forward",
        "push motion exercise standing",
        "chest press standing exercise",
        "push forward motion full body",
    ],
    "run_forward": [
        "running in place exercise",
        "high knees exercise",
        "jogging in place workout",
        "run in place cardio",
    ],
}


def search_and_download(
    query: str,
    output_dir: str,
    max_videos: int = 5,
    max_duration: int = 30,
) -> list[str]:
    """Search YouTube and download short clips. Returns list of downloaded file paths."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "yt-dlp",
        f"ytsearch{max_videos}:{query}",
        "--format", "worst[height>=240][ext=mp4]/worst[ext=mp4]/worst",
        "--max-filesize", "20M",
        "--match-filter", f"duration<={max_duration}",
        "--no-playlist",
        "--output", os.path.join(output_dir, "%(id)s.%(ext)s"),
        "--print", "after_move:filepath",
        "--no-warnings",
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        return [p for p in paths if os.path.exists(p)]
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    WARNING: search failed for '{query}': {e}")
        return []


def extract_skeleton_from_video(
    video_path: str,
    pose: MediaPipePoseExtractor,
    max_frames: int = 300,
) -> np.ndarray | None:
    """Extract MediaPipe skeleton sequence from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    skeletons = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        skeleton = pose.extract(frame)
        if np.any(skeleton):
            skeletons.append(skeleton.copy())

    cap.release()

    if len(skeletons) < 15:
        return None
    return np.stack(skeletons).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Download action clips and extract skeletons")
    parser.add_argument("--output", "-o", default="data/extracted")
    parser.add_argument("--videos-per-query", type=int, default=5)
    parser.add_argument("--max-duration", type=int, default=30, help="Max video duration in seconds")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to extract per video")
    parser.add_argument("--actions", nargs="+", default=ACTIONS, help="Actions to download")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only extract")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tmp_video_dir = os.path.join(args.output, "_videos")
    os.makedirs(tmp_video_dir, exist_ok=True)

    # Initialize MediaPipe
    print("Initializing MediaPipe Pose...")
    pose = MediaPipePoseExtractor(
        num_joints=33,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        input_width=0,
        input_height=0,
    )

    total_saved = 0

    for action in args.actions:
        if action not in SEARCH_QUERIES:
            print(f"WARNING: No search queries for '{action}', skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Action: {action.upper()}")
        print(f"{'=' * 60}")

        action_video_dir = os.path.join(tmp_video_dir, action)
        os.makedirs(action_video_dir, exist_ok=True)

        # Download videos
        if not args.skip_download:
            for query in SEARCH_QUERIES[action]:
                print(f"\n  Searching: '{query}'...")
                paths = search_and_download(
                    query,
                    action_video_dir,
                    max_videos=args.videos_per_query,
                    max_duration=args.max_duration,
                )
                print(f"    Downloaded: {len(paths)} videos")

        # Extract skeletons from all downloaded videos
        video_files = sorted(
            f for f in os.listdir(action_video_dir)
            if f.endswith((".mp4", ".mkv", ".webm", ".avi"))
        )

        action_saved = 0
        for i, vf in enumerate(video_files):
            video_path = os.path.join(action_video_dir, vf)
            out_name = f"{action}_yt_{os.path.splitext(vf)[0]}.npy"
            out_path = os.path.join(args.output, out_name)

            if os.path.exists(out_path):
                action_saved += 1
                continue

            print(f"  Extracting [{i+1}/{len(video_files)}]: {vf}...", end=" ", flush=True)
            skeleton_seq = extract_skeleton_from_video(video_path, pose, args.max_frames)

            if skeleton_seq is not None:
                np.save(out_path, skeleton_seq)
                action_saved += 1
                print(f"{skeleton_seq.shape[0]} frames")
            else:
                print("FAILED")

        total_saved += action_saved
        print(f"\n  {action}: {action_saved} sequences extracted")

    pose.close()

    # Write labels
    labels_path = os.path.join(args.output, "labels.txt")
    with open(labels_path, "w") as f:
        for i, act in enumerate(ACTIONS):
            f.write(f"{i} {act}\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")

    by_label = {}
    for f in sorted(os.listdir(args.output)):
        if not f.endswith(".npy"):
            continue
        label = f.split("_", 1)[0]
        arr = np.load(os.path.join(args.output, f))
        by_label.setdefault(label, []).append(arr.shape[0])

    for action in ACTIONS:
        counts = by_label.get(action, [])
        if counts:
            print(f"  {action:12s}: {len(counts):3d} sequences, {sum(counts):5d} frames")
        else:
            print(f"  {action:12s}:   0 sequences")

    print(f"\n  Total: {total_saved} sequences")
    print(f"  Output: {args.output}/")
    print(f"  Next: python tools/train_model.py --data {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
