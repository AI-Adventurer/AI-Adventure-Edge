"""
Complete pipeline: extract skeletons from video dataset → train CTR-GCN.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    python tools/train_from_videos.py \
        --videos data/clips_same_format_5classes \
        --output models/best_v2.pt \
        --base-channel 16 \
        --window 30 \
        --epochs 80
"""
import argparse
import copy
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adventure_game_jetson.inference.model.ctrgcn import Model
from adventure_game_jetson.inference.pose_extractor import MediaPipePoseExtractor

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


# ── Step 1: Extract skeletons ──────────────────────────────────


def extract_all_skeletons(video_dir: str, cache_dir: str, complexity: int = 1) -> dict[str, list[np.ndarray]]:
    """Extract MediaPipe skeletons from all videos, with caching."""
    os.makedirs(cache_dir, exist_ok=True)

    # Check cache
    cache_file = os.path.join(cache_dir, "_all_skeletons.npz")
    if os.path.exists(cache_file):
        print(f"Loading cached skeletons from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        result = {}
        for action in ACTIONS:
            seqs = data.get(action, [])
            if len(seqs) > 0:
                result[action] = list(seqs)
        total = sum(len(v) for v in result.values())
        print(f"  Loaded {total} sequences from cache")
        return result

    pose = MediaPipePoseExtractor(
        num_joints=33,
        model_complexity=complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        input_width=0,
        input_height=0,
    )

    video_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    print(f"\nExtracting skeletons from {len(video_files)} videos...")

    result: dict[str, list[np.ndarray]] = {a: [] for a in ACTIONS}
    failed = 0

    for i, filename in enumerate(video_files):
        # Parse action from filename
        action = None
        for candidate in sorted(ACTIONS, key=len, reverse=True):
            if filename.startswith(candidate + "_"):
                action = candidate
                break
        if action is None:
            continue

        video_path = os.path.join(video_dir, filename)
        cap = cv2.VideoCapture(video_path)
        skeletons = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to match inference camera resolution (640x480)
            # so MediaPipe produces consistent skeleton characteristics.
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, 480))
            skeleton = pose.extract(frame)
            if np.any(skeleton):
                skeletons.append(skeleton.copy())

        cap.release()

        if len(skeletons) >= 10:
            arr = np.stack(skeletons).astype(np.float32)
            result[action].append(arr)
        else:
            failed += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(video_files):
            total = sum(len(v) for v in result.values())
            print(f"  [{i+1}/{len(video_files)}] extracted={total}, failed={failed}")

    pose.close()

    # Save cache
    save_dict = {}
    for action, seqs in result.items():
        if seqs:
            save_dict[action] = np.array(seqs, dtype=object)
    np.savez(cache_file, **save_dict)
    print(f"  Cached to {cache_file}")

    # Summary
    print("\nExtraction summary:")
    for action in ACTIONS:
        seqs = result[action]
        if seqs:
            frames = [s.shape[0] for s in seqs]
            print(f"  {action:12s}: {len(seqs):3d} sequences, "
                  f"frames: {min(frames)}-{max(frames)} (avg {np.mean(frames):.0f})")
        else:
            print(f"  {action:12s}:   0 sequences (WARNING!)")

    return result


# ── Step 2: Data augmentation ──────────────────────────────────


def augment_mirror(seq: np.ndarray) -> np.ndarray:
    s = seq.copy()
    s[:, :, 0] = 1.0 - s[:, :, 0]
    for l, r in MIRROR_PAIRS:
        s[:, l, :], s[:, r, :] = s[:, r, :].copy(), s[:, l, :].copy()
    return s


def augment_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def augment_scale(seq: np.ndarray, lo: float = 0.90, hi: float = 1.10) -> np.ndarray:
    return seq * np.random.uniform(lo, hi)


def augment_shift(seq: np.ndarray, max_s: float = 0.05) -> np.ndarray:
    s = seq.copy()
    s[:, :, 0] += np.random.uniform(-max_s, max_s)
    s[:, :, 1] += np.random.uniform(-max_s, max_s)
    return s


def augment_temporal_stretch(seq: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    T, V, C = seq.shape
    factor = np.random.uniform(lo, hi)
    new_T = max(2, int(T * factor))
    old_t = np.linspace(0, T - 1, new_T, dtype=np.float32)
    idx = np.clip(old_t.astype(int), 0, T - 2)
    frac = old_t - idx
    return (seq[idx] * (1 - frac[:, None, None]) + seq[idx + 1] * frac[:, None, None]).astype(np.float32)


def augment_joint_mask(seq: np.ndarray, mask_rate: float = 0.1) -> np.ndarray:
    """Randomly zero out some joints for entire sequence (forces model to use other joints)."""
    s = seq.copy()
    mask = np.random.rand(33) > mask_rate
    s[:, ~mask, :] = 0.0
    return s


def preprocess(seq: np.ndarray) -> np.ndarray:
    """Clip to [0,1] then centralize around hip midpoint. Matches inference."""
    s = np.clip(seq.copy(), 0.0, 1.0)
    if s.shape[1] >= 25:
        cx = (s[:, 23, 0] + s[:, 24, 0]) / 2.0
        cy = (s[:, 23, 1] + s[:, 24, 1]) / 2.0
        cz = (s[:, 23, 2] + s[:, 24, 2]) / 2.0
        s[:, :, 0] -= cx[:, None]
        s[:, :, 1] -= cy[:, None]
        s[:, :, 2] -= cz[:, None]
    return s


# ── Step 3: Dataset ────────────────────────────────────────────


class ActionDataset(Dataset):
    def __init__(
        self,
        sequences: list[tuple[np.ndarray, int]],
        window_size: int = 30,
        augment: bool = True,
        samples_per_seq: int = 10,
    ):
        self.sequences = sequences
        self.window_size = window_size
        self.augment = augment
        self.samples_per_seq = samples_per_seq

    def __len__(self):
        return len(self.sequences) * self.samples_per_seq

    def __getitem__(self, idx):
        raw, label = self.sequences[idx % len(self.sequences)]
        raw = np.array(raw, dtype=np.float32, copy=True)

        if self.augment:
            if random.random() < 0.5:
                raw = augment_temporal_stretch(raw, 0.7, 1.3)
            if random.random() < 0.5:
                raw = augment_mirror(raw)
            if random.random() < 0.8:
                raw = augment_noise(raw, sigma=0.008)
            if random.random() < 0.5:
                raw = augment_scale(raw, 0.85, 1.15)
            if random.random() < 0.5:
                raw = augment_shift(raw, 0.06)
            if random.random() < 0.3:
                raw = augment_joint_mask(raw, 0.1)

        processed = preprocess(raw)
        T = processed.shape[0]

        if T >= self.window_size:
            if self.augment:
                start = random.randint(0, T - self.window_size)
            else:
                start = (T - self.window_size) // 2
            window = processed[start:start + self.window_size]
        else:
            pad = self.window_size - T
            window = np.concatenate([processed, np.tile(processed[-1:], (pad, 1, 1))], axis=0)

        return torch.from_numpy(window.copy()).float(), label


# ── Step 4: Training ───────────────────────────────────────────


def build_model(base_channel: int = 16, dropout: float = 0.15) -> nn.Module:
    return Model(
        num_class=len(ACTIONS),
        num_point=33,
        num_person=1,
        graph="adventure_game_jetson.inference.graph.mediapipe_pose.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
        drop_out=dropout,
        adaptive=True,
        base_channel=base_channel,
    )


def split_stratified(
    data: dict[str, list[np.ndarray]],
    val_ratio: float = 0.2,
) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    train, val = [], []
    for action in ACTIONS:
        label = ACTIONS.index(action)
        seqs = data.get(action, [])
        random.shuffle(seqs)
        n_val = max(1, int(len(seqs) * val_ratio))
        for s in seqs[:n_val]:
            val.append((s, label))
        for s in seqs[n_val:]:
            train.append((s, label))
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def load_live_npy(live_dir: str, slice_len: int = 60, stride: int = 30) -> dict[str, list[np.ndarray]]:
    """Load .npy skeleton recordings and slice into shorter sequences."""
    result: dict[str, list[np.ndarray]] = {a: [] for a in ACTIONS}
    npy_files = sorted(f for f in os.listdir(live_dir) if f.endswith(".npy"))
    print(f"\nLoading live data from {live_dir} ({len(npy_files)} files)...")

    for filename in npy_files:
        action = None
        for candidate in sorted(ACTIONS, key=len, reverse=True):
            if filename.startswith(candidate + "_"):
                action = candidate
                break
        if action is None:
            continue

        arr = np.load(os.path.join(live_dir, filename)).astype(np.float32)
        T = arr.shape[0]
        # Slice long recordings into overlapping chunks
        if T <= slice_len:
            result[action].append(arr)
        else:
            for start in range(0, T - slice_len + 1, stride):
                result[action].append(arr[start:start + slice_len])

    for action in ACTIONS:
        print(f"  {action:12s}: {len(result[action]):3d} slices")
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Extract skeletons from videos
    cache_dir = os.path.join(os.path.dirname(args.videos), "skeleton_cache")
    data = extract_all_skeletons(args.videos, cache_dir, args.mp_complexity)

    # Merge live .npy data if provided
    if args.live_data and os.path.isdir(args.live_data):
        live_data = load_live_npy(args.live_data, slice_len=60, stride=30)
        for action in ACTIONS:
            n_before = len(data.get(action, []))
            data.setdefault(action, []).extend(live_data.get(action, []))
            n_after = len(data[action])
            if n_after > n_before:
                print(f"  {action:12s}: {n_before} → {n_after} (+{n_after - n_before} live)")

    # Split
    train_seqs, val_seqs = split_stratified(data, val_ratio=0.2)
    print(f"\nSplit: {len(train_seqs)} train, {len(val_seqs)} val")
    for action in ACTIONS:
        label = ACTIONS.index(action)
        n_train = sum(1 for _, l in train_seqs if l == label)
        n_val = sum(1 for _, l in val_seqs if l == label)
        print(f"  {action:12s}: train={n_train}, val={n_val}")

    # Datasets
    train_ds = ActionDataset(train_seqs, window_size=args.window, augment=True,
                             samples_per_seq=args.samples_per_seq)
    val_ds = ActionDataset(val_seqs, window_size=args.window, augment=False,
                           samples_per_seq=5)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # Model
    model = build_model(base_channel=args.base_channel, dropout=args.dropout)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: base_channel={args.base_channel}, {total_params:,} params")

    # Optimizer with warm-up
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        nesterov=True, weight_decay=args.weight_decay,
    )

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 4), T_mult=2,
        eta_min=args.lr * 0.01,
    )

    # Warm-up: linearly increase LR for first N epochs
    warmup_epochs = min(5, args.epochs // 10)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    print(f"\n{'=' * 75}")
    print(f"  Training: epochs={args.epochs}, window={args.window}, "
          f"base_ch={args.base_channel}, lr={args.lr}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 75}\n")

    for epoch in range(1, args.epochs + 1):
        # Warm-up LR
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            N, T, V, C = inputs.shape
            inputs = inputs.view(N, T, V * C)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * N
            train_correct += outputs.argmax(1).eq(targets).sum().item()
            train_total += N

        if epoch > warmup_epochs:
            scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        cls_correct = [0] * len(ACTIONS)
        cls_total = [0] * len(ACTIONS)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                N, T, V, C = inputs.shape
                inputs = inputs.view(N, T, V * C)
                preds = model(inputs).argmax(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += N
                for i in range(N):
                    lbl = targets[i].item()
                    cls_total[lbl] += 1
                    if preds[i].item() == lbl:
                        cls_correct[lbl] += 1

        train_acc = train_correct / max(1, train_total) * 100
        val_acc = val_correct / max(1, val_total) * 100
        avg_loss = train_loss / max(1, train_total)
        lr_now = optimizer.param_groups[0]["lr"]

        cls_str = " | ".join(
            f"{ACTIONS[i][:5]}:{cls_correct[i]/max(1,cls_total[i])*100:4.0f}%"
            for i in range(len(ACTIONS))
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " *BEST*" if is_best else ""
        print(
            f"Ep {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"train={train_acc:5.1f}% | val={val_acc:5.1f}% | "
            f"lr={lr_now:.5f} | {cls_str}{marker}"
        )

        # Early stopping
        if patience_counter >= 20 and epoch > 30:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for 20 epochs)")
            break

    # Save
    if best_state is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(best_state, args.output)
        print(f"\nBest model saved: {args.output}")
        print(f"Best val accuracy: {best_val_acc:.1f}%")
    else:
        print("\nWARNING: No model saved")

    # Final summary
    print(f"\n{'=' * 55}")
    print("Final per-class accuracy (best epoch):")
    for i, action in enumerate(ACTIONS):
        acc = cls_correct[i] / max(1, cls_total[i]) * 100
        print(f"  {action:12s}: {acc:5.1f}%")
    print(f"  {'Overall':12s}: {best_val_acc:5.1f}%")
    print(f"{'=' * 55}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train CTR-GCN from video dataset")
    parser.add_argument("--videos", "-v", required=True, help="Video dataset directory")
    parser.add_argument("--output", "-o", default="models/best_v2.pt")
    parser.add_argument("--base-channel", type=int, default=16)
    parser.add_argument("--window", "-w", type=int, default=30)
    parser.add_argument("--epochs", "-e", type=int, default=80)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--samples-per-seq", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mp-complexity", type=int, default=1)
    parser.add_argument("--live-data", default=None, help="Directory with .npy live recordings to mix in")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
