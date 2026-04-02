"""
Train CTR-GCN action recognition model from collected skeleton data.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson

    # Train from scratch on collected data:
    python tools/train_model.py --data data/collected

    # Fine-tune from existing weights:
    python tools/train_model.py --data data/collected --finetune models/best.pt

    # Custom settings:
    python tools/train_model.py --data data/collected --epochs 80 --window 40 --lr 0.005

Output: models/best.pt (overwrites existing model)
"""
import argparse
import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adventure_game_jetson.inference.model.ctrgcn import Model
from adventure_game_jetson.inference.graph.mediapipe_pose import Graph

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

# MediaPipe left-right joint swap pairs for mirror augmentation
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6),       # eyes
    (7, 8),                         # ears
    (9, 10),                        # mouth
    (11, 12),                       # shoulders
    (13, 14),                       # elbows
    (15, 16),                       # wrists
    (17, 18),                       # pinkies
    (19, 20),                       # index fingers
    (21, 22),                       # thumbs
    (23, 24),                       # hips
    (25, 26),                       # knees
    (27, 28),                       # ankles
    (29, 30),                       # heels
    (31, 32),                       # foot indices
]


# ── Data Augmentation ──────────────────────────────────────────


def augment_mirror(skeleton: np.ndarray) -> np.ndarray:
    """Left-right mirror: flip x-axis and swap left/right joints."""
    s = skeleton.copy()
    # Mirror x coordinate (raw MediaPipe x is in [0,1])
    s[:, :, 0] = 1.0 - s[:, :, 0]
    # Swap left-right joint pairs
    for l, r in MIRROR_PAIRS:
        s[:, l, :], s[:, r, :] = s[:, r, :].copy(), s[:, l, :].copy()
    return s


def augment_noise(skeleton: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to joint positions."""
    noise = np.random.randn(*skeleton.shape).astype(np.float32) * sigma
    return skeleton + noise


def augment_scale(skeleton: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """Random uniform scaling."""
    scale = np.random.uniform(lo, hi)
    return skeleton * scale


def augment_shift(skeleton: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    """Random spatial translation on x, y."""
    s = skeleton.copy()
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    s[:, :, 0] += dx
    s[:, :, 1] += dy
    return s


def augment_temporal_stretch(skeleton: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Random temporal stretch/compress via linear interpolation."""
    T, V, C = skeleton.shape
    factor = np.random.uniform(lo, hi)
    new_T = max(2, int(T * factor))
    old_t = np.linspace(0, T - 1, new_T, dtype=np.float32)
    indices = np.clip(old_t.astype(int), 0, T - 2)
    frac = old_t - indices
    out = skeleton[indices] * (1 - frac[:, None, None]) + skeleton[indices + 1] * frac[:, None, None]
    return out.astype(np.float32)


def augment_joint_dropout(skeleton: np.ndarray, drop_rate: float = 0.05) -> np.ndarray:
    """Randomly zero out some joints entirely."""
    s = skeleton.copy()
    T, V, C = s.shape
    mask = np.random.rand(V) > drop_rate
    s[:, ~mask, :] = 0.0
    return s


def preprocess_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Match inference preprocessing: clip to [0,1] then centralize to hip midpoint."""
    s = np.clip(skeleton, 0.0, 1.0)
    # Centralize around hip midpoint (joints 23, 24)
    if s.shape[1] >= 25:
        cx = (s[:, 23, 0] + s[:, 24, 0]) / 2.0  # (T,)
        cy = (s[:, 23, 1] + s[:, 24, 1]) / 2.0
        cz = (s[:, 23, 2] + s[:, 24, 2]) / 2.0
        s[:, :, 0] -= cx[:, None]
        s[:, :, 1] -= cy[:, None]
        s[:, :, 2] -= cz[:, None]
    return s


def interpolate_60fps(skeleton: np.ndarray) -> np.ndarray:
    """Interpolate to 60fps by inserting mid-frames (matching inference)."""
    T, V, C = skeleton.shape
    out = np.empty((T * 2 - 1, V, C), dtype=np.float32)
    out[0::2] = skeleton
    out[1::2] = (skeleton[:-1] + skeleton[1:]) * 0.5
    return out


# ── Dataset ────────────────────────────────────────────────────


class SkeletonDataset(Dataset):
    def __init__(
        self,
        sequences: list[tuple[np.ndarray, int]],
        window_size: int = 64,
        augment: bool = True,
        do_interpolate_60fps: bool = True,
        samples_per_sequence: int = 20,
    ):
        self.sequences = sequences  # list of (raw_skeleton (T,33,3), label_idx)
        self.window_size = window_size
        self.augment = augment
        self.do_interpolate_60fps = do_interpolate_60fps
        self.samples_per_sequence = samples_per_sequence

    def __len__(self):
        return len(self.sequences) * self.samples_per_sequence

    def __getitem__(self, idx):
        seq_idx = idx % len(self.sequences)
        raw, label = self.sequences[seq_idx]  # raw: (T, 33, 3)

        # Augmentation (before preprocessing, on raw coordinates)
        if self.augment:
            # Temporal stretch
            if random.random() < 0.5:
                raw = augment_temporal_stretch(raw, 0.8, 1.2)
            # Mirror
            if random.random() < 0.5:
                raw = augment_mirror(raw)
            # Noise
            if random.random() < 0.7:
                raw = augment_noise(raw, sigma=0.005)
            # Scale
            if random.random() < 0.5:
                raw = augment_scale(raw, 0.9, 1.1)
            # Shift
            if random.random() < 0.5:
                raw = augment_shift(raw, 0.04)
            # Joint dropout
            if random.random() < 0.3:
                raw = augment_joint_dropout(raw, 0.05)

        # Interpolate to 60fps (matching inference pipeline)
        if self.do_interpolate_60fps:
            raw = interpolate_60fps(raw)

        # Preprocess (clip + centralize, matching inference)
        processed = preprocess_skeleton(raw)

        # Random crop to window_size
        T = processed.shape[0]
        if T >= self.window_size:
            if self.augment:
                start = random.randint(0, T - self.window_size)
            else:
                start = (T - self.window_size) // 2  # center crop for val
            window = processed[start : start + self.window_size]
        else:
            # Pad by repeating last frame
            pad_len = self.window_size - T
            window = np.concatenate(
                [processed, np.tile(processed[-1:], (pad_len, 1, 1))], axis=0
            )

        # (window_size, 33, 3) → tensor
        tensor = torch.from_numpy(window.copy()).float()
        return tensor, label


# ── Training ───────────────────────────────────────────────────


def build_model(num_classes: int = 5, drop_out: float = 0.0) -> nn.Module:
    """Build CTR-GCN model with MediaPipe graph."""
    model = Model(
        num_class=num_classes,
        num_point=33,
        num_person=1,
        graph="adventure_game_jetson.inference.graph.mediapipe_pose.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
        drop_out=drop_out,
        adaptive=True,
    )
    return model


def load_data(data_dir: str) -> list[tuple[np.ndarray, int]]:
    """Load all .npy skeleton files from data directory."""
    sequences = []
    npy_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))

    if not npy_files:
        print(f"ERROR: No .npy files found in {data_dir}")
        sys.exit(1)

    for filename in npy_files:
        # Parse action name from filename prefix.
        # Supports: <action>_r00.npy, <action>_syn_0000.npy, <action>_yt_xxx.npy
        action_name = None
        for candidate in sorted(ACTIONS, key=len, reverse=True):
            if filename.startswith(candidate + "_") or filename == candidate + ".npy":
                action_name = candidate
                break
        if action_name is None:
            continue

        label = ACTIONS.index(action_name)
        data = np.load(os.path.join(data_dir, filename)).astype(np.float32)
        sequences.append((data, label))
        print(f"  Loaded: {filename:30s} label={label} ({action_name:12s}) frames={data.shape[0]}")

    print(f"\nTotal: {len(sequences)} sequences, "
          f"{sum(s[0].shape[0] for s in sequences)} frames")

    # Check class balance
    counts = {}
    for _, label in sequences:
        counts[label] = counts.get(label, 0) + 1
    print("Class distribution:")
    for i, action in enumerate(ACTIONS):
        print(f"  {action:12s}: {counts.get(i, 0)} sequences")

    return sequences


def split_data(
    sequences: list[tuple[np.ndarray, int]],
    val_ratio: float = 0.2,
) -> tuple[list, list]:
    """Stratified split into train and validation sets."""
    by_class: dict[int, list] = {}
    for seq, label in sequences:
        by_class.setdefault(label, []).append((seq, label))

    train, val = [], []
    for label, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def train(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data from: {args.data}")
    sequences = load_data(args.data)
    train_seqs, val_seqs = split_data(sequences, val_ratio=args.val_ratio)
    print(f"\nSplit: {len(train_seqs)} train, {len(val_seqs)} val")

    # Datasets
    train_ds = SkeletonDataset(
        train_seqs,
        window_size=args.window,
        augment=True,
        do_interpolate_60fps=args.interpolate,
        samples_per_sequence=args.samples_per_seq,
    )
    val_ds = SkeletonDataset(
        val_seqs,
        window_size=args.window,
        augment=False,
        do_interpolate_60fps=args.interpolate,
        samples_per_sequence=10,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch: {len(val_loader)}")

    # Build model
    model = build_model(num_classes=len(ACTIONS), drop_out=args.dropout)

    if args.finetune:
        print(f"\nFine-tuning from: {args.finetune}")
        state = torch.load(args.finetune, map_location=device, weights_only=True)
        cleaned = {}
        for k, v in state.items():
            cleaned[k.replace("module.", "")] = v
        # Load with strict=False in case of class count mismatch
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  Missing keys (will be random-init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
    else:
        print("\nTraining from scratch")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    # LR scheduler: cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, args.epochs // 3),
        T_mult=2,
        eta_min=args.lr * 0.01,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    output_path = args.output

    print(f"\n{'=' * 70}")
    print(f"  Training: {args.epochs} epochs, window={args.window}, lr={args.lr}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Model expects (N, C, T, V, M) but forward() handles (N, T, V*C) too
            # Our input: (N, T, 33, 3) — use the model's built-in reshape
            N, T, V, C = inputs.shape
            inputs = inputs.view(N, T, V * C)  # (N, T, 99)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * N
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += N

        scheduler.step()

        train_acc = train_correct / max(1, train_total) * 100
        avg_loss = train_loss / max(1, train_total)

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = [0] * len(ACTIONS)
        class_total = [0] * len(ACTIONS)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                N, T, V, C = inputs.shape
                inputs = inputs.view(N, T, V * C)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += N

                for i in range(N):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

        val_acc = val_correct / max(1, val_total) * 100
        lr_now = optimizer.param_groups[0]["lr"]

        # Per-class accuracy
        class_acc_str = " | ".join(
            f"{ACTIONS[i][:5]}:{class_correct[i]/max(1,class_total[i])*100:4.0f}%"
            for i in range(len(ACTIONS))
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        marker = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"train={train_acc:5.1f}% | "
            f"val={val_acc:5.1f}% | "
            f"lr={lr_now:.5f} | "
            f"{class_acc_str}{marker}"
        )

    # Save best model
    if best_model_state is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(best_model_state, output_path)
        print(f"\nBest model saved to: {output_path}")
        print(f"Best validation accuracy: {best_val_acc:.1f}%")
    else:
        print("\nWARNING: No model saved (no improvement over baseline)")

    # Final per-class breakdown
    print(f"\n{'=' * 50}")
    print("Final validation per-class accuracy:")
    for i, action in enumerate(ACTIONS):
        total = class_total[i]
        correct = class_correct[i]
        acc = correct / max(1, total) * 100
        print(f"  {action:12s}: {correct:3d}/{total:3d} = {acc:5.1f}%")
    print(f"  {'Overall':12s}: {val_correct:3d}/{val_total:3d} = {best_val_acc:5.1f}%")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Train CTR-GCN action recognition model")
    parser.add_argument("--data", "-d", required=True, help="Data directory with .npy files")
    parser.add_argument("--output", "-o", default="models/best.pt", help="Output model path")
    parser.add_argument("--finetune", "-f", default=None, help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--epochs", "-e", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0004, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--window", "-w", type=int, default=64, help="Training window size (frames)")
    parser.add_argument("--interpolate", action=argparse.BooleanOptionalAction, default=True,
                        help="Interpolate 30fps→60fps (match inference)")
    parser.add_argument("--samples-per-seq", type=int, default=20,
                        help="Random crops per sequence per epoch")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
