"""
Generate synthetic MediaPipe skeleton sequences for pre-training CTR-GCN.

Creates procedurally animated skeleton data for 5 action classes.
Motion patterns are designed to match MediaPipe Pose output characteristics
(normalized [0,1] coordinates, 33 joints, z as relative depth).

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson
    python tools/generate_synthetic.py --output data/synthetic --sequences-per-class 200

Then train:
    python tools/train_model.py --data data/synthetic --epochs 30 --window 64
"""
import argparse
import os
import sys

import numpy as np

ACTIONS = ["stand", "jump", "crouch", "push", "run_forward"]

# ── Base skeleton definition (MediaPipe 33 joints, front-facing) ──────
# Coordinates are normalized [0,1] image space: x=left-right, y=top-bottom
# z is relative depth (negative = closer to camera, as on Jetson ARM)

def make_base_skeleton() -> np.ndarray:
    """Create a natural standing pose (33, 3) in MediaPipe coordinates."""
    s = np.zeros((33, 3), dtype=np.float32)

    # Head / face landmarks
    s[0]  = [0.500, 0.200, -0.10]  # nose
    s[1]  = [0.490, 0.185, -0.08]  # left eye inner
    s[2]  = [0.480, 0.183, -0.07]  # left eye
    s[3]  = [0.470, 0.185, -0.06]  # left eye outer
    s[4]  = [0.510, 0.185, -0.08]  # right eye inner
    s[5]  = [0.520, 0.183, -0.07]  # right eye
    s[6]  = [0.530, 0.185, -0.06]  # right eye outer
    s[7]  = [0.460, 0.200, -0.05]  # left ear
    s[8]  = [0.540, 0.200, -0.05]  # right ear
    s[9]  = [0.490, 0.215, -0.09]  # mouth left
    s[10] = [0.510, 0.215, -0.09]  # mouth right

    # Upper body
    s[11] = [0.430, 0.320, -0.05]  # left shoulder
    s[12] = [0.570, 0.320, -0.05]  # right shoulder
    s[13] = [0.400, 0.450, -0.03]  # left elbow
    s[14] = [0.600, 0.450, -0.03]  # right elbow
    s[15] = [0.400, 0.560, -0.01]  # left wrist
    s[16] = [0.600, 0.560, -0.01]  # right wrist

    # Hands (close to wrists)
    s[17] = [0.395, 0.575, -0.01]  # left pinky
    s[18] = [0.605, 0.575, -0.01]  # right pinky
    s[19] = [0.398, 0.580, -0.01]  # left index
    s[20] = [0.602, 0.580, -0.01]  # right index
    s[21] = [0.405, 0.565,  0.00]  # left thumb
    s[22] = [0.595, 0.565,  0.00]  # right thumb

    # Lower body
    s[23] = [0.470, 0.560, -0.02]  # left hip
    s[24] = [0.530, 0.560, -0.02]  # right hip
    s[25] = [0.470, 0.700, -0.01]  # left knee
    s[26] = [0.530, 0.700, -0.01]  # right knee
    s[27] = [0.470, 0.860,  0.00]  # left ankle
    s[28] = [0.530, 0.860,  0.00]  # right ankle
    s[29] = [0.465, 0.875,  0.02]  # left heel
    s[30] = [0.535, 0.875,  0.02]  # right heel
    s[31] = [0.470, 0.880, -0.02]  # left foot index
    s[32] = [0.530, 0.880, -0.02]  # right foot index

    return s


# ── Joint groups for applying motion ──────────────────────────────

HEAD = list(range(0, 11))
L_ARM = [11, 13, 15, 17, 19, 21]
R_ARM = [12, 14, 16, 18, 20, 22]
L_LEG = [23, 25, 27, 29, 31]
R_LEG = [24, 26, 28, 30, 32]
TORSO = [11, 12, 23, 24]
ALL_JOINTS = list(range(33))
BODY = list(range(33))  # everything moves together for global transforms


def add_noise(skeleton: np.ndarray, sigma: float = 0.003) -> np.ndarray:
    """Add per-frame per-joint Gaussian noise."""
    return skeleton + np.random.randn(*skeleton.shape).astype(np.float32) * sigma


def randomize_base(base: np.ndarray) -> np.ndarray:
    """Slight random variation of base pose (body proportions, position)."""
    s = base.copy()
    # Random horizontal shift
    dx = np.random.uniform(-0.1, 0.1)
    s[:, 0] += dx
    # Random vertical shift (taller/shorter person)
    dy = np.random.uniform(-0.05, 0.05)
    s[:, 1] += dy
    # Random scale (body size)
    scale = np.random.uniform(0.85, 1.15)
    center_x = (s[23, 0] + s[24, 0]) / 2
    center_y = (s[23, 1] + s[24, 1]) / 2
    s[:, 0] = center_x + (s[:, 0] - center_x) * scale
    s[:, 1] = center_y + (s[:, 1] - center_y) * scale
    # Random z offset
    s[:, 2] += np.random.uniform(-0.1, 0.1)
    return s


# ── Motion generators per action ──────────────────────────────────


def gen_stand(n_frames: int) -> np.ndarray:
    """Standing still with natural micro-movements (sway, breathing)."""
    base = randomize_base(make_base_skeleton())
    seq = np.tile(base, (n_frames, 1, 1))

    # Subtle breathing (torso expands/contracts)
    breath_freq = np.random.uniform(0.1, 0.3)
    breath_amp = np.random.uniform(0.002, 0.005)
    t = np.arange(n_frames, dtype=np.float32)
    breath = np.sin(2 * np.pi * breath_freq * t / 30.0) * breath_amp

    for j in [11, 12]:  # shoulders
        seq[:, j, 1] += breath
    for j in HEAD:
        seq[:, j, 1] += breath * 0.5

    # Subtle sway
    sway_freq = np.random.uniform(0.05, 0.15)
    sway_amp = np.random.uniform(0.002, 0.008)
    sway = np.sin(2 * np.pi * sway_freq * t / 30.0) * sway_amp
    seq[:, :, 0] += sway[:, None]

    return add_noise(seq, sigma=0.003)


def gen_jump(n_frames: int) -> np.ndarray:
    """Repeated jumps: crouch down slightly, launch up, land."""
    base = randomize_base(make_base_skeleton())
    seq = np.tile(base, (n_frames, 1, 1))
    t = np.arange(n_frames, dtype=np.float32)

    # Jump cycle parameters
    jump_freq = np.random.uniform(0.8, 1.5)  # jumps per second
    jump_height = np.random.uniform(0.06, 0.15)
    arm_raise = np.random.uniform(0.05, 0.12)

    cycle = (t / 30.0) * jump_freq * 2 * np.pi
    # Asymmetric jump: quick up, slower down
    raw = np.sin(cycle)
    # Only take the positive part (airborne) and a slight negative (crouch prep)
    jump_y = np.where(raw > 0, -raw * jump_height, raw * jump_height * 0.3)

    # Move whole body up during jump
    seq[:, :, 1] += jump_y[:, None]

    # Arms go up during jump
    arm_motion = np.clip(-raw, 0, 1) * arm_raise
    for j in L_ARM + R_ARM:
        seq[:, j, 1] -= arm_motion  # arms go up (lower y)
    # Arms go out slightly
    for j in L_ARM:
        seq[:, j, 0] -= arm_motion * 0.3
    for j in R_ARM:
        seq[:, j, 0] += arm_motion * 0.3

    # Knees bend during prep phase
    prep = np.clip(raw * 0.5, -1, 0) * 0.03
    for j in [25, 26]:  # knees
        seq[:, j, 1] += prep  # knees go down slightly
        seq[:, j, 0] += np.where(j == 25, -1, 1) * prep * 0.3

    # Z motion: closer to camera when airborne
    z_motion = np.where(raw > 0, -raw * 0.05, 0)
    seq[:, :, 2] += z_motion[:, None]

    return add_noise(seq, sigma=0.004)


def gen_crouch(n_frames: int) -> np.ndarray:
    """Repeated crouch/squat motions: hips drop, knees bend, torso leans."""
    base = randomize_base(make_base_skeleton())
    seq = np.tile(base, (n_frames, 1, 1))
    t = np.arange(n_frames, dtype=np.float32)

    # Crouch cycle
    freq = np.random.uniform(0.4, 1.0)
    depth = np.random.uniform(0.08, 0.18)

    cycle = (t / 30.0) * freq * 2 * np.pi
    crouch = (1 - np.cos(cycle)) / 2.0  # smooth 0→1→0

    # Hips drop
    for j in [23, 24]:
        seq[:, j, 1] += crouch * depth

    # Knees bend outward and forward
    for j in [25, 26]:
        seq[:, j, 1] += crouch * depth * 0.6
        seq[:, j, 2] -= crouch * 0.04  # knees come forward (closer to camera)
        side = -1 if j == 25 else 1
        seq[:, j, 0] += side * crouch * 0.02  # knees go outward

    # Ankles stay mostly in place (feet planted)
    for j in [27, 28, 29, 30, 31, 32]:
        seq[:, j, 1] += crouch * depth * 0.1  # slight down

    # Upper body leans forward slightly
    lean = crouch * 0.03
    for j in HEAD + [11, 12]:
        seq[:, j, 1] += crouch * depth * 0.5  # head drops
        seq[:, j, 2] -= lean  # lean forward

    # Arms come forward for balance
    for j in L_ARM + R_ARM:
        seq[:, j, 1] += crouch * depth * 0.3
        seq[:, j, 2] -= crouch * 0.03

    return add_noise(seq, sigma=0.004)


def gen_push(n_frames: int) -> np.ndarray:
    """Pushing motion: arms extend forward, body leans, steps forward."""
    base = randomize_base(make_base_skeleton())
    seq = np.tile(base, (n_frames, 1, 1))
    t = np.arange(n_frames, dtype=np.float32)

    # Push cycle
    freq = np.random.uniform(0.5, 1.2)
    reach = np.random.uniform(0.06, 0.14)

    cycle = (t / 30.0) * freq * 2 * np.pi
    push = (1 - np.cos(cycle)) / 2.0  # 0→1→0

    # Arms extend forward (z = closer to camera = more negative)
    for j in [15, 16, 17, 18, 19, 20, 21, 22]:  # wrists + hands
        seq[:, j, 2] -= push * 0.15  # push toward camera
        seq[:, j, 1] -= push * 0.05  # arms rise slightly

    # Elbows extend
    for j in [13, 14]:
        seq[:, j, 2] -= push * 0.08
        seq[:, j, 1] -= push * 0.03

    # Body leans forward
    lean = push * 0.03
    for j in HEAD + [11, 12]:
        seq[:, j, 2] -= lean
        seq[:, j, 1] += lean * 0.5

    # One leg steps forward
    step_leg = np.random.choice([0, 1])
    step_joints = L_LEG if step_leg == 0 else R_LEG
    for j in step_joints:
        seq[:, j, 2] -= push * 0.03

    # Slight hip rotation
    hip_twist = push * 0.01
    seq[:, 23, 2] -= hip_twist  # left hip forward
    seq[:, 24, 2] += hip_twist  # right hip back

    # Z depth change for the whole body
    seq[:, :, 2] -= push[:, None] * 0.02

    return add_noise(seq, sigma=0.004)


def gen_run_forward(n_frames: int) -> np.ndarray:
    """Running in place: alternating arm/leg swing, body bounce."""
    base = randomize_base(make_base_skeleton())
    seq = np.tile(base, (n_frames, 1, 1))
    t = np.arange(n_frames, dtype=np.float32)

    # Run cycle (faster than other actions)
    freq = np.random.uniform(1.5, 3.0)  # steps per second
    stride_size = np.random.uniform(0.04, 0.10)

    cycle = (t / 30.0) * freq * 2 * np.pi

    # Vertical bounce (body goes up/down with each step)
    bounce = np.abs(np.sin(cycle)) * 0.03
    seq[:, :, 1] -= bounce[:, None]  # whole body bounces up

    # Alternating leg motion
    leg_swing = np.sin(cycle) * stride_size
    # Left leg
    seq[:, 25, 1] -= np.clip(leg_swing, 0, None)  # knee up
    seq[:, 25, 2] -= np.clip(leg_swing, 0, None) * 0.5  # knee forward
    seq[:, 27, 1] -= np.clip(leg_swing, 0, None) * 0.8
    seq[:, 27, 2] -= np.clip(leg_swing, 0, None) * 0.3
    # Right leg (opposite phase)
    seq[:, 26, 1] -= np.clip(-leg_swing, 0, None)
    seq[:, 26, 2] -= np.clip(-leg_swing, 0, None) * 0.5
    seq[:, 28, 1] -= np.clip(-leg_swing, 0, None) * 0.8
    seq[:, 28, 2] -= np.clip(-leg_swing, 0, None) * 0.3

    # Feet follow legs
    for foot, knee in [(29, 25), (31, 25), (30, 26), (32, 26)]:
        seq[:, foot, 1] = seq[:, knee, 1] + (base[foot, 1] - base[knee, 1]) * 0.9
        seq[:, foot, 2] = seq[:, knee, 2]

    # Alternating arm swing (opposite to legs)
    arm_swing = np.sin(cycle) * stride_size * 0.7
    # Left arm swings forward when right leg is forward
    for j in [13, 15, 17, 19, 21]:
        seq[:, j, 2] -= arm_swing * 0.5  # forward/back
        seq[:, j, 1] -= arm_swing * 0.3  # up/down
    for j in [14, 16, 18, 20, 22]:
        seq[:, j, 2] += arm_swing * 0.5
        seq[:, j, 1] += arm_swing * 0.3

    # Arms bend more during run
    for j in [13, 14]:  # elbows
        seq[:, j, 1] -= 0.03  # elbows higher

    # Slight body lean forward
    for j in HEAD + [11, 12]:
        seq[:, j, 2] -= 0.02

    # Body sway (torso rotates slightly)
    sway = np.sin(cycle) * 0.01
    seq[:, :, 0] += sway[:, None] * 0.3

    return add_noise(seq, sigma=0.005)


# ── Generator dispatch ─────────────────────────────────────────

GENERATORS = {
    "stand": gen_stand,
    "jump": gen_jump,
    "crouch": gen_crouch,
    "push": gen_push,
    "run_forward": gen_run_forward,
}


# ── Mirror augmentation ────────────────────────────────────────

MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    s = seq.copy()
    s[:, :, 0] = 1.0 - s[:, :, 0]
    for l, r in MIRROR_PAIRS:
        s[:, l, :], s[:, r, :] = s[:, r, :].copy(), s[:, l, :].copy()
    return s


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic skeleton data")
    parser.add_argument("--output", "-o", default="data/synthetic")
    parser.add_argument("--sequences-per-class", "-n", type=int, default=200)
    parser.add_argument("--min-frames", type=int, default=60, help="Min frames per sequence (at 30fps)")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames per sequence (at 30fps)")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Also generate mirrored versions (doubles dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    total = 0
    for action in ACTIONS:
        gen_fn = GENERATORS[action]
        print(f"\n{'=' * 50}")
        print(f"  Generating: {action.upper()} ({args.sequences_per_class} sequences)")
        print(f"{'=' * 50}")

        for i in range(args.sequences_per_class):
            n_frames = np.random.randint(args.min_frames, args.max_frames + 1)
            seq = gen_fn(n_frames)  # (T, 33, 3)

            # Clip to valid range (MediaPipe outputs [0,1] for x,y)
            seq[:, :, 0] = np.clip(seq[:, :, 0], 0.0, 1.0)
            seq[:, :, 1] = np.clip(seq[:, :, 1], 0.0, 1.0)
            # z can be negative (depth relative to hip)
            seq[:, :, 2] = np.clip(seq[:, :, 2], -0.5, 0.5)

            filename = f"{action}_syn_{i:04d}.npy"
            np.save(os.path.join(args.output, filename), seq)
            total += 1

            # Mirror version
            if args.mirror:
                m_seq = mirror_sequence(seq)
                m_filename = f"{action}_syn_{i:04d}_mirror.npy"
                np.save(os.path.join(args.output, m_filename), m_seq)
                total += 1

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{args.sequences_per_class} generated")

        count = args.sequences_per_class * (2 if args.mirror else 1)
        print(f"  {action}: {count} sequences")

    # Write labels
    labels_path = os.path.join(args.output, "labels.txt")
    with open(labels_path, "w") as f:
        for i, action in enumerate(ACTIONS):
            f.write(f"{i} {action}\n")

    print(f"\n{'=' * 60}")
    print(f"  DONE! Generated {total} synthetic sequences")
    print(f"  Output: {args.output}/")
    print(f"  Labels: {labels_path}")
    print(f"\n  Next steps:")
    print(f"    1. Pre-train:  python tools/train_model.py --data {args.output} --epochs 30")
    print(f"    2. Collect real data: python tools/collect_data.py --output data/collected --rounds 1")
    print(f"    3. Fine-tune:  python tools/train_model.py --data data/collected --finetune models/best.pt --epochs 40")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
