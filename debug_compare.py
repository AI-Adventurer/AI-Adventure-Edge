"""Diagnostic: compare original row2_ws pipeline vs AI_Adventure_Edge pipeline."""
import time
import numpy as np
import torch

# ── 1. Load model ──────────────────────────────────────────────
config_path = "models/config.yaml"
weights_path = "models/best.pt"

from adventure_game_jetson.inference.backends.pytorch_ctrgcn import (
    PyTorchCTRGCNBackend,
    load_model_from_config,
)
from adventure_game_jetson.inference.ctrgcn_runner import softmax_np

print("=== Loading model ===")
model_cpu = load_model_from_config(config_path, weights_path, torch.device("cpu"))
print(f"Model loaded on CPU. Parameters: {sum(p.numel() for p in model_cpu.parameters()):,}")

# ── 2. Create test input (20 frames of random skeleton) ───────
np.random.seed(42)
window = np.random.rand(20, 33, 3).astype(np.float32) * 0.5 + 0.25  # normalized 0.25~0.75

# ── 3. Original row2_ws method (torch, direct) ────────────────
print("\n=== Method A: Original row2_ws style (torch direct, CPU) ===")
arr_a = np.stack(list(window), axis=0).astype(np.float32)  # (T,V,3)
arr_a = arr_a.transpose(2, 0, 1)                           # (3,T,V)
arr_a = arr_a[None, :, :, :, None]                         # (1,3,T,V,1)
inp_a = torch.from_numpy(arr_a).to(torch.device("cpu"))

with torch.inference_mode():
    logits_a = model_cpu(inp_a)
    probs_a = torch.softmax(logits_a, dim=1)[0].detach().cpu().numpy()

labels = ["stand", "jump", "crouch", "push", "run_forward"]
print("  Logits:", logits_a[0].detach().cpu().numpy())
print("  Probs: ", {l: f"{p:.4f}" for l, p in zip(labels, probs_a)})
print(f"  Predicted: {labels[np.argmax(probs_a)]} ({probs_a.max():.4f})")

# ── 4. New AI_Adventure_Edge method (via backend) ─────────────
print("\n=== Method B: AI_Adventure_Edge PyTorchCTRGCNBackend (CPU) ===")
backend_cpu = PyTorchCTRGCNBackend(config_path, weights_path, "cpu")
logits_b = backend_cpu.infer(window)
probs_b = softmax_np(logits_b)
print("  Logits:", logits_b)
print("  Probs: ", {l: f"{p:.4f}" for l, p in zip(labels, probs_b)})
print(f"  Predicted: {labels[np.argmax(probs_b)]} ({probs_b.max():.4f})")
print(f"  Max diff vs Method A: {np.max(np.abs(probs_a - probs_b)):.2e}")

# ── 5. CUDA if available ──────────────────────────────────────
if torch.cuda.is_available():
    print("\n=== Method C: AI_Adventure_Edge PyTorchCTRGCNBackend (CUDA) ===")
    backend_gpu = PyTorchCTRGCNBackend(config_path, weights_path, "cuda")
    logits_c = backend_gpu.infer(window)
    probs_c = softmax_np(logits_c)
    print("  Logits:", logits_c)
    print("  Probs: ", {l: f"{p:.4f}" for l, p in zip(labels, probs_c)})
    print(f"  Predicted: {labels[np.argmax(probs_c)]} ({probs_c.max():.4f})")
    print(f"  Max diff vs Method A (CPU): {np.max(np.abs(probs_a - probs_c)):.2e}")

# ── 6. Speed benchmark ───────────────────────────────────────
print("\n=== Speed benchmark (50 iterations) ===")
N = 50

# CPU
t0 = time.perf_counter()
for _ in range(N):
    backend_cpu.infer(window)
cpu_ms = (time.perf_counter() - t0) / N * 1000
print(f"  CPU:  {cpu_ms:.1f} ms/inference")

if torch.cuda.is_available():
    # warmup
    for _ in range(5):
        backend_gpu.infer(window)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        backend_gpu.infer(window)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  CUDA: {gpu_ms:.1f} ms/inference")
    print(f"  → {'CPU' if cpu_ms < gpu_ms else 'CUDA'} is faster for this model")

# ── 7. Test with real-ish skeleton (standing pose) ────────────
print("\n=== Test with standing-like skeleton ===")
stand_skel = np.zeros((33, 3), dtype=np.float32)
# approximate standing: head on top, feet on bottom, centered
stand_skel[0] = [0.5, 0.15, 0.0]   # nose
stand_skel[11] = [0.45, 0.3, 0.0]  # left shoulder
stand_skel[12] = [0.55, 0.3, 0.0]  # right shoulder
stand_skel[13] = [0.4, 0.45, 0.0]  # left elbow
stand_skel[14] = [0.6, 0.45, 0.0]  # right elbow
stand_skel[15] = [0.4, 0.55, 0.0]  # left wrist
stand_skel[16] = [0.6, 0.55, 0.0]  # right wrist
stand_skel[23] = [0.47, 0.55, 0.0] # left hip
stand_skel[24] = [0.53, 0.55, 0.0] # right hip
stand_skel[25] = [0.47, 0.7, 0.0]  # left knee
stand_skel[26] = [0.53, 0.7, 0.0]  # right knee
stand_skel[27] = [0.47, 0.85, 0.0] # left ankle
stand_skel[28] = [0.53, 0.85, 0.0] # right ankle

# centralize (like the runtime does)
cx = (stand_skel[23, 0] + stand_skel[24, 0]) / 2.0
cy = (stand_skel[23, 1] + stand_skel[24, 1]) / 2.0
cz = (stand_skel[23, 2] + stand_skel[24, 2]) / 2.0
stand_skel[:, 0] -= cx
stand_skel[:, 1] -= cy
stand_skel[:, 2] -= cz

# Fill 20-frame window with the same standing skeleton
stand_window = np.tile(stand_skel, (20, 1, 1))  # (20, 33, 3)
logits_stand = backend_cpu.infer(stand_window)
probs_stand = softmax_np(logits_stand)
print("  Probs:", {l: f"{p:.4f}" for l, p in zip(labels, probs_stand)})
print(f"  Predicted: {labels[np.argmax(probs_stand)]} ({probs_stand.max():.4f})")

print("\nDone.")
