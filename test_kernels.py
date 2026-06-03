"""Standalone kernel test — validate NKI kernels produce correct output.

Runs on a single NeuronCore (no TP needed). Compares NKI kernel output
against PyTorch reference implementation.

Usage:
    python test_kernels.py   (on a Neuron instance)
"""

import os
import time
import math
import torch
import numpy as np

os.environ.setdefault("NEURON_RT_NUM_CORES", "1")

import torch_neuronx
from torch_neuronx.nki_hop import wrap_nki

from kernels.rope import qwen3_rope
from kernels.prefill_attention import prefill_gqa_flash_attention


# ═══════════════════════════════════════════════════════════════════════
# Test config
# ═══════════════════════════════════════════════════════════════════════
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
SEQ_LEN = 256  # 2 tiles to test multi-tile correctness
HALF_D = HEAD_DIM // 2

torch.manual_seed(42)
torch.neuron.set_device(0)
DEVICE = torch.device("neuron")


def check_close(name, nki_out, ref_out, rtol=1e-2, atol=1e-2):
    """Compare NKI output to reference."""
    nki_np = nki_out.float().cpu().numpy()
    ref_np = ref_out.float().cpu().numpy()
    max_diff = np.abs(nki_np - ref_np).max()
    mean_diff = np.abs(nki_np - ref_np).mean()
    match = np.allclose(nki_np, ref_np, rtol=rtol, atol=atol)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if not match:
        print(f"         Shapes: nki={nki_out.shape}, ref={ref_out.shape}")
        print(f"         NKI first 5: {nki_np.flatten()[:5]}")
        print(f"         Ref first 5: {ref_np.flatten()[:5]}")
    return match


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: RoPE kernel
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# TEST 0: nc_matmul semantics — determine if it's A@B or A^T@B
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 0: nc_matmul semantics")
print("=" * 60)

import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def test_matmul(a, b):
    P = 128
    out = nl.ndarray((P, P), dtype=a.dtype, buffer=nl.shared_hbm)
    a_sb = nl.ndarray((P, P), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sb, src=a)
    b_sb = nl.ndarray((P, P), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=b)
    out_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(out_psum, a_sb, b_sb)
    # PSUM -> SBUF f32 -> SBUF bf16 -> HBM
    out_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    out_f32[...] = nl.copy(out_psum, dtype=nl.float32)
    out_bf16 = nl.copy(out_f32, dtype=a.dtype)
    nisa.dma_copy(dst=out, src=out_bf16)
    return out

# Simple test: A = [[1,2],[3,4],...] padded to 128x128, B = identity
A = torch.zeros(128, 128, dtype=torch.bfloat16)
A[0, 0] = 1; A[0, 1] = 2; A[1, 0] = 3; A[1, 1] = 4
B = torch.eye(128, dtype=torch.bfloat16)

matmul_kernel = wrap_nki(test_matmul)
result = matmul_kernel(A.to(DEVICE), B.to(DEVICE)).cpu()

print(f"  A[0:2, 0:2] = [[1,2],[3,4]]")
print(f"  B = identity")
print(f"  nc_matmul(A, B)[0:2, 0:2] = {result[0:2, 0:2].float().numpy()}")
print(f"  If A@B:   expect [[1,2],[3,4]]")
print(f"  If A^T@B: expect [[1,3],[2,4]]")

if result[0, 1].item() == 2.0:
    print(f"  => nc_matmul = A @ B (no transpose)")
elif result[0, 1].item() == 3.0:
    print(f"  => nc_matmul = A^T @ B (transposes stationary)")
else:
    print(f"  => UNEXPECTED result")

print()
print("=" * 60)
print("TEST 1: RoPE kernel (qwen3_rope)")
print("=" * 60)

# Generate test data
x = torch.randn(SEQ_LEN, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)

# Generate cos/sin (half-dim, will be doubled by kernel)
inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
positions = torch.arange(SEQ_LEN, dtype=torch.float32)
freqs = torch.outer(positions, inv_freq)  # [seq_len, D/2]
cos_vals = freqs.cos()  # [seq_len, D/2]
sin_vals = freqs.sin()  # [seq_len, D/2]

# Kernel expects cos_sin as [seq_len, D] with first half cos, second half sin
cos_sin = torch.cat([cos_vals, sin_vals], dim=-1).float()  # [seq_len, D]

# PyTorch reference (rotate_half style)
def rope_ref(x, cos, sin):
    """Reference RoPE: x * cos_full + rotate_half(x) * sin_full."""
    cos_full = torch.cat([cos, cos], dim=-1)  # [seq, D]
    sin_full = torch.cat([sin, sin], dim=-1)  # [seq, D]
    x_f32 = x.float()
    x1, x2 = x_f32[..., :HALF_D], x_f32[..., HALF_D:]
    rotated = torch.cat([-x2, x1], dim=-1)
    out = x_f32 * cos_full.unsqueeze(1) + rotated * sin_full.unsqueeze(1)
    return out.to(torch.bfloat16)

ref_out = rope_ref(x, cos_vals, sin_vals)

# NKI kernel — tensors must be on Neuron device
print(f"  Input: x={x.shape}, cos_sin={cos_sin.shape}")
print(f"  Moving to Neuron device...")
x_dev = x.to(DEVICE)
cos_sin_dev = cos_sin.to(DEVICE)

print(f"  Running NKI kernel...")
t0 = time.time()

rope_kernel = wrap_nki(qwen3_rope)
nki_out = rope_kernel(x_dev, cos_sin_dev, num_heads=NUM_Q_HEADS, head_dim=HEAD_DIM)

t1 = time.time()
print(f"  NKI time: {t1-t0:.3f}s (includes compilation)")

nki_out_cpu = nki_out.cpu()
rope_pass = check_close("RoPE", nki_out_cpu, ref_out)

# Run again (cached)
t0 = time.time()
nki_out2 = rope_kernel(x_dev, cos_sin_dev, num_heads=NUM_Q_HEADS, head_dim=HEAD_DIM)
t1 = time.time()
print(f"  Cached time: {t1-t0:.4f}s")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Prefill Flash Attention
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Prefill Flash Attention (prefill_gqa_flash_attention)")
print("=" * 60)

# Generate Q, K, V — Q is [heads, seq, D], K is [heads, D, seq], V is [heads, seq, D]
q = torch.randn(NUM_Q_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
k = torch.randn(NUM_KV_HEADS, HEAD_DIM, SEQ_LEN, dtype=torch.bfloat16)
v = torch.randn(NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
identity = torch.eye(128, dtype=torch.bfloat16)
scale = 1.0 / math.sqrt(HEAD_DIM)

# Build full causal mask [seq_len, seq_len]: mask[i, j] = -inf if j > i
causal_mask = torch.triu(
    torch.full((SEQ_LEN, SEQ_LEN), float('-inf'), dtype=torch.bfloat16),
    diagonal=1
)

# PyTorch reference (causal attention with GQA)
def attention_ref(q, k, v, scale, num_q_heads, num_kv_heads):
    """Reference causal GQA attention.
    q: (num_q_heads, seq, D), k: (num_kv_heads, D, seq), v: (num_kv_heads, seq, D)
    """
    seq_len = q.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    out = torch.zeros(num_q_heads, seq_len, HEAD_DIM, dtype=torch.bfloat16)

    for q_h in range(num_q_heads):
        kv_h = q_h // gqa_ratio
        # Q: [seq, D]
        q_t = q[q_h].float()  # [seq, D]
        # K: [D, seq] -> [seq, D]
        k_t = k[kv_h].T.float()  # [seq, D]
        # V: [seq, D]
        v_t = v[kv_h].float()  # [seq, D]

        # scores = Q @ K^T
        scores = q_t @ k_t.T * scale  # [seq, seq]

        # Causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Output
        out[q_h] = (attn @ v_t).to(torch.bfloat16)

    return out

print(f"  Input: q={q.shape}, k={k.shape}, v={v.shape}")
print(f"  Computing reference...")
ref_attn_out = attention_ref(q, k, v, scale, NUM_Q_HEADS, NUM_KV_HEADS)

# Move to Neuron device
print(f"  Moving to Neuron device...")
q_dev = q.to(DEVICE)
k_dev = k.to(DEVICE)
v_dev = v.to(DEVICE)
id_dev = identity.to(DEVICE)
mask_dev = causal_mask.to(DEVICE)

print(f"  Running NKI kernel...")
t0 = time.time()

prefill_kernel = wrap_nki(prefill_gqa_flash_attention)
nki_attn_out = prefill_kernel(q_dev, k_dev, v_dev, id_dev, mask_dev, scale,
                               num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
                               head_dim=HEAD_DIM)

t1 = time.time()
print(f"  NKI time: {t1-t0:.3f}s (includes compilation)")

nki_attn_out_cpu = nki_attn_out.cpu()
attn_pass = check_close("Prefill Attention", nki_attn_out_cpu, ref_attn_out)

# Per-tile diff analysis
nki_np = nki_attn_out_cpu.float().numpy()
ref_np = ref_attn_out.float().numpy()
for tile_i in range(SEQ_LEN // 128):
    s, e = tile_i * 128, (tile_i + 1) * 128
    tile_diff = np.abs(nki_np[:, s:e, :] - ref_np[:, s:e, :]).max()
    print(f"    Tile {tile_i} (pos {s}-{e-1}): max_diff={tile_diff:.6f}")

# Diagnostic: check which ROWS have largest error (head 0, tile 0)
row_diffs = np.abs(nki_np[0, :128, :] - ref_np[0, :128, :]).max(axis=1)
worst_rows = np.argsort(row_diffs)[-5:]
print(f"    Worst rows in head0/tile0: {worst_rows} with diffs: {row_diffs[worst_rows]}")
print(f"    Best rows in head0/tile0: rows 0-4 diffs: {row_diffs[:5]}")

# Run again (cached)
t0 = time.time()
nki_attn_out2 = prefill_kernel(q_dev, k_dev, v_dev, id_dev, mask_dev, scale,
                                num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
                                head_dim=HEAD_DIM)
t1 = time.time()
print(f"  Cached time: {t1-t0:.4f}s")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  RoPE:             {'PASS' if rope_pass else 'FAIL'}")
print(f"  Prefill Attention: {'PASS' if attn_pass else 'FAIL'}")

if rope_pass and attn_pass:
    print("\n  All tests passed! Kernels ready to wire into model.py")
else:
    print("\n  Some tests FAILED. Fix before wiring in.")
