"""RoPE NKI kernel for Qwen3-VL decoder — rotate_half style.

Adapted from rolling-forcing causal_rope_rotation.
Qwen3-VL uses rotate_half: (-x2, x1) pattern (not interleaved pairs).

IO layouts:
  - x:       (seq_len, num_heads, head_dim) bf16
  - cos_sin: (seq_len, head_dim) float32
             columns [0, D/2): cos values
             columns [D/2, D): sin values
  - out:     (seq_len, num_heads, head_dim) bf16

seq_len must be a multiple of 128 (pad at call site).

Uses nl.sequential_range for outer loop (>8 tiles causes SBUF corruption
with affine_range due to software pipelining — see rolling-forcing notes).
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def qwen3_rope(x, cos_sin, num_heads=4, head_dim=128):
    """Apply RoPE with rotate_half pattern to Q or K tensor.

    rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])
    out = x * cos + rotate_half(x) * sin

    Args:
        x:       (seq_len, num_heads, head_dim) bf16
        cos_sin: (seq_len, head_dim) float32
                 First D/2 columns: cos, last D/2 columns: sin
        num_heads: number of heads (4 for Q per rank, 1 for K per rank)
        head_dim: 128

    Returns:
        out: (seq_len, num_heads, head_dim) bf16
    """
    seq_len = x.shape[0]
    N = num_heads
    D = head_dim
    half_D = D // 2
    P = nl.tile_size.pmax  # 128

    assert seq_len % P == 0
    num_tiles = seq_len // P

    out = nl.ndarray((seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P

        # Load cos_sin: [P, D] — first half is cos, second half is sin
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, half_D)]      # [P, D/2]
        sin_tile = cs_sb[:, nl.ds(half_D, half_D)]  # [P, D/2]

        # Expand cos/sin to full head_dim: cat(cos, cos) and cat(sin, sin)
        cos_full = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        sin_full = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        cos_full[:, nl.ds(0, half_D)] = cos_tile
        cos_full[:, nl.ds(half_D, half_D)] = cos_tile
        sin_full[:, nl.ds(0, half_D)] = sin_tile
        sin_full[:, nl.ds(half_D, half_D)] = sin_tile

        # Load x tile: [P, N, D]
        x_sb = nl.load(x[nl.ds(ts, P), :, :])

        out_sb = nl.ndarray((P, N, D), dtype=x.dtype, buffer=nl.sbuf)

        for n in nl.affine_range(N):
            xh = x_sb[:, n, :]  # [P, D] bf16
            xh_f32 = nl.copy(xh, dtype=nl.float32)

            # x * cos
            x_cos = nl.multiply(xh_f32, cos_full)

            # rotate_half(x): cat(-x[D/2:], x[:D/2])
            x_rot = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            # First half of output = -x[D/2:]
            x_rot[:, nl.ds(0, half_D)] = nl.multiply(
                xh_f32[:, nl.ds(half_D, half_D)], -1.0)
            # Second half of output = x[:D/2]
            x_rot[:, nl.ds(half_D, half_D)] = xh_f32[:, nl.ds(0, half_D)]

            # rotate_half(x) * sin
            x_sin = nl.multiply(x_rot, sin_full)

            # out = x*cos + rotate_half(x)*sin
            result = nl.add(x_cos, x_sin)
            out_sb[:, n, :] = nl.copy(result, dtype=x.dtype)

        nl.store(out[nl.ds(ts, P), :, :], out_sb)

    return out
