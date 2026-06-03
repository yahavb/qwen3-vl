"""Prefill flash attention NKI kernel for Qwen3-VL decoder.

Causal self-attention with online softmax, GQA, tiled over seq_len.
Adapted from rolling-forcing wan_flash_self_attn.

For Qwen3-VL-8B TP-8:
  - num_q_heads = 4, num_kv_heads = 1, head_dim = 128
  - seq_len padded to multiple of 128

IO layouts (all bf16):
  - q:      (num_q_heads, seq_len, head_dim) — standard layout
  - k:      (num_kv_heads, head_dim, seq_len) — transposed for K matmul
  - v:      (num_kv_heads, seq_len, head_dim)
  - mask:   (seq_len, seq_len) — full causal mask: 0 valid, -inf masked
            mask[i, j] = 0 if j <= i, else -inf
  - identity: (128, 128) — for transpose trick
  - out:    (num_q_heads, seq_len, head_dim)

seq_len MUST be multiple of 128.
mask is built by caller for each Q tile group (simplifies kernel).
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def prefill_gqa_flash_attention(q, k, v, identity, mask, softmax_scale,
                                 num_q_heads=4, num_kv_heads=1, head_dim=128):
    """Flash causal self-attention for prefill with GQA.

    Args:
        q:        (num_q_heads, seq_len, D) bf16
        k:        (num_kv_heads, D, seq_len) bf16
        v:        (num_kv_heads, seq_len, D) bf16
        identity: (128, 128) bf16
        mask:     (seq_len, seq_len) bf16 — full causal mask (0 valid, -inf future)
        softmax_scale: float
        num_q_heads: int
        num_kv_heads: int
        head_dim: int

    Returns:
        out: (num_q_heads, seq_len, D) bf16
    """
    seq_len = q.shape[1]
    P = nl.tile_size.pmax  # 128
    D = head_dim
    num_tiles = seq_len // P
    gqa_ratio = num_q_heads // num_kv_heads

    out = nl.ndarray((num_q_heads, seq_len, D), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity into SBUF
    id_sbuf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for kv_h in range(num_kv_heads):
        # Load full K: [D, seq_len] — stays in SBUF for all Q heads in this GQA group
        k_full = nl.ndarray((D, seq_len), dtype=k.dtype, buffer=nl.sbuf)
        for ti in range(num_tiles):
            nisa.dma_copy(dst=k_full[:, nl.ds(ti * P, P)],
                          src=k[kv_h, :, nl.ds(ti * P, P)])

        # Load full V: [P, num_tiles, D] tiles
        v_tiles = nl.ndarray((P, num_tiles, D), dtype=v.dtype, buffer=nl.sbuf)
        for ti in range(num_tiles):
            nisa.dma_copy(dst=v_tiles[:, ti, :],
                          src=v[kv_h, nl.ds(ti * P, P), :])


        for q_offset in range(gqa_ratio):
            q_h = kv_h * gqa_ratio + q_offset

            for grp_i in range(num_tiles):
                # Load Q tile: [P, D] — Q is (heads, seq_len, D)
                q_tile = nl.ndarray((P, D), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile, src=q[q_h, nl.ds(grp_i * P, P), :])

                # Load mask rows for this Q group: mask[grp_i*P:(grp_i+1)*P, :]
                mask_grp = nl.ndarray((P, seq_len), dtype=nl.float32, buffer=nl.sbuf)
                for mti in range(num_tiles):
                    mask_tile = nl.ndarray((P, P), dtype=mask.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=mask_tile, src=mask[nl.ds(grp_i * P, P), nl.ds(mti * P, P)])
                    mask_grp[:, nl.ds(mti * P, P)] = nl.copy(mask_tile, dtype=nl.float32)

                # Online softmax state
                r_max = nl.full((P, 1), fill_value=float('-inf'), dtype=nl.float32)
                r_sum = nl.zeros((P, 1), dtype=nl.float32)
                pv_acc = nl.zeros((P, D), dtype=nl.float32)

                # Tile over K
                for k_ti in range(num_tiles):
                    # Scores = Q[P,D] @ K_slice[D,P] = [P,P]
                    # nc_matmul: out[P,F] = stat[P,C] @ mov[C,F]
                    # stat = q_tile[P, D] (partition=P=128, contract=D=128)
                    # mov = k_slice[D, P] (contract=D=128, free=P=128)
                    k_slice = k_full[:, nl.ds(k_ti * P, P)]
                    score_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(score_psum, q_tile, k_slice)
                    scores = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    scores[...] = nl.copy(score_psum, dtype=nl.float32)
                    scores = nl.multiply(scores, softmax_scale)

                    # Add causal mask slice for this K tile
                    mask_slice = mask_grp[:, nl.ds(k_ti * P, P)]
                    scores = nl.add(scores, mask_slice)

                    # Online softmax
                    tile_max = nl.max(scores, axis=1, keepdims=True)  # [P, 1]
                    old_max = nl.copy(r_max)
                    new_max = nl.maximum(old_max, tile_max)
                    corr = nl.exp(nl.subtract(old_max, new_max))
                    r_max[...] = new_max

                    shifted = nl.subtract(scores, new_max)
                    exp_scores = nl.exp(shifted)
                    tile_sum = nl.sum(exp_scores, axis=1, keepdims=True)

                    old_sum = nl.copy(r_sum)
                    r_sum[...] = nl.add(nl.multiply(old_sum, corr), tile_sum)

                    # PV: transpose exp_scores[P,P] then matmul with V[P,D]
                    exp_bf16 = nl.copy(exp_scores, dtype=nl.bfloat16)
                    exp_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(exp_T_psum, exp_bf16, id_sbuf)
                    exp_T_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    exp_T_f32[...] = nl.copy(exp_T_psum, dtype=nl.float32)
                    exp_T = nl.copy(exp_T_f32, dtype=nl.bfloat16)

                    v_tile = v_tiles[:, k_ti, :]
                    pv_psum = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv_psum, exp_T, v_tile)
                    pv_tile = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                    pv_tile[...] = nl.copy(pv_psum, dtype=nl.float32)

                    # Correct and accumulate
                    pv_acc[...] = nl.add(nl.multiply(pv_acc, corr), pv_tile)

                # Normalize and store
                rcp = nl.reciprocal(r_sum)
                result = nl.multiply(pv_acc, rcp)
                result_bf16 = nl.copy(result, dtype=q.dtype)
                nisa.dma_copy(dst=out[q_h, nl.ds(grp_i * P, P), :],
                              src=result_bf16)

    return out
