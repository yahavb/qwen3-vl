"""Prefill flash attention NKI kernel for Qwen3-VL decoder.

Causal self-attention for the prefill pass (seq_q = seq_k = prompt_len).
Adapted from rolling-forcing wan_flash_self_attn with:
  - GQA support (num_q_heads != num_kv_heads)
  - Causal masking built-in (lower-triangular)
  - head_dim=128 fixed

For Qwen3-VL-8B TP-8:
  - num_q_heads = 4 per rank
  - num_kv_heads = 1 per rank
  - head_dim = 128
  - seq_len padded to multiple of 128

IO layouts (all bf16):
  - q:   (num_q_heads, head_dim, seq_len) — Q after RoPE
  - k:   (num_kv_heads, head_dim, seq_len) — K after RoPE
  - v:   (num_kv_heads, seq_len, head_dim) — V
  - out: (num_q_heads, seq_len, head_dim)

seq_len MUST be a multiple of 128. Pad at call site.
Causal mask is generated internally (no mask tensor needed).
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def prefill_gqa_flash_attention(q, k, v, identity, softmax_scale,
                                 num_q_heads=4, num_kv_heads=1, head_dim=128):
    """Flash causal self-attention for prefill with GQA.

    Uses online softmax tiled over K in sections of P=128.
    Causal mask: position i can attend to positions [0, i].

    Args:
        q:        (num_q_heads, D, seq_len) bf16
        k:        (num_kv_heads, D, seq_len) bf16
        v:        (num_kv_heads, seq_len, D) bf16
        identity: (128, 128) bf16 — for transpose trick in PV matmul
        softmax_scale: float (1/sqrt(head_dim))
        num_q_heads: int — Q heads per rank
        num_kv_heads: int — KV heads per rank
        head_dim: int

    Returns:
        out: (num_q_heads, seq_len, D) bf16
    """
    seq_len = q.shape[2]
    P = nl.tile_size.pmax  # 128
    D = head_dim
    num_tiles = seq_len // P
    gqa_ratio = num_q_heads // num_kv_heads

    out = nl.ndarray((num_q_heads, seq_len, D), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity for transpose trick
    id_sbuf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for kv_h in range(num_kv_heads):
        # Load full K for this KV head: [D, seq_len]
        k_full = nl.ndarray((D, seq_len), dtype=k.dtype, buffer=nl.sbuf)
        for ti in range(num_tiles):
            nisa.dma_copy(dst=k_full[:, nl.ds(ti * P, P)],
                          src=k[kv_h, :, nl.ds(ti * P, P)])

        # Load full V: tiles of [P, D]
        v_tiles = nl.ndarray((P, num_tiles, D), dtype=v.dtype, buffer=nl.sbuf)
        for ti in range(num_tiles):
            nisa.dma_copy(dst=v_tiles[:, ti, :],
                          src=v[kv_h, nl.ds(ti * P, P), :])

        for q_offset in range(gqa_ratio):
            q_h = kv_h * gqa_ratio + q_offset

            # Process Q in groups of P=128 tokens
            for grp_i in range(num_tiles):

                # Load Q tile: [D, P]
                q_tile = nl.ndarray((D, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile, src=q[q_h, :, nl.ds(grp_i * P, P)])

                # Online softmax state for this Q group
                r_max = nl.full((P, 1), fill_value=float('-inf'), dtype=nl.float32)
                r_sum = nl.zeros((P, 1), dtype=nl.float32)
                pv_acc = nl.zeros((P, D), dtype=nl.float32)

                # Tile over K positions
                for k_ti in range(num_tiles):
                    # Score: Q_tile^T @ K_tile = [P_q, P_k]
                    # nc_matmul: out[P, F] = stationary[P, C] @ moving[C, F]
                    # q_tile is [D, P_q], k_tile is [D, P_k]
                    # We want q_tile^T @ k_tile = [P_q, D]^T ... no
                    # Score[P_q, P_k] = Q[P_q, D] @ K[D, P_k]
                    # nc_matmul: stationary=q_tile[D, P_q]... wait, that has D in partition

                    # Correct: nc_matmul out[P, F] = stat[P, C] @ mov[C, F]
                    # Want: scores[P_q, P_k]
                    #   stat must have P in partition = q_tile^T[P_q, D] → partition=P_q, contract=D
                    #   mov = k_slice[D, P_k] → contract=D, free=P_k
                    # But q_tile is stored as [D, P_q] in SBUF...
                    # We need q_tile transposed to [P_q, D] for stationary

                    # Use identity trick to transpose q_tile[D, P] → [P, D]
                    # Hmm that only works for [P, P] shapes
                    # Since D=P=128, q_tile[D, P] = [128, 128] — identity trick works!

                    q_tile_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(q_tile_T_psum, q_tile, id_sbuf)
                    q_tile_T_f32 = nl.copy(q_tile_T_psum, dtype=nl.float32)
                    q_tile_T = nl.copy(q_tile_T_f32, dtype=nl.bfloat16)
                    # q_tile_T is [P_q, D=P] = [128, 128]

                    # Now compute scores: q_tile_T[P, D] @ k_slice[D, P] = [P, P]
                    k_slice = k_full[:, nl.ds(k_ti * P, P)]  # [D, P]

                    score_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(score_psum, q_tile_T, k_slice)
                    scores = nl.copy(score_psum, dtype=nl.float32)
                    scores = nl.multiply(scores, softmax_scale)

                    # Causal mask: Q position grp_i*P+qi can attend to K position k_ti*P+ki
                    # Valid if k_ti*P+ki <= grp_i*P+qi
                    # For k_ti > grp_i: all masked (-inf)
                    # For k_ti < grp_i: all valid (0)
                    # For k_ti == grp_i: lower triangular
                    if k_ti > grp_i:
                        # All future — mask everything
                        scores = nl.full((P, P), fill_value=float('-inf'), dtype=nl.float32)
                    elif k_ti == grp_i:
                        # Diagonal tile — apply causal (upper triangle = -inf)
                        causal_mask = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                        for row in range(P):
                            for col in range(P):
                                if col > row:
                                    causal_mask[row, col] = float('-inf')
                                else:
                                    causal_mask[row, col] = 0.0
                        scores = nl.add(scores, causal_mask)
                    # else k_ti < grp_i: all valid, no mask needed

                    # Online softmax
                    tile_max = nl.max(scores, axis=1, keepdims=True)  # [P, 1]
                    old_max = nl.copy(r_max)
                    new_max = nl.maximum(old_max, tile_max)
                    corr = nl.exp(nl.subtract(old_max, new_max))
                    r_max[...] = new_max

                    shifted = nl.subtract(scores, new_max)
                    exp_scores = nl.exp(shifted)  # [P, P]
                    tile_sum = nl.sum(exp_scores, axis=1, keepdims=True)  # [P, 1]

                    old_sum = nl.copy(r_sum)
                    r_sum[...] = nl.add(nl.multiply(old_sum, corr), tile_sum)

                    # PV: exp_scores[P_q, P_k] @ V_tile[P_k, D] = [P_q, D]
                    # Need to transpose exp_scores for nc_matmul
                    # exp_scores[P, P]: stationary
                    # v_tile[P, D]: need as moving[P, D] with contract=P, free=D
                    # nc_matmul: out[P, F] = stat[P, C] @ mov[C, F]
                    # Want: pv[P_q, D] = exp_scores[P_q, P_k] @ V[P_k, D]
                    # stat = exp_scores^T[P_k, P_q]... no, partition must be P_k
                    # This doesn't directly fit nc_matmul...

                    # Use identity transpose trick (same as rolling-forcing):
                    # 1. exp_bf16[P, P] → transpose via id matmul → exp_T[P, P]
                    # 2. nc_matmul(exp_T[P, P], v_tile[P, D]) → [P, D]
                    exp_bf16 = nl.copy(exp_scores, dtype=nl.bfloat16)
                    exp_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(exp_T_psum, exp_bf16, id_sbuf)
                    exp_T_f32 = nl.copy(exp_T_psum, dtype=nl.float32)
                    exp_T = nl.copy(exp_T_f32, dtype=nl.bfloat16)

                    v_tile = v_tiles[:, k_ti, :]  # [P, D]
                    pv_psum = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv_psum, exp_T, v_tile)
                    pv_tile = nl.copy(pv_psum, dtype=nl.float32)

                    # Update accumulator with online softmax correction
                    pv_acc[...] = nl.add(nl.multiply(pv_acc, corr), pv_tile)

                # Normalize and store
                rcp = nl.reciprocal(r_sum)
                result = nl.multiply(pv_acc, rcp)
                result_bf16 = nl.copy(result, dtype=q.dtype)
                nisa.dma_copy(dst=out[q_h, nl.ds(grp_i * P, P), :],
                              src=result_bf16)

    return out
