"""Load Qwen3-VL-8B weights into fused-submodule decoder.

Same weight loading as loader.py, but builds AttentionBlock and MLPBlock
modules that get compiled as whole submodules instead of individual matmuls.

Compilation strategy:
  - AttentionBlock: norm + QKV + QK-norm + RoPE compiled as ONE NEFF
  - MLPBlock: norm + gate_up + SiLU + down compiled as ONE NEFF
  - OProjection: o_proj as a separate small NEFF (for decode seq_len=1 reuse)
  - lm_head: compiled separately

Total: 3 NEFFs per layer + 1 for lm_head = 3*36+1 = 109 compiled modules
(vs 4*36+1 = 145 in the per-matmul approach, but each NEFF does more work)
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from model_fused import (
    AttentionBlock, OProjection, MLPBlock, FusedLinear, RMSNorm,
    RotaryEmbedding, DecoderLayer, Qwen3VLDecoderFused,
)


def shard_column(weight, rank, tp):
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def load_model(model_path, rank, world_size, device, compile_backend='neuron'):
    """Load HF model, build fused submodules, compile, return decoder.

    Returns:
        decoder: Qwen3VLDecoderFused with compiled fused submodules
        vision_model: HF vision encoder on device (eager)
        processor: HF processor for tokenization
        hf_model: full HF model reference (for vision encoder access)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    TP = world_size
    t0 = time.time()

    if rank == 0:
        print(f"[LOAD] Loading HF model from {model_path}...")

    processor = AutoProcessor.from_pretrained(model_path)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().requires_grad_(False)

    lang_model = hf_model.model.language_model
    config = lang_model.config

    NUM_LAYERS = config.num_hidden_layers
    HIDDEN = config.hidden_size
    NUM_Q_HEADS_TOTAL = config.num_attention_heads
    NUM_KV_HEADS_TOTAL = config.num_key_value_heads
    HEAD_DIM = HIDDEN // NUM_Q_HEADS_TOTAL
    INTER = config.intermediate_size

    NUM_Q_HEADS = NUM_Q_HEADS_TOTAL // TP
    NUM_KV_HEADS = NUM_KV_HEADS_TOTAL // TP
    INTER_PER_RANK = INTER // TP

    if rank == 0:
        print(f"  Model: {NUM_LAYERS} layers, hidden={HIDDEN}, heads={NUM_Q_HEADS_TOTAL}/{NUM_KV_HEADS_TOTAL}")
        print(f"  TP-{TP}: Q={NUM_Q_HEADS}/rank, KV={NUM_KV_HEADS}/rank, inter={INTER_PER_RANK}/rank")
        print(f"  Strategy: Fused submodule compilation (AttentionBlock + MLPBlock)")

    layers = []
    for i in range(NUM_LAYERS):
        attn = lang_model.layers[i].self_attn
        mlp = lang_model.layers[i].mlp

        # Fuse QKV: column-shard each, concatenate, transpose for x @ W
        q_w = shard_column(attn.q_proj.weight.data, rank, TP)
        k_w = shard_column(attn.k_proj.weight.data, rank, TP)
        v_w = shard_column(attn.v_proj.weight.data, rank, TP)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).t().contiguous()

        # Fuse QKV bias (if present)
        qkv_bias = None
        if attn.q_proj.bias is not None:
            q_b = shard_column(attn.q_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1)
            k_b = shard_column(attn.k_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1)
            v_b = shard_column(attn.v_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1)
            qkv_bias = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

        # O proj: row-parallel, transpose for x @ W
        o_w = shard_row(attn.o_proj.weight.data, rank, TP).t().contiguous()

        # Fuse gate+up: column-shard each, concatenate, transpose
        gate_w = shard_column(mlp.gate_proj.weight.data, rank, TP)
        up_w = shard_column(mlp.up_proj.weight.data, rank, TP)
        gate_up_w = torch.cat([gate_w, up_w], dim=0).t().contiguous()

        # Down proj: row-parallel, transpose
        down_w = shard_row(mlp.down_proj.weight.data, rank, TP).t().contiguous()

        # Norm weights
        input_norm_w = lang_model.layers[i].input_layernorm.weight.data.clone()
        post_norm_w = lang_model.layers[i].post_attention_layernorm.weight.data.clone()

        # QK norm weights
        if hasattr(attn, 'q_norm'):
            q_norm_w = attn.q_norm.weight.data.clone()
            k_norm_w = attn.k_norm.weight.data.clone()
        else:
            q_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
            k_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16)

        # Build fused submodules
        attn_block = AttentionBlock(
            qkv_weight=qkv_w,
            qkv_bias=qkv_bias,
            o_proj_weight=o_w,
            input_norm_weight=input_norm_w,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            eps=config.rms_norm_eps,
        )

        o_proj = OProjection(o_w)

        mlp_block = MLPBlock(
            gate_up_weight=gate_up_w,
            down_weight=down_w,
            post_norm_weight=post_norm_w,
            eps=config.rms_norm_eps,
        )

        layers.append(DecoderLayer(
            attn_block=attn_block,
            o_proj=o_proj,
            mlp_block=mlp_block,
        ))

    # lm_head: column-parallel
    lm_head_chunks = hf_model.lm_head.weight.data.shape[0] // TP
    lm_head_w = hf_model.lm_head.weight.data[
        rank * lm_head_chunks : (rank + 1) * lm_head_chunks
    ].contiguous()
    lm_head = FusedLinear(lm_head_w.t().contiguous())

    # embed_tokens (replicated)
    embed_tokens = lang_model.embed_tokens

    # Final norm (replicated)
    if hasattr(lang_model, 'norm'):
        final_norm_weight = lang_model.norm.weight.data.clone()
    elif hasattr(lang_model, 'final_layernorm'):
        final_norm_weight = lang_model.final_layernorm.weight.data.clone()
    else:
        raise AttributeError(f"Cannot find final norm")
    final_norm = RMSNorm(final_norm_weight, eps=config.rms_norm_eps)

    # RoPE
    rope_theta = getattr(config, 'rope_theta', None)
    if rope_theta is None:
        rope_scaling = getattr(config, 'rope_scaling', {}) or {}
        rope_theta = rope_scaling.get('rope_theta', 1000000.0)
    max_seq = int(os.environ.get("MAX_SEQ_LEN", "4096"))
    rotary = RotaryEmbedding(HEAD_DIM, max_seq_len=max_seq, base=rope_theta)

    if rank == 0:
        print(f"  Weights fused+sharded: {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Move to device + compile WHOLE SUBMODULES
    # ═══════════════════════════════════════════════════════════════════
    t0 = time.time()
    if rank == 0:
        print(f"[COMPILE] Moving to Neuron + compiling fused submodules...")
        print(f"  Strategy: 3 NEFFs/layer (attn_block + o_proj + mlp_block)")

    _compile = lambda m: torch.compile(
        m.to(device), backend=compile_backend, dynamic=False, fullgraph=True
    )

    for i, layer in enumerate(layers):
        layer.attn_block = _compile(layer.attn_block)
        layer.o_proj = _compile(layer.o_proj)
        layer.mlp_block = _compile(layer.mlp_block)

    lm_head = _compile(lm_head)
    embed_tokens = embed_tokens.to(device)
    final_norm = final_norm.to(device)

    if rank == 0:
        print(f"  Compiled {NUM_LAYERS*3+1} modules: {time.time()-t0:.1f}s (lazy)")

    # Build decoder
    decoder = Qwen3VLDecoderFused(
        layers=layers,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        rotary=rotary,
        tp_size=TP,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq,
        device=device,
    )

    # Vision encoder: keep on device, no compile (eager)
    if hasattr(hf_model.model, 'visual'):
        vision_model = hf_model.model.visual.to(device)
    elif hasattr(hf_model.model, 'vision_model'):
        vision_model = hf_model.model.vision_model.to(device)
    else:
        vision_model = None
        if rank == 0:
            print("  WARNING: No vision encoder found")

    if rank == 0:
        print(f"[READY] Fused decoder + vision encoder loaded.")
        print(f"  Decoder: {NUM_LAYERS} layers, {NUM_LAYERS*3+1} compiled modules")
        print(f"  Each layer: AttentionBlock(norm+QKV+QK-norm+RoPE) + OProj + MLPBlock(norm+gate_up+silu+down)")
        print(f"  Vision: eager on all ranks")

    return decoder, vision_model, processor, hf_model
