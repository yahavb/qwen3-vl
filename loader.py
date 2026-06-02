"""Load Qwen3-VL-8B weights into our custom fused decoder.

Handles:
  - Loading full HF model on CPU
  - Fusing Q+K+V into single weight matrix per layer
  - Fusing gate+up into single weight matrix per layer
  - TP sharding (column for QKV/gate_up, row for O/down)
  - Moving to Neuron device
  - torch.compile on fused projections
  - Vision encoder kept as-is (eager, replicated)
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from model import (
    FusedLinear, RMSNorm, RotaryEmbedding, DecoderLayer, Qwen3VLDecoder,
)


def shard_column(weight, rank, tp):
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def load_model(model_path, rank, world_size, device, compile_backend='neuron'):
    """Load HF model, fuse weights, shard, compile, return custom decoder + vision encoder.

    Returns:
        decoder: Qwen3VLDecoder with compiled fused projections
        vision_model: HF vision encoder on device (eager)
        processor: HF processor for tokenization
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

    # Fuse and shard weights
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

        # O proj: row-parallel, transpose for x @ W (no bias in Qwen3)
        o_w = shard_row(attn.o_proj.weight.data, rank, TP).t().contiguous()

        # Fuse gate+up: column-shard each, concatenate, transpose
        gate_w = shard_column(mlp.gate_proj.weight.data, rank, TP)
        up_w = shard_column(mlp.up_proj.weight.data, rank, TP)
        gate_up_w = torch.cat([gate_w, up_w], dim=0).t().contiguous()

        # Down proj: row-parallel, transpose
        down_w = shard_row(mlp.down_proj.weight.data, rank, TP).t().contiguous()

        # Norms (replicated, not compiled)
        input_norm = RMSNorm(
            lang_model.layers[i].input_layernorm.weight.data.clone(),
            eps=config.rms_norm_eps,
        )
        post_norm = RMSNorm(
            lang_model.layers[i].post_attention_layernorm.weight.data.clone(),
            eps=config.rms_norm_eps,
        )

        layers.append(DecoderLayer(
            qkv=FusedLinear(qkv_w, qkv_bias),
            o_proj=FusedLinear(o_w),
            gate_up=FusedLinear(gate_up_w),
            down=FusedLinear(down_w),
            input_norm=input_norm,
            post_norm=post_norm,
        ))

    # lm_head: column-parallel
    lm_head_chunks = hf_model.lm_head.weight.data.shape[0] // TP
    lm_head_w = hf_model.lm_head.weight.data[
        rank * lm_head_chunks : (rank + 1) * lm_head_chunks
    ].contiguous()
    lm_head = FusedLinear(lm_head_w.t().contiguous())

    # embed_tokens (replicated)
    embed_tokens = lang_model.embed_tokens

    # Final norm (replicated) — HF uses 'norm' or 'final_layernorm'
    if hasattr(lang_model, 'norm'):
        final_norm_weight = lang_model.norm.weight.data.clone()
    elif hasattr(lang_model, 'final_layernorm'):
        final_norm_weight = lang_model.final_layernorm.weight.data.clone()
    else:
        raise AttributeError(f"Cannot find final norm. Attributes: {[n for n in dir(lang_model) if 'norm' in n.lower()]}")
    final_norm = RMSNorm(final_norm_weight, eps=config.rms_norm_eps)

    # RoPE — rope_theta may be top-level or inside rope_scaling
    rope_theta = getattr(config, 'rope_theta', None)
    if rope_theta is None:
        rope_scaling = getattr(config, 'rope_scaling', {}) or {}
        rope_theta = rope_scaling.get('rope_theta', 1000000.0)
    max_seq = int(os.environ.get("MAX_SEQ_LEN", "4096"))
    rotary = RotaryEmbedding(HEAD_DIM, max_seq_len=max_seq, base=rope_theta)

    if rank == 0:
        print(f"  Weights fused+sharded: {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Move to device + compile
    # ═══════════════════════════════════════════════════════════════════
    t0 = time.time()
    if rank == 0:
        print(f"[COMPILE] Moving to Neuron + compiling fused projections...")

    _compile = lambda m: torch.compile(
        m.to(device), backend=compile_backend, dynamic=False, fullgraph=True
    )

    for i, layer in enumerate(layers):
        layer.qkv = _compile(layer.qkv)
        layer.o_proj = _compile(layer.o_proj)
        layer.gate_up = _compile(layer.gate_up)
        layer.down = _compile(layer.down)
        layer.input_norm = layer.input_norm.to(device)
        layer.post_norm = layer.post_norm.to(device)

    lm_head = _compile(lm_head)
    embed_tokens = embed_tokens.to(device)
    final_norm = final_norm.to(device)

    if rank == 0:
        print(f"  Compiled {NUM_LAYERS*4+1} modules: {time.time()-t0:.1f}s (lazy)")

    # Build decoder
    decoder = Qwen3VLDecoder(
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
        print(f"[READY] Decoder + vision encoder loaded.")
        print(f"  Decoder: {NUM_LAYERS} layers, {NUM_LAYERS*4+1} compiled modules")
        print(f"  Vision: eager on all ranks")

    return decoder, vision_model, processor, hf_model
