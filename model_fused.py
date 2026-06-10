"""Qwen3-VL-8B decoder — whole-submodule compilation for Neuron.

Key difference from model.py: instead of compiling individual matmuls and
running everything else eagerly, we compile entire AttentionBlock and MLPBlock
nn.Modules. This lets the Neuron compiler fuse:
  - AttentionBlock: input_norm → QKV matmul → QK head-norm → RoPE → O-proj
  - MLPBlock: post_norm → gate_up matmul → SiLU → down matmul

Attention scores/softmax and KV cache updates remain eager (dynamic shapes).
But the surrounding ops are now fused, eliminating HBM roundtrips between
norm/projection/activation that the per-matmul approach causes.

Compilation targets (per layer, 2 NEFFs instead of 4):
  - AttentionBlock: norm → QKV → QK-norm → RoPE → reshape → O-proj
  - MLPBlock: norm → gate_up → SiLU*up → down

Eager (not compiled):
  - Attention scores (Q@K^T), softmax, V accumulation (dynamic KV length)
  - KV cache read/write (in-place mutations)
  - TP all_reduce (collective, not compilable)
  - Embedding, final norm, lm_head (small relative cost)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class RotaryEmbedding:
    """Precomputes and caches cos/sin for RoPE."""
    def __init__(self, head_dim, max_seq_len=8192, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos_cached = freqs.cos().to(torch.bfloat16)
        self.sin_cached = freqs.sin().to(torch.bfloat16)

    def get(self, seq_len, offset=0, device=None):
        cos = self.cos_cached[offset:offset+seq_len]
        sin = self.sin_cached[offset:offset+seq_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        return cos, sin


class AttentionBlock(nn.Module):
    """Fused attention projection block — compiled as ONE NEFF.

    Compiles: input_norm → QKV matmul → split → QK per-head norm → RoPE → O-proj reshape
    Output: (q, k, v, attn_out_weight) all ready for eager attention.

    Actually returns q, k, v after RoPE so the caller can do attention eagerly.
    O-proj is also inside this module so it gets fused with the norms/projections.
    """
    def __init__(self, qkv_weight, qkv_bias, o_proj_weight,
                 input_norm_weight, q_norm_weight, k_norm_weight,
                 num_q_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.qkv_weight = nn.Parameter(qkv_weight, requires_grad=False)
        self.qkv_bias = nn.Parameter(qkv_bias, requires_grad=False) if qkv_bias is not None else None
        self.input_norm_weight = nn.Parameter(input_norm_weight, requires_grad=False)
        self.q_norm_weight = nn.Parameter(q_norm_weight, requires_grad=False)
        self.k_norm_weight = nn.Parameter(k_norm_weight, requires_grad=False)
        self.o_proj_weight = nn.Parameter(o_proj_weight, requires_grad=False)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_q_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.eps = eps

    def forward(self, hidden, cos, sin):
        """Returns (q, k, v, o_proj_weight) — q/k have RoPE applied.

        Input: hidden [bsz, seq_len, hidden_dim], cos/sin [seq_len, head_dim/2]
        Output: q [bsz, num_q_heads, seq_len, head_dim]
                k [bsz, num_kv_heads, seq_len, head_dim]
                v [bsz, num_kv_heads, seq_len, head_dim]
        """
        bsz, seq_len, _ = hidden.shape

        # RMSNorm (fused into this NEFF)
        h_fp32 = hidden.float()
        rms = torch.rsqrt(h_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        h = (h_fp32 * rms).to(hidden.dtype) * self.input_norm_weight

        # QKV projection (fused into same NEFF)
        qkv = h @ self.qkv_weight
        if self.qkv_bias is not None:
            qkv = qkv + self.qkv_bias

        # Split and reshape
        q = qkv[..., :self.q_dim].view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., self.q_dim:self.q_dim+self.kv_dim].view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., self.q_dim+self.kv_dim:].view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head QK RMSNorm (fused)
        q_fp32 = q.float()
        q_rms = torch.rsqrt(q_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        q = (q_fp32 * q_rms).to(q.dtype) * self.q_norm_weight

        k_fp32 = k.float()
        k_rms = torch.rsqrt(k_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        k = (k_fp32 * k_rms).to(k.dtype) * self.k_norm_weight

        # RoPE (fused)
        cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
        sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)

        q1, q2 = q.chunk(2, dim=-1)
        q = q * cos_full + torch.cat((-q2, q1), dim=-1) * sin_full

        k1, k2 = k.chunk(2, dim=-1)
        k = k * cos_full + torch.cat((-k2, k1), dim=-1) * sin_full

        return q, k, v


class OProjection(nn.Module):
    """O-proj as separate compiled module (seq_len=1 for decode reuse)."""
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return x @ self.weight


class MLPBlock(nn.Module):
    """Fused MLP block — compiled as ONE NEFF.

    Compiles: post_norm → gate_up matmul → split → SiLU(gate)*up → down matmul
    All in one compiled graph — no HBM roundtrips between norm, gate, silu, down.
    """
    def __init__(self, gate_up_weight, down_weight, post_norm_weight, eps=1e-6):
        super().__init__()
        self.gate_up_weight = nn.Parameter(gate_up_weight, requires_grad=False)
        self.down_weight = nn.Parameter(down_weight, requires_grad=False)
        self.post_norm_weight = nn.Parameter(post_norm_weight, requires_grad=False)
        self.eps = eps

    def forward(self, hidden):
        """Input/output: [bsz, seq_len, hidden_dim]"""
        # RMSNorm (fused)
        h_fp32 = hidden.float()
        rms = torch.rsqrt(h_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        h = (h_fp32 * rms).to(hidden.dtype) * self.post_norm_weight

        # Gate+Up projection (fused)
        gate_up = h @ self.gate_up_weight

        # SiLU * up (fused)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up

        # Down projection (fused)
        return h @ self.down_weight


class FusedLinear(nn.Module):
    """Simple matmul for lm_head."""
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class DecoderLayer:
    """One transformer layer — compiled attention + MLP submodules."""
    def __init__(self, attn_block, o_proj, mlp_block):
        self.attn_block = attn_block  # compiled AttentionBlock
        self.o_proj = o_proj          # compiled OProjection
        self.mlp_block = mlp_block    # compiled MLPBlock


class Qwen3VLDecoderFused:
    """Optimized decoder: whole-submodule compilation, fewer NEFFs, less HBM traffic.

    Per layer: 3 compiled modules (attn_block, o_proj, mlp_block) instead of 4.
    Each module fuses multiple ops that were previously eager.
    """
    def __init__(self, layers, lm_head, embed_tokens, final_norm, rotary,
                 tp_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, device):
        self.layers = layers
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        self.final_norm = final_norm
        self.rotary = rotary
        self.tp_size = tp_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.num_layers = len(layers)
        self.q_dim = num_q_heads * head_dim

        self.causal_mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float('-inf'), dtype=torch.bfloat16, device=device),
            diagonal=1
        )

    def init_kv_cache(self, batch_size=1):
        cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                          dtype=torch.bfloat16, device=self.device)
            v = torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                          dtype=torch.bfloat16, device=self.device)
            cache.append((k, v))
        return cache

    def forward_layer(self, hidden, layer_idx, kv_cache, start_pos, seq_len):
        layer = self.layers[layer_idx]
        bsz = hidden.shape[0]

        # === Attention (compiled block does norm+QKV+QK-norm+RoPE) ===
        residual = hidden
        cos, sin = self.rotary.get(seq_len, offset=start_pos, device=self.device)
        q, k, v = layer.attn_block(hidden, cos, sin)

        # KV cache update (eager — dynamic positions)
        k_cache, v_cache = kv_cache[layer_idx]
        k_cache[:, :, start_pos:start_pos+seq_len, :] = k
        v_cache[:, :, start_pos:start_pos+seq_len, :] = v

        # Attention scores (eager — variable KV length)
        k_full = k_cache
        v_full = v_cache

        if self.num_kv_heads < self.num_q_heads:
            repeat = self.num_q_heads // self.num_kv_heads
            k_full = k_full.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(bsz, self.num_q_heads, self.max_seq_len, self.head_dim)
            v_full = v_full.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(bsz, self.num_q_heads, self.max_seq_len, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn_mask = self.causal_mask[start_pos:start_pos+seq_len, :self.max_seq_len]
        attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_out = torch.matmul(attn_weights, v_full)

        # O-proj (compiled) + TP all_reduce
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)
        attn_out = layer.o_proj(attn_out)
        dist.all_reduce(attn_out, op=dist.ReduceOp.SUM)

        hidden = residual + attn_out

        # === MLP (compiled block does norm+gate_up+silu+down) ===
        residual = hidden
        mlp_out = layer.mlp_block(hidden)
        dist.all_reduce(mlp_out, op=dist.ReduceOp.SUM)
        hidden = residual + mlp_out

        return hidden

    def forward(self, input_ids, start_pos, kv_cache):
        bsz, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)

        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, i, kv_cache, start_pos, seq_len)

        hidden = self.final_norm(hidden)
        last_hidden = hidden[:, -1:, :]

        local_logits = self.lm_head(last_hidden)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        logits = torch.cat(gathered, dim=-1)
        return logits

    def prefill(self, input_ids, kv_cache):
        return self.forward(input_ids, start_pos=0, kv_cache=kv_cache)

    def decode_step(self, token_id, pos, kv_cache):
        input_ids = token_id.view(1, 1)
        return self.forward(input_ids, start_pos=pos, kv_cache=kv_cache)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, eos_token_id=151645, verbose=False):
        import time as _time
        bsz, prompt_len = input_ids.shape
        kv_cache = self.init_kv_cache(batch_size=bsz)

        t0 = _time.time()
        logits = self.prefill(input_ids, kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1)
        if verbose and dist.get_rank() == 0:
            print(f"    prefill: {_time.time()-t0:.2f}s")

        generated_tokens = [next_token.item()]

        for i in range(max_new_tokens - 1):
            pos = prompt_len + i
            t0 = _time.time()
            logits = self.decode_step(next_token, pos, kv_cache)
            next_token = logits[:, -1, :].argmax(dim=-1)
            tok = next_token.item()
            if verbose and dist.get_rank() == 0 and (i < 10 or i % 10 == 0):
                print(f"    decode[{i}]: {_time.time()-t0:.2f}s tok={tok}")
            if tok == eos_token_id:
                break
            generated_tokens.append(tok)

        return generated_tokens

    def generate_with_embeds(self, inputs_embeds, max_new_tokens=256, eos_token_id=151645,
                             deepstack_embeds=None, visual_mask=None):
        bsz, prompt_len, _ = inputs_embeds.shape
        kv_cache = self.init_kv_cache(batch_size=bsz)

        hidden = inputs_embeds
        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, i, kv_cache, start_pos=0, seq_len=prompt_len)

            if deepstack_embeds is not None and visual_mask is not None and i < len(deepstack_embeds):
                hidden = hidden.clone()
                hidden[0, visual_mask] = hidden[0, visual_mask] + deepstack_embeds[i].to(hidden.dtype)

        hidden = self.final_norm(hidden)
        last_hidden = hidden[:, -1:, :]
        local_logits = self.lm_head(last_hidden)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        logits = torch.cat(gathered, dim=-1)

        next_token = logits[:, -1, :].argmax(dim=-1)
        generated_tokens = [next_token.item()]

        for i in range(max_new_tokens - 1):
            pos = prompt_len + i
            logits = self.decode_step(next_token, pos, kv_cache)
            next_token = logits[:, -1, :].argmax(dim=-1)
            tok = next_token.item()
            if tok == eos_token_id:
                break
            generated_tokens.append(tok)

        return generated_tokens
