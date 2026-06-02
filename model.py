"""Custom Qwen3-VL-8B text decoder for Neuron — fused projections, static KV cache.

No HuggingFace forward/generate. All operations explicit.
Vision encoder is loaded from HF but runs in eager mode separately.

Compilation targets (per layer):
  - fused_qkv: [4096, 768] matmul (Q+K+V in one shot)
  - o_proj: [512, 4096] matmul
  - fused_gate_up: [4096, 3072] matmul (gate+up in one shot)
  - down_proj: [1536, 4096] matmul
  - lm_head: [4096, 18992] matmul (per rank)

Eager (not compiled):
  - RoPE, attention scores, softmax, KV cache updates, layernorms, silu
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class FusedLinear(nn.Module):
    """x @ weight + bias — single matmul, compiled as one NEFF."""
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


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
    """Precomputes and caches cos/sin for RoPE. CPU only."""
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


def apply_rotary(x, cos, sin):
    """Apply RoPE (rotate_half style, matching Qwen3/Llama) to x: [batch, heads, seq, head_dim]."""
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    # cos/sin are [seq, head_dim/2], need to broadcast to [1, 1, seq, head_dim]
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def head_rms_norm(x, weight, eps=1e-6):
    """Per-head RMSNorm: x is [batch, heads, seq, head_dim], weight is [head_dim]."""
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp32 * rms).to(x.dtype) * weight


class DecoderLayer:
    """One transformer layer — stores compiled fused modules + eager norms."""
    def __init__(self, qkv, o_proj, gate_up, down, input_norm, post_norm, q_norm_weight, k_norm_weight):
        self.qkv = qkv
        self.o_proj = o_proj
        self.gate_up = gate_up
        self.down = down
        self.input_norm = input_norm
        self.post_norm = post_norm
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight


class Qwen3VLDecoder:
    """Custom decoder: fused projections, static KV cache, explicit generate loop.

    Args:
        layers: list of DecoderLayer
        lm_head: compiled FusedLinear for logit projection
        embed_tokens: nn.Embedding (on Neuron, not compiled)
        final_norm: RMSNorm
        rotary: RotaryEmbedding
        tp_size: tensor parallel world size
        num_q_heads: Q heads per rank
        num_kv_heads: KV heads per rank
        head_dim: dimension per head
        max_seq_len: max sequence length for KV cache
        device: neuron device
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
        self.kv_dim = num_kv_heads * head_dim

    def init_kv_cache(self, batch_size=1):
        """Allocate static KV cache at KV head count."""
        cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                          dtype=torch.bfloat16, device=self.device)
            v = torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                          dtype=torch.bfloat16, device=self.device)
            cache.append((k, v))
        return cache

    def forward_layer(self, hidden, layer_idx, kv_cache, start_pos, seq_len):
        """One layer forward. Returns updated hidden states."""
        layer = self.layers[layer_idx]
        bsz = hidden.shape[0]

        # Pre-attn norm
        residual = hidden
        h = layer.input_norm(hidden)

        # Fused QKV projection (compiled)
        qkv = layer.qkv(h)
        q = qkv[..., :self.q_dim].view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., self.q_dim:self.q_dim+self.kv_dim].view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., self.q_dim+self.kv_dim:].view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head QK RMSNorm before RoPE (critical for Qwen3)
        q = head_rms_norm(q, layer.q_norm_weight)
        k = head_rms_norm(k, layer.k_norm_weight)

        # RoPE (eager)
        cos, sin = self.rotary.get(seq_len, offset=start_pos, device=self.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # KV cache update (eager, in-place)
        k_cache, v_cache = kv_cache[layer_idx]
        k_cache[:, :, start_pos:start_pos+seq_len, :] = k
        v_cache[:, :, start_pos:start_pos+seq_len, :] = v

        # Attention — use full static buffer with masking for fixed shapes
        # This avoids Neuron eager re-dispatch for each new KV length
        valid_len = start_pos + seq_len
        k_full = k_cache  # [bsz, kv_heads, max_seq_len, head_dim]
        v_full = v_cache

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads < self.num_q_heads:
            repeat = self.num_q_heads // self.num_kv_heads
            k_full = k_full.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(bsz, self.num_q_heads, self.max_seq_len, self.head_dim)
            v_full = v_full.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(bsz, self.num_q_heads, self.max_seq_len, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        # Mask: -inf for all positions >= valid_len (not yet written)
        # Build mask on CPU then move (avoids Neuron dispatch for mask construction)
        attn_mask = torch.zeros(1, 1, seq_len, self.max_seq_len, dtype=torch.bfloat16, device=self.device)
        attn_mask[:, :, :, valid_len:] = float('-inf')
        # Causal mask for prefill
        if seq_len > 1:
            causal = torch.triu(
                torch.full((seq_len, valid_len), float('-inf'), dtype=torch.bfloat16, device=self.device),
                diagonal=start_pos + 1
            )
            attn_mask[:, :, :, :valid_len] = causal

        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_out = torch.matmul(attn_weights, v_full)

        # O proj (compiled) + TP all_reduce
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)
        attn_out = layer.o_proj(attn_out)
        dist.all_reduce(attn_out, op=dist.ReduceOp.SUM)

        hidden = residual + attn_out

        # Post-attn norm + MLP
        residual = hidden
        h = layer.post_norm(hidden)

        # Fused gate+up (compiled)
        gate_up = layer.gate_up(h)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up

        # Down proj (compiled) + TP all_reduce
        h = layer.down(h)
        dist.all_reduce(h, op=dist.ReduceOp.SUM)

        hidden = residual + h
        return hidden

    def forward(self, input_ids, start_pos, kv_cache):
        """Full forward pass through all layers. Returns logits for last token."""
        bsz, seq_len = input_ids.shape

        # Embedding (eager, replicated)
        hidden = self.embed_tokens(input_ids)

        # All decoder layers
        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, i, kv_cache, start_pos, seq_len)

        # Final norm
        hidden = self.final_norm(hidden)

        # Only compute logits for last token (saves compute during decode)
        last_hidden = hidden[:, -1:, :]

        # lm_head (compiled) + all_gather for full vocab
        local_logits = self.lm_head(last_hidden)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        logits = torch.cat(gathered, dim=-1)

        return logits

    def prefill(self, input_ids, kv_cache):
        """Prefill: process full prompt, return logits for next token."""
        return self.forward(input_ids, start_pos=0, kv_cache=kv_cache)

    def decode_step(self, token_id, pos, kv_cache):
        """Decode: single token, fixed shape [1,1]."""
        input_ids = token_id.view(1, 1)
        return self.forward(input_ids, start_pos=pos, kv_cache=kv_cache)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, eos_token_id=151645, verbose=False):
        """Custom generate loop with static decode shape.

        1. Prefill: process full prompt (compiles for prefill bucket)
        2. Decode: one token at a time with seq_len=1 (compiles once, reuses)
        """
        import time as _time
        bsz, prompt_len = input_ids.shape
        kv_cache = self.init_kv_cache(batch_size=bsz)

        # Prefill
        t0 = _time.time()
        logits = self.prefill(input_ids, kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1)
        if verbose and dist.get_rank() == 0:
            print(f"    prefill: {_time.time()-t0:.2f}s")

        generated_tokens = [next_token.item()]

        # Decode loop — always seq_len=1, same compiled NEFF reused
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

    def generate_with_embeds(self, inputs_embeds, max_new_tokens=256, eos_token_id=151645):
        """Generate from embeddings (after vision encoder merges visual tokens).

        Same as generate() but starts from embeddings instead of token IDs.
        """
        bsz, prompt_len, _ = inputs_embeds.shape
        kv_cache = self.init_kv_cache(batch_size=bsz)

        # Prefill from embeddings (bypass embed_tokens)
        hidden = inputs_embeds
        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, i, kv_cache, start_pos=0, seq_len=prompt_len)
        hidden = self.final_norm(hidden)
        last_hidden = hidden[:, -1:, :]
        local_logits = self.lm_head(last_hidden)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        logits = torch.cat(gathered, dim=-1)

        next_token = logits[:, -1, :].argmax(dim=-1)
        generated_tokens = [next_token.item()]

        # Decode loop
        for i in range(max_new_tokens - 1):
            pos = prompt_len + i
            logits = self.decode_step(next_token, pos, kv_cache)
            next_token = logits[:, -1, :].argmax(dim=-1)
            tok = next_token.item()
            if tok == eos_token_id:
                break
            generated_tokens.append(tok)

        return generated_tokens
