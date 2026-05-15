"""Qwen3-VL-8B-Instruct TP-16 benchmark on PyTorch Native (Neuron).

Architecture:
  - 16 NeuronCores, 1 per rank (torchrun --nproc_per_node=16)
  - LLM decoder: TP-16 with KV-replicated sharding
  - Vision encoder: replicated on all ranks

Model dims (Qwen3-VL-8B):
  hidden_size=4096, num_heads=32, num_kv_heads=8, intermediate=12288
  head_dim=128, 36 layers, vocab=151936

TP-16 sharding:
  Q: 32 heads -> 2/rank (256 dim/rank) — column-parallel
  K: 8 KV heads -> REPLICATED (each rank gets all 8 KV heads)
  V: 8 KV heads -> REPLICATED (each rank gets all 8 KV heads)
  O: row-parallel (input dim = 2 heads * 128 = 256/rank)
  gate_proj/up_proj: 12288 -> 768/rank (column)
  down_proj: row-parallel (input 768/rank)
  lm_head: 151936 -> 9496/rank (column)
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request


# ═══════════════════════════════════════════════════════════════════════
# TP Wrapper Modules
# ═══════════════════════════════════════════════════════════════════════

class TPAttention(nn.Module):
    """Wraps attention with all_reduce after row-parallel o_proj."""
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, *args, **kwargs):
        out = self.attn(*args, **kwargs)
        if isinstance(out, tuple):
            attn_out = out[0]
            dist.all_reduce(attn_out, op=dist.ReduceOp.SUM)
            return (attn_out,) + out[1:]
        else:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out


class TPMLP(nn.Module):
    """Wraps MLP with all_reduce after row-parallel down_proj."""
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        out = self.mlp(*args, **kwargs)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class TPLMHead(nn.Module):
    """Column-parallel lm_head with all_gather."""
    def __init__(self, lm_head, tp_size):
        super().__init__()
        self.lm_head = lm_head
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        local_logits = self.lm_head(*args, **kwargs)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        return torch.cat(gathered, dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Sharding helpers
# ═══════════════════════════════════════════════════════════════════════

def shard_column(weight, rank, tp):
    """Split output dimension (dim=0) into tp chunks."""
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    """Split input dimension (dim=1) into tp chunks."""
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Init distributed
# ═══════════════════════════════════════════════════════════════════════
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
TP = world_size
assert TP == 16, f"Expected 16 ranks, got {TP}"

torch.neuron.set_device(rank)
NEURON_DEVICE = torch.device("neuron")

if rank == 0:
    print("=" * 60)
    print(f"  Qwen3-VL-8B-Instruct TP-16 on PyTorch Native (Neuron)")
    print(f"  World size: {TP}")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load and shard model
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n[STEP 1] Loading model on CPU and sharding for TP-16...")

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Qwen3-VL-8B-Instruct")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().requires_grad_(False)

lang_model = model.model.language_model

# ─── TP-16 sharding: Q sharded, K/V replicated ──────────────────
# Q: 32 heads / 16 = 2 heads/rank (column-parallel)
# K/V: 8 KV heads — REPLICATED (each rank keeps all 8)
# O: row-parallel (input = 2*128=256 per rank)

# Config update: Q heads per rank, KV heads stay full
lang_model.config.num_attention_heads = 32 // TP   # 2 per rank
lang_model.config.num_key_value_heads = 8          # full (replicated)

for layer in lang_model.layers:
    attn = layer.self_attn
    mlp = layer.mlp

    # Q: column-parallel (shard by TP=16)
    attn.q_proj.weight = nn.Parameter(
        shard_column(attn.q_proj.weight.data, rank, TP), requires_grad=False)
    if attn.q_proj.bias is not None:
        attn.q_proj.bias = nn.Parameter(
            shard_column(attn.q_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1),
            requires_grad=False)

    # K: REPLICATED — keep full weights (no sharding)
    # V: REPLICATED — keep full weights (no sharding)

    # O: row-parallel (shard input dim by TP=16)
    attn.o_proj.weight = nn.Parameter(
        shard_row(attn.o_proj.weight.data, rank, TP), requires_grad=False)

    # MLP: column-parallel gate/up, row-parallel down
    mlp.gate_proj.weight = nn.Parameter(
        shard_column(mlp.gate_proj.weight.data, rank, TP), requires_grad=False)
    mlp.up_proj.weight = nn.Parameter(
        shard_column(mlp.up_proj.weight.data, rank, TP), requires_grad=False)
    mlp.down_proj.weight = nn.Parameter(
        shard_row(mlp.down_proj.weight.data, rank, TP), requires_grad=False)

# lm_head: column-parallel
lm_head_chunks = model.lm_head.weight.data.shape[0] // TP
model.lm_head.weight = nn.Parameter(
    model.lm_head.weight.data[rank * lm_head_chunks : (rank + 1) * lm_head_chunks].contiguous(),
    requires_grad=False)

if rank == 0:
    print(f"  Sharded 36 layers for TP-{TP}:")
    print(f"    Q: 2 heads/rank (column-parallel)")
    print(f"    K/V: 8 heads/rank (REPLICATED)")
    print(f"    O: row-parallel (256 input dim/rank)")
    print(f"    MLP: gate/up 768/rank, down row-parallel")
    print(f"    lm_head: {lm_head_chunks} vocab/rank")
    print(f"    Vision encoder: replicated on all ranks")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Move to Neuron, compile, wrap with TP
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n[STEP 2] Moving to Neuron...")
model = model.to(NEURON_DEVICE)

if rank == 0:
    print("  Compiling with torch.compile(backend='neuron')...")
compile_start = time.time()

for layer in lang_model.layers:
    layer.self_attn = torch.compile(layer.self_attn, backend='neuron', dynamic=False)
    layer.mlp = torch.compile(layer.mlp, backend='neuron', dynamic=False)
model.lm_head = torch.compile(model.lm_head, backend='neuron', dynamic=False)

compile_time = time.time() - compile_start
if rank == 0:
    print(f"  torch.compile setup: {compile_time:.2f}s")

# Wrap with TP modules (all_reduce/all_gather outside compiled graph)
if rank == 0:
    print("  Wrapping with TP modules...")
for layer in lang_model.layers:
    layer.self_attn = TPAttention(layer.self_attn)
    layer.mlp = TPMLP(layer.mlp)
model.lm_head = TPLMHead(model.lm_head, TP)

dist.barrier()
if rank == 0:
    print(f"\n  TP-{TP} setup complete!")
    print(f"  Each rank: ~0.5B params on 1 NeuronCore")
    print()

# ═══════════════════════════════════════════════════════════════════════
# Helper: run generate and return (response_text, elapsed_time)
# ═══════════════════════════════════════════════════════════════════════
def timed_generate(inputs, max_new_tokens=128):
    """All ranks run generate. Returns (text, time) on rank 0."""
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start

    text = None
    if rank == 0:
        output_ids_cpu = output_ids.cpu()
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids_cpu[:, input_len:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text, elapsed


# ═══════════════════════════════════════════════════════════════════════
# IMAGE BENCHMARK: 1 warmup + 3 timed runs
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  IMAGE BENCHMARK: 1 warmup + 3 timed runs")
    print("=" * 60)

IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
if rank == 0:
    print(f"  Downloading: {IMAGE_URL}")
with urllib.request.urlopen(IMAGE_URL) as resp:
    image_data = resp.read()
image = Image.open(BytesIO(image_data)).convert("RGB")
if rank == 0:
    print(f"  Image size: {image.size}")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image in detail."},
]}]
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
img_inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
img_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in img_inputs.items()}

# Warmup
if rank == 0:
    print("\n  [WARMUP] Image inference (triggers compilation)...")
img_warmup_text, img_warmup_time = timed_generate(img_inputs)
if rank == 0:
    print(f"  WARMUP: {img_warmup_time:.2f}s")
    print(f"  Response: {img_warmup_text}")
    print()
dist.barrier()

# 3 timed runs
img_times = []
for i in range(1, 4):
    if rank == 0:
        print(f"  [IMAGE RUN {i}]...")
    text, elapsed = timed_generate(img_inputs)
    img_times.append(elapsed)
    if rank == 0:
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Response: {text}")
        print()
    dist.barrier()

if rank == 0:
    img_avg = sum(img_times) / len(img_times)
    print(f"  IMAGE SUMMARY:")
    print(f"    Warmup:  {img_warmup_time:.2f}s")
    for i, t in enumerate(img_times):
        print(f"    Run {i+1}:   {t:.2f}s")
    print(f"    Average: {img_avg:.2f}s")
    print(f"    Speedup: {img_warmup_time/img_avg:.1f}x (warmup vs avg cached)")
    print()

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# VIDEO BENCHMARK: 1 warmup + 3 timed runs
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  VIDEO BENCHMARK: 1 warmup + 3 timed runs")
    print("=" * 60)

VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
VIDEO_PATH = "/tmp/sample_video.mp4"
VIDEO_INPUTS_PATH = "/tmp/video_inputs.pt"

if rank == 0:
    print(f"  Downloading: {VIDEO_URL}")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print(f"  Video saved to {VIDEO_PATH}")

    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "video", "video": VIDEO_PATH, "nframes": 4},
        {"type": "text", "text": "Describe what is happening in this video."},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    vid_inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    torch.save(vid_inputs, VIDEO_INPUTS_PATH)
    print(f"  Video preprocessed")

dist.barrier()

vid_inputs = torch.load(VIDEO_INPUTS_PATH, weights_only=False)
vid_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in vid_inputs.items()}

# Warmup
if rank == 0:
    print("\n  [WARMUP] Video inference (triggers compilation)...")
vid_warmup_text, vid_warmup_time = timed_generate(vid_inputs)
if rank == 0:
    print(f"  WARMUP: {vid_warmup_time:.2f}s")
    print(f"  Response: {vid_warmup_text}")
    print()
dist.barrier()

# 3 timed runs
vid_times = []
for i in range(1, 4):
    if rank == 0:
        print(f"  [VIDEO RUN {i}]...")
    text, elapsed = timed_generate(vid_inputs)
    vid_times.append(elapsed)
    if rank == 0:
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Response: {text}")
        print()
    dist.barrier()

if rank == 0:
    vid_avg = sum(vid_times) / len(vid_times)
    print(f"  VIDEO SUMMARY:")
    print(f"    Warmup:  {vid_warmup_time:.2f}s")
    for i, t in enumerate(vid_times):
        print(f"    Run {i+1}:   {t:.2f}s")
    print(f"    Average: {vid_avg:.2f}s")
    print(f"    Speedup: {vid_warmup_time/vid_avg:.1f}x (warmup vs avg cached)")
    print()

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Model:        Qwen3-VL-8B-Instruct")
    print(f"  Backend:      torch.compile(backend='neuron') + eager attention")
    print(f"  Parallelism:  TP-{TP} (16 NeuronCores, KV-replicated)")
    print(f"  Compile time: {compile_time:.2f}s")
    print(f"")
    print(f"  IMAGE:")
    print(f"    Warmup:     {img_warmup_time:.2f}s")
    print(f"    Avg cached: {sum(img_times)/len(img_times):.2f}s")
    print(f"    Speedup:    {img_warmup_time/(sum(img_times)/len(img_times)):.1f}x")
    print(f"")
    print(f"  VIDEO:")
    print(f"    Warmup:     {vid_warmup_time:.2f}s")
    print(f"    Avg cached: {sum(vid_times)/len(vid_times):.2f}s")
    print(f"    Speedup:    {vid_warmup_time/(sum(vid_times)/len(vid_times)):.1f}x")
    print(f"")
    print("  All tests complete!")

dist.destroy_process_group()
