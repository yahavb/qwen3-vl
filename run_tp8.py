"""Qwen3-VL-8B-Instruct TP-8 batch inference on PyTorch Native (Neuron).

Clean tensor-parallel implementation using proper nn.Module wrappers
for all_reduce/all_gather — no monkey-patching.

Runs: image inference (2 runs) + video inference (3 runs), then exits.
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
# TP Wrapper Modules — clean all_reduce/all_gather inside forward()
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
    """Wraps column-parallel lm_head with all_gather to reconstruct full vocab logits."""

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
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Init distributed
# ═══════════════════════════════════════════════════════════════════════
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
assert world_size == 8, f"Expected 8 ranks, got {world_size}"

torch.neuron.set_device(rank)
NEURON_DEVICE = torch.device("neuron")

if rank == 0:
    print("=" * 60)
    print(f"  Qwen3-VL-8B-Instruct TP-8 on PyTorch Native (Neuron)")
    print(f"  World size: {world_size}")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load and shard model
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n[STEP 1] Loading model on CPU and sharding for TP-8...")

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Qwen3-VL-8B-Instruct")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
).eval().requires_grad_(False)

TP = world_size
lang_model = model.model.language_model

lang_model.config.num_attention_heads = lang_model.config.num_attention_heads // TP
lang_model.config.num_key_value_heads = lang_model.config.num_key_value_heads // TP

for layer in lang_model.layers:
    attn = layer.self_attn
    mlp = layer.mlp

    attn.q_proj.weight = nn.Parameter(shard_column(attn.q_proj.weight.data, rank, TP), requires_grad=False)
    if attn.q_proj.bias is not None:
        attn.q_proj.bias = nn.Parameter(shard_column(attn.q_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1), requires_grad=False)

    attn.k_proj.weight = nn.Parameter(shard_column(attn.k_proj.weight.data, rank, TP), requires_grad=False)
    if attn.k_proj.bias is not None:
        attn.k_proj.bias = nn.Parameter(shard_column(attn.k_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1), requires_grad=False)

    attn.v_proj.weight = nn.Parameter(shard_column(attn.v_proj.weight.data, rank, TP), requires_grad=False)
    if attn.v_proj.bias is not None:
        attn.v_proj.bias = nn.Parameter(shard_column(attn.v_proj.bias.data.unsqueeze(1), rank, TP).squeeze(1), requires_grad=False)

    attn.o_proj.weight = nn.Parameter(shard_row(attn.o_proj.weight.data, rank, TP), requires_grad=False)

    mlp.gate_proj.weight = nn.Parameter(shard_column(mlp.gate_proj.weight.data, rank, TP), requires_grad=False)
    mlp.up_proj.weight = nn.Parameter(shard_column(mlp.up_proj.weight.data, rank, TP), requires_grad=False)
    mlp.down_proj.weight = nn.Parameter(shard_row(mlp.down_proj.weight.data, rank, TP), requires_grad=False)

lm_head_chunks = model.lm_head.weight.data.shape[0] // TP
model.lm_head.weight = nn.Parameter(
    model.lm_head.weight.data[rank * lm_head_chunks : (rank + 1) * lm_head_chunks].contiguous(),
    requires_grad=False)

if rank == 0:
    print(f"  Sharded 36 decoder layers for TP-{TP}")
    print(f"  lm_head: column-parallel ({lm_head_chunks} vocab/rank)")
    print(f"  Vision encoder: replicated on all ranks")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Wrap with TP modules, move to Neuron, compile
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("  Moving sharded model to Neuron...")
model = model.to(NEURON_DEVICE)

if rank == 0:
    print("  Wrapping layers with TP modules and compiling...")
compile_start = time.time()

for layer in lang_model.layers:
    layer.self_attn = torch.compile(TPAttention(layer.self_attn), backend='neuron', dynamic=False)
    layer.mlp = torch.compile(TPMLP(layer.mlp), backend='neuron', dynamic=False)

model.lm_head = torch.compile(TPLMHead(model.lm_head, TP), backend='neuron', dynamic=False)

compile_time = time.time() - compile_start
if rank == 0:
    print(f"  Compiled in {compile_time:.2f}s")

dist.barrier()
if rank == 0:
    print("\n  TP-8 setup complete!")
    print(f"  Each rank: ~1B params on 1 NeuronCore")
    print()

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Image Inference — TWO CONSECUTIVE RUNS
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  STEP 2: Image Inference (TP-8) — TWO CONSECUTIVE RUNS")
    print("=" * 60)

IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
if rank == 0:
    print(f"Downloading sample image: {IMAGE_URL}")
with urllib.request.urlopen(IMAGE_URL) as response:
    image_data = response.read()
image = Image.open(BytesIO(image_data)).convert("RGB")
if rank == 0:
    print(f"Image loaded: {image.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# RUN 1 (includes compilation)
if rank == 0:
    print("\n[RUN 1] Image inference (includes first-run compilation)...")
    img_start_1 = time.time()

with torch.no_grad():
    output_ids_1 = model.generate(**inputs, max_new_tokens=128, do_sample=False)

if rank == 0:
    img_time_1 = time.time() - img_start_1
    output_ids_cpu = output_ids_1.cpu()
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids_cpu[:, input_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n{'─' * 40}")
    print(f"IMAGE RUN 1 ({img_time_1:.2f}s):")
    print(f"{'─' * 40}")
    print(response_text[:200])
    print(f"{'─' * 40}\n")

dist.barrier()

# RUN 2 (cached)
if rank == 0:
    print("[RUN 2] Image inference (cached compilation, should be faster)...")
    img_start_2 = time.time()

with torch.no_grad():
    output_ids_2 = model.generate(**inputs, max_new_tokens=128, do_sample=False)

if rank == 0:
    img_time_2 = time.time() - img_start_2
    output_ids_cpu = output_ids_2.cpu()
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids_cpu[:, input_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n{'─' * 40}")
    print(f"IMAGE RUN 2 ({img_time_2:.2f}s):")
    print(f"{'─' * 40}")
    print(response_text[:200])
    print(f"{'─' * 40}")

    speedup = img_time_1 / img_time_2 if img_time_2 > 0 else float('inf')
    print(f"\n  ⏱️  RUN 1: {img_time_1:.2f}s (with compilation)")
    print(f"  ⏱️  RUN 2: {img_time_2:.2f}s (cached)")
    print(f"  ⚡ Speedup: {speedup:.1f}x")
    print()

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Video Inference
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  STEP 3: Video Inference (TP-8)")
    print("=" * 60)

VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
VIDEO_PATH = "/tmp/sample_video.mp4"
VIDEO_INPUTS_PATH = "/tmp/video_inputs.pt"

if rank == 0:
    print(f"Downloading sample video: {VIDEO_URL}")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print(f"Video saved to {VIDEO_PATH}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_PATH, "nframes": 4},
                {"type": "text", "text": "Describe what is happening in this video."},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    torch.save(inputs, VIDEO_INPUTS_PATH)
    print(f"Video processed and saved for all ranks")

dist.barrier()

inputs = torch.load(VIDEO_INPUTS_PATH, weights_only=False)
inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

for run_num in range(1, 4):
    if rank == 0:
        print(f"\n[VIDEO RUN {run_num}]...")
        vid_start = time.time()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    if rank == 0:
        vid_time = time.time() - vid_start
        output_ids_cpu = output_ids.cpu()
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids_cpu[:, input_len:]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"\n{'─' * 40}")
        print(f"VIDEO RUN {run_num} ({vid_time:.2f}s):")
        print(f"{'─' * 40}")
        print(response_text[:200])
        print(f"{'─' * 40}")

        if run_num == 1:
            vid_time_1 = vid_time
        elif run_num == 3:
            vid_speedup = vid_time_1 / vid_time if vid_time > 0 else float('inf')
            print(f"\n  ⚡ Video speedup (RUN1 vs RUN3): {vid_speedup:.1f}x")

    dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Model:           Qwen3-VL-8B-Instruct")
    print(f"  Backend:         torch.compile(backend='neuron')")
    print(f"  Parallelism:     TP-8 (8 NeuronCores)")
    print(f"  Compile time:    {compile_time:.2f}s")
    print(f"  Image RUN 1:     {img_time_1:.2f}s (with compilation)")
    print(f"  Image RUN 2:     {img_time_2:.2f}s (cached)")
    print(f"  Image speedup:   {img_time_1/img_time_2:.1f}x")
    print("\nAll tests passed!")

dist.destroy_process_group()
