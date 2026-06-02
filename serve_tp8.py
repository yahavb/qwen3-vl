"""Qwen3-VL-8B TP-8 server — fused projections, static KV cache, custom generate.

No HuggingFace model.generate(). Custom decode loop with:
  - Fused QKV (1 matmul, not 3)
  - Fused gate+up (1 matmul, not 2)
  - Static KV cache (fixed shape, no reallocation)
  - Decode always seq_len=1 (one NEFF, reused every token)
  - Prefill compiles per bucket (reused for same-bucket inputs)

Usage:
    torchrun --nproc_per_node=8 serve_tp8.py
"""

import os
import sys
import time

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# Init distributed
# ═══════════════════════════════════════════════════════════════════════
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
assert world_size == 8, f"Expected 8 ranks, got {world_size}"

torch.neuron.set_device(rank)
NEURON_DEVICE = torch.device("neuron")

MAX_NFRAMES = int(os.environ.get("QWEN3_VL_MAX_NFRAMES", "4"))
MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN3_VL_MAX_NEW_TOKENS", "256"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Qwen3-VL-8B-Instruct")

if rank == 0:
    print("=" * 60)
    print(f"  Qwen3-VL-8B TP-8 — Fused Projections + Static KV Cache")
    print(f"  Custom generate loop (no HF generate)")
    print(f"  World size: {world_size}")
    print(f"  MAX_NFRAMES: {MAX_NFRAMES}, MAX_NEW_TOKENS: {MAX_NEW_TOKENS_DEFAULT}")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════════
from loader import load_model

decoder, vision_model, processor, hf_model = load_model(
    MODEL_PATH, rank, world_size, NEURON_DEVICE
)

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# WARMUP: Image — triggers NEFF compilation for prefill + decode shapes
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n" + "=" * 60)
    print("  WARMUP: Text-only (compiles prefill + decode NEFFs)")
    print("=" * 60)

# Use a simple text prompt to validate the decoder without vision encoder
prompt = "The capital of France is"
input_ids = processor(text=[prompt], return_tensors="pt")["input_ids"].to(NEURON_DEVICE)

if rank == 0:
    print(f"  Prompt: '{prompt}'")
    print(f"  Input seq_len={input_ids.shape[-1]}")
    print(f"  Running generate(max_new_tokens=10)...")
    t0 = time.time()

generated_tokens = decoder.generate(input_ids, max_new_tokens=10)

if rank == 0:
    compile_time = time.time() - t0
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"  First call (compile): {compile_time:.1f}s")
    print(f"  Generated: {text}")

dist.barrier()

# Cached run — same shape, should reuse NEFFs
if rank == 0:
    print(f"\n  Running again (cached NEFFs)...")
    t0 = time.time()

generated_tokens = decoder.generate(input_ids, max_new_tokens=10)

if rank == 0:
    cached_time = time.time() - t0
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"  Cached: {cached_time:.1f}s  (speedup: {compile_time/max(cached_time,0.01):.1f}x)")
    print(f"  Generated: {text}")
    print(f"  Decode throughput: {10/cached_time:.2f} tok/s")

dist.barrier()

# Longer generation
if rank == 0:
    print(f"\n  Running 50 tokens (verbose, first 5 decode steps timed)...")
    t0 = time.time()

generated_tokens = decoder.generate(input_ids, max_new_tokens=50, verbose=True)

if rank == 0:
    t50 = time.time() - t0
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"  50 tokens: {t50:.1f}s  ({50/t50:.2f} tok/s)")
    print(f"  Generated: {text[:200]}")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# IMAGE TEST: Run vision encoder + merged generate
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("\n" + "=" * 60)
    print("  IMAGE TEST: Vision encoder + decoder generate")
    print("=" * 60)

from PIL import Image
from io import BytesIO
import urllib.request
from vision import get_vision_embeddings, merge_vision_embeddings

IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
if rank == 0:
    print(f"  Downloading image...")
with urllib.request.urlopen(IMAGE_URL) as response:
    image_data = response.read()
image = Image.open(BytesIO(image_data)).convert("RGB")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")

input_ids = inputs["input_ids"].to(NEURON_DEVICE)
pixel_values = inputs.get("pixel_values")
image_grid_thw = inputs.get("image_grid_thw")

if pixel_values is not None:
    pixel_values = pixel_values.to(NEURON_DEVICE)
if image_grid_thw is not None:
    image_grid_thw = image_grid_thw.to(NEURON_DEVICE)

if rank == 0:
    print(f"  Input seq_len={input_ids.shape[-1]}")
    print(f"  pixel_values shape={pixel_values.shape if pixel_values is not None else None}")
    print(f"  image_grid_thw={image_grid_thw}")

# Run vision encoder (eager)
if rank == 0:
    print(f"  Running vision encoder...")
    t0 = time.time()

image_embeds, _ = get_vision_embeddings(hf_model, pixel_values, image_grid_thw)

if rank == 0:
    vis_time = time.time() - t0
    print(f"  Vision encoder: {vis_time:.2f}s")
    if image_embeds is not None:
        print(f"  image_embeds shape={image_embeds.shape}")

# Merge vision embeddings into text
inputs_embeds = merge_vision_embeddings(decoder.embed_tokens, input_ids, image_embeds=image_embeds)

if rank == 0:
    print(f"  Merged inputs_embeds shape={inputs_embeds.shape}")
    print(f"  Generating with vision (max_new_tokens=30)...")
    t0 = time.time()

generated_tokens = decoder.generate_with_embeds(inputs_embeds, max_new_tokens=30)

if rank == 0:
    img_time = time.time() - t0
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"  Image generate: {img_time:.1f}s ({30/img_time:.2f} tok/s)")
    print(f"  Generated: {text[:300]}")

dist.barrier()

if rank == 0:
    print("\n[DONE] Profiling complete. Exiting.")
import os as _os
_os._exit(0)
