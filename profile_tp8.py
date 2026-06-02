"""Qwen3-VL-8B TP-8 profiling script — find where time goes.

Measures:
  1. Model load + shard time
  2. Move to Neuron time
  3. Vision encoder forward (image) — is it compiling?
  4. Single prefill forward (no generate loop) — first call triggers NEFF compilation
  5. Single decode step — second shape triggers NEFF compilation
  6. Full generate() with max_new_tokens=10 — total time for 10 tokens
  7. Same for video input

Each phase prints wall-clock time. No server, no warmup loop — just profiling.
"""

import os
import sys
import time

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from io import BytesIO
import urllib.request


def log(msg, rank=0):
    if dist.get_rank() == rank:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# TP Wrappers (same as serve_tp8.py)
# ═══════════════════════════════════════════════════════════════════════

class TPAttention(nn.Module):
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
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        out = self.mlp(*args, **kwargs)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class TPLMHead(nn.Module):
    def __init__(self, lm_head, tp_size):
        super().__init__()
        self.lm_head = lm_head
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        local_logits = self.lm_head(*args, **kwargs)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        return torch.cat(gathered, dim=-1)


def shard_column(weight, rank, tp):
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()

def shard_row(weight, rank, tp):
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Init
# ═══════════════════════════════════════════════════════════════════════
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
TP = world_size
assert TP == 8, f"Expected 8 ranks, got {TP}"

torch.neuron.set_device(rank)
NEURON_DEVICE = torch.device("neuron")

log("=" * 60)
log("  PROFILING: Qwen3-VL-8B TP-8 — timing each phase")
log("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Load + shard
# ═══════════════════════════════════════════════════════════════════════
t0 = time.time()
log("PHASE 1: Loading model + sharding...")

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Qwen3-VL-8B-Instruct")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().requires_grad_(False)

lang_model = model.model.language_model
lang_model.config.num_attention_heads = 32 // TP
lang_model.config.num_key_value_heads = 8 // TP

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

log(f"  PHASE 1 done: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Move to Neuron
# ═══════════════════════════════════════════════════════════════════════
t0 = time.time()
log("PHASE 2: Moving to Neuron device...")
model = model.to(NEURON_DEVICE)
log(f"  PHASE 2 done: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: torch.compile setup + TP wrappers
# ═══════════════════════════════════════════════════════════════════════
t0 = time.time()
log("PHASE 3: torch.compile + TP wrappers...")

_compile = lambda m: torch.compile(m, backend='neuron', dynamic=False, fullgraph=True)

for i, layer in enumerate(lang_model.layers):
    layer.self_attn.q_proj = _compile(layer.self_attn.q_proj)
    layer.self_attn.k_proj = _compile(layer.self_attn.k_proj)
    layer.self_attn.v_proj = _compile(layer.self_attn.v_proj)
    layer.self_attn.o_proj = _compile(layer.self_attn.o_proj)
    layer.mlp.gate_proj = _compile(layer.mlp.gate_proj)
    layer.mlp.up_proj = _compile(layer.mlp.up_proj)
    layer.mlp.down_proj = _compile(layer.mlp.down_proj)

model.lm_head = _compile(model.lm_head)

for layer in lang_model.layers:
    layer.self_attn = TPAttention(layer.self_attn)
    layer.mlp = TPMLP(layer.mlp)
model.lm_head = TPLMHead(model.lm_head, TP)

log(f"  PHASE 3 done: {time.time()-t0:.1f}s (lazy, no compilation yet)")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Prepare image input
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 4: Preparing image input...")
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
with urllib.request.urlopen(IMAGE_URL) as response:
    image_data = response.read()
image = Image.open(BytesIO(image_data)).convert("RGB")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

log(f"  Input shapes: input_ids={inputs['input_ids'].shape}, pixel_values={inputs.get('pixel_values', torch.tensor([])).shape}")
log(f"  image_grid_thw={inputs.get('image_grid_thw', 'N/A')}")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: generate() with max_new_tokens=5 (image)
# This triggers: vision encoder + prefill compilation + 5 decode steps
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 5: generate(max_new_tokens=5) — IMAGE (first call, triggers compilation)...")
t0 = time.time()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

phase5_time = time.time() - t0
log(f"  PHASE 5 done: {phase5_time:.1f}s (image, 5 tokens, includes NEFF compilation)")

if rank == 0:
    output_ids_cpu = output_ids.cpu()
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids_cpu[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    log(f"  Output: {text}")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: generate() with max_new_tokens=5 AGAIN (image, same shape)
# Should reuse compiled NEFFs — measures pure inference speed
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 6: generate(max_new_tokens=5) — IMAGE (cached, no compilation)...")
t0 = time.time()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

phase6_time = time.time() - t0
log(f"  PHASE 6 done: {phase6_time:.1f}s (cached inference)")
log(f"  Speedup: {phase5_time/phase6_time:.1f}x")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: generate() with max_new_tokens=50 (image, same input)
# Tests decode throughput
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 7: generate(max_new_tokens=50) — IMAGE (cached, 50 tokens)...")
t0 = time.time()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

phase7_time = time.time() - t0
log(f"  PHASE 7 done: {phase7_time:.1f}s (50 tokens)")
log(f"  Throughput: {50/phase7_time:.2f} tok/s")

if rank == 0:
    output_ids_cpu = output_ids.cpu()
    generated = output_ids_cpu[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    log(f"  Output: {text[:150]}")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# PHASE 8: Video input — prepare
# ═══════════════════════════════════════════════════════════════════════
MAX_NFRAMES = int(os.environ.get("QWEN3_VL_MAX_NFRAMES", "4"))
log(f"PHASE 8: Preparing video input (nframes={MAX_NFRAMES})...")

VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
VIDEO_PATH = "/tmp/sample_video.mp4"
VIDEO_INPUTS_PATH = "/tmp/video_inputs.pt"

if rank == 0:
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "video", "video": VIDEO_PATH, "nframes": MAX_NFRAMES},
        {"type": "text", "text": "Describe this video."},
    ]}]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    vid_inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    torch.save(vid_inputs, VIDEO_INPUTS_PATH)

dist.barrier()

vid_inputs = torch.load(VIDEO_INPUTS_PATH, weights_only=False)
vid_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in vid_inputs.items()}

log(f"  Video input shapes: input_ids={vid_inputs['input_ids'].shape}")
log(f"  video_grid_thw={vid_inputs.get('video_grid_thw', 'N/A')}")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 9: generate() with max_new_tokens=5 (video, first call)
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 9: generate(max_new_tokens=5) — VIDEO (first call, may recompile)...")
t0 = time.time()

with torch.no_grad():
    vid_output_ids = model.generate(**vid_inputs, max_new_tokens=5, do_sample=False)

phase9_time = time.time() - t0
log(f"  PHASE 9 done: {phase9_time:.1f}s (video, 5 tokens)")

if rank == 0:
    vid_output_cpu = vid_output_ids.cpu()
    vid_input_len = vid_inputs["input_ids"].shape[-1]
    vid_generated = vid_output_cpu[:, vid_input_len:]
    vid_text = processor.batch_decode(vid_generated, skip_special_tokens=True)[0]
    log(f"  Output: {vid_text}")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# PHASE 10: generate() again (video, cached)
# ═══════════════════════════════════════════════════════════════════════
log("PHASE 10: generate(max_new_tokens=5) — VIDEO (cached)...")
t0 = time.time()

with torch.no_grad():
    vid_output_ids = model.generate(**vid_inputs, max_new_tokens=5, do_sample=False)

phase10_time = time.time() - t0
log(f"  PHASE 10 done: {phase10_time:.1f}s (video cached)")
log(f"  Speedup: {phase9_time/phase10_time:.1f}x")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("  PROFILING SUMMARY")
log("=" * 60)
log(f"  Image compile (5 tok):  {phase5_time:.1f}s")
log(f"  Image cached  (5 tok):  {phase6_time:.1f}s  ({phase5_time/phase6_time:.1f}x speedup)")
log(f"  Image cached  (50 tok): {phase7_time:.1f}s  ({50/phase7_time:.2f} tok/s)")
log(f"  Video compile (5 tok):  {phase9_time:.1f}s")
log(f"  Video cached  (5 tok):  {phase10_time:.1f}s  ({phase9_time/phase10_time:.1f}x speedup)")
log("=" * 60)

dist.barrier()
log("Done. Exiting.")
