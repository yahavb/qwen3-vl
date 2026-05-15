"""Qwen3-VL-8B-Instruct TP-8 server on PyTorch Native (Neuron).

Clean tensor-parallel implementation using proper nn.Module wrappers
for all_reduce/all_gather — no monkey-patching.

Architecture:
  - 8 NeuronCores, 1 per rank (torchrun --nproc_per_node=8)
  - LLM decoder: TP-8 sharded (column-parallel Q/K/V/gate/up, row-parallel O/down)
  - Vision encoder: replicated on all ranks
  - embed_tokens: replicated, lm_head: column-parallel with all_gather

Model dims (Qwen3-VL-8B):
  hidden_size=4096, num_heads=32, num_kv_heads=8, intermediate=12288
  head_dim=128, 36 layers, vocab=151936
"""

import os
import sys
import time
import threading

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
    """Split output dimension (dim=0) into tp chunks, take rank's chunk."""
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    """Split input dimension (dim=1) into tp chunks, take rank's chunk."""
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
    attn_implementation="eager",  # Force eager attention (not SDPA) for Neuron compatibility
).eval().requires_grad_(False)

# ─── Shard LLM decoder layers ──────────────────────────────────────
TP = world_size
lang_model = model.model.language_model

# Update config for sharded head counts
lang_model.config.num_attention_heads = lang_model.config.num_attention_heads // TP
lang_model.config.num_key_value_heads = lang_model.config.num_key_value_heads // TP

for layer in lang_model.layers:
    attn = layer.self_attn
    mlp = layer.mlp

    # Attention: column-parallel Q/K/V, row-parallel O
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

    # MLP: column-parallel gate/up, row-parallel down
    mlp.gate_proj.weight = nn.Parameter(shard_column(mlp.gate_proj.weight.data, rank, TP), requires_grad=False)
    mlp.up_proj.weight = nn.Parameter(shard_column(mlp.up_proj.weight.data, rank, TP), requires_grad=False)
    mlp.down_proj.weight = nn.Parameter(shard_row(mlp.down_proj.weight.data, rank, TP), requires_grad=False)

# lm_head: column-parallel
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

# PURE EAGER TEST — no torch.compile at all
# This isolates whether the TP logic itself is correct
if rank == 0:
    print("  [PURE EAGER] Skipping torch.compile — running entirely in eager mode")
compile_start = time.time()

# Wrap with TP modules (all_reduce/all_gather)
if rank == 0:
    print("  Wrapping with TP modules (all_reduce/all_gather)...")

for layer in lang_model.layers:
    layer.self_attn = TPAttention(layer.self_attn)
    layer.mlp = TPMLP(layer.mlp)

model.lm_head = TPLMHead(model.lm_head, TP)

compile_time = time.time() - compile_start
if rank == 0:
    print(f"  Compiled in {compile_time:.2f}s")

dist.barrier()
if rank == 0:
    print("\n  TP-8 setup complete!")
    print(f"  Each rank: ~1B params on 1 NeuronCore")
    print()

# ═══════════════════════════════════════════════════════════════════════
# WARMUP: Image inference (triggers compilation)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  WARMUP: Image Inference (triggers compilation)")
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

if rank == 0:
    print("\n[WARMUP] Running first inference (includes compilation)...")
    warmup_start = time.time()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

if rank == 0:
    warmup_time = time.time() - warmup_start
    output_ids_cpu = output_ids.cpu()
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids_cpu[:, input_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n{'─' * 40}")
    print(f"WARMUP COMPLETE ({warmup_time:.2f}s):")
    print(f"{'─' * 40}")
    print(response_text[:200])
    print(f"{'─' * 40}\n")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# WARMUP 2: Video inference (triggers compilation for video seq lengths)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("=" * 60)
    print("  WARMUP 2: Video Inference (triggers compilation)")
    print("=" * 60)

VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
VIDEO_PATH = "/tmp/sample_video.mp4"
VIDEO_INPUTS_PATH = "/tmp/video_warmup_inputs.pt"

if rank == 0:
    print(f"Downloading sample video: {VIDEO_URL}")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print(f"Video saved to {VIDEO_PATH}")

    from qwen_vl_utils import process_vision_info

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
    image_inputs, video_inputs = process_vision_info(messages)
    vid_inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    torch.save(vid_inputs, VIDEO_INPUTS_PATH)
    print(f"Video processed and saved for all ranks")

dist.barrier()

vid_inputs = torch.load(VIDEO_INPUTS_PATH, weights_only=False)
vid_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in vid_inputs.items()}

if rank == 0:
    print("\n[WARMUP 2] Video inference (includes compilation for video seq length)...")
    warmup2_start = time.time()

with torch.no_grad():
    vid_output_ids = model.generate(**vid_inputs, max_new_tokens=128, do_sample=False)

if rank == 0:
    warmup2_time = time.time() - warmup2_start
    vid_output_ids_cpu = vid_output_ids.cpu()
    vid_input_len = vid_inputs["input_ids"].shape[-1]
    vid_generated = vid_output_ids_cpu[:, vid_input_len:]
    vid_response = processor.batch_decode(vid_generated, skip_special_tokens=True)[0]

    print(f"\n{'─' * 40}")
    print(f"VIDEO WARMUP COMPLETE ({warmup2_time:.2f}s):")
    print(f"{'─' * 40}")
    print(vid_response[:200])
    print(f"{'─' * 40}\n")

dist.barrier()

if rank == 0:
    print("[READY] Model warmed up (image + video), starting HTTP server...")

# ═══════════════════════════════════════════════════════════════════════
# SERVING: FastAPI on rank 0, all ranks participate in generate
# ═══════════════════════════════════════════════════════════════════════
INPUTS_PATH = "/tmp/current_inputs.pt"


def run_inference(inputs_path, max_tokens=256):
    """All ranks load inputs from file and run generate together."""
    loaded_inputs = torch.load(inputs_path, weights_only=False)
    loaded_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v for k, v in loaded_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**loaded_inputs, max_new_tokens=max_tokens, do_sample=False)

    if rank == 0:
        output_ids_cpu = output_ids.cpu()
        input_len = loaded_inputs["input_ids"].shape[-1]
        generated_ids = output_ids_cpu[:, input_len:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return None


if rank == 0:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Any

    app = FastAPI(title="Qwen3-VL-8B-Instruct (PyTorch Native, TP-8)")

    inference_lock = threading.Lock()

    class ChatMessage(BaseModel):
        role: str
        content: Any

    class ChatRequest(BaseModel):
        model: Optional[str] = "Qwen3-VL-8B-Instruct"
        messages: List[ChatMessage]
        max_tokens: Optional[int] = 256
        temperature: Optional[float] = 0.0
        stream: Optional[bool] = False

    class ChatChoice(BaseModel):
        index: int = 0
        message: dict
        finish_reason: str = "stop"

    class ChatResponse(BaseModel):
        id: str = "chatcmpl-qwen3vl"
        object: str = "chat.completion"
        model: str = "Qwen3-VL-8B-Instruct"
        choices: List[ChatChoice]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/readiness")
    def readiness():
        return {"status": "ready"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        with inference_lock:
            start_time = time.time()

            # Parse messages into Qwen VL format
            qwen_messages = []
            for msg in request.messages:
                if isinstance(msg.content, str):
                    qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
                else:
                    content_items = []
                    for item in msg.content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                content_items.append({"type": "image", "image": url})
                            elif item.get("type") == "video_url":
                                url = item.get("video_url", {}).get("url", "")
                                if url.startswith("data:video/"):
                                    # Decode base64 data URI to temp file
                                    import base64 as b64mod
                                    import tempfile as tmpmod
                                    header, b64data = url.split(",", 1)
                                    video_bytes = b64mod.b64decode(b64data)
                                    tmp = tmpmod.NamedTemporaryFile(suffix=".mp4", delete=False)
                                    tmp.write(video_bytes)
                                    tmp.close()
                                    print(f"  Decoded video data URI to {tmp.name} ({len(video_bytes)} bytes)")
                                    content_items.append({"type": "video", "video": tmp.name, "nframes": 4})
                                else:
                                    content_items.append({"type": "video", "video": url, "nframes": 4})
                            elif item.get("type") in ("image", "video", "text"):
                                content_items.append(item)
                            else:
                                content_items.append(item)
                        else:
                            content_items.append(item)
                    qwen_messages.append({"role": msg.role, "content": content_items})

            from qwen_vl_utils import process_vision_info

            text_prompt = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(qwen_messages)

            req_inputs = processor(
                text=[text_prompt],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt"
            )

            # Save inputs for other ranks
            torch.save(req_inputs, INPUTS_PATH)

            # Signal other ranks
            max_tokens_tensor = torch.tensor([request.max_tokens or 256], dtype=torch.long).to(NEURON_DEVICE)
            dist.broadcast(max_tokens_tensor, src=0)

            # Run inference (all ranks participate)
            response_text = run_inference(INPUTS_PATH, max_tokens=request.max_tokens or 256)

            elapsed = time.time() - start_time
            print(f"  Request completed in {elapsed:.2f}s")

            return ChatResponse(
                choices=[ChatChoice(
                    message={"role": "assistant", "content": response_text}
                )]
            )

    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print(f"\n[SERVER] FastAPI running on port 8000")

    # Rank 0 main thread: keep alive
    while True:
        time.sleep(0.1)

else:
    # Non-rank-0: wait for broadcast signals from rank 0
    while True:
        try:
            max_tokens_tensor = torch.tensor([256], dtype=torch.long).to(NEURON_DEVICE)
            dist.broadcast(max_tokens_tensor, src=0)
            max_tokens = max_tokens_tensor.item()
            run_inference(INPUTS_PATH, max_tokens=max_tokens)
        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")
            time.sleep(1)
