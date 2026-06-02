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
import threading

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.distributed as dist
from PIL import Image
from io import BytesIO
import urllib.request


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
    print("  WARMUP: Image inference (compiles prefill + decode NEFFs)")
    print("=" * 60)

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

# For now, use text-only path (embed input_ids directly) to validate decoder
# Vision embedding merge will be added once decoder is validated
input_ids = inputs["input_ids"].to(NEURON_DEVICE)

if rank == 0:
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

# Cached run
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
    print(f"\n  Running 50 tokens (cached)...")
    t0 = time.time()

generated_tokens = decoder.generate(input_ids, max_new_tokens=50)

if rank == 0:
    t50 = time.time() - t0
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"  50 tokens: {t50:.1f}s  ({50/t50:.2f} tok/s)")
    print(f"  Generated: {text[:200]}")

dist.barrier()

if rank == 0:
    print(f"\n[READY] Warmup complete. Starting server on port 8000.")

# ═══════════════════════════════════════════════════════════════════════
# SERVING: FastAPI
# ═══════════════════════════════════════════════════════════════════════
INPUTS_PATH = "/tmp/current_inputs.pt"


def run_inference(input_ids_neuron, max_tokens=256):
    """Run generate on all ranks."""
    generated = decoder.generate(input_ids_neuron, max_new_tokens=max_tokens)
    if rank == 0:
        return processor.decode(generated, skip_special_tokens=True)
    return None


if rank == 0:
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional, Any

    app = FastAPI(title="Qwen3-VL-8B (TP-8, Fused, Static KV)")

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
                                content_items.append({"type": "video", "video": url, "nframes": MAX_NFRAMES})
                            elif item.get("type") in ("image", "video", "text"):
                                content_items.append(item)
                            else:
                                content_items.append(item)
                        else:
                            content_items.append(item)
                    qwen_messages.append({"role": msg.role, "content": content_items})

            text_prompt = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            tok_inputs = processor(text=[text_prompt], return_tensors="pt")
            input_ids = tok_inputs["input_ids"].to(NEURON_DEVICE)

            # Broadcast input to all ranks
            seq_len_tensor = torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(seq_len_tensor, src=0)
            dist.broadcast(input_ids, src=0)

            max_tok = request.max_tokens or MAX_NEW_TOKENS_DEFAULT
            response_text = run_inference(input_ids, max_tokens=max_tok)

            elapsed = time.time() - start_time
            print(f"  Request: {elapsed:.2f}s")
            return ChatResponse(choices=[ChatChoice(message={"role": "assistant", "content": response_text})])

    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print(f"\n[SERVER] Running on port 8000")
    while True:
        time.sleep(0.1)

else:
    # Non-rank-0: wait for rank-0 to broadcast inputs
    while True:
        try:
            seq_len_tensor = torch.tensor([0], dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(seq_len_tensor, src=0)
            seq_len = seq_len_tensor.item()
            input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(input_ids, src=0)
            run_inference(input_ids, max_tokens=MAX_NEW_TOKENS_DEFAULT)
        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")
            time.sleep(1)
