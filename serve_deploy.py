"""Qwen3-VL-8B TP-8 deployment server — OpenAI-compatible API.

Long-running server with /v1/chat/completions endpoint.
Supports image_url and video_url content types (including base64 data URIs).

Usage:
    torchrun --nproc_per_node=8 serve_deploy.py
"""

import os
import sys
import time
import threading
import base64
import tempfile

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.distributed as dist

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
    print(f"  Qwen3-VL-8B TP-8 — Deployment Server")
    print(f"  OpenAI-compatible /v1/chat/completions")
    print(f"  Image + Video understanding")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════════
from loader import load_model
from vision import prepare_vision_embeds

decoder, vision_model, processor, hf_model = load_model(
    MODEL_PATH, rank, world_size, NEURON_DEVICE
)

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# Warmup — compile prefill + decode NEFFs with a short text prompt
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print("[WARMUP] Compiling with text prompt...")
    t0 = time.time()

prompt = "Hello"
input_ids = processor(text=[prompt], return_tensors="pt")["input_ids"].to(NEURON_DEVICE)
decoder.generate(input_ids, max_new_tokens=5)

if rank == 0:
    print(f"[WARMUP] Done in {time.time()-t0:.1f}s")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# Server state
# ═══════════════════════════════════════════════════════════════════════
server_ready = True
inference_lock = threading.Lock()


def run_vision_inference(inputs_dict, max_tokens):
    """Process a request with potential vision content. All ranks participate."""
    inputs_embeds, deepstack_embeds, visual_mask = prepare_vision_embeds(
        hf_model, processor, inputs_dict, NEURON_DEVICE
    )
    generated_tokens = decoder.generate_with_embeds(
        inputs_embeds, max_new_tokens=max_tokens,
        deepstack_embeds=deepstack_embeds, visual_mask=visual_mask
    )
    if rank == 0:
        return processor.decode(generated_tokens, skip_special_tokens=True)
    return None


def run_text_inference(input_ids, max_tokens):
    """Process a text-only request. All ranks participate."""
    generated_tokens = decoder.generate(input_ids, max_new_tokens=max_tokens)
    if rank == 0:
        return processor.decode(generated_tokens, skip_special_tokens=True)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Rank 0: FastAPI server
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Any

    app = FastAPI(title="Qwen3-VL-8B-Instruct")

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

    class ModelInfo(BaseModel):
        id: str = "Qwen3-VL-8B-Instruct"
        object: str = "model"
        owned_by: str = "local"

    class ModelList(BaseModel):
        object: str = "list"
        data: List[ModelInfo]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/readiness")
    def readiness():
        if server_ready:
            return {"status": "ready"}
        raise HTTPException(status_code=503, detail="Not ready")

    @app.get("/v1/models")
    def list_models():
        return ModelList(data=[ModelInfo()])

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        with inference_lock:
            start_time = time.time()

            # Parse messages to detect image/video content
            has_vision = False
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
                                if url.startswith("data:image/"):
                                    # Decode base64 data URI to temp file
                                    header, b64data = url.split(",", 1)
                                    img_bytes = base64.b64decode(b64data)
                                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                    tmp.write(img_bytes)
                                    tmp.close()
                                    content_items.append({"type": "image", "image": tmp.name})
                                else:
                                    content_items.append({"type": "image", "image": url})
                                has_vision = True
                            elif item.get("type") == "video_url":
                                url = item.get("video_url", {}).get("url", "")
                                if url.startswith("data:video/"):
                                    header, b64data = url.split(",", 1)
                                    vid_bytes = base64.b64decode(b64data)
                                    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                                    tmp.write(vid_bytes)
                                    tmp.close()
                                    content_items.append({"type": "video", "video": tmp.name, "nframes": MAX_NFRAMES})
                                else:
                                    content_items.append({"type": "video", "video": url, "nframes": MAX_NFRAMES})
                                has_vision = True
                            elif item.get("type") == "text":
                                content_items.append(item)
                            else:
                                content_items.append(item)
                        else:
                            content_items.append(item)
                    qwen_messages.append({"role": msg.role, "content": content_items})

            max_tokens = request.max_tokens or MAX_NEW_TOKENS_DEFAULT

            if has_vision:
                from qwen_vl_utils import process_vision_info

                text_prompt = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(qwen_messages)
                inputs_dict = processor(
                    text=[text_prompt],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    return_tensors="pt"
                )

                # Signal other ranks: vision request
                signal = torch.tensor([1, max_tokens], dtype=torch.long, device=NEURON_DEVICE)
                dist.broadcast(signal, src=0)
                torch.save(inputs_dict, "/tmp/current_request.pt")
                dist.barrier()

                response_text = run_vision_inference(inputs_dict, max_tokens)
            else:
                text_prompt = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
                input_ids = processor(text=[text_prompt], return_tensors="pt")["input_ids"].to(NEURON_DEVICE)

                # Signal other ranks: text request
                signal = torch.tensor([0, max_tokens], dtype=torch.long, device=NEURON_DEVICE)
                dist.broadcast(signal, src=0)
                # Broadcast input_ids shape and data
                seq_len = torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=NEURON_DEVICE)
                dist.broadcast(seq_len, src=0)
                dist.broadcast(input_ids, src=0)

                response_text = run_text_inference(input_ids, max_tokens)

            elapsed = time.time() - start_time
            print(f"  Request ({('vision' if has_vision else 'text')}): {elapsed:.2f}s")

            return ChatResponse(
                choices=[ChatChoice(message={"role": "assistant", "content": response_text})]
            )

    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print(f"\n[SERVER] Running on port 8000")
    print(f"  POST /v1/chat/completions — image_url, video_url supported")
    print(f"  GET  /v1/models")
    print(f"  GET  /health, /readiness")

    # Keep rank 0 alive
    while True:
        time.sleep(1)

else:
    # Non-rank-0: wait for signals from rank 0
    while True:
        try:
            signal = torch.tensor([0, 0], dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(signal, src=0)
            request_type = signal[0].item()
            max_tokens = signal[1].item()

            if request_type == 1:
                # Vision request — load inputs from file
                dist.barrier()
                inputs_dict = torch.load("/tmp/current_request.pt", weights_only=False)
                run_vision_inference(inputs_dict, max_tokens)
            else:
                # Text request — receive input_ids via broadcast
                seq_len = torch.tensor([0], dtype=torch.long, device=NEURON_DEVICE)
                dist.broadcast(seq_len, src=0)
                input_ids = torch.zeros(1, seq_len.item(), dtype=torch.long, device=NEURON_DEVICE)
                dist.broadcast(input_ids, src=0)
                run_text_inference(input_ids, max_tokens)

        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")
            time.sleep(1)
