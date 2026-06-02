# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-VL-8B-Instruct running on AWS Trainium2 via PyTorch Native with `torch.compile(backend='neuron')` and manual tensor parallelism. This is NOT a standard HuggingFace inference setup — the model is manually sharded, compiled per-layer, and wrapped with TP collective modules.

## Running

All scripts are launched with `torchrun` on Trainium2 nodes (Kubernetes):

```bash
# TP-4 serving (primary deployment)
torchrun --nproc_per_node=4 serve_tp4.py

# TP-8 serving
torchrun --nproc_per_node=8 serve_tp8.py

# TP-8 batch inference (runs, benchmarks, exits)
torchrun --nproc_per_node=8 run_tp8.py

# TP-16 benchmark (GQA-aware KV replication)
torchrun --nproc_per_node=16 serve_tp16.py
```

Deploy to Kubernetes:
```bash
kubectl apply -f qwen3-vl-deploy.yaml   # TP-4 serving deployment
kubectl apply -f qwen3-vl-job.yaml      # Batch inference job
```

## Architecture

Each `serve_tp*.py` is a self-contained script (no shared modules) that:
1. Initializes `dist.init_process_group(backend="neuron")`
2. Loads full model on CPU, shards weights in-place per rank
3. Moves to Neuron device, wraps attn/MLP/lm_head with `torch.compile(backend='neuron', dynamic=False)`
4. Wraps compiled modules with TP wrappers (`TPAttention`, `TPMLP`, `TPLMHead`) that call `all_reduce`/`all_gather` outside the compiled graph
5. Runs warmup inferences (triggers Neuron compilation)
6. Rank 0 starts a FastAPI server; other ranks loop on `dist.broadcast` waiting for work

The TP wrappers are defined at the top of each file — they are NOT imported from a shared library.

## Key Constraints

- **`do_sample=False` is mandatory** — sampling causes rank divergence in TP, leading to mismatched collectives and hangs
- **Input shapes determine compilation** — each unique sequence length triggers a new Neuron compilation (~45 min). Same-shape inputs reuse cached NEFFs
- **Vision encoder is NOT compiled** — only the LLM decoder layers and lm_head go through `torch.compile`
- **Video decode serialization** — only rank 0 processes video with `qwen_vl_utils`; the processed tensors are saved to disk and loaded by all ranks
- **TP-16 requires GQA-aware sharding** — 8 KV heads can't evenly divide across 16 ranks, so rank pairs share a KV head (non-standard)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `/tmp/Qwen3-VL-8B-Instruct` | Local path to model weights |
| `QWEN3_VL_MAX_NFRAMES` | `4` | Max video frames (affects compilation shape) |
| `QWEN3_VL_MAX_NEW_TOKENS` | `256` | Max generation length |
| `NEURON_CC_FLAGS` | `--model-type=transformer` | Neuron compiler flags |

## API

Serving scripts expose an OpenAI-compatible endpoint:
- `POST /v1/chat/completions` — supports image_url and video_url content types
- `GET /health` — liveness
- `GET /readiness` — model loaded and warmed up

## DRA Resource Claims

| Claim | Neuron Devices | Cores | TP Degree |
|-------|---------------|-------|-----------|
| `s-trn2` / `s-lnc2-trn2` | 1 | 4 | TP-4 |
| `m-trn2` | 2 | 8 | TP-8 |
| `l-trn2` | 4 | 16 | TP-16 |

## Dependencies

Installed at runtime in the container (not a requirements.txt):
```
transformers qwen-vl-utils accelerate Pillow av
fastapi uvicorn python-multipart
torchvision (from github, v0.25.0, no-deps)
```
