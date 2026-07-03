# caption-endpoint branch — Qwen3-VL for Kinetics video captioning

This branch adds a **dedicated Qwen3-VL captioning endpoint** used to generate a
motion-focused prompt corpus from the Kinetics dataset (Stage 1 of a WAN2.2 →
StreamDiffusion-v2 distillation project). It branches from `fused-layer-compile`
(the deployed eval server) and leaves that server untouched.

Client-side tooling lives in `github.com/yahavb/video-distill`.

## What this branch changes vs `fused-layer-compile`

```
 caption-vl-deploy.yaml | +168   (new: separate Service + Deployment)
 vision.py              |  ±62   (fix: video deepstack / visual_mask sizing)
```

### 1. `vision.py` — video deepstack / visual-mask size fix

**Bug:** captioning arbitrary-resolution Kinetics clips returned HTTP 500 with
either `hidden[0, visual_mask] += deepstack_embeds` size mismatch (e.g. 683 vs
1152) or `shape mismatch: value tensor [1152,4096] -> [683,4096]`.

**Root cause:** the Neuron vision path is a **static-shape compiled NEFF** — it
emits a constant visual-token count (1152) for the one grid it compiled at
warmup. But the CPU-side processor computes the `<video>` placeholder count from
each clip's real `video_grid_thw`, so it varies per clip (683, 692, …). When the
placeholder count ≠ the vision output count, `prepare_vision_embeds`' embed-merge
and the decoder's deepstack-add mismatch and crash.

**Fix (in `prepare_vision_embeds`, video branch):** reconcile everything to
`n_fill = min(video_features.shape[0], num_video_positions)`. Fill exactly
`n_fill` placeholder positions with the first `n_fill` main features, mark only
those in `visual_mask`, and truncate the deepstack to the same `n_fill` — so main
features, mask positions, and deepstack rows always match at decode time. Handles
video-only (the captioning case) and mixed image+video requests.

**Note:** with the client normalizing every clip to a fixed 448×448 (see below),
the grid is constant and this reconciliation is effectively a no-op backstop —
it stays as defense against any un-normalized input rather than crashing.

### 2. `caption-vl-deploy.yaml` — dedicated caption endpoint

New `caption-qwen3-vl` **Service + Deployment** (namespace `default`, port 8000,
svc DNS `caption-qwen3-vl.default.svc.cluster.local`). Identical to the eval
`qwen3-vl` deploy except:
- names → `caption-qwen3-vl`
- clones **this** branch (`caption-endpoint`) so it runs the vision fix
- leaves the existing `qwen3-vl` eval Service/Deployment completely untouched

Still TP-8 (`serve_deploy_fused.py` asserts `world_size == 8`); one trn2 chip per
replica via the `s-lnc1-trn2` resource claim. Scaling throughput = more replicas
(each its own chip), matched by a higher client `--concurrency`.

## The critical operational insight: fixed input resolution

A static compiled vision graph is only correct for the grid it compiled. The
robust fix for heterogeneous Kinetics resolutions is **client-side**: re-encode
every clip to a fixed **448×448** before sending (`caption_kinetics.py --resize
448` in video-distill). Then the server compiles ONE video NEFF on the first
request (~59s) and reuses it for all others (~10s each). This is what turned the
run from ~35–65% HTTP 500s into 100% success.

- 448 / 28 (patch·merge) = 16×16 grid → 512 vision tokens at 4 frames, well under
  `MAX_SEQ_LEN=4096`. 8 frames = 1024 tokens, still fits.
- `QWEN3_VL_MAX_NFRAMES=4` gives good prose motion/camera descriptions; the
  categorical motion field is weak at ~2.5s frame spacing (bump to 8 if needed).

## Deploy

```bash
kubectl apply -f caption-vl-deploy.yaml
kubectl get pods -l app=caption-qwen3-vl -w      # wait Ready (warmup + compile)
```

## Result

Used to caption the full k400 validation split: **19,881 clips → ~19,877
prompts, 4 corrupt-clip failures (0.02%)**, ~7.1s/clip effective with 2 replicas
+ client concurrency 2. See video-distill for the client, jobs, and corpus.
