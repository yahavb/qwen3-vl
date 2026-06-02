# Qwen3-VL-8B-Instruct on PyTorch Native (Neuron)

## Summary

Running Qwen3-VL-8B-Instruct with `torch.compile(backend='neuron')` on AWS Trainium2 using tensor parallelism.

### Results — TP-4 Cross-Content (LATEST, commit 2ae56f9)

✅ **Compile once → reuse forever CONFIRMED** across different content!

| Metric | Value |
|--------|-------|
| Image warmup (compile with image-2.png) | 5373s |
| Image cached (avg 5 runs with **image-3.png**) | **278.67s** |
| Image speedup | **19.3x** |
| Video warmup (compile with tmpmmr4w9n6.mp4) | 5784s |
| Video cached (avg 5 runs with **tmpugehbe0r.mp4**) | **261.06s** |
| Video speedup | **22.2x** |
| All 5 image runs cached? | ✅ YES |
| All 5 video runs cached? | ✅ YES |

**Key findings:**
- **Different content, same shape = NO recompilation!** This is the demo use case.
- Image: compile once with any 512×512 image → serve any new 512×512 image at ~279s/256 tokens (~0.9 tok/s TP-4 with profiling overhead)
- Video: compile once with any 4-frame video → serve any new 4-frame video at ~261s/256 tokens (~1.0 tok/s TP-4 with profiling overhead)
- Profiling overhead adds ~40% to inference time (278s vs 195s without profiling)
- Profile artifacts saved at `/var/mdl/qwen3_vl/profiles/cross_content_20260519_200344/`

### Results — TP-4 Same-Content (commit d4b7e4e, no profiling)

| Metric | Value |
|--------|-------|
| Image warmup (compilation) | 5250s |
| Image cached (avg of 5 runs, same image) | **194.81s** |
| Image speedup | **27.0x** |
| Video warmup (compilation) | — |
| Video cached | — |

**Without profiling overhead:** ~195s per image inference (256 tokens, ~1.3 tok/s)

### Results — TP-4 with Static Shapes (older, commit 9186be9)

| Metric | Value |
|--------|-------|
| Image warmup (compilation) | 2800s |
| Image cached (avg of 3 runs) | 42.5s |
| Image speedup (cached vs compile) | **65.9x** |
| Video warmup (compilation) | 3124s |
| Video cached (avg of 3 runs) | 108.7s |
| Video speedup (cached vs compile) | **28.7x** |
| User image (reuse compiled graph) | 2907s ⚠️ |
| User video (reuse compiled graph) | 2333s ⚠️ |

**Key findings (old run):**
- ⚠️ User image/video triggered recompilation because max_new_tokens differed (128 vs 256)
- Fixed in later commits by standardizing max_new_tokens=256 and using same-shape PVC content

### Results — TP-8 baseline (original, no static shapes)

| Metric | Value |
|--------|-------|
| Image RUN 1 (with compilation) | 1471s |
| Image RUN 2 (cached) | 32s |
| Video RUN 1 (with compilation) | 1882s |
| Video RUN 2 (cached) | 38s |
| Video RUN 3 (cached) | 38s |
| Image speedup (cached vs compile) | 46.4x |
| Video speedup (cached vs compile) | 49.2x |

**Comparison notes:**
- TP-8 baseline: 32s image, 38s video (but 800+ compiled graphs, every new input recompiles)
- TP-4 static: 42.5s image, 108.7s video (fewer cores, but same-shape inputs reuse graphs)
- The padded bucket approach works for repeated same-prompt calls but user inputs still miss the cache

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Qwen3-VL-8B-Instruct                              │
│  ┌───────────┐  ┌──────────────────────────────┐   │
│  │ Vision    │  │ LLM Decoder (36 layers)      │   │
│  │ Encoder   │  │ TP-8 sharded:                │   │
│  │ (ViT)     │  │   Q: 32→4 heads/rank         │   │
│  │ replicated│  │   KV: 8→1 head/rank          │   │
│  │ on all    │  │   gate/up: col-parallel       │   │
│  │ ranks     │  │   down/o: row-parallel        │   │
│  └───────────┘  │   lm_head: col-parallel       │   │
│                  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### TP Sharding Details

| Layer | Dimension | TP-8 per rank |
|-------|-----------|---------------|
| Q proj | [4096, 4096] | [512, 4096] (4 heads × 128) |
| K proj | [1024, 4096] | [128, 4096] (1 KV head × 128) |
| V proj | [1024, 4096] | [128, 4096] |
| O proj | [4096, 4096] | [4096, 512] (row-parallel) |
| gate_proj | [12288, 4096] | [1536, 4096] |
| up_proj | [12288, 4096] | [1536, 4096] |
| down_proj | [4096, 12288] | [4096, 1536] (row-parallel) |
| lm_head | [151936, 4096] | [18992, 4096] |

### Collectives

- `all_reduce(SUM)` after self_attn output (before residual add)
- `all_reduce(SUM)` after MLP output (before residual add)
- `all_gather` for lm_head logits (reconstruct full vocab)

---

## TP Scaling Analysis

### TP-8 (current — `m-trn2`)

- **Resource:** 2 Neuron devices × LNC=2 = 4 cores/device × 2 devices = **8 NeuronCores**
- **Status:** ✅ Working, validated

### TP-16 (potential — `l-trn2`)

- **Resource:** 4 Neuron devices × LNC=2 = 4 cores/device × 4 devices = **16 NeuronCores**
- **Status:** ❌ **NOT POSSIBLE** for Qwen3-VL-8B

**Blocker:** `num_kv_heads = 8` is NOT divisible by 16.

With GQA (Grouped Query Attention), you cannot split 8 KV heads across 16 ranks (0.5 head/rank is invalid).

| Dimension | ÷ 8 | ÷ 16 | Divisible by 16? |
|-----------|-----|------|------------------|
| Q heads: 32 | 4 | 2 | ✅ |
| KV heads: 8 | 1 | 0.5 | ❌ |
| intermediate: 12288 | 1536 | 768 | ✅ |
| hidden: 4096 | 512 | 256 | ✅ |
| vocab: 151936 | 18992 | 9496 | ✅ |

**Options for higher parallelism:**
1. **TP-8 is maximum** for Qwen3-VL-8B (8 KV heads ÷ 8 = 1 per rank minimum)
2. **GQA-aware TP-16**: Replicate KV projections across rank pairs (complex, non-standard)
3. **Use Qwen3-VL-72B**: Has `num_kv_heads=64`, divisible by 16/32/64

---

## DRA Resource Claims

| Claim | Devices | LNC | Total Cores | TP Degree |
|-------|---------|-----|-------------|-----------|
| `s-trn2` | 1 | 2 | 4 | TP-4 (won't work for 8B: KV heads not divisible) |
| `m-trn2` | 2 | 2 | 8 | TP-8 ✅ |
| `l-trn2` | 4 | 2 | 16 | TP-16 (blocked by KV heads) |

---

## Serving Architecture

```
┌─ Service: qwen3-vl:8000 ─────────────────────────┐
│                                                    │
│  Deployment (replicas: 1)                          │
│  ┌──────────────────────────────────────────────┐  │
│  │ Pod (m-trn2: 8 NeuronCores)                  │  │
│  │                                              │  │
│  │ torchrun --nproc_per_node=8                  │  │
│  │   ├─ Rank 0: FastAPI server (port 8000)      │  │
│  │   │           + model inference               │  │
│  │   ├─ Rank 1-7: model inference only          │  │
│  │   └─ All ranks: torch.compile + all_reduce   │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### API Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible (image + video)
- `GET /health` — Liveness
- `GET /readiness` — Model loaded + warmed up

### Warmup Strategy

At startup, the server runs:
1. One image inference (compiles for image seq_len)
2. One video inference with 4 frames (compiles for video seq_len)

After warmup (~3400s first time, instant with NEFF cache), the server marks ready.

---

## Neuron Explorer Profile — BASELINE (before static shapes)

**Profile session:** `20260517_103017` on `i-0212e98f28fe0ba79` (trn2.48xlarge)  
**Workload:** TP-4, image + video inference with `model.generate()` (dynamic shapes)

### Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Total wall time | 24,295 seconds | 6.7 hours profiled run |
| Total NeuronCore exec time | 0.996 seconds | 🔴 Near-zero |
| Total active time | 0.677 seconds | 🔴 0.000000108% utilization |
| NEFF executions | 7,291,122 | 🔴 Extreme dispatch overhead |
| Unique NEFFs compiled | 800+ | 🔴 Extreme graph fragmentation |
| Dropped notifications | After 123,435 of 7,291,122 | Buffer overflow |
| MFU (estimated) | 0.0000000024% | 🔴 Catastrophically low |
| Max achievable MFU | 0.014% | Not compute-bound |

### Hardware Utilization Breakdown

| Engine | Instruction Time (s) | % of Exec | Active Time (s) |
|--------|---------------------|-----------|-----------------|
| GPSIMD (scalar) | 0.584 | 58.6% | 0.538 |
| Tensor Engine (matmul) | 0.387 | 38.8% | 0.081 |
| Vector Engine | 0.147 | 14.7% | 0.091 |
| Scalar Engine | 0.238 | 23.9% | 0.143 |
| Sync Engine | 0.065 | 6.5% | 0.054 |
| CC (collective compute) | 0.044 | 4.4% | 0.002 |

**Key finding:** GPSIMD dominates (58%) over Tensor Engine (39%). Most time is in scalar ops (reshapes, masks, indexing) not matmuls.

### Memory & Data Movement

| Metric | Value |
|--------|-------|
| HBM reads | 49.5 GB |
| HBM writes | 15.6 GB |
| Weight size | 1.8 GB |
| Total inputs+outputs+weights moved | 309 GB |
| SBUF writes | 44.7 GB |
| SBUF reads | 11.2 GB |
| Matmul arithmetic intensity | 82,164 |

### System Events

| Event | Count |
|-------|-------|
| nrt_dma_mem_dealloc | 524,139 |
| nrt_tensor_free | 149 |
| Trace events | 13,732,233 |
| Total events | 7,960,610 |

### Root Causes Identified

1. **800+ compiled subgraphs** — `model.generate()` with dynamic shapes causes recompilation for every unique sequence length
2. **7.3M NEFF dispatches** — Each token decode dispatches 800+ kernels sequentially
3. **99.99997% idle** — Host-side Python overhead between dispatches dwarfs compute
4. **GPSIMD dominance** — Attention masking, reshapes, and indexing run on scalar cores, not fused into tensor engine
5. **524K memory deallocations** — Constant alloc/dealloc of tiny buffers between fragmented ops

### Fix Applied (commit 61b1ead)

- Pad inputs to fixed bucket sizes (512 for image, 2048 for video)
- Use `cache_implementation="static"` for fixed KV cache shapes during decode
- Resize all inputs to 512×512
- Expected: ~3-5 compiled graphs instead of 800+

---

## Key Learnings

1. **`do_sample=False` is REQUIRED** — With TP, sampling randomness causes rank divergence → mismatched collectives → crash
2. **Each unique seq_len triggers recompilation** — Different input sizes (image vs video) compile separately
3. **Video decode must be serialized** — Multiple ranks decoding video with av/swscaler simultaneously causes ENOMEM
4. **Vision encoder runs on Neuron but is NOT compiled** — Only LLM decoder layers + lm_head use torch.compile
5. **`aten::contiguous` warnings are benign** — View operators fall back correctly on Neuron

---

## How to Profile

### Step 1: Enable profiling in the inference job

Add these environment variables to the container spec:

```yaml
env:
  - name: NEURON_RT_INSPECT_ENABLE
    value: "1"
  - name: NEURON_RT_INSPECT_OUTPUT_DIR
    value: "/tmp/neuron_profile"
  - name: NEURON_RT_INSPECT_DEVICE_PROFILE
    value: "session"
```

Then at the end of the job script, copy artifacts to PVC:

```bash
PROFILE_DEST="/var/mdl/qwen3_vl/profiles/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DEST"
cp -r /tmp/neuron_profile/* "$PROFILE_DEST/" 2>/dev/null || true
```

### Step 2: Analyze with neuron-explorer

Use the `neuron-explorer-job.yaml` to run analysis on the profile data:

```bash
# Apply the explorer job
kubectl delete job neuron-explorer --ignore-not-found
kubectl apply -f neuron-explorer-job.yaml

# View results (logs persist after job completion)
kubectl logs job/neuron-explorer
```

### Step 3: neuron-explorer CLI commands

```bash
# Summary text output (human-readable per-model stats)
neuron-explorer view -d <profile_dir> --output-format summary-text

# JSON output (for scripting/parsing)
neuron-explorer view -d <profile_dir> --output-format json --output-file output.json

# Skip invalid DMA trace errors (common with tiny ops)
neuron-explorer view -d <profile_dir> --output-format summary-text --ignore-dma-trace

# Launch interactive UI (requires port-forward)
neuron-explorer view -d <profile_dir>
```

### Step 4: Understanding NEFF files

NEFFs (Neuron Executable File Format) are compiled graph kernels. The profile directory contains:

```
<profile_dir>/
├── *.ntff          # Neuron Trace Format Files (execution timeline)
├── *.neff          # Compiled kernels (one per unique graph)
└── metadata/       # Session info
```

**Key analysis commands:**

```bash
# Count total unique compiled graphs
ls <profile_dir>/*.neff | wc -l

# Size distribution (large = good fused ops, tiny = fragmented)
ls -lhS <profile_dir>/*.neff | head -20   # Largest
ls -lhS <profile_dir>/*.neff | tail -20   # Smallest

# Total NEFF sizes
ls -l <profile_dir>/*.neff | awk '{sum+=$5} END {printf "%.2f MB\n", sum/1024/1024}'
```

**What to look for:**

| Indicator | Good | Bad |
|-----------|------|-----|
| Total NEFFs | 3-10 | 100+ |
| NEFF size distribution | Most >1MB (fused ops) | Mostly <10KB (fragmented) |
| NEFF executions | ~thousands | millions (dispatch overhead) |
| total_active_time_percent | >50% | <1% (idle/overhead-bound) |
| tensor_engine vs gpsimd | tensor > gpsimd | gpsimd > tensor (scalar fallback) |
| MFU | >10% | <1% |

### Step 5: Interpreting summary-text output

Key metrics from `summary-text`:

```
total_time              — Wall clock time for entire profile session
total_exec_time         — Time spent executing on NeuronCores
total_active_time       — Time cores were actually computing (subset of exec)
total_active_time_percent — CRITICAL: should be >50%, <1% means overhead-bound

tensor_engine_instruction_time  — Time in matmul ops (want this high)
gpsimd_engine_instruction_time  — Time in scalar ops (want this low)

mfu_estimated_percent           — Model FLOPS Utilization
mm_arithmetic_intensity         — FLOPS/byte for matmuls (higher = better)

event_count            — Total profiled events
trace_count            — Total trace entries
```

---

## Files

| File | Purpose |
|------|---------|
| `qwen3-vl-job.yaml` | Batch job (TP-4, static shapes, image + video) |
| `qwen3-vl-deploy.yaml` | Serving deployment (TP-8, FastAPI, m-trn2) |
| `neuron-explorer-job.yaml` | Profile analysis job (runs neuron-explorer on PVC data) |
| `QWEN3_VL_NEURON.md` | This document |
