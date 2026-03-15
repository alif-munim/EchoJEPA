# Probe Training Efficiency

Optimizations for speeding up the Nature Medicine probe runs (18 UHN tasks x 5+ models = 90+ runs).

## HP Grid Narrowing (2026-03-13)

### Analysis

Extracted per-head val MAE from TAPSE (4 models complete) and LVEF (EchoJEPA-G complete) checkpoints. The multihead system trains 20 parallel probe heads with different (LR, WD) combos, selecting the best per model.

**Original grid (20 heads):** 5 LRs x 4 WDs

```
LR: [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
WD: [0.001, 0.01, 0.1, 0.4]
```

### Per-Head Results — TAPSE (val MAE in cm, lower = better)

| Head | LR | WD | EchoJEPA-G | EchoJEPA-L | EchoPrime | PanEcho |
|------|----|----|------------|------------|-----------|---------|
| 0 | 1e-3 | 0.001 | 0.271 | 0.338 | 0.327 | 0.348 |
| 1 | 1e-3 | 0.01 | 0.270 | 0.339 | 0.326 | 0.346 |
| 2 | 1e-3 | 0.1 | 0.271 | 0.334 | 0.327 | 0.350 |
| 3 | 1e-3 | 0.4 | 0.275 | 0.347 | 0.328 | 0.354 |
| 4 | 5e-4 | 0.001 | 0.268 | 0.332 | 0.324 | 0.353 |
| 5 | 5e-4 | 0.01 | 0.267 | 0.331 | 0.324 | 0.348 |
| 6 | 5e-4 | 0.1 | 0.268 | 0.332 | 0.325 | 0.346 |
| 7 | 5e-4 | 0.4 | 0.271 | 0.335 | 0.326 | 0.350 |
| 8 | 1e-4 | 0.001 | 0.266 | **0.325** | 0.322 | 0.341 |
| 9 | 1e-4 | 0.01 | 0.265 | 0.326 | 0.322 | 0.341 |
| 10 | 1e-4 | 0.1 | 0.265 | 0.326 | 0.322 | 0.343 |
| 11 | 1e-4 | 0.4 | 0.265 | 0.329 | 0.324 | 0.341 |
| 12 | 5e-5 | 0.001 | 0.264 | 0.328 | 0.321 | 0.340 |
| 13 | 5e-5 | 0.01 | **0.264** | 0.329 | 0.321 | 0.340 |
| 14 | 5e-5 | 0.1 | 0.264 | 0.329 | 0.321 | 0.341 |
| 15 | 5e-5 | 0.4 | 0.264 | 0.330 | 0.322 | 0.341 |
| 16 | 1e-5 | 0.001 | 0.265 | 0.337 | 0.321 | **0.339** |
| 17 | 1e-5 | 0.01 | 0.266 | 0.338 | **0.321** | 0.340 |
| 18 | 1e-5 | 0.1 | 0.267 | 0.339 | 0.321 | 0.340 |
| 19 | 1e-5 | 0.4 | 0.266 | 0.338 | 0.321 | 0.340 |

**Winners:** G→head 13 (LR=5e-5, WD=0.01), L→head 8 (LR=1e-4, WD=0.001), EchoPrime→head 17 (LR=1e-5, WD=0.01), PanEcho→head 16 (LR=1e-5, WD=0.001).

### Per-Head Results — LVEF EchoJEPA-G (val MAE in %, lower = better)

Best: head 13 (LR=5e-5, WD=0.01) MAE=4.462. Same winner as TAPSE.

Average by LR: 1e-3 → 4.616, 5e-4 → 4.545, 1e-4 → 4.483, 5e-5 → **4.468**, 1e-5 → 4.536.

### Key Findings

1. **LR matters most, WD matters little.** LR explains 3-5x more variance than WD across all models.

2. **Optimal LR differs by model:**
   - EchoJEPA-G/L: prefer LR=5e-5 to 1e-4 (mid-range)
   - EchoPrime/PanEcho: prefer LR=1e-5 (lowest)
   - No model prefers LR=1e-3 (highest) on any task

3. **LR=1e-3 consistently worst.** Always the worst LR tier across all 4 models and both tasks.

4. **WD=0.4 consistently worst.** Always the worst WD tier. Never wins.

5. **WD=0.001 and WD=0.01 essentially tied.** Difference < 0.002 cm on TAPSE across all models.

### Recommended Grid: 12 heads (4 LRs x 3 WDs)

```
LR: [5e-4, 1e-4, 5e-5, 1e-5]
WD: [0.001, 0.01, 0.1]
```

**Dropped:** LR=1e-3 (always worst), WD=0.4 (always worst).

All 4 models' TAPSE winners and EchoJEPA-G's LVEF winner survive. **40% fewer heads → 40% less training compute.**

### More Aggressive: 8 heads (4 LRs x 2 WDs)

```
LR: [5e-4, 1e-4, 5e-5, 1e-5]
WD: [0.001, 0.01]
```

Also drops WD=0.1. All winners still survive. **60% less compute.** Slightly riskier on unseen tasks where WD=0.1 might matter.

### Fairness Note

Different models selecting different optimal HPs from the same grid is expected and desired — the grid IS the fairness mechanism. Fixing a single (LR, WD) for all models would favor whichever model happens to match that setting. The key constraint is that the grid covers every model's optimal region.

## Adopted Efficiency Settings

### Increase Batch Size — VERIFIED, ADOPTED

**Verified 2026-03-13.** BS2 × 4 GPUs × 12 heads uses only **5.6 GB/GPU** (ViT-L). Far below the 80 GB limit.

| Model | Heads | Memory/GPU (BS1×8GPU, 20h) | Memory/GPU (BS2×4GPU, 12h) |
|-------|-------|---------------------------|---------------------------|
| EchoJEPA-L (ViT-L) | 12 | ~7.2 GB (20 heads) | **5.6 GB** |
| PanEcho (ConvNeXt-T) | 12 | ~9.7 GB (20 heads) | ~7 GB (est.) |
| EchoJEPA-G (ViT-G) | 12 | ~15-20 GB (est.) | ~12 GB (est.) |

Note: BS1 memory was higher because it ran 20 heads. BS2 × 12 heads is actually lower memory per GPU despite doubling batch size, because 12/20 = 0.6× fewer probe instances.

BS2 × 4 GPUs gives eff_bs=8, identical to BS1 × 8 GPUs. Training dynamics are unchanged — no HP re-validation needed. Adopted for all remaining Nature Medicine probe runs.

### Reduce num_segments — NOT RECOMMENDED
Current: `num_segments: 2` (2 non-overlapping temporal crops per clip). Each crop spans `frames_per_clip(16) × frame_step(2) = 32 raw frames`. With NS=2, the model sees ~64 raw frames across 2 crops, covering approximately 2 cardiac cycles.

Critically, segments are **not** late-fused via prediction averaging. The `ClipAggregation` wrapper (`vit_encoder_multiclip.py:132-135`) passes each clip through the frozen encoder independently, then concatenates their output tokens along the temporal dimension: `torch.cat(outputs, dim=1).flatten(1, 2)` → `[B, num_clips × T × S, D]`. For ViT models with NS=2, this produces 2 × 1568 = 3136 tokens. The d=1 attentive probe then cross-attends over all 3136 tokens simultaneously, enabling the probe to reason across both cardiac cycles in a single forward pass.

(Late prediction averaging applies to `num_views_per_segment` — spatial augmentation views — which is set to 1 during training. It also applies to multi-clip study-level aggregation at inference. Neither is affected by `num_segments`.)

Reducing to NS=1 would halve encoder forward passes per sample but restrict the probe to 1568 tokens from ~32 frames (≈1 cardiac cycle), eliminating cross-beat reasoning. **This is clinically inappropriate for echocardiography.** Multi-beat coverage is essential for:
- **Wall motion assessment**: Regional abnormalities may be intermittent or load-dependent; comparing across beats improves sensitivity
- **Valve severity grading**: Beat-to-beat variation in regurgitant jets, especially in atrial fibrillation, requires multi-cycle assessment
- **Diastolic function**: E/A ratios, deceleration time, and tissue Doppler parameters are assessed across multiple beats per clinical guidelines
- **Arrhythmia detection**: Rhythm irregularities, ectopic beats, and conduction abnormalities require temporal context beyond a single cycle

The ~2x encoder speedup does not justify the clinical information loss. **Keep `num_segments: 2`.**

### Reduce Epochs — ADOPTED (15 epochs)
TAPSE best epochs were 15 (G) and 18 (L/EchoPrime/PanEcho). Adopted 15 epochs for all remaining runs. The 4 already-complete models (TAPSE G/L/EchoPrime, LVEF G) ran 20 epochs; their epoch-15 results can be extracted from checkpoints for consistent reporting.

### GPU Concurrency (2 jobs × 4 GPUs) — ADOPTED

Split the 8×A100 node into 2 concurrent jobs of 4 GPUs each. Combined with BS2 (eff_bs=8 maintained), this gives ~2× throughput vs sequential.

| Config | GPUs/job | BS | eff_bs | iters/ep (25K studies) | Jobs in parallel | Net throughput |
|--------|----------|-----|--------|----------------------|-----------------|----------------|
| Current | 8 | 1 | 8 | ~3,125 | 1 | 1× |
| Split, BS1 | 4 | 1 | 4 | ~6,250 | 2 | ~1× (each job 2× slower, but 2 concurrent) |
| **Split, BS2** | **4** | **2** | **8** | **~3,125** | **2** | **~2×** |

The key insight: **BS2 × 4 GPUs has the same effective batch size (8) as BS1 × 8 GPUs**, so training dynamics are identical — no HP re-validation needed beyond confirming BS2 fits in memory. Each job takes roughly the same wall time as the current setup (same eff_bs, same iters/epoch), but two tasks run simultaneously → **~2× total throughput**.

**Implementation:** `run_uhn_probe.sh` supports `DEVICES`, `MASTER_PORT`, and `RESUME` environment variables:

```bash
# Terminal A: GPUs 0-3
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 \
  bash scripts/run_uhn_probe.sh --models "echojepa-g echojepa-l echomae" tapse

# Terminal B: GPUs 4-7
DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 \
  bash scripts/run_uhn_probe.sh --models "echoprime panecho" tapse

# Resume a killed job
RESUME=true DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 \
  bash scripts/run_uhn_probe.sh --models echojepa-l lvef
```

**CRITICAL: Exclusive GPU sets are mandatory.** On 2026-03-13 we observed what happens when 3 jobs share all 8 GPUs without isolation:

| Job | Model | Time/epoch (shared) | Expected solo | Slowdown |
|-----|-------|--------------------|--------------:|----------|
| TAPSE PanEcho | PanEcho (42M) | ~23 min | ~18 min | 1.3× |
| LVEF EchoJEPA-L | ViT-L (304M) | **~86 min** | ~18 min | **~5×** |
| LVEF EchoJEPA-L-K | ViT-L (304M) | **~89 min** | ~18 min | **~5×** |

Three jobs on shared GPUs gave ~5× slowdown per ViT-L job, for **net ~0.6× throughput** — worse than running sequentially. The lightweight PanEcho job was only mildly affected (1.3×).

**Root cause: CPU data loading, not GPU memory.** With 3 jobs × 8 GPUs × 8 `num_workers` = 192 data loader workers on 96 CPU cores, the loaders couldn't keep up with the GPUs. The ViT-L encoder finishes its forward pass faster than data arrives, so GPUs starve. PanEcho's tiny encoder (42M) is slow enough that even degraded data loading keeps it fed. GPU memory was fine (total ~24GB/GPU across all 3 jobs).

**Lesson:** Always use `CUDA_VISIBLE_DEVICES` or exclusive `--devices` sets to prevent GPU sharing between jobs.

**Potential contention with proper isolation (exclusive GPUs):**
- **CPU/data loading:** The main risk. With `num_workers: 8` and 4 GPUs/job, each job uses 32 data loader workers. Two jobs = 64 workers on 96 cores — safe. Rule of thumb: **total workers ≤ CPU cores (96)**. At 3 concurrent jobs it would be 96 workers, the limit.
- **NVLink bandwidth:** DDP all-reduce for the d=1 probe is tiny (~2M params, ~8MB). Two jobs on separate GPU sets sharing the NVLink fabric should have negligible contention.
- **GPU memory:** With exclusive GPU sets, no cross-job sharing. At BS2 + ViT-L, ~14GB/GPU. Even ViT-G at BS2 is ~30-40GB/GPU — well within 80GB.
- **TCPStore ports:** Each DDP job needs a unique `MASTER_PORT`. The run script now accepts this as an env var.

**4 jobs × 2 GPUs:** Possible for ViT-L (BS4 × 2 GPUs = eff_bs 8, ~28GB/GPU). Not safe for ViT-G (~60-80GB/GPU). CPU contention worse with 4 jobs. Consider for non-ViT-G models only if 2×4 proves stable.

### Pre-cache Encoder Features
Extract and cache frozen encoder outputs, then train probes on cached features. Would eliminate redundant encoder forward passes across epochs. **Not planned** — breaks the prediction averaging pipeline and adds complexity.

## Final Recommendations — ALL ADOPTED (2026-03-13, updated 2026-03-14)

| # | Change | Speedup | Risk | Status |
|---|--------|---------|------|--------|
| 1 | GPU concurrency: 2 jobs × 4 GPUs, BS2 | **~2× total throughput** | None — eff_bs unchanged (8) | **ADOPTED** ✓ |
| 2 | HP grid 20→12 heads | ~35% probe compute | None — all models' optima preserved | **ADOPTED** ✓ |
| 3 | Epochs 20→15 | ~25% wall time | Low — TAPSE best epochs were 15-18 | **ADOPTED** ✓ |
| 4 | Increase `val_batch_size` 16→64 | Negligible | None — see note below | Set to 64, neutral impact |
| 5 | Reduce `num_segments` 2→1 | ~2x encoder speed | **High** — clinically inappropriate | **DO NOT ADOPT** |
| 6 | Class-balanced study sampling (`--balance 3`) | **3-8× for imbalanced tasks** | None — no-op on balanced data | **ADOPTED** ✓ (default) |

### Observed Performance (BS2 × 4 GPUs × 12 heads × 15 epochs, verified 2026-03-13)

| Metric | Value |
|--------|-------|
| Memory/GPU (ViT-L, training) | **5.6 GB** (of 80 GB) |
| Memory/GPU (ViT-L, val @ BS64) | **17.3 GB** (peak, forward-only) |
| Iter rate (ViT-L) | **~280 iters/min** per job |
| TAPSE iters/epoch | 3,134 (eff_bs=8, ~13K studies) |
| LVEF iters/epoch | 2,956 (eff_bs=8, ~12K studies) |
| Epoch time (TAPSE, ViT-L, incl. val) | **14.7 min** (stable over 7 epochs) |
| Epoch time (LVEF, ViT-L, incl. val) | **14.0 min** (stable over 8 epochs) |
| Per-model time (15 epochs) | **~3.5-3.75 hrs** |
| GPU utilization | 52-83% |
| Data loader workers | 2 × 4 × 8 = 64 (of 96 CPU cores) |
| System load average | ~40 (2 concurrent jobs) |

**val_batch_size=64 note:** Tested vs val_batch_size=16. Val iterations are slightly faster (~2 min vs ~3 min for ~200 val iters), but peak memory jumps from 5.6→17.3 GB/GPU and checkpoint save time increases. Net epoch time unchanged (14.7 vs 14.3 min). Left at 64 since it's neutral — not harmful, not helpful. The bottleneck is study sampling: val uses 1 clip/study (~13K for TAPSE), so only ~200 val iters regardless of batch size.

### Previous Run: TAPSE + LVEF (launched 06:55 UTC 2026-03-13, COMPLETE)

Logs: `logs/jobA_tapse_lvef_vbs64_20260313_*.log`, `logs/jobB_lvef_vbs64_20260313_*.log`

### Class-Balanced Study Sampling (2026-03-14) — ADOPTED

**Problem:** Classification tasks with severe class imbalance (MR severity: 25.8× majority/minority, AS severity: 41.1×) inflate dataset size with redundant majority-class studies. Weighted CE adjusts gradients but doesn't fix wasted forward passes.

**Solution:** `class_balance_ratio` parameter in `DistributedStudySampler`. Caps each class at `ratio × min_class_count` studies. Deterministic downsample (seeded). Applied to train only — val is always unbalanced (full evaluation).

**Implementation:** 4 files changed:
- `src/datasets/study_sampler.py` — core balancing logic in `__init__`
- `src/datasets/video_dataset.py` — passes param through
- `src/datasets/data_manager.py` — passes param through
- `evals/video_classification_frozen/eval.py` — reads from config, passes to train loader only

**Default behavior:** `run_uhn_probe.sh` auto-sets `--balance 3` for classification + study_sampling tasks. Self-regulating: no-op on balanced datasets (cap only fires when a class exceeds 3× minority). Override with `--no-balance` or `--balance <N>`.

| Task | Studies (before) | Studies (after, 3×) | Batches/epoch | Reduction |
|------|-----------------|--------------------|--------------:|-----------|
| MR severity | 89,597 | 28,605 | 3,576 | **3.1×** |
| AS severity | 130,863 | 21,917 | 2,740 | **6.0×** |

### Current Run: MR + AS Severity (launched 07:22 UTC 2026-03-14)

**Job A** (GPUs 0-3, MASTER_PORT=29500): `mr_severity --balance 3`
**Job B** (GPUs 4-7, MASTER_PORT=29501): `as_severity --balance 3`
- 5 models each: echojepa-g, echojepa-l, echojepa-l-k, echoprime, panecho
- 15 epochs, BS2 × 4 GPUs, 12 HP heads

Logs:
- `logs/jobA_mr_severity_bal3_20260314_0722.log`
- `logs/jobB_as_severity_bal3_20260314_0722.log`

**Measured epoch times (ViT-G, epoch 1):**
- MR: ~16.7 min train + ~3 min val = **~20 min/epoch**
- AS: ~14 min train + ~3 min val = **~17 min/epoch**

**Estimated per-model times:**

| Model | Tokens | MR (Job A) | AS (Job B) |
|-------|--------|-----------|-----------|
| EchoJEPA-G | 1568 | 5.0 hrs | 4.25 hrs |
| EchoJEPA-L | 784 | 2.75 hrs | 2.35 hrs |
| EchoJEPA-L-K | 784 | 2.75 hrs | 2.35 hrs |
| EchoPrime | 1 | 1.5 hrs | 1.3 hrs |
| PanEcho | 1 | 1.5 hrs | 1.3 hrs |
| **Total** | | **13.5 hrs** | **11.5 hrs** |

**ETA: ~21:00 UTC Mar 14 (Sat ~9 PM)** — bounded by Job A (MR severity).

Compared to unbalanced run (killed after ~35 min):
- Unbalanced estimate was **~56 hrs** (Sun 2 PM)
- Balanced estimate is **~13.5 hrs** (Sat 9 PM) — **4.1× faster**

### Already-Complete Models (20 heads, 20 epochs, eff_bs=8)

These 4 models completed before the optimization. Their results are valid — the 12-head grid is a subset of their 20-head grid. For consistent reporting, extract the 12 relevant heads (indices [4,5,6,8,9,10,12,13,14,16,17,18] from the 20-head checkpoint) and use epoch-15 results.

| Task | Model | Epochs | Heads | eff_bs |
|------|-------|--------|-------|--------|
| TAPSE | echojepa-g | 20 | 20 | 8 |
| TAPSE | echojepa-l | 20 | 20 | 8 |
| TAPSE | echoprime | 20 | 20 | 8 |
| LVEF | echojepa-g | 20 | 20 | 8 |

Old 20-head checkpoints for re-run models backed up to `*.bak_20head/` directories.

### Rejected: num_segments reduction (#5)

At `num_segments: 2`, `ClipAggregation` concatenates encoder tokens from both temporal crops into a single 3136-token tensor that the probe cross-attends over simultaneously — this is early fusion, not late averaging (same design as original V-JEPA 2). The probe can reason across ~2 cardiac cycles in one pass. Reducing to NS=1 would cut this to 1568 tokens from ~1 cycle, eliminating cross-beat reasoning. For echocardiography, multi-beat assessment is clinically essential: wall motion abnormalities may be intermittent, valve severity varies beat-to-beat (especially in atrial fibrillation), diastolic parameters require multi-cycle measurement per guidelines, and rhythm assessment demands temporal context beyond one cycle. The ~2x encoder speedup does not justify this clinical information loss.

### Combined Impact (observed)

| Setup | Per-model (ViT-L) | TAPSE+LVEF (6 models ea.) | vs. Sequential |
|-------|-------------------|--------------------------|----------------|
| Sequential (BS1 × 8 GPUs, 20h, 20ep) | ~6 hrs | ~36 hrs | baseline |
| Old concurrent (BS1 × 4 GPUs, 20h, 20ep) | ~6 hrs/job | ~43 hrs | **1.2× slower** (eff_bs=4) |
| **Optimal (BS2 × 4 GPUs, 12h, 15ep)** | **~3.5 hrs** | **~15 hrs** | **2.4× faster** |

Speedup breakdown: 2× from GPU concurrency (BS2 restores eff_bs=8), 40% from 20→12 heads, 25% from 20→15 epochs. Multiplicative: 1/2 × 12/20 × 15/20 = 0.45× the sequential wall time.

The old concurrent setup (BS1 × 4 GPUs) was actually slower than sequential because eff_bs=4 doubled iters/epoch without doubling throughput. The fix was BS2, which restores eff_bs=8 while enabling true 2× concurrency.

## Data Sources

- TAPSE checkpoints: `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/tapse-{model}/best.pt`
- LVEF checkpoints: `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/lvef-{model}/best.pt`
- Per-head metrics stored in checkpoint: `val_acc_per_head`, `best_val_acc_per_head`, `mean_val_acc_per_head`, `min_val_acc_per_head`, `best_epoch_per_head`
- HP grid defined in `scripts/run_uhn_probe.sh` under `multihead_kwargs`
