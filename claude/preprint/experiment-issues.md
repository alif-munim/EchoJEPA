# ICML Preprint: Experiment Issues & Lessons Learned

Issues discovered during early experimental testing (Aug-Sep 2025) and subsequent code review (Feb-Mar 2026). These were exploratory runs done while developing the probe pipeline. Documented here for reference so the same pitfalls are avoided in Nature Medicine.

## 1. Batch Size Scaling Failure (Aug 2025)

### Setup

Early probe training (view classification / RV function, d=4 attentive, ViT-G) used `batch_size: 8` per GPU (eff_bs=64 on 8 GPUs). Attempted to speed up training by increasing to `batch_size: 48` (eff_bs=384), with and without linear LR scaling.

### Dataset

`a4c_rvfx_labels_train_tiny.csv`: **2,000 training clips** (binary RV function classification). This was a deliberately small development split used during probe pipeline iteration.

### Results

| Run | BS | LR | Best Test Acc | Outcome |
|-----|----|----|--------------|---------|
| BS8 baseline | 8 | 0.02 | 83.0% | Stable convergence |
| BS48 / no LR scale | 48 | 0.03 | 79.4% | Under-converged |
| BS48 / scaled LR v1 | 48 | 0.06 | 83.3% | Lucky convergence |
| BS48 / scaled LR v2 | 48 | 0.06 | 58.1% | Diverged/trapped |
| BS48 / scaled LR v3 | 48 | 0.05 | 62.0% | Diverged/trapped |

Three runs with the same BS48 + scaled LR config produced wildly different results (58-83%). The BS8 baseline was reproducible.

**Log files:** `logs/rvfx_cooldown_h16_b4_0824_keepe96_b8.log` (BS8), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0826_scaledLR.log` (xLR v1), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v2.log` (v2), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v3.log` (v3).

**Plots:** `notebooks/probe_hps.png` (all HP variants), `notebooks/probe_scaledLR.png` (scaled LR train+test curves). Generated in `notebooks/plot.ipynb` (cells 4, 9).

### Root Cause: Dataset too small for batch size

| Config | Effective BS | Iters/Epoch | Fraction of dataset/step | Total grad updates (170 ep) |
|--------|-------------|-------------|--------------------------|---------------------------|
| BS8 | 64 | ~31 | 3.2% | ~5,300 |
| BS48 | 384 | ~5 | **19.2%** | ~850 |

With BS48, each gradient step sampled nearly 1/5 of the entire 2,000-clip dataset. Each epoch had only ~5 gradient updates. Three compounding factors:

1. **LR schedule compressed to ~5 steps/epoch.** Cosine decay with warmup needs hundreds of steps to function. With 5 steps/epoch, the warmup alone barely completes before the LR starts decaying. The schedule essentially collapses to a near-constant LR.

2. **Each step's gradient dominates the trajectory.** When one step sees 19% of the dataset, the random composition of that mini-batch determines the gradient direction. With so few total steps (~850), there's no law-of-large-numbers averaging. Different random seeds → different gradient trajectories → different basins.

3. **d=4 attentive probe (16H, 4B) was too large for ~850 updates.** The probe had ~10-20M trainable parameters. 850 gradient updates on 2,000 samples is massively under-determined. BS8's 5,300 updates gave 6x more signal.

### Lesson for Nature Medicine

The current Nature Medicine probes use BS1 (eff_bs=8), d=1 attentive probes, and much larger datasets (25K-100K+ studies). Even at BS8 (eff_bs=64), TAPSE's 25K studies would give ~390 iters/epoch — far above the danger zone. The ICML batch size failure was specific to the pathological combination of tiny dataset + large batch + deep probe.

The HP grid analysis (see `claude/dev/efficiency.md`) suggests the Nature Medicine probes could safely use BS2-4 with these larger datasets, though BS1 remains the conservative default.

---

## 2. Attentive Probe Inversion (d=4 on non-ViT models)

### Issue

The ICML preprint used d=4 attentive probes (16 heads, 4 self-attention blocks) for all models. Non-ViT baselines showed inverted results: linear probes outperformed attentive probes on view classification.

| Model | Tokens/clip | Attentive (d=4) | Linear | Status |
|-------|-------------|-----------------|--------|--------|
| EchoJEPA-G | 1568 | 87.4% | 80.9% | Expected |
| EchoJEPA-L | 1568 | 85.5% | 70.8% | Expected |
| EchoMAE-L | 1568 | 40.4% | 59.2% | **Inverted** |
| EchoPrime | 1 | 42.1% | 57.7% | **Inverted** |
| PanEcho | 1 | 41.9% | 52.8% | **Inverted** |

### Root Cause: Token starvation + normalization bugs

Two issues compounded:

1. **Token starvation**: EchoPrime and PanEcho output 1 pre-pooled token per clip. Cross-attention over a single token is degenerate (softmax trivially 1.0). The d=4 probe's 3 self-attention blocks became pure optimization overhead, causing overfitting. See `probe-architecture-analysis.md`.

2. **Normalization bugs** (bug 002): PanEcho was double-normalized (ImageNet applied twice), EchoPrime received ImageNet-normed input instead of [0,255], EchoFM received ImageNet-normed instead of [0,1]. All three baselines were running on corrupted inputs, further depressing their results. See `claude/dev/bugs/002-normalization-bugs.md`.

### Resolution for Nature Medicine

Nature Medicine adopted **Strategy E: d=1 attentive probes for all models**. At depth=1, there are no self-attention blocks — only a single cross-attention layer. Verification (UHN view classification, 22K clips) confirmed d=1 helps ALL models including EchoPrime (+9.3pp) and PanEcho (+7.1pp). The ICML inversion was an artifact of d=4 + normalization bugs + token starvation. Normalization bugs were fixed and affected models re-extracted.

---

## 3. Encoder Normalization Bugs (discovered Mar 2026)

Three of five baseline encoder adapters had incorrect input normalization. See `claude/dev/bugs/002-normalization-bugs.md` for full details.

| Model | Bug | Impact |
|-------|-----|--------|
| PanEcho | Double ImageNet normalization | Shifted input distribution |
| EchoPrime | Expected [0,255], received [-2,2] | Wrong input range entirely |
| EchoFM | Expected [0,1], received [-2,2] | Wrong input range entirely |

All ICML preprint baseline numbers for PanEcho, EchoPrime, and EchoFM were affected. The bugs were fixed and MIMIC embeddings re-extracted (2026-03-08). EchoJEPA and EchoMAE were unaffected (they use ImageNet normalization natively).

**Note:** The controlled comparison (EchoJEPA-L vs EchoMAE-L) is unaffected since both use the same ViT-L encoder adapter with correct normalization.

---

## 4. Shuffle Bug in Embedding Extraction (discovered Feb 2026)

`DistributedSampler(shuffle=True)` was used during embedding extraction, but the extracted embeddings were stored in shuffled order rather than the original CSV order. This meant embeddings were misaligned with their labels. See `claude/dev/bugs/001-shuffle-bug.md`.

**Impact:** All 7 models' MIMIC clip-level embeddings were in wrong order. Post-hoc reordering was applied using the saved shuffle permutation. UHN extractions fixed at the code level before extraction.

---

## 5. Pretraining Loss Divergence (ViT-L)

Visible in `notebooks/plot.ipynb` cell 16: the ViT-L pretraining loss drops from ~0.53 to ~0.455 by epoch 10, then **rises continuously** back to ~0.52 by epoch 200. This is classic overfitting on the pretraining objective.

The loss divergence is why the anneal (LR cooldown) stage was critical — it recovers representation quality by reducing the LR to allow the model to settle into a better basin. The anneal checkpoint (`vitl-pt-210-an25.pt`, 210 pretrain + 25 anneal epochs) is what's used for downstream probes, not the raw pretrain checkpoint.

The Kinetics-initialized ViT-L (`vitl-kinetics-pt220-an55.pt`) and ViT-G (`pt-280-an81.pt`) both went through similar anneal stages. Pretraining loss curves for these are in cells 47 and 49 of `plot.ipynb`.

---

## 6. Video Decode Failures (ongoing)

The S3-streaming video pipeline (decord) has a non-zero decode failure rate. Failed videos are silently substituted with the next valid video in the dataset. In the BS48 experiments, decode failure counts scaled with batch size:

| Config | Decode failures |
|--------|----------------|
| BS8 / 96 Ep (178 actual) | 358 |
| BS48 / xLR (170 ep) | 170 |
| BS48 / xLR v2 (64 ep) | 64 |

Roughly ~1 failure per epoch regardless of batch size. The substitution doesn't affect label alignment (bug 004 was fixed to track substitutions), but it does introduce subtle noise in training — the model sees a different video than intended. With BS48's few gradient updates per epoch, each substitution has proportionally more impact.

---

## Summary: What Survived vs What Didn't

| Finding | Survived to Nature Medicine? | Notes |
|---------|------|-------|
| JEPA > MAE controlled comparison | Yes | Unaffected by any bugs (same encoder, same probe) |
| EchoJEPA-G dominance over baselines | Partially | Magnitudes inflated by d=4 inversion + normalization bugs. Re-evaluated with d=1 |
| Attentive probe as primary eval | No | Replaced by d=1 attentive (Strategy E) after discovering d=4 degeneration |
| Batch size scaling | No | Specific to tiny dev dataset. NM uses BS1 on large datasets |
| Sample efficiency (1% labels) | Yes | Controlled comparison, probe mismatch constant across fractions |
| Pediatric transfer | Yes | Tests representation directly, not probe design |

## Related Documents

- `probe-architecture-analysis.md` — attentive vs linear inversion root cause
- `encoder-fairness.md` — confound analysis (dim, scale, data, tokens)
- `claim-validity.md` — which claims survive the fairness analysis
- `hindsight-recommendations.md` — what we'd change if redoing the preprint
- `claude/dev/efficiency.md` — HP grid analysis for Nature Medicine probes
- `claude/dev/bugs/` — all 6 bugs with severity, status, and fixes
