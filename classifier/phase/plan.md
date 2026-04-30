# Cross-View JEPA Adaptation — Multi-View Pretraining from IN21K e100

## Context

We have a 100-epoch MIMIC JEPA checkpoint trained from ImageNet-21K init (`checkpoints/jepa_in21k_vitl_e100.pt`). The NeurIPS manuscript already contrasts this against a 200-epoch single-view continuation and SALT. This plan adds a **third continuation axis**: cross-view JEPA where the predictor maps context tokens from view A to target tokens from view B of the same study, rather than predicting masked tokens within a single clip.

**Why now.** JEPA's distinguishing property vs BYOL-family methods is *local* prediction — the predictor maps context-region tokens to target-region tokens at specific positions. If we replace this with pooled summary tokens we reduce cross-view JEPA to "BYOL with cross-view targets." The design below preserves token-level local prediction in the cross-view setting by using target position embeddings as predictor queries and learning soft cross-view correspondence via cross-attention.

**Primary comparison set** (all adapted from the same `jepa_in21k_vitl_e100.pt`):
1. No-adaptation baseline (evaluate e100 directly).
2. 200-epoch single-view continuation (exists: `checkpoints/pretrain_21/mimic/vjepa2_1_vitl_224px_16f/` reaches e200; **uses a different init though — the current MIMIC JEPA run; note this is an imperfect control**).
3. Cross-view JEPA from e100, same-study targets.
4. Cross-view JEPA from e100, shuffled-study control (matched target-view marginals).
5. Optional: continued single-view from `jepa_in21k_vitl_e100.pt` for an init-matched single-view control if #2 is deemed not comparable.

Intended outcome: downstream regression/classification probes comparing all four. Expected positive signal on cross-view-specific tasks (RVSP multi-view, missing-view robustness, cross-view retrieval).

---

## Files to create / modify

### New files

- **`app/vjepa_mv/__init__.py`** — makes subpackage discoverable by `app.scaffold`.
- **`app/vjepa_mv/train.py`** — new main training loop for `app: vjepa_mv`. Starts as a fork of `app/vjepa/train.py` with phi-JEPA/phase-mask branches stripped out; adds cross-view forward path and loss.
- **`app/vjepa_mv/utils.py`** — `init_mv_video_model(...)`, `init_opt(...)`, `load_checkpoint(strict_predictor=False, strict_encoder=True, ...)`. Reuses `src/utils/wrappers.MultiSeqWrapper` for the encoder. The cross-view predictor replaces `PredictorMultiSeqWrapper`; the existing single-view predictor may optionally be kept in the same model for a within-view loss branch.
- **`src/models/cross_view_predictor.py`** — `CrossViewPredictor(nn.Module)`. 4–6 layer cross-attention transformer (cross-attn → self-attn → FFN per block). Query path: target spatiotemporal position embedding + target view embedding (additive). KV path: context tokens with view-aware position embedding added. Output: `[B, N_tgt, D]`. Exposes per-layer attention weights for entropy/concentration diagnostics.
- **`src/models/view_aware_pos_embed.py`** — shared utility: `ViewAwarePosEmbed(num_views, grid_dims, embed_dim)`. Implements `pos_embed_view = pos_embed_base + view_embed[v]` (additive offset, the default). Alternative `per_view_table` path gated by config.
- **`src/models/view_target_summarizer.py`** *(optional, only if we land `predictor_type: summary_pooling` as a baseline)* — wraps `AttentivePooler` with K=8 learned queries plus view embedding. Present as a behind-flag baseline; do NOT wire as default.
- **`src/datasets/paired_study_dataset.py`** *(thin wrapper)* — adapter around `VideoGroupDataset` that (a) sets `group_size=2` for context+target clips per row, (b) returns the two normalized view IDs, (c) supports a `shuffled_study_control=True` mode where target clips are drawn from a different study each epoch while preserving the target-view marginal distribution. Implementation choice: subclass `VideoGroupDataset` or compose it; prefer composition since `video_group_dataset.py` expects header-schema CSVs and we will build a fresh one.
- **`scripts/neurips/build_mv_pairs_csv.py`** — offline CSV builder. Inputs: `data/csv/mimic_annotations_s3.csv` (525,328 clips) and `user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv` (525,329 rows; columns `s3_uri,view,view_confidence,view_status`). Join on `s3_uri`. Filter `view_status == "OK"` and drop `view == "Exclude"`. Extract study_id via the same regex as `src/datasets/study_sampler.py::_extract_study_id` (`/s(\d+)/\d+_\d+\.mp4$`). Group rows by study; for each study, emit one row per paired sample: `context_uri context_view target_uri target_view label=0`. Sampling policy: for each study with ≥2 OK-view clips, draw one (context, target) pair per epoch-independent row with probabilities `same_view / related_view / random_view` as configured; if the study has only one clip, drop it (log count). Produce two outputs:
  1. `mimic_mv_pairs_train.csv` — paired rows (same study).
  2. `mimic_mv_pairs_train_shuffled.csv` — same contexts, target drawn from a different study, matching the target-view marginal of the non-shuffled file.
  Both write to `/opt/dlami/nvme/data/csv/` on HyperPod; also keep a copy in `data/csv/` for documentation. Logs: per-view counts, pair-type counts, dropped-study count, fallback counts.
- **`src/masks/mv_mask_collator.py`** — wrapper that applies the existing `MaskCollator` from `src/masks/multiseq_multiblock3d.py` twice per sample (once per clip in the pair) and re-zips to emit `(batch, masks_enc_A, masks_pred_A, masks_enc_B, masks_pred_B, view_ids_A, view_ids_B)`. Does NOT modify `multiseq_multiblock3d.py`; the existing modifications there are preserved.
- **`configs/train/vitl16/pretrain-mv-mimic-224px-16f-from-e100.yaml`** — same-study multi-view adaptation from `jepa_in21k_vitl_e100.pt`.
- **`configs/train/vitl16/pretrain-mv-mimic-224px-16f-shuffled-from-e100.yaml`** — shuffled-study control.
- **`configs/train/vitl16/pretrain-mv-mimic-224px-16f-singleview-from-e100.yaml`** *(optional)* — init-matched continued single-view control if the 200ep run is not directly comparable.
- **`docs/multiview_jepa.md`** — README (see §README below).
- **`tests/app/test_vjepa_mv_smoke.py`** — smoke test.

### Modified files

- **`app/scaffold.py`** — register `vjepa_mv` → `app.vjepa_mv.train:main`.
- **`src/datasets/data_manager.py`** — add `pairedstudydataset` branch alongside the existing `videodataset` (L72–L100) and `videogroupdataset` (L102–L130) branches. Pattern: follow `make_videogroupdataset`, but route through the new wrapper class and propagate the `shuffled_study_control: bool` flag from the YAML.

### Files intentionally NOT modified

- `app/vjepa/train.py` (972 lines) stays intact; single-view JEPA path is unchanged.
- `src/models/predictor.py` stays intact.
- `src/masks/multiseq_multiblock3d.py` stays intact (the already-in-progress edit is preserved).
- `src/datasets/video_group_dataset.py` stays intact; the new paired-study wrapper composes around it.
- `src/datasets/video_dataset.py`, `src/datasets/study_sampler.py`, `src/utils/wrappers.py` — read-only reuse.

---

## Model / objective — cross-attention dense prediction

**Online context path.** `context_tokens = encoder(context_video)` — reuse existing `MultiSeqWrapper` (`src/utils/wrappers.py` L9–L27). Shape `[B, N_ctx, D]`. For ViT-L at 224×224, 16 frames, tubelet 2, patch 16: `N_ctx ≈ 8 · 14 · 14 = 1568`, `D = 1024`.

**EMA target path.** `target_tokens = target_encoder(target_video)` with `torch.no_grad()` and post-LayerNorm (mirroring `app/vjepa/train.py` L751–L755). Shape `[B, N_tgt, D]`, `N_tgt = N_ctx` here. EMA update copies the existing `torch._foreach_mul_/_foreach_add_` block (`train.py` L796–L805).

**Target query construction.** Queries are deterministic functions of target position and target view — not learned summary tokens:
```
q_i = pos_embed_view(target_view)[i] + optional_learned_offset[i]
```
Shape `[B, N_tgt, D]`. Implemented via `ViewAwarePosEmbed` in `src/models/view_aware_pos_embed.py`.

**Cross-view predictor.** `src/models/cross_view_predictor.py`. 6 layers by default. Each layer: cross-attention (target queries → context keys/values) → self-attention (target queries attend to each other) → FFN. Predictor embedding dim matches the existing single-view predictor embedding dim (`predictor_embed_dim` in config, typically 384 for ViT-L). Output head projects back to encoder `D=1024`. Output shape `[B, N_tgt, D]`.

**View-aware position embeddings.** Additive view offset by default:
```
pos_embed_total = pos_embed_base + view_embed[view_id]
```
Applied to BOTH context tokens (before they enter the predictor) and target queries. `view_embed` has shape `[V, D]` where `V = |view families|`. View label normalization: uppercase → strip → alias map → OTHER bucket for unknowns. View families from config (apical / parasternal / subcostal / doppler / other). `pos_embed_base` is a learned `[N_tgt, D]` table initialized from the existing encoder's position embeddings if available (load once, freeze optional).

**Loss.**
- `cross_view_loss = mean_L1(pred_target_tokens, stop_gradient(target_tokens))` — token-level at each of the N_tgt positions. Same form as `app/vjepa/train.py::loss_fn` L767 but over unmasked full target-token grids (no `apply_masks`, since the cross-view predictor attends softly rather than at fixed masked positions).
- `within_view_loss` — standard single-view JEPA: encode context view, run the *existing* `VisionTransformerPredictor` on masked context, compare to masked target-encoder output of the *same view*. Reuses the loss path from `app/vjepa/train.py` L751–L767.
- `total_loss = within_view_weight · within_view_loss + cross_view_weight · cross_view_loss`. Defaults: `within_view_weight=1.0`, `cross_view_weight=0.5`. Do not fall back to cross-only by default.

**Failure mode: attention collapse.** Diagnostic logging (see §Logging).

---

## Dataset and sampling

Reuse the existing CSV and regex-based study_id extraction; all new logic lives in `scripts/neurips/build_mv_pairs_csv.py` and `src/datasets/paired_study_dataset.py`.

**Join.** `mimic_annotations_s3.csv` (525,328 clips) ⟕ `mimic_view_predictions.csv` (525,329 rows) on full S3 URI. Drop rows where `view_status != OK` or `view == Exclude`. Log counts: raw, after filter, per-view histogram.

**Pair sampling (same_study_multiview mode).** For each study with ≥2 OK clips:
- sample context view/clip uniformly from the study;
- sample target view/clip according to configured schedule:
  - `same_view`: different clip of the same view (if available);
  - `related_view`: different view in the same family (apical/parasternal/subcostal/doppler);
  - `random_view`: any other clip in the study;
  - fallbacks logged.
- Schedule example (curriculum): start at 0.50/0.40/0.10, shift to 0.30/0.50/0.20 after 10 epochs, 0.20/0.40/0.40 after 30.

**Pair sampling (shuffled_study_control mode).** For each row in the same-study CSV, keep the context but redraw the target from a *different study* whose clip has the same normalized view as the intended target. This preserves the target-view marginal while breaking patient-specific content. Implementation: precompute per-view pools, sample with replacement. Guarantee `context_study_id != target_study_id`. (Patient ID via MIMIC `p{nn}/p{patient_id}` path segment — extract and also require `context_patient != target_patient` to prevent within-patient leakage.)

**Output CSV format.** Header: `context_uri target_uri label context_view target_view`. Label column is dummy 0 (pretraining). Space-delimited to align with existing readers. Two files per split:
```
mimic_mv_pairs_train.csv
mimic_mv_pairs_train_shuffled.csv
```

**View family normalization.** Implemented in `src/datasets/paired_study_dataset.py` (shared with CSV builder): uppercase, strip, alias map (e.g. `APICAL_4CH`→`A4C`), unknown → `OTHER`. Config-driven family table from the YAML's `view_families` block, not hardcoded.

---

## Checkpoint loading

Load `checkpoints/jepa_in21k_vitl_e100.pt` with `strict=False` on the predictor and cross-view modules:

- `encoder.*` — strict load.
- `target_encoder.*` — strict load if present; otherwise deepcopy encoder (matches `app/vjepa/train.py` L310 / L448 pattern).
- `predictor.*` (legacy) — strict load IF `within_view_weight > 0`; otherwise skipped.
- `cross_view_predictor.*`, `view_embed.*`, `view_aware_pos_embed.*` — initialized fresh (random or from encoder pos-embeds).
- Log all missing/unexpected keys explicitly. Assert the *only* missing keys are the new cross-view modules.

Pattern reused from `app/vjepa/utils.py::load_checkpoint` L98–L158 (the existing `strict_predictor=False` path for phi-JEPA). New output directory: `/opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_mv_from_e100/`.

---

## Config schema

New top-level block added to the existing config shape. Template: copy `configs/train/vitl16/pretrain-mimic-224px-16f-in21k-phase-mask.yaml`, delete the `phase_mask` block (L65–L86) and `phase_metadata_csv` (L32), set `dataset_type: PairedStudyDataset`, set `anneal_ckpt: /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt` + `force_load_pretrain: true`, and add:

```yaml
multiview:
  enabled: true
  shuffled_study_control: false
  context_views_per_sample: 1
  target_views_per_sample: 1
  predictor_type: cross_attention_dense   # default; alternative: summary_pooling
  predictor:
    num_layers: 6
    num_heads: 8
    use_self_attention: true
    feedforward_dim_mult: 4
  position_embed:
    style: additive_view_offset           # alternative: per_view_table
    learnable: true
  # Retained for summary_pooling baseline only:
  target_summary_tokens: 8
  target_pooling: query
  pair_sampling:
    schedule:
      - until_epoch: 10
        same_view: 0.50
        related_view: 0.40
        random_view: 0.10
      - until_epoch: 30
        same_view: 0.30
        related_view: 0.50
        random_view: 0.20
      - until_epoch: null
        same_view: 0.20
        related_view: 0.40
        random_view: 0.40
  loss:
    within_view_weight: 1.0
    cross_view_weight: 0.5
  attention_diagnostics:
    log_entropy: true
    log_concentration: true
    entropy_warning_threshold: 0.95
  view_families:
    apical:    [A4C, A2C, A3C, A5C]
    parasternal: [PLAX, PSAX]
    subcostal: [SUBCOSTAL, IVC]
    doppler:   [TR_DOPPLER, MV_DOPPLER, AV_DOPPLER, LVOT_DOPPLER, CW_DOPPLER, PW_DOPPLER]
    other:     [OTHER, UNKNOWN]
  residual_target_prediction:
    enabled: false                        # stub only for MVP
    lambda: 0.5
    prototype_momentum: 0.99
```

Three YAMLs (all `app: vjepa_mv`):
1. `pretrain-mv-mimic-224px-16f-from-e100.yaml` — same-study, `shuffled_study_control: false`.
2. `pretrain-mv-mimic-224px-16f-shuffled-from-e100.yaml` — `shuffled_study_control: true`; same compute budget.
3. *(optional, if user wants init-matched single-view control)* `pretrain-mv-mimic-224px-16f-singleview-from-e100.yaml` — reuses existing `app: vjepa` but with `anneal_ckpt: jepa_in21k_vitl_e100.pt`. One-line README note may suffice in place of this.

Epoch budget: 50 extra epochs (= matched to 100→150 single-view extension we can run in parallel). Batch size: 64 on 8×H100 (half of vanilla 128 to offset the 2x clip memory). `use_activation_checkpointing: true`.

---

## Logging

Extend the existing CSV logger (`app/vjepa/train.py::CSVLogger` pattern). Per-iteration:
- `loss_total`, `loss_within_view`, `loss_cross_view`.
- `ema_momentum`.
- `attn_entropy_mean_per_layer` (list of L floats, one per cross-attn layer; mean across heads and batch).
- `attn_concentration_top1_mean` (mean top-1 attention weight across target queries and batch).
- Warn if `attn_entropy_mean / log(N_ctx) > entropy_warning_threshold` (0.95 default) for >10% of an epoch. Warning surfaces as a log line prefixed `[ATTN_COLLAPSE_WARN]` plus a flag in `log_r{rank}.csv`.

Per-epoch JSON dump (`view_pair_log_epoch{E}.json`): context-view histogram, target-view histogram, pair-type histogram (same_view/related_view/random_view), A→B pair counts (top 20), fallback counts, fraction shuffled (should be 0 for same-study, 1.0 for shuffled control).

Representation diagnostics (cheap, every ~50 iters): cosine similarity between `pred_target_tokens` and `target_tokens` (mean across positions and batch); `||pred||`, `||target||` norms.

---

## Cross-view retrieval diagnostic (v1, promoted from TODO)

Script: `scripts/neurips/mv_retrieval_diag.py`. Runs after each epoch on a fixed held-out set of N=200 studies (split from training CSV, written once to `data/csv/mimic_mv_retrieval_heldout.csv`).

For each held-out study:
1. Encode one A4C clip and one PSAX clip with the online encoder.
2. `pred_target_psax = CrossViewPredictor(A4C_tokens, target_view=PSAX)`; mean-pool across positions → single vector.
3. `actual_psax = mean_pool(target_encoder(PSAX_clip))`.

Gallery = all actual_psax vectors. For each study, query with its pred_target_psax, compute cosine similarity to every gallery entry. Metrics: Recall@1, Recall@5, MRR. Report same-study R@1 vs shuffled-control R@1 — if they are within ±2%, the predictor is learning view priors, not patient-specific structure (hard failure signal).

Output: per-epoch `retrieval_diag_epoch{E}.json`. Also emit a v1 stub for (a)/(b)/(c) directly comparable across runs (skip (c) continued single-view since it has no cross-view predictor).

---

## Smoke test

`tests/app/test_vjepa_mv_smoke.py`. Runs in-process without SLURM. Checks:
1. `PairedStudyDataset.__getitem__` returns `(clip_A, clip_B, view_A, view_B, study_id, patient_id, is_shuffled)`.
2. `MVMaskCollator` emits the 7-tuple of tensors.
3. One forward pass through `CrossViewPredictor`; shape assertion `pred.shape == target_tokens.shape == [B, N_tgt, D]`.
4. `target_encoder` parameter grads are zero after backward: `all(p.grad is None or p.grad.abs().sum() == 0 for p in target_encoder.parameters())`.
5. One `optimizer.step()`, verify online encoder grads are non-zero.
6. EMA update works: target_encoder params move by (1−m)·(encoder − target_encoder).
7. Attention entropy and concentration are logged per cross-attn layer; log dict contains `attn_entropy_mean_per_layer` and `attn_concentration_top1_mean`.
8. Run ≥20 iters; record final `N_ctx`, `N_tgt`, peak GPU memory, sec/iter.
9. Checkpoint save + reload round-trip; missing-key assertion on reload matches expected set.
10. Shuffled-control mode: assert `context_study_id != target_study_id` across all samples in a batch.

Batch=2, `num_workers=0`, 1 GPU. No S3 reads — use `tests/fixtures/` local fake clips or skip if fixtures are heavy.

---

## Implementation order

1. Inspect repo (done; see §Files).
2. Build `scripts/neurips/build_mv_pairs_csv.py` + produce the two paired CSVs.
3. `src/models/view_aware_pos_embed.py` (shared utility; no model deps yet).
4. `src/models/cross_view_predictor.py` (standalone module, unit-testable).
5. `src/datasets/paired_study_dataset.py` + `src/masks/mv_mask_collator.py`.
6. `app/vjepa_mv/train.py` + `app/vjepa_mv/utils.py` (loss, EMA, checkpoint load).
7. `app/scaffold.py` + `src/datasets/data_manager.py` wiring.
8. Three YAML configs.
9. Attention entropy/concentration logging.
10. Shuffled-study control path end-to-end.
11. `scripts/neurips/mv_retrieval_diag.py` retrieval v1.
12. `tests/app/test_vjepa_mv_smoke.py` smoke test + run it.
13. `docs/multiview_jepa.md` README with design rationale and failure-mode warning.

---

## Launch commands

```bash
# 1. Build paired CSVs (once)
python scripts/neurips/build_mv_pairs_csv.py \
  --clips   data/csv/mimic_annotations_s3.csv \
  --views   /home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv \
  --out-dir data/csv/ \
  --shuffled-control

# 2. Smoke test (local single-GPU)
pytest tests/app/test_vjepa_mv_smoke.py -x -s

# 3. Same-study multi-view adaptation (HyperPod)
sbatch scripts/neurips/multiview/pretrain_mv_from_e100.sbatch

# 4. Shuffled-study control (HyperPod)
sbatch scripts/neurips/multiview/pretrain_mv_shuffled_from_e100.sbatch

# 5. Per-epoch retrieval diagnostic (runs inside train loop; manual standalone:)
python scripts/neurips/mv_retrieval_diag.py \
  --ckpt   /opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_mv_from_e100/e10.pt \
  --heldout data/csv/mimic_mv_retrieval_heldout.csv
```

---

## Open assumptions to confirm during implementation

- `src/datasets/video_group_dataset.py` accepts space-delimited headerless CSVs — we reuse that for the paired CSV format.
- `N_ctx = N_tgt = 1568` for ViT-L at 224px/16f/tubelet=2/patch=16. Smoke test will confirm.
- View label confidence threshold: default 0.0 (include all OK rows). Flag `--min-confidence` in the CSV builder for ablation.
- Whether to use `jepa_in21k_vitl_e100_as_salt_teacher.pt` (1.2GB, encoder-only) vs the full `jepa_in21k_vitl_e100.pt` (5.1GB, with predictor + target + optimizer state) — prefer the full one so the within-view predictor is also loaded.

---

## Verification plan

**Unit / smoke (local):**
- `pytest tests/app/test_vjepa_mv_smoke.py` — all 10 checks pass.
- Assert `N_ctx`, `N_tgt`, peak mem, and sec/iter are reported.

**Single-GPU short training (1 node):**
- Launch 500-iter run on 1 GPU, batch=2. Verify `loss_within_view` and `loss_cross_view` both decrease, cosine(pred, target) increases, attention entropy does not saturate.

**HyperPod end-to-end:**
- Same-study run (50 epochs, 8×H100, batch=64 per node) — checkpoints every 10 epochs.
- Shuffled control at same compute.
- Retrieval diagnostic compares same-study R@1 vs shuffled R@1 after 10 and 50 epochs. Signal: same-study R@1 > shuffled R@1 + 5pp.

**Downstream probes (later, not in this plan scope):**
- LVEF, RVSP, CAMUS segmentation using existing `configs/eval/vitl/` probes. Swap encoder ckpt to the new mv-adapted ckpt. Compare against e100, e200 single-view, and SALT baselines.

---

## Reporting requirements

On completion, report:
1. Files changed (with diffs summarized).
2. New config file paths.
3. Exact launch commands (templated `anneal_ckpt` path).
4. What is implemented vs TODO (residual target prediction is TODO by default).
5. Smoke test result: all 10 checks, `N_ctx`/`N_tgt` values, peak memory, sec/iter.
6. Assumptions about dataset metadata fields.
7. Issues found with view labels (unknown label counts, study skip counts, shuffled fallback counts).
8. Whether attention entropy diagnostics fired during the smoke test.
9. Any sign of attention collapse in smoke / early training.
10. Memory footprint of `cross_attention_dense` vs (if implemented) `summary_pooling`.
