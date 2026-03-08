# Issue 001: DistributedSampler shuffle corrupts embedding order

**Severity**: CRITICAL
**Discovered**: 2026-03-07
**Status**: FIXED (all embeddings reordered, downstream regenerated)

## Root Cause

`src/datasets/video_dataset.py:106` hardcodes `DistributedSampler(shuffle=True)`. Both extraction scripts (`extract_embeddings.py` for MIMIC, `extract_uhn_embeddings.py` for UHN) called `init_data()` → `make_videodataset()`, inheriting the shuffle. Neither script originally overrode this default.

With `shuffle=True`, the sampler applies a deterministic permutation (seed=0, epoch=0) to dataset indices before distributing across ranks. Each rank processes a subset of the permuted indices. The merge step uses a `global_idx` formula that assigns sequential IDs based on batch/rank position, then sorts by these IDs to "restore order." But this restores the **interleaved permutation order**, not the **CSV order**.

Downstream scripts (`remap_embeddings.py`, `pool_embeddings.py`) and `clip_index.npz` all assume **CSV order** alignment. The result: embeddings at master NPZ row `i` correspond to CSV row `perm[i]`, not CSV row `i`.

## Impact

### UHN (18M clips)
- **EchoJEPA-G**: Extracted with shuffle=True. `clip_embeddings.npz` (95GB) and `study_embeddings.npz` have wrong ordering. Study-level pooling assigned clips to wrong studies.
- **EchoJEPA-L**: Partially extracted (~75%), also with shuffle=True. Existing chunks need to be discarded and re-extracted.
- **EchoJEPA-L-K**: Extraction started AFTER the fix. No issue.

### MIMIC (525K clips, 7 models)
- **All 7 models** (echojepa_g, echojepa_l, echojepa_l_kinetics, echomae, echofm, panecho, echoprime) have shuffled master NPZs.
- `remap_embeddings.py` creates indices by matching source CSV paths (CSV order) to positions — but the master NPZ rows are in permuted order, so indices reference wrong rows.
- `pool_embeddings.py` uses `clip_index.npz` (CSV order study_ids) with the shuffled master NPZ — study pooling assigns wrong embeddings to wrong studies.
- **All study-level NPZs** (`{model}_study_level/{task}.npz`) have mismatched embedding-label pairs.
- **All precomputed splits** (`{model}_splits/{task}/train.npz`) inherit the wrong pairings.
- Probe training on these splits would learn random associations (AUC ~0.5, R² ~0).

## Fixes Applied

### Code fixes (prevent future occurrences)
1. `evals/extract_uhn_embeddings.py:141` — Added `data_loader.sampler.shuffle = False` (2026-03-07)
2. `evals/extract_embeddings.py` — Added `loader.sampler.shuffle = False` (2026-03-07)

### UHN EchoJEPA-G post-hoc fix
`evals/fix_shuffle_order.py` reconstructs the exact permutation (PyTorch Generator, seed=0, epoch=0) and reorders existing `clip_embeddings.npz` to CSV order, then re-pools to study level. Backs up originals as `.shuffled_backup`.

### MIMIC post-hoc fix (no re-extraction needed for shuffle)
`evals/fix_mimic_shuffle.py` reconstructs the permutation from `(n_dataset=525319, seed=0)` and reorders all 7 master NPZs to CSV order. Verified via label matching: **100.00% match** on all 7 models. The fix takes ~4 seconds per model.

Key detail: the dataset had 525,319 rows at extraction time; DistributedSampler padded to 525,320 (divisible by 8) and the merge produced 525,312 embeddings (7 fewer due to padding dedup). The permutation must be generated from n=525,319 (not 525,312) to get 100% label match.

**Note**: 3 of the 7 MIMIC models (PanEcho, EchoPrime, EchoFM) still need re-extraction due to normalization bugs (see Issue 002). The shuffle fix corrects their ordering but the embedding values are still wrong.

After shuffle fix, the downstream pipeline (remap, pool, split) must be re-run to regenerate study-level NPZs and train/val/test splits.

## Verification

The fix scripts verify correctness by label matching:
- MIMIC: reordered labels vs source CSV labels (100.00% match for all 7 models)
- UHN: re-pooled study embeddings using clip_index.npz (correct study assignments)

## Timeline

| Date | Event |
|------|-------|
| 2026-03-07 | Bug discovered during UHN resume investigation |
| 2026-03-07 | `extract_uhn_embeddings.py` fixed (shuffle=False) |
| 2026-03-07 | `fix_shuffle_order.py` created for post-hoc UHN fix |
| 2026-03-07 | `extract_embeddings.py` fixed (shuffle=False) |
| 2026-03-07 | MIMIC impact analysis: all 7 models affected |
| 2026-03-07 | `fix_mimic_shuffle.py` created, dry run verified 100% match on all 7 models |
| 2026-03-07 | MIMIC fix applied (all 7 models reordered) |
| 2026-03-07 | UHN EchoJEPA-G fix applied (clip reordered + study re-pooled: 319,815 studies) |
| 2026-03-07 | MIMIC downstream regenerated: all 7 models × 23 tasks (study-level + train/val/test splits) via `regenerate_mimic_downstream.py` |
| TBD | Re-extract PanEcho, EchoPrime, EchoFM (normalization bug, see Issue 002) |
