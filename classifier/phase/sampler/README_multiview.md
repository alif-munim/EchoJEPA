# Phase-matched multi-view JEPA — quick reference

Turn 3 scaffold. Loss + data path are smoke-tested; the long-run
launcher driver is Turn 4.

## Run order (before the first pilot)

```bash
# 0. Verify the environment.
python classifier/phase/sampler/check_anchor_math.py

# 1. Subject-level split (strict, high-only).
python classifier/phase/sampler/build_subject_splits.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --out-dir classifier/phase/splits/ \
    --quality-tiers high --rr-filter-mode strict \
    --val-frac 0.05 --test-frac 0.10 --seed 0

# 2. Hard-fail if subjects leak.
python classifier/phase/sampler/check_phase_split_integrity.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --split-csv classifier/phase/splits/subjects_split.csv \
    --fail-on-leakage

# 3. Sampler dry-run.
python classifier/phase/sampler/phase_matched_sampler.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --quality-tiers high --rr-filter-mode strict \
    --sampling-mode uniform_phase --phase-tolerance 0.15 \
    --frames-per-clip 16 --frame-step 1 --pairs-per-study 1

# 4. Anchor-loading sanity (uses on-disk DICOMs).
python classifier/phase/sampler/check_anchor_loading.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --dicom-dir classifier/phase/dicoms --n-pairs 5

# 5. One-batch loss + gradient smoke (CPU, ~5 min at ViT-Tiny 128px).
python classifier/phase/sampler/check_multiview_loss_smoke.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --dicom-dir classifier/phase/dicoms \
    --batch-size 2 --img-size 128 --frames-per-clip 8 --frame-step 1 --cpu

# 6. DDP logical disjointness.
python classifier/phase/sampler/check_ddp_disjoint.py \
    --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
    --world-size 2 --epoch 0

# 7. Pre-baseline Δ_within (required before launch).
python classifier/phase/sampler/prepost_delta_within.py \
    --pre /home/sagemaker-user/user-default-efs/vjepa2/checkpoints/vitl.pt \
    --out-dir /tmp/prepost_baseline
```

## Actual pilot launch (once Turn 4's `app/vjepa_multiview/train.py::main`
is wired)

```bash
# 1-node 8-GPU pilot on HyperPod (same launcher contract as vjepa).
python -m app.main \
    --fname configs/train/vitl16/pretrain-multiview-phase-high.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

# 2-GPU single-node smoke test before running the pilot (under 10 min):
python -m app.main \
    --fname configs/train/vitl16/pretrain-multiview-phase-high.yaml \
    --devices cuda:0 cuda:1
```

## DDP smoke via torchrun

### CPU sanity (this dev box, already passes)

```bash
cd classifier/phase
python sampler/run_debug_training.py \
    --config /path/to/vjepa2/configs/train/vitl16/debug-multiview-phase-high.yaml \
    --dicom-dir dicoms
# -> 2 epochs * 5 steps = 10 optimizer steps in ~1.5 min on CPU.
# Losses decrease; checkpoint lands at /tmp/multiview_debug_phase_high/latest.pt.
```

### HyperPod 2-GPU smoke (pending — run on cluster)

The HyperPod controller stages code via S3 + srun (see
``claude/dev/hyperpod-deployment.md``). Once deployed:

```bash
# On a compute node with 2+ GPUs:
cd /opt/vjepa2
torchrun --nproc_per_node=2 -m app.main \
    --fname configs/train/vitl16/debug-multiview-phase-high.yaml

# For a real run (after the smoke):
torchrun --nproc_per_node=8 -m app.main \
    --fname configs/train/vitl16/pretrain-multiview-phase-high.yaml
```

### Pilot launch via SLURM sbatch (HyperPod)

```bash
# Rank 0 / world=8, single node, 25-epoch phase-matched pilot.
# Pattern from scripts/neurips/phase/*.sbatch:
sbatch -p h100 --gres=gpu:8 --cpus-per-task=64 --mem=900G \
    scripts/neurips/phase/pretrain-multiview-phase-high.sbatch
```

(`.sbatch` wrapper: Turn 4a. Writes SLURM output to `/tmp/`, deploys
via `~/deploy.sh`, runs `python -m app.main --fname <yaml>` with the
appropriate `--devices` flag.)

### Logical disjointness (already verified)

`check_ddp_disjoint.py` shows world=2 → 3,194 records per rank with
zero overlap and exact reconstruction of the world=1 record set (no
padding, no dropped records). Real DDP run will still reproduce this
because the sampler's rank slicing is deterministic given
`(seed, epoch, rank, world_size)`.

## What Turn 3 does NOT include

- `app/vjepa_multiview/train.py::main` (full-scale launcher with
  optimizer / EMA / checkpoint resume): Turn 4.
- Training-loop-wired `refresh_epoch` call: Turn 4.
- Predictor phase-relation token: config stub present, implementation
  deferred.
- Raw ECG fusion, LVET/Weissler phase, Echo-SyncNet DTW: not in scope.
