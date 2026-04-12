# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoJEPA — a latent predictive foundation model for echocardiography built on V-JEPA 2 (Video Joint-Embedding Predictive Architecture) from Meta. Trained on 18M echocardiograms via self-supervised latent prediction. The package installs as `vjepa2` (Python >=3.11, tested with 3.12).

### Nature Medicine Manuscript

The active research objective is a **Nature Medicine paper** ("Towards a cardiac world model") demonstrating that EchoJEPA's frozen representations constitute a cardiac world model: a system that predicts unobserved clinical states at three levels. (1) **Cross-modal**: hemodynamic severity from B-mode alone (MR 0.860, AS 0.908). (2) **Cross-system**: mortality, blood biomarkers from cardiac imaging. (3) **Cross-temporal**: new-onset cardiomyopathy (0.793) from normal baseline echos across 93K longitudinal pairs. The paper additionally covers disease detection, SAE interpretability, and fairness analysis. All downstream tasks use **frozen depth=1 attentive probes** (no fine-tuning) run from video through frozen encoders, with **prediction averaging** across clips per study (Strategy E, adopted 2026-03-11). The companion ICML preprint covers the method and standard benchmarks; Nature Medicine covers novel clinical findings. Datasets: UHN (18M echos, pretraining for G), MIMIC-IV-Echo (pretraining for L and L-K, plus linked to MIMIC-IV clinical data for outcomes, labs, ICD codes, notes).

### Reference Documentation (`claude/`)

The `claude/` directory contains persistent reference docs organized by topic. See `claude/DIRECTORY.md` for the full index with file-level descriptions.

- **`claude/architecture/`** — codebase internals: pretraining pipeline, probe system (attentive/linear/MLP), classifier pipeline
- **`claude/data/`** — datasets and manuscript: `data/` directory layout, Nature Medicine scope, UHN database schemas, MIMIC-IV linkage
- **`claude/dev/`** — development log: bug tracker (6 issues), changelog, code review findings, UHN extraction ops guide, `hyperpod-ops.md` (cluster provisioning, S3 code deployment, non-interactive SSM job submission, 15 troubleshooting issues). See `dev/README.md` for the bug index and planned fixes
- **`claude/nature_medicine/`** — inference tracking for prediction averaging: `inference-tracker.md` (primary 3-model G/EP/Pan NPZ inventory, 69 ready + 57 pending), `inference-tracker-additional.md` (supplementary models B/L-K/L, 59 NPZs). Copies synced to `uhn_echo/nature_medicine/context_files/dev/`
- **`claude/preprint/`** — ICML preprint analysis: encoder fairness confounds, probe architecture mismatch (attentive vs linear inversion), claim validity assessment, hindsight recommendations for camera-ready. Full preprint LaTeX at `user-default-efs/vjepa2/claude/preprint/icml_preprint.tex`
- **`claude/rebuttals/`** — ICML rebuttal (scores 2/3/3/4, all maintained post-rebuttal). See `rebuttals/README.md` for the full file index. Start with **`13-post-rebuttal-outcome.md`** for current status (10-20% acceptance, decision with AC). Key docs: `08-rebuttal-v2.md` (strategy & framing), `10-rebuttal-experiment-results.md` (consolidated results & key findings), `09-three-way-comparison-results.md` (JEPA/BYOL/MAE 50ep detail), `12-checkpoint-reference.md` (all checkpoint paths). **`experiments/frame-shuffling.md`** documents the 6-condition temporal ablation (dropped from ICML rebuttal, intended for NeurIPS). All rebuttal experiments feed Nature Medicine and NeurIPS resubmission
- **`claude/goodfire/`** — Goodfire interpretability report (`goodfire_mar20.pdf`): 31-page analysis of EchoJEPA-L-K vs EchoJEPA-L vs EchoMAE-L representation geometry, attribution, and SAE concept discovery. Key rebuttal-relevant findings: frame shuffling (Fig 25-26), temporal Fourier power (Fig 29), spatial intrinsic dimension (Fig 9). SAE results (single feature rho=0.50 for LVEF) reserved for Nature Medicine.

- **`claude/neurips/`** — NeurIPS 2025 resubmission plan. Start with `README.md` for the high-level plan, experiment status matrix, and 2×2 SALT design. `completed-experiments.md` inventories all done experiments with numbers. `new-experiments.md` lists what needs to be run (SALT is the critical path). `paper-outline.md` has the section-by-section structure. `competitive-landscape.md` covers US-JEPA, SALT paper, and concurrent work positioning

Additional source references: `uhn_echo/nature_medicine/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/mimic/CLAUDE.md`

### Data Auditing Reference

**`uhn_echo/nature_medicine/context_files/data-auditing.md`** — Comprehensive best practices for auditing clinical AI labels and data pipelines. 62 items across 8 categories (label FP/FN, temporal integrity, leakage, confounds, pipeline bugs, statistical pitfalls, documentation), pre-flight and publication checklists, precision audit protocol. Distilled from 100+ real issues encountered across UHN (v1-v7.2) and MIMIC (v1-v4.2) label builds. Read before building or auditing any label set.

## Common Commands

### Setup
```bash
pip install -e .          # editable install (setup script is scripts/setup.py)
```

### Tests
```bash
pytest tests              # all tests
pytest tests/models/test_models.py   # single test file
```

### Linting (CI runs these on app/, evals/*.py, src/, tests/)
```bash
python -m black --check app evals/*.py src tests
python -m isort app evals/*.py src tests --check
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
```

### Pretraining
```bash
python -m app.main --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml --devices cuda:0 cuda:1
python -m app.main_distributed --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml  # SLURM
```

### HyperPod (H100 clusters)
```bash
# Active cluster: echojepa-h100-neurips (training plan: EchoJEPA-NeurIPS, Apr 12 – May 2)
# Compute node: ip-10-0-50-241 (ml.p5.48xlarge, 8x H100 80GB)
# Connect to controller via SSM (see claude/dev/hyperpod-ops.md for full details)
CLUSTER=echojepa-h100-neurips
CLUSTER_ID=n9we8xfqjv3p
GROUP_PREFIX=echojepa-neurips
CTRL_ID="$(aws sagemaker list-cluster-nodes --cluster-name $CLUSTER --region us-west-2 \
  --query "sort_by(ClusterNodeSummaries[?InstanceGroupName=='${GROUP_PREFIX}-controller'], &LaunchTime)[-1].InstanceId" --output text)"
aws ssm start-session --region us-west-2 \
  --target "sagemaker-cluster:${CLUSTER_ID}_${GROUP_PREFIX}-controller-${CTRL_ID}"

# Previous cluster (compute scaled to 0): echojepa-h100-march (ID: yyepvbne5vzr)

# On the controller — deploy latest code and launch any job:
cd ~/EchoJEPA-repo && git pull   # get latest changes (requires GitHub SSH key)
~/deploy.sh                      # push code to compute node(s) via srun
sbatch scripts/<job>.sbatch      # all sbatch scripts use /opt/vjepa2
```

**IMPORTANT: Always deploy code before every `sbatch` submission.** Compute nodes are in a private subnet with no GitHub access. Two deployment workflows exist (see `claude/dev/hyperpod-ops.md`):
1. **Git workflow** (if controller has GitHub SSH key): `cd ~/EchoJEPA-repo && git pull && ~/deploy.sh`
2. **S3 tarball workflow** (current — no git SSH on echojepa-h100-neurips): Create tarball on SageMaker → upload to S3 → deploy via non-interactive SSM + srun. See "Code Deployment via S3" in hyperpod-ops.md.

### Non-Interactive SSM Commands (for Claude Code)

Run commands on the HyperPod controller without an interactive session. **Must** use the `script -q -c "timeout N ..."` wrapper — raw `aws ssm start-session` hangs or drops output without it. The `AWS-StartNonInteractiveCommand` document is only available on the controller, not compute nodes.

```bash
# Template: run a command on the controller
TARGET="sagemaker-cluster:${CLUSTER_ID}_${GROUP_PREFIX}-controller-${CTRL_ID}"
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu COMMAND_HERE\"]}'" /dev/null 2>&1 \
  | grep -v "Exiting session\|Starting session"

# Check SLURM queue
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu squeue -u ubuntu -l\"]}'" /dev/null 2>&1

# Check completed jobs
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu sacct --starttime=2026-04-12 --format=JobID,JobName%30,State%12,Elapsed,ExitCode -n\"]}'" /dev/null 2>&1

# Read job output from compute node (via srun from controller)
script -q -c "timeout 60 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu srun -N1 --ntasks=1 bash -c \\\"tail -80 /tmp/JOB_OUTPUT_FILE\\\"\"]}'" /dev/null 2>&1

# Deploy code via S3 tarball and submit job
script -q -c "timeout 120 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu bash -c \\\"aws s3 cp s3://BUCKET/setup/vjepa2-src.tar.gz /tmp/ --quiet && srun -N1 --ntasks=1 bash -c \\\\\\\"sudo tar xzf /tmp/vjepa2-src.tar.gz -C /opt/vjepa2\\\\\\\" && echo DEPLOY_OK\\\"\"]}'" /dev/null 2>&1

# Submit sbatch (extract script to controller first, then submit)
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu sbatch /tmp/vjepa2-ctrl/scripts/JOB.sbatch\"]}'" /dev/null 2>&1
```

**Key gotchas:**
- `AWS-StartNonInteractiveCommand` is NOT available on compute nodes (only controller)
- Without `script -q -c "timeout N ..."`, SSM sessions hang indefinitely or produce no output
- Escape nested quotes carefully: outer `'...'`, then `\"...\"`; for srun subcommands use `\\\"...\\\"`
- sbatch scripts run from the controller but execute on compute nodes — the script must exist on the controller (e.g., `/tmp/vjepa2-ctrl/scripts/`)
- Job stdout/stderr files (e.g., `/tmp/nmed_a4c-23.out`) are on the compute node, not the controller — use `srun` to read them

### Probe Evaluation (Primary — d=1 attentive probes from video, Strategy E)
```bash
# Nature Medicine primary evaluation: d=1 attentive probe + prediction averaging
python -m evals.main --fname configs/eval/vitg-384/view/verification/verify-echojepa-g-d1.yaml --devices cuda:0 cuda:1
python -m evals.main --fname configs/inference/vitl/lvef.yaml --devices cuda:0 --val_only  # inference only
```

See `README.md` Quickstart sections for full examples.

## Code Style

- **black** (line-length 119), **isort** (profile=black), **flake8** (max-line-length 119, ignore E203/E701/W503)
- Config: `pyproject.toml` for black/isort, `.flake8` for flake8
- Dev lint deps: `pip install -r requirements-test.txt`

## Architecture

### Entry Points & Dispatch

Training and evaluation use a scaffold pattern:
- `app.main` / `app.main_distributed` → `app.scaffold.main(app_name)` → `app.<app_name>.train.main()`
- `evals.main` / `evals.main_distributed` → `evals.scaffold.main(eval_name)` → `evals.<eval_name>.eval.main()`

The `app_name` / `eval_name` comes from the YAML config (`app:` or `eval_name:` field) and maps directly to a Python subpackage.

### Core Library (`src/`)

- **`src/models/`**: VisionTransformer encoder (ViT-T through ViT-g with RoPE, SDPA, activation checkpointing), VisionTransformerPredictor (latent target prediction), AttentivePooler/Classifier/Regressor (cross-attention probe heads), LinearPooler variants
- **`src/datasets/`**: `VideoDataset` (single-view, decord-based, S3+local), `VideoGroupDataset` (multi-view studies), `data_manager.init_data()` factory, `DistributedStudySampler` (1 random clip per study per epoch for study-level tasks)
- **`src/masks/`**: `MaskCollator` for spatio-temporal block masking (context + target masks)
- **`src/utils/`**: distributed init (NCCL, supports both local multi-GPU and SLURM/submitit), LR schedulers (WarmupCosine, WSD), checkpoint loading with retry logic

### Training Objective

Encoder processes visible (context) tokens; predictor takes encoder output + mask token positions to predict target encoder's representations of masked regions. Target encoder updated via EMA.

### Eval Modules

- `evals/video_classification_frozen/` — single-view probe (classification or regression)
- `evals/video_classification_frozen_multi/` — multi-view probe (multiple echo views per study)
- Both support `probe_type: attentive | linear | mlp` in config, and contain `modelcustom/` backbone adapters for VJepa2, EchoPrime, PanEcho, VideoMAE, EchoFM

### Probe Types

Three probe architectures in `src/models/`. Set via `experiment.classifier.probe_type` in YAML config:
- **`attentive`** (default) — learned cross-attention pooling + self-attention blocks. **Nature Medicine primary (Strategy E): depth=1 (`num_probe_blocks: 1`) for all models.** At depth=1, only cross-attention (no SA blocks), which is fair for both 1568-token and 1-token models. Config: `num_heads`, `num_probe_blocks`. Verification configs: `configs/eval/vitg-384/view/verification/`
- **`linear`** — mean-pool + LayerNorm + single linear layer. Config: `use_layernorm`, `dropout`
- **`mlp`** — mean-pool + LayerNorm + 2-layer MLP. Middle ground

See `claude/architecture/probe-system.md` for full details including attentive-vs-linear comparison and hyperparameter guidance.

### Legacy Embedding Pipeline (`evals/`) — SUPERSEDED

The NPZ-based embedding pipeline (extract → mean-pool → sklearn) has been superseded by Strategy E (d=1 attentive probes from video + prediction averaging). These scripts remain in the codebase but are no longer used for the Nature Medicine evaluation:

- `evals/extract_embeddings.py` — clip-level extraction → master NPZ
- `evals/remap_embeddings.py` — per-task label NPZs
- `evals/pool_embeddings.py` — mean-pool clip → study-level
- `evals/train_probe.py` — sklearn linear probes on NPZ embeddings

### Config System

All experiments driven by YAML configs in `configs/`. Each subdirectory has a README.md with full details:
- `configs/train/` — pretraining and cooldown, by model size (vitl16, vith16, vitg16). Naming: `{phase}-{dataset}-{resolution}-{frames}.yaml`
- `configs/eval/` — probe training, by model (vitg-384, vitl) and task (lvef, rvsp, view, color, quality, zoom, tapse). `old/` contains archived configs. `multihead_kwargs` list trains multiple probes in parallel with different hyperparameters.
- `configs/inference/` — inference-only configs (set `val_only: true`, require `probe_checkpoint`). Specialized subdirs: `depth_attenuation/`, `echonet-dynamic/`, `echonet-pediatric/`

### Study-Level Evaluation (Nature Medicine)

For study-level tasks (MIMIC clinical outcomes), CSVs contain ALL clips per study. The `study_sampling: true` config flag activates `DistributedStudySampler` (`src/datasets/study_sampler.py`), which:
1. Groups CSV rows by study_id (extracted from S3 path via regex)
2. Each epoch, selects 1 random clip per study (cross-view augmentation)
3. Distributes across ranks (drop-in replacement for `DistributedSampler`)

At validation/test time, all clips are scored independently and predictions are averaged per study.

Config: `configs/eval/vitg-384/nature_medicine/`. CSV builder: `experiments/nature_medicine/mimic/build_probe_csvs.py`.

### Prediction Averaging Results (UHN)

Study-level prediction averaging outputs are at:
```
evals/vitg-384/nature_medicine/uhn/video_classification_frozen/{task}-predavg-{model}/
```
Each directory contains `study_predictions.csv` (one row per study with ground truth, predictions, and per-class probabilities) and `log_r0.csv` (epoch-level metrics). **139 files** across 27 tasks × 6 models (echojepa-g, echojepa-b, echojepa-l, echojepa-l-k, echoprime, panecho). Newer runs also produce `clip_outputs.npz` (clip-level predictions and features for all HP heads).

To recompute statistics from the study_predictions.csv files:
```bash
python scripts/compute_predavg_stats.py                          # all tasks, all models
python scripts/compute_predavg_stats.py --task mr_severity       # filter by task
python scripts/compute_predavg_stats.py --model echojepa-g       # filter by model
python scripts/compute_predavg_stats.py --csv output.csv         # save to CSV
```

### Dataset CSV Formats

- **Pretraining**: `path/to/video.mp4 0` (space-delimited, dummy label)
- **Classification**: `path/to/video.mp4 <int_label>` (space-delimited)
- **Regression**: `path/to/video.mp4 <raw_float>` (space-delimited, raw values; Z-score normalized at runtime)
- **Multi-view**: `path/to/view1.mp4 path/to/view2.mp4 <float_label>` (space-delimited)

### Distributed Training

- Local: `mp.Process` per GPU with `CUDA_VISIBLE_DEVICES`
- SLURM: `submitit` for cluster jobs
- Both use NCCL backend via `src/utils/distributed.init_distributed()`
- Debug mode: `--debugmode True` runs rank 0 in main process

### Pretrained Checkpoints

Available via torch.hub (`facebookresearch/vjepa2`): `vjepa2_vit_large`, `vjepa2_vit_huge`, `vjepa2_vit_giant`, `vjepa2_vit_giant_384`. Hub entry point is `scripts/hubconf.py`.

### Classifier Pipeline (`classifier/`)

ConvNeXt/Swin image classifiers for echo view/color/quality/zoom, plus distributed inference on the 18M dataset. See `claude/architecture/classifier-pipeline.md` for full details.

### Data Directory (`data/`)

JEPA-format splits, raw labels, scalers, notebooks, and scripts. See `claude/data/data-directory.md` for full details. **Do not rename files in `csv/`** without updating corresponding eval configs.

`data/data/` contains the UHN deidentification key and mapping notebooks (deid_key.csv, identifiers.ipynb, ecs_mappings.ipynb, syngo.ipynb). Working copies of the key mapping files are at `experiments/nature_medicine/uhn/mapping/` (gitignored). See `claude/data/uhn-mapping.md` for the full mapping chain documentation.

### Other Root Directories

- `checkpoints/` — all model weights: pretrain, anneal, cooldown, eval probes, SSv2 probe
- `indices/` — S3 URI manifests for the 18M dataset (`master_index_18M.csv`, `master_index_18M_cleaned.csv`, `s3_pretrain.csv`, annotations)
- `experiments/` — precomputed frozen embeddings organized by paper: `icml/` (UHN benchmarks), `nature_medicine/mimic/` (7 models, 23 tasks), `nature_medicine/uhn/mapping/` (DICOM↔Syngo deid keys, gitignored). UHN label provenance docs: `nature_medicine/uhn/DATASET_PROVENANCE.md` (all tasks), `nature_medicine/uhn/UHN_DISEASE_PROVENANCE.md` (disease v1-v7.2), `nature_medicine/uhn/CLASS_MAPS.md` (class mappings). See `experiments/README.md`, `claude/data/embedding-pipeline.md`, `claude/data/uhn-mapping.md`
- `predictions/` — probe and classifier prediction CSVs (LVEF, RVSP, EchoNet, view, quality, zoom)
- `results/` — data efficiency experiment runs (epoch checkpoints + logs)
- `scripts/` — SBATCH scripts, Python utilities, demos, `run_details.md`
- `notebooks/` — project-level analysis and demo Jupyter notebooks
- `figures/` — publication-quality UMAP plots (PDF + PNG)
- `logs/` — training and evaluation logs (`.log` files, SLURM `.out` files, training plots)
- `uhn_echo/` — UHN research data, Nature Medicine analysis, MIMIC-IV linkage
