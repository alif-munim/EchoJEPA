# HyperPod Deployment and Job Submission

Environment deployment, job submission, monitoring, and multi-node training on SageMaker HyperPod. For cluster creation see [hyperpod-cluster-creation.md](hyperpod-cluster-creation.md). For connectivity and remote commands see [hyperpod-ops.md](hyperpod-ops.md).

## Code Deployment (Default Workflow)

The git repo is cloned on the **controller** at `~/EchoJEPA-repo`. Use `scripts/deploy.sh` to push code to compute nodes:

```bash
# On the controller:
cd ~/EchoJEPA-repo
git pull                        # get latest from GitHub
~/deploy.sh                     # deploy to ip-10-0-50-241
~/deploy.sh ip-10-0-50-83      # deploy to other node (optional)
sbatch ~/vjepa2_pretrain_h100.sbatch
```

The deploy script tars the repo (excluding .git, checkpoints, large data) and unpacks it on the compute node via srun. This is the standard workflow because:
- Controller and compute nodes have **no shared filesystem**
- Compute nodes are in a private subnet with **no GitHub access**
- The controller has SSH access to GitHub for git pull/push

After editing code on the controller (e.g. via Claude Code), always run `~/deploy.sh` before launching training.

## Code Deployment via S3 (No Git on Controller)

The `echojepa-h100-neurips` controller does **not** have GitHub SSH keys configured. Use the S3 tarball approach to deploy code from SageMaker:

```bash
# 1. On SageMaker: create minimal source tarball using file list (see Issue 18 in hyperpod-troubleshooting.md)
cd /path/to/vjepa2
# Build file list (Python source only, ~280 KB)
find app/ src/ configs/ -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.txt' -o -name '*.cfg' \) ! -path '*__pycache__*' > /tmp/deploy_files.txt
find evals/ -maxdepth 1 -name '*.py' >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/eval.py" >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/models.py" >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/utils.py" >> /tmp/deploy_files.txt
find evals/video_classification_frozen/modelcustom/ -maxdepth 1 -name '*.py' >> /tmp/deploy_files.txt
find evals/video_classification_frozen/modelcustom/PanEcho/ -type f -name '*.py' >> /tmp/deploy_files.txt
echo "setup.py" >> /tmp/deploy_files.txt && echo "pyproject.toml" >> /tmp/deploy_files.txt && echo ".flake8" >> /tmp/deploy_files.txt
echo "scripts/<your_job>.sbatch" >> /tmp/deploy_files.txt
rm -f /tmp/vjepa2-src.tar.gz  # MUST delete first -- tar may append to existing
tar czf /tmp/vjepa2-src.tar.gz -T /tmp/deploy_files.txt

# 2. Upload to S3 (artifacts bucket, not lifecycle bucket)
S3_ARTIFACTS="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
aws s3 cp /tmp/vjepa2-src.tar.gz ${S3_ARTIFACTS}/setup/vjepa2-src.tar.gz

# 3. Deploy: download on controller, then download+extract on compute node via srun
#    (controller and compute don't share /tmp -- compute must download from S3 directly)
TARGET="sagemaker-cluster:n9we8xfqjv3p_echojepa-neurips-controller-i-0415ce8f417564270"

# Download tarball on controller (for sbatch extraction)
script -q -c "timeout 60 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"bash -c \\\"aws s3 cp ${S3_ARTIFACTS}/setup/vjepa2-src.tar.gz /tmp/vjepa2-src.tar.gz --quiet && echo DOWNLOAD_OK\\\"\"]}'" /dev/null 2>&1

# Download+extract on compute node (must use S3, not controller's /tmp)
script -q -c "timeout 120 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"bash -c \\\"srun -N1 --ntasks=1 bash -c \\\\\\\"aws s3 cp ${S3_ARTIFACTS}/setup/vjepa2-src.tar.gz /tmp/vjepa2-src.tar.gz --quiet && sudo tar xzf /tmp/vjepa2-src.tar.gz -C /opt/vjepa2 && echo DEPLOY_OK\\\\\\\"\\\"\"]}'" /dev/null 2>&1

# Extract sbatch to controller
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"bash -c \\\"mkdir -p /tmp/vjepa2-ctrl && tar xzf /tmp/vjepa2-src.tar.gz -C /tmp/vjepa2-ctrl scripts/<your_job>.sbatch && echo EXTRACT_OK\\\"\"]}'" /dev/null 2>&1

# 4. Submit sbatch from controller
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target '$TARGET' --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"sudo -u ubuntu sbatch /tmp/vjepa2-ctrl/scripts/<your_job>.sbatch\"]}'" /dev/null 2>&1
```

**Key lessons from 2026-04-13 session:**
- Controller and compute nodes do NOT share `/tmp` -- the tarball must be downloaded independently on each
- `sudo -u ubuntu bash -c "..."` swallows stdout in SSM; use `bash -c "..."` (runs as root) for commands that need visible output
- `srun` on a fully-occupied node blocks -- use `srun --jobid=<ID> --overlap` to share resources with a running job
- Output dir must be writable by `ubuntu` -- use `/tmp/...` not `/opt/vjepa2/...` (see Issue 16 in [hyperpod-troubleshooting.md](hyperpod-troubleshooting.md))

**CRITICAL -- rebuild the tarball before every VideoMAE (or other S3-backed pretraining) submission.** Stale tarballs containing the pre-`d91b4d4` `s3_dataset.py` cause silent loss collapse on S3 hiccups. See [Issue 20](hyperpod-troubleshooting.md#20-videomae-loss-silently-collapses-to-0-mid-epoch-1-2026-04-21) and [bug 020](bugs/020-videomae-dummy-zeros-loss-collapse.md). Quick verification after upload:
```bash
aws s3 cp ${S3_ARTIFACTS}/setup/vjepa2-src.tar.gz - | tar xzO s3_dataset.py | grep -c max_retries
# Expect 1 (has fix) or 0 (stale -- DO NOT submit)
```

**Important**: sbatch runs on the controller, not the compute node. The sbatch script itself is read by Slurm on the controller, so it must be accessible there (extracted to `/tmp/vjepa2-ctrl/`). The actual Python code runs on the compute node at `/opt/vjepa2`.

**CRITICAL — srun vs direct python for eval/probe jobs:**
- **Pretraining** (`app.main`, `run_mae_pretraining.py`): Use `srun --ntasks-per-node=8` — these expect one process per GPU via DDP.
- **Eval/probe** (`evals.main`): Use `--ntasks-per-node=1` and run `python3 -m evals.main` directly (NO srun). The eval code uses `mp.Process` internally to spawn one worker per GPU. Using `srun --ntasks-per-node=8` causes each rank to spawn 8 workers (64 total), `init_distributed()` returns `world_size=1`, and `DistributedSampler` doesn't split data — resulting in 8x slowdown (7465 vs 934 iters/epoch on EchoNet-Dynamic).

**Gotcha**: conda activate scripts fail under `set -u` (unbound `CONDA_PREFIX`). Use `set -eo pipefail` before `source activate`, then `set -u` after.

**CRITICAL — AWS credential expiration on compute nodes (2026-04-20):**
Compute nodes use the IAM instance metadata service for credentials (no static credentials file). `mp.Process` workers (used by `evals.main` and `app.main`) cache the initial STS token and do NOT refresh it. Jobs running >1h will fail with `Unable to locate credentials` when the token expires.

**Fix:** Run `setup_aws_creds.sh` on each compute node (deployed to S3 at `setup/setup_aws_creds.sh`). This:
1. Fetches current IAM role credentials from the metadata service
2. Writes them to `/home/ubuntu/.aws/credentials`
3. Sets up a cron job to refresh every 30 minutes

The sbatch env vars `AWS_SHARED_CREDENTIALS_FILE=/home/ubuntu/.aws/credentials` then point to the refreshed file. Run this once after node provisioning or reboot:
```bash
srun -p dev -w ip-10-0-50-39 -N1 --ntasks=1 bash -c \
  "aws s3 cp s3://.../setup/setup_aws_creds.sh /tmp/ --quiet && bash /tmp/setup_aws_creds.sh"
```

## The conda-pack Approach (One-Time Setup)

The conda environment (`/opt/vjepa2-312`) is deployed separately since it rarely changes:

1. **Create conda env** on source machine (A100 notebook instance)
2. **Pack with conda-pack**: `conda pack -n vjepa2-312 -o vjepa2-312.tar.gz` (~4.5 GB)
3. **Upload to S3**: `aws s3 cp vjepa2-312.tar.gz s3://<bucket>/setup/`
4. **Unpack on compute node**: `tar xzf vjepa2-312.tar.gz -C /opt/vjepa2-312 && source /opt/vjepa2-312/bin/activate && conda-unpack`

## Legacy: Source Code via S3

Before the git+deploy workflow was set up, code was deployed via S3 tarball. This is still used by the lifecycle script for initial node provisioning:

```bash
# On notebook instance: create tarball excluding large files
cd /path/to/vjepa2
tar czf /tmp/vjepa2-src.tar.gz \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.pkl' --exclude='*.zip' \
    --exclude='*.dcm' --exclude='*.csv' --exclude='checkpoints' --exclude='experiments' \
    --exclude='evals/video_classification_frozen/*/output*' \
    --exclude='evals/video_classification_frozen/modelcustom/EchoPrime/model_data*' \
    --exclude='data/samsung' --exclude='data/data' --exclude='.git' \
    .

# Include specific CSVs needed
tar rf /tmp/vjepa2-src.tar data/csv/mimic_annotations_s3.csv

aws s3 cp /tmp/vjepa2-src.tar.gz s3://<bucket>/setup/
```

## Lifecycle Script Approach

HyperPod runs `on_create.sh` during node provisioning. This can automate environment setup:

```bash
# Detect compute node by GPU presence
if command -v nvidia-smi &>/dev/null; then
    # Download and unpack conda env, source code, checkpoint
    aws s3 cp s3://<bucket>/setup/vjepa2-312.tar.gz /tmp/
    tar xzf /tmp/vjepa2-312.tar.gz -C /opt/vjepa2-312
    # ... etc
fi
```

**Caveats**:
- The lifecycle script is downloaded from S3 at provisioning time. If you update S3 after node launch, the running node uses the OLD script
- `batch-replace-cluster-nodes` triggers re-provisioning with the current S3 script
- Background processes (`nohup`, `&`) survive the script exit but are harder to debug
- Use `set +e` in subshells to prevent provisioning failures from killing the node

## S3 Bucket Layout (echojepa-h100-neurips)

```
s3://sagemaker-echojepa-h100-neurips-f85ad7df-bucket/
  on_create.sh                          # lifecycle script (references h100-march bucket for setup artifacts)
  lifecycle_script.py                   # standard HyperPod provisioning
  provisioning_parameters.json          # cluster provisioning params (MUST list all instance groups)
  apply_hotfix.sh                       # hotfix runner (globs hotfix/*.sh)
  hotfix/                               # REQUIRED -- apply_hotfix.sh fails if missing
    hold-lustre-client.sh
    mock-gpu-driver-deb.sh
  utils/                                # Slurm utilities, enroot, keypair scripts
  observability/                        # DCGM/NCCL/OTel metric exporters
  multi_headnode_setup/                 # multi-controller scripts
  setup/
    setup_done_<hostname>.txt           # completion markers (written by on_create.sh)
```

The `on_create.sh` pulls heavy setup artifacts (conda env, source code) from the echojepa-h100-march bucket's `setup/` directory to avoid duplicating ~10 GB of data. It does NOT auto-launch training -- use `~/deploy.sh` + `sbatch` after provisioning.

## S3 Bucket Layout (echojepa-h100-march)

```
s3://sagemaker-echojepa-h100-march-0d224785-bucket/
  on_create.sh                          # lifecycle script
  setup/
    vjepa2-312.tar.gz                   # conda env (4.5 GB)
    vjepa2-src.tar.gz                   # source code (5.3 MB)
    latest.pt                           # checkpoint (4.8 GB, epoch 118)
    mimic_annotations_s3.csv            # training data CSV (42 MB)
    pretrain-21-mimic-224px-16f-h100.yaml  # H100 training config
    setup_done_<hostname>.txt           # completion markers
    training_started_<hostname>.txt     # training launch markers
  checkpoints/
    vjepa-2.1-l-h100/                   # V-JEPA 2.1 ViT-L checkpoints
    byol-vitl-imagenet/                 # BYOL v1 (stopped -- EMA plateau)
    byol-vitl-imagenet-v2/              # BYOL v2 (active -- constant EMA)
```

## Job Submission

### From the Controller

Always submit jobs from the controller node via Slurm. Never run training directly on compute nodes.

```bash
sudo su - ubuntu

# Submit
JOBID=$(sbatch --parsable ~/vjepa2_pretrain_h100.sbatch)
echo "JOBID=$JOBID"
squeue -j "$JOBID"

# Monitor
NODE=$(squeue -j $JOBID -h -o %N)
srun --jobid="$JOBID" --overlap -N 1 -n 1 -w "$NODE" bash -lc \
  "tail -n 200 -F /tmp/vjepa21_pretrain-${JOBID}.out /tmp/vjepa21_pretrain-${JOBID}.err"

# Status
sacct -j "$JOBID" --format=JobID,JobName,State,Elapsed,NodeList%30,AllocTRES%40
```

### Example sbatch Script (V-JEPA 2.1 Pretrain on H100)

See `scripts/vjepa2_pretrain_h100.sbatch` for the canonical version. Key points:
- The `LD_LIBRARY_PATH` override is **required** to avoid the cuBLAS 12.9 bf16 bug (see Issue 9 in [hyperpod-troubleshooting.md](hyperpod-troubleshooting.md))
- `/dev/shm` cleanup prevents shared memory exhaustion from prior crashes

```bash
#!/bin/bash
#SBATCH --job-name=vjepa21-pretrain
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/vjepa21_pretrain-%j.out
#SBATCH --error=/tmp/vjepa21_pretrain-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=ip-10-0-50-241

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
cd /opt/vjepa2

# Force PyTorch's bundled CUDA 12.8 libs instead of system CUDA 12.9
# (cuBLAS 12.9.1.4 has a bf16 gemm bug on H100)
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

rm -rf /dev/shm/* 2>/dev/null || true

python -m app.main \
    --fname configs/train/vitl16/pretrain-21-mimic-224px-16f-h100.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

### Resuming VideoMAE Pretraining After a Crash

VideoMAE pretraining (`run_mae_pretraining.py`) supports resume via `--auto_resume` (default on; the sbatch does not pass `--no_auto_resume`). Two mechanisms work together:

1. **`checkpoint-latest.pth` every epoch** (added 2026-04-21 in `evals/video_classification_frozen/modelcustom/VideoMAE/run_mae_pretraining.py:313`). Written unconditionally at each epoch boundary, overwriting in place. Negligible cost (~5 s / epoch for ViT-B). `auto_load_model` in `utils.py` prefers this file whenever its `epoch` field is >= the highest-numbered `checkpoint-N.pth`, so the most recent state wins regardless of `save_ckpt_freq`.
2. **Numbered `checkpoint-N.pth` every `save_ckpt_freq` epochs** (default 5). These persist; `checkpoint-latest` is overwritten.

Because each SLURM job gets a new `WORKDIR=/opt/dlami/nvme/<job>_<SLURM_JOB_ID>/`, checkpoints from the failed run are not visible to the new run's `auto_resume` unless we pull them from S3 first. The sbatch script handles this via `RESUME_FROM_JOB`:

```bash
# On the controller: resume ViT-B MAE from job 298
RESUME_FROM_JOB=298 sbatch scripts/videomae_pretrain_mimic_vitb_in21k.sbatch

# Cold start (no resume)
sbatch scripts/videomae_pretrain_mimic_vitb_in21k.sbatch
```

When `RESUME_FROM_JOB` is set, the sbatch runs `aws s3 sync s3://<bucket>/runs/mae_vitb_in21k_${RESUME_FROM_JOB}/training_folder/ ${OUTPUT_DIR}/` before launch, so whatever `.pth` files were uploaded by the periodic `sync_ckpts` trap (every 15 min) or the `on_exit` trap are available locally.

**What gets synced:** any `.pth`, `.pt`, `.yaml`, `.json`, `.csv` under the training folder. The `checkpoint-latest.pth` is included.

**Limit:** If the crashed job died mid-epoch-1 (as job 298 did), no checkpoint was ever written, so there is nothing to resume from. Only useful once the run has completed at least one epoch.

**Prerequisite on resubmit:** rebuild and re-upload `vjepa2-src.tar.gz` so the compute node picks up the latest `run_mae_pretraining.py` / `utils.py` with the `checkpoint-latest` logic. See Issue 18 in [hyperpod-troubleshooting.md](hyperpod-troubleshooting.md) for the file-list tar approach.

### Installing Claude Code on Compute Nodes

```bash
# From controller, get a shell on compute node
srun -N1 -w ip-10-0-50-241 --ntasks=1 --pty bash

# Install Node.js and Claude Code
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
claude
```

## Monitoring Running Jobs

### Claude Code Context

Claude Code can run on the **controller node** directly (via SSM interactive session) where it can use `squeue`, `sinfo`, `sacct`, and `srun` natively. From a **remote machine** (e.g., SageMaker notebook), use the non-interactive SSM pattern in [hyperpod-ops.md](hyperpod-ops.md). SSH from controller to compute nodes fails (`Permission denied (publickey)`), so all compute-node commands must go through `srun`.

### Checking Training Progress

Use `srun --overlap` to run commands on compute nodes without interfering with the running job:

```bash
JOBID=183
NODE=ip-10-0-50-83

# Check latest training iteration (CSV log)
srun --jobid=$JOBID --nodes=1 --nodelist=$NODE --ntasks=1 --overlap \
  bash -c "tail -5 /opt/vjepa2/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet/log_r0.csv"

# Check checkpoint file timestamps and sizes
srun --jobid=$JOBID --nodes=1 --nodelist=$NODE --ntasks=1 --overlap \
  bash -c "ls -lht /opt/vjepa2/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet/*.pt | head -10"

# Check GPU utilization
srun --jobid=$JOBID --nodes=1 --nodelist=$NODE --ntasks=1 --overlap \
  bash -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Check S3 checkpoint archive
aws s3 ls s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/

# Check disk space (critical -- checkpoint saves need ~5GB headroom)
srun --jobid=$JOBID --nodes=1 --nodelist=$NODE --ntasks=1 --overlap \
  bash -c "df -h /opt/vjepa2"
```

### Key Nuances

1. **`slurmstepd: error: couldn't chdir` warnings are harmless** -- they appear because the controller's CWD doesn't exist on compute nodes. The command still runs from /tmp.

2. **stdout is fully buffered under srun** -- `grep` on the sbatch stdout file (e.g., `/tmp/byol_2node-183.out`) may show nothing even while training is active. Python stdout is not line-buffered when piped through srun. Use the CSV log files (`log_r0.csv`) or checkpoint timestamps for reliable progress monitoring.

3. **CSV log format**: `epoch,iter,loss,total_ms,gpu_ms,unstable_flag` -- the last column is 0 for normal, 1 if loss was unstable.

4. **Checkpoint saves happen at epoch boundaries** -- if a job just started and you see stale checkpoint timestamps, it may just mean the first epoch hasn't completed yet. Check `log_r0.csv` to see current epoch/iter and estimate when the next save will occur (`ipe` iterations per epoch x time per iter).

5. **Disk space on /opt**: Compute nodes start with ~15GB free on /opt. The S3 setup cache (`latest.pt` ~4.8GB, `vjepa2-312.tar.gz` ~4.5GB) in /tmp can fill the disk. If space is tight: `sudo rm -f /tmp/latest.pt /tmp/vjepa2-312.tar.gz`.

## Multi-Node Distributed Training

### Architecture

- **Single-node**: `app.main` spawns `mp.Process` per GPU, sets `CUDA_VISIBLE_DEVICES` itself
- **Multi-node**: `app.main_srun` is called once per srun task; SLURM manages processes. Each task = 1 GPU = 1 DDP rank. GPU assignment via `SLURM_LOCALID`, rank via `SLURM_PROCID`/`SLURM_NTASKS`

### No Shared Filesystem

The biggest HyperPod constraint for multi-node training. Key implications:

1. **Code must be deployed to ALL nodes** before launching. The sbatch script runs from the first node's `/opt/vjepa2`, but srun spawns tasks on all nodes. Each node must have identical code at `/opt/vjepa2/`.
   ```bash
   ~/deploy.sh ip-10-0-50-83    # deploy to node 1
   ~/deploy.sh ip-10-0-50-184   # deploy to node 2
   ```

2. **Checkpoints must be synced** before resuming. Only rank 0 (on the first node) saves checkpoints. When restarting, other nodes won't have the latest checkpoint unless explicitly copied. The sbatch script handles this:
   ```bash
   CKPT_DIR=/opt/vjepa2/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet
   for node in $OTHER_NODES; do
       srun --nodes=1 --nodelist=$node --ntasks=1 bash -c "cat > $CKPT_DIR/$f" < "$CKPT_DIR/$f"
   done
   ```

3. **Checkpoint saves only happen on the first node** -- rank 0 saves locally and uploads to S3. Other nodes never write checkpoints. This means S3 is the durable copy.

### MASTER_ADDR Bug (Fixed 2026-03-27)

`src/utils/distributed.py` previously always overwrote `MASTER_ADDR` with `localhost` or `HOSTNAME`, which breaks multi-node NCCL since each node would set its own hostname as master. Fix: only set `MASTER_ADDR` if not already configured by the sbatch script.

### Batch Size Scaling

When going from 1 node (8 GPUs) to 2 nodes (16 GPUs), halve per-GPU `batch_size` to keep effective batch constant. Example: `batch_size: 64` (single-node, effective=512) -> `batch_size: 32` (2-node, effective=512). This preserves training dynamics and allows controlled speedup comparison.

**Note**: The repo config (`pretrain-byol-mimic-224px-16f-h100.yaml`) uses `batch_size: 64` for the 2-node run (64 x 16 = 1024 effective, matching V-JEPA). For single-node, use `batch_size: 128` to maintain the same effective batch.

### Performance

- Single-node (8x H100): ~7.0s/iter
- 2-node (16x H100): ~3.6s/iter (1.94x speedup)
- Scaling efficiency: 97% (near-linear)

### 2-Node sbatch Script

See `~/byol_pretrain_2node.sbatch` on the controller. Key differences from single-node:
- `--nodes=2`, `--ntasks-per-node=8`, `--nodelist=ip-10-0-50-83,ip-10-0-50-184`
- Sets `MASTER_ADDR` from SLURM node list
- Syncs checkpoints from first node to other nodes before launch
- Cleans `/dev/shm` on all nodes via `srun --ntasks-per-node=1`
- Launches via `srun python -m app.main_srun` (not `app.main`)

## BYOL-Video ViT-L Training Status

### V2 Run (2026-03-27, Active)

Fresh start with matched V-JEPA config. Constant EMA, full batch size, ImageNet-21k init.

- **Job**: 241 (2-node, 16x H100), resumed from epoch 10
- **Config**: `configs/train/vitl16/pretrain-byol-mimic-224px-16f-h100.yaml`
- **Key settings**: EMA [0.99925, 0.99925] constant, batch 64x16=1024, 240 epochs, warmup 40
- **S3**: `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/`
- **Periodic saves**: every 2 epochs, 1 kept locally (all archived to S3)
- **Iter time**: ~7.2s, ~18 min/epoch, ETA ~3 days for full 240 epochs
- **Loss trajectory** (improving monotonically, no plateau):

| Epoch | Avg Loss |
|-------|----------|
| 1 | -1.659 |
| 2 | -1.826 |
| 5 | -1.930 |
| 10 | -1.967 |

### V1 Run (2026-03-27, Stopped -- representation degradation)

- **Problem**: Cosine EMA [0.996, 1.0] froze target encoder by epoch ~12, causing representation degradation
- **Evidence**: LVEF probe Pearson r dropped from 0.151 (e10) to 0.089 (e40) -- online encoder got WORSE
- **Loss**: plateaued at -1.987 (epoch 12), drifted to -1.955 by epoch 44
- **S3**: `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet/` (e10, e15, e42, latest)

### V-JEPA 2.1 ViT-L (2026-03-26, Completed)

- **A100 run** (SageMaker notebook, 8x A100 80GB): Epoch 118/240, ~8s/iter
- **H100 run** (HyperPod echojepa-h100-march): Completed, ~2.9s/iter GPU time
- Config: `configs/train/vitl16/pretrain-21-mimic-224px-16f-h100.yaml`
- S3: `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/vjepa-2.1-l-h100/`

### H100 Environment (March 2026)

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 580.126.09 |
| System CUDA | 13.0, 12.9, 12.8, 12.6 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12 (conda env at /opt/vjepa2-312) |
| Instance | ml.p5.48xlarge (8x H100 80GB HBM3, sm_90) |
