# HyperPod Cluster Operations Guide

Operational guide for SageMaker HyperPod GPU clusters used for EchoJEPA training. Covers cluster setup, connectivity, environment deployment, and job submission. Distilled from the echojepa-h100-march setup (2026-03-26).

## Cluster Inventory

| Cluster | ID | Instance Type | GPUs | Training Plan | Status |
|---------|------|--------------|------|---------------|--------|
| echojepa-v10 | swcxoboj2tln | ml.p5e.48xlarge | 8x H200 | EchoJEPA | InService |
| echojepa-h200 | 2paq9e2d06dk | ml.p5e.48xlarge | 8x H200 | EchoJEPA-H200 | InService |
| echojepa-h100-march | yyepvbne5vzr | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NMED-H100-V1 | InService |
| echojepa-h100-nmed | 8186qucd36mr | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NMED-H100-V1 | Failed |

## Cluster Creation Checklist

### Prerequisites

1. **Service quota**: Increase for desired instance type (e.g., ml.p5.48xlarge, ml.p5e.48xlarge)
2. **Training plan reservation**: Create under SageMaker Training Plans

### Network Config (Critical)

1. Identify the **Availability Zone** pinned to the Training Plan (e.g., usw2-az3)
2. Create a **private subnet** in that specific AZ within VPC `vpc-0a306d982844ee4e9`
3. **Associate the subnet with the Route Table** (`rtb-0dc170...`) that contains the S3 Gateway Endpoint. Default route tables often lack this, causing S3 timeout errors
4. Ensure the VPC has active **Interface Endpoints** for `ssm`, `ssmmessages`, and `ec2messages`

### Security & IAM

- **Security Group**: Use `sg-0c1b3b9f78325dc0c` (patched to allow traffic to VPC endpoints)
- **IAM Role**: `AmazonSageMaker-ExecutionRole-20250409T120880`
  - Must have `AmazonSSMManagedInstanceCore` attached
  - Must have `ec2.amazonaws.com` in the trust policy (in addition to `sagemaker.amazonaws.com` and `codebuild.amazonaws.com`) for SSM agent registration
  - Has custom inline policy for SSM access

### Launch

- Use default lifecycle scripts; create a new S3 bucket per cluster
- Verify subnet AZ matches instance group AZ
- Provisioning takes ~30-60 minutes for p5.48xlarge (Slurm, Docker/Enroot, EFA, NCCL setup)

## Connecting to Clusters

### SSM Session (Interactive)

HyperPod uses a special SSM target format. Standard `aws ssm start-session --target i-xxxxx` does NOT work.

```bash
REGION=us-west-2
CLUSTER=echojepa-h100-march

CLUSTER_ARN="$(aws sagemaker describe-cluster --cluster-name "$CLUSTER" --region "$REGION" --query ClusterArn --output text)"
CLUSTER_ID="${CLUSTER_ARN##*/}"

# Controller
CTRL_ID="$(aws sagemaker list-cluster-nodes --cluster-name "$CLUSTER" --region "$REGION" \
  --query "sort_by(ClusterNodeSummaries[?InstanceGroupName=='echojepa-h100-controller'], &LaunchTime)[-1].InstanceId" --output text)"

aws ssm start-session --region "$REGION" \
  --target "sagemaker-cluster:${CLUSTER_ID}_echojepa-h100-controller-${CTRL_ID}"
```

**Requires**: SSM Session Manager plugin installed locally. On SageMaker notebook instances:
```bash
curl -sL "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o /tmp/session-manager-plugin.deb
sudo dpkg -i /tmp/session-manager-plugin.deb
```

### SSH from Controller to Compute Nodes

SSH keys are NOT configured by default on HyperPod. Use Slurm's `srun` instead:
```bash
sudo su - ubuntu
srun -N1 -w ip-10-0-50-184 --ntasks=1 --pty bash   # interactive shell
srun -N1 -w ip-10-0-50-184 --ntasks=1 bash -c "hostname && nvidia-smi"  # one-off command
```

### Network Topology

- **SageMaker notebook instance**: Default VPC (172.31.x.x) -- cannot SSH to cluster
- **HyperPod cluster**: Custom VPC `vpc-0a306d982844ee4e9` (10.0.50.0/24)
- **Cross-VPC access**: Not possible without VPC peering or bastion. Use SSM from notebook or local machine.

## Environment Deployment

### The conda-pack Approach

Since compute nodes have no shared filesystem with the notebook instance, we deploy a portable conda environment via S3:

1. **Create conda env** on source machine (A100 notebook instance)
2. **Pack with conda-pack**: `conda pack -n vjepa2-312 -o vjepa2-312.tar.gz` (~4.5 GB)
3. **Upload to S3**: `aws s3 cp vjepa2-312.tar.gz s3://<bucket>/setup/`
4. **Unpack on compute node**: `tar xzf vjepa2-312.tar.gz -C /opt/vjepa2-312 && source /opt/vjepa2-312/bin/activate && conda-unpack`

### Source Code Deployment

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

### Lifecycle Script Approach

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

### S3 Bucket Layout (echojepa-h100-march)

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
    vjepa-2.1-l-h100/                   # S3 checkpoint uploads (every 5 epochs)
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
- The `LD_LIBRARY_PATH` override is **required** to avoid the cuBLAS 12.9 bf16 bug (see Issue 9 below)
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
#SBATCH --nodelist=ip-10-0-50-184

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

### Installing Claude Code on Compute Nodes

```bash
# From controller, get a shell on compute node
srun -N1 -w ip-10-0-50-184 --ntasks=1 --pty bash

# Install Node.js and Claude Code
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
claude
```

## Cluster Management Commands

```bash
# List clusters
aws sagemaker list-clusters --region us-west-2

# List nodes
aws sagemaker list-cluster-nodes --cluster-name echojepa-h100-march --region us-west-2

# Replace nodes (triggers re-provisioning with current lifecycle script)
aws sagemaker batch-replace-cluster-nodes --cluster-name echojepa-h100-march --region us-west-2 \
  --node-ids i-xxxxx

# Check lifecycle logs (CloudWatch)
aws logs get-log-events --region us-west-2 \
  --log-group-name "/aws/sagemaker/Clusters/echojepa-h100-march/yyepvbne5vzr" \
  --log-stream-name "LifecycleConfig/echojepa-h100-compute/i-xxxxx" \
  --limit 50 --query 'events[*].message' --output text
```

## Troubleshooting

### Issues Encountered During echojepa-h100-march Setup (2026-03-26)

#### 1. SSM "TargetNotConnected" / "InvalidInstanceId"

**Symptom**: `aws ssm start-session --target i-xxxxx` fails with TargetNotConnected.

**Root cause**: HyperPod uses a special SSM target format, not plain instance IDs. The correct format is:
```
sagemaker-cluster:{CLUSTER_ID}_{INSTANCE_GROUP_NAME}-{INSTANCE_ID}
```

**Also required**: The IAM execution role needs both:
- `AmazonSSMManagedInstanceCore` policy attached
- `ec2.amazonaws.com` in the trust policy

The SSM agent comes pre-installed on HyperPod AMI but cannot register without proper IAM. `aws ssm describe-instance-information` will show empty results even though the agent is running.

#### 2. SSH from Notebook Instance Fails

**Symptom**: SSH to cluster IPs (10.0.50.x) times out.

**Root cause**: Notebook instance is in a different VPC (172.31.x.x default VPC) than the cluster (10.0.50.0/24 in vpc-0a306d982844ee4e9). No network path exists.

**Solution**: Use SSM from the notebook (with session manager plugin) or from a local machine. On the controller, use `srun` instead of `ssh` to reach compute nodes.

#### 3. SSH from Controller to Compute Fails (Permission Denied)

**Symptom**: `ssh ip-10-0-50-184` returns "Permission denied (publickey)".

**Solution**: Use Slurm's `srun` instead:
```bash
srun -N1 -w ip-10-0-50-184 --ntasks=1 bash -c "command here"
```

#### 4. Lifecycle Script Timing Issue

**Symptom**: Nodes provision but don't have the expected environment.

**Root cause**: `batch-replace-cluster-nodes` triggers re-provisioning. The node downloads `on_create.sh` from S3 at provision time. If the S3 upload happens AFTER the node starts provisioning, it gets the old script.

**Solution**: Always upload the lifecycle script to S3 BEFORE calling `batch-replace-cluster-nodes`. Verify upload timestamp vs node launch time.

#### 5. Training Crashes with FileNotFoundError

**Symptom**: `FileNotFoundError: '/opt/vjepa2/data/csv/mimic_annotations_s3.csv'`

**Root cause**: The source tarball excluded CSVs to keep size manageable (the full `data/csv/` directory is 20+ GB). The training CSV must be deployed separately.

**Solution**: Upload the specific CSV to S3 and copy to the node:
```bash
srun -N1 -w ip-10-0-50-184 --ntasks=1 bash -c \
  "aws s3 cp s3://<bucket>/setup/mimic_annotations_s3.csv /opt/vjepa2/data/csv/"
```

#### 6. VPC Endpoint Security Group Blocking SSM

**Symptom**: SSM agent running but can't register. VPC endpoints exist but aren't reachable.

**Root cause**: The VPC endpoint security group (`sg-0c1b3b9f78325dc0c`) only allowed self-referencing ingress, not traffic from the cluster's security group.

**Fix**: Add inbound rule allowing TCP 443 from the cluster SG:
```bash
aws ec2 authorize-security-group-ingress --group-id sg-0c1b3b9f78325dc0c \
  --protocol tcp --port 443 --source-group sg-0a98dd2539c5bb0e9
```

#### 7. /dev/shm Exhaustion

**Symptom**: OOM or DataLoader errors after crashes.

**Root cause**: PyTorch DataLoader workers use shared memory (`/dev/shm`) for IPC. Ungraceful kills leave orphaned segments that accumulate across restarts.

**Fix**: Always clean before launching: `rm -rf /dev/shm/* 2>/dev/null || true`

#### 8. 403 Errors on S3 Video Data

**Symptom**: AccessDenied when loading video data from S3 during training.

**Fix**: Remove stale credentials: `rm -rf ~/.aws/credentials`

#### 9. CUBLAS_STATUS_INVALID_VALUE with bf16 on H100

**Symptom**: Training crashes on the first backward pass with:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx(..., CUDA_R_16BF, ...)`
```
Even a trivial 64×1024 bf16 matmul backward reproduces it.

**Root cause**: The HyperPod AMI ships CUDA 12.6/12.8/12.9/13.0. `LD_LIBRARY_PATH` puts `/usr/local/cuda-12.9/lib` first, so PyTorch (compiled with CUDA 12.8) loads the system's cuBLAS 12.9.1.4 instead of its bundled cuBLAS. This specific cuBLAS version has a bf16 GemmEx bug on H100 (sm_90).

**Diagnosis**:
```bash
# Check which cuBLAS PyTorch actually loads (should show the conda env path, NOT /usr/local/cuda-12.9/)
srun -w <node> --gpus-per-node=1 bash -c '
  source /opt/vjepa2-312/bin/activate
  python -c "
import torch, os; torch.randn(2,2).cuda()
for line in open(f\"/proc/{os.getpid()}/maps\").read().split(chr(10)):
    if \"cublas\" in line and \".so\" in line: print(line.split(\"/\",1)[-1]); break
"'
```

**Fix**: Prepend PyTorch's bundled NVIDIA libraries to `LD_LIBRARY_PATH` in the sbatch script (see `scripts/vjepa2_pretrain_h100.sbatch`):
```bash
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:...:$LD_LIBRARY_PATH"
```

**Verification** (should print "PASSED"):
```bash
srun -w <node> --gpus-per-node=1 bash -c '
  source /opt/vjepa2-312/bin/activate
  export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
  python -c "
import torch
a = torch.randn(200704, 1024, device=\"cuda\", dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(1024, 4096, device=\"cuda\", dtype=torch.bfloat16)
torch.matmul(a, b).sum().backward()
print(\"PASSED\")
"'
```

#### 10. CUDA_VISIBLE_DEVICES Override in train.py

**Symptom**: All 8 ranks initialize but crash during first backward pass. Only 1 GPU is actually used.

**Root cause**: `app/vjepa_2_1/train.py` top-level: `os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]`. With `--ntasks-per-node=1`, `SLURM_LOCALID=0` for all spawned processes, overriding `main.py`'s per-rank GPU assignment and forcing all 8 ranks onto GPU 0.

**Fix**: Commented out in train.py. `main.py` handles per-rank `CUDA_VISIBLE_DEVICES` correctly.

#### 11. GradScaler with bfloat16

**Symptom**: Not a direct crash cause, but the A100 checkpoint carries a GradScaler scale factor of 2^33 (~8.6 billion), which amplifies gradients unnecessarily during backward.

**Root cause**: The training code enables GradScaler for all mixed-precision modes including bf16. Unlike fp16 (5-bit exponent), bf16 has the same 8-bit exponent as fp32, so loss scaling is unnecessary.

**Fix**: Disabled GradScaler when `dtype == torch.bfloat16` in train.py. Changed scaler usage checks from `if mixed_precision:` to `if scaler is not None:`.

## V-JEPA 2.1 ViT-L Training Status (2026-03-26)

- **A100 run** (SageMaker notebook, 8x A100 80GB): Epoch 118/240, ~8s/iter
- **H100 run** (HyperPod echojepa-h100-march): **Active**, resumed from epoch 118 checkpoint, ~3.4s/iter GPU time, ~4.1s wall, 27 GB VRAM per GPU
- Checkpoints saved every 5 epochs to S3
- S3 checkpoint mirror: `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/vjepa-2.1-l-h100/`
- Config: `configs/train/vitl16/pretrain-21-mimic-224px-16f-h100.yaml`
- Known collapse risk: V-JEPA 2.1 context loss collapses on small repetitive datasets (observed at epochs 153/176/182 on ViT-B). Monitor for sudden loss drops

### H100 Environment (March 2026)

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 580.126.09 |
| System CUDA | 13.0, 12.9, 12.8, 12.6 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12 (conda env at /opt/vjepa2-312) |
| Instance | ml.p5.48xlarge (8x H100 80GB HBM3, sm_90) |
