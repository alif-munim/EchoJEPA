# HyperPod Cluster Operations Guide

Operational guide for SageMaker HyperPod GPU clusters used for EchoJEPA training. Covers cluster provisioning (step-by-step), connectivity, environment deployment, and job submission. Distilled from echojepa-h100-march (2026-03-26) and echojepa-h100-neurips (2026-04-12) setups. Includes 15 troubleshooting issues with fixes.

## Cluster Inventory

| Cluster | ID | Instance Type | GPUs | Training Plan | Status |
|---------|------|--------------|------|---------------|--------|
| echojepa-h100-neurips | n9we8xfqjv3p | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NeurIPS | InService |
| echojepa-v10 | swcxoboj2tln | ml.p5e.48xlarge | 8x H200 | EchoJEPA | InService |
| echojepa-h200 | 2paq9e2d06dk | ml.p5e.48xlarge | 8x H200 | EchoJEPA-H200 | InService |
| echojepa-h100-march | yyepvbne5vzr | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NeurIPS | InService (compute scaled to 0) |
| echojepa-h100-nmed | 8186qucd36mr | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NMED-H100-V1 | Failed |

## Training Plans

| Plan Name | Instance Type | Instances | AZ | Start | End | Duration |
|-----------|--------------|-----------|-----|-------|-----|----------|
| EchoJEPA-NeurIPS | ml.p5.48xlarge | 1 | us-west-2c | 2026-04-12 | 2026-05-02 | 480h (two blocks: 144h + 336h) |
| EchoJEPA-NMED-H100-V1 | ml.p5.48xlarge | 1 | us-west-2c | (expired) | (expired) | — |
| EchoJEPA-H200 | ml.p5e.48xlarge | 1 | — | — | — | — |
| EchoJEPA | ml.p5e.48xlarge | 1 | — | — | — | — |

**EchoJEPA-NeurIPS** has two reserved capacity blocks:
- Block 1 (Active): Apr 12 – Apr 18 (144h), ARN `reserved-capacity/9xnkd1mgivt4a00c34i66qamg`
- Block 2 (Scheduled): Apr 18 – May 2 (336h), ARN `reserved-capacity/dtgufn3txwmx24e53tw4ppmnr`

## Creating a New Cluster (Step-by-Step)

Complete procedure for provisioning a new HyperPod cluster. Distilled from the echojepa-h100-neurips setup (2026-04-12), which required 4 attempts due to `provisioning_parameters.json` mismatches (Issue 13) and a missing `hotfix/` directory (Issue 15).

### Step 0: Prerequisites

1. **Service quota**: Verify quota for the desired instance type (e.g., `ml.p5.48xlarge`, `ml.p5e.48xlarge`). Check at: SageMaker console → Service quotas
2. **Training plan**: Create or identify an active training plan under SageMaker → Training Plans. Note the **Availability Zone** — the cluster subnet must match
3. **Network**: Ensure a **private subnet** exists in the training plan's AZ within VPC `vpc-0a306d982844ee4e9`. The subnet must be associated with the Route Table (`rtb-0dc170...`) that has the S3 Gateway Endpoint. VPC Interface Endpoints must exist for `ssm`, `ssmmessages`, and `ec2messages`

### Step 1: Choose Names

Pick a cluster name and derive instance group names. These names must be consistent across the API call AND `provisioning_parameters.json` — mismatches cause immediate provisioning failure (see Issue 13 below).

```bash
CLUSTER_NAME="echojepa-h100-neurips"
CONTROLLER_GROUP="echojepa-neurips-controller"
COMPUTE_GROUP="echojepa-neurips-compute"
PARTITION_NAME="ml-p5-48xlarge"   # must match instance type: ml.p5.48xlarge → ml-p5-48xlarge
```

### Step 2: Create S3 Bucket and Upload Lifecycle Files

Each cluster gets its own S3 bucket for lifecycle scripts. You can copy the base lifecycle scripts from an existing cluster and share heavy setup artifacts (conda env, source code) across buckets.

```bash
BUCKET="sagemaker-${CLUSTER_NAME}-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')-bucket"
aws s3 mb "s3://${BUCKET}" --region us-west-2

# Copy base lifecycle scripts from an existing cluster bucket
SOURCE_BUCKET="sagemaker-echojepa-h100-march-0d224785-bucket"
aws s3 sync "s3://${SOURCE_BUCKET}/" "s3://${BUCKET}/" \
  --exclude "setup/*" --exclude "checkpoints/*" --exclude "gpu_tests/*"
# NOTE: Do NOT exclude hotfix/ — apply_hotfix.sh globs hotfix/*.sh and fails if the dir is missing
```

### Step 3: Upload provisioning_parameters.json (CRITICAL)

**This is the #1 cause of provisioning failures.** HyperPod validates that ALL instance group names in the `create-cluster` API call appear in this file. Missing groups → immediate rollback.

```bash
cat << EOF | aws s3 cp - "s3://${BUCKET}/provisioning_parameters.json"
{
    "workload_manager": "slurm",
    "controller_group": "${CONTROLLER_GROUP}",
    "worker_groups": [
        {
            "instance_group_name": "${COMPUTE_GROUP}",
            "partition_name": "${PARTITION_NAME}"
        }
    ],
    "login_group": ""
}
EOF
```

**Validation checklist before proceeding:**
- [ ] `controller_group` matches the controller instance group name exactly
- [ ] `worker_groups[*].instance_group_name` matches each compute instance group name exactly
- [ ] `partition_name` follows the `ml-<family>-<size>` convention (e.g., `ml-p5-48xlarge`)
- [ ] No stale group names from a previous cluster (common when copying from another bucket)

### Step 4: Upload on_create.sh

Customize the lifecycle script for your cluster. The script runs on every node during provisioning. Key considerations:
- Use `command -v nvidia-smi` to detect compute vs controller nodes
- Reference setup artifacts from a shared bucket to avoid duplicating large files
- Wrap compute-node setup in `( set +e; ... ) &` to prevent failures from killing provisioning
- Restart the SSM agent at the end so the controller is reachable via SSM immediately
- Do NOT auto-launch training — use `deploy.sh` + `sbatch` after provisioning is complete

```bash
aws s3 cp /path/to/on_create.sh "s3://${BUCKET}/on_create.sh"
```

### Step 5: Create the Cluster

```bash
TRAINING_PLAN_ARN="arn:aws:sagemaker:us-west-2:495467399120:training-plan/<YOUR_PLAN>"
EXECUTION_ROLE="arn:aws:iam::495467399120:role/service-role/AmazonSageMaker-ExecutionRole-20250409T120880"
SECURITY_GROUP="sg-0a98dd2539c5bb0e9"
SUBNET="subnet-0c98c74de56238192"   # must be in the training plan's AZ

aws sagemaker create-cluster \
  --cluster-name "${CLUSTER_NAME}" \
  --region us-west-2 \
  --instance-groups "[
    {
      \"InstanceGroupName\": \"${COMPUTE_GROUP}\",
      \"InstanceType\": \"ml.p5.48xlarge\",
      \"InstanceCount\": 1,
      \"LifeCycleConfig\": {
        \"SourceS3Uri\": \"s3://${BUCKET}\",
        \"OnCreate\": \"on_create.sh\"
      },
      \"ExecutionRole\": \"${EXECUTION_ROLE}\",
      \"ThreadsPerCore\": 1,
      \"InstanceStorageConfigs\": [{\"EbsVolumeConfig\": {\"VolumeSizeInGB\": 500}}],
      \"TrainingPlanArn\": \"${TRAINING_PLAN_ARN}\",
      \"OverrideVpcConfig\": {
        \"SecurityGroupIds\": [\"${SECURITY_GROUP}\"],
        \"Subnets\": [\"${SUBNET}\"]
      }
    },
    {
      \"InstanceGroupName\": \"${CONTROLLER_GROUP}\",
      \"InstanceType\": \"ml.m5.2xlarge\",
      \"InstanceCount\": 1,
      \"LifeCycleConfig\": {
        \"SourceS3Uri\": \"s3://${BUCKET}\",
        \"OnCreate\": \"on_create.sh\"
      },
      \"ExecutionRole\": \"${EXECUTION_ROLE}\",
      \"ThreadsPerCore\": 1,
      \"InstanceStorageConfigs\": [{\"EbsVolumeConfig\": {\"VolumeSizeInGB\": 500}}],
      \"OverrideVpcConfig\": {
        \"SecurityGroupIds\": [\"${SECURITY_GROUP}\"],
        \"Subnets\": [\"${SUBNET}\"]
      }
    }
  ]" \
  --vpc-config "{
    \"SecurityGroupIds\": [\"${SECURITY_GROUP}\"],
    \"Subnets\": [\"${SUBNET}\"]
  }" \
  --node-recovery "Automatic"
```

**Notes:**
- Only the compute group gets `TrainingPlanArn` — the controller does not
- `ThreadsPerCore: 1` disables hyperthreading (standard for GPU workloads)
- 500GB EBS gives headroom for checkpoints, conda env, and data

### Step 6: Monitor Provisioning

Provisioning takes **30-60 minutes** for `ml.p5.48xlarge`. Monitor status:

```bash
# Quick status check
aws sagemaker describe-cluster --cluster-name "${CLUSTER_NAME}" --region us-west-2 \
  --query '{Status:ClusterStatus,FailureMessage:FailureMessage}' --output json

# Node-level detail
aws sagemaker list-cluster-nodes --cluster-name "${CLUSTER_NAME}" --region us-west-2

# Poll until done (run in background)
while true; do
  STATUS=$(aws sagemaker describe-cluster --cluster-name "${CLUSTER_NAME}" \
    --region us-west-2 --query ClusterStatus --output text)
  echo "$(date +%H:%M:%S) - $STATUS"
  [ "$STATUS" != "Creating" ] && break
  sleep 300
done
```

**If provisioning fails** (status: `RollingBack` → `Failed`):
1. Check the `FailureMessage` — it usually identifies the exact issue
2. Fix the root cause (most commonly `provisioning_parameters.json`)
3. Delete the failed cluster: `aws sagemaker delete-cluster --cluster-name "${CLUSTER_NAME}" --region us-west-2`
4. Rollbacks for `ml.p5.48xlarge` can take **30-90 minutes** — you must wait for `Failed` status before deleting
5. Recreate with the same command (the S3 bucket and lifecycle files persist)

### Step 7: Post-Provisioning Setup

Once status is `InService`:

```bash
# 1. Connect to the controller via SSM
CLUSTER_ID="$(aws sagemaker describe-cluster --cluster-name "${CLUSTER_NAME}" \
  --region us-west-2 --query ClusterArn --output text | rev | cut -d/ -f1 | rev)"
CTRL_ID="$(aws sagemaker list-cluster-nodes --cluster-name "${CLUSTER_NAME}" --region us-west-2 \
  --query "sort_by(ClusterNodeSummaries[?InstanceGroupName=='${CONTROLLER_GROUP}'], &LaunchTime)[-1].InstanceId" --output text)"
aws ssm start-session --region us-west-2 \
  --target "sagemaker-cluster:${CLUSTER_ID}_${CONTROLLER_GROUP}-${CTRL_ID}"

# 2. On the controller: clone the repo and set up deploy.sh
sudo su - ubuntu
git clone git@github.com:<org>/EchoJEPA.git ~/EchoJEPA-repo

# 3. Create deploy.sh for code deployment to compute nodes
#    (see "Code Deployment" section below)

# 4. Deploy code and verify compute node environment
~/deploy.sh
srun -N1 --ntasks=1 bash -c "ls /opt/vjepa2 && nvidia-smi"
```

### Reusable Config Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| VPC | `vpc-0a306d982844ee4e9` | 10.0.50.0/24 |
| Subnet (us-west-2c) | `subnet-0c98c74de56238192` | Private, S3 gateway endpoint via route table |
| Security Group | `sg-0a98dd2539c5bb0e9` | Cluster SG, allows EFA + VPC endpoint traffic |
| VPC Endpoint SG | `sg-0c1b3b9f78325dc0c` | Must allow TCP 443 from cluster SG |
| IAM Role | `AmazonSageMaker-ExecutionRole-20250409T120880` | SSM + ec2 trust policy |
| Conda Env | `/opt/vjepa2-312` | Python 3.12, PyTorch 2.10.0+cu128 |
| Source Code | `/opt/vjepa2` | Deployed via `deploy.sh` from controller |
| Setup Artifacts Bucket | `sagemaker-echojepa-h100-march-0d224785-bucket/setup/` | Shared conda env + source tarball |

## Connecting to Clusters

### SSM Session (Interactive)

HyperPod uses a special SSM target format. Standard `aws ssm start-session --target i-xxxxx` does NOT work.

```bash
REGION=us-west-2
CLUSTER=echojepa-h100-neurips   # or echojepa-h100-march
INSTANCE_GROUP_PREFIX=echojepa-neurips   # or echojepa-h100 for the march cluster

CLUSTER_ARN="$(aws sagemaker describe-cluster --cluster-name "$CLUSTER" --region "$REGION" --query ClusterArn --output text)"
CLUSTER_ID="${CLUSTER_ARN##*/}"

# Controller
CTRL_ID="$(aws sagemaker list-cluster-nodes --cluster-name "$CLUSTER" --region "$REGION" \
  --query "sort_by(ClusterNodeSummaries[?InstanceGroupName=='${INSTANCE_GROUP_PREFIX}-controller'], &LaunchTime)[-1].InstanceId" --output text)"

aws ssm start-session --region "$REGION" \
  --target "sagemaker-cluster:${CLUSTER_ID}_${INSTANCE_GROUP_PREFIX}-controller-${CTRL_ID}"
```

**Instance group names by cluster:**
| Cluster | Controller Group | Compute Group |
|---------|-----------------|---------------|
| echojepa-h100-neurips | echojepa-neurips-controller | echojepa-neurips-compute |
| echojepa-h100-march | echojepa-h100-controller | echojepa-h100-compute |

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

### Code Deployment (Default Workflow)

The git repo is cloned on the **controller** at `~/EchoJEPA-repo`. Use `scripts/deploy.sh` to push code to compute nodes:

```bash
# On the controller:
cd ~/EchoJEPA-repo
git pull                        # get latest from GitHub
~/deploy.sh                     # deploy to ip-10-0-50-184
~/deploy.sh ip-10-0-50-83      # deploy to other node (optional)
sbatch ~/vjepa2_pretrain_h100.sbatch
```

The deploy script tars the repo (excluding .git, checkpoints, large data) and unpacks it on the compute node via srun. This is the standard workflow because:
- Controller and compute nodes have **no shared filesystem**
- Compute nodes are in a private subnet with **no GitHub access**
- The controller has SSH access to GitHub for git pull/push

After editing code on the controller (e.g. via Claude Code), always run `~/deploy.sh` before launching training.

### The conda-pack Approach (One-Time Setup)

The conda environment (`/opt/vjepa2-312`) is deployed separately since it rarely changes:

1. **Create conda env** on source machine (A100 notebook instance)
2. **Pack with conda-pack**: `conda pack -n vjepa2-312 -o vjepa2-312.tar.gz` (~4.5 GB)
3. **Upload to S3**: `aws s3 cp vjepa2-312.tar.gz s3://<bucket>/setup/`
4. **Unpack on compute node**: `tar xzf vjepa2-312.tar.gz -C /opt/vjepa2-312 && source /opt/vjepa2-312/bin/activate && conda-unpack`

### Legacy: Source Code via S3

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

### S3 Bucket Layout (echojepa-h100-neurips)

```
s3://sagemaker-echojepa-h100-neurips-f85ad7df-bucket/
  on_create.sh                          # lifecycle script (references h100-march bucket for setup artifacts)
  lifecycle_script.py                   # standard HyperPod provisioning
  provisioning_parameters.json          # cluster provisioning params (MUST list all instance groups)
  apply_hotfix.sh                       # hotfix runner (globs hotfix/*.sh)
  hotfix/                               # REQUIRED — apply_hotfix.sh fails if missing
    hold-lustre-client.sh
    mock-gpu-driver-deb.sh
  utils/                                # Slurm utilities, enroot, keypair scripts
  observability/                        # DCGM/NCCL/OTel metric exporters
  multi_headnode_setup/                 # multi-controller scripts
  setup/
    setup_done_<hostname>.txt           # completion markers (written by on_create.sh)
```

The `on_create.sh` pulls heavy setup artifacts (conda env, source code) from the echojepa-h100-march bucket's `setup/` directory to avoid duplicating ~10 GB of data. It does NOT auto-launch training — use `~/deploy.sh` + `sbatch` after provisioning.

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
    vjepa-2.1-l-h100/                   # V-JEPA 2.1 ViT-L checkpoints
    byol-vitl-imagenet/                 # BYOL v1 (stopped — EMA plateau)
    byol-vitl-imagenet-v2/              # BYOL v2 (active — constant EMA)
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

# List nodes (substitute cluster name as needed)
aws sagemaker list-cluster-nodes --cluster-name echojepa-h100-neurips --region us-west-2

# Replace nodes (triggers re-provisioning with current lifecycle script)
aws sagemaker batch-replace-cluster-nodes --cluster-name echojepa-h100-neurips --region us-west-2 \
  --node-ids i-xxxxx

# Check lifecycle logs (CloudWatch)
# echojepa-h100-neurips: cluster ID n9we8xfqjv3p, compute group echojepa-neurips-compute
aws logs get-log-events --region us-west-2 \
  --log-group-name "/aws/sagemaker/Clusters/echojepa-h100-neurips/n9we8xfqjv3p" \
  --log-stream-name "LifecycleConfig/echojepa-neurips-compute/i-xxxxx" \
  --limit 50 --query 'events[*].message' --output text

# echojepa-h100-march: cluster ID yyepvbne5vzr, compute group echojepa-h100-compute
aws logs get-log-events --region us-west-2 \
  --log-group-name "/aws/sagemaker/Clusters/echojepa-h100-march/yyepvbne5vzr" \
  --log-stream-name "LifecycleConfig/echojepa-h100-compute/i-xxxxx" \
  --limit 50 --query 'events[*].message' --output text
```

## Troubleshooting

### Issues Encountered During Cluster Setup (2026-03 – 2026-04)

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

#### 12. Activation Checkpointing Required at Batch Size 128 (2026-03-26)

**Symptom**: OOM crash when running with `use_activation_checkpointing: false` on H100 80GB.

**Details**: Tested on ip-10-0-50-83 (job 70, 15-minute test run). Without activation checkpointing, the model uses ~79.1 GB out of 79.2 GB available, crashing with `torch.OutOfMemoryError` in the predictor forward pass (`predictor.py:257`, `torch.stack` over argsort indices). With checkpointing enabled, VRAM usage is ~27.8 GB — a 2.8x reduction.

**Conclusion**: Activation checkpointing is **mandatory** for ViT-L at batch size 128 on 80GB GPUs. To disable it, batch size would need to be reduced significantly (~40-50% of current), which would reduce throughput more than checkpointing's ~30% recompute overhead. The current config (checkpointing ON, batch 128) is optimal for H100 80GB.

**Alternatives not yet tested**:
- Reducing batch size (e.g., 64) without checkpointing — may not be faster due to lower GPU utilization
- Gradient accumulation with smaller micro-batch — adds complexity
- Multi-node training (2 nodes × 8 GPUs) — would allow larger effective batch or lower per-GPU memory

#### 13. provisioning_parameters.json Group Name Mismatch (2026-04-12)

**Symptom**: Cluster immediately enters `RollingBack` with:
```
The instance group names provided by the request do not match the instance group names
provided by the provisioning_parameters.json file from the lifecycle scripts S3 bucket.
```

**Root cause**: HyperPod validates that EVERY instance group name in the `create-cluster` API call has a matching entry in `provisioning_parameters.json`. This fails when:
1. The `controller_group` field doesn't match the controller instance group name
2. The `worker_groups` array is missing or doesn't include all compute instance group names
3. You copied `provisioning_parameters.json` from another cluster's bucket without updating the group names

**Example of a WRONG file** (missing `worker_groups`):
```json
{
    "workload_manager": "slurm",
    "controller_group": "echojepa-neurips-controller",
    "login_group": ""
}
```

**Example of a CORRECT file:**
```json
{
    "workload_manager": "slurm",
    "controller_group": "echojepa-neurips-controller",
    "worker_groups": [
        {
            "instance_group_name": "echojepa-neurips-compute",
            "partition_name": "ml-p5-48xlarge"
        }
    ],
    "login_group": ""
}
```

**Prevention**: See Step 3 in "Creating a New Cluster" — always validate the file before `create-cluster`. The validation checklist catches this.

**Recovery**: Rollbacks for `ml.p5.48xlarge` take 30-90 minutes. You must wait for `Failed` status before deleting and recreating. Fix the JSON in S3, delete the failed cluster, then recreate with the same command.

#### 14. Slow Rollbacks for p5 Instances (2026-04-12)

**Symptom**: Cluster stuck in `RollingBack` for 30-90 minutes after a provisioning failure. Compute nodes show `ShuttingDown` for the entire duration.

**Root cause**: Normal behavior for `ml.p5.48xlarge` (and likely `ml.p5e.48xlarge`). The p5 instance deprovisioning involves EFA teardown, NCCL cleanup, and EBS volume detachment, all of which take time.

**What to do**: There is no way to accelerate this. Just poll and wait:
```bash
while true; do
  STATUS=$(aws sagemaker describe-cluster --cluster-name <name> --region us-west-2 --query ClusterStatus --output text)
  echo "$(date +%H:%M:%S) - $STATUS"
  [ "$STATUS" = "Failed" ] && break
  sleep 300
done
```

You cannot delete a cluster in `RollingBack` state. You cannot create a new cluster with the same name while the old one exists. Plan for up to 90 minutes of dead time per failed attempt — which is why getting `provisioning_parameters.json` right on the first try matters.

#### 15. Lifecycle Script Fails: hotfix/*.sh Glob Not Found (2026-04-12)

**Symptom**: Controller lifecycle script fails with exit code 127:
```
bash: /tmp/.../hotfix/*.sh: No such file or directory
subprocess.CalledProcessError: Command '['sudo', 'bash', './apply_hotfix.sh', ...]' returned non-zero exit status 127.
```

**Root cause**: `apply_hotfix.sh` runs `for i in $BIN_DIR/hotfix/*.sh; do bash -x "$i"; done`. If the `hotfix/` directory is missing from the S3 bucket (e.g., excluded during `aws s3 sync`), the glob doesn't expand and bash tries to execute the literal string `hotfix/*.sh`, which fails.

**Fix**: Copy the `hotfix/` directory from the source bucket:
```bash
aws s3 sync s3://<source-bucket>/hotfix/ s3://<new-bucket>/hotfix/
```

**Prevention**: When syncing lifecycle files to a new bucket, do NOT exclude `hotfix/`. The directory is small (~1.5 KB) and required by the standard HyperPod lifecycle scripts.

## Remote Command Execution via SSM (Non-Interactive)

### The Problem

`aws ssm start-session` requires a PTY (pseudo-terminal). From environments without a TTY — SageMaker notebook instances, CI/CD pipelines, Claude Code on a remote machine — the session opens but immediately exits with `Cannot perform start session: EOF`.

Standard alternatives (`ssm send-command`, `describe-instance-information`) don't work with HyperPod because it uses a custom SSM target format (`sagemaker-cluster:...`) that only `start-session` supports.

### The Solution: `script` + Non-Interactive SSM Document

Wrap the SSM call in `script -q -c '...' /dev/null` to allocate a PTY, and use the `AWS-StartNonInteractiveCommand` document to run a command and exit cleanly:

```bash
CLUSTER_ID=n9we8xfqjv3p
TARGET="sagemaker-cluster:${CLUSTER_ID}_echojepa-neurips-controller-i-0415ce8f417564270"

# Basic pattern
script -q -c 'timeout 20 aws ssm start-session --region us-west-2 \
  --target "'$TARGET'" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters "{\"command\":[\"<your command here>\"]}"' /dev/null
```

**Key details:**
- `script -q -c '...' /dev/null` — allocates a PTY without creating a typescript file
- `timeout 20` — prevents hanging if the session stalls (adjust as needed for longer commands)
- `AWS-StartNonInteractiveCommand` — runs the command and exits (no interactive shell)
- The `command` parameter takes a **single string** — multiple commands must be joined with `&&` or `;`

### Examples

```bash
# Check Slurm status
script -q -c 'timeout 20 aws ssm start-session --region us-west-2 \
  --target "'$TARGET'" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters "{\"command\":[\"sudo -u ubuntu bash -c \\\"sinfo -N && squeue\\\"\"]}"' /dev/null

# Run nvidia-smi on compute node via srun
script -q -c 'timeout 20 aws ssm start-session --region us-west-2 \
  --target "'$TARGET'" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters "{\"command\":[\"sudo -u ubuntu srun -N1 --ntasks=1 bash -c \\\"nvidia-smi\\\"\"]}"' /dev/null

# Submit an sbatch job (write script via echo|tee, then sbatch)
script -q -c 'timeout 30 aws ssm start-session --region us-west-2 \
  --target "'$TARGET'" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters "{\"command\":[\"bash -c \\\"echo -e '"'"'#!/bin/bash\\n#SBATCH --job-name=test\\n#SBATCH --partition=ml-p5-48xlarge\\n#SBATCH --nodes=1\\n#SBATCH --gpus-per-node=8\\n#SBATCH --output=/tmp/test-%j.out\\n#SBATCH --time=0:05:00\\nnvidia-smi\\necho PASSED'"'"' | tee /tmp/test.sbatch && sudo -u ubuntu sbatch /tmp/test.sbatch\\\"\"]}"' /dev/null

# Read job output from compute node (no shared filesystem)
script -q -c 'timeout 20 aws ssm start-session --region us-west-2 \
  --target "'$TARGET'" \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters "{\"command\":[\"sudo -u ubuntu srun -N1 -w ip-10-0-50-241 --ntasks=1 bash -c \\\"cat /tmp/test-3.out\\\"\"]}"' /dev/null
```

### Escaping Rules

The JSON parameters go through multiple layers of escaping (outer shell → JSON → inner shell). Rules of thumb:
- Inner double quotes: `\\\"`
- Inner single quotes: use `'"'"'` (end outer single-quote, add escaped single-quote, restart outer single-quote)
- Newlines in strings: `\\n` (for `echo -e`)
- `%` in sbatch directives (e.g., `%j`): use `%%j` if inside a `printf`, plain `%j` inside `echo -e`
- When in doubt, break multi-step operations into separate SSM calls rather than fighting the escaping

### Limitations

- **No streaming output**: the full output appears only after the command completes
- **Timeout required**: without `timeout`, a hung command blocks forever
- **Single command string**: the `command` parameter takes exactly one value, not an array
- **Output includes session boilerplate**: `Starting session...` and `Exiting session...` lines wrap the actual output
- **Job output on compute node**: sbatch stdout/stderr files are on the compute node, not the controller. Read them via `srun ... cat /tmp/<file>.out`

## Monitoring Running Jobs

### Claude Code Context

Claude Code can run on the **controller node** directly (via SSM interactive session) where it can use `squeue`, `sinfo`, `sacct`, and `srun` natively. From a **remote machine** (e.g., SageMaker notebook), use the non-interactive SSM pattern above. SSH from controller to compute nodes fails (`Permission denied (publickey)`), so all compute-node commands must go through `srun`.

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

# Check disk space (critical — checkpoint saves need ~5GB headroom)
srun --jobid=$JOBID --nodes=1 --nodelist=$NODE --ntasks=1 --overlap \
  bash -c "df -h /opt/vjepa2"
```

### Key Nuances

1. **`slurmstepd: error: couldn't chdir` warnings are harmless** — they appear because the controller's CWD doesn't exist on compute nodes. The command still runs from /tmp.

2. **stdout is fully buffered under srun** — `grep` on the sbatch stdout file (e.g., `/tmp/byol_2node-183.out`) may show nothing even while training is active. Python stdout is not line-buffered when piped through srun. Use the CSV log files (`log_r0.csv`) or checkpoint timestamps for reliable progress monitoring.

3. **CSV log format**: `epoch,iter,loss,total_ms,gpu_ms,unstable_flag` — the last column is 0 for normal, 1 if loss was unstable.

4. **Checkpoint saves happen at epoch boundaries** — if a job just started and you see stale checkpoint timestamps, it may just mean the first epoch hasn't completed yet. Check `log_r0.csv` to see current epoch/iter and estimate when the next save will occur (`ipe` iterations per epoch × time per iter).

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

3. **Checkpoint saves only happen on the first node** — rank 0 saves locally and uploads to S3. Other nodes never write checkpoints. This means S3 is the durable copy.

### MASTER_ADDR Bug (Fixed 2026-03-27)

`src/utils/distributed.py` previously always overwrote `MASTER_ADDR` with `localhost` or `HOSTNAME`, which breaks multi-node NCCL since each node would set its own hostname as master. Fix: only set `MASTER_ADDR` if not already configured by the sbatch script.

### Batch Size Scaling

When going from 1 node (8 GPUs) to 2 nodes (16 GPUs), halve per-GPU `batch_size` to keep effective batch constant. Example: `batch_size: 64` (single-node, effective=512) → `batch_size: 32` (2-node, effective=512). This preserves training dynamics and allows controlled speedup comparison.

**Note**: The repo config (`pretrain-byol-mimic-224px-16f-h100.yaml`) uses `batch_size: 64` for the 2-node run (64 × 16 = 1024 effective, matching V-JEPA). For single-node, use `batch_size: 128` to maintain the same effective batch.

### Performance

- Single-node (8x H100): ~7.0s/iter
- 2-node (16x H100): ~3.6s/iter (1.94× speedup)
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
- **Key settings**: EMA [0.99925, 0.99925] constant, batch 64×16=1024, 240 epochs, warmup 40
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

### V1 Run (2026-03-27, Stopped — representation degradation)

- **Problem**: Cosine EMA [0.996, 1.0] froze target encoder by epoch ~12, causing representation degradation
- **Evidence**: LVEF probe Pearson r dropped from 0.151 (e10) to 0.089 (e40) — online encoder got WORSE
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
