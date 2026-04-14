# HyperPod Troubleshooting

19 issues encountered during SageMaker HyperPod cluster setup and operation (2026-03 -- 2026-04). For cluster operations see [hyperpod-ops.md](hyperpod-ops.md). For deployment and job submission see [hyperpod-deployment.md](hyperpod-deployment.md).

## Issues Encountered During Cluster Setup (2026-03 -- 2026-04)

### 1. SSM "TargetNotConnected" / "InvalidInstanceId"

**Symptom**: `aws ssm start-session --target i-xxxxx` fails with TargetNotConnected.

**Root cause**: HyperPod uses a special SSM target format, not plain instance IDs. The correct format is:
```
sagemaker-cluster:{CLUSTER_ID}_{INSTANCE_GROUP_NAME}-{INSTANCE_ID}
```

**Also required**: The IAM execution role needs both:
- `AmazonSSMManagedInstanceCore` policy attached
- `ec2.amazonaws.com` in the trust policy

The SSM agent comes pre-installed on HyperPod AMI but cannot register without proper IAM. `aws ssm describe-instance-information` will show empty results even though the agent is running.

### 2. SSH from Notebook Instance Fails

**Symptom**: SSH to cluster IPs (10.0.50.x) times out.

**Root cause**: Notebook instance is in a different VPC (172.31.x.x default VPC) than the cluster (10.0.50.0/24 in vpc-0a306d982844ee4e9). No network path exists.

**Solution**: Use SSM from the notebook (with session manager plugin) or from a local machine. On the controller, use `srun` instead of `ssh` to reach compute nodes.

### 3. SSH from Controller to Compute Fails (Permission Denied)

**Symptom**: `ssh ip-10-0-50-241` returns "Permission denied (publickey)".

**Solution**: Use Slurm's `srun` instead:
```bash
srun -N1 -w ip-10-0-50-241 --ntasks=1 bash -c "command here"
```

### 4. Lifecycle Script Timing Issue

**Symptom**: Nodes provision but don't have the expected environment.

**Root cause**: `batch-replace-cluster-nodes` triggers re-provisioning. The node downloads `on_create.sh` from S3 at provision time. If the S3 upload happens AFTER the node starts provisioning, it gets the old script.

**Solution**: Always upload the lifecycle script to S3 BEFORE calling `batch-replace-cluster-nodes`. Verify upload timestamp vs node launch time.

### 5. Training Crashes with FileNotFoundError

**Symptom**: `FileNotFoundError: '/opt/vjepa2/data/csv/mimic_annotations_s3.csv'`

**Root cause**: The source tarball excluded CSVs to keep size manageable (the full `data/csv/` directory is 20+ GB). The training CSV must be deployed separately.

**Solution**: Upload the specific CSV to S3 and copy to the node:
```bash
srun -N1 -w ip-10-0-50-241 --ntasks=1 bash -c \
  "aws s3 cp s3://<bucket>/setup/mimic_annotations_s3.csv /opt/vjepa2/data/csv/"
```

### 6. VPC Endpoint Security Group Blocking SSM

**Symptom**: SSM agent running but can't register. VPC endpoints exist but aren't reachable.

**Root cause**: The VPC endpoint security group (`sg-0c1b3b9f78325dc0c`) only allowed self-referencing ingress, not traffic from the cluster's security group.

**Fix**: Add inbound rule allowing TCP 443 from the cluster SG:
```bash
aws ec2 authorize-security-group-ingress --group-id sg-0c1b3b9f78325dc0c \
  --protocol tcp --port 443 --source-group sg-0a98dd2539c5bb0e9
```

### 7. /dev/shm Exhaustion

**Symptom**: OOM or DataLoader errors after crashes.

**Root cause**: PyTorch DataLoader workers use shared memory (`/dev/shm`) for IPC. Ungraceful kills leave orphaned segments that accumulate across restarts.

**Fix**: Always clean before launching: `rm -rf /dev/shm/* 2>/dev/null || true`

### 8. 403 Errors on S3 Video Data

**Symptom**: AccessDenied when loading video data from S3 during training.

**Fix**: Remove stale credentials: `rm -rf ~/.aws/credentials`

### 9. CUBLAS_STATUS_INVALID_VALUE with bf16 on H100

**Symptom**: Training crashes on the first backward pass with:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx(..., CUDA_R_16BF, ...)`
```
Even a trivial 64x1024 bf16 matmul backward reproduces it.

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

### 10. CUDA_VISIBLE_DEVICES Override in train.py

**Symptom**: All 8 ranks initialize but crash during first backward pass. Only 1 GPU is actually used.

**Root cause**: `app/vjepa_2_1/train.py` top-level: `os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]`. With `--ntasks-per-node=1`, `SLURM_LOCALID=0` for all spawned processes, overriding `main.py`'s per-rank GPU assignment and forcing all 8 ranks onto GPU 0.

**Fix**: Commented out in train.py. `main.py` handles per-rank `CUDA_VISIBLE_DEVICES` correctly.

### 11. GradScaler with bfloat16

**Symptom**: Not a direct crash cause, but the A100 checkpoint carries a GradScaler scale factor of 2^33 (~8.6 billion), which amplifies gradients unnecessarily during backward.

**Root cause**: The training code enables GradScaler for all mixed-precision modes including bf16. Unlike fp16 (5-bit exponent), bf16 has the same 8-bit exponent as fp32, so loss scaling is unnecessary.

**Fix**: Disabled GradScaler when `dtype == torch.bfloat16` in train.py. Changed scaler usage checks from `if mixed_precision:` to `if scaler is not None:`.

### 12. Activation Checkpointing Required at Batch Size 128 (2026-03-26)

**Symptom**: OOM crash when running with `use_activation_checkpointing: false` on H100 80GB.

**Details**: Tested on ip-10-0-50-83 (job 70, 15-minute test run). Without activation checkpointing, the model uses ~79.1 GB out of 79.2 GB available, crashing with `torch.OutOfMemoryError` in the predictor forward pass (`predictor.py:257`, `torch.stack` over argsort indices). With checkpointing enabled, VRAM usage is ~27.8 GB -- a 2.8x reduction.

**Conclusion**: Activation checkpointing is **mandatory** for ViT-L at batch size 128 on 80GB GPUs. To disable it, batch size would need to be reduced significantly (~40-50% of current), which would reduce throughput more than checkpointing's ~30% recompute overhead. The current config (checkpointing ON, batch 128) is optimal for H100 80GB.

**Alternatives not yet tested**:
- Reducing batch size (e.g., 64) without checkpointing -- may not be faster due to lower GPU utilization
- Gradient accumulation with smaller micro-batch -- adds complexity
- Multi-node training (2 nodes x 8 GPUs) -- would allow larger effective batch or lower per-GPU memory

### 13. provisioning_parameters.json Group Name Mismatch (2026-04-12)

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

**Prevention**: See Step 3 in [hyperpod-cluster-creation.md](hyperpod-cluster-creation.md) -- always validate the file before `create-cluster`. The validation checklist catches this.

**Recovery**: Rollbacks for `ml.p5.48xlarge` take 30-90 minutes. You must wait for `Failed` status before deleting and recreating. Fix the JSON in S3, delete the failed cluster, then recreate with the same command.

### 14. Slow Rollbacks for p5 Instances (2026-04-12)

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

You cannot delete a cluster in `RollingBack` state. You cannot create a new cluster with the same name while the old one exists. Plan for up to 90 minutes of dead time per failed attempt -- which is why getting `provisioning_parameters.json` right on the first try matters.

### 15. Lifecycle Script Fails: hotfix/*.sh Glob Not Found (2026-04-12)

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

### 16. PermissionError Writing to /opt/vjepa2 (2026-04-13)

**Symptom**: Inference job exits immediately with:
```
PermissionError: [Errno 13] Permission denied: '/opt/vjepa2/evals/vitg-384'
```

**Root cause**: The sbatch config set `folder: /opt/vjepa2/evals/vitg-384/nature_medicine/uhn` as the output directory. Code at `/opt/vjepa2` is owned by `root` (deployed via `sudo tar`), but the job runs as `ubuntu`. `os.makedirs()` fails when trying to create subdirectories.

**Fix**: Set the output path to a writable location like `/tmp`:
```yaml
folder: /tmp/nmed_a4c/output   # NOT /opt/vjepa2/evals/...
```
Then sync results to S3 at the end of the job.

### 17. Scaling Compute Nodes Up/Down (2026-04-13)

**Procedure**: Use `update-cluster` with the full instance group config (all fields must be present, including `OverrideVpcConfig`):

```bash
aws sagemaker update-cluster --cluster-name echojepa-h100-neurips --region us-west-2 \
  --instance-groups '[
    {
      "InstanceGroupName":"echojepa-neurips-controller",
      "InstanceType":"ml.m5.2xlarge",
      "InstanceCount":1,
      "LifeCycleConfig":{"SourceS3Uri":"s3://sagemaker-echojepa-h100-neurips-f85ad7df-bucket","OnCreate":"on_create.sh"},
      "ExecutionRole":"arn:aws:iam::495467399120:role/service-role/AmazonSageMaker-ExecutionRole-20250409T120880",
      "ThreadsPerCore":1,
      "InstanceStorageConfigs":[{"EbsVolumeConfig":{"VolumeSizeInGB":500}}],
      "OverrideVpcConfig":{"SecurityGroupIds":["sg-0a98dd2539c5bb0e9"],"Subnets":["subnet-0c98c74de56238192"]}
    },
    {
      "InstanceGroupName":"echojepa-neurips-compute",
      "InstanceType":"ml.p5.48xlarge",
      "InstanceCount":1,
      "LifeCycleConfig":{"SourceS3Uri":"s3://sagemaker-echojepa-h100-neurips-f85ad7df-bucket","OnCreate":"on_create.sh"},
      "ExecutionRole":"arn:aws:iam::495467399120:role/service-role/AmazonSageMaker-ExecutionRole-20250409T120880",
      "ThreadsPerCore":1,
      "InstanceStorageConfigs":[{"EbsVolumeConfig":{"VolumeSizeInGB":500}}],
      "TrainingPlanArn":"arn:aws:sagemaker:us-west-2:495467399120:training-plan/EchoJEPA-NeurIPS",
      "OverrideVpcConfig":{"SecurityGroupIds":["sg-0a98dd2539c5bb0e9"],"Subnets":["subnet-0c98c74de56238192"]}
    }
  ]'
```

**Key gotchas**:
- `OverrideVpcConfig` MUST be included for each group -- omitting it causes `ValidationException: Updating fields OverrideVpcConfig on an InstanceGroup not supported`
- `ExecutionRole` is required for each group
- `TrainingPlanArn` only on compute group (not controller)
- Set `InstanceCount: 0` to scale down, `1` to scale up
- Scaling up takes 5-15 minutes; monitor with `describe-cluster`

### 18. S3 Tarball Deploy: Vendored Model Repos Bloat (2026-04-13)

**Symptom**: Code tarball is 1.5-6 GB instead of expected ~300 KB.

**Root cause**: `evals/video_classification_frozen/modelcustom/` contains vendored repos (EchoPrime 5.5 GB, EchoFM 1.2 GB) with model weights and data. A naive `tar czf ... evals/` includes everything.

**Fix**: Use a file list approach to include only Python source:
```bash
cd /path/to/vjepa2
find app/ src/ configs/ -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.txt' -o -name '*.cfg' \) ! -path '*__pycache__*' > /tmp/deploy_files.txt
find evals/ -maxdepth 1 -name '*.py' >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/eval.py" >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/models.py" >> /tmp/deploy_files.txt
echo "evals/video_classification_frozen/utils.py" >> /tmp/deploy_files.txt
find evals/video_classification_frozen/modelcustom/ -maxdepth 1 -name '*.py' >> /tmp/deploy_files.txt
find evals/video_classification_frozen/modelcustom/PanEcho/ -type f -name '*.py' >> /tmp/deploy_files.txt
echo "setup.py" >> /tmp/deploy_files.txt
echo "pyproject.toml" >> /tmp/deploy_files.txt
echo ".flake8" >> /tmp/deploy_files.txt
echo "scripts/<your_job>.sbatch" >> /tmp/deploy_files.txt
rm -f /tmp/vjepa2-src.tar.gz  # IMPORTANT: delete old file first, tar may append
tar czf /tmp/vjepa2-src.tar.gz -T /tmp/deploy_files.txt
```
Result: ~280 KB instead of 5 GB. Always `rm -f` the old tarball before creating a new one -- tar may silently append to existing files.

### 19. Root filesystem full -- sbatch jobs fail silently with exit code 1:0

**Symptom**: All sbatch jobs fail immediately (1 second) with exit code `1:0`. Both stdout and stderr output files are 0 bytes. Even a minimal script (`echo HELLO`) produces no output. `srun` commands work fine.

**Root cause**: The compute node root filesystem (`/dev/root`, 97G) is 100% full. Bash cannot start when `/tmp` (used for shell temp files) has no space. Job 36's A4C ablation cached ~27G of model checkpoints and outputs under `/tmp/nmed_a4c/` (16G encoder, 8.8G probes, 2.3G outputs), consuming all available space.

**Diagnosis**:
```bash
# slurmd log shows the exit but no error message:
[54.batch] stepd_cleanup: done with step (rc[0x100]:Unknown error 256)

# The actual problem:
df -h /
# /dev/root  97G  93G  0  100%  /

du -sh /tmp/*  # shows the large cache directory
```

**Fix**: Clean up the cached files and use NVMe ephemeral storage for large downloads:
```bash
rm -rf /tmp/nmed_a4c  # or whatever is consuming space
df -h /  # verify space recovered

# In sbatch scripts, use NVMe instead of /tmp:
LOCAL="/opt/dlami/nvme/my_job"  # 28TB NVMe ephemeral storage
export TORCH_HOME="/opt/dlami/nvme/torch_cache"
```

**Prevention**: Always add cleanup at the end of sbatch scripts (after S3 upload of results) and use `/opt/dlami/nvme/` for large temporary files (model weights, outputs). The root filesystem has only ~30G free after the OS and packages.
