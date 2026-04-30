# HyperPod Cluster Operations Guide

Operational guide for SageMaker HyperPod GPU clusters used for EchoJEPA training. Covers connectivity, cluster management, and remote command execution. Split across 4 files:

| File | Contents |
|------|----------|
| **`hyperpod-ops.md`** (this file) | Cluster inventory, connectivity, management commands, non-interactive SSM |
| [`hyperpod-cluster-creation.md`](hyperpod-cluster-creation.md) | Step-by-step cluster provisioning (7 steps, reusable config reference) |
| [`hyperpod-deployment.md`](hyperpod-deployment.md) | Code deployment (git + S3 tarball), conda-pack, job submission, monitoring, multi-node training, BYOL status |
| [`hyperpod-troubleshooting.md`](hyperpod-troubleshooting.md) | 18 troubleshooting issues with fixes (2026-03 -- 2026-04) |

Distilled from echojepa-h100-march (2026-03-26) and echojepa-h100-neurips (2026-04-12/13) setups.

## Cluster Inventory

| Cluster | ID | Instance Type | GPUs | Training Plan | Status |
|---------|------|--------------|------|---------------|--------|
| echojepa-h100-neurips | n9we8xfqjv3p | ml.p5.48xlarge | 8x H100 80GB | EchoJEPA-NeurIPS | InService (compute node: ip-10-0-50-35, i-065188ac6aa4aaadb; controller: i-0415ce8f417564270) |
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

### Direct SSM to Compute Nodes

When a job is running and occupying the full node, `srun` blocks. You can SSM directly to the compute node to read logs:

```bash
# Get compute node instance ID
COMPUTE_ID="$(aws sagemaker list-cluster-nodes --cluster-name echojepa-h100-neurips --region us-west-2 \
  --query "ClusterNodeSummaries[?InstanceGroupName=='echojepa-neurips-compute'].InstanceId" --output text)"

# Read job output directly (doesn't compete with running job)
script -q -c "timeout 30 aws ssm start-session --region us-west-2 \
  --target 'sagemaker-cluster:n9we8xfqjv3p_echojepa-neurips-compute-${COMPUTE_ID}' \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters '{\"command\":[\"tail -80 /tmp/nmed_predavg-16.out\"]}'" /dev/null
```

**When to use**: Reading stdout/stderr of running jobs. The sbatch output files (`/tmp/*-{jobid}.out`) are on the compute node, not the controller. `srun` to an occupied node blocks until the job finishes.

### SSH from Controller to Compute Nodes

SSH keys are NOT configured by default on HyperPod. Use Slurm's `srun` instead:
```bash
sudo su - ubuntu
srun -N1 -w ip-10-0-50-35 --ntasks=1 --pty bash   # interactive shell
srun -N1 -w ip-10-0-50-35 --ntasks=1 bash -c "hostname && nvidia-smi"  # one-off command
```

**Note**: When a job is using the full node (all GPUs/CPUs), `srun` blocks waiting for resources. Use `srun --jobid=<ID> --overlap` to share resources with a running job, or use direct SSM to the compute node (see above).

### Network Topology

- **SageMaker notebook instance**: Default VPC (172.31.x.x) -- cannot SSH to cluster
- **HyperPod cluster**: Custom VPC `vpc-0a306d982844ee4e9` (10.0.50.0/24)
- **Cross-VPC access**: Not possible without VPC peering or bastion. Use SSM from notebook or local machine.

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

The JSON parameters go through multiple layers of escaping (outer shell -> JSON -> inner shell). Rules of thumb:
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
