# HyperPod Cluster Creation Guide

Complete procedure for provisioning a new SageMaker HyperPod cluster. Distilled from the echojepa-h100-neurips setup (2026-04-12), which required 4 attempts due to `provisioning_parameters.json` mismatches (Issue 13) and a missing `hotfix/` directory (Issue 15). See [hyperpod-troubleshooting.md](hyperpod-troubleshooting.md) for full issue details.

## Step 0: Prerequisites

1. **Service quota**: Verify quota for the desired instance type (e.g., `ml.p5.48xlarge`, `ml.p5e.48xlarge`). Check at: SageMaker console -> Service quotas
2. **Training plan**: Create or identify an active training plan under SageMaker -> Training Plans. Note the **Availability Zone** -- the cluster subnet must match
3. **Network**: Ensure a **private subnet** exists in the training plan's AZ within VPC `vpc-0a306d982844ee4e9`. The subnet must be associated with the Route Table (`rtb-0dc170...`) that has the S3 Gateway Endpoint. VPC Interface Endpoints must exist for `ssm`, `ssmmessages`, and `ec2messages`

## Step 1: Choose Names

Pick a cluster name and derive instance group names. These names must be consistent across the API call AND `provisioning_parameters.json` -- mismatches cause immediate provisioning failure (see Issue 13 in [hyperpod-troubleshooting.md](hyperpod-troubleshooting.md)).

```bash
CLUSTER_NAME="echojepa-h100-neurips"
CONTROLLER_GROUP="echojepa-neurips-controller"
COMPUTE_GROUP="echojepa-neurips-compute"
PARTITION_NAME="ml-p5-48xlarge"   # must match instance type: ml.p5.48xlarge -> ml-p5-48xlarge
```

## Step 2: Create S3 Bucket and Upload Lifecycle Files

Each cluster gets its own S3 bucket for lifecycle scripts. You can copy the base lifecycle scripts from an existing cluster and share heavy setup artifacts (conda env, source code) across buckets.

```bash
BUCKET="sagemaker-${CLUSTER_NAME}-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')-bucket"
aws s3 mb "s3://${BUCKET}" --region us-west-2

# Copy base lifecycle scripts from an existing cluster bucket
SOURCE_BUCKET="sagemaker-echojepa-h100-march-0d224785-bucket"
aws s3 sync "s3://${SOURCE_BUCKET}/" "s3://${BUCKET}/" \
  --exclude "setup/*" --exclude "checkpoints/*" --exclude "gpu_tests/*"
# NOTE: Do NOT exclude hotfix/ -- apply_hotfix.sh globs hotfix/*.sh and fails if the dir is missing
```

## Step 3: Upload provisioning_parameters.json (CRITICAL)

**This is the #1 cause of provisioning failures.** HyperPod validates that ALL instance group names in the `create-cluster` API call appear in this file. Missing groups -> immediate rollback.

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

## Step 4: Upload on_create.sh

Customize the lifecycle script for your cluster. The script runs on every node during provisioning. Key considerations:
- Use `command -v nvidia-smi` to detect compute vs controller nodes
- Reference setup artifacts from a shared bucket to avoid duplicating large files
- Wrap compute-node setup in `( set +e; ... ) &` to prevent failures from killing provisioning
- Restart the SSM agent at the end so the controller is reachable via SSM immediately
- Do NOT auto-launch training -- use `deploy.sh` + `sbatch` after provisioning is complete

```bash
aws s3 cp /path/to/on_create.sh "s3://${BUCKET}/on_create.sh"
```

## Step 5: Create the Cluster

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
- Only the compute group gets `TrainingPlanArn` -- the controller does not
- `ThreadsPerCore: 1` disables hyperthreading (standard for GPU workloads)
- 500GB EBS gives headroom for checkpoints, conda env, and data

## Step 6: Monitor Provisioning

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

**If provisioning fails** (status: `RollingBack` -> `Failed`):
1. Check the `FailureMessage` -- it usually identifies the exact issue
2. Fix the root cause (most commonly `provisioning_parameters.json`)
3. Delete the failed cluster: `aws sagemaker delete-cluster --cluster-name "${CLUSTER_NAME}" --region us-west-2`
4. Rollbacks for `ml.p5.48xlarge` can take **30-90 minutes** -- you must wait for `Failed` status before deleting
5. Recreate with the same command (the S3 bucket and lifecycle files persist)

## Step 7: Post-Provisioning Setup

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
#    (see hyperpod-deployment.md)

# 4. Deploy code and verify compute node environment
~/deploy.sh
srun -N1 --ntasks=1 bash -c "ls /opt/vjepa2 && nvidia-smi"
```

## Reusable Config Reference

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
