---
name: HyperPod deployment workflow
description: Default workflow for deploying code to H100 compute nodes on SageMaker HyperPod cluster echojepa-h100-march
type: project
---

Default code deployment workflow on the HyperPod cluster:

1. Controller node (ip-10-0-50-52) has the git repo at `~/EchoJEPA-repo`
2. `git pull` on controller to get latest changes
3. `~/deploy.sh` pushes code to BOTH compute nodes via srun (tar+unpack to /opt/vjepa2). Pass a node name to deploy to just one: `~/deploy.sh ip-10-0-50-83`
4. `sbatch scripts/<job>.sbatch` — all sbatch scripts use `/opt/vjepa2` (deployed code), NOT code.tar from S3

**Why:** Controller and compute nodes have no shared filesystem. Compute nodes are in a private subnet with no GitHub access. The old code.tar approach (downloading from S3) caused Bug 017a — stale code deployed after fixes were committed. Switched to deploy.sh on 2026-03-29.

**How to apply:**
- Always run `~/deploy.sh` after editing code on the controller before launching jobs
- deploy.sh defaults to both nodes; no need to run twice
- NEVER use code.tar in sbatch scripts — use `REPO_DIR="/opt/vjepa2"` and `cd "$REPO_DIR"` instead
- Sbatch scripts still download env, data CSVs, and checkpoints from S3 (those are large and node-local)
