---
name: Always deploy before sbatch
description: MUST run ~/deploy.sh before every sbatch submission on HyperPod — compute nodes have no git access
type: feedback
---

Always run `~/deploy.sh` before submitting any sbatch job on HyperPod.

**Why:** Compute nodes (ip-10-0-50-83, ip-10-0-50-184) are in a private subnet with no GitHub access. Code lives at `/opt/vjepa2` on each node, placed there by `deploy.sh`. If you skip the deploy, the nodes run stale code. This caused Bug 017a (stale code.tar) and a wasted LVEF probe retrain (probe trained without z-score normalization because deployed code was outdated).

**How to apply:** Before every `sbatch` call, run `~/deploy.sh` (deploys to both nodes by default). The sequence is always: `git pull` (if needed) → `~/deploy.sh` → `sbatch`. No exceptions, even for "just resubmitting the same job."
