---
name: HyperPod job monitoring from controller
description: How to check running jobs and training progress when Claude Code is on the HyperPod controller node
type: feedback
---

Claude Code runs directly on the HyperPod controller node (ip-10-0-50-52). Do NOT attempt SSH or SSM to reach compute nodes — use srun directly.

**Why:** SSH from controller to compute fails (no keys configured). SSM requires external access. srun --overlap works without interfering with running jobs.

**How to apply:**
- Use `squeue`, `sinfo`, `sacct` directly (no wrapping needed)
- Use `srun --jobid=X --nodes=1 --nodelist=NODE --ntasks=1 --overlap bash -c "CMD"` to run commands on compute nodes
- Ignore `slurmstepd: error: couldn't chdir` warnings — they're harmless (controller CWD doesn't exist on compute)
- Check training progress via CSV logs (`log_r0.csv`), NOT sbatch stdout files (stdout is fully buffered under srun, appears empty)
- Check checkpoint health via `ls -lht *.pt` timestamps + `aws s3 ls` for S3 archive
- Checkpoint saves happen at epoch boundaries — stale timestamps may just mean the current epoch hasn't finished yet
