# claude/dev/

Development log for the EchoJEPA project: bug tracker, changelog, operational guides, and code review findings. This is the single source of truth for what's broken, what's been fixed, and what's planned.

## Bug Tracker

| # | Title | Severity | Status | File |
|---|-------|----------|--------|------|
| 001 | [DistributedSampler shuffle corrupts embedding order](bugs/001-shuffle-bug.md) | **CRITICAL** | **FIXED** | `extract_embeddings.py`, `extract_uhn_embeddings.py` |
| 002 | [Encoder adapter normalization errors](bugs/002-normalization-bugs.md) | **HIGH** | Fixed in code, MIMIC re-extraction needed | `panecho_encoder.py`, `echo_prime_encoder.py`, `echofm_encoder.py` |
| 003 | [EchoFM temporal padding](bugs/003-echofm-padding.md) | Moderate | **FIXED** | `echofm_encoder.py` |
| 004 | [Silent index misalignment on video load failure](bugs/004-video-load-substitution.md) | **HIGH** | **FIXED** (tracking added) | `src/datasets/video_dataset.py` |
| 005 | [`drop_last` not forwarded to DataLoader](bugs/005-drop-last-not-forwarded.md) | **MEDIUM** | **FIXED** | `src/datasets/data_manager.py` |
| 006 | [train_probe --labels with --train/--val broken](bugs/006-labels-trainval-mode.md) | **MEDIUM** | Open (latent) | `evals/train_probe.py` |
| 007 | [Probe checkpoint loss — no backup, no S3 push](bugs/007-checkpoint-loss.md) | **CRITICAL** | **FIXED** | `scripts/run_uhn_probe.sh` |

## Planned Fixes

| Priority | Bug | Action | Blocked by |
|----------|-----|--------|------------|
| P0 | 007 | Retrain LVEF, TAPSE, MR sev, AS sev (20 runs) + AV Vmax G/L/L-K (3 runs) | GPU time |
| P1 | 002 | Re-extract PanEcho, EchoPrime, EchoFM on MIMIC | GPU time |
| P2 | 006 | Add validation guard in `train_probe.py` | — |

## Other Files

| File | Contents |
|------|----------|
| [roadmap.md](roadmap.md) | Consolidated outstanding work: blocking extractions, MVP tasks, strong additions, what's done |
| [changelog.md](changelog.md) | Chronological record of code changes, fixes, and extraction runs |
| [ops.md](ops.md) | Operational guide: UHN 18M extraction (launch commands, performance tuning, S3 bottleneck, crash recovery, timing) |
| [code-review.md](code-review.md) | Full-repo code review findings: encoder adapters, extraction scripts, pooling, probes, eval scaffold |
| [efficiency.md](efficiency.md) | Probe training efficiency: HP grid analysis (per-head TAPSE/LVEF results), narrowing from 20→12 heads, other speedup options |
