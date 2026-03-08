# Bug 006: `--labels` with `--train`/`--val` mode broken in train_probe.py

**Severity**: MEDIUM (latent — not triggered by documented workflows)
**Discovered**: 2026-03-07
**Status**: Open

## Root Cause

`evals/train_probe.py` lines 510-511. When `--labels` is combined with `--train`/`--val`, the same label-only NPZ (with its `indices` array) is applied to both train and val embedding files:

```python
X_train, y_train, p_train = load_npz(args.train[idx], labels_path=args.labels)
X_val, y_val, p_val = load_npz(args.val[idx], labels_path=args.labels)
```

The `indices` in the label NPZ are row positions into the *master* embedding file. If `--train` and `--val` are pre-split study-level NPZs (not the master), these indices are meaningless and will either cause an IndexError or silently select wrong rows.

## Impact

**Not triggered by current workflows.** The documented usage:
- k-fold mode (`--data` + `--labels`): works correctly (line 485)
- train/val mode (`--train`/`--val`): uses pre-split NPZs with embedded labels, `--labels` is not passed

The bug would only manifest if someone mixed the two modes.

## Recommended Fix

Error out if `--labels` is combined with `--train`/`--val`:
```python
if args.labels and (args.train or args.val):
    parser.error("--labels cannot be used with --train/--val (use pre-split NPZs with embedded labels)")
```
