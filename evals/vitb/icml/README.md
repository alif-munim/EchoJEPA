# EchoJEPA-B (V-JEPA 2.1) ICML Rebuttal Probe Runs

ICML rebuttal Tier 2, Experiment 5: ViT-B scaling analysis (B -> L -> G).

## Status (2026-03-28)

| Task | Epochs | Status | Best Val |
|------|--------|--------|----------|
| LVEF | 19/20 | **Stopped** (epoch 20 training interrupted at batch ~6K/22K) | R²=0.650, MAE=5.24, r=0.806 (ep19) |
| RVSP | 0/20 | Not started | — |
| View | 0/20 | Not started | — |

## Resume Commands

### LVEF (finish epoch 20)
The config has `resume_checkpoint: false` — change to `true` to resume from `latest.pt`:
```bash
# Edit config first: set resume_checkpoint: true
python -m evals.main \
    --fname configs/eval/vitb/icml/echojepa_b_lvef_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

### RVSP / View (start fresh)
```bash
python -m evals.main \
    --fname configs/eval/vitb/icml/echojepa_b_rvsp_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

python -m evals.main \
    --fname configs/eval/vitb/icml/echojepa_b_view_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

### Queue all 3 sequentially (with L-K PA at end)
```bash
nohup bash /tmp/queue_vitb_then_lk.sh > logs/vitb_then_lk_queue.log 2>&1 &
```

## EchoMAE-L Runs

### EchoMAE-L RVSP (ep163, full dataset)

| Task | Checkpoint | Dataset | Epochs | Status | Best Val |
|------|-----------|---------|--------|--------|----------|
| RVSP | videomae-ep163.pth | Full (41K train / 5K val) | 0/20 | **Running** | — |

```bash
PYTHONUNBUFFERED=1 nohup python -m evals.main \
    --fname configs/eval/vitb/icml/echomae_l_rvsp_d4_ep163.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    > logs/echomae_l_rvsp_ep163_full.log 2>&1 &
```

### EchoMAE-L Rebuttal Results (5K subsets, ep99)

| Task | Epochs | Best Val | Notes |
|------|--------|----------|-------|
| LVEF | 20/20 | R² ≈ 0, MAE ~8.0 | No signal (task-dependent, not degenerate ckpt) |
| RVSP | 20/20 | MAE ~150 | **Invalid** — stale zscore_params.json had LVEF values |
| View | 20/20 | Acc 44.1%, AUROC 0.847 | Good performance |

## Known Bug: DDP Crash from TMPDIR / LD_LIBRARY_PATH Env Vars

**Symptom:** Rank 0 initializes as `world_size=1` while ranks 1-7 initialize as `world_size=8`, causing NCCL `socketPollConnect: Connection refused` followed by `_verify_param_shape_across_processes` crash.

**Root cause:** Setting `TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH` before launch interferes with the distributed initialization in `src/utils/distributed.py`. Rank 0 gets a different environment view and initializes a solo process group, while other ranks find each other via NCCL but cannot connect to rank 0.

**Fix:** Do NOT set `TMPDIR` or `LD_LIBRARY_PATH` when launching distributed eval. Just run:
```bash
PYTHONUNBUFFERED=1 nohup python -m evals.main --fname <config> --devices cuda:0 ... cuda:7 > logs/run.log 2>&1 &
```

**Also watch for:** Stale `zscore_params.json` files. The eval code loads z-score params from `os.path.dirname(train_data_path)`. If a parent-level `zscore_params.json` exists with wrong task values (e.g., LVEF mean/std in an RVSP directory), it will silently produce garbage MAE. Fix: either delete stale files or set explicit `target_mean` / `target_std` in the YAML config (these override file-based params).

## Config Details

- **Checkpoint:** `checkpoints/vjepa2_1_vitb_mimic_p169_c60.pt` (V-JEPA 2.1, ViT-B 86M)
- **Encoder adapter:** `vit_encoder_multiclip_v21` (handles 2.1 differences: norms_block, modality embeddings, on-the-fly RoPE)
- **Critical config fix:** `use_rope: true` must be in `pretrain_kwargs.encoder` (V-JEPA 2.1 uses RoPE, not pos_embed; omitting this causes `AttributeError: 'VisionTransformer' has no attribute 'pos_embed'`)
- **Probe:** d=4 attentive, 16 heads (ICML protocol)
- **HP grid:** 6 combos — LR {1e-4, 5e-5} x WD {0.01, 0.1, 0.4}
- **Training:** 20 epochs, batch_size=1/GPU x 8 GPUs, bfloat16, 224px, 16 frames
- **Data format:** Raw values (LVEF in %, RVSP in mmHg), z-normalized at runtime

## Checkpoints

```
evals/vitb/icml/lvef/video_classification_frozen/icml-echojepa-b-lvef-d4/
├── best.pt          # best val R² (epoch 19)
├── latest.pt        # epoch 19 (for resume)
├── epoch_001.pt ... epoch_019.pt
└── log_r0.csv       # per-epoch val metrics
```

## LVEF Results (19 epochs)

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 7.125 | 6.555 | 0.459 | 0.697 |
| 5 | 6.244 | 5.691 | 0.583 | 0.765 |
| 10 | 5.980 | 5.573 | 0.583 | 0.780 |
| 15 | 5.772 | 5.291 | 0.642 | 0.802 |
| 19 | 5.683 | 5.244 | 0.650 | 0.806 |

Best: **epoch 19 — Val MAE 5.24, R² 0.650, Pearson 0.806**

## Scaling Context

| Model | Params | Data | LVEF R² |
|-------|--------|------|---------|
| EchoJEPA-B (V-JEPA 2.1) | 86M | MIMIC 525K | 0.650* |
| EchoJEPA-L (V-JEPA 2.0) | 300M | MIMIC 525K | TBD |
| EchoJEPA-G (V-JEPA 2.0) | 1.1B | UHN 18M | 0.778 |

*19/20 epochs, plateauing. Epoch 20 would add ~0.001-0.002 R².

**Caveat:** B->L confounds model scale with architecture version (2.1 vs 2.0). L->G is the clean scaling point.
