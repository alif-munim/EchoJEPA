# Operational Notes for NeurIPS Experiments

Lessons learned from running experiments on A100 (SageMaker) and HyperPod H100 clusters. Read before launching any new runs.

---

## Probe training: never chain runs

**Problem (2026-04-05):** Chaining probe training runs with `&&` in a single background task produces one shared output file. The eval logs don't include the model/checkpoint name, so when grep-ing for val MAE scores, results from different runs get mixed together. This caused JEPA IN21K e75 val MAE to be misattributed.

**Rules:**
1. **Always launch each probe as its own separate background task.** One model per task = one output file per model. Unambiguous.
2. **Never chain `evals.main` calls with `&&`.** Even with `echo` markers between them, the output interleaving is error-prone.
3. **Use absolute paths** for all output files (CSVs, logs). Relative paths resolve to the background shell's cwd, which may differ from the repo root. (This caused the first frame shuffling severity run to "lose" all output files.)
4. **Use `PYTHONUNBUFFERED=1`** for all background Python jobs. Without it, stdout buffers and progress isn't visible until the process exits.
5. **After a run completes, always verify the output file exists and has the expected content** before reporting results.

## Config file gotchas

- **`resume_checkpoint`**: Set to `true` only when you want to resume from `latest.pt` in the output folder. After a run completes, reset to `false` — otherwise the next model's run will try to load the previous model's checkpoint. The `sed` in-chain approach is fragile; prefer separate config files or manual reset.
- **`tasks_per_node`**: Must match the number of `--devices` passed. If you launch on 7 GPUs but config says `tasks_per_node: 8`, rank 7 will crash.

## Frame shuffling severity script

- The script `scripts/rebuttal/frame_shuffle_severity.py` has hardcoded model configs in `PT50_END_CONFIGS` and `ALL_CONFIGS` dicts. When adding new models, update both dicts.
- Use `--models MODEL_NAME` to run a single model per GPU. Do NOT use `--all` in a single process — it runs models sequentially on one GPU.
- For parallel runs, launch one process per model per GPU with separate output files.

## Killing GPU processes

- `pkill -f <script_name>` kills the main process but spawned multiprocessing workers may survive.
- Always follow with: `ps aux | grep multiprocessing | grep -v grep | awk '{print $2}' | xargs kill -9`
- Verify with `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` — if memory is still allocated, use `fuser -k /dev/nvidiaN`.
- Some zombie processes hold ~2GB indefinitely. These usually clear on their own after a few minutes.

## S3 checkpoint downloads

- BYOL v2 original training (e0-e50): `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/`
- BYOL v2 resume (e55-e105): `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/byol-vitl-imagenet-v2-resume/`
- VideoMAE original (e4-e54): `.../runs/videomae_matched_2n_245/training_folder/checkpoint-N.pth`
- VideoMAE resume (e58-e116): `.../runs/videomae_resume_e54_354/training_folder/checkpoint-N.pth`
- JEPA IN21K (e0-e100): `.../runs/jepa_in21k_pretrain_376/checkpoints/eN.pt`
- SALT S2 (e4-e79): `.../runs/salt_s2_pretrain_388/checkpoints/eN.pt`

Note: BYOL original and resume runs use **different S3 buckets**. The training code's `s3_checkpoint_uri` determines which bucket checkpoints land in.

## Init confound reference

Always check the encoder init before comparing models:

| Model | Init | How to verify |
|-------|------|--------------|
| JEPA pt50 | Fully-trained 235ep JEPA | `anneal_ckpt: vitl.pt` in config |
| JEPA IN21K | ImageNet-21K | Job 376 |
| BYOL | ImageNet-21K | `anneal_ckpt: vitl_in21k.pt` |
| MAE | ImageNet | `args.finetune` in checkpoint |
| SALT S1 | ImageNet-21K | `force_load_pretrain: true` + `anneal_ckpt` |
| SALT S2 | Random (student) | `force_load_pretrain: false` — correct per SALT paper |

**Do not use JEPA pt50 in the primary comparison table.**

## SALT configuration discrepancies (discovered 2026-04-06)

The initial SALT S1+S2 runs (jobs 379, 388, 391, 392) had several hyperparameter mismatches vs the SALT paper. These are now fixed in the configs but the existing checkpoints were trained with the old values.

| Parameter | SALT Paper | Our Initial Config | Fixed Config | Impact |
|-----------|-----------|-------------------|-------------|--------|
| **S2 Loss** | L1 (SALT paper Eq 2.1: \|\|...\|\|_1) | L1 (`loss_exp: 1.0`) | L1 (`loss_exp: 1.0`) | Was incorrectly changed to 2.0, now reverted |
| **Peak LR** | 0.000625 | 1.75e-4 | 2.55e-4 (sqrt-scaled) | v1 too low; v2 unscaled; v3 sqrt-scaled |
| **Start LR** | 0.0002 | 3.33e-5 | 8.2e-5 (sqrt-scaled) | v1 too low; v2 unscaled; v3 sqrt-scaled |
| **Final LR** | 1e-6 (cosine decay) | 1.75e-4 (constant) | 1e-6 | No LR decay → never fine-tunes |
| **Weight decay ramp** | 0.04→0.4 (cosine) | 0.04→0.04 (constant) | 0.04→0.4 | Missing regularization ramp |
| **ipe_scale** | 1.0 | 1.25 | 1.0 | 25% phantom steps (SALT paper explicitly disables) |
| **Resize scale** | [0.3, 1.0] | [0.5, 1.0] | [0.3, 1.0] | Less aggressive cropping |
| **Resize aspect** | [0.75, 1.35] | [0.9, 1.1] | [0.75, 1.35] | Less aspect variation |
| **Batch size** | 3072 | 512 (64×8) | 512 (GPU-limited) | 6× smaller — cannot fix without more GPUs |

**Files fixed (round 1, 2026-04-06):**
- `app/salt/train.py`: `loss_exp` default changed from 1.0 to 2.0
- `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml`: LR, WD, ipe_scale, augmentation
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`: LR, WD, ipe_scale, augmentation, loss_exp

**Files fixed (round 2, 2026-04-07, after full paper audit):**
- `app/salt/train.py`: `loss_exp` default reverted to 1.0 (SALT paper uses L1, not L2 — the round 1 "fix" was wrong)
- `app/salt/train.py`: Stage 2 teacher/student forward mode changed to single-level (`training_mode=False`) unless `n_output_distillation` is explicitly set. SALT paper uses standard V-JEPA embeddings, NOT V-JEPA 2.1 hierarchical output.
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`: `loss_exp` reverted to 1.0, `pred_num_heads` fixed from 12 to 16

**Files fixed (round 3, 2026-04-07, LR batch-size scaling):**
- `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml`: warmup 40→33, lr 6.25e-4→2.55e-4, start_lr 2e-4→8.2e-5
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`: same warmup/LR changes
- `claude/architecture/salt-training-reference.md`: updated to match sqrt-scaled values

**LR scaling decision:** sqrt scaling chosen over linear. Paper's 6.25e-4 at batch 3072 → 2.55e-4 at batch 512 (factor = sqrt(512/3072) ≈ 0.408). Linear would give 1.04e-4, which is even lower than v1's accidentally-low 1.75e-4. Prior S1 loss curves suggest unscaled 6.25e-4 is too high (worse final loss than 1.75e-4), and linear may be too conservative for only 20 S1 epochs. V-JEPA 2 uses linear scaling but at a smaller reduction (3×, not 6×). See `claude/architecture/salt-training-reference.md` § "LR Scaling Rationale" for full analysis.

**Remaining deliberate deviation:**
- ImageNet-21K init for S1 teacher: deliberate for controlled comparison, but deviates from paper (trains from scratch).

**Revised implication (2026-04-07):** The table above characterizes v3 as the "fully corrected" run, implying v1 results are invalid. **This framing is out of date.** Both v1 (jobs 388/391/392) and v3 (job 446) are **legitimate SALT variants** with different design choices:

| | v1 (hierarchical) | v3 (paper-spec) |
|---|---|---|
| S2 predictor | Hierarchical 4-layer (V-JEPA 2.1 extension) | Single-level (SALT paper Eq 2.1) |
| Loss | L1 (`loss_exp: 1.0`) | L1 (`loss_exp: 1.0`) |
| Peak LR | 1.75e-4 constant | 2.55e-4 cosine decay to 1e-6 |
| Weight decay | 0.04 constant | 0.04→0.4 cosine ramp |
| ipe_scale | 1.25 (virtual early stopping) | 1.0 |
| Augmentation | weak ([0.9, 1.1] aspect, [0.5, 1.0] scale) | paper ([0.75, 1.35], [0.3, 1.0]) |
| `pred_num_heads` | 12 | 16 |
| Test MAE (EchoNet-Dynamic) | **6.66 (best)** | 7.03 |
| Test R² (EchoNet-Dynamic) | **0.414 (best)** | 0.348 |

**Both v1 and v3 used L1 loss** (the `loss_exp: 1.0` is in both checkpoints' saved configs). The "round 1 fix" mentioned above (`loss_exp` default changed from 1.0 to 2.0) was briefly in `app/salt/train.py` but the configs always specified 1.0, which overrides the default. Neither v1 nor v3 was ever trained with L2 loss.

**Both variants are valid SALT implementations.** v1 uses the V-JEPA 2.1 hierarchical distillation extension; v3 matches paper Eq 2.1 strictly. v1 happens to outperform v3 on EchoNet LVEF, likely because hierarchical features help the d=4 attentive regression probe. The gap to EMA-based methods is robust across both variants (JEPA 0.652, BYOL 0.511, MAE 0.447 vs SALT v1 0.414, v3 0.348).

**For the paper: use v1 e79 as the primary SALT row** (best test R² among SALT variants). No retraining required. The consistent finding across v1, v1 e199, and v3 is that SALT underperforms all three EMA-based objectives by 0.03–0.24 R². See `claude/neurips/experiments/salt-comparison.md` for the full writeup and conservative framing.

## Probe result extraction

When extracting val MAE from logs, always:
1. Identify the run boundaries (look for `Epoch 1` or model-loading messages)
2. Count exactly 20 val entries per run (20 probe epochs)
3. The `best.pt` checkpoint corresponds to the lowest val MAE across all 6 HP heads, not the last epoch
4. Cross-check: `best.pt` file timestamp should match the epoch with the lowest val MAE
