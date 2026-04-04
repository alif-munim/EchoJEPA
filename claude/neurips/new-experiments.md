# New Experiments Needed for NeurIPS

Experiments not yet run that are needed for the NeurIPS submission. Organized by priority.

---

## P0 — Critical Path (Blocks Paper)

### SALT Pretraining (ViT-L, 50 epochs, MIMIC 525K)

The single most important new experiment. Completes the {pixel, latent} × {EMA, frozen} 2×2 design.

**Stage 1 (V-Pixel teacher):**
- Train pixel reconstruction teacher with multi-block masking
- Config: `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f.yaml`
- Output: `checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f/latest.pt`
- Compute: ~2-3 days on 8×A100 or 8×H100
- Code: `app/salt/train.py` (stage=1)

**Stage 2 (Frozen-teacher student):**
- Freeze S1 encoder, train new student + predictor to predict frozen teacher's latents
- Config: `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f.yaml` (update `teacher_checkpoint` to point to S1 output)
- Output: `checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/latest.pt`
- Compute: ~2-3 days on 8×A100 or 8×H100
- Code: `app/salt/train.py` (stage=2)
- Depends on: S1 checkpoint

**Key question this answers:** Does the latent target or the EMA mechanism drive noise filtering?
- SALT S2 ≈ JEPA → EMA is unnecessary, frozen teacher suffices
- SALT S2 > MAE but < JEPA → both mechanisms contribute
- SALT S1 ≈ MAE on downstream → pixel reconstruction profile same regardless of masking strategy

**Epoch matching:** Must be 50 epochs to match existing JEPA/BYOL/MAE comparison. The SALT paper uses 240K total steps split between S1 and S2; for 50 epoch-matched comparison, allocate ~15-20 epochs to S1 and ~30-35 to S2 (following the paper's finding that student should get more compute). Alternatively, train S1 for 50ep and S2 for 50ep (total 100ep of compute, but S1 is cheaper — no predictor, just encoder + lightweight decoder).

**Decision needed:** How to epoch-match SALT fairly. Options:
1. S1=50ep + S2=50ep (100ep total compute, but S1 is cheaper per-step)
2. S1=15ep + S2=35ep (50ep total, matching the paper's allocation guidance)
3. S1=50ep + S2=50ep, report total FLOPs for fair comparison (SALT paper approach)

### SALT 5-Task Evaluation Battery

Once S2 checkpoint exists, run the same evaluation as the existing 3-way:

| Task | Config Template | Data | Probe Epochs | Estimated Time |
|------|----------------|------|-------------|---------------|
| LVEF (UHN, 10K/53K) | `echojepa_l_pt50_lvef_d4.yaml` (adapt) | `lvef_train_10k.csv` | 20 | ~4 hours |
| RVSP (UHN, 41K/5K, multi-view) | `echojepa_l_pt50_rvsp_d4_full.yaml` (adapt) | `rvsp_{train,val,test}_{a4c,psax}.csv` | 20 | ~8 hours |
| CAMUS segmentation | `scripts/rebuttal/camus_frozen_*` (adapt) | CAMUS | 50 | ~4 hours |
| EchoNet-Dynamic LVEF | `echojepa_l_pt50_end_lvef_d4.yaml` (adapt) | `echonet_dynamic_{train,val,test}_*.csv` | 20 | ~4 hours |
| Pediatric zero-shot | No training — run UHN probe on pediatric test | `echonet_pediatric_test_*.csv` | 0 | ~30 min |

**Total probe compute:** ~1 day on 2-4 GPUs

### SALT EchoBench (Noise Robustness)

Run the same perturbation matrix as existing models:
- 3 perturbation types (depth attenuation, acoustic shadow, haze) × 3 severity levels
- On EchoNet-Dynamic LVEF (1,277 test) and CAMUS segmentation (50 test patients)
- Script: `scripts/rebuttal/noised_inference.py` (adapt for SALT encoder)
- Compute: ~4 hours

### SALT Frame Shuffling (6 Conditions)

Run the same temporal ablation as existing models:
- 6 conditions: clean, tubelet, reverse, matched, shuffle (3 seeds), matched_frame
- On EchoNet-Dynamic test (1,277 videos) with frozen LVEF probe
- Pipeline: `evals.main` with VideoDataset temporal ablation hook
- Compute: ~2 hours

### SALT Speckle Probing

Run `scripts/rebuttal/information_probing.py` on SALT encoder embeddings:
- Extract mean-pooled features from 2,554 EchoNet-Dynamic clips
- Train linear probes for speckle energy, mean intensity, texture variance
- Compute partial R² controlling for intensity
- Compute: ~1 hour

---

## P1 — High Priority (Strengthens Paper)

### V-JEPA 2.1 Probe Evaluation

V-JEPA 2.1 adds dense hierarchical supervision (predict all tokens, not just masked). If a checkpoint exists on MIMIC, evaluate it on the 5-task battery. This tests whether dense supervision improves spatial encoding (closing the segmentation gap) while preserving functional performance.

- Check: Does `checkpoints/` contain a V-JEPA 2.1 ViT-L MIMIC checkpoint?
- The ViT-B 2.1 checkpoint exists (`vjepa2_1_vitb_mimic_p169_c60.pt`) but is a different model size
- If no ViT-L 2.1 checkpoint: skip this experiment (training from scratch is too expensive for the insight gained)

---

## P1.5 — Frame Shuffling Extensions (Deepens Mechanistic Story)

These experiments extend the existing 6-condition frame shuffling results (see `experiments/frame-shuffling.md`) with follow-ups that strengthen the NeurIPS mechanistic evidence section.

### a) Shuffle Severity Gradient

**Question:** Is temporal integration global (every frame depends on every other) or local (only adjacent frames matter)?

**Design:** Instead of all-or-nothing frame shuffle, shuffle only a fraction of frames: 0% (clean), 25%, 50%, 75%, 100%. For each severity, randomly select that fraction of frame positions and shuffle only those while leaving the rest in original order. Use matched_frame variant (RoPE remapped) to remove positional confound.

**Prediction:**
- If JEPA's degradation scales **linearly** with shuffle percentage → global temporal integration (features depend on the full sequence)
- If MAE's degradation is **sublinear** (most damage from first 25%, diminishing returns) → local temporal structure (only adjacent-frame relationships matter, and disrupting a few is enough to break those)
- BYOL should show **concave** degradation (rapid collapse then plateau) — once the global pool loses temporal coherence, additional shuffling doesn't help

**Output:** Degradation curves (R² vs shuffle percentage) per model — a clean additional figure for NeurIPS §4.

**Implementation:** Modify `src/datasets/video_dataset.py` temporal ablation to accept a `shuffle_fraction` parameter. Or create a new script extending `scripts/rebuttal/frame_shuffle_task.py`. Use the `evals.main` pipeline for prediction averaging consistency.

**Compute:** ~3 hours (5 severity levels × 3 models × 3 seeds each, reusing existing probes)
**Depends on:** Existing END LVEF probes (already trained)

### b) CAMUS Segmentation Under Frame Shuffling

**Question:** Is temporal dependence task-specific? Does the ranking inversion (MAE > JEPA on segmentation, JEPA > MAE on LVEF) persist under temporal disruption?

**Design:** Run frame shuffling (matched_frame condition, the most rigorous) on CAMUS segmentation. Use the same frozen segmentation decoders from the noise robustness experiment.

**Prediction:** All three models should show **minimal Dice degradation** under frame shuffling, because segmentation is a per-frame spatial task — frame order should be irrelevant for localizing the endocardial border. If confirmed:
- Temporal encoding drives functional task performance but is irrelevant for spatial tasks
- The ranking inversion (MAE wins segmentation, JEPA wins LVEF) is explained by different spatial information types (boundary precision vs chamber geometry), not by temporal encoding
- This directly connects frame shuffling to the anatomy-function dissociation

**If the prediction fails** (some model degrades substantially on segmentation under shuffling): that would mean temporal context aids even spatial tasks, which is interesting but complicates the clean story. Still publishable as a nuance.

**Implementation:** Adapt `scripts/rebuttal/noised_segmentation.py` to apply frame shuffling instead of noise perturbations. Needs the `matched_frame` shuffle type from `src/datasets/video_dataset.py`.

**Compute:** ~1 hour (3 models × 1 condition × 50 test patients)
**Depends on:** Existing CAMUS segmentation decoders (already trained)

### c) View Classification Under Frame Shuffling

**Question:** Is view classification (a largely static/spatial task) invariant to temporal disruption, like segmentation?

**Design:** Run frame shuffling (matched_frame) on frozen view classification probes. View identity is primarily determined by anatomy visible in a single frame, so shuffling should have minimal effect.

**Prediction:** All models should show <5% accuracy degradation. This adds a third task type (classification) to the temporal-dependence-by-task analysis, strengthening the generalization from "LVEF is temporal, segmentation is not" to "functional tasks are temporal, spatial tasks are not."

**Compute:** ~30 min per model
**Depends on:** View classification probes (may need to train if not already done — check `new-experiments.md` P2)

### d) SALT Temporal-Spatial Decomposition

Once SALT experiments are complete (P0), run the same matched_frame decomposition to determine SALT's temporal/spatial split. This answers whether the frozen pixel teacher produces temporal representations like MAE (low temporal %) or like JEPA (higher temporal %).

**Prediction:**
- SALT S1 (V-Pixel teacher) should decompose similarly to MAE (~70-75% spatial)
- SALT S2 (latent student) is the key question: if it decomposes like JEPA (~60% spatial), the latent target matters more than the teacher type

**Compute:** ~2 hours (already included in P0 SALT frame shuffling)
**Depends on:** SALT S2 checkpoint (P0)

---

## P2 — Medium Priority (Broadens Coverage)

### View Classification (All 4 Paradigms)

Attentive probe for 5-class echo view classification. Broadens the task coverage beyond regression + segmentation. Quick to run since probes are small.

- Models: JEPA, BYOL, MAE, SALT
- Data: UHN view classification splits (existing CSVs)
- Compute: ~2 hours per model

### EchoBench Reference Baselines

Add DINOv2 (ViT-L, public checkpoint) and a randomly initialized ViT-L as EchoBench reference baselines. This transforms EchoBench from "our evaluation tool" to "community benchmark with reference points."

- DINOv2: `facebookresearch/dinov2` via torch.hub
- Random ViT-L: same architecture, random weights
- Run through noise robustness matrix + LVEF/CAMUS evaluation
- Compute: ~4 hours total

---

## P3 — Nice to Have (Deepens Story)

### Training Dynamics: Noise Filtering Emergence

Track speckle probing, intrinsic dimensionality, and LVEF R² across pretraining epochs. Shows *when* the noise-filtering mechanism emerges during training.

- Requires: epoch checkpoints for JEPA and MAE (check if available at epochs 10, 20, 30, 40, 50)
- For each checkpoint: extract embeddings on EchoNet-Dynamic, run speckle probing, train quick LVEF probe
- Would produce a compelling training dynamics figure
- Compute: ~1 day (many small inference runs)

### Biplane LVEF (A4C + A2C Multi-View)

97% of UHN studies have both A4C and A2C views. Biplane Simpson's is the clinical gold standard for LVEF. Would demonstrate multi-view probe on the most clinically important view combination.

- Data availability confirmed: 48,397 of 49,894 studies have both views
- Requires: building biplane CSVs, training multi-view probes
- Compute: ~1 day
- Risk: may overlap with Nature Medicine scope (check deconfliction)

---

## Dependency Chain

```
SALT S1 pretrain (P0, ~3 days)
    └── SALT S2 pretrain (P0, ~3 days)
            ├── SALT 5-task evaluation (P0, ~1 day)
            │       ├── SALT EchoBench (P0, ~4 hours)
            │       └── SALT frame shuffling + decomposition (P0/P1.5d, ~2 hours)
            └── SALT speckle probing (P0, ~1 hour)

Existing checkpoints (can start immediately):
    ├── Shuffle severity gradient (P1.5a, ~3 hours)
    ├── CAMUS segmentation under shuffling (P1.5b, ~1 hour)
    ├── View classification under shuffling (P1.5c, ~1.5 hours) [needs view probes]
    ├── View classification probes (P2, ~8 hours) [if not already trained]
    ├── DINOv2 + random baselines (P2, ~4 hours)
    ├── Training dynamics (P3, ~1 day)
    └── Biplane LVEF (P3, ~1 day)

V-JEPA 2.1 checkpoint check (P1, ~5 min)
    └── V-JEPA 2.1 probes (P1, ~1 day) [only if checkpoint exists]
```

**Critical path:** SALT S1 → S2 → evaluation. P1.5a-c can start immediately on existing checkpoints. P1.5d depends on SALT.
