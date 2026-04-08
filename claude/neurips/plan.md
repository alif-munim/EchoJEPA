# NeurIPS 2025 Execution Plan

**Last updated:** 2026-04-07
**Deadline:** May 4, 2026 (abstract), ~4 weeks remaining
**Status:** Week 1 — frame shuffling experiments complete, moving to core experiments

---

**Operational notes:** See `ops-notes.md` for lessons learned (never chain probe runs, absolute paths, SALT config audit, etc.)

---

## Completed Experiments

### Frame Shuffling (§4 — DONE)

- [x] **Severity gradient (0/25/50/75/100%)** — 13 models: JEPA IN21K e25/50/75/100, BYOL e24/50/75/100, MAE e24/50/74/99, SALT S2 e79. Results: `experiments/severity-gradient.md`
- [x] **6-condition (clean/tubelet/reverse/matched/shuffle/matched_frame)** — 12 models: JEPA IN21K e25/50/75/100, BYOL e24/50/75/100, MAE e24/50/74/99. Results: `experiments/6-condition-shuffling.md`
- [x] **Key finding: three temporal encoding regimes** — JEPA consolidates (−17%), BYOL stabilizes (−38%), MAE invariant (−4%). MAE temporal encoding is transient. JEPA shuffled > BYOL clean.
- [x] **Tube masking does not prevent the shortcut (2026-04-08 reframe)** — Confirmed our VideoMAE ViT-L used `--mask_type tube --mask_ratio 0.9` (canonical Tong et al. 2022 recipe across all three MIMIC sbatches). The shortcut MAE finds must therefore be within-frame spatial interpolation, not cross-frame copying. No masking intervention can fix this. Reframes §4 from observation to "tube masking, the community standard, fails." Results: `experiments/tube-masking-failure.md`

### Training Dynamics Probes (appendix — DONE)

- [x] **JEPA IN21K** e25/50/75/100 LVEF probes: val MAE 6.48 / 5.55 / 5.27 / 5.22
- [x] **BYOL** e24/50/75/100 LVEF probes: val MAE 6.37 / 6.17 / 5.99 / 5.94
- [x] **MAE** e24/50/74/99 LVEF probes: val MAE 7.54 / 6.41 / 6.11 / 6.05

### SALT S2 Evaluation — VALID across two implementation variants (revised 2026-04-07)

- [x] SALT v1 e79 probe: val MAE 6.47 → **test MAE 6.66, R²=0.414** (pred-avg, best SALT variant)
- [x] SALT v1 e199 probe: val MAE 6.73 → test MAE 7.02, R²=0.360 (extended training, slightly worse)
- [x] SALT v3 e79 probe: val MAE 6.84 → test MAE 7.03, R²=0.348 (paper-spec single-level predictor)
- [x] SALT severity gradient: collapses at 25% shuffle (v1 e79)
- [x] Effective dimensionality: SALT v1 e79 RankMe = 203 (JEPA/BYOL/MAE are 245/221/206 — **no dim collapse**)
- **✓ Results valid.** Earlier "invalidated / retrain" notes were based on a false claim that v1 used L2 loss. Both v1 and v3 configs have `loss_exp: 1.0` (L1, matching SALT paper Eq 2.1). The two variants differ in predictor architecture (hierarchical vs single-level) and hyperparameter regime, but the qualitative finding holds across both: **SALT underperforms all three EMA-based methods by 0.03–0.24 R² on EchoNet-Dynamic LVEF**. The gap is about teacher dynamics, not representational capacity.
- Full writeup: `claude/neurips/experiments/salt-comparison.md`
- Paper inclusion: one row in §3 comparison table + two-sentence framing in §4.5 (web Claude recommendation, 2026-04-07)

### Infrastructure

- [x] JEPA IN21K pretraining completed (job 376, e100)
- [x] SALT S2 v1 pretraining completed (jobs 388/391/392, e79/e100/e199) — hierarchical predictor, constant LR
- [x] SALT S2 v3 pretraining completed (job 446, e79) — single-level predictor, paper-spec HP regime
- [x] All checkpoints + probes uploaded to GDrive (`echo_foundation/nature_medicine/neurips/`)
- [x] Fetal US dataset downloaded (774 videos, `data/fetal_ultrasound/DatasetV3/`)
- [x] Allen Brain Observatory SDK set up (`allen` conda env)
- [x] Init confound discovered and documented (JEPA pt50 = fully-trained 235ep init)
- [x] EchoBench static perturbation framing issue documented
- [x] SALT implementation fully audited against paper — all fixes applied to configs
- [x] Noise autocorrelation sweep completed — hypothesis not supported (static noise worst, not iid). Demoted to appendix.
- [x] CAMUS segmentation probes at e100: MAE 0.827, BYOL 0.825, JEPA 0.815 (ranking inversion confirmed)
- [x] EchoBench LVEF at e100: JEPA −20%, BYOL −22%, MAE −51% (MAE collapses on functional tasks)
- [x] EchoBench CAMUS at e100: JEPA −10%, MAE −13%, BYOL −29% (JEPA most robust on both tasks)
- [x] Pediatric zero-shot at e100: JEPA Pearson=0.670, MAE 0.617, BYOL 0.500 (JEPA leads transfer)
- [x] Pediatric EchoBench at e100: JEPA −9%, MAE −8%, BYOL −19%
- [x] Speckle probing at e100: BYOL 0.716 < JEPA 0.848 < MAE 0.885 (ordering CHANGED from pt50 — init confound)
- [x] Layer-wise speckle probing: BYOL filters most across depth (−31%), MAE retains most (−4%)
- [x] Token-level speckle probing: MAE 0.941 > JEPA 0.926 > BYOL 0.891
- [x] Temporal consistency: BYOL 0.976 > JEPA 0.954 > MAE 0.950 (JEPA filtering hypothesis NOT supported)
- [x] Effective dimensionality: ⚠️ REVISED — JEPA 245, BYOL 221, MAE 206, SALT 203 (all 200-245 range, no collapse; prior MAE=63 not reproducible; `scripts/rebuttal/rankme.py`)
- [x] Model registry updated with e100 models (JEPA-IN21K-e100, BYOL-L-e100, MAE-L-e99)
- [x] VideoMAE token extraction fixed (hook on model.norm for pre-pooled tokens)

---

## Priority Stack (TODO)

### P0 — Required for paper (week 1-2)

| # | Experiment | Why | Compute | Depends On |
|---|-----------|-----|---------|-----------|
| ~~1~~ | ~~Noise autocorrelation sweep~~ | ~~Causal proof~~ | — | **DONE** — hypothesis not supported. Static noise worst. Demoted to appendix. |
| 2 | **5-task probes for JEPA IN21K e100** | Primary comparison table needs UHN LVEF, RVSP, Pediatric (CAMUS + END LVEF done) | ~1-2 days on HyperPod | JEPA IN21K e100 checkpoint (done) |
| 3 | **5-task probes for BYOL e100 + MAE e99** | Complete the 3-way on UHN LVEF, RVSP, Pediatric | ~1-2 days on HyperPod | Existing checkpoints (done) |
| ~~4~~ | ~~EchoBench at e100~~ | — | — | **DONE** — JEPA most robust on both LVEF (−20%) and CAMUS (−10%). |

### P1 — Strengthens paper (week 2-3)

| # | Experiment | Why | Compute | Depends On |
|---|-----------|-----|---------|-----------|
| ~~5~~ | ~~SALT retrain with corrected configs~~ | ~~Decision gate~~ | — | **NOT NEEDED (2026-04-07).** Earlier "invalidated" claim was based on a false L1/L2 assumption. v1 and v3 are both valid SALT variants with consistent results (R²=0.35–0.41, all below EMA baselines). See `experiments/salt-comparison.md`. Frees ~3 days HyperPod for 5-task probes or calcium imaging. |
| 6 | **Fetal US appendix** | Cross-anatomy transfer. Both tasks spatial → MAE leads both → confirms thesis. | ~2 hrs | Fetal dataset (downloaded) |
| 7 | **Speckle probing on e100 models** | §4.3 needs init-matched speckle partial R² | ~1 hr | e100 checkpoints (done) |

### P2 — Bonus if time permits (week 3-4)

| # | Experiment | Why | Compute | Depends On |
|---|-----------|-----|---------|-----------|
| 8 | **Calcium imaging viability test** | ViT-B pretrain on 1 session, 10ep. If non-degenerate → proceed. | ~2 hrs | Allen SDK (set up) |
| 9 | **Calcium imaging full** | Strongest second modality if it works. | ~1 week | Viability test passing |
| 10 | **BYOL/MAE e200 probes** | Insurance for "undertrained" reviewer concern. | ~1 day | e200 checkpoints (training on HyperPod) |

### Dropped

- ~~EchoJEPA-G scaling~~ — confounds prediction target with scale, conflicts with Nature Medicine
- ~~Training dynamics EchoBench~~ — running all 12 models through EchoBench is excessive for appendix material
- ~~6-condition on SALT~~ — SALT used as single-row probe only (web Claude conservative framing), full 6-condition sweep not needed
- ~~Frame-gap MAE intervention (ViT-B pilot)~~ — **cancelled 2026-04-08.** Our VideoMAE ViT-L was already pretrained with tube masking 90% (canonical Tong et al. 2022 recipe), which blocks cross-frame patch copying by construction. Yet MAE e99 is still invariant to frame shuffling (−4%, flat across all six conditions). The temporal shortcut must therefore arise from **within-frame spatial interpolation** (reconstructing a masked patch from its visible spatial neighbors at the same timestep), which neither tube masking nor frame-gap masking can prevent. The frame-gap experiment was testing a hypothesis the existing ViT-L run already refutes. Saves ~2 days HyperPod compute. The finding reframes §4 from "MAE is temporally flat" to "tube masking, the community-standard defense, fails — the prediction target is the bottleneck." See `experiments/tube-masking-failure.md` and `paper-outline.md` §4.6.

---

## What Each Paper Section Needs

| Section | What's done | What's missing |
|---------|------------|----------------|
| **§1 Intro** | Framing complete | Writing |
| **§2 Design** | 3-way comparison defined, init-matched | Writing |
| **§3 Core finding** | END LVEF + CAMUS + Pediatric ZS at e100. Ranking inversion confirmed (JEPA leads functional, MAE leads spatial). | **UHN LVEF, RVSP** (P0 #2-3, HyperPod) |
| **§4.1 6-condition** | Complete (12 models × 6 conditions) | Writing + Fig 2a |
| **§4.2 Severity gradient** | Complete (13 models × 5 fractions) | Writing + Fig 2b,c |
| **§4.3 Effective dim + speckle** | **REVISED.** Prior d_eff numbers retracted. Consistent 4-model RankMe: JEPA 245, BYOL 221, MAE 206, SALT 203 (all 200-245, no collapse). Demoted to appendix. Speckle + temporal consistency unchanged. | Writing |
| **§4.4 Autocorrelation sweep** | **DONE** — hypothesis not supported. Appendix result. | — |
| **§4.5 SALT** | **VALID across v1 and v3 variants.** SALT v1 e79 test R²=0.414 (best), v3 e79 R²=0.348. Both underperform EMA baselines (JEPA 0.652, BYOL 0.511, MAE 0.447). Conservative framing adopted: one row + two sentences. No retrain needed. | Writing (two-sentence framing in §4.5 per `salt-comparison.md`) |
| **§5 EchoBench** | **DONE** — LVEF + CAMUS at e100 init-matched. JEPA most robust on both. | Writing + figures |
| **§6 Discussion** | Framing complete | Writing |
| **Appendix** | Training dynamics, frame shuffling tables | Fetal US (P1 #6), figures |

---

## HyperPod Status

| Job | Model | Node | Status |
|-----|-------|------|--------|
| 376 | JEPA IN21K (e100) | — | **Complete** |
| 388 | SALT S2 v1 (e79) | — | **Complete, VALID** (hierarchical predictor variant) |
| 391 | SALT S2 v1 resume (e100) | — | **Complete, VALID** |
| 392 | SALT S2 v1 resume (e199) | — | **Complete, VALID** (marginal improvement) |
| 446 | SALT S2 v3 (e79) | — | **Complete, VALID** (single-level paper-spec variant) |
| 393 | BYOL (e108→e200) | 184 | Running |
| 394 | MAE (e124→e200) | 83 | Running |

---

## Immediate Next Actions

1. **Start noise autocorrelation sweep implementation** — modify `echo_perturbations.py` for per-frame noise with controllable temporal correlation. Run on JEPA IN21K e100 / BYOL e100 / MAE e99 with existing END LVEF probes. All 8 A100 GPUs free.

2. **Prepare 5-task probe configs for JEPA IN21K e100** — need configs for UHN LVEF (53K), RVSP (41K multi-view), CAMUS segmentation, Pediatric zero-shot. These can run on HyperPod once BYOL/MAE e200 jobs free up nodes.

3. ~~**Submit SALT S1+S2 retrain on HyperPod**~~ — **NOT NEEDED.** Existing v1 and v3 variants are both valid SALT implementations. Use v1 e79 (best test R²=0.414) as the primary SALT row in the §3 comparison table. See `experiments/salt-comparison.md` and `paper-outline.md` §4.5.
