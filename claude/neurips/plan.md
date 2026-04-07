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

### Training Dynamics Probes (appendix — DONE)

- [x] **JEPA IN21K** e25/50/75/100 LVEF probes: val MAE 6.48 / 5.55 / 5.27 / 5.22
- [x] **BYOL** e24/50/75/100 LVEF probes: val MAE 6.37 / 6.17 / 5.99 / 5.94
- [x] **MAE** e24/50/74/99 LVEF probes: val MAE 7.54 / 6.41 / 6.11 / 6.05

### SALT S2 Evaluation (preliminary — INVALIDATED by config bugs)

- [x] SALT S2 e79 probe: val MAE 6.47 (trained with wrong config: hierarchical output, wrong num_heads)
- [x] SALT S2 e199 probe: val MAE 6.73 (worse than e79 — plateau, but config was wrong)
- [x] SALT severity gradient: collapses at 25% shuffle
- **⚠️ ALL SALT results invalidated.** Config had: hierarchical output (not in SALT paper), pred_num_heads 12 vs 16, wrong LR/WD initially. Retrain required.

### Infrastructure

- [x] JEPA IN21K pretraining completed (job 376, e100)
- [x] SALT S2 pretraining completed (jobs 388/391/392, e200) — but with wrong config
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
- [x] Model registry updated with e100 models (JEPA-IN21K-e100, BYOL-L-e100, MAE-L-e99)

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
| 5 | **SALT retrain with corrected configs** | Decision gate: include as mechanistic probe or cut. All previous SALT results invalidated. | ~3 days S1+S2 on HyperPod | Corrected configs (done) |
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
- ~~6-condition on SALT~~ — SALT results invalidated, retrain first

---

## What Each Paper Section Needs

| Section | What's done | What's missing |
|---------|------------|----------------|
| **§1 Intro** | Framing complete | Writing |
| **§2 Design** | 3-way comparison defined, init-matched | Writing |
| **§3 Core finding** | END LVEF + CAMUS + Pediatric ZS at e100. Ranking inversion confirmed (JEPA leads functional, MAE leads spatial). | **UHN LVEF, RVSP** (P0 #2-3, HyperPod) |
| **§4.1 6-condition** | Complete (12 models × 6 conditions) | Writing + Fig 2a |
| **§4.2 Severity gradient** | Complete (13 models × 5 fractions) | Writing + Fig 2b,c |
| **§4.3 Speckle probing** | **DONE** at e100. Ordering changed: BYOL < JEPA < MAE (was JEPA < BYOL < MAE at pt50). Init confound. | Revise narrative |
| **§4.4 Autocorrelation sweep** | **DONE** — hypothesis not supported. Appendix result. | — |
| **§4.5 SALT** | Invalidated | **Retrain** (P1 #5) |
| **§5 EchoBench** | **DONE** — LVEF + CAMUS at e100 init-matched. JEPA most robust on both. | Writing + figures |
| **§6 Discussion** | Framing complete | Writing |
| **Appendix** | Training dynamics, frame shuffling tables | Fetal US (P1 #6), figures |

---

## HyperPod Status

| Job | Model | Node | Status |
|-----|-------|------|--------|
| 376 | JEPA IN21K (e100) | — | **Complete** |
| 388 | SALT S2 (e79) | — | **Complete** (config was wrong) |
| 391 | SALT S2 resume (e100) | — | **Complete** (config was wrong) |
| 392 | SALT S2 resume (e200) | — | **Complete** (config was wrong) |
| 393 | BYOL (e108→e200) | 184 | Running |
| 394 | MAE (e124→e200) | 83 | Running |

---

## Immediate Next Actions

1. **Start noise autocorrelation sweep implementation** — modify `echo_perturbations.py` for per-frame noise with controllable temporal correlation. Run on JEPA IN21K e100 / BYOL e100 / MAE e99 with existing END LVEF probes. All 8 A100 GPUs free.

2. **Prepare 5-task probe configs for JEPA IN21K e100** — need configs for UHN LVEF (53K), RVSP (41K multi-view), CAMUS segmentation, Pediatric zero-shot. These can run on HyperPod once BYOL/MAE e200 jobs free up nodes.

3. **Submit SALT S1+S2 retrain on HyperPod** — corrected configs ready. Queue after BYOL/MAE e200 finishes on node 83/184.
