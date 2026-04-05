# NeurIPS 2025 Execution Plan

**Last updated:** 2026-04-05
**Deadline:** May 4, 2026 (abstract), ~6 weeks
**Status:** Week 0 — infrastructure + initial experiments done

---

## Priority Stack

| Rank | Experiment | Why | Compute | Timeline |
|------|-----------|-----|---------|----------|
| **P0** | Noise autocorrelation sweep | Causal proof — turn inversion on/off with one parameter. Nobody has done this. | ~2 days, existing pipeline | Week 1 |
| **P0** | JEPA IN21K e100 probes | Primary comparison table needs init-matched JEPA | ~4 hrs on 8 GPUs | Week 1 (after job 376 finishes) |
| **P1** | SALT e200 evaluation | If clean → mechanistic probe (frozen teacher vs EMA). If noisy → cut. | Already queued on HyperPod | Week 2 decision gate |
| **P1** | Fetal US appendix | 1-day cross-anatomy transfer. Both tasks spatial → MAE leads both → confirms thesis. | ~2 hrs probe training | Week 1-2 |
| **P2** | Calcium imaging viability test | ViT-B pretrain on 1 session, 10ep. If features non-degenerate → proceed. | ~2 hrs on 1 GPU | Week 2-3 |
| **P2** | Calcium imaging full | Pretrain ViT-L on 5-10 sessions, eval segmentation + transient detection. | ~1 week | Weeks 3-4, hard kill at week 4 |
| **P3** | Extend BYOL/MAE to e200 | Removes "undertrained" concern. Resume from e108/e116 on S3. | ~1-2 days on HyperPod | After JEPA/SALT finish |

**Dropped:** EchoJEPA-G scaling section. G uses private 18M UHN data, confounds prediction target with 36× data + 4× params. Conflicts with Nature Medicine. One sentence in discussion instead. Reclaimed 0.5 pages for §4 mechanism.

---

## Current State (2026-04-05 evening)

### Running

| Where | Job | Progress | ETA |
|-------|-----|----------|-----|
| **A100 GPUs 5-7** | SALT S2 e29 LVEF probe | Epoch 6/20 | ~2.5 hrs |
| **HyperPod node 184** | JEPA IN21K e100 pretrain (job 376) | ~e80/100 | ~5 hrs |
| **HyperPod node 83** | SALT S2 e80→e100 (job 391) | Running | ~2 hrs |
| **HyperPod node 83** | SALT S2 e100→e200 (job 392) | Queued after 391 | ~13 hrs after 391 |

### Completed Today

- [x] Downloaded BYOL e24/e75/e100, MAE e24/e74/e99, SALT S2 e29/e49/e79 checkpoints
- [x] Trained 8 LVEF probes: BYOL e24/e75/e100, MAE e24/e74/e99, SALT S2 e79, SALT S2 e29 (in progress)
- [x] Ran frame shuffling severity gradient on BYOL e50/e100, MAE e50/e99, SALT S2 e79
- [x] Discovered init confound: JEPA pt50 init from fully-trained 235ep encoder
- [x] Discovered static perturbation framing issue in EchoBench
- [x] Discovered MAE temporal encoding is transient (invariant to shuffling at e99, not at e50)
- [x] Downloaded fetal US dataset (774 videos, `data/fetal_ultrasound/DatasetV3/`)
- [x] Set up Allen Brain Observatory SDK (`allen` conda env + Jupyter kernel)
- [x] Updated checkpoint reference, NeurIPS README, paper outline

### Key Results From Today

**Severity gradient (R² vs shuffle fraction):**

| Fraction | BYOL e50 | BYOL e100 | MAE e50 | MAE e99 | SALT S2 e79 |
|----------|----------|-----------|---------|---------|-------------|
| 0% | 0.427 | 0.468 | 0.141 | 0.445 | 0.293 |
| 25% | 0.360 | 0.410 | 0.091 | 0.421 | -0.037 |
| 50% | 0.278 | 0.336 | -0.103 | 0.436 | -0.277 |
| 75% | 0.220 | 0.300 | -0.271 | 0.414 | -0.382 |
| 100% | 0.219 | 0.291 | -0.301 | 0.428 | -0.397 |

**Training dynamics (Val MAE on EchoNet-Dynamic):**

| Model | e24 | e50 | e75 | e100 |
|-------|-----|-----|-----|------|
| BYOL | 6.37 | 6.17 | 5.99 | 5.94 |
| MAE | 7.54 | 6.41 | 6.11 | 6.05 |

**SALT S2 e79 LVEF probe:** Val MAE = 6.47 (weakest of all models)

---

## Week-by-Week Plan

### Week 1 (Apr 6-12): Core experiments

- [ ] **Noise autocorrelation sweep** — Implement per-frame noise with controllable temporal correlation in `echo_perturbations.py`. Sweep τ from ∞ (static) to 0 (iid). Run on JEPA/BYOL/MAE frozen encoders + existing END LVEF probes. Target: one figure showing R² vs noise autocorrelation for each model.
- [ ] **JEPA IN21K probes** — After job 376 finishes: download checkpoint, train END LVEF probe (8 GPUs, ~4 hrs), run severity gradient. This gives the init-matched primary comparison.
- [ ] **Fetal appendix experiment** — Freeze MIMIC-pretrained JEPA/BYOL/MAE encoders, write fetal dataloader (512×512 AVI → 224×224 clips), train AOP regression + segmentation probes. Expect MAE leads both (both tasks spatial).
- [ ] **SALT decision** — After SALT S2 e200 finishes on HyperPod: download, train END LVEF probe, run speckle probing. If frozen teacher → higher speckle partial R² than EMA → include. Otherwise cut.

### Week 2 (Apr 13-19): Primary results + second modality decision

- [ ] **Primary comparison table** — JEPA IN21K e100 / BYOL e100 / MAE e99 on all 5 tasks. Init-matched at ~100 total epochs.
- [ ] **SALT go/no-go** — Review e200 results. Decide inclusion.
- [ ] **Calcium imaging viability test** — Download 1 Cux2 session from Allen S3. Pretrain ViT-B MAE + JEPA for 10ep. Check: do features separate cells from background in a linear probe? If yes → proceed to full experiment. If no → kill.
- [ ] **Resume BYOL/MAE to e200** — Submit on freed HyperPod nodes. Insurance for "undertrained" reviewer concern.

### Week 3-4 (Apr 20 - May 3): Calcium imaging (if viable) + writing

- [ ] **Calcium imaging full** (if viability passed) — Download 5-10 sessions, pretrain ViT-L MAE + JEPA (~3 days each), evaluate segmentation + transient detection. **Hard kill at end of week 4 if results aren't clean.**
- [ ] **Start writing** — Sections 1-4 can be drafted now with existing results.
- [ ] **Figures** — Fig 1 (method diagram), Fig 2 (mechanistic panel), Fig 3 (ranking inversion), Fig 4 (noise robustness), Fig 5 (speckle probing)

### Week 5-6 (May 4-17): Writing + polish

- [ ] **Complete draft** — All sections, all figures, all tables
- [ ] **Internal review** — Check every claim against actual results
- [ ] **Appendix** — Training dynamics, fetal transfer, CAMUS per-structure, perturbation matrices
- [ ] **Submit**

---

## Decision Gates

### SALT (Week 2)
**Include if:** SALT S2 e200 speckle partial R² > MAE (confirms frozen teacher retains noise like MAE) AND SALT S2 e200 END LVEF val MAE < 6.0 (competitive with BYOL/MAE).
**Cut if:** Results noisy, SALT still worst by large margin, or speckle probing doesn't differentiate frozen vs EMA teacher.
**Fallback:** 3-way paper (JEPA/BYOL/MAE) with noise autocorrelation sweep as mechanistic centerpiece.

### Calcium Imaging (Week 4)
**Include if:** ViT-L pretrain converges, features are non-degenerate, segmentation + transient detection show ranking inversion (JEPA leads transient, MAE leads segmentation).
**Cut if:** Pretrain doesn't converge, features are degenerate, or no clear inversion.
**Fallback:** Paper with echo-only results + fetal appendix + noise autocorrelation sweep. Still strong.

---

## Files Created/Modified Today

| File | Description |
|------|-------------|
| `scripts/rebuttal/frame_shuffle_severity.py` | Severity gradient script, supports all 10 model configs |
| `scripts/rebuttal/frame_shuffle_segmentation.py` | P1.5b CAMUS segmentation under shuffling |
| `configs/eval/vitl/icml/echobyol_l_e{24,75,100}_end_lvef_d4.yaml` | BYOL training dynamics probe configs |
| `configs/eval/vitl/icml/echomae_l_e{24,74,99}_end_lvef_d4.yaml` | MAE training dynamics probe configs |
| `configs/eval/vitl/icml/echojepa_l_e100_end_lvef_d4.yaml` | JEPA e100 probe config (wrong init — do not use) |
| `configs/eval/vitl/icml/salt_s2_e{29,49,79}_end_lvef_d4.yaml` | SALT S2 probe configs at 3 checkpoints |
| `data/csv/echonet_dynamic_train_local_raw.csv` | Local-path training CSV for EchoNet-Dynamic |
| `data/fetal_ultrasound/DatasetV3/` | Fetal intrapartum US dataset (774 videos) |
| `claude/neurips/README.md` | Comprehensive update with all results and framing |
| `claude/neurips/paper-outline.md` | Updated with severity results, autocorrelation sweep, framing notes |
| `claude/neurips/plan.md` | This file |
| `claude/rebuttals/12-checkpoint-reference.md` | Full checkpoint inventory including S3 paths for all training runs |

## HyperPod Job Reference

| Job | Model | Node | Epochs | Status |
|-----|-------|------|--------|--------|
| 376 | JEPA IN21K | 184 | 0→100 | Running (~e80) |
| 388 | SALT S2 | 83 | 0→79 | **Done** (16 checkpoints on S3) |
| 391 | SALT S2 resume | 83 | 80→100 | Running |
| 392 | SALT S2 resume | 83 | 100→200 | Queued after 391 |
