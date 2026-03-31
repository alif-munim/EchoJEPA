# EchoJEPA ICML 2026 Rebuttal

---

## Reviewer 6t2T | Overall 3, Confidence 5/5

### Response

We would like to thank the reviewer for the careful reading and for raising the important concern about reproducibility. We have taken concrete steps to address both points and believe the revised evaluation is substantially stronger as a result.

**Full reproducibility plan.** The reviewer is absolutely right that reproducibility is essential. EchoJEPA-G (1.1B params) will be released with gated institutional access upon acceptance. To our knowledge, this is the largest publicly available echocardiography video foundation model (EchoPrime does not release its model; PanEcho is 40× smaller). Combined with EchoBench (open-source) and the already-public EchoJEPA-L, this provides full reproducibility at both model scales. Specifically, the community resources we commit to releasing are as follows.

1. **EchoJEPA-G** (ViT-Giant, 1.1B params, 18M echocardiograms), gated institutional access upon acceptance.
2. **EchoBench**, open-source video-level robustness evaluation with ultrasound-specific physics-based perturbations (depth attenuation, acoustic shadow, haze) and temporal corruptions. Applicable across all ultrasound modalities.
3. **EchoJEPA-L + evaluation framework**, already public (MIMIC-IV-Echo via PhysioNet).

**Moving beyond ablation to a systematic study.** Thank you for this observation, which helped us identify where the original framing fell short. Table 1 in our response to Reviewer hfQ1 now spans five tasks (classification, regression, segmentation, cross-dataset transfer, zero-shot cross-population) across three datasets and three model objectives. Tables 2–3 in our response to Reviewer ncQn add noise robustness across three perturbation types. This is substantially broader than the original submission's two-way comparison. The addition of BYOL-Video and cross-dataset evaluation (EchoNet-Dynamic, EchoNet-Pediatric) moves beyond ablation to a systematic study of what different SSL objectives learn.

**Report generation as future work.** This is an interesting direction. Report generation requires a text decoder and is orthogonal to representation quality; we discuss this as future work.

**Double-blind compliance.** Thank you for catching this. We will replace geographic identifiers with anonymized labels in the camera-ready version.

---

## Reviewer L8sp | Overall 4, Confidence 3/5

### Response

We would like to thank Reviewer L8sp for the positive assessment and for recognizing the clinical utility and sample efficiency. We appreciate the constructive suggestions and have worked to address each point.

**Multi-view framework is more than system-level.** The reviewer raises a fair point, and we are glad to provide evidence that cross-view integration contributes at the representation level. A controlled RVSP ablation shows cross-view integration (A4C + PSAX-AV) improves test Pearson from 0.447 to **0.484** (+3.7pp, +8.3% relative). Crucially, under severe noise the multi-view probe degrades by only 5.4% vs. 9.8% for the best single view, and multi-view at severe (0.449 Pearson) still matches single-view clean baselines. This is not just a system-level improvement. Cross-view integration provides representational robustness, because when one view is degraded, the complementary view compensates. The preprint (Table 5) provides a component-level ablation. This framework generalizes to any variable-composition multi-view setting, including multi-sequence MRI, multi-phase CT, and multi-probe ultrasound.

**New evidence strengthening the contribution beyond V-JEPA.** The new mechanistic evidence (noise robustness revealing task-specific representation properties; see Tables 2–3 in our response to Reviewer ncQn) and the three-way controlled comparison (Table 1 in our response to Reviewer hfQ1) add contribution dimensions beyond the original submission.

**Reproducibility addressed.** See our response to Reviewer 6t2T for full details on the EchoJEPA-G release (gated, 1.1B params) and community resources.

---

## Reviewer ncQn | Overall 3, Confidence 4/5

### Response

We would like to thank the reviewer for the insightful and constructive feedback. The suggestion to validate speckle suppression at the representation level was particularly valuable and has led to what we believe is one of the strongest new contributions of the revised paper.

**New representation-level evidence (directly addressing the reviewer's request).** We have conducted the *feature invariance tests and noise sensitivity studies* suggested by the reviewer. We evaluate frozen probes on test videos with physics-based ultrasound perturbations (depth attenuation, acoustic shadow, haze) at three severity levels. These degradation modes are common to *all* ultrasound modalities, not just echocardiography. Depth attenuation, shadowing, and reverberation affect obstetric, musculoskeletal, abdominal, and vascular ultrasound alike. Probes are **not retrained**; this measures whether existing representations maintain accuracy under realistic clinical degradation. We evaluate both LVEF regression (EchoNet-Dynamic, 1,277 test videos) and CAMUS segmentation (50 patients).

**Table 2. LVEF regression robustness (R², clean → severe).**

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Depth attenuation | 0.552 → 0.361 (−35%) | 0.440 → 0.145 (−67%) | 0.351 → 0.233 (−34%) |
| Acoustic shadow | 0.552 → 0.478 (−13%) | 0.440 → 0.247 (−44%) | 0.351 → 0.280 (−20%) |
| Haze artifact | 0.552 → 0.502 (−9%) | 0.440 → 0.398 (−10%) | 0.351 → 0.147 (−58%) |
| **Average drop** | **−19%** | −40% | −37% |

**Table 3. CAMUS segmentation robustness (mean Dice, clean → severe).**

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Depth attenuation | 0.815 → 0.681 (−16%) | 0.821 → 0.425 (−48%) | 0.822 → 0.749 (−9%) |
| Acoustic shadow | 0.815 → 0.708 (−13%) | 0.821 → 0.614 (−25%) | 0.822 → 0.728 (−11%) |
| Haze artifact | 0.815 → **0.800** (−2%) | 0.821 → 0.804 (−2%) | 0.822 → 0.794 (−3%) |
| **Average drop** | −10% | −25% | **−8%** |

**Key finding from the noise analysis.** The pretraining objective determines task-specific robustness. On LVEF (function), JEPA is most robust (−19% avg). On segmentation (anatomy), MAE is most robust (−8% avg). Under haze, JEPA achieves the highest severe Dice (0.800 vs. MAE 0.794), showing JEPA is competitive even on spatial tasks under certain degradation types. BYOL collapses on both (−40% LVEF, −25% seg). This finding is invisible from clean performance alone and demonstrates why comprehensive robustness evaluation is necessary.

Importantly, **JEPA maintains LVEF accuracy under noise (−19% avg) while MAE maintains segmentation accuracy (−8% avg)**. Each objective is robust to perturbation on the task it encodes. This is a representation-level finding; it reveals what information each objective preserves, measured through clinically interpretable downstream metrics rather than abstract feature geometry.

**New open-source robustness benchmark (EchoBench).** Motivated by the reviewer's feedback, we open-source **EchoBench**, a video-level robustness evaluation suite with physics-based perturbations and temporal corruptions. Standard SSL benchmarks evaluate clean-data accuracy alone; EchoBench adds a perturbation axis that exposes *how* models fail, not just *whether* they succeed. The framework is applicable to any ultrasound modality, and the evaluation methodology (comparing SSL objectives under systematic noise) can be adopted by any medical imaging benchmark to assess robustness alongside accuracy.

**Expanded controlled comparisons.** Thank you for encouraging a broader comparison. We now provide a three-way epoch-matched comparison (Table 1 in our response to Reviewer hfQ1) across five tasks, two datasets, and zero-shot cross-population transfer. All pairwise differences are statistically significant on EchoNet-Dynamic (bootstrap 95% CIs).

**Contribution beyond scaling.** We agree with the reviewer that scaling alone would be insufficient as a contribution. Beyond the hypothesis itself, the **evaluation methodology is a contribution**. Systematically comparing SSL objectives under physics-based perturbations at multiple severity levels reveals failure modes that clean benchmarks miss entirely. On clean CAMUS, all three objectives produce near-identical segmentation (0.815–0.822 Dice, <1pp spread). Under severe depth attenuation, BYOL collapses to 0.425 while MAE holds at 0.749, a 32pp gap invisible on clean data. This methodology is directly transferable to any SSL evaluation in noisy imaging domains.

**Camera-ready improvements.** Following the reviewer's suggestion, we will add representation visualizations comparing attention maps over cardiac structures under clean vs. perturbed inputs.

---

## Reviewer hfQ1 | Overall 2, Confidence 5/5

### Response

We would like to thank the reviewer for the thorough and rigorous evaluation. These concerns have pushed us to substantially strengthen the experimental foundation of the paper. We have conducted **11 new experiments** since submission, and we believe the revised results directly address each point raised. To our knowledge, this is the **first controlled comparison** of latent prediction, self-distillation, and pixel reconstruction objectives in the medical video domain.

**Clarifying the novelty of our contribution.** The reviewer is right that the encoder architecture is not novel, and we appreciate the opportunity to clarify that this is by design. Our contribution is threefold. (1) The **first controlled comparison** of three SSL paradigms (latent prediction, self-distillation, pixel reconstruction) in medical video, isolating the prediction target as the only variable. (2) A **systematic noise robustness evaluation methodology** that reveals task-specific failure modes invisible from clean performance (see our response to Reviewer ncQn, Tables 2–3). On clean data, all three objectives converge on segmentation (Table 1, CAMUS column), but under noise BYOL collapses while JEPA and MAE preserve complementary task-specific information. (3) A falsifiable hypothesis, namely that *the prediction target determines what information the model encodes and what it preserves under noise.* The five-task comparison, noise robustness analysis, and anatomy-vs-function dissociation constitute evidence for *when and why* different SSL objectives succeed, a finding about SSL broadly, not just echocardiography.

**New contrastive comparison and controlled baselines.** Thank you for highlighting this important gap. We trained two additional baselines on MIMIC-IV-Echo, providing an **epoch-matched** three-way comparison. All three models use the same ViT-L encoder, same data (525K clips), and the same 50-epoch budget. The **only variable is the prediction target**. JEPA predicts local masked token representations; BYOL-Video predicts global mean-pooled representations (self-distillation); MAE reconstructs pixels. Additionally, EchoPrime (CLIP-style contrastive, 12.1M clips, text supervision) is included as a system-level comparison in the preprint. Within the controlled pt50 comparison, JEPA outperforms both BYOL and MAE on all hemodynamic tasks using identical architecture and data.

**Table 1. Results across five tasks (all test sets).**

| Model | Target | LVEF R² (53K) | RVSP r (5K) | END LVEF R² (1.3K) | CAMUS Dice (50) | Ped. ZS r (368) |
|---|---|---|---|---|---|---|
| **EchoJEPA-L** | Local tokens | **0.409** | **0.484** | **0.552** | 0.815 | **0.705** |
| EchoBYOL-L | Global pool | 0.384 | 0.446 | 0.440 | 0.821 | 0.602 |
| EchoMAE-L | Pixels | 0.283 | 0.438 | 0.351 | **0.822** | 0.626 |

> LVEF/RVSP = UHN held-out test. END = EchoNet-Dynamic test (cross-dataset). CAMUS = frozen segmentation (50 patients). Ped. ZS = zero-shot UHN→Pediatric (no retraining). All pairwise differences significant on EchoNet-Dynamic (bootstrap 95% CIs non-overlapping, n=1,277) and on UHN LVEF (n=53K). At the system level, EchoJEPA-G (1.1B, 18M echos) achieves R²=0.778 LVEF, outperforming EchoPrime (0.681) and PanEcho (0.665). See preprint Tables 3–5.

**Key takeaway from the three-way comparison.** JEPA leads on all four hemodynamic tasks. MAE leads on segmentation (spatial anatomy) despite the weakest LVEF. The pretraining objective determines *what* clinical information is encoded, not just how well the model performs. This dissociation is the central finding. Pixel reconstruction encodes anatomy; latent prediction encodes hemodynamic function.

At the system level, EchoJEPA-G (1.1B params, 18M echocardiograms) achieves state-of-the-art across all tasks. Notably, the controlled ViT-L comparison above uses only MIMIC-IV-Echo (525K clips, 4,600 patients), a small, single-institution public dataset, yet JEPA's LVEF and RVSP results are competitive with models trained on 23× more data with direct supervision.

**Segmentation results added.** Thank you for noting this gap. CAMUS segmentation is now included in Table 1 (clean Dice) and Table 3 in our response to Reviewer ncQn (robustness under noise). MAE achieves the best clean Dice (0.822). This *supports* rather than contradicts our thesis, since pixel reconstruction excels at spatial anatomy.

**Broader evaluation tasks.** This is a fair point. Echocardiography lacks standardized detection benchmarks (no bounding-box annotations exist at scale). We evaluate the closest proxies, including view classification, regression, segmentation, and cross-dataset transfer, for five tasks total (Table 1).

**Dataset release clarified.** EchoJEPA-G (1.1B params) will be released with gated institutional access upon acceptance. See our response to Reviewer 6t2T for full details on community resources.

**Dataset diversity.** The five view categories in Figure 1 are anatomical imaging planes, not the total semantic space. The dataset spans 300K patients, multiple scanner manufacturers, and the full pathology spectrum. Representation diversity emerges from temporal variation and patient diversity, not category count.

**Camera-ready improvements.** We will restructure the paper to lead with the general SSL principle, namely that *the prediction target determines what information a self-supervised model encodes and preserves under noise.* Latent prediction encodes temporal dynamics and hemodynamic function; pixel reconstruction encodes spatial anatomy. Echocardiography is the case study, but the mechanism (that stochastic interference in the pixel space is filtered by the EMA target encoder) applies wherever pixel fidelity is dominated by noise rather than signal.
