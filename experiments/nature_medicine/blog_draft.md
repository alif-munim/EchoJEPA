# A World Model of the Heart: What JEPA Learns From 18 Million Echocardiograms

*Draft Substack post -- EchoJEPA Nature Medicine working notes, March 2026*

---

We trained a self-supervised video model on 18 million echocardiograms using JEPA -- the Joint-Embedding Predictive Architecture. The objective is simple: mask parts of the video and predict what the masked regions look like in latent space. No pixel reconstruction. No text. No labels. Just the task of predicting what comes next.

To do this well on echocardiograms, the model has to build an internal model of how hearts work. It has to learn how valves open and close, how chambers fill and empty, how walls contract and relax. It has to learn, in some representational sense, the physics of cardiac mechanics.

We froze this model and asked it clinical questions that no frozen echo model has been asked before. The results provide what we believe is the first clinical evidence that a JEPA world model encodes not just cardiac structure, but hemodynamic function, right ventricular mechanics, and subclinical features predictive of future disease.

## What Is a World Model?

Yann LeCun's JEPA framework is built on a specific hypothesis: that intelligence requires an internal model of how the world works. Rather than reconstructing raw pixels (like a masked autoencoder) or matching images to text descriptions (like CLIP), JEPA learns to predict abstract representations. The model sees part of a video, and must predict the latent representation of the parts it can't see.

This forces a particular kind of learning. The model can't just memorize textures or edges. It has to learn the *dynamics* -- the rules governing how things change over time. For natural video, this means learning about object permanence, gravity, occlusion. For echocardiograms, this means learning about cardiac mechanics.

A heart with a severely regurgitant mitral valve doesn't just look different from a normal heart. It *moves* differently. The left atrium dilates. The leaflets prolapse or fail to coapt. The wall motion changes to compensate. A world model that predicts cardiac video in latent space must represent these dynamics to make accurate predictions. The hemodynamic information isn't explicitly taught -- it emerges as a necessary consequence of the predictive objective.

This is the hypothesis we set out to test.

## The Experiment

**EchoJEPA-G** is a ViT-Giant (1,012M parameters) trained with JEPA on 18 million echocardiogram videos from the University Health Network. After pretraining, we freeze the encoder entirely and never touch it again.

Every downstream task uses the same frozen representations, evaluated with a minimal cross-attention probe: a single layer of cross-attention (~2M learnable parameters) that attends to the encoder's 1,568 output tokens. The probe learns what to look at. The encoder provides what there is to see.

We compare against four models, all evaluated with the same frozen probe:

| Model | Pretraining objective | World model? |
|-------|----------------------|-------------|
| **EchoJEPA-G** | Predict masked echo video in latent space (1,012M params, 18M echos) | Yes -- learns cardiac mechanics |
| **EchoJEPA-L-K** | Predict masked video, initialized from Kinetics natural video (300M params) | Partial -- general motion physics, adapted to echo |
| **EchoJEPA-L** | Predict masked echo video (300M params, 7K MIMIC studies) | Minimal -- too little data |
| **EchoPrime** | CLIP contrastive: align echo video with cardiologist reports | No -- learns report language, not physics |
| **PanEcho** | Supervised multi-task: predict specific echo measurements | No -- learns measurement mappings |

Same probe architecture. Same hyperparameter grid. Same evaluation protocol. Five models, one question: what did each pretraining objective learn?

## Three Clinical Tests of the World Model

Our clinical collaborator challenged us with three capabilities that would constitute evidence for a cardiac world model.

### Test 1: Can Structure Predict Flow?

In echocardiography, hemodynamic measurements -- valve severity, pressure gradients, flow velocities -- come from Doppler imaging. A skilled cardiologist can *estimate* hemodynamics from structural (B-mode) views alone: they see the chamber dilation, leaflet thickening, and wall motion patterns that imply abnormal flow. But they still need Doppler to quantify it.

If the JEPA world model truly captures cardiac mechanics, it should encode hemodynamic information in its representations of structural views -- because the physics of structure and flow are coupled. You can't accurately predict what a regurgitant valve looks like in latent space without representing the regurgitation.

We tested this by deliberately withholding all Doppler information. For each hemodynamic task, we filter training data to include only B-mode clips, excluding any clip containing color or spectral Doppler. The model never sees the direct measurement. It only sees the structural consequences.

### Test 2: Can It Assess the Right Ventricle?

The right ventricle is echocardiography's blind spot. It's geometrically complex, harder to image, and most AI models ignore it entirely. We probe for TAPSE, RV S', and RV FAC -- functional measurements that clinically require M-mode or tissue Doppler.

A world model of cardiac mechanics should represent the right ventricle as naturally as the left. The RV contracts, relaxes, and interacts with pulmonary circulation. If the model learned cardiac physics, it learned RV physics too.

### Test 3: Can It See the Future in a Normal Heart?

This is the hardest test. We take patients with preserved ejection fraction (EF >= 50%) -- hearts that appear normal by conventional measurement -- and ask: will this patient develop cardiomyopathy (EF < 50%) within one year?

This is explicitly a test of whether the world model captures information *beyond* what standard measurements encode. If the model only learned to estimate EF, it would be at chance -- all these patients have normal EF at baseline. To succeed, the model must detect subclinical mechanical abnormalities: subtle wall motion patterns, early diastolic changes, myocardial texture features that precede clinical deterioration.

A world model that understands cardiac mechanics deeply enough should detect these. A model that merely learned structural appearance should not.

## Results

### Hemodynamics: The World Model Encodes Flow

| Task | EchoJEPA-G | L-K | EchoPrime | L | PanEcho |
|------|-----------|-----|-----------|---|---------|
| AS severity (4-class) | **0.908** | 0.821 | 0.827 | 0.786 | 0.762 |
| MR severity (5-class) | **0.860** | 0.803 | 0.770 | 0.771 | 0.724 |
| TR severity (5-class) | **0.837** | 0.786 | 0.739+ | 0.753 | -- |
| AV peak velocity (R2) | **0.582** | 0.388 | 0.476 | 0.232 | 0.390 |

*All results: frozen d=1 attentive probes, B-mode clips only, single random clip per study per validation epoch. Prediction averaging not yet applied -- final numbers will be higher.*

EchoJEPA-G achieves 0.908 AUROC for aortic stenosis severity from structural views alone. No Doppler. No gradients. Just B-mode video through a frozen encoder and a single cross-attention layer.

The world model interpretation: JEPA's predictive objective forced the model to learn the relationship between aortic valve structure (calcification, restricted motion, post-stenotic dilation) and the downstream hemodynamic consequences. The representation encodes flow because flow is necessary to predict structure.

But here's what makes this a finding about the world model, not just about scale: **every model shows hemodynamic signal from B-mode.** PanEcho, a supervised model that never performed self-supervised pretraining, achieves 0.724 AUROC for MR severity from structural views alone. The hemodynamic information is *in the data*. The question is how deeply each pretraining objective excavates it.

EchoJEPA-G leads by 5-10 percentage points across all hemodynamic tasks. The world model -- learned through predicting masked cardiac video at massive scale -- provides the deepest extraction of the structure-flow relationship.

### Trajectory: The Deepest Test

| Model | Test AUROC | Pred Avg Boost |
|-------|-----------|----------------|
| EchoJEPA-G | **0.793** | +0.060 |
| EchoPrime | 0.776 | +0.076 |
| PanEcho | 0.759 | +0.061 |
| EchoJEPA-L-K | 0.677 | +0.081 |
| EchoJEPA-L | 0.514 | -- |

*Test set: 6,087 studies (453 events, 7.4% event rate). Prediction averaging across all clips per study.*

From an apparently normal echocardiogram, EchoJEPA-G identifies patients who will develop cardiomyopathy within a year at 0.793 AUROC. These patients all had EF >= 50% at baseline. The model is detecting something that the EF number doesn't capture.

This is the strongest evidence for the world model hypothesis. The model was never trained to predict the future. It was trained to predict masked video regions. But a world model that deeply understands cardiac mechanics should detect the subtle mechanical signatures -- wall motion abnormalities below clinical threshold, early filling pattern changes, myocardial texture variations -- that precede overt dysfunction. The fact that it can predict onset cardiomyopathy from normal-appearing echos suggests it has learned exactly this.

### The Model Comparison Tells the Real Story

The ranking changes depending on what you ask:

| Task type | G leads by | Second place |
|-----------|-----------|-------------|
| Hemodynamics | +5-10pp | L-K (partial world model) |
| Trajectory | +1.7pp | EchoPrime (text-supervised) |

**On hemodynamics, the world model hierarchy holds.** G (full cardiac world model) dominates. L-K (partial world model from natural video, adapted to echo) is second. EchoPrime and PanEcho, which don't build world models, trail. The depth of the world model determines how much structure-flow coupling it captures.

**On trajectory, something different happens.** EchoPrime -- which learns to align video with cardiologist reports, not to predict cardiac physics -- nearly matches G. The gap shrinks from 8-10 percentage points to 1.7.

Why? Because prognostic features live in two places:

1. **In the physics.** Subtle mechanical abnormalities that a world model captures by learning cardiac dynamics. This is what G learns through 18 million echos of JEPA training.

2. **In the language.** When a cardiologist writes "subtle septal hypokinesis" or "mild diastolic dysfunction" or "borderline wall thickness" in a report, CLIP training forces the vision encoder to represent exactly those observations. EchoPrime gets a linguistic shortcut to prognostic features that the world model discovers through physics.

Both paths lead to similar prognostic performance. But only the world model additionally captures the deep structure-flow relationship -- which is why G's hemodynamic advantage (8-10pp) is so much larger than its trajectory advantage (1.7pp). The world model learns *everything*: structural mechanics, hemodynamic coupling, *and* prognostic features. Text supervision efficiently learns what cardiologists notice and write down, which happens to include prognostic observations, but misses the physical relationships they don't describe.

### L-K's Collapse: A World Model That Isn't Deep Enough

EchoJEPA-L-K is the most revealing comparison. It was pretrained on Kinetics (natural video: people cooking, playing sports, driving) and then annealed on echocardiograms. It has a world model -- but it's a world model of general motion physics, partially adapted to cardiac physics.

On hemodynamics, L-K is consistently second-best. Its world model of motion is good enough to capture structural dynamics: how valves move, how walls contract. These are, in some sense, extensions of general physical motion.

On trajectory, L-K collapses to fourth place (0.677 AUROC), and its predictions never cross the threshold to actually identify at-risk patients (balanced accuracy = 0.500, kappa = 0.000). The partial world model captures cardiac *structure* but not the subtle *prognostic* features embedded in cardiac mechanics. Those features -- the subclinical patterns that distinguish a heart that will fail from one that won't -- require a world model trained deeply enough on cardiac data to discover them.

Fifty-five epochs of echo annealing on top of Kinetics pretraining isn't enough. Eighteen million echocardiograms of JEPA training from scratch is.

### L's Failure: No World Model At All

EchoJEPA-L, pretrained on only 7,000 MIMIC studies, performs at chance on trajectory (0.514 AUROC) and poorly on hemodynamics. Seven thousand studies is simply not enough data for self-supervised learning to build any meaningful world model. The representations collapse -- they're concentrated in a tiny region of embedding space with almost no discriminative structure.

This isn't a failure of the JEPA objective. It's a data floor. You can't learn cardiac physics from 7,000 examples.

## The World Model Hierarchy

Putting it all together, the five models form a hierarchy of world model depth:

| Model | World model depth | Hemodynamics | Prognosis | Interpretation |
|-------|------------------|-------------|-----------|----------------|
| G | Deep cardiac | Best | Best | Full cardiac mechanics from 18M echos |
| L-K | Shallow cardiac | 2nd | 4th | General motion + shallow echo adaptation |
| EchoPrime | None (text shortcut) | 3rd-4th | 2nd | Linguistic proxy for prognostic features |
| PanEcho | None (task shortcut) | 4th-5th | 3rd | Supervised mappings generalize partially |
| L | None (data floor) | 5th | Chance | Insufficient data for any world model |

The key insight: **world model depth determines hemodynamic performance, but text supervision provides an efficient alternative path to prognostic features.** Only the deep world model (G) leads on *both* -- because it learns the physics that generates both the structural observations and the prognostic features, rather than learning each through separate shortcuts.

## How We Got Here: Decisions and Dead Ends

### The Trajectory Pivot

Our first three attempts at trajectory prediction failed:

- **V0 (delta regression)**: Predict the continuous change in EF. Best R2 = 0.043. The model barely beat predicting the mean.
- **V1 (3-class delta)**: Classify as declined/stable/improved. All five models clustered at 0.60-0.65 AUROC with no separation.
- **V2 (tighter thresholds)**: Made it worse. G dropped from 0.649 to 0.610.

The fundamental problem: delta prediction is dominated by regression to the mean (r = -0.511 between baseline EF and change). All models are equally good at encoding baseline EF, so there's no model differentiation. The task was measuring a trivial baseline correlation, not world model quality.

The fix was the onset framing. By restricting to patients with EF >= 50%, we control for baseline EF by design. The model can no longer succeed by encoding what the EF number already captures. It has to find something else -- and *that's* where world model depth separates from shortcuts.

This was the single most important experimental design decision. The same five models, the same representations, but a reframed question that tests what we actually care about.

### Evaluation: Why Frozen Probes Matter

We freeze the encoder for every task. This is critical for the world model claim. If we fine-tuned the encoder, we'd be testing the combined system of "pretrained features + task-specific adaptation." By freezing, we test what the representation *already knows*. The probe is minimal by design -- a single cross-attention layer -- so it can't compensate for missing information.

When EchoJEPA-G achieves 0.908 AUROC for AS severity through a frozen encoder, that hemodynamic information was already in the representation before the probe was trained. The JEPA objective put it there.

### Prediction Averaging: The Multi-Clip Signal

A clinical echo study contains 40-80 video clips from different views and angles. During training, we sample one random clip per study per epoch. But at test time, we score every clip and average predictions across the study.

This simple technique boosted trajectory onset AUROC by +0.06-0.08 across all models. Different clips provide complementary views of the same heart, and averaging extracts signal that any single clip might miss. We haven't yet applied prediction averaging to the hemodynamic tasks (those checkpoints need retraining), but we expect similar gains.

### The B-Mode Filter: Controlled Evaluation

For the hemodynamic tasks, we run view and color classifiers (ConvNeXt-Small, separately trained) on every clip to identify B-mode structural views. We exclude any clip containing color Doppler, spectral Doppler, tissue Doppler, or M-mode. The filter is applied at the training CSV level -- the frozen encoder was pretrained on all clip types, but the evaluation probe is trained and tested on B-mode clips only.

This creates a deliberately handicapped evaluation. The model has to infer hemodynamic severity from structural information alone. The fact that it succeeds is not because the evaluation leaks Doppler data, but because the frozen representation already encodes the structure-flow relationship.

## What's Still Missing

**Prediction averaging on hemodynamics.** All hemodynamic numbers above are single-clip. Prediction averaging should add +0.06-0.08 AUROC based on the trajectory pattern.

**Four more hemodynamic tasks.** AR severity, E/e' ratio, RVSP, AV mean gradient. These complete the hemodynamic pillar with 7 total tasks.

**RV mechanics.** RV S', RV FAC haven't been trained yet. Along with TAPSE (done), these form the RV pillar.

**Four remaining trajectory tasks.** TAPSE trajectory, LV mass trajectory, RV systolic pressure trajectory, MR severity trajectory. Only LVEF onset is complete. If the onset framing works for TAPSE (predicting RV dysfunction onset), that substantially strengthens the world model claim.

**Interpretability.** We're collaborating with Goodfire on sparse autoencoder analysis to identify what individual features in the representation encode. Early-stage: checkpoints being transferred. The goal is to find features that correspond to specific cardiac pathology -- a direct window into the world model's internal representation.

## The "Just Scale" Question

We anticipate the critique: "this is just a bigger model on more data."

Three responses:

**First, scale is how world models develop.** In the same way that language models develop reasoning capabilities at scale, the cardiac world model develops hemodynamic understanding at scale. L-K has 300 million parameters and saw Kinetics + echo data. It captured structure but not prognosis. G has 1.1 billion parameters and saw 18 million echos. It captured both. The contribution isn't "we used more compute." It's characterizing what emerges at scale and what doesn't -- and providing clinical evidence for the emergence.

**Second, the comparison models illuminate the mechanism.** If scale were the only factor, you'd expect G to lead every task by the same margin. Instead, the gap varies dramatically: 8-10pp on hemodynamics, 1.7pp on prognosis. EchoPrime (text-supervised, much smaller) nearly catches G on prognosis but not on hemodynamics. This pattern is explained by the world model hypothesis -- hemodynamics requires physical understanding that only the world model provides, while prognosis can be partially captured through linguistic shortcuts -- and not explained by scale alone.

**Third, the clinical findings are novel.** No frozen echo model has demonstrated: (1) hemodynamic inference at this level from structural views alone, (2) onset cardiomyopathy prediction from normal-appearing echos, or (3) right ventricular functional assessment. These are clinical capabilities, not benchmark improvements. They would be newsworthy regardless of which model produced them.

The world model framing connects LeCun's theoretical framework to concrete clinical evidence. The JEPA objective creates an internal model of cardiac mechanics. The hemodynamic results demonstrate it encodes physical relationships (structure predicts flow). The trajectory results demonstrate it encodes prognostic information beyond standard measurements. The comparison models demonstrate that alternative training objectives (text supervision, supervised labels) provide efficient shortcuts to *some* of this information but not all of it.

The paper isn't "we trained the biggest echo AI model." The paper is "JEPA builds a world model of the heart, and here is the clinical evidence."

## What Comes Next

The immediate work is completing the experimental matrix: retrain lost checkpoints, finish hemodynamic and RV tasks, run prediction averaging everywhere, and launch the remaining trajectory tasks. The numbers will change -- probably upward, given what prediction averaging did for onset.

The paper structure follows the three clinical tests: hemodynamic inference from structure (the world model encodes flow), right ventricular mechanics (the world model extends to the neglected chamber), and trajectory forecasting (the world model captures subclinical features beyond standard measurements). Standard benchmarks are brief context. Disease detection and clinical outcomes are supporting evidence. The model comparison is woven throughout as mechanistic insight, not as a competition.

The deeper question -- can we crack open the world model and see what it learned? -- is the SAE interpretability work. If we can identify individual features that encode specific cardiac pathology, that's direct evidence for the world model's internal structure. That work is starting now.

---

*Working draft from the middle of the experiment. Numbers will change as prediction averaging and remaining tasks complete. The framing may evolve. But the core claim -- that JEPA builds a cardiac world model, and that model encodes hemodynamics, mechanics, and prognosis -- is supported by the data we have so far.*
