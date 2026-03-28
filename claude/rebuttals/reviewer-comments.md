OpenReview.net
Search OpenReview...
Notifications1
Activity
Tasks
Alif Munim 
back arrowBack to the profile of Alif Munim
EchoJEPA: A Latent Predictive Foundation Model for Echocardiography
Download PDF
Alif Munim, Adibvafa Fallahpour, Teodora Szasz, Ahmadreza Attarpour, River Jiang, Brana Sooriyakanthan, Maala Sooriyakanthan, Heather M. Whitney, Jeremy Slivnick, Barry B Rubin, Wendy Tsang, BO WANG 
23 Jan 2026 (modified: 25 Mar 2026)
ICML 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors, Ethics Reviewers, Ethics Chairs
Revisions
CC BY 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: We introduce EchoJEPA, a foundation model for echocardiography that learns by predicting in latent space rather than reconstructing pixels, yielding superior sample efficiency, robustness, and generalization.
Abstract:
Foundation models for echocardiography promise to reduce annotation burden and improve diagnostic consistency by learning generalizable representations from large unlabeled video archives. However, current approaches fail to disentangle anatomical signal from the stochastic speckle and acquisition artifacts that dominate ultrasound imagery. We present EchoJEPA, a state-of-the-art foundation model for echocardiography trained on 18 million echocardiograms across 300K patients, the largest pretraining corpus for this modality to date. We also introduce a novel multi-view probing framework with factorized stream embeddings that standardizes evaluation under frozen backbones. Compared to prior methods, EchoJEPA reduces left ventricular ejection fraction estimation error by 19% and achieves 87.4% view classification accuracy. EchoJEPA exhibits strong sample efficiency, reaching 78.6% accuracy with only 1% of labeled data versus 42.1% for the best baseline trained on 100%. Under acoustic perturbations, EchoJEPA degrades by only 2.3% compared to 17% for the next best model, and transfers zero-shot to pediatric patients with 15% lower error than the next best model, outperforming all fine-tuned baselines. These results establish EchoJEPA as the new standard for modeling echocardiography.

Primary Area: Applications->Health / Medicine
Keywords: echocardiography, foundation models, self-supervised learning, V-JEPA, video representation learning, medical imaging
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission allows Policy B.
Proceedings-only Option: If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).
Reciprocal Reviewing Status: This submission is NOT exempt from the Reciprocal Reviewing requirement. (We expect most submissions to fall in this category.)
Reciprocal Reviewing Author:  Adibvafa Fallahpour
Submission Number: 29062
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
4 / 4 replies shown
Add:
Official Review of Submission29062 by Reviewer 6t2T
Official Reviewby Reviewer 6t2T13 Mar 2026, 04:42 (modified: 25 Mar 2026, 04:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Ethics Reviewers, Ethics Chairs, Reviewer 6t2TRevisions
Summary:
This paper presents EchoJEPA, an advancement in echocardiography foundation models that utilizes a JEPA to overcome the inherent noise and artifacts of US imaging. By training on a massive dataset of 18 million videos, the largest to date for this modality, the authors successfully demonstrate that latent prediction is superior to traditional pixel-reconstruction or contrastive methods. The methodology is technically sound and addresses a critical domain-specific challenge: the presence of stochastic speckle and acoustic shadowing that often leads reconstruction-based models to overfit on non-anatomical noise. The introduction of a novel multi-view probing framework with factorized stream embeddings is a particularly clever contribution, as it standardizes evaluation across diverse cardiac views without requiring view-specific components. This work establishes a new standard for echocardiographic modeling. By shifting the focus from pixel fidelity to representation quality, the authors provide a robust framework that is both clinically relevant and computationally efficient.

Strengths And Weaknesses:
Strenghts

The most significant strength is the shift from reconstructing raw pixels to predicting latent embeddings. By training the model to predict the abstract representation of masked areas rather than the noisy pixels themselves, the model successfully filters out random ultrasound "speckle" while focusing on stable, clinically relevant heart structures.
The introduction of factorized stream embeddings is a clever architectural choice. It allows the model to integrate data from multiple heart views efficiently without requiring a fixed number of views, making it robust to real-world clinical studies where certain views might be missing.
Weaknesses

While the authors provide a version of the model trained on public data for reproducibility, the headline "state-of-the-art" results are tied to a massive private dataset. This makes it difficult for other researchers to fully replicate or verify the flagship model's performance on the same scale.
There are no comparison with other state-of-the-art methods, especially in downstream tasks: classification and regression. The comparison is mainly on EchoJEPA-based methods, which looks like ablation studies but no direct comparison with other sequential methods
Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
How this model will help other downstream tasks, such as echocardiogram report generation or segmentation?
Limitations:
Yes.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Ethical Review Flag: Flag this paper for an ethics review.
Ethics Expertise Needed: Other Expertise
Ethical Review Concerns:
The internal dataset names suggest the affiliation of the authors, which can break double-blind submission rules.

Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission29062 by Reviewer L8sp
Official Reviewby Reviewer L8sp11 Mar 2026, 23:04 (modified: 25 Mar 2026, 04:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Ethics Reviewers, Ethics Chairs, Reviewer L8spRevisions
Summary:
This paper introduces EchoJEPA, a foundation model for echocardiography based on the Joint-Embedding Predictive Architecture (JEPA). EchoJEPA is pretrained on an extremely large dataset of 18M ECG videos from 300K patients, making it one of the largest ultrasound corpora used for self-supervised learning. The paper also introduces a multi-view probing framework with factorized stream embeddings, designed to evaluate models across incomplete clinical studies where certain ultrasound views may be missing.

Strengths And Weaknesses:
Strengths
The scale of pretraining (18M videos) is a major contribution and represents a significant step toward foundation models for echocardiography. Large-scale pretraining is still rare in medical imaging, making this dataset and training effort particularly impactful.
The authors conduct a controlled comparison between EchoJEPA-L and a compute-matched baseline trained with the VideoMAE objective. This “apples-to-apples” comparison clearly shows that latent prediction significantly outperforms pixel reconstruction for ultrasound data.
The results demonstrate clinical utility, including, Accurate Task specific estimation and robust to perturbation.
Strong sample efficiency (competitive results with only 1% labels)
The paper is well written and organized.
Weaknesses
The core architecture is largely based on V-JEPA. Although, the main contribution here is the application and scaling of the approach to ECG, along with domain-specific adjustments such as temporal resolution and geometry. These changes are incremental.

Multi-View Probe Contribution: The proposed multi-view attentive probe is practical for handling incomplete clinical studies, but its technical novelty is limited. The design relies on standard attention mechanisms, making it more of a system-level improvement than a new methodological contribution.

Reproducibility Concerns: The strongest results (EchoJEPA-G) are trained on a large proprietary dataset that cannot be released. Although the authors provide a model trained on the public MIMIC‑IV‑Echo dataset, the community cannot reproduce the flagship results using the full dataset.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 2: fair
Key Questions For Authors:
See Weakness

Limitations:
The authors clearly acknowledge two important limitations:

Dependence on proprietary training data.
Synthetic perturbations used in robustness experiments instead of real difficult-to-image clinical cases.
Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission29062 by Reviewer ncQn
Official Reviewby Reviewer ncQn11 Mar 2026, 21:53 (modified: 25 Mar 2026, 04:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Ethics Reviewers, Ethics Chairs, Reviewer ncQnRevisions
Summary:
EchoJEPA is proposed in this paper as a foundation model for echocardiography video representation learning. The method builds on the V-JEPA framework and uses a latent prediction objective instead of pixel reconstruction for self-supervised pretraining. The model extracts representations from visible video tubelets through a context encoder, then uses a predictor to predict the latent embeddings of masked regions, where the prediction targets come from an EMA target encoder. The training objective minimizes the L1 loss between predicted embeddings and target embeddings. The authors pretrain on a large-scale echocardiography dataset and propose a multi-view attentive probing framework for downstream task evaluation, such as LVEF estimation and view classification. Experimental results show that under the same architecture, dataset, and compute budget, EchoJEPA-L outperforms EchoMAE-L across multiple tasks.

Strengths And Weaknesses:
Strength:

Large-scale dataset The paper pretrains on 18M echocardiography videos from ~300K patients, which is a very large data scale in medical imaging and helps learn more generalizable representations.

Practical relevance for medical imaging The method is adapted for ultrasound signal characteristics such as speckle noise and cardiac temporal dynamics, and has practical application value in the direction of echocardiography foundation models.

Weakness:

The proposed method largely follows the existing V-JEPA[1] framework and applies it to the echocardiography domain. The overall architecture and the latent prediction objective remain essentially unchanged from prior work. As a result, the main contribution lies more in scaling the approach to a medical dataset.

The paper argues that the performance improvement mainly comes from the ability of latent prediction to suppress speckle noise and thus learn better anatomical representations. However, this causal chain is not directly validated in the experiments. The paper mainly shows improvements on downstream tasks (e.g., LVEF regression and view classification), but does not provide representation-level analyses to demonstrate that the learned embeddings are indeed more robust to speckle noise or better aligned with anatomical structures. Without additional analyses such as feature invariance tests, representation visualization, or noise sensitivity studies, it remains unclear whether the gains truly arise from noise suppression or from other factors such as dataset scale or temporal modeling.

Although the paper attempts a compute-matched comparison between EchoJEPA-L and EchoMAE-L, this appears to be the only strictly controlled baseline where architecture, data, and compute are identical. Other comparisons involve models with different architectures, training paradigms, or significantly larger proprietary datasets, making it difficult to draw strong conclusions about the advantage of the latent prediction objective and Multi-view probing.

[1] Assran, Mido, et al. "V-jepa 2: Self-supervised video models enable understanding, prediction and planning." arXiv preprint arXiv:2506.09985 (2025).

Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Since the method largely follows the existing V-JEPA framework, could the authors clarify what the main methodological novelty is beyond scaling the approach to a large echocardiography dataset?

The paper attributes performance gains to latent prediction suppressing speckle noise, but this is mainly supported by downstream task improvements. Do the authors have representation-level analyses that directly verify this claim?

The strictly controlled comparison appears limited to EchoJEPA-L vs EchoMAE-L. Could the authors provide additional controlled comparisons to better demonstrate that the improvements come from the latent prediction objective and multi-view probing, rather than specific experimental settings?

Limitations:
yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission29062 by Reviewer hfQ1
Official Reviewby Reviewer hfQ120 Feb 2026, 01:58 (modified: 25 Mar 2026, 04:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Ethics Reviewers, Ethics Chairs, Reviewer hfQ1Revisions
Summary:
This paper introduces a foundational echocardiography model based on the JEPA policy. The authors prepared a large-scale dataset (comprising 1.8 million videos from 300,000 individuals) and trained EchoJEPA upon it. Furthermore, they proposed a unified probe for evaluating foundational model representations, based on cardiac ultrasound cross-sectional video frame sampling, to assess EchoJEPA's representation-extraction capability. They also assessed the model's resilience to adversarial perturbations and its performance on downstream task evaluations.

Strengths And Weaknesses:
Strengths:

Employing the JEPA strategy for pre-training foundational models in echocardiography visualisation is a sound approach, yielding richer semantic representations and enhanced trainability compared to the MAE strategy.
Proposing a novel dataset holds significant value for the community, likely to garner widespread attention.
The authors' comparative fairness analysis of foundational models and their evaluation using physically grounded adversarial perturbations tailored to ultrasound applications represent novel contributions. These constitute critical considerations for ultrasound vision foundational models.
Weaknesses:

While the idea of leveraging JEPA for echocardiography foundation model training is encouraging, the technical contribution appears limited. The proposed method is largely built upon V-JEPA2, with only marginal architectural modifications. From a methodological standpoint, the level of technical innovation does not meet the high bar typically expected at ICML.
The authors do not clearly justify why JEPA should be preferable to MAE for constructing echocardiography foundation models. In addition, no comparisons are provided between JEPA and contrastive pretraining approaches in this domain. The vague motivation, unclear positioning, and insufficient baseline comparisons significantly weaken the claimed contributions.
The downstream evaluation protocol of EchoJEPA is somewhat confusing. Typically, a vision foundation model is expected to demonstrate general-purpose visual capabilities across a range of downstream tasks, such as classification, detection (if applicable), segmentation, and generation. In echocardiography, segmentation datasets are widely available with well-established baselines. Potential readers would reasonably expect evaluation on these canonical visual tasks. However, the current study only evaluates classification and regression tasks, which limits the assessment of general visual representation quality.
Data plays a crucial role in foundation model development, yet it remains unclear whether the dataset will be publicly released. If public release is intended, no ethical approval documentation is provided. If the dataset cannot be released, this would further diminish the practical and community-level impact of the work.
The design of the Multi-view Probing Framework is somewhat confusing. The Attentive Probe appears to be used solely for LEVF evaluation, which is essentially a regression task. While this may demonstrate certain aspects of representation strength, foundation model representations are typically expected to support broader visual analysis tasks (e.g., classification and segmentation). The current probing design feels narrow and somewhat misaligned with the stated goal of constructing a vision-centric foundation model.
Overall, although the motivation of the work is reasonably clear, many technical details lack sufficient explanation and clarification. Conceptually, applying JEPA to the echocardiography foundation model construction is a worthwhile direction. However, from a technical standpoint, the current work does not yet meet the acceptance standards of ICML.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Have the authors discussed the role of contrastive learning and other self-supervised learning strategies in constructing echocardiography foundation models? At present, the evaluation is limited to a comparison with MAE-style pretraining, which appears insufficient to justify the choice of JEPA over alternative self-supervised paradigms.
The authors should consider including a broader range of visual analysis tasks to more comprehensively demonstrate the advantages of EchoJEPA over existing vision foundation models. A wider downstream evaluation would help substantiate the claimed generalization and representation benefits.
The authors claim that EchoJEPA is trained on a relatively large-scale custom dataset. However, it is generally understood that in self-supervised pretraining, diversity in semantic category space often contributes more significantly than mere dataset size expansion. Although video is the primary modality in echocardiography, as illustrated in Figure 1, the dataset appears to contain only five categories despite its scale. This aspect requires further clarification and justification regarding how representation diversity is ensured.
Limitations:
The author discussed the limitations in the paper.

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Ethics Expertise Needed: Responsible Research Practice (e.g., IRB, documentation, research ethics)
Ethical Review Concerns:
If the author considers making the dataset public, supporting documents for ethical review should be provided.

Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
About OpenReview
Hosting a Venue
All Venues
Contact
Sponsors
Donate
FAQ
Terms of Use / Privacy Policy
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

EchoJEPA: A Latent Predictive Foundation Model for Echocardiography | OpenReview