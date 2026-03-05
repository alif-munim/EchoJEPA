# Encoder Comparison Fairness Analysis

## Encoder Output Summary

The `modelcustom/` encoder adapters produce the same `[B, N, D]` interface but differ significantly in what they output:

| Encoder | Module | Backbone | Params | `embed_dim` | Tokens per clip | Output type |
|---------|--------|----------|--------|-------------|-----------------|-------------|
| EchoJEPA-G | `vit_encoder_multiclip` | ViT-g | ~1B | 1408 | 14×14×8 = 1568 | Full spatial-temporal token grid |
| EchoJEPA-L | `vit_encoder_multiclip` | ViT-L | ~300M | 1024 | 14×14×8 = 1568 | Full spatial-temporal token grid |
| VideoMAE | `videomae_encoder` | ViT-L | ~300M | 1024 | Full token grid | Full spatial-temporal token grid |
| PanEcho | `panecho_encoder` | ConvNeXt-T + Transformer | ~30M | 768 | **1** (pre-pooled) | Single global embedding |
| EchoPrime | `echo_prime_encoder` | MViT-v2-S | ~35M | 512 | **1** (pre-pooled) | Single global embedding |

## Fairness Confounds

**The comparison between encoders is NOT apples-to-apples.** Several confounds exist:

### 1. Embedding dimensionality

A linear probe on 1408-d (EchoJEPA-G) has 1408 free parameters per output class vs 512 for EchoPrime. Higher dimensionality gives the probe more capacity regardless of representation quality. This inflates EchoJEPA-G's advantage over smaller models.

### 2. Token structure asymmetry (attentive probe bias)

This is the most important confound for attentive probes. EchoJEPA and VideoMAE return the **full spatial-temporal token grid** (1568 tokens), while PanEcho and EchoPrime return a **single pre-pooled token**:

- **EchoJEPA/VideoMAE (1568 tokens)**: The attentive probe's cross-attention queries can selectively attend to specific spatial regions and temporal frames. This is the design intent — learn which tokens matter for the task.
- **PanEcho/EchoPrime (1 token)**: Cross-attention over a single element is degenerate — softmax over one element is trivially 1.0. The attentive probe's parameters (query embeddings, multi-head attention, self-attention blocks) become pure optimization overhead with no representational benefit. This makes attentive probes **strictly harder to train** than a linear layer for these models, leading to worse performance from overfitting.

**This is why attentive probes underperformed linear probes for PanEcho/EchoPrime on view classification in the ICML preprint.** It is not a statement about representation quality — it's a probe architecture mismatch.

For `extract_embeddings.py`, the mean-pool step (line 173: `o.mean(dim=1)`) aggregates spatial information for EchoJEPA/VideoMAE but is a no-op for PanEcho/EchoPrime (already 1 token).

### 3. Model scale

ViT-g (1B) vs MViT-v2-S (35M) is a ~30× parameter difference. Larger models learn richer representations regardless of training objective. This confounds any JEPA-vs-supervised comparison.

### 4. Pretraining data

| Model | Pretraining data |
|-------|-----------------|
| EchoJEPA-G | 18M UHN echos (domain-matched, massive) |
| EchoJEPA-L | MIMIC-IV-Echo (525K echos) |
| VideoMAE | Kinetics (general video) or MIMIC (depends on checkpoint) |
| PanEcho | Own echo dataset (published) |
| EchoPrime | Own echo dataset (published) |

Data quantity and domain match both affect representation quality independently of the training objective.

## Controlled Comparisons

For a **fair objective comparison** (JEPA vs MAE), the only controlled pair is **EchoJEPA-L vs VideoMAE-L** — same ViT-L backbone, same embed_dim (1024), same token structure, trainable on the same data.

For EchoJEPA-G vs PanEcho/EchoPrime, this is a **system-level comparison** ("our best vs their best"), which is standard in clinical papers but should not be used to claim JEPA > supervised learning.

## Mitigations for Nature Medicine Paper

The Nature Medicine paper uses linear probes on mean-pooled embeddings, which helps:

- **Linear probes eliminate the token-count confound** — all models are mean-pooled to a single vector before the probe, so the comparison depends only on embedding content, not token structure
- **Report embedding dimensionality** in all tables so reviewers can assess the capacity confound
- **Always report EchoJEPA-L alongside EchoJEPA-G** — the L model is a fairer size/data comparator
- **Optional**: PCA-project all embeddings to a common dimensionality (e.g., 512-d) before probing to control for the dimensionality confound

## Related Documents
- `probe-architecture-analysis.md` — empirical analysis of attentive vs linear probe inversion, rebuttal strategy, task-specific behavior
- `probe_dim_analysis.md` — full conversation log (raw source)
- `../architecture/probe-system.md` — probe architecture details, config reference
