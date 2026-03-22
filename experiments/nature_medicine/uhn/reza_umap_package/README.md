# UMAP Visualization Package for Reza

UHN study-level mean-pooled embeddings + labels for Nature Medicine UMAP figures.

## Models (4 available, PanEcho pending extraction)

| Model | File | Dim | N Studies |
|-------|------|-----|-----------|
| EchoJEPA-G | `embeddings/echojepa_g.npz` | 1408 | 319,815 |
| EchoJEPA-L | `embeddings/echojepa_l.npz` | 1024 | 319,802 |
| EchoJEPA-L-K | `embeddings/echojepa_l_kinetics.npz` | 1024 | 319,818 |
| EchoPrime | `embeddings/echoprime.npz` | 512 | 319,818 |
| PanEcho | *not yet extracted* | 768 | — |

## Loading

```python
import numpy as np

# Load embeddings
emb = np.load('embeddings/echojepa_g.npz', allow_pickle=True)
X = emb['embeddings']       # (319815, 1408)
study_ids = emb['study_ids'] # (319815,) — DICOM StudyInstanceUID strings

# Load labels
lab = np.load('labels/mr_severity.npz', allow_pickle=True)
label_ids = lab['study_ids']  # (133303,)
labels = lab['labels']        # (133303,) — integer class labels
splits = lab['splits']        # (133303,) — 'train'/'val'/'test'

# Align: only keep studies that have both embeddings and labels
id_to_idx = {sid: i for i, sid in enumerate(study_ids)}
mask = [id_to_idx[sid] for sid in label_ids if sid in id_to_idx]
label_mask = [i for i, sid in enumerate(label_ids) if sid in id_to_idx]

X_aligned = X[mask]           # (~133K, 1408)
y_aligned = labels[label_mask] # (~133K,)
```

## Priority tasks for main figure

### Tier 1 (main text figure)
- **mr_severity** — 5 classes (trivial/mild/moderate/mod-severe/severe), 133K studies, AUROC 0.860. Best result, ordinal grades should show graded UMAP structure.
- **rv_function** — 5 classes (normal/mildly/moderately/severely reduced/hyperdynamic), 125K studies. Paper has explicit TBD placeholder for this.
- **tr_severity** — 5 classes, 139K studies, AUROC 0.838.

### Tier 2 (extended data / supplementary)
- **as_severity** — 4 classes, 176K studies, AUROC 0.908.
- **disease_hcm** — binary, 73K studies. Rare disease, showed decent separation in MIMIC t-SNE.
- **lv_systolic_function** — ordinal LV function grade.
- **diastolic_function** — diastolic filling grade.

### Tier 3 (supplementary only, if space)
- **cardiac_rhythm** — 6 classes, 45K studies. Includes AFib.
- **disease_***: amyloidosis, DCM, bicuspid AV, endocarditis, etc. Binary, small positives.

## Recommended approach

1. **Use UMAP** (not t-SNE) — better global structure preservation, Nature Medicine convention.
2. **L2-normalize** embeddings before UMAP — models have different magnitude scales.
3. **PCA to 50-100 dims** before UMAP — denoises, speeds up neighbor graph.
4. **Layout**: 1x4 panel (G, L-K, EP, L) per task, same color scheme. Add PanEcho when available.
5. **Color by ordinal grade** for severity tasks — use a sequential colormap (e.g., viridis) to show the graded organization.
6. **Add silhouette score** as a small annotation in each panel (compute in original embedding space, not UMAP space).
7. **Drop K-means column** from previous t-SNE plots.

## Quick start

```python
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap

# Load and align (as above)
X_norm = normalize(X_aligned, norm='l2')
X_pca = PCA(n_components=50).fit_transform(X_norm)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X_pca)

# Plot colored by severity grade
import matplotlib.pyplot as plt
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_aligned, cmap='viridis', s=0.5, alpha=0.3)
plt.colorbar(label='MR Severity Grade')
```

## Notes
- These are UHN embeddings (318K+ studies), not MIMIC (7K studies). Much larger and covers the paper's headline hemodynamic results.
- Embeddings are mean-pooled over all spatiotemporal tokens (1568 tokens for G, 1 token for EchoPrime). This is the standard SSL visualization approach.
- Shuffle alignment verified 2026-03-18: all 4 models pass functional probe test (balanced acc 0.31-0.44 on 5-class MR severity, well above 0.20 random).
- PanEcho extraction pending — will be added to embeddings/ when complete.
