#!/usr/bin/env bash
# One-shot orchestration for PR-N1/N1b/N2 manifest pipeline.
#
# Stages:
#   1. build_manifest       phase_annotations.parquet ⋈ mimic_classifications.csv → clip manifest
#   2. splits               assign patient-level train/val/test on hashed patient_id
#   3. add_quality_buckets  train-cohort tertiles → quality_bucket column
#   4. dedup (metadata-only) annotate n_duplicates + is_duplicate_of
#   5. element_grouping     study_element_manifest (3-tuple key)
#   6. sample_K             study_clip_sample_K8_seed0 (train split)
#   7. view_modality_coverage_audit  reports/echoset_jepa/coverage_audit.{md,json}
#
# Usage:
#   scripts/echoset_jepa/build_and_audit.sh [OUT_DIR]
# Default OUT_DIR = /tmp/echoset_pr_n2
#
# No GPU. Runs in ~90s end-to-end on 214k MIMIC clips.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-/tmp/echoset_pr_n2}"
mkdir -p "$OUT_DIR" "reports/echoset_jepa"

PA="classifier/phase/phase_annotations/phase_annotations.parquet"
CLS="/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_classifications.csv"

CLIP="$OUT_DIR/study_clip_manifest.parquet"
CLIP_SPLIT="$OUT_DIR/study_clip_manifest_split.parquet"
CLIP_FINAL="$OUT_DIR/study_clip_manifest_final.parquet"
CLIP_DEDUP="$OUT_DIR/study_clip_manifest_dedup.parquet"
ELEM="$OUT_DIR/study_element_manifest.parquet"
K8="$OUT_DIR/study_clip_sample_K8_seed0_train.parquet"

echo "=== 1. build_manifest ==="
python -m experiments.echoset_jepa.build_manifest \
  --phase_annotations "$PA" \
  --classifications_csv "$CLS" \
  --out "$CLIP"

echo "=== 2. splits ==="
python -m experiments.echoset_jepa.splits \
  --manifest "$CLIP" --out "$CLIP_SPLIT"

echo "=== 3. add_quality_buckets ==="
python - <<PY
import pandas as pd
from experiments.echoset_jepa.build_manifest import add_quality_buckets
df = pd.read_parquet("$CLIP_SPLIT")
train_sids = df[df.split == "train"].study_id.unique().tolist()
add_quality_buckets("$CLIP_SPLIT", train_sids, "$CLIP_FINAL")
PY

echo "=== 4. dedup (metadata-only) ==="
python -m experiments.echoset_jepa.dedup \
  --manifest "$CLIP_FINAL" --out "$CLIP_DEDUP"

echo "=== 5. element_grouping ==="
python -m experiments.echoset_jepa.element_grouping \
  --clip_manifest "$CLIP_DEDUP" --out "$ELEM"

echo "=== 6. sample_K (K=8, train) ==="
python -m experiments.echoset_jepa.sample_K \
  --clip_manifest "$CLIP_DEDUP" --out "$K8" \
  --K 8 --seed 0 --split train

echo "=== 7. coverage audit ==="
python -m experiments.echoset_jepa.view_modality_coverage_audit \
  --clip_manifest "$CLIP_DEDUP" \
  --element_manifest "$ELEM" \
  --k_sample_manifest "$K8" \
  --out_dir reports/echoset_jepa \
  || echo "[WARN] coverage audit failed gates (see reports/echoset_jepa/coverage_audit.md)"

echo
echo "=== Summary ==="
echo "  clip manifest:    $CLIP_DEDUP"
echo "  element manifest: $ELEM"
echo "  K=8 (train):      $K8"
echo "  report:           reports/echoset_jepa/coverage_audit.md"
