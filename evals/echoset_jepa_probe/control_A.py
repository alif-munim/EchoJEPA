"""Control A — V-JEPA + per-clip prediction averaging (plan §7).

For each of the K selected clips (from the shared K-manifest), run a trained
single-clip attentive probe (d=1); study prediction is the arithmetic mean of
clip predictions.

No new training: reuses the existing single-clip probe checkpoint produced by
``evals/video_classification_frozen``. Control A is an INFERENCE-ONLY path.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

logger = logging.getLogger(__name__)


def run(
    k_manifest: str,                       # study_clip_sample_K{K}_seed{S}.parquet
    clip_probe_checkpoint: str,
    task_targets_csv: str,
    out_csv: str,
) -> None:
    """Compute study-level predictions by arithmetic-mean over K clip scores.

    Algorithm:
      1. Load k_manifest → per study, list of (clip_id, cached_cclip_s3).
      2. For each clip, score = clip_probe(cached_cclip) — reuse existing d=1 head.
      3. Group by study, take mean → study_prediction.
      4. Join against task_targets_csv (study_id, target) → per-study (pred, gt).
      5. Write out_csv.

    This path does not touch the Stage-2 transformer at all. It exists only so
    that A, B1, B2, D, E and EchoSet are all scored on the same study manifest
    with identical ground-truth alignment.
    """
    raise NotImplementedError(
        "Control A scaffold — wires up after PR-1 manifest and existing d=1 probe "
        "checkpoint paths are confirmed. No new model training needed."
    )


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_manifest", required=True)
    ap.add_argument("--clip_probe_checkpoint", required=True)
    ap.add_argument("--task_targets_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.k_manifest, args.clip_probe_checkpoint, args.task_targets_csv, args.out_csv)


if __name__ == "__main__":
    _main()
