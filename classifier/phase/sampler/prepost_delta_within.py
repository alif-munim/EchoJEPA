"""Pre/post Δ_within harness for measuring whether multi-view phase-matched
JEPA training induces cross-clip phase alignment in the encoder.

Runs ``embedding_substrate_validation`` twice — once with the pre-training
checkpoint, once with the post-training checkpoint — and emits a comparison
table of Δ_within, Δ_specificity, frac>0, and Wilcoxon p-values.

Why this exists
---------------
The frozen-encoder substrate test on JEPA-IN21K-e100 gave Δ_within ≈ 0
(README §329-354). That is the question of whether a pretrained encoder
*already* exhibits cross-clip phase alignment, not whether training with
phase-matched positives *induces* it. This harness is the latter test:
Δ_within_post − Δ_within_pre. Positive delta means training did the work
the phase-annotations pipeline was built to support.

Inputs
------
--pre and --post each point at a .pt checkpoint with a ``target_encoder``
state dict (the format ``build_encoder`` already consumes).

Usage
-----
    python prepost_delta_within.py \\
        --pre  /path/to/echojepa-l-k-e100.pt \\
        --post /path/to/echojepa-l-k-mv-e110.pt \\
        --out-dir /tmp/prepost_delta \\
        --n-within 30 --n-cross 30 --n-anchors 8

Artifacts written under --out-dir:
  pre/embedding_validation_results.csv
  pre/embedding_validation_summary.txt
  post/embedding_validation_results.csv
  post/embedding_validation_summary.txt
  prepost_comparison.csv       # delta table
  prepost_comparison.txt       # human summary

The harness does not require the post checkpoint to exist to be useful:
it also runs pre-only if --post is omitted, letting the substrate numbers
be captured before the multi-view fine-tune is launched.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE_DIR = HERE.parent  # classifier/phase
SUBSTRATE_SCRIPT = PHASE_DIR / "embedding_substrate_validation.py"


def run_substrate(
    checkpoint: Path,
    out_dir: Path,
    n_within: int,
    n_cross: int,
    n_anchors: int,
    device: str | None,
) -> Path:
    """Invoke embedding_substrate_validation.py with a monkey-patched
    CHECKPOINT. Caches under the substrate script's own ``embedding_cache``
    but the summary/results CSVs are copied into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{PHASE_DIR}")
import embedding_substrate_validation as esv
esv.CHECKPOINT = Path(r"{checkpoint}")
sys.argv = ["esv", "--skip-sanity", "--n-within", "{n_within}",
            "--n-cross", "{n_cross}", "--n-anchors", "{n_anchors}"]
{f'sys.argv += ["--device", "{device}"]' if device else ""}
esv.main()
"""
    r = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"substrate run failed for {checkpoint}")

    results = PHASE_DIR / "embedding_validation_results.csv"
    summary = PHASE_DIR / "embedding_validation_summary.txt"
    out_results = out_dir / "embedding_validation_results.csv"
    out_summary = out_dir / "embedding_validation_summary.txt"
    if results.exists():
        out_results.write_bytes(results.read_bytes())
    if summary.exists():
        out_summary.write_bytes(summary.read_bytes())
    return out_results


def _stats(delta: np.ndarray) -> dict:
    if not len(delta):
        return {"n": 0}
    out = {
        "n": int(len(delta)),
        "median": float(np.median(delta)),
        "q25": float(np.quantile(delta, 0.25)),
        "q75": float(np.quantile(delta, 0.75)),
        "frac_gt_0": float((delta > 0).mean()),
    }
    try:
        from scipy.stats import wilcoxon

        w = wilcoxon(delta, alternative="greater")
        out["wilcoxon_p_greater"] = float(w.pvalue)
    except Exception:
        out["wilcoxon_p_greater"] = float("nan")
    return out


def load_deltas(results_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract Δ_within (phase − random, within) and Δ_specificity (within −
    cross) per record from the substrate-validation CSV."""
    df = pd.read_csv(results_csv)
    need = {"sim_phase_within", "sim_random_within", "sim_phase_cross"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{results_csv} missing columns {missing}")
    d_within = (df.sim_phase_within - df.sim_random_within).to_numpy()
    d_spec = (df.sim_phase_within - df.sim_phase_cross).to_numpy()
    return d_within, d_spec


def compare(pre_csv: Path, post_csv: Path | None, out_dir: Path) -> dict:
    dw_pre, ds_pre = load_deltas(pre_csv)
    rows = [
        {"ckpt": "pre", "metric": "delta_within", **_stats(dw_pre)},
        {"ckpt": "pre", "metric": "delta_specificity", **_stats(ds_pre)},
    ]
    if post_csv is not None:
        dw_post, ds_post = load_deltas(post_csv)
        rows += [
            {"ckpt": "post", "metric": "delta_within", **_stats(dw_post)},
            {"ckpt": "post", "metric": "delta_specificity", **_stats(ds_post)},
        ]
        # Paired diff (requires same anchor ordering across runs). The
        # substrate script's pair sampler is seeded, so pre and post should
        # align row-for-row. Guard anyway.
        if len(dw_pre) == len(dw_post):
            rows += [
                {
                    "ckpt": "post_minus_pre",
                    "metric": "delta_within",
                    **_stats(dw_post - dw_pre),
                },
                {
                    "ckpt": "post_minus_pre",
                    "metric": "delta_specificity",
                    **_stats(ds_post - ds_pre),
                },
            ]

    out_csv = out_dir / "prepost_comparison.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)

    out_txt = out_dir / "prepost_comparison.txt"
    lines = []
    for r in rows:
        tag = f"[{r['ckpt']:<15}] {r['metric']:<20}"
        if r.get("n", 0) == 0:
            lines.append(f"{tag}  (no records)")
            continue
        lines.append(
            f"{tag}  n={r['n']:4d}  median={r['median']:+.4f}  "
            f"IQR=[{r['q25']:+.4f}, {r['q75']:+.4f}]  frac>0={r['frac_gt_0']:.2f}  "
            f"wilcoxon_p={r['wilcoxon_p_greater']:.2e}"
        )
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"rows": rows, "csv": out_csv, "txt": out_txt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=Path, required=True,
                    help=".pt checkpoint for the pre-training encoder.")
    ap.add_argument("--post", type=Path, default=None,
                    help=".pt checkpoint after multi-view phase-matched "
                         "fine-tune. Omit to run pre only.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-within", type=int, default=30)
    ap.add_argument("--n-cross", type=int, default=30)
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # embedding_substrate_validation caches embeddings at
    # classifier/phase/embedding_cache/<clip>.npz keyed ONLY by clip id, so
    # swapping the encoder checkpoint without clearing the cache returns
    # the previous encoder's embeddings. We clear the cache between pre and
    # post runs so the delta reflects a genuine encoder change.
    import shutil
    EMBED_CACHE = PHASE_DIR / "embedding_cache"

    def _clear_cache():
        if EMBED_CACHE.exists():
            shutil.rmtree(EMBED_CACHE)
        EMBED_CACHE.mkdir(parents=True, exist_ok=True)

    _clear_cache()
    pre_dir = args.out_dir / "pre"
    pre_csv = run_substrate(
        args.pre, pre_dir, args.n_within, args.n_cross, args.n_anchors, args.device,
    )

    post_csv = None
    if args.post is not None:
        _clear_cache()
        post_dir = args.out_dir / "post"
        post_csv = run_substrate(
            args.post, post_dir, args.n_within, args.n_cross, args.n_anchors, args.device,
        )

    compare(pre_csv, post_csv, args.out_dir)


if __name__ == "__main__":
    main()
