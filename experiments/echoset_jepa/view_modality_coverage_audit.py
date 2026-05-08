"""View × modality coverage audit for EchoSet-JEPA (plan §14.2).

Run before any GPU time is spent:
  1. Confirm the MIMIC cohort has enough view/modality diversity for the K=8
     fairness protocol to be meaningful (≥90% of studies have ≥2 view
     families in their K-sample; ≥80% of studies with any color keep at
     least one color slot).
  2. Document known-unknowns (measurement_site mostly unknown;
     phase_bucket mostly full_cycle).
  3. Expose modality-presence leakage signals that downstream probe
     protocols must disarm (e.g. color-present-only baseline for MR).

Inputs (all landed by PR-N1b):
  - study_clip_manifest_final.parquet  (clip-level, with split + quality_bucket)
  - study_element_manifest.parquet     (element-level, 3-tuple key)
  - study_clip_sample_K8_seed0_train.parquet  (K=8 sample, train)

Outputs:
  - reports/echoset_jepa/coverage_audit.md   (human-readable)
  - reports/echoset_jepa/coverage_audit.json (machine-readable)
  - exit code 0 if all gate thresholds pass; 1 otherwise
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


GATE_THRESHOLDS = {
    # ≥ 90% of train studies must have ≥2 distinct view_families after K=8
    "min_frac_studies_ge2_view_families": 0.90,
    # ≥ 80% of studies with any color_doppler clip must retain >=1 in K=8
    "min_frac_color_retention": 0.80,
}


def _study_clip_stats(clip_df) -> Dict:
    cps = clip_df.groupby("study_id").size()
    return {
        "n_clips": int(len(clip_df)),
        "n_studies": int(clip_df["study_id"].nunique()),
        "n_patients": int(clip_df["patient_id"].nunique()),
        "clips_per_study_median": float(cps.median()),
        "clips_per_study_p75": float(cps.quantile(0.75)),
        "clips_per_study_p95": float(cps.quantile(0.95)),
        "clips_per_study_max": float(cps.max()),
    }


def _view_modality_crosstab(clip_df) -> Dict:
    import pandas as pd
    ct = pd.crosstab(clip_df["view_family"], clip_df["modality"])
    return {str(v): {str(m): int(ct.loc[v, m]) for m in ct.columns} for v in ct.index}


def _clipcount_only_baseline(clip_df) -> Dict:
    """What signal does clip-count alone carry per study?

    Proxy: for each study, count clips and per-modality counts. Downstream
    probes that 'accidentally' rely on color-being-present will be caught
    because the presence/absence pattern correlates with certain endpoints.
    """
    g = clip_df.groupby("study_id")
    out = {
        "clip_count_mean": float(g.size().mean()),
        "clip_count_std": float(g.size().std()),
        "frac_studies_with_color": float(
            (g["modality"].apply(lambda s: (s == "color_doppler").any())).mean()
        ),
        "frac_studies_bmode_only": float(
            (g["modality"].apply(lambda s: (s == "b_mode").all())).mean()
        ),
    }
    return out


def _k_sample_diagnostics(k_sample_df) -> Dict:
    """Per-study diagnostics of the K=8 manifest."""
    g = k_sample_df.groupby("study_id")
    n_vf = g["view_family"].nunique()
    n_mod = g["modality"].nunique()
    n_color_per_study = g.apply(lambda df: int((df["modality"] == "color_doppler").sum()))
    n_bmode_per_study = g.apply(lambda df: int((df["modality"] == "b_mode").sum()))
    return {
        "n_studies_in_sample": int(k_sample_df["study_id"].nunique()),
        "n_clips_in_sample": int(len(k_sample_df)),
        "view_families_per_study": {
            "mean": float(n_vf.mean()),
            "p10": float(n_vf.quantile(0.10)),
            "median": float(n_vf.median()),
            "frac_ge2": float((n_vf >= 2).mean()),
        },
        "modalities_per_study": {
            "mean": float(n_mod.mean()),
            "frac_both": float((n_mod >= 2).mean()),
        },
        "color_slots_per_study": {
            "mean": float(n_color_per_study.mean()),
            "median": float(n_color_per_study.median()),
            "frac_ge1": float((n_color_per_study >= 1).mean()),
        },
        "bmode_slots_per_study": {
            "mean": float(n_bmode_per_study.mean()),
            "median": float(n_bmode_per_study.median()),
        },
    }


def _color_retention(manifest_df, k_sample_df) -> float:
    """For studies that have ≥1 color clip in the full manifest, what
    fraction retain ≥1 color clip in the K=8 sample?"""
    has_color_full = manifest_df.groupby("study_id").apply(
        lambda df: (df["modality"] == "color_doppler").any()
    )
    studies_with_color = set(has_color_full[has_color_full].index)
    if not studies_with_color:
        return 1.0
    has_color_k = k_sample_df.groupby("study_id").apply(
        lambda df: (df["modality"] == "color_doppler").any()
    )
    retained = sum(1 for s in studies_with_color if has_color_k.get(s, False))
    return retained / len(studies_with_color)


def _element_manifest_stats(element_df) -> Dict:
    eps = element_df.groupby("study_id").size()
    return {
        "n_elements": int(len(element_df)),
        "elements_per_study_median": float(eps.median()),
        "elements_per_study_p75": float(eps.quantile(0.75)),
        "elements_per_study_p95": float(eps.quantile(0.95)),
        "elements_per_study_max": float(eps.max()),
        "frac_studies_with_ge2_elements": float((eps >= 2).mean()),
        "n_distinct_element_keys": int(len(
            element_df.groupby(["view_family", "modality", "phase_bucket"]).size()
        )),
    }


def _modality_presence_leakage(clip_df) -> Dict:
    """Per-study modality presence — the minimum diagnostic for leakage.

    These counts are what Control D (metadata-only) and the
    color-present-only / clip-count-only baselines will train on.
    If a downstream endpoint (e.g. MR severity) correlates strongly with
    'does the study have color?', metadata-only will appear to work —
    and that's what the control is there to quantify.
    """
    g = clip_df.groupby("study_id")
    return {
        "per_study_color_doppler_count": {
            "mean": float(g.apply(lambda d: (d["modality"] == "color_doppler").sum()).mean()),
            "std":  float(g.apply(lambda d: (d["modality"] == "color_doppler").sum()).std()),
        },
        "per_study_bmode_count": {
            "mean": float(g.apply(lambda d: (d["modality"] == "b_mode").sum()).mean()),
            "std":  float(g.apply(lambda d: (d["modality"] == "b_mode").sum()).std()),
        },
        "per_study_view_families": {
            "mean": float(g["view_family"].nunique().mean()),
            "std": float(g["view_family"].nunique().std()),
        },
    }


def _quality_bucket_stats(clip_df) -> Dict:
    if "quality_bucket" not in clip_df.columns:
        return {"note": "quality_bucket not populated; run add_quality_buckets first"}
    import pandas as pd
    ct = pd.crosstab(clip_df.get("split", "unknown"), clip_df["quality_bucket"])
    return {
        str(split): {str(b): int(ct.loc[split, b]) for b in ct.columns}
        for split in ct.index
    }


def run_audit(
    clip_manifest_path: str,
    element_manifest_path: str,
    k_sample_path: str,
    out_dir: str,
) -> Dict:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clip_df = pd.read_parquet(clip_manifest_path)
    element_df = pd.read_parquet(element_manifest_path)
    k_df = pd.read_parquet(k_sample_path)

    summary: Dict = {
        "clip_manifest": clip_manifest_path,
        "element_manifest": element_manifest_path,
        "k_sample_manifest": k_sample_path,
        "overall": _study_clip_stats(clip_df),
        "view_modality_crosstab": _view_modality_crosstab(clip_df),
        "clipcount_only_baseline": _clipcount_only_baseline(clip_df),
        "elements": _element_manifest_stats(element_df),
        "k_sample": _k_sample_diagnostics(k_df),
        "color_retention_in_K8": _color_retention(clip_df, k_df),
        "modality_presence_leakage": _modality_presence_leakage(clip_df),
        "quality_bucket_by_split": _quality_bucket_stats(clip_df),
    }

    if "split" in clip_df.columns:
        summary["per_split"] = {}
        for sp in ["train", "val", "test"]:
            sub = clip_df[clip_df["split"] == sp]
            if len(sub):
                summary["per_split"][sp] = _study_clip_stats(sub)

    # --- gate evaluation ---------------------------------------------------
    ks = summary["k_sample"]
    frac_ge2_vf = ks["view_families_per_study"]["frac_ge2"]
    color_ret = summary["color_retention_in_K8"]

    gates = {
        "frac_studies_ge2_view_families_in_K8": frac_ge2_vf,
        "color_retention_in_K8": color_ret,
    }
    gate_pass = (
        frac_ge2_vf >= GATE_THRESHOLDS["min_frac_studies_ge2_view_families"]
        and color_ret >= GATE_THRESHOLDS["min_frac_color_retention"]
    )
    summary["gates"] = gates
    summary["gate_thresholds"] = GATE_THRESHOLDS
    summary["gate_passed"] = bool(gate_pass)

    (out / "coverage_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "coverage_audit.md").write_text(_render_markdown(summary))
    logger.info("coverage audit → %s (gate_passed=%s)", out, gate_pass)
    return summary


def _render_markdown(s: Dict) -> str:
    lines = [
        "# EchoSet-JEPA view × modality coverage audit",
        "",
        f"- clip_manifest: `{s['clip_manifest']}`",
        f"- element_manifest: `{s['element_manifest']}`",
        f"- k_sample_manifest: `{s['k_sample_manifest']}`",
        "",
        "## Gate status",
        "",
        f"- **passed**: {s['gate_passed']}",
        f"- frac_studies_ge2_view_families_in_K8 = {s['gates']['frac_studies_ge2_view_families_in_K8']:.3f}  (threshold {s['gate_thresholds']['min_frac_studies_ge2_view_families']})",
        f"- color_retention_in_K8 = {s['gates']['color_retention_in_K8']:.3f}  (threshold {s['gate_thresholds']['min_frac_color_retention']})",
        "",
        "## Overall",
        "",
        f"- clips: {s['overall']['n_clips']:,}",
        f"- studies: {s['overall']['n_studies']:,}",
        f"- patients: {s['overall']['n_patients']:,}",
        f"- clips/study: median={s['overall']['clips_per_study_median']:.0f}, p75={s['overall']['clips_per_study_p75']:.0f}, p95={s['overall']['clips_per_study_p95']:.0f}, max={s['overall']['clips_per_study_max']:.0f}",
        "",
        "## Per-split",
        "",
    ]
    for sp, stats in (s.get("per_split") or {}).items():
        lines.append(f"### {sp}")
        lines.append(f"- {stats['n_clips']:,} clips / {stats['n_studies']:,} studies / {stats['n_patients']:,} patients")
        lines.append(f"- clips/study median={stats['clips_per_study_median']:.0f}")
        lines.append("")
    lines += [
        "## View × modality crosstab",
        "",
        "| view_family | " + " | ".join(sorted({m for row in s['view_modality_crosstab'].values() for m in row})) + " |",
    ]
    modalities = sorted({m for row in s['view_modality_crosstab'].values() for m in row})
    lines.append("|" + "---|" * (len(modalities) + 1))
    for vf, row in s["view_modality_crosstab"].items():
        lines.append("| " + vf + " | " + " | ".join(str(row.get(m, 0)) for m in modalities) + " |")
    lines += [
        "",
        "## K=8 sampler diagnostics",
        "",
        f"- studies sampled: {s['k_sample']['n_studies_in_sample']:,}",
        f"- clips sampled: {s['k_sample']['n_clips_in_sample']:,}",
        f"- view_families/study: mean={s['k_sample']['view_families_per_study']['mean']:.2f}, median={s['k_sample']['view_families_per_study']['median']:.1f}, frac_ge2={s['k_sample']['view_families_per_study']['frac_ge2']:.3f}",
        f"- color slots/study: mean={s['k_sample']['color_slots_per_study']['mean']:.2f}, median={s['k_sample']['color_slots_per_study']['median']:.1f}, frac_ge1={s['k_sample']['color_slots_per_study']['frac_ge1']:.3f}",
        f"- bmode slots/study: mean={s['k_sample']['bmode_slots_per_study']['mean']:.2f}",
        "",
        "## Element manifest",
        "",
        f"- elements: {s['elements']['n_elements']:,}",
        f"- elements/study: median={s['elements']['elements_per_study_median']:.0f}, p95={s['elements']['elements_per_study_p95']:.0f}, max={s['elements']['elements_per_study_max']:.0f}",
        f"- distinct (view, modality, phase_bucket) keys: {s['elements']['n_distinct_element_keys']}",
        "",
        "## Modality-presence leakage signals (for Control D / color-present-only baselines)",
        "",
        f"- frac studies with any color_doppler: {s['clipcount_only_baseline']['frac_studies_with_color']:.3f}",
        f"- frac studies B-mode-only: {s['clipcount_only_baseline']['frac_studies_bmode_only']:.3f}",
        f"- per-study color count: mean={s['modality_presence_leakage']['per_study_color_doppler_count']['mean']:.2f} ± {s['modality_presence_leakage']['per_study_color_doppler_count']['std']:.2f}",
        "",
        "## Quality bucket × split",
        "",
        "```",
        json.dumps(s["quality_bucket_by_split"], indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_manifest", required=True)
    ap.add_argument("--element_manifest", required=True)
    ap.add_argument("--k_sample_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_audit(args.clip_manifest, args.element_manifest, args.k_sample_manifest, args.out_dir)
    sys.exit(0 if s["gate_passed"] else 1)


if __name__ == "__main__":
    _main()
