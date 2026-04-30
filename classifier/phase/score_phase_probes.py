"""Post-process phase-probe sin/cos predictions into phase metrics.

Joins sin and cos test-set prediction CSVs per arm, reconstructs the
predicted phase φ̂ via atan2(ŝin, ĉos) / (2π), and scores against the
ground-truth phase pulled from the anchors diagnostics CSV.

Inputs:
  --pred-dir               local dir with 6 CSVs (mirror of
                           runs/phase_<arm>_<target>_test_<jid>/predictions/)
                           or equivalent flat dir
  --anchors-csv            diagnostic CSV with ground-truth phase per clip
                           (written by build_phase_probe_csvs.py:
                           data/csv/mimic_phase_anchors_10k.csv)
  --train-anchors-csv      (optional) anchors CSV restricted to train split —
                           used for the constant-baseline phase distribution
  --arms                   space-separated arm tags (default: e100 phase_542 sv_548)
  --train-logs             map arm → train log_r0.csv for best-epoch report
  --out-dir                where to write per-arm+sin/cos joined CSVs + summary

Outputs:
  <out>/per_arm/<arm>_joined.csv      per-test-clip rows with
                                      [dicom_id, sin_true, cos_true, sin_pred,
                                       cos_pred, phi_true, phi_pred, dphi, bin_true,
                                       bin_pred]
  <out>/summary.csv                   one row per arm with all reported metrics
  <out>/per_bin.csv                   per-arm × per-phase-bin rows (counts, MAE)
  <out>/best_epochs.csv               sin/cos best epoch + val MAE per arm

Decision flag:
  Prints DECISION: {POSITIVE|NEUTRAL} based on whether phase+25 beats both
  sv+25 and e100 on circular MAE or macro phase-bin accuracy.
"""
from __future__ import annotations
import argparse, csv, math, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrap-aware absolute phase difference in [0, 0.5] cycles."""
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1.0 - d)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def load_pred(pred_dir: Path, arm: str, target: str) -> pd.DataFrame:
    """Locate the test-set prediction CSV for (arm, target).
    Accepts either flat layout (arm_target.csv) or
    '<arm_tag>_<target>_test.csv' naming from the sbatch pipeline."""
    candidates = [
        pred_dir / f"phase_{arm}_{target}_test.csv",
        pred_dir / f"{arm}_{target}_test.csv",
        pred_dir / f"phase_{arm}_{target}.csv",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    # Last resort: glob
    matches = list(pred_dir.glob(f"*{arm}*{target}*.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(
        f"No prediction CSV for arm={arm} target={target} under {pred_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def uri_to_dicom_id(uri: str) -> str:
    return Path(str(uri)).stem


def join_arm(pred_dir: Path, arm: str, anchors: pd.DataFrame) -> pd.DataFrame:
    """Load sin and cos predictions, join by video_path, attach ground-truth
    phase from anchors. Returns per-clip dataframe."""
    p_sin = load_pred(pred_dir, arm, 'sin')[['video_path', 'label_real', 'pred_real']].copy()
    p_cos = load_pred(pred_dir, arm, 'cos')[['video_path', 'label_real', 'pred_real']].copy()
    p_sin.columns = ['video_path', 'sin_true_csv', 'sin_pred_raw']
    p_cos.columns = ['video_path', 'cos_true_csv', 'cos_pred_raw']
    df = p_sin.merge(p_cos, on='video_path', how='inner')
    if len(df) == 0:
        raise RuntimeError(f"arm={arm}: sin∩cos join produced 0 rows")
    df['dicom_id'] = df.video_path.map(uri_to_dicom_id)

    # Attach ground-truth phase + bin from anchors CSV
    gt = anchors[['dicom_id', 'phase', 'sin', 'cos', 'subject_id', 'split']].copy()
    gt.columns = ['dicom_id', 'phi_true', 'sin_true', 'cos_true', 'subject_id', 'split']
    df = df.merge(gt, on='dicom_id', how='inner')

    # Sanity: CSV-embedded target should match anchors-derived target
    sin_drift = float((df['sin_true_csv'] - df['sin_true']).abs().max())
    cos_drift = float((df['cos_true_csv'] - df['cos_true']).abs().max())
    if max(sin_drift, cos_drift) > 1e-3:
        print(f"WARN arm={arm}: sin/cos target drift vs anchors "
              f"(sin max={sin_drift:.4f} cos max={cos_drift:.4f})")

    # Unit-circle-normalize prediction before reconstructing phase
    norm = np.sqrt(df['sin_pred_raw'] ** 2 + df['cos_pred_raw'] ** 2).clip(lower=1e-8)
    df['sin_pred'] = df['sin_pred_raw'] / norm
    df['cos_pred'] = df['cos_pred_raw'] / norm

    df['phi_pred'] = (np.arctan2(df['sin_pred'], df['cos_pred']) / (2 * math.pi)) % 1.0
    df['phi_true_wrap'] = df['phi_true'] % 1.0  # Safety, phases are already in [0,1)
    df['dphi'] = circular_diff(df['phi_true_wrap'].to_numpy(), df['phi_pred'].to_numpy())

    # 10-bin assignment on [0,1)
    bins = np.linspace(0, 1, 11)
    df['bin_true'] = np.digitize(df['phi_true_wrap'], bins[1:-1])
    df['bin_pred'] = np.digitize(df['phi_pred'], bins[1:-1])

    return df


def metrics_for_arm(df: pd.DataFrame, train_phis: np.ndarray) -> dict:
    """Compute all reported metrics from the joined dataframe.

    Constant-baseline: predict a constant φ̂ chosen to minimize train-set
    circular MAE. Because phase is cyclic, the optimum isn't the mean;
    scan the 10 train-bin centers and take argmin."""
    sin_true = df['sin_true'].to_numpy()
    cos_true = df['cos_true'].to_numpy()
    sin_pred = df['sin_pred'].to_numpy()
    cos_pred = df['cos_pred'].to_numpy()

    cmae_cycles = float(df['dphi'].mean())
    cmae_deg = cmae_cycles * 360.0

    bin_acc = float((df['bin_true'] == df['bin_pred']).mean())

    # Per-bin circular MAE, macro-averaged
    per_bin = []
    for b in range(10):
        sub = df[df['bin_true'] == b]
        if len(sub) == 0:
            per_bin.append({'bin': b, 'n': 0, 'cmae_cycles': float('nan'),
                            'cmae_deg': float('nan'), 'bin_acc': float('nan')})
            continue
        per_bin.append({
            'bin': b,
            'n': int(len(sub)),
            'cmae_cycles': float(sub['dphi'].mean()),
            'cmae_deg': float(sub['dphi'].mean()) * 360.0,
            'bin_acc': float((sub['bin_true'] == sub['bin_pred']).mean()),
        })
    macro_cmae = float(np.nanmean([b['cmae_cycles'] for b in per_bin]))
    macro_bin_acc = float(np.nanmean([b['bin_acc'] for b in per_bin]))

    # Constant baseline
    bin_centers = np.linspace(0.05, 0.95, 10)
    best_const = None; best_const_cmae = float('inf')
    test_phi = df['phi_true_wrap'].to_numpy()
    for c in bin_centers:
        cmae_c = float(circular_diff(test_phi, np.full_like(test_phi, c)).mean())
        if cmae_c < best_const_cmae:
            best_const_cmae = cmae_c; best_const = float(c)

    return {
        'n_test': int(len(df)),
        'sin_R2':  r2(sin_true, sin_pred),
        'cos_R2':  r2(cos_true, cos_pred),
        'mean_component_R2': 0.5 * (r2(sin_true, sin_pred) + r2(cos_true, cos_pred)),
        'circular_MAE_cycles': cmae_cycles,
        'circular_MAE_deg':    cmae_deg,
        'phase_bin_acc':       bin_acc,
        'macro_circular_MAE_cycles': macro_cmae,
        'macro_circular_MAE_deg':    macro_cmae * 360.0,
        'macro_phase_bin_acc':       macro_bin_acc,
        'const_baseline_phi':        best_const,
        'const_baseline_cmae_cycles': best_const_cmae,
        'const_baseline_cmae_deg':    best_const_cmae * 360.0,
    }, per_bin


def load_train_log(path: Path, target: str) -> dict:
    """Parse a probe log_r0.csv. Tolerates re-header lines (resumed runs)."""
    if not path.exists():
        return {'best_ep': None, 'best_val_mae': None}
    rows = []
    with path.open() as f:
        for ln in f:
            parts = ln.strip().split(',')
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            try:
                rows.append((int(parts[0]), float(parts[2])))  # epoch, val_mae
            except ValueError:
                pass
    if not rows:
        return {'best_ep': None, 'best_val_mae': None}
    best = min(rows, key=lambda r: r[1])
    return {'best_ep': int(best[0]), 'best_val_mae': float(best[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', type=Path, required=True,
                    help='dir with 6 prediction CSVs (phase_<arm>_<target>_test.csv)')
    ap.add_argument('--anchors-csv', type=Path, required=True,
                    help='data/csv/mimic_phase_anchors_10k.csv')
    ap.add_argument('--arms', nargs='+', default=['e100', 'phase_542', 'sv_548'])
    ap.add_argument('--train-logs-dir', type=Path, default=None,
                    help='optional: dir with <arm>_<target>/log_r0.csv files')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'per_arm').mkdir(parents=True, exist_ok=True)

    anchors = pd.read_csv(args.anchors_csv)
    train_phis = anchors[anchors.split == 'train']['phase'].to_numpy()

    all_summary = []
    all_per_bin = []
    all_best = []
    for arm in args.arms:
        print(f"\n=== arm={arm} ===")
        df = join_arm(args.pred_dir, arm, anchors)
        df.to_csv(args.out_dir / 'per_arm' / f'{arm}_joined.csv', index=False)

        summary, per_bin = metrics_for_arm(df, train_phis)
        summary['arm'] = arm
        all_summary.append(summary)

        for pb in per_bin:
            pb['arm'] = arm
            all_per_bin.append(pb)

        # Best epoch lookup (if provided)
        if args.train_logs_dir is not None:
            sin_best = load_train_log(args.train_logs_dir / f'phase_{arm}_sin' / 'log_r0.csv', 'sin')
            cos_best = load_train_log(args.train_logs_dir / f'phase_{arm}_cos' / 'log_r0.csv', 'cos')
            row = {
                'arm': arm,
                'sin_best_ep': sin_best['best_ep'],
                'sin_best_val_mae': sin_best['best_val_mae'],
                'cos_best_ep': cos_best['best_ep'],
                'cos_best_val_mae': cos_best['best_val_mae'],
                'best_epochs_differ': (sin_best['best_ep'] != cos_best['best_ep']),
            }
            all_best.append(row)

        print(f"  n_test={summary['n_test']}")
        print(f"  sin R²={summary['sin_R2']:+.3f}  cos R²={summary['cos_R2']:+.3f}  "
              f"mean component R²={summary['mean_component_R2']:+.3f}")
        print(f"  circular MAE: {summary['circular_MAE_cycles']:.3f} cycles "
              f"({summary['circular_MAE_deg']:.1f}°)")
        print(f"  macro circular MAE (by bin): {summary['macro_circular_MAE_cycles']:.3f} cycles "
              f"({summary['macro_circular_MAE_deg']:.1f}°)")
        print(f"  phase-bin (10) acc: {summary['phase_bin_acc']:.3f}  "
              f"macro: {summary['macro_phase_bin_acc']:.3f}")
        print(f"  constant baseline (φ={summary['const_baseline_phi']:.2f}) "
              f"cmae: {summary['const_baseline_cmae_cycles']:.3f} cycles")

    # Write summary.csv
    summary_df = pd.DataFrame(all_summary)
    summary_df = summary_df[['arm'] + [c for c in summary_df.columns if c != 'arm']]
    summary_df.to_csv(args.out_dir / 'summary.csv', index=False)
    print(f"\nwrote {args.out_dir/'summary.csv'}")

    pd.DataFrame(all_per_bin).to_csv(args.out_dir / 'per_bin.csv', index=False)
    if all_best:
        pd.DataFrame(all_best).to_csv(args.out_dir / 'best_epochs.csv', index=False)

    # --- Decision ---
    sdf = summary_df.set_index('arm')
    required = {'e100', 'phase_542', 'sv_548'}
    if not required.issubset(sdf.index):
        print(f"\nDECISION: INCOMPLETE — missing arms {required - set(sdf.index)}")
        return

    phase = sdf.loc['phase_542']
    sv    = sdf.loc['sv_548']
    e100  = sdf.loc['e100']

    # Positive = phase+25 beats both controls on circular MAE OR macro bin acc
    beats_on_cmae = (phase['circular_MAE_cycles'] < sv['circular_MAE_cycles']) \
                 and (phase['circular_MAE_cycles'] < e100['circular_MAE_cycles'])
    beats_on_macro_acc = (phase['macro_phase_bin_acc'] > sv['macro_phase_bin_acc']) \
                     and (phase['macro_phase_bin_acc'] > e100['macro_phase_bin_acc'])

    phase_vs_sv_cmae_gap = float(sv['circular_MAE_cycles'] - phase['circular_MAE_cycles'])

    if beats_on_cmae or beats_on_macro_acc:
        verdict = 'POSITIVE mechanism signal'
        reason = (f"phase+25 beats sv+25 and e100 "
                  f"{'on circular MAE' if beats_on_cmae else ''}"
                  f"{' and ' if (beats_on_cmae and beats_on_macro_acc) else ''}"
                  f"{'on macro bin acc' if beats_on_macro_acc else ''}.")
    elif abs(phase_vs_sv_cmae_gap) < 0.005:  # < 0.5% cycle = 1.8° — within noise
        verdict = 'NEUTRAL (phase+25 ≈ sv+25)'
        reason = (f"phase+25 circular MAE = {phase['circular_MAE_cycles']:.4f} vs "
                  f"sv+25 = {sv['circular_MAE_cycles']:.4f} — difference "
                  f"{phase_vs_sv_cmae_gap:+.4f} cycles within noise threshold. "
                  f"Phase-matched sampling did not clearly improve explicit phase "
                  f"decodability at +25.")
    else:
        verdict = 'NEGATIVE (sv+25 better than phase+25)'
        reason = (f"sv+25 circular MAE = {sv['circular_MAE_cycles']:.4f} < "
                  f"phase+25 = {phase['circular_MAE_cycles']:.4f} "
                  f"(gap {phase_vs_sv_cmae_gap:+.4f} cycles).")

    print(f"\nDECISION: {verdict}")
    print(f"  {reason}")
    (args.out_dir / 'decision.txt').write_text(f"{verdict}\n{reason}\n")


if __name__ == '__main__':
    main()
