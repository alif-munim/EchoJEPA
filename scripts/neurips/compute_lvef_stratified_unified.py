"""Unified per-stratum MAE/Pearson/R² breakdown for EchoNet-Dynamic LVEF test (N=1,277)
with continuous clinical-range binning (ASE/AHA HFmrEF = [40, 50), etc.).

Two output tables:
  §05 controlled: JEPA e100 / BYOL e100 / MAE e99 / SALT v1 e79
  §08 phase-rel:  V-JEPA†-e125 / V4-e25

Also computes paired bootstrap CIs for Δ vs V-JEPA†-e125 on the §08 table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

RNG = np.random.default_rng(0)
B = 10000

STRATA = [
    ("Reduced <40",            lambda y: y < 40),
    ("Mildly reduced 40–49",   lambda y: (y >= 40) & (y < 50)),
    ("Normal 50–69",           lambda y: (y >= 50) & (y < 70)),
    ("Hyperdynamic ≥70",       lambda y: y >= 70),
]

# §05 controlled: load clean predictions from noised_inference CSVs
CONTROLLED = {
    "JEPA e100": "/tmp/e100_lvef/jepa_in21k_e100.csv",
    "BYOL e100": "/tmp/e100_lvef/byol_e100.csv",
    "MAE e99":   "/tmp/e100_lvef/mae_e99.csv",
    "SALT e79":  "/tmp/e100_lvef/salt_v1_e79.csv",
}
# §08 phase-rel: standard prediction CSVs (video_path, label_real, pred_real)
PHASE_REL = {
    "V-JEPA†-e125": "/tmp/lvef_strat/base_e125.csv",
    "V4-e25":       "/tmp/lvef_strat/v4_e25.csv",
}


def load_controlled(path):
    df = pd.read_csv(path)
    df = df[df.condition == "clean"].sort_values("sample_idx").reset_index(drop=True)
    y = df["label"].to_numpy()
    yhat = df["prediction"].to_numpy()
    return y, yhat


def load_phase_rel(path):
    df = pd.read_csv(path).sort_values("video_path").reset_index(drop=True)
    y = df["label_real"].to_numpy()
    yhat = df["pred_real"].to_numpy()
    return y, yhat


def metrics(y, yhat):
    if len(y) < 3:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(y, yhat)
    if y.std() > 0 and yhat.std() > 0:
        pr = pearsonr(y, yhat)[0]
        r2 = r2_score(y, yhat)
    else:
        pr, r2 = np.nan, np.nan
    return mae, pr, r2


def paired_delta(y, yh_a, yh_b, B=B, rng=RNG):
    n = len(y)
    if n < 10:
        return "—"
    d_mae = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        d_mae.append(mean_absolute_error(yy, yh_b[idx]) - mean_absolute_error(yy, yh_a[idx]))
    d = np.array(d_mae)
    p = (d > 0).mean()
    sig = "✅" if (np.percentile(d, 2.5) > 0) else ("❌" if np.percentile(d, 97.5) < 0 else "")
    return f"{d.mean():+.2f} [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}] P(A>B)={p:.2f} {sig}"


def table(name, data):
    """data: dict of {model_name: (y, yhat)}"""
    print("=" * 90)
    print(f"{name} — per-stratum MAE (continuous clinical-range binning)")
    print("=" * 90)
    # Check all models have same labels
    models = list(data.keys())
    y_ref = data[models[0]][0]
    for m in models[1:]:
        assert np.allclose(sorted(y_ref), sorted(data[m][0]), atol=1e-3), f"{m} labels mismatch"
    y = y_ref
    print(f"\n{'Stratum':<25} {'N':>5}", *(f"{m:>14}" for m in models), sep=" ")
    for stratum_name, pred in STRATA:
        mask = pred(y)
        n = int(mask.sum())
        row = []
        for m in models:
            y_m, yh_m = data[m]
            # Re-mask per-model if labels aren't in same order (for controlled they are; for phase_rel sorted by path)
            mae, _, _ = metrics(y_m[mask], yh_m[mask])
            row.append(f"{mae:>14.2f}")
        print(f"{stratum_name:<25} {n:>5}", *row, sep=" ")
    # Overall
    row = []
    for m in models:
        y_m, yh_m = data[m]
        mae, _, _ = metrics(y_m, yh_m)
        row.append(f"{mae:>14.2f}")
    print(f"{'Full cohort':<25} {len(y):>5}", *row, sep=" ")
    print()


def table_with_paired(name, data, ref):
    """data: dict of {model: (y, yhat)}; ref: name of reference model for paired Δ"""
    print("=" * 95)
    print(f"{name} — per-stratum MAE + paired bootstrap Δ vs {ref}")
    print("=" * 95)
    models = list(data.keys())
    y_ref = data[models[0]][0]
    for m in models[1:]:
        assert np.allclose(sorted(y_ref), sorted(data[m][0]), atol=1e-3), f"{m} labels mismatch"
    # Use ref-model's ordering as canonical; require all aligned on label
    # For §08 we sorted by video_path so labels should be aligned one-to-one
    # Validate by checking element-wise equality
    for m in models[1:]:
        if not np.allclose(data[models[0]][0], data[m][0], atol=1e-3):
            print(f"  WARN: {m} labels not aligned with {models[0]} — sorting by label for within-stratum comparison")
    y = data[ref][0]

    print(f"\n{'Stratum':<25} {'N':>5}   {'V-JEPA†-e125 MAE':>18}   {'V4-e25 MAE':>14}   {'ΔMAE (base-V4, pos=V4 better)':>40}")
    for stratum_name, pred in STRATA:
        mask = pred(y)
        n = int(mask.sum())
        if n < 3:
            continue
        ref_y, ref_yh = data[ref]
        mae_ref = mean_absolute_error(ref_y[mask], ref_yh[mask])
        lines = []
        for m in models:
            if m == ref: continue
            y_m, yh_m = data[m]
            mae_m = mean_absolute_error(y_m[mask], yh_m[mask])
            delta = paired_delta(ref_y[mask], yh_m[mask], ref_yh[mask])
            lines.append((m, mae_m, delta))
        for m, mae_m, delta in lines:
            print(f"{stratum_name:<25} {n:>5}   {mae_ref:>18.2f}   {mae_m:>14.2f}   {delta}")
    # Full cohort
    ref_y, ref_yh = data[ref]
    mae_ref = mean_absolute_error(ref_y, ref_yh)
    for m in models:
        if m == ref: continue
        y_m, yh_m = data[m]
        mae_m = mean_absolute_error(y_m, yh_m)
        delta = paired_delta(ref_y, yh_m, ref_yh)
        print(f"{'Full cohort':<25} {len(y):>5}   {mae_ref:>18.2f}   {mae_m:>14.2f}   {delta}")
    print()


def main():
    # §05 controlled
    controlled = {name: load_controlled(path) for name, path in CONTROLLED.items()}
    table("§05 Controlled (e100 init-matched)", controlled)

    # §08 phase-rel: V-JEPA†-e125 + V4-e25
    phase_rel = {name: load_phase_rel(path) for name, path in PHASE_REL.items()}
    # Make sure labels match one-to-one
    y0, _ = phase_rel["V-JEPA†-e125"]
    y1, _ = phase_rel["V4-e25"]
    assert np.allclose(y0, y1, atol=1e-3), "phase-rel labels diverged"
    table_with_paired("§08 Phase-relational (V-JEPA†-e125 vs V4-e25)", phase_rel, ref="V-JEPA†-e125")


if __name__ == "__main__":
    main()
