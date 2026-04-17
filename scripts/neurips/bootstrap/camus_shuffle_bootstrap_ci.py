"""
Bootstrap CIs for CAMUS frame shuffling: severity gradient + 6-condition.

Computes 95% CIs for mean Dice and degradation (Δ from clean) using
paired bootstrap (same test samples in clean and shuffled).
"""

import csv
import numpy as np
from pathlib import Path

MODELS = ['jepa_in21k_e100', 'byol_e100', 'mae_e99', 'salt_s2v1_e79']
LABELS = ['JEPA IN21K', 'BYOL', 'MAE', 'SALT S2']
SAMPLES_DIR = Path('scripts/neurips/samples')
N_BOOTSTRAP = 10000
SEED = 42


def load_csv(path):
    """Load CSV into list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return np.mean(values), lo, hi


def bootstrap_delta_ci(clean_vals, shuffled_vals, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for the degradation (clean - shuffled) / clean."""
    rng = np.random.RandomState(seed)
    clean_vals = np.array(clean_vals)
    shuffled_vals = np.array(shuffled_vals)
    n = len(clean_vals)
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        c = np.mean(clean_vals[idx])
        s = np.mean(shuffled_vals[idx])
        if c > 0:
            boot_deltas.append((c - s) / c * 100)
    boot_deltas = np.array(boot_deltas)
    lo, hi = np.percentile(boot_deltas, [2.5, 97.5])
    mean_c = np.mean(clean_vals)
    mean_s = np.mean(shuffled_vals)
    point = (mean_c - mean_s) / mean_c * 100 if mean_c > 0 else 0
    return point, lo, hi


def get_per_sample_dices(rows):
    """Average per-seed results into a single vector of per-sample-like Dice values.

    Since we don't have per-sample data, we treat each seed's mean_dice as one observation.
    For paired bootstrap on degradation, we use the per-structure Dice across seeds.
    """
    return [float(r['mean_dice']) for r in rows]


def get_structure_dices(rows, structure='mean'):
    """Get Dice values for a specific structure across seeds."""
    key = f'{structure}_dice' if structure != 'mean' else 'mean_dice'
    return [float(r[key]) for r in rows]


# ============================================================
# Severity gradient
# ============================================================
print('=' * 90)
print('SEVERITY GRADIENT — Bootstrap 95% CIs')
print('=' * 90)

# Header
print(f"\n{'Fraction':>8}  ", end='')
for label in LABELS:
    print(f"  {label:>22}", end='')
print()
print('-' * 100)

fractions = [0.0, 0.25, 0.50, 0.75, 1.0]

# Store clean values for delta computation
clean_by_model = {}

for frac in fractions:
    pct = int(frac * 100)
    row = f"{pct:>6}%  "
    for model in MODELS:
        data = load_csv(SAMPLES_DIR / f'{model}_camus_severity.csv')
        vals = [float(r['mean_dice']) for r in data if abs(float(r['fraction']) - frac) < 0.01]
        if frac == 0.0:
            clean_by_model[model] = vals
        mean, lo, hi = bootstrap_ci(vals)
        row += f"  {mean:.4f} [{lo:.4f},{hi:.4f}]"
    print(row)

# Degradation with CIs
print(f"\nDegradation (%) from clean:")
print(f"{'Fraction':>8}  ", end='')
for label in LABELS:
    print(f"  {label:>22}", end='')
print()
print('-' * 100)

for frac in [0.25, 0.50, 0.75, 1.0]:
    pct = int(frac * 100)
    row = f"{pct:>6}%  "
    for model in MODELS:
        data = load_csv(SAMPLES_DIR / f'{model}_camus_severity.csv')
        clean_vals = np.array(clean_by_model[model])
        shuf_vals = np.array([float(r['mean_dice']) for r in data
                              if abs(float(r['fraction']) - frac) < 0.01])
        # Since we only have 3 seeds per condition, bootstrap on those
        point, lo, hi = bootstrap_delta_ci(clean_vals, shuf_vals)
        row += f"   {point:>+5.1f}% [{lo:>+5.1f},{hi:>+5.1f}]"
    print(row)


# ============================================================
# 6-condition
# ============================================================
print()
print('=' * 90)
print('6-CONDITION ABLATION — Bootstrap 95% CIs')
print('=' * 90)

conditions = ['clean', 'reverse', 'tubelet', 'matched', 'shuffle', 'matched_frame']

# Dice values
print(f"\n{'Condition':>15}  ", end='')
for label in LABELS:
    print(f"  {label:>22}", end='')
print()
print('-' * 105)

clean6_by_model = {}

for cond in conditions:
    row = f"{cond:>15}  "
    for model in MODELS:
        data = load_csv(SAMPLES_DIR / f'{model}_camus_6cond.csv')
        vals = [float(r['mean_dice']) for r in data if r['condition'] == cond]
        if cond == 'clean':
            clean6_by_model[model] = vals
        mean, lo, hi = bootstrap_ci(vals)
        if len(vals) == 1:
            row += f"  {mean:.4f}                "
        else:
            row += f"  {mean:.4f} [{lo:.4f},{hi:.4f}]"
    print(row)

# Degradation with CIs
print(f"\nDegradation (%) from clean:")
print(f"{'Condition':>15}  ", end='')
for label in LABELS:
    print(f"  {label:>22}", end='')
print()
print('-' * 105)

for cond in conditions[1:]:
    row = f"{cond:>15}  "
    for model in MODELS:
        data = load_csv(SAMPLES_DIR / f'{model}_camus_6cond.csv')
        clean_vals = np.array(clean6_by_model[model])
        shuf_vals = np.array([float(r['mean_dice']) for r in data if r['condition'] == cond])
        point, lo, hi = bootstrap_delta_ci(clean_vals, shuf_vals)
        row += f"   {point:>+5.1f}% [{lo:>+5.1f},{hi:>+5.1f}]"
    print(row)


# ============================================================
# Per-structure breakdown for 6-condition (full shuffle only)
# ============================================================
print()
print('=' * 90)
print('PER-STRUCTURE BREAKDOWN — matched_frame condition')
print('=' * 90)

structures = ['LV', 'MYO', 'LA']
for struct in structures:
    print(f"\n{struct}:")
    print(f"  {'Model':>12}  {'Clean':>8}  {'Shuffled':>22}  {'Δ%':>22}")
    print(f"  {'-'*70}")
    for model, label in zip(MODELS, LABELS):
        data = load_csv(SAMPLES_DIR / f'{model}_camus_6cond.csv')
        clean_val = [float(r[f'{struct}_dice']) for r in data if r['condition'] == 'clean']
        shuf_vals = [float(r[f'{struct}_dice']) for r in data if r['condition'] == 'matched_frame']
        mean_c = np.mean(clean_val)
        mean_s, lo_s, hi_s = bootstrap_ci(shuf_vals)
        drop = (mean_c - mean_s) / mean_c * 100
        print(f"  {label:>12}  {mean_c:.4f}  {mean_s:.4f} [{lo_s:.4f},{hi_s:.4f}]  {drop:>+5.1f}%")


# ============================================================
# ED vs ES breakdown for matched_frame
# ============================================================
print()
print('=' * 90)
print('ED vs ES PHASE — matched_frame condition')
print('=' * 90)

for phase in ['ed', 'es']:
    print(f"\n{phase.upper()} Mean Dice:")
    print(f"  {'Model':>12}  {'Clean':>8}  {'Shuffled':>22}  {'Δ%':>8}")
    print(f"  {'-'*60}")
    for model, label in zip(MODELS, LABELS):
        data = load_csv(SAMPLES_DIR / f'{model}_camus_6cond.csv')
        # Compute mean across structures for this phase
        clean_rows = [r for r in data if r['condition'] == 'clean']
        shuf_rows = [r for r in data if r['condition'] == 'matched_frame']

        clean_dice = np.mean([
            np.mean([float(r[f'{phase}_{s}_dice']) for s in structures])
            for r in clean_rows
        ])
        shuf_dices = [
            np.mean([float(r[f'{phase}_{s}_dice']) for s in structures])
            for r in shuf_rows
        ]
        mean_s, lo_s, hi_s = bootstrap_ci(shuf_dices)
        drop = (clean_dice - mean_s) / clean_dice * 100
        print(f"  {label:>12}  {clean_dice:.4f}  {mean_s:.4f} [{lo_s:.4f},{hi_s:.4f}]  {drop:>+5.1f}%")
