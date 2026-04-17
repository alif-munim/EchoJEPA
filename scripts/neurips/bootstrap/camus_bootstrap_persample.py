"""
Bootstrap CIs from per-sample CAMUS frame shuffling data.

Paired bootstrap over n=100 test samples (50 patients × 2 views),
10K resamples, 95% percentile CIs.
"""

import csv
import numpy as np
from pathlib import Path

MODELS = ['jepa_in21k_e100', 'byol_e100', 'mae_e99', 'salt_s2v1_e79']
LABELS = ['JEPA', 'BYOL', 'MAE', 'SALT']
SAMPLES_DIR = Path('scripts/neurips/samples')
N_BOOT = 10000
RNG = np.random.RandomState(42)


def load_persample(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bootstrap_mean_ci(values):
    """95% CI for mean via bootstrap."""
    values = np.array(values, dtype=float)
    n = len(values)
    boots = np.array([np.mean(RNG.choice(values, n, replace=True)) for _ in range(N_BOOT)])
    return np.mean(values), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired_bootstrap_delta(clean_vals, shuf_vals):
    """Paired bootstrap for % degradation = (clean - shuf) / clean * 100."""
    clean_vals = np.array(clean_vals, dtype=float)
    shuf_vals = np.array(shuf_vals, dtype=float)
    n = len(clean_vals)
    deltas = []
    for _ in range(N_BOOT):
        idx = RNG.choice(n, n, replace=True)
        c = np.mean(clean_vals[idx])
        s = np.mean(shuf_vals[idx])
        deltas.append((c - s) / c * 100 if c > 0 else 0)
    point = (np.mean(clean_vals) - np.mean(shuf_vals)) / np.mean(clean_vals) * 100
    return point, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def get_samples(data, key_field, key_val, seed=None, metric='mean_dice'):
    """Get per-sample values for a condition (averaged across seeds if seed=None)."""
    if seed is not None:
        rows = [r for r in data if r[key_field] == str(key_val) and r['seed'] == str(seed)]
        return [float(r[metric]) for r in rows]
    else:
        # Average across seeds per sample_idx
        from collections import defaultdict
        by_sample = defaultdict(list)
        for r in data:
            if r[key_field] == str(key_val):
                by_sample[r['sample_idx']].append(float(r[metric]))
        return [np.mean(v) for v in sorted(by_sample.items())]


def get_samples_avg_seeds(data, key_field, key_val, metric='mean_dice'):
    """Get per-sample values averaged across seeds."""
    from collections import defaultdict
    by_sample = defaultdict(list)
    for r in data:
        # Float-safe comparison for fraction field
        try:
            match = abs(float(r[key_field]) - float(key_val)) < 1e-6
        except (ValueError, TypeError):
            match = r[key_field] == str(key_val)
        if match:
            by_sample[int(r['sample_idx'])].append(float(r[metric]))
    # Sort by sample_idx for consistent pairing
    return np.array([np.mean(v) for _, v in sorted(by_sample.items())])


# ============================================================
# SEVERITY GRADIENT
# ============================================================
print('=' * 100)
print('SEVERITY GRADIENT — Paired Bootstrap 95% CIs (n=100 samples, 10K resamples)')
print('=' * 100)

fractions = ['0.00', '0.25', '0.50', '0.75', '1.00']
frac_labels = ['0%', '25%', '50%', '75%', '100%']

# Mean Dice
print(f"\n{'Frac':>6}", end='')
for label in LABELS:
    print(f"  {'':>4}{label:>6} [95% CI]{'':>5}", end='')
print()
print('-' * 100)

clean_persample = {}

for frac, flab in zip(fractions, frac_labels):
    print(f"{flab:>6}", end='')
    for model in MODELS:
        data = load_persample(SAMPLES_DIR / f'{model}_camus_severity_persample.csv')
        vals = get_samples_avg_seeds(data, 'fraction', frac)
        if frac == '0.00':
            clean_persample[model] = vals
        mean, lo, hi = bootstrap_mean_ci(vals)
        print(f"  {mean:.4f} [{lo:.4f}, {hi:.4f}]", end='')
    print()

# Degradation
print(f"\nDegradation (%) from clean [paired bootstrap]:")
print(f"{'Frac':>6}", end='')
for label in LABELS:
    print(f"  {'':>3}{label:>6} [95% CI]{'':>4}", end='')
print()
print('-' * 100)

for frac, flab in zip(fractions[1:], frac_labels[1:]):
    print(f"{flab:>6}", end='')
    for model in MODELS:
        data = load_persample(SAMPLES_DIR / f'{model}_camus_severity_persample.csv')
        shuf = get_samples_avg_seeds(data, 'fraction', frac)
        clean = clean_persample[model]
        point, lo, hi = paired_bootstrap_delta(clean, shuf)
        print(f"  {point:>+5.1f}% [{lo:>+5.1f}, {hi:>+5.1f}]", end='')
    print()


# ============================================================
# 6-CONDITION
# ============================================================
print()
print('=' * 100)
print('6-CONDITION — Paired Bootstrap 95% CIs (n=100 samples, 10K resamples)')
print('=' * 100)

conditions = ['clean', 'reverse', 'tubelet', 'matched', 'shuffle', 'matched_frame']

# Mean Dice
print(f"\n{'Cond':>15}", end='')
for label in LABELS:
    print(f"  {'':>4}{label:>6} [95% CI]{'':>5}", end='')
print()
print('-' * 105)

clean6_persample = {}

for cond in conditions:
    print(f"{cond:>15}", end='')
    for model in MODELS:
        data = load_persample(SAMPLES_DIR / f'{model}_camus_6cond_persample.csv')
        vals = get_samples_avg_seeds(data, 'condition', cond)
        if cond == 'clean':
            clean6_persample[model] = vals
        mean, lo, hi = bootstrap_mean_ci(vals)
        print(f"  {mean:.4f} [{lo:.4f}, {hi:.4f}]", end='')
    print()

# Degradation
print(f"\nDegradation (%) from clean [paired bootstrap]:")
print(f"{'Cond':>15}", end='')
for label in LABELS:
    print(f"  {'':>3}{label:>6} [95% CI]{'':>4}", end='')
print()
print('-' * 105)

for cond in conditions[1:]:
    print(f"{cond:>15}", end='')
    for model in MODELS:
        data = load_persample(SAMPLES_DIR / f'{model}_camus_6cond_persample.csv')
        shuf = get_samples_avg_seeds(data, 'condition', cond)
        clean = clean6_persample[model]
        point, lo, hi = paired_bootstrap_delta(clean, shuf)
        print(f"  {point:>+5.1f}% [{lo:>+5.1f}, {hi:>+5.1f}]", end='')
    print()


# ============================================================
# PER-STRUCTURE (matched_frame)
# ============================================================
print()
print('=' * 100)
print('PER-STRUCTURE — matched_frame [paired bootstrap]')
print('=' * 100)

structures = ['LV', 'MYO', 'LA']
for struct in structures:
    metric = f'mean_{struct}_dice' if struct != 'mean' else 'mean_dice'
    metric = f'{struct}_dice'
    print(f"\n{struct}:")
    print(f"  {'Model':>6}  {'Clean':>18}  {'Shuffled':>22}  {'Δ% [95% CI]':>22}")
    print(f"  {'-'*75}")
    for model, label in zip(MODELS, LABELS):
        data_clean = load_persample(SAMPLES_DIR / f'{model}_camus_6cond_persample.csv')
        clean = get_samples_avg_seeds(data_clean, 'condition', 'clean', f'mean_{struct}_dice')
        shuf = get_samples_avg_seeds(data_clean, 'condition', 'matched_frame', f'mean_{struct}_dice')
        mc, lc, hc = bootstrap_mean_ci(clean)
        ms, ls, hs = bootstrap_mean_ci(shuf)
        dp, dl, dh = paired_bootstrap_delta(clean, shuf)
        print(f"  {label:>6}  {mc:.4f} [{lc:.4f},{hc:.4f}]  {ms:.4f} [{ls:.4f},{hs:.4f}]  {dp:>+5.1f}% [{dl:>+4.1f},{dh:>+4.1f}]")


# ============================================================
# ED vs ES (matched_frame)
# ============================================================
print()
print('=' * 100)
print('ED vs ES — matched_frame [paired bootstrap]')
print('=' * 100)

for phase in ['ed', 'es']:
    print(f"\n{phase.upper()} (mean across LV/MYO/LA):")
    print(f"  {'Model':>6}  {'Clean':>18}  {'Shuffled':>22}  {'Δ% [95% CI]':>22}")
    print(f"  {'-'*75}")
    for model, label in zip(MODELS, LABELS):
        data = load_persample(SAMPLES_DIR / f'{model}_camus_6cond_persample.csv')
        # Per-sample: average across 3 structures for this phase
        from collections import defaultdict

        def get_phase_dice(cond_name):
            by_sample = defaultdict(list)
            for r in data:
                if r['condition'] == cond_name:
                    phase_dice = np.mean([float(r[f'{phase}_{s}_dice']) for s in structures])
                    by_sample[int(r['sample_idx'])].append(phase_dice)
            return np.array([np.mean(v) for _, v in sorted(by_sample.items())])

        clean = get_phase_dice('clean')
        shuf = get_phase_dice('matched_frame')
        mc, lc, hc = bootstrap_mean_ci(clean)
        ms, ls, hs = bootstrap_mean_ci(shuf)
        dp, dl, dh = paired_bootstrap_delta(clean, shuf)
        print(f"  {label:>6}  {mc:.4f} [{lc:.4f},{hc:.4f}]  {ms:.4f} [{ls:.4f},{hs:.4f}]  {dp:>+5.1f}% [{dl:>+4.1f},{dh:>+4.1f}]")
