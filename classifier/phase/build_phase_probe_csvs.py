"""Build phase-probe train/val/test CSVs for the evals.video_classification_frozen
pipeline.

Emits two CSVs per split (sin and cos), space-delimited:
    <s3_uri> <target>

Target column is either sin(2π·phase_anchor) or cos(2π·phase_anchor).

Anchor = middle of the confident-mask run in the clip's per-frame phase
vector. The existing VideoDataset samples a random 16-frame window, so
we cannot guarantee that the specific anchor frame is inside every
clip in the batch — the frozen encoder consumes the full clip and the
probe reads a pooled representation, so what we actually train on is
"representation of a clip whose middle frame has this phase." That is
the exact quantity we want to probe.

Filters applied before sampling:
    quality_tier ∈ {high}   (strict baseline)
    strict RR-consistency  (inherited via splits/)
    confident anchor (≥8 confident frames, anchor = median of them)

Subsampling:
    cap per-subject clips
    phase-bin balance via 10 bins in [0,1)
    target sizes: 10000 / 1500 / 3000

Out:
    data/csv/mimic_phase_sin_{train,val,test}_10k.csv
    data/csv/mimic_phase_cos_{train,val,test}_10k.csv
    data/csv/mimic_phase_anchors_10k.csv   (phase + dicom_id, for diagnostics)
"""
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_PARQ   = HERE / 'phase_annotations' / 'phase_annotations.parquet'
DEFAULT_SPLITS = HERE / 'splits' / 'dicoms_split.csv'


def pick_anchor(phases, mask):
    if not phases or not mask or len(phases) != len(mask):
        return None
    confs = [(i, p) for i, (p, m) in enumerate(zip(phases, mask)) if m and p is not None]
    if len(confs) < 8:
        return None
    mid_idx = len(confs) // 2
    return confs[mid_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=DEFAULT_PARQ)
    ap.add_argument('--splits',  type=Path, default=DEFAULT_SPLITS)
    ap.add_argument('--out-dir', type=Path, default=Path('data/csv'))
    ap.add_argument('--n-train', type=int, default=10000)
    ap.add_argument('--n-val',   type=int, default=1500)
    ap.add_argument('--n-test',  type=int, default=3000)
    ap.add_argument('--max-per-subject', type=int, default=6)
    ap.add_argument('--phase-bins', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quality-tiers', nargs='+', default=['high'])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    splits = pd.read_csv(args.splits)
    df = pd.read_parquet(args.parquet, columns=[
        'dicom_id', 'subject_id', 'study_id', 's3_uri',
        'per_frame_phase_json', 'confident_mask_json', 'quality_tier',
    ])
    df = df[df.quality_tier.isin(args.quality_tiers)]
    m = df.merge(splits[['dicom_id', 'split']], on='dicom_id', how='inner')
    print(f"After quality filter + split merge: {len(m)} clips "
          f"({m['split'].value_counts().to_dict()})")

    records = []
    for _, r in m.iterrows():
        try:
            phases = json.loads(r['per_frame_phase_json'])
            mask   = json.loads(r['confident_mask_json'])
        except Exception:
            continue
        anc = pick_anchor(phases, mask)
        if anc is None:
            continue
        anchor_idx, phase = anc
        records.append({
            'dicom_id': r['dicom_id'],
            'subject_id': r['subject_id'],
            'study_id': r['study_id'],
            's3_uri': r['s3_uri'],
            'split': r['split'],
            'anchor_idx': anchor_idx,
            'phase': phase,
            'sin': math.sin(2 * math.pi * phase),
            'cos': math.cos(2 * math.pi * phase),
        })
    anchors = pd.DataFrame(records)
    print(f"With confident anchor: {len(anchors)} clips "
          f"({anchors['split'].value_counts().to_dict()})")

    # Phase-bin-balanced subsample per split, capped per subject
    def subsample(split_df, n_target):
        bins = np.linspace(0, 1, args.phase_bins + 1)
        split_df = split_df.copy()
        split_df['bin'] = np.digitize(split_df.phase, bins[1:-1])
        per_bin = max(1, n_target // args.phase_bins)
        chosen = []
        for b in range(args.phase_bins):
            bin_df = split_df[split_df.bin == b]
            # Shuffle then cap per subject
            bin_df = bin_df.sample(frac=1, random_state=args.seed + b).reset_index(drop=True)
            by_sub = defaultdict(int)
            out = []
            for _, row in bin_df.iterrows():
                if by_sub[row.subject_id] >= args.max_per_subject:
                    continue
                out.append(row)
                by_sub[row.subject_id] += 1
                if len(out) >= per_bin:
                    break
            chosen.extend(out)
        # If we undershot, top up with random across bins (still subject-capped)
        need = n_target - len(chosen)
        chosen_ids = {r.dicom_id for r in chosen}
        if need > 0:
            extra_pool = split_df[~split_df.dicom_id.isin(chosen_ids)]
            extra_pool = extra_pool.sample(frac=1, random_state=args.seed + 999).reset_index(drop=True)
            by_sub = defaultdict(int)
            for r in chosen:
                by_sub[r.subject_id] += 1
            for _, row in extra_pool.iterrows():
                if by_sub[row.subject_id] >= args.max_per_subject:
                    continue
                chosen.append(row)
                by_sub[row.subject_id] += 1
                if len(chosen) >= n_target:
                    break
        return pd.DataFrame(chosen).reset_index(drop=True)

    parts = {}
    for split, n in [('train', args.n_train), ('val', args.n_val), ('test', args.n_test)]:
        parts[split] = subsample(anchors[anchors.split == split], n)
        print(f"{split}: {len(parts[split])} clips, "
              f"{parts[split]['subject_id'].nunique()} subjects, "
              f"phase bin distribution (10 bins): "
              f"{np.bincount(np.digitize(parts[split].phase, np.linspace(0,1,args.phase_bins+1)[1:-1]), minlength=args.phase_bins).tolist()}")

    # Assert subject-disjoint
    s_train = set(parts['train'].subject_id.astype(str).unique())
    s_val   = set(parts['val'].subject_id.astype(str).unique())
    s_test  = set(parts['test'].subject_id.astype(str).unique())
    assert s_train.isdisjoint(s_val),  f"train/val subject overlap: {len(s_train & s_val)}"
    assert s_train.isdisjoint(s_test), f"train/test subject overlap: {len(s_train & s_test)}"
    assert s_val.isdisjoint(s_test),   f"val/test subject overlap: {len(s_val & s_test)}"
    print("Subject-disjoint splits verified.")

    # Emit probe CSVs: space-delimited, three columns
    # (s3_uri, anchor_frame, target). VideoDataset auto-detects 3-col
    # format and switches on anchor-aware sampling.
    #
    # Map the parquet's DICOM-source URI to the MP4 mirror used by
    # downstream probes (decord cannot read DICOMs over S3):
    #   s3://echodata25/mimic-raw-staging/....dcm
    #   → s3://echodata25/mimic-echo-224px/....mp4
    def dcm_to_mp4(u):
        return u.replace('/mimic-raw-staging/', '/mimic-echo-224px/').replace('.dcm', '.mp4')

    for split, p in parts.items():
        for target in ('sin', 'cos'):
            out = args.out_dir / f'mimic_phase_{target}_{split}_10k.csv'
            with out.open('w') as f:
                for _, r in p.iterrows():
                    f.write(f"{dcm_to_mp4(r.s3_uri)} {int(r.anchor_idx)} {r[target]:.6f}\n")
            print(f"wrote {out} ({len(p)} rows)")

    # Diagnostics CSV with full anchor info
    diag = pd.concat([v.assign(split=k) for k, v in parts.items()])
    diag[['dicom_id','subject_id','study_id','split','anchor_idx','phase','sin','cos','s3_uri']].to_csv(
        args.out_dir / 'mimic_phase_anchors_10k.csv', index=False
    )
    print(f"wrote {args.out_dir/'mimic_phase_anchors_10k.csv'}")


if __name__ == '__main__':
    main()
