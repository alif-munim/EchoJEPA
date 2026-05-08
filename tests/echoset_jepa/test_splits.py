"""Patient-level split tests."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from experiments.echoset_jepa.splits import assign_split, build_split


def test_assign_split_deterministic():
    assert assign_split("p1", seed=0) == assign_split("p1", seed=0)
    assert assign_split("p42", seed=7) == assign_split("p42", seed=7)


def test_assign_split_seed_changes_assignment():
    # Different seeds should (typically) move some patients between splits.
    diffs = sum(
        1 for i in range(100) if assign_split(f"p{i}", seed=0) != assign_split(f"p{i}", seed=1)
    )
    assert diffs > 0, "seed has no effect on split assignment"


def test_assign_split_respects_fractions():
    n = 10000
    assignments = [assign_split(f"p{i}", seed=0, train_frac=0.85, val_frac=0.075) for i in range(n)]
    counts = Counter(assignments)
    # Within ~2% of target for n=10k
    assert abs(counts["train"] / n - 0.85) < 0.02
    assert abs(counts["val"] / n - 0.075) < 0.015
    assert abs(counts["test"] / n - 0.075) < 0.015


def test_build_split_no_patient_leakage(tmp_path):
    # 5 patients × 3 clips each
    rows = []
    for pid in range(5):
        for i in range(3):
            rows.append({"patient_id": f"p{pid}", "study_id": f"s{pid}", "clip_id": f"c{pid}_{i}"})
    df = pd.DataFrame(rows)
    in_path = tmp_path / "m.parquet"
    df.to_parquet(in_path, index=False)
    out_path = tmp_path / "m_split.parquet"
    build_split(str(in_path), str(out_path), seed=0)
    out = pd.read_parquet(out_path)
    per_pt = out.groupby("patient_id")["split"].nunique()
    assert (per_pt == 1).all(), "patient leaked across splits"


def test_build_split_raises_on_invalid_fractions(tmp_path):
    df = pd.DataFrame([{"patient_id": "p1", "study_id": "s1", "clip_id": "c1"}])
    in_path = tmp_path / "m.parquet"
    df.to_parquet(in_path, index=False)
    with pytest.raises(ValueError):
        build_split(str(in_path), str(tmp_path / "out.parquet"), train_frac=0.9, val_frac=0.2)
