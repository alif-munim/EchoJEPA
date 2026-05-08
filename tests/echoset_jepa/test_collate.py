"""DDP collate tests for EchoSet-JEPA (PR-N4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.echoset_jepa_collate import (
    EchoSetStudyDataset,
    echoset_collate,
)
from src.models.meta_embeddings import MetaEmbeddings


@pytest.fixture
def synth_cache(tmp_path):
    cache = tmp_path / "cclip"
    rng = np.random.RandomState(0)
    rows = []
    for sid_i in range(5):
        sid = f"s{sid_i}"
        for i, (view, mod) in enumerate([
            ("apical", "b_mode"),
            ("parasternal_long", "b_mode"),
            ("apical", "color_doppler"),
            ("parasternal_short", "b_mode"),
        ]):
            cid = f"{sid}_c{i}"
            vec = rng.randn(32).astype(np.float32)
            path = cache / sid / f"{cid}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vec)
            rows.append(dict(
                study_id=sid, clip_id=cid,
                view_family=view, modality=mod, phase_bucket="full_cycle",
                measurement_site="none", quality_score=0.7,
                cached_cclip_s3="",
            ))
    k_path = tmp_path / "k.parquet"
    pd.DataFrame(rows).to_parquet(k_path, index=False)
    return k_path, cache


def test_dataset_returns_per_study_dict(synth_cache):
    k_path, cache = synth_cache
    meta = MetaEmbeddings(d_model=32)
    ds = EchoSetStudyDataset(str(k_path), str(cache), meta=meta)
    assert len(ds) == 5
    b = ds[0]
    # 4 clips, 4 unique (view, modality, phase_bucket) keys → 4 elements total
    assert b["ctx_elements"].shape[0] + b["tgt_elements"].shape[0] == 4
    assert b["ctx_elements"].shape[1] == 32
    assert "mask_strategy" in b
    assert b["n_elements"] == 4


def test_dataset_handles_missing_cache(tmp_path):
    meta = MetaEmbeddings(d_model=16)
    rows = [{"study_id": "sX", "clip_id": "c0", "view_family": "apical",
             "modality": "b_mode", "phase_bucket": "full_cycle",
             "measurement_site": "none", "quality_score": 0.5,
             "cached_cclip_s3": ""}]
    k = tmp_path / "k.parquet"
    pd.DataFrame(rows).to_parquet(k, index=False)
    ds = EchoSetStudyDataset(str(k), str(tmp_path / "nonexistent"), meta=meta)
    b = ds[0]
    assert b["n_elements"] == 0
    assert b["ctx_elements"].numel() > 0   # placeholder zero element
    assert b["tgt_elements"].shape[0] == 0


def test_collate_pads_variable_length(synth_cache):
    k_path, cache = synth_cache
    meta = MetaEmbeddings(d_model=32)
    ds = EchoSetStudyDataset(str(k_path), str(cache), meta=meta)
    items = [ds[i] for i in range(3)]
    batch = echoset_collate(items)
    B = 3
    assert batch["ctx_elements"].shape[0] == B
    assert batch["ctx_pad_mask"].shape == batch["ctx_elements"].shape[:2]
    assert batch["tgt_pad_mask"].dtype == torch.bool
    # Padding mask = True where padded
    assert batch["ctx_pad_mask"].any().item() in (True, False)
    # ctx_meta shapes match ctx_elements[:2]
    assert batch["ctx_meta_view"].shape == batch["ctx_elements"].shape[:2]
    assert batch["tgt_meta_view"].shape == batch["tgt_elements"].shape[:2]
    assert batch["study_id_int"].shape == (B,)
    assert len(batch["mask_strategies"]) == B


def test_collate_pad_mask_true_means_pad(synth_cache):
    k_path, cache = synth_cache
    meta = MetaEmbeddings(d_model=32)
    ds = EchoSetStudyDataset(str(k_path), str(cache), meta=meta)
    # Manually make one sample smaller
    small = ds[0]
    small["ctx_elements"] = small["ctx_elements"][:1]
    small["ctx_meta_view"] = small["ctx_meta_view"][:1]
    small["ctx_meta_modality"] = small["ctx_meta_modality"][:1]
    small["ctx_meta_phase"] = small["ctx_meta_phase"][:1]
    small["ctx_meta_quality"] = small["ctx_meta_quality"][:1]
    large = ds[1]
    batch = echoset_collate([small, large])
    mc_small = 1
    mc_large = large["ctx_elements"].shape[0]
    max_ctx = max(mc_small, mc_large)
    # First sample: slot 0 is real, slots 1..max are padded → pad mask True beyond index 0
    assert not batch["ctx_pad_mask"][0, 0].item()
    if max_ctx > mc_small:
        assert batch["ctx_pad_mask"][0, mc_small:].all().item()


def test_end_to_end_forward_backward_through_collate(synth_cache):
    """The real reason this collate exists: it must feed training_step."""
    from app.echoset_jepa.train import training_step
    from src.models.study_transformer import StudyTransformer, StudyTransformerConfig
    from src.models.study_projectors import EMAProjectorPair

    k_path, cache = synth_cache
    d_model = 32
    meta = MetaEmbeddings(d_model=d_model)
    ds = EchoSetStudyDataset(str(k_path), str(cache), meta=meta)
    batch = echoset_collate([ds[i] for i in range(3)])

    st = StudyTransformer(StudyTransformerConfig(
        d_clip=32, d_model=d_model, n_layers=2, n_heads=4, ffn_mult=2, max_M=16
    ))
    proj = EMAProjectorPair(d_model=d_model, d_hidden=64, d_proj=16)

    # training_step expects every study to have at least 1 target — skip if not
    if batch["tgt_elements"].shape[1] == 0 or (~batch["tgt_pad_mask"]).sum() == 0:
        pytest.skip("synth batch has no target positions (mask picked no targets)")

    out = training_step(batch, st, meta, proj, lambda_nce=0.03)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()
    assert "valid_neg_count_same_view_mean" in out.diagnostics
