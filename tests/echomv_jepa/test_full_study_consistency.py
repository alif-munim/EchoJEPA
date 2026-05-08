"""Dataset-level invariants for EchoMV-JEPA (PR-2).

Verifies that the new collate builds ``full_elements`` and index tensors that
are consistent with the already-padded ``ctx_elements`` and ``tgt_elements``.
"""

from __future__ import annotations

import torch

from src.datasets.echomv_jepa_dataset import EchoMVJEPADataset, echomv_collate


def _batch(synth_cache, meta32):
    k_path, cache = synth_cache
    ds = EchoMVJEPADataset(str(k_path), str(cache), meta=meta32)
    items = [ds[i] for i in range(4)]
    return echomv_collate(items)


def test_full_elements_is_ctx_then_tgt(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    ctx = b["ctx_elements"]
    tgt = b["tgt_elements"]
    full = b["full_elements"]
    assert full.shape[1] == ctx.shape[1] + tgt.shape[1]
    assert full.shape[0] == ctx.shape[0] == tgt.shape[0]
    assert torch.equal(full[:, : ctx.shape[1]], ctx)
    assert torch.equal(full[:, ctx.shape[1] :], tgt)


def test_target_idx_in_full_gathers_tgt_elements(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    full = b["full_elements"]
    tgt = b["tgt_elements"]
    tgt_idx = b["target_idx_in_full"]
    idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
    gathered = torch.gather(full, dim=1, index=idx_exp)
    assert torch.equal(gathered, tgt)


def test_context_idx_in_full_gathers_ctx_elements(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    full = b["full_elements"]
    ctx = b["ctx_elements"]
    ctx_idx = b["context_idx_in_full"]
    idx_exp = ctx_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
    gathered = torch.gather(full, dim=1, index=idx_exp)
    assert torch.equal(gathered, ctx)


def test_ctx_and_tgt_index_sets_disjoint(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    for i in range(b["ctx_elements"].shape[0]):
        ctx_i = set(b["context_idx_in_full"][i].tolist())
        tgt_i = set(b["target_idx_in_full"][i].tolist())
        assert ctx_i.isdisjoint(tgt_i), f"row {i}: ctx∩tgt = {ctx_i & tgt_i}"


def test_full_pad_mask_matches_ctx_tgt_concat(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    pad = torch.cat([b["ctx_pad_mask"], b["tgt_pad_mask"]], dim=1)
    assert torch.equal(pad, b["full_pad_mask"])


def test_full_meta_concat_matches(synth_cache, meta32):
    b = _batch(synth_cache, meta32)
    for k in ("view", "modality", "phase", "quality"):
        concat = torch.cat([b[f"ctx_meta_{k}"], b[f"tgt_meta_{k}"]], dim=1)
        assert torch.equal(concat, b[f"full_meta_{k}"])
