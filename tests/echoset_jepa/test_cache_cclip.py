"""Cache cclip tests — path helpers, resume filter, index merge.

These tests do not touch GPU. The GPU-dependent worker path is exercised
by the --dry_run 8 mode in the sbatch pipeline before a full cache job.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.echoset_jepa.cache_cclip import (
    CacheConfig,
    _cache_path,
    _filter_to_cache,
    merge_index,
    _write_npy,
    _exists,
)


def test_cache_path_local():
    p = _cache_path("/tmp/foo", "sABC", "clip1")
    assert p == "/tmp/foo/sABC/clip1.npy"


def test_cache_path_s3():
    p = _cache_path("s3://bucket/prefix", "sABC", "clip1")
    assert p == "s3://bucket/prefix/sABC/clip1.npy"


def test_cache_path_trims_trailing_slash():
    p = _cache_path("s3://bucket/prefix/", "sABC", "clip1")
    assert p == "s3://bucket/prefix/sABC/clip1.npy"


def test_write_and_read_local(tmp_path):
    arr = np.random.randn(1024).astype(np.float32)
    target = str(tmp_path / "sX" / "c1.npy")
    _write_npy(target, arr)
    assert _exists(target)
    back = np.load(target)
    assert np.allclose(back, arr)


def test_filter_to_cache_resume(tmp_path):
    # Create one cached file; filter should drop its row
    target = tmp_path / "s1" / "c1.npy"
    target.parent.mkdir(parents=True)
    np.save(target, np.zeros(4))
    rows = [
        {"study_id": "s1", "clip_id": "c1"},    # already cached
        {"study_id": "s1", "clip_id": "c2"},    # not cached
        {"study_id": "s2", "clip_id": "c3"},    # not cached
    ]
    kept = _filter_to_cache(rows, str(tmp_path), force=False)
    assert len(kept) == 2
    assert all(r["clip_id"] != "c1" for r in kept)


def test_filter_to_cache_force_keeps_all(tmp_path):
    target = tmp_path / "s1" / "c1.npy"
    target.parent.mkdir(parents=True)
    np.save(target, np.zeros(4))
    rows = [{"study_id": "s1", "clip_id": "c1"}, {"study_id": "s1", "clip_id": "c2"}]
    kept = _filter_to_cache(rows, str(tmp_path), force=True)
    assert len(kept) == 2


def test_merge_index_concats_shards(tmp_path):
    shard0 = pd.DataFrame([{"clip_id": "a", "study_id": "s1", "cached_path": "/tmp/s1/a.npy",
                           "checkpoint_id": "ckpt", "checksum": "aa", "embedding_dim": 1024,
                           "dtype": "float32", "created_at": "t"}])
    shard1 = pd.DataFrame([{"clip_id": "b", "study_id": "s2", "cached_path": "/tmp/s2/b.npy",
                           "checkpoint_id": "ckpt", "checksum": "aa", "embedding_dim": 1024,
                           "dtype": "float32", "created_at": "t"}])
    shard0.to_parquet(tmp_path / "cache_index_rank0.parquet", index=False)
    shard1.to_parquet(tmp_path / "cache_index_rank1.parquet", index=False)
    out = tmp_path / "cache_index.parquet"
    merge_index(str(tmp_path), str(out), world_size=2)
    merged = pd.read_parquet(out)
    assert len(merged) == 2
    assert set(merged["clip_id"]) == {"a", "b"}


def test_merge_index_dedupes_on_clip_id(tmp_path):
    shard0 = pd.DataFrame([{"clip_id": "a", "study_id": "s1", "cached_path": "/p0",
                           "checkpoint_id": "ck", "checksum": "h", "embedding_dim": 1024,
                           "dtype": "float32", "created_at": "t"}])
    shard1 = pd.DataFrame([{"clip_id": "a", "study_id": "s1", "cached_path": "/p1",
                           "checkpoint_id": "ck", "checksum": "h", "embedding_dim": 1024,
                           "dtype": "float32", "created_at": "t"}])
    shard0.to_parquet(tmp_path / "cache_index_rank0.parquet", index=False)
    shard1.to_parquet(tmp_path / "cache_index_rank1.parquet", index=False)
    out = tmp_path / "cache_index.parquet"
    merge_index(str(tmp_path), str(out), world_size=2)
    merged = pd.read_parquet(out)
    assert len(merged) == 1


def test_cache_config_dataclass():
    cfg = CacheConfig(config_path="x.yaml", cache_prefix="/tmp/c")
    assert cfg.batch_size == 16
    assert cfg.dry_run == 0
