"""EchoSet-JEPA downstream probe + controls (plan §7, §8).

All controls consume the same ``study_clip_sample_K{K}_seed{S}.parquet`` and
apply the same probe head (plan §8.6). Each ``control_*`` module has a
``main(cfg)`` entry that mirrors the shape of ``evals.scaffold.main``.
"""
