"""EchoMV-JEPA model package.

See ``docs/echomv_jepa_architecture_plan.md`` §7 for the full architecture.
Stage-1 MVP exports the new EMA full-study teacher wrapper, a per-modality
projector, and the core losses. CALA, target anchoring, and adapter-joint
clip training are deferred follow-ups.
"""

from .ema import ema_update_
from .losses import (
    cosine_regress,
    covariance_penalty,
    layernorm_cosine,
    matched_nce,
    matched_rank_metrics,
    prioritized_neg_pool,
)
from .modality_projector import ModalityProjectorPair
from .study_target_encoder import StudyTransformerEMA
from .token_study_transformer import TokenStudyTransformer

# OnlineVJepaEncoder lives behind a lazy import — the module imports heavy
# V-JEPA evaluation modules that aren't needed for the pooled-cache path and
# that complicate pytest collection on CPU-only environments.

__all__ = [
    "StudyTransformerEMA",
    "ModalityProjectorPair",
    "TokenStudyTransformer",
    "ema_update_",
    "cosine_regress",
    "covariance_penalty",
    "layernorm_cosine",
    "matched_nce",
    "matched_rank_metrics",
    "prioritized_neg_pool",
]
