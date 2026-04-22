"""Random-init ViT-L for the diff-probe random-encoder control.

Uses the same ClipAggregation wrapper as vit_encoder_multiclip.py but
**skips the checkpoint load**, seeding a fresh ViT via torch.manual_seed.
The seed is read from the RANDOM_ENCODER_SEED env var to keep the public
API (which takes `checkpoint: str`) unchanged — the checkpoint path is
ignored when this module is selected.
"""

import logging
import os

import torch

import src.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import (
    ClipAggregation,
)

logger = logging.getLogger(__name__)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,  # ignored
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    seed_str = os.environ.get("RANDOM_ENCODER_SEED", "42")
    seed = int(seed_str)
    logger.info(f"RANDOM-ENCODER mode: seed={seed}, ignoring checkpoint={checkpoint}")

    enc_kwargs = model_kwargs["encoder"]
    enc_model_name = enc_kwargs.get("model_name")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, **enc_kwargs
    )
    logger.info(f"Random-init ViT ({enc_model_name}) built with seed={seed}")

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    return model
