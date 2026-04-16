"""
Shared model registry for rebuttal scripts.

Centralizes checkpoint paths and model configs so frame_shuffling.py,
cka_speckle.py, and noise_level_probe.py can all use --models to select
which models to run, enabling parallelization across GPUs/machines.

Usage:
    from scripts.neurips.model_registry import ALL_MODELS, get_models

    models = get_models(args.models)  # None = all, or ["JEPA-L-pt50", "BYOL-L-pt50"]
"""

# All available models with their configs
ALL_MODELS = {
    # --- Fully-trained (preprint models) ---
    "JEPA-G": {
        "checkpoint": "checkpoints/anneal/keep/pt-280-an81.pt",
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "fully-trained",
    },
    "JEPA-L": {
        "checkpoint": "checkpoints/anneal/keep/vitl-pt-210-an25.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "fully-trained",
    },
    "MAE-L": {
        "checkpoint": "checkpoints/videomae-ep163.pth",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
        "type": "videomae",
        "group": "fully-trained",
    },
    # --- pt50 controlled comparison ---
    "JEPA-L-pt50": {
        "checkpoint": "checkpoints/echojepa-l-pt50.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "pt50",
    },
    "BYOL-L-pt50": {
        "checkpoint": "checkpoints/byol_vitl_imagenet_v2_e50.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "pt50",
    },
    "MAE-L-pt50": {
        "checkpoint": "checkpoints/videomae_l_mimic_ep50.pth",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
        "type": "videomae",
        "group": "pt50",
    },
    # --- e100 init-matched comparison (NeurIPS primary) ---
    "JEPA-IN21K-e100": {
        "checkpoint": "checkpoints/jepa_in21k_vitl_e95.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "e100",
    },
    "BYOL-L-e100": {
        "checkpoint": "checkpoints/byol_vitl_imagenet_v2_e100.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "e100",
    },
    "MAE-L-e99": {
        "checkpoint": "checkpoints/videomae_l_mimic_ep99.pth",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
        "type": "videomae",
        "group": "e100",
    },
    # --- SALT (frozen teacher) ---
    "SALT-S2v1-e79": {
        "checkpoint": "checkpoints/pretrain/mimic/salt_s2v1_e79.pt",
        "model_name": "vit_large",
        "checkpoint_key": "encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "salt",
    },
    "SALT-S2v3-e79": {
        "checkpoint": "checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/latest.pt",
        "model_name": "vit_large",
        "checkpoint_key": "encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
        "type": "vjepa",
        "group": "salt",
    },
    # --- System-level baselines ---
    "EchoPrime": {
        "checkpoint": "checkpoints/echo_prime_encoder.pt",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
        "type": "echoprime",
        "group": "baseline",
    },
    "PanEcho": {
        "checkpoint": "checkpoints/panecho.pt",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
        "type": "panecho",
        "group": "baseline",
    },
}

# Preset groups for convenience
MODEL_GROUPS = {
    "fully-trained": ["JEPA-G", "JEPA-L", "MAE-L"],
    "pt50": ["JEPA-L-pt50", "BYOL-L-pt50", "MAE-L-pt50"],
    "controlled": ["JEPA-L-pt50", "BYOL-L-pt50", "MAE-L-pt50"],  # alias
    "e100": ["JEPA-IN21K-e100", "BYOL-L-e100", "MAE-L-e99"],
    "all-5": ["JEPA-G", "JEPA-L", "MAE-L", "EchoPrime", "PanEcho"],
    "salt": ["SALT-S2v1-e79", "SALT-S2v3-e79"],
    "baselines": ["EchoPrime", "PanEcho"],
}


def get_models(selection=None):
    """
    Get model configs based on selection.

    Args:
        selection: None (all models), or list of model names or group names.
            Examples: ["JEPA-L-pt50", "BYOL-L-pt50"], ["pt50"], ["all-5"]

    Returns:
        dict of {name: config} for selected models
    """
    if selection is None:
        return ALL_MODELS

    selected = {}
    for s in selection:
        if s in MODEL_GROUPS:
            for name in MODEL_GROUPS[s]:
                selected[name] = ALL_MODELS[name]
        elif s in ALL_MODELS:
            selected[s] = ALL_MODELS[s]
        else:
            available = list(ALL_MODELS.keys()) + list(MODEL_GROUPS.keys())
            raise ValueError(f"Unknown model/group: {s}. Available: {available}")
    return selected


def add_model_args(parser):
    """Add --models argument to an argparse parser."""
    available = list(ALL_MODELS.keys()) + list(MODEL_GROUPS.keys())
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to run (default: all). Names or groups: {available}"
    )
