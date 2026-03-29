"""
Echo-specific perturbations for rebuttal experiments.

Wraps USAugment perturbation types (depth attenuation, acoustic shadow, haze)
as pure PyTorch functions operating on video tensors [C, T, H, W].

All perturbations use fixed severity levels (not random ranges) for controlled
experiments. Scan masks are auto-detected from the first frame.

Perturbation types and their clinical motivation:
  - Depth attenuation: signal falls off with depth (obese patients, poor windows)
  - Acoustic shadow: localized signal dropout (ribs, calcification)
  - Haze artifact: reverberation haze (poor acoustic windows)

References:
  - Depth attenuation, haze: Ostvik et al. 2021 (via USAugment)
  - Gaussian shadow: Smistad et al. 2018 (via USAugment)

Usage:
    from scripts.rebuttal.echo_perturbations import apply_perturbation, PERTURBATIONS

    clip = load_clip(...)  # [C, T, H, W], float32 [0, 1]
    perturbed = apply_perturbation(clip, "depth_attenuation", "severe", seed=42)
"""

import torch
import numpy as np


# --- Severity levels (fixed parameters per perturbation type) ---
# Each level maps to the parameters of the underlying USAugment transform.

PERTURBATIONS = {
    # Rate controls exponential decay with depth. USAugment default range: (0.0, 3.0).
    # At rate=2.5 the deepest point retains ~6% intensity.
    "depth_attenuation": {
        "mild": {"attenuation_rate": 1.0, "max_attenuation": 0.0},
        "moderate": {"attenuation_rate": 1.8, "max_attenuation": 0.0},
        "severe": {"attenuation_rate": 2.5, "max_attenuation": 0.0},
    },
    # Strength = darkness (0-1). Sigma = spread as fraction of image size.
    # USAugment defaults: strength=(0.25, 0.8), sigma=(0.01, 0.2).
    # Both strength and sigma increase with severity (darker AND broader shadow).
    "gaussian_shadow": {
        "mild": {"strength": 0.5, "sigma_x": 0.14, "sigma_y": 0.14},
        "moderate": {"strength": 0.75, "sigma_x": 0.20, "sigma_y": 0.20},
        "severe": {"strength": 0.95, "sigma_x": 0.28, "sigma_y": 0.28},
    },
    # Haze intensity controls contrast reduction (blend towards gray).
    # Radius = distance from transducer where haze peaks. Sigma = spatial spread.
    # Based on USAugment (Ostvik 2021) but using contrast-reduction blending
    # instead of additive brightness — real reverberation washes out detail
    # rather than just brightening.
    "haze_artifact": {
        "mild": {"haze_intensity": 0.15, "radius": 0.3, "sigma": 0.25},
        "moderate": {"haze_intensity": 0.30, "radius": 0.3, "sigma": 0.30},
        "severe": {"haze_intensity": 0.50, "radius": 0.3, "sigma": 0.35},
    },
}

SEVERITY_LEVELS = ["mild", "moderate", "severe"]

# Transducer position presets (normalized x, y coords)
TRANSDUCER_PRESETS = {
    "standard": (0.5, 0.0),   # top-center (EchoNet, UHN, most echo videos)
    "camus": (0.0, 0.5),      # left-center (CAMUS NIfTI storage orientation)
}


def create_scan_mask(frame, threshold=10):
    """
    Create a binary scan mask from a single frame by thresholding.

    Args:
        frame: [C, H, W] or [H, W] tensor, float32 [0, 1]
        threshold: intensity threshold (0-255 scale, applied after *255)

    Returns:
        [H, W] float32 tensor, 1.0 = scan region, 0.0 = background
    """
    if frame.dim() == 3:
        gray = frame.mean(dim=0)  # [H, W]
    else:
        gray = frame
    mask = (gray * 255 > threshold).float()
    return mask


# --- Depth Attenuation (Ostvik et al. 2021) ---

def _depth_attenuation_map(height, width, attenuation_rate, max_attenuation, scan_mask,
                           transducer_pos=(0.5, 0.0)):
    """
    Generate depth attenuation map. Signal decays exponentially with distance
    from the transducer position.

    Args:
        transducer_pos: (x, y) normalized coords of transducer. Default (0.5, 0.0) = top-center.

    Returns: [H, W] float32 tensor, values in [max_attenuation, 1.0]
    """
    tx, ty = transducer_pos
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    distances = torch.sqrt((xv - tx) ** 2 + (yv - ty) ** 2)

    atten_map = (1 - max_attenuation) * torch.exp(-attenuation_rate * distances) + max_attenuation
    atten_map = torch.where(scan_mask > 0.5, atten_map, torch.ones_like(atten_map))
    return atten_map


def apply_depth_attenuation(clip, attenuation_rate, max_attenuation, scan_mask,
                            transducer_pos=(0.5, 0.0)):
    """
    Apply depth attenuation to a video clip.

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        scan_mask: [H, W] float32
        transducer_pos: (x, y) normalized coords of transducer
    Returns:
        [C, T, H, W] perturbed clip
    """
    _, _, H, W = clip.shape
    atten_map = _depth_attenuation_map(H, W, attenuation_rate, max_attenuation, scan_mask,
                                       transducer_pos)
    return clip * atten_map.unsqueeze(0).unsqueeze(0)


# --- Gaussian Shadow (Smistad et al. 2018) ---

def _gaussian_shadow_map(height, width, strength, sigma_x, sigma_y, scan_mask, seed=0):
    """
    Generate a Gaussian shadow map. Simulates acoustic shadow from
    ribs or calcification blocking the ultrasound beam.

    Shadow position is deterministic given the seed.

    Returns: [H, W] float32 tensor, values in [1-strength, 1.0]
    """
    rng = np.random.RandomState(seed)
    # Shadow center biased towards center region
    mu_x = rng.uniform(0.2, 0.8) * width
    mu_y = rng.uniform(0.2, 0.8) * height

    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yv, xv = torch.meshgrid(y, x, indexing="ij")

    sig_x_px = sigma_x * width
    sig_y_px = sigma_y * height

    shadow = 1.0 - strength * torch.exp(
        -((xv - mu_x) ** 2 / (2 * sig_x_px ** 2) + (yv - mu_y) ** 2 / (2 * sig_y_px ** 2))
    )
    shadow = torch.where(scan_mask > 0.5, shadow, torch.ones_like(shadow))
    return shadow


def apply_gaussian_shadow(clip, strength, sigma_x, sigma_y, scan_mask, seed=0):
    """
    Apply a static Gaussian shadow to a video clip.
    Same shadow position across all frames (consistent per-video).

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        scan_mask: [H, W] float32
        seed: RNG seed for shadow position
    Returns:
        [C, T, H, W] perturbed clip
    """
    _, _, H, W = clip.shape
    shadow = _gaussian_shadow_map(H, W, strength, sigma_x, sigma_y, scan_mask, seed)
    return clip * shadow.unsqueeze(0).unsqueeze(0)


# --- Haze Artifact (Ostvik et al. 2021) ---

def _haze_maps(height, width, haze_intensity, radius, sigma, scan_mask, seed=0,
               transducer_pos=(0.5, 0.0)):
    """
    Generate haze maps for combined additive + contrast-reduction blending.
    Simulates reverberation artifacts: brightens dark regions (additive)
    AND washes out detail (contrast reduction), with textured spatial variation.

    Args:
        transducer_pos: (x, y) normalized coords of transducer. Haze is strongest
            at `radius` distance from this point.

    Returns: (additive_map, blend_map) — both [H, W] float32 tensors
    """
    rng = np.random.RandomState(seed)
    tx, ty = transducer_pos

    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt((xv - tx) ** 2 + (yv - ty) ** 2)

    # Spatial envelope — strongest near transducer, falls off with depth
    envelope = torch.exp(-((r - radius) ** 2) / (2 * sigma ** 2))

    # Noise texture (USAugment-style) for natural spatial variation
    noise = torch.from_numpy(rng.uniform(0, 1, (height, width)).astype(np.float32))

    # Additive component: textured brightness (noise × envelope)
    # Brightens dark areas, gives the haze its textured quality
    additive = haze_intensity * 0.35 * noise * envelope * scan_mask

    # Contrast-reduction component: smooth envelope only (no noise)
    # Washes out detail by blending towards gray
    blend = haze_intensity * 0.5 * envelope * scan_mask

    return additive, blend


def apply_haze_artifact(clip, haze_intensity, radius, sigma, scan_mask, seed=0,
                        transducer_pos=(0.5, 0.0)):
    """
    Apply haze artifact to a video clip.
    Combines additive brightness (textured, from USAugment) with contrast
    reduction (blend towards gray). Real reverberation does both: it brightens
    dark regions AND washes out fine detail.

    Same haze pattern across all frames (consistent per-video).

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        scan_mask: [H, W] float32
        seed: RNG seed for haze texture
        transducer_pos: (x, y) normalized coords of transducer
    Returns:
        [C, T, H, W] perturbed clip
    """
    _, _, H, W = clip.shape
    additive, blend = _haze_maps(H, W, haze_intensity, radius, sigma, scan_mask, seed,
                                 transducer_pos)

    # [H, W] -> [1, 1, H, W] for broadcasting
    add_map = additive.unsqueeze(0).unsqueeze(0)
    blend_map = blend.unsqueeze(0).unsqueeze(0)

    # Contrast reduction (blend towards dark gray)
    gray_value = 0.25
    result = clip * (1.0 - blend_map) + gray_value * blend_map
    # Additive textured brightness on top
    result = result + add_map
    return torch.clamp(result, 0.0, 1.0)


# --- Unified interface ---

def _apply_depth_attenuation(clip, params, mask, seed, transducer_pos):
    return apply_depth_attenuation(
        clip, params["attenuation_rate"], params["max_attenuation"], mask, transducer_pos
    )


def _apply_gaussian_shadow(clip, params, mask, seed, transducer_pos):
    return apply_gaussian_shadow(
        clip, params["strength"], params["sigma_x"], params["sigma_y"], mask, seed
    )


def _apply_haze_artifact(clip, params, mask, seed, transducer_pos):
    return apply_haze_artifact(
        clip, params["haze_intensity"], params["radius"], params["sigma"], mask, seed,
        transducer_pos
    )


_APPLY_FNS = {
    "depth_attenuation": _apply_depth_attenuation,
    "gaussian_shadow": _apply_gaussian_shadow,
    "haze_artifact": _apply_haze_artifact,
}


def apply_perturbation(clip, perturbation_type, severity, scan_mask=None, seed=0,
                       transducer_pos=(0.5, 0.0)):
    """
    Apply an echo-specific perturbation to a video clip.

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        perturbation_type: one of "depth_attenuation", "gaussian_shadow", "haze_artifact"
        severity: one of "mild", "moderate", "severe"
        scan_mask: [H, W] float32 mask (auto-detected from first frame if None)
        seed: RNG seed for stochastic components (shadow position, haze texture)
        transducer_pos: (x, y) normalized position of the transducer.
            Use TRANSDUCER_PRESETS["standard"] = (0.5, 0.0) for standard echo orientation.
            Use TRANSDUCER_PRESETS["camus"] = (0.0, 0.5) for CAMUS NIfTI orientation.

    Returns:
        [C, T, H, W] perturbed clip
    """
    if perturbation_type not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {perturbation_type}. Choose from {list(PERTURBATIONS)}")
    if severity not in PERTURBATIONS[perturbation_type]:
        raise ValueError(f"Unknown severity: {severity}. Choose from {SEVERITY_LEVELS}")

    if scan_mask is None:
        scan_mask = create_scan_mask(clip[:, 0, :, :])  # first frame

    params = PERTURBATIONS[perturbation_type][severity]
    return _APPLY_FNS[perturbation_type](clip, params, scan_mask, seed, transducer_pos)
