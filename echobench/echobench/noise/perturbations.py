"""
Echo-specific acoustic perturbations for robustness benchmarking.

Four perturbation types modelling real clinical degradation in echocardiography:
  - Depth attenuation: signal falls off with depth (obese patients, poor windows)
  - Gaussian shadow: localized signal dropout (ribs, calcification)
  - Haze artifact: reverberation haze (poor acoustic windows)
  - Speckle reduction: variable despeckling (scanner post-processing)

All perturbations are temporally consistent (same spatial map across frames),
deterministic from seed, and respect the ultrasound scan mask.

References:
  - Depth attenuation, haze: Ostvik et al. 2021 (via USAugment)
  - Gaussian shadow, speckle reduction: Smistad et al. 2018 (via USAugment)

Usage:
    from echobench.noise import apply_perturbation, NOISE_TYPES

    clip = ...  # [C, T, H, W], float32 [0, 1]
    perturbed = apply_perturbation(clip, "depth_attenuation", "high", seed=42)
"""

import numpy as np
import torch

__all__ = [
    "apply_perturbation",
    "create_scan_mask",
    "NOISE_TYPES",
    "PERTURBATIONS",
    "SEVERITY_LEVELS",
]


# ---------------------------------------------------------------------------
# Severity levels (fixed parameters per perturbation type)
# ---------------------------------------------------------------------------

PERTURBATIONS = {
    "depth_attenuation": {
        "low": {"attenuation_rate": 0.75, "max_attenuation": 0.0},
        "medium": {"attenuation_rate": 1.50, "max_attenuation": 0.0},
        "high": {"attenuation_rate": 2.15, "max_attenuation": 0.0},
    },
    "gaussian_shadow": {
        "low": {"strength": 0.4, "sigma_x": 0.15, "sigma_y": 0.15},
        "medium": {"strength": 0.6, "sigma_x": 0.20, "sigma_y": 0.20},
        "high": {"strength": 0.8, "sigma_x": 0.25, "sigma_y": 0.25},
    },
    "haze_artifact": {
        "low": {"haze_intensity": 0.15, "radius": 0.3, "sigma": 0.25},
        "medium": {"haze_intensity": 0.30, "radius": 0.3, "sigma": 0.30},
        "high": {"haze_intensity": 0.50, "radius": 0.3, "sigma": 0.35},
    },
    "speckle_reduction": {
        "low": {"sigma_spatial": 0.5, "sigma_color": 0.2, "window_size": 5},
        "medium": {"sigma_spatial": 1.0, "sigma_color": 0.5, "window_size": 5},
        "high": {"sigma_spatial": 2.0, "sigma_color": 1.0, "window_size": 5},
    },
}

NOISE_TYPES = list(PERTURBATIONS.keys())
SEVERITY_LEVELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Scan mask detection
# ---------------------------------------------------------------------------


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
        gray = frame.mean(dim=0)
    else:
        gray = frame
    return (gray * 255 > threshold).float()


# ---------------------------------------------------------------------------
# Depth Attenuation (Ostvik et al. 2021)
# ---------------------------------------------------------------------------


def _depth_attenuation_map(height, width, attenuation_rate, max_attenuation, scan_mask):
    """Exponential signal decay with distance from transducer (top-center)."""
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    distances = torch.sqrt((xv - 0.5) ** 2 + yv**2)

    atten_map = (1 - max_attenuation) * torch.exp(-attenuation_rate * distances) + max_attenuation
    atten_map = torch.where(scan_mask > 0.5, atten_map, torch.ones_like(atten_map))
    return atten_map


def apply_depth_attenuation(clip, attenuation_rate, max_attenuation, scan_mask):
    """Apply depth attenuation to [C, T, H, W] clip."""
    _, _, H, W = clip.shape
    atten_map = _depth_attenuation_map(H, W, attenuation_rate, max_attenuation, scan_mask)
    return clip * atten_map.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Gaussian Shadow (Smistad et al. 2018)
# ---------------------------------------------------------------------------


def _gaussian_shadow_map(height, width, strength, sigma_x, sigma_y, scan_mask, seed=0):
    """Localized dark region from beam-blocking structures. Position deterministic from seed."""
    rng = np.random.RandomState(seed)
    mu_x = rng.uniform(0.2, 0.8) * width
    mu_y = rng.uniform(0.2, 0.8) * height

    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yv, xv = torch.meshgrid(y, x, indexing="ij")

    sig_x_px = sigma_x * width
    sig_y_px = sigma_y * height

    shadow = 1.0 - strength * torch.exp(
        -((xv - mu_x) ** 2 / (2 * sig_x_px**2) + (yv - mu_y) ** 2 / (2 * sig_y_px**2))
    )
    shadow = torch.where(scan_mask > 0.5, shadow, torch.ones_like(shadow))
    return shadow


def apply_gaussian_shadow(clip, strength, sigma_x, sigma_y, scan_mask, seed=0):
    """Apply static Gaussian shadow to [C, T, H, W] clip."""
    _, _, H, W = clip.shape
    shadow = _gaussian_shadow_map(H, W, strength, sigma_x, sigma_y, scan_mask, seed)
    return clip * shadow.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Haze Artifact (Ostvik et al. 2021)
# ---------------------------------------------------------------------------


def _haze_maps(height, width, haze_intensity, radius, sigma, scan_mask, seed=0):
    """Combined additive brightness + contrast-reduction reverberation haze."""
    rng = np.random.RandomState(seed)

    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt((xv - 0.5) ** 2 + yv**2)

    envelope = torch.exp(-((r - radius) ** 2) / (2 * sigma**2))
    noise = torch.from_numpy(rng.uniform(0, 1, (height, width)).astype(np.float32))

    additive = haze_intensity * 0.35 * noise * envelope * scan_mask
    blend = haze_intensity * 0.5 * envelope * scan_mask
    return additive, blend


def apply_haze_artifact(clip, haze_intensity, radius, sigma, scan_mask, seed=0):
    """Apply haze artifact to [C, T, H, W] clip."""
    _, _, H, W = clip.shape
    additive, blend = _haze_maps(H, W, haze_intensity, radius, sigma, scan_mask, seed)

    add_map = additive.unsqueeze(0).unsqueeze(0)
    blend_map = blend.unsqueeze(0).unsqueeze(0)

    gray_value = 0.25
    result = clip * (1.0 - blend_map) + gray_value * blend_map
    result = result + add_map
    return torch.clamp(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Speckle Reduction (Smistad et al. 2018)
# ---------------------------------------------------------------------------


def apply_speckle_reduction(clip, sigma_spatial, sigma_color, window_size=5, scan_mask=None):
    """
    Apply bilateral-filter speckle reduction to each frame independently.

    Unlike the other 3 perturbations (which broadcast a single [H,W] map),
    bilateral filtering is a per-pixel nonlinear operation applied per-frame.
    This is deterministic (no random component).

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        sigma_spatial: spatial smoothing extent (higher = more blur)
        sigma_color: intensity smoothing range (higher = smoother across edges)
        window_size: bilateral filter window size
        scan_mask: [H, W] float32, optional
    Returns:
        [C, T, H, W] filtered clip
    """
    from skimage.restoration import denoise_bilateral

    C, T, H, W = clip.shape
    result = clip.clone()

    for t in range(T):
        frame = clip[:, t, :, :].permute(1, 2, 0).numpy()  # [H, W, C]
        filtered = denoise_bilateral(
            frame,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            win_size=window_size,
            channel_axis=-1,
        ).astype(np.float32)
        filtered_t = torch.from_numpy(filtered).permute(2, 0, 1)  # [C, H, W]

        if scan_mask is not None:
            mask = scan_mask.unsqueeze(0)  # [1, H, W]
            result[:, t, :, :] = filtered_t * mask + clip[:, t, :, :] * (1.0 - mask)
        else:
            result[:, t, :, :] = filtered_t

    return result


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

_APPLY_FNS = {
    "depth_attenuation": lambda clip, params, mask, seed: apply_depth_attenuation(
        clip, params["attenuation_rate"], params["max_attenuation"], mask
    ),
    "gaussian_shadow": lambda clip, params, mask, seed: apply_gaussian_shadow(
        clip, params["strength"], params["sigma_x"], params["sigma_y"], mask, seed
    ),
    "haze_artifact": lambda clip, params, mask, seed: apply_haze_artifact(
        clip, params["haze_intensity"], params["radius"], params["sigma"], mask, seed
    ),
    "speckle_reduction": lambda clip, params, mask, seed: apply_speckle_reduction(
        clip, params["sigma_spatial"], params["sigma_color"], params.get("window_size", 5), mask
    ),
}


def apply_perturbation(clip, noise_type, severity, scan_mask=None, seed=0):
    """
    Apply an echo-specific perturbation to a video clip.

    Args:
        clip: [C, T, H, W] float32 [0, 1]
        noise_type: one of NOISE_TYPES
        severity: one of "low", "medium", "high"
        scan_mask: [H, W] float32 mask (auto-detected from first frame if None)
        seed: RNG seed for stochastic components (shadow position, haze texture)

    Returns:
        [C, T, H, W] perturbed clip
    """
    if noise_type not in PERTURBATIONS:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")
    if severity not in PERTURBATIONS[noise_type]:
        raise ValueError(f"Unknown severity: {severity}. Choose from {SEVERITY_LEVELS}")

    if scan_mask is None:
        scan_mask = create_scan_mask(clip[:, 0, :, :])

    params = PERTURBATIONS[noise_type][severity]
    return _APPLY_FNS[noise_type](clip, params, scan_mask, seed)
