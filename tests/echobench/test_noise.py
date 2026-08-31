"""Tests for echobench.noise perturbation module."""

import torch
import pytest

from echobench.noise import (
    NOISE_TYPES,
    PERTURBATIONS,
    SEVERITY_LEVELS,
    apply_perturbation,
    create_scan_mask,
)
from echobench.noise.perturbations import (
    apply_depth_attenuation,
    apply_gaussian_shadow,
    apply_haze_artifact,
    apply_speckle_reduction,
)


@pytest.fixture
def sample_clip():
    """A synthetic [C, T, H, W] clip with a bright center region."""
    torch.manual_seed(42)
    clip = torch.rand(3, 4, 64, 64)
    return clip


@pytest.fixture
def scan_mask():
    """A circular scan mask."""
    H, W = 64, 64
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    mask = ((x**2 + y**2) < 0.8).float()
    return mask


class TestCreateScanMask:
    def test_output_shape(self, sample_clip):
        mask = create_scan_mask(sample_clip[:, 0, :, :])
        assert mask.shape == (64, 64)

    def test_binary_values(self, sample_clip):
        mask = create_scan_mask(sample_clip[:, 0, :, :])
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_2d_input(self):
        frame = torch.ones(64, 64) * 0.5
        mask = create_scan_mask(frame)
        assert mask.sum() > 0  # bright frame should have nonzero mask


class TestNoiseTypes:
    def test_all_four_types_exist(self):
        assert len(NOISE_TYPES) == 4
        assert "depth_attenuation" in NOISE_TYPES
        assert "gaussian_shadow" in NOISE_TYPES
        assert "haze_artifact" in NOISE_TYPES
        assert "speckle_reduction" in NOISE_TYPES

    def test_all_three_severities(self):
        assert SEVERITY_LEVELS == ["low", "medium", "high"]

    def test_perturbations_dict_complete(self):
        for nt in NOISE_TYPES:
            assert nt in PERTURBATIONS
            for sev in SEVERITY_LEVELS:
                assert sev in PERTURBATIONS[nt]


class TestApplyPerturbation:
    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    @pytest.mark.parametrize("severity", SEVERITY_LEVELS)
    def test_output_shape(self, sample_clip, scan_mask, noise_type, severity):
        result = apply_perturbation(sample_clip, noise_type, severity, scan_mask=scan_mask, seed=0)
        assert result.shape == sample_clip.shape

    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    def test_values_in_range(self, sample_clip, scan_mask, noise_type):
        result = apply_perturbation(sample_clip, noise_type, "high", scan_mask=scan_mask, seed=0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    def test_deterministic(self, sample_clip, scan_mask, noise_type):
        r1 = apply_perturbation(sample_clip, noise_type, "medium", scan_mask=scan_mask, seed=42)
        r2 = apply_perturbation(sample_clip, noise_type, "medium", scan_mask=scan_mask, seed=42)
        assert torch.allclose(r1, r2)

    @pytest.mark.parametrize("noise_type", ["gaussian_shadow", "haze_artifact"])
    def test_different_seeds_differ(self, sample_clip, scan_mask, noise_type):
        r1 = apply_perturbation(sample_clip, noise_type, "medium", scan_mask=scan_mask, seed=0)
        r2 = apply_perturbation(sample_clip, noise_type, "medium", scan_mask=scan_mask, seed=999)
        assert not torch.allclose(r1, r2)

    def test_invalid_noise_type(self, sample_clip):
        with pytest.raises(ValueError, match="Unknown noise type"):
            apply_perturbation(sample_clip, "invalid_noise", "low")

    def test_invalid_severity(self, sample_clip):
        with pytest.raises(ValueError, match="Unknown severity"):
            apply_perturbation(sample_clip, "depth_attenuation", "extreme")

    def test_auto_mask(self, sample_clip):
        """Should auto-detect scan mask if not provided."""
        result = apply_perturbation(sample_clip, "depth_attenuation", "low")
        assert result.shape == sample_clip.shape


class TestDepthAttenuation:
    def test_attenuates_deeper_more(self):
        # Use full mask so all pixels are in scan region
        full_mask = torch.ones(64, 64)
        clip = torch.ones(3, 2, 64, 64)
        result = apply_depth_attenuation(clip, attenuation_rate=2.0, max_attenuation=0.0, scan_mask=full_mask)
        # Top rows (near transducer) should be brighter than bottom rows (deeper)
        top_mean = result[:, :, 5, 32].mean()
        bot_mean = result[:, :, 60, 32].mean()
        assert top_mean > bot_mean


class TestGaussianShadow:
    def test_darkens_image(self, sample_clip, scan_mask):
        result = apply_gaussian_shadow(sample_clip, strength=0.8, sigma_x=0.2, sigma_y=0.2, scan_mask=scan_mask)
        # Shadow should darken at least some pixels
        assert result.mean() < sample_clip.mean()


class TestHazeArtifact:
    def test_modifies_image(self, sample_clip, scan_mask):
        result = apply_haze_artifact(sample_clip, haze_intensity=0.3, radius=0.3, sigma=0.3, scan_mask=scan_mask)
        assert not torch.allclose(result, sample_clip)


class TestSpeckleReduction:
    def test_smoothing_effect(self):
        """Bilateral filter should reduce variance (smooth the image)."""
        torch.manual_seed(0)
        noisy = torch.rand(3, 2, 32, 32)  # small for speed
        mask = torch.ones(32, 32)
        result = apply_speckle_reduction(noisy, sigma_spatial=2.0, sigma_color=1.0, scan_mask=mask)
        # Filtered should have less variance than noisy
        assert result.var() < noisy.var()

    def test_respects_scan_mask(self):
        """Pixels outside scan mask should be unchanged."""
        clip = torch.rand(3, 2, 32, 32)
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0  # only center is scan region
        result = apply_speckle_reduction(clip, sigma_spatial=1.0, sigma_color=0.5, scan_mask=mask)
        # Outside mask should be identical
        assert torch.allclose(result[:, :, 0, 0], clip[:, :, 0, 0])
        assert torch.allclose(result[:, :, 30, 30], clip[:, :, 30, 30])
