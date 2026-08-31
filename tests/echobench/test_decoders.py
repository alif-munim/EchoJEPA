"""Tests for echobench segmentation decoders and spatial feature extraction."""

import torch
import pytest

from echobench.tasks.decoders import LinearSegDecoder, SmallConvSegDecoder
from echobench.tasks.segmentation import extract_spatial_features


class TestLinearSegDecoder:
    def test_output_shape(self):
        decoder = LinearSegDecoder(embed_dim=1024, num_classes=4, target_size=224)
        x = torch.randn(2, 14, 14, 1024)  # [B, H', W', D]
        out = decoder(x)
        assert out.shape == (2, 4, 224, 224)

    def test_small_embed_dim(self):
        decoder = LinearSegDecoder(embed_dim=512, num_classes=4, target_size=224)
        x = torch.randn(1, 7, 7, 512)
        out = decoder(x)
        assert out.shape == (1, 4, 224, 224)


class TestSmallConvSegDecoder:
    def test_output_shape(self):
        decoder = SmallConvSegDecoder(embed_dim=1024, num_classes=4)
        x = torch.randn(2, 14, 14, 1024)
        out = decoder(x)
        assert out.shape == (2, 4, 224, 224)


class TestExtractSpatialFeatures:
    def test_vjepa_shape(self):
        tokens = torch.randn(2, 1568, 1024)  # 8 temporal x 14x14 spatial
        t_idx = torch.tensor([3, 5])
        spatial = extract_spatial_features(tokens, "vjepa", t_idx)
        assert spatial.shape == (2, 14, 14, 1024)

    def test_videomae_shape(self):
        tokens = torch.randn(2, 1568, 1024)
        t_idx = torch.tensor([0, 7])
        spatial = extract_spatial_features(tokens, "videomae", t_idx)
        assert spatial.shape == (2, 14, 14, 1024)

    def test_echoprime_shape(self):
        tokens = torch.randn(1, 392, 512)  # 8 temporal x 7x7 spatial
        t_idx = torch.tensor([2])
        spatial = extract_spatial_features(tokens, "echoprime", t_idx)
        assert spatial.shape == (1, 7, 7, 512)

    def test_panecho_shape(self):
        tokens = torch.randn(1, 784, 768)  # 16 temporal x 7x7 spatial
        t_idx = torch.tensor([3])  # tubelet index, will be mapped to frame index
        spatial = extract_spatial_features(tokens, "panecho", t_idx)
        assert spatial.shape == (1, 7, 7, 768)

    def test_batch_indexing(self):
        """Each sample in batch should get its own temporal index."""
        tokens = torch.randn(3, 1568, 1024)
        # Make temporal tokens distinguishable
        for t in range(8):
            tokens[:, t * 196 : (t + 1) * 196, :] = t * 1.0

        t_idx = torch.tensor([0, 4, 7])
        spatial = extract_spatial_features(tokens, "vjepa", t_idx)

        assert spatial[0].mean().item() == pytest.approx(0.0, abs=0.01)
        assert spatial[1].mean().item() == pytest.approx(4.0, abs=0.01)
        assert spatial[2].mean().item() == pytest.approx(7.0, abs=0.01)


class TestAdapterImports:
    """Smoke tests that adapter modules import without crashing."""

    def test_import_echojepa(self):
        from echobench.adapters.echojepa import EchoJEPAAdapter
        assert EchoJEPAAdapter is not None

    def test_import_videomae(self):
        from echobench.adapters.videomae import VideoMAEAdapter
        assert VideoMAEAdapter is not None

    def test_import_echoprime(self):
        from echobench.adapters.echoprime import EchoPrimeAdapter
        assert EchoPrimeAdapter is not None

    def test_import_panecho_module(self):
        """PanEcho module imports, but instantiation needs source repo."""
        from echobench.adapters.panecho import PanEchoAdapter
        assert PanEchoAdapter is not None

    def test_import_echofm_module(self):
        """EchoFM module imports, but instantiation needs source repo."""
        from echobench.adapters.echofm import EchoFMAdapter
        assert EchoFMAdapter is not None

    def test_get_adapter_invalid(self):
        from echobench.adapters import get_adapter
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent")
