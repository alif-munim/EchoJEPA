"""
Pixel reconstruction decoder for SALT Stage 1 (V-Pixel).

Reconstructs raw pixel values for masked patches given visible encoder
embeddings. Uses the same Block class as the V-JEPA 2 encoder/predictor
to ensure RoPE compatibility.
"""

import torch
import torch.nn as nn

from app.vjepa_2_1.models.utils.modules import Block
from src.masks.utils import apply_masks
from src.utils.tensors import trunc_normal_


class PixelDecoder(nn.Module):
    """Lightweight ViT decoder that reconstructs masked pixel patches."""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        img_size=224,
        num_frames=16,
        use_rope=True,
        use_activation_checkpointing=False,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.use_rope = use_rope
        self.use_activation_checkpointing = use_activation_checkpointing
        self.decoder_embed_dim = decoder_embed_dim
        self.init_std = init_std

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size

        # Project encoder visible embeddings to decoder space
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token substituted at masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=init_std)

        # Decoder transformer blocks (same Block as encoder/predictor)
        dpr = [x.item() for x in torch.linspace(0, 0.0, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    patch_size=patch_size,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Project to pixel space: C * tubelet_size * patch_size^2 values per patch
        pixel_dim = in_chans * tubelet_size * patch_size * patch_size
        self.decoder_head = nn.Linear(decoder_embed_dim, pixel_dim, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_vis, masks_enc, masks_pred):
        """
        Reconstruct pixel values for masked patches.

        Args:
            x_vis: [B*len(masks_enc), N_vis, encoder_embed_dim]
                   Visible encoder output (already gathered by apply_masks in encoder).
            masks_enc: list of [B, N_vis] index tensors — which patches are visible.
            masks_pred: list of [B, N_pred] index tensors — which patches to reconstruct.

        Returns:
            pixel_pred: [B*len(masks_pred), N_pred, C*t*p*p] predicted pixel values.
        """
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        B = len(x_vis) // len(masks_enc)

        # 1. Project visible embeddings to decoder space
        x_vis = self.encoder_to_decoder(x_vis)
        _, N_vis, D = x_vis.shape

        # 2. Create mask tokens for predicted (masked) positions
        pred_tokens = self.mask_token.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_pred)
        pred_tokens = pred_tokens.repeat(len(masks_enc), 1, 1)

        # 3. Concatenate visible + mask tokens
        x_vis = x_vis.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x_vis, pred_tokens], dim=1)

        # 4. Build position indices for RoPE and sort by position
        masks_enc_cat = torch.cat(masks_enc, dim=0)
        masks_pred_cat = torch.cat(masks_pred, dim=0)
        # Replicate enc masks for each pred mask and vice versa
        masks_enc_rep = masks_enc_cat.repeat(len(masks_pred), 1)
        masks_pred_rep = masks_pred_cat.repeat(len(masks_enc), 1)
        masks = torch.cat([masks_enc_rep, masks_pred_rep], dim=1)

        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        # 5. Run through decoder blocks
        for blk in self.decoder_blocks:
            if self.use_activation_checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(blk, x, masks, use_reentrant=False)
            else:
                x, _ = blk(x, mask=masks)

        x = self.decoder_norm(x)

        # 6. Reverse argsort, extract only mask-token positions
        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
        x = x[:, N_vis:, :]  # Keep only predicted (masked) positions

        # 7. Project to pixel space
        x = self.decoder_head(x)
        return x
