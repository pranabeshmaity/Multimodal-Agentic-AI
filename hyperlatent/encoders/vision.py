"""Vision encoder for the Hyper-Latent Fusion architecture.

A compact ViT-style encoder that performs patch embedding followed by a
stack of pre-norm transformer blocks. The first block optionally accepts
a ``cross_modal_context`` tensor so that visual tokens can attend to
other modalities early — this is the hook used by the
``CrossModalAttention`` fusion novelty.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """Image -> sequence of patch tokens via a single strided Conv2d.

    Args:
        image_size: Height/width of the input image (assumed square).
        patch_size: Side length of a square patch.
        in_channels: Number of input image channels.
        d_model: Output embedding dimension.

    Shape:
        Input:  ``(B, C, H, W)``
        Output: ``(B, N, D)`` with ``N = (H / P) * (W / P)``.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project an image tensor into a patch-token sequence.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Patch tokens of shape ``(B, N, D)``.
        """
        x = self.proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with optional cross-attention.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        mlp_ratio: Multiplier for the MLP hidden dimension.
        dropout: Dropout probability.
        use_cross_attn: If True, the block owns a cross-attention module
            that is applied when a context is supplied to ``forward``.

    Shape:
        Input:  ``(B, N, D)``
        Output: ``(B, N, D)``
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_ctx = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.cross_gate = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a single transformer step.

        Args:
            x: Token sequence of shape ``(B, N, D)``.
            context: Optional cross-modal context ``(B, M, D)``.

        Returns:
            Updated tokens of shape ``(B, N, D)``.
        """
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn_out
        if self.use_cross_attn and context is not None:
            h_q = self.norm_ctx(x)
            cx, _ = self.cross_attn(h_q, context, context, need_weights=False)
            gate = torch.sigmoid(self.cross_gate(h_q))
            x = x + gate * cx
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-style vision encoder with a first-layer cross-modal hook.

    Args:
        image_size: Spatial size of input images.
        patch_size: Patch side length.
        in_channels: Image channel count.
        d_model: Hidden dimension.
        n_heads: Attention heads per block.
        depth: Number of transformer blocks.
        mlp_ratio: MLP expansion factor.
        dropout: Dropout probability.

    Shape:
        Input:  ``(B, C, H, W)``
        Output: ``(B, 1 + N, D)`` (CLS token prepended).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_cross_attn=(i == 0),
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal init for CLS and positional embeddings."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        cross_modal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: Input images of shape ``(B, C, H, W)``.
            cross_modal_context: Optional tensor of shape ``(B, M, D)``
                injected at layer 1 via cross-attention.

        Returns:
            Encoded tokens of shape ``(B, 1 + N, D)``.
        """
        b = images.shape[0]
        x = self.patch_embed(images)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            ctx = cross_modal_context if i == 0 else None
            x = block(x, context=ctx)
        return self.norm(x)
