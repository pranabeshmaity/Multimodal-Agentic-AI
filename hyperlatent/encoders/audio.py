"""Audio encoder for Hyper-Latent Fusion.

A Wave2Vec-style feature extractor: a stack of strided 1D convolutions
transforms the raw waveform into a sequence of frame embeddings that is
then refined by a transformer stack. Layer 1 exposes the same
cross-modal context hook as the vision encoder.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vision import TransformerBlock


class ConvFeatureExtractor(nn.Module):
    """Strided 1D conv stack mapping waveform -> frame embeddings.

    Args:
        in_channels: Number of input audio channels (1 for mono).
        channels: Output channel counts per conv layer.
        kernel_sizes: Kernel sizes for each conv layer.
        strides: Strides for each conv layer.
        d_model: Final projection dimension.

    Shape:
        Input:  ``(B, C_in, T)``
        Output: ``(B, T', D)`` where ``T'`` depends on strides.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3),
        strides: Tuple[int, ...] = (5, 2, 2, 2),
        d_model: int = 512,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels/kernels/strides must match in length")
        layers: List[nn.Module] = []
        c_prev = in_channels
        for c, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(c_prev, c, kernel_size=k, stride=s, bias=False))
            layers.append(nn.GroupNorm(1, c))
            layers.append(nn.GELU())
            c_prev = c
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(c_prev, d_model)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract frame features from a raw waveform tensor.

        Args:
            wav: Waveform of shape ``(B, C_in, T)``.

        Returns:
            Frame features of shape ``(B, T', D)``.
        """
        x = self.convs(wav)
        x = rearrange(x, "b c t -> b t c")
        return self.proj(x)


class AudioEncoder(nn.Module):
    """Wave2Vec-style encoder + transformer refinement.

    Args:
        in_channels: Audio channels.
        d_model: Hidden dimension.
        n_heads: Attention heads.
        depth: Transformer depth.
        max_frames: Maximum number of frames supported by positional
            embeddings.
        mlp_ratio: MLP expansion factor.
        dropout: Dropout probability.

    Shape:
        Input:  ``(B, C, T)``
        Output: ``(B, T', D)``
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        max_frames: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(
            in_channels=in_channels, d_model=d_model
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        wav: torch.Tensor,
        cross_modal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of waveforms.

        Args:
            wav: Input waveform tensor of shape ``(B, C, T)``.
            cross_modal_context: Optional context ``(B, M, D)`` injected
                at layer 1 via cross-attention.

        Returns:
            Frame embeddings of shape ``(B, T', D)``.
        """
        x = self.feature_extractor(wav)
        t_prime = x.shape[1]
        if t_prime > self.pos_embed.shape[1]:
            raise ValueError(
                f"audio sequence length {t_prime} exceeds max_frames "
                f"{self.pos_embed.shape[1]}"
            )
        x = x + self.pos_embed[:, :t_prime]
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            ctx = cross_modal_context if i == 0 else None
            x = block(x, context=ctx)
        return self.norm(x)
