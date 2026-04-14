"""Text encoder for Hyper-Latent Fusion.

A standard transformer encoder with token and learned positional
embeddings. Includes a cross-modal hook at layer 1 to mirror the vision
and audio encoders.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import TransformerBlock


class TextEncoder(nn.Module):
    """Transformer text encoder with learned positional embeddings.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Hidden dimension.
        n_heads: Attention heads.
        depth: Number of transformer layers.
        max_seq_len: Maximum supported sequence length.
        mlp_ratio: MLP expansion factor.
        dropout: Dropout probability.
        pad_token_id: Id reserved for padding tokens (zeros the gradient
            of the corresponding embedding row).

    Shape:
        Input:  ``(B, T)`` (long tensor of token ids)
        Output: ``(B, T, D)``
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        max_seq_len: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
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
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        cross_modal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of token id sequences.

        Args:
            tokens: Long tensor of shape ``(B, T)``.
            cross_modal_context: Optional context ``(B, M, D)`` injected
                at layer 1 via cross-attention.

        Returns:
            Contextualized token embeddings of shape ``(B, T, D)``.
        """
        b, t = tokens.shape
        if t > self.pos_embed.shape[1]:
            raise ValueError(
                f"text sequence length {t} exceeds max_seq_len "
                f"{self.pos_embed.shape[1]}"
            )
        x = self.token_embed(tokens) + self.pos_embed[:, :t]
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            ctx = cross_modal_context if i == 0 else None
            x = block(x, context=ctx)
        return self.norm(x)
