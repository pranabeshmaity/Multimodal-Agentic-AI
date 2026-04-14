"""Novel cross-modal attention with modality-type embeddings and a gated
residual.

This module is the workhorse of the Hyper-Latent Fusion first stage. A
set of *query* tokens (e.g. vision tokens) attend to a *concatenated*
sequence of key/value tokens drawn from the other modalities. Each
participating token is tagged with a learnable modality-type embedding
so the attention can differentiate where information originated from.

A learned sigmoid gate conditioned on the query decides how much of the
attention output is mixed back into the residual stream — this lets the
network adaptively ignore irrelevant cross-modal signal rather than
forcing a full update.

Optionally, a rotary position embedding (RoPE) is applied to queries and
keys for translation-equivariant position handling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) caches for rotary position embedding.

    Args:
        seq_len: Length of the sequence.
        head_dim: Per-head dimension (must be even).
        device: Device to create tensors on.
        base: Frequency base.

    Returns:
        Tuple ``(cos, sin)`` each of shape ``(1, 1, seq_len, head_dim)``.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.einsum("t,f->tf", t, freqs)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape ``(B, H, T, D)``."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CrossModalAttention(nn.Module):
    """Query-from-one-modality, KV-from-others cross-modal attention.

    The layer expects a single query tensor and a *dictionary* of context
    tensors keyed by modality name. Each context is decorated with a
    learnable modality-type embedding, concatenated into a single
    key/value sequence, and attended to by the query. The result passes
    through a sigmoid gate and is added back to ``query`` as a residual.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        modalities: Ordered names of modalities whose type embeddings
            must be registered (e.g. ``("vision", "audio", "text")``).
        dropout: Attention dropout probability.
        use_rope: If True, apply rotary position embedding to Q and K.
        rope_max_len: Cache length for the rotary embedding tables.

    Shape:
        Query input:   ``(B, N_q, D)``
        Context input: ``{modality: (B, N_m, D)}``
        Output:        ``(B, N_q, D)``
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        modalities: Sequence[str] = ("vision", "audio", "text"),
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_max_len: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modalities: Tuple[str, ...] = tuple(modalities)
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_max_len = rope_max_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable modality-type embeddings, one vector per registered
        # modality. Added to tokens before projecting to K/V.
        self.modality_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, d_model)) for m in self.modalities}
        )
        # Also tag the query stream with its own modality token.
        self.query_modality_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, d_model)) for m in self.modalities}
        )
        for p in self.modality_embed.values():
            nn.init.trunc_normal_(p, std=0.02)
        for p in self.query_modality_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        # Adaptive gate: sigmoid(W_g @ q) scales the attention residual.
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_proj.bias)
        # Bias init negative so initial gate sits near 0.5 but slightly
        # below — lets the model "open" the gate during training.
        nn.init.normal_(self.gate_proj.weight, std=0.02)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        if use_rope:
            cos, sin = _build_rope_cache(
                rope_max_len, self.head_dim, device=torch.device("cpu")
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T, D)`` -> ``(B, H, T, Dh)``."""
        return rearrange(x, "b t (h d) -> b h t d", h=self.n_heads)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, H, T, Dh)`` -> ``(B, T, D)``."""
        return rearrange(x, "b h t d -> b t (h d)")

    def forward(
        self,
        query: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
        query_modality: str,
        context_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run cross-modal attention.

        Args:
            query: Query stream of shape ``(B, N_q, D)`` coming from the
                modality named ``query_modality``.
            contexts: Mapping modality-name -> tensor ``(B, N_m, D)``.
                The ``query_modality`` entry, if present, is ignored so
                the query does not attend to itself.
            query_modality: Name of the query modality; must be a key in
                the registered ``modalities``.
            context_masks: Optional per-modality boolean masks of shape
                ``(B, N_m)`` where ``True`` marks *valid* positions.

        Returns:
            Updated query tokens of shape ``(B, N_q, D)``.
        """
        if query_modality not in self.modalities:
            raise KeyError(f"unknown query modality: {query_modality}")

        b, n_q, _ = query.shape

        # Tag query with its modality.
        q_in = self.norm_q(query + self.query_modality_embed[query_modality])

        # Build concatenated KV from the *other* modalities.
        kv_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        for name in self.modalities:
            if name == query_modality:
                continue
            if name not in contexts:
                continue
            c = contexts[name]
            if c.shape[-1] != self.d_model:
                raise ValueError(
                    f"context '{name}' has dim {c.shape[-1]}, expected {self.d_model}"
                )
            c = c + self.modality_embed[name]
            kv_chunks.append(c)
            if context_masks is not None and name in context_masks:
                mask_chunks.append(context_masks[name])
            else:
                mask_chunks.append(
                    torch.ones(c.shape[:2], dtype=torch.bool, device=c.device)
                )

        if not kv_chunks:
            # Nothing to attend to — identity.
            return query

        kv = torch.cat(kv_chunks, dim=1)
        kv = self.norm_kv(kv)
        kv_mask = torch.cat(mask_chunks, dim=1)  # (B, N_kv)

        q = self._split_heads(self.q_proj(q_in))
        k = self._split_heads(self.k_proj(kv))
        v = self._split_heads(self.v_proj(kv))

        if self.use_rope:
            n_kv = k.shape[2]
            if max(n_q, n_kv) > self.rope_max_len:
                raise ValueError("sequence length exceeds rope_max_len")
            cos_q = self.rope_cos[:, :, :n_q].to(q.device)
            sin_q = self.rope_sin[:, :, :n_q].to(q.device)
            cos_k = self.rope_cos[:, :, :n_kv].to(k.device)
            sin_k = self.rope_sin[:, :, :n_kv].to(k.device)
            q = _apply_rope(q, cos_q, sin_q)
            k = _apply_rope(k, cos_k, sin_k)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply key mask: positions with False become -inf.
        key_mask = kv_mask[:, None, None, :]  # (B, 1, 1, N_kv)
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

        attn = attn_logits.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        # Adaptive gated residual.
        gate = torch.sigmoid(self.gate_proj(q_in))
        return query + gate * out


def _smoke_test() -> None:
    """Instantiate a ``CrossModalAttention`` and run a dummy forward."""
    torch.manual_seed(0)
    d_model, n_heads = 512, 8
    batch = 2
    layer = CrossModalAttention(
        d_model=d_model,
        n_heads=n_heads,
        modalities=("vision", "audio", "text"),
        dropout=0.1,
        use_rope=True,
        rope_max_len=1024,
    )
    vision = torch.randn(batch, 197, d_model)  # 14*14 patches + CLS
    audio = torch.randn(batch, 128, d_model)
    text = torch.randn(batch, 32, d_model)
    contexts = {"vision": vision, "audio": audio, "text": text}
    out = layer(query=vision, contexts=contexts, query_modality="vision")
    assert out.shape == vision.shape, f"unexpected shape {out.shape}"
    print(f"[ok] CrossModalAttention forward -> {tuple(out.shape)}")
    # Backward.
    out.mean().backward()
    print("[ok] backward pass completed")


if __name__ == "__main__":
    _smoke_test()
