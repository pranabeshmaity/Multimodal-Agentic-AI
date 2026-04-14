"""Cross-modal mixture-of-experts router.

Four directional experts specialise in a single cross-modal interaction:

* ``v2t`` — vision  -> text
* ``t2a`` — text    -> audio
* ``a2v`` — audio   -> vision
* ``tri`` — tri-modal fusion

A linear gating network predicts per-example logits over experts; the
top-``k`` are selected, re-normalised, and their weighted outputs are
summed. We also return the Switch-Transformer style load-balancing
auxiliary loss so expert utilisation stays roughly uniform.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Two-layer MLP expert with GELU.

    Args:
        d_model: Hidden / output dimension.
        hidden_dim: MLP inner dimension.
        dropout: Dropout probability.

    Shape:
        Input/Output: ``(B, d_model)``.
    """

    def __init__(
        self, d_model: int = 512, hidden_dim: int = 1024, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert MLP.

        Args:
            x: Tensor of shape ``(B, d_model)``.

        Returns:
            Tensor of shape ``(B, d_model)``.
        """
        return self.net(x)


class CrossModalMoERouter(nn.Module):
    """Top-k MoE router over four cross-modal experts.

    Args:
        d_model: Feature dimension for router input & expert I/O.
        hidden_dim: Expert MLP inner dimension.
        top_k: Number of experts activated per example.
        dropout: Expert dropout.
        expert_names: Names of the four experts (ordered).
        noise_std: Gaussian noise std added to gate logits during
            training to break ties (Noisy top-k gating).

    Shape:
        Input: ``(B, d_model)``
        Output: ``(B, d_model)``
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 1024,
        top_k: int = 2,
        dropout: float = 0.0,
        expert_names: Sequence[str] = ("v2t", "t2a", "a2v", "tri"),
        noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        if len(expert_names) != 4:
            raise ValueError("exactly 4 expert names required")
        if not (1 <= top_k <= 4):
            raise ValueError("top_k must be in [1, 4]")
        self.d_model = d_model
        self.top_k = top_k
        self.expert_names: Tuple[str, ...] = tuple(expert_names)
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, len(self.expert_names))
        self.experts = nn.ModuleDict(
            {
                name: ExpertMLP(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
                for name in self.expert_names
            }
        )

    def _load_balancing_loss(
        self, gate_probs: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Switch-Transformer load-balancing auxiliary loss.

        Args:
            gate_probs: Full softmax over experts, shape ``(B, E)``.
            top_k_indices: Indices of chosen experts, shape ``(B, top_k)``.

        Returns:
            Scalar auxiliary loss encouraging uniform expert utilisation.
        """
        n_experts = gate_probs.shape[-1]
        # Fraction of routing decisions dispatched to each expert.
        one_hot = F.one_hot(top_k_indices, num_classes=n_experts).float()
        dispatch_frac = one_hot.sum(dim=(0, 1)) / (
            gate_probs.shape[0] * self.top_k
        )
        # Mean gate probability per expert across the batch.
        mean_prob = gate_probs.mean(dim=0)
        # Classic formulation: n_experts * <f, P>.
        return n_experts * torch.sum(dispatch_frac * mean_prob)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Route ``x`` through the top-k experts.

        Args:
            x: Tensor of shape ``(B, d_model)``.

        Returns:
            Tuple ``(y, info)`` where ``y`` is the routed output
            ``(B, d_model)`` and ``info`` contains:

            * ``router_probs``: full softmax over experts, ``(B, E)``.
            * ``top_k_indices``: selected expert indices, ``(B, top_k)``.
            * ``top_k_weights``: renormalised weights, ``(B, top_k)``.
            * ``load_balancing_loss``: scalar auxiliary loss.
        """
        if x.dim() != 2:
            raise ValueError(f"expected (B, D) input; got {tuple(x.shape)}")

        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = logits.softmax(dim=-1)

        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute every expert once (cheap at this batch scale), then
        # gather the ones actually selected per example.
        stacked = torch.stack(
            [self.experts[name](x) for name in self.expert_names], dim=1
        )  # (B, E, D)

        gathered = torch.gather(
            stacked,
            dim=1,
            index=top_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )  # (B, top_k, D)
        y = (gathered * top_w.unsqueeze(-1)).sum(dim=1)

        aux = self._load_balancing_loss(probs, top_idx)

        info: Dict[str, torch.Tensor] = {
            "router_probs": probs,
            "top_k_indices": top_idx,
            "top_k_weights": top_w,
            "load_balancing_loss": aux,
        }
        return y, info
