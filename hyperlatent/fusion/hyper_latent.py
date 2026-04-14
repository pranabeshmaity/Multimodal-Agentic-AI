"""Hyper-Latent projector and VICReg regulariser.

Each modality is mapped into the shared space Z via a near-isometric
MLP ``phi_m``. We encourage the projections to preserve structure and
to be informative using the VICReg triple objective (invariance,
variance, covariance).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsometricMLP(nn.Module):
    """Small MLP whose final linear layer is initialised near-isometric.

    Near-isometry is approximated by orthogonal initialisation of the
    final linear map and disabling its bias; combined with LayerNorm on
    the input this keeps distances well-behaved at init.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden dimension.
        out_dim: Output (shared) dimension.
        depth: Number of hidden blocks (``>= 1``).
        dropout: Dropout probability.

    Shape:
        Input:  ``(B, in_dim)``
        Output: ``(B, out_dim)``
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.final = nn.Linear(prev, out_dim, bias=False)
        # Orthogonal init promotes approximate isometry.
        nn.init.orthogonal_(self.final.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` into the shared space.

        Args:
            x: Input features of shape ``(B, in_dim)``.

        Returns:
            Projected vector of shape ``(B, out_dim)``.
        """
        return self.final(self.trunk(x))


class HyperLatentProjector(nn.Module):
    """Bank of modality-specific projectors ``phi_m`` into shared Z.

    Args:
        modality_dims: Mapping from modality name to its input feature
            dimension.
        hidden_dim: Hidden size for each projector's MLP.
        z_dim: Dimensionality of the shared latent space Z.
        depth: Depth of each projector MLP.
        dropout: Dropout in the projectors.
        pool: Pooling strategy for inputs of shape ``(B, T, D)``. Use
            ``"mean"`` (default) or ``"cls"`` (first token).

    Shape:
        Input per modality: ``(B, D_m)`` or ``(B, T_m, D_m)``.
        Output per modality: ``(B, z_dim)``.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        hidden_dim: int = 1024,
        z_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool not in ("mean", "cls"):
            raise ValueError("pool must be 'mean' or 'cls'")
        self.pool = pool
        self.z_dim = z_dim
        self.projectors = nn.ModuleDict(
            {
                name: IsometricMLP(
                    in_dim=dim,
                    hidden_dim=hidden_dim,
                    out_dim=z_dim,
                    depth=depth,
                    dropout=dropout,
                )
                for name, dim in modality_dims.items()
            }
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence tensor to a single vector."""
        if x.dim() == 2:
            return x
        if self.pool == "cls":
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project every available modality into Z.

        Args:
            features: Mapping modality-name -> tensor. Each tensor may be
                either ``(B, D)`` or ``(B, T, D)``.

        Returns:
            Mapping modality-name -> projected latent of shape
            ``(B, z_dim)``.
        """
        out: Dict[str, torch.Tensor] = {}
        for name, x in features.items():
            if name not in self.projectors:
                raise KeyError(f"no projector registered for modality '{name}'")
            pooled = self._pool(x)
            out[name] = self.projectors[name](pooled)
        return out


class VICRegLoss(nn.Module):
    """VICReg tri-term loss on a set of projected latents.

    * Invariance:  MSE between pairs of modalities (pulls them together).
    * Variance:    hinge on per-dim standard deviation to avoid collapse.
    * Covariance:  off-diagonal Gram penalty to decorrelate features.

    Args:
        inv_weight: Weight of the invariance term.
        var_weight: Weight of the variance (std-hinge) term.
        cov_weight: Weight of the covariance term.
        gamma: Target per-dimension standard deviation used in the hinge.
        eps: Numerical stabiliser for the std computation.

    Shape:
        Input: mapping modality-name -> ``(B, D)``.
        Output: scalar tensor.
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma
        self.eps = eps

    @staticmethod
    def _variance_term(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
        """Std-hinge term per VICReg."""
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return F.relu(gamma - std).mean()

    @staticmethod
    def _covariance_term(z: torch.Tensor) -> torch.Tensor:
        """Sum of squared off-diagonal elements of the covariance matrix."""
        b, d = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(b - 1, 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return (off_diag.pow(2).sum()) / d

    def forward(
        self, latents: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined VICReg loss.

        Args:
            latents: Mapping modality-name -> tensor ``(B, D)``. At least
                two modalities are required for the invariance term; a
                single-modality call still returns variance + covariance.

        Returns:
            Tuple ``(total_loss, components)`` where ``components`` is a
            dict of the individual scalar terms.
        """
        names = list(latents.keys())
        if not names:
            raise ValueError("at least one modality required")

        var_terms: List[torch.Tensor] = []
        cov_terms: List[torch.Tensor] = []
        for name in names:
            z = latents[name]
            if z.dim() != 2:
                raise ValueError(
                    f"latent '{name}' must be 2D (B, D); got {tuple(z.shape)}"
                )
            var_terms.append(self._variance_term(z, self.gamma, self.eps))
            cov_terms.append(self._covariance_term(z))

        var_loss = torch.stack(var_terms).mean()
        cov_loss = torch.stack(cov_terms).mean()

        inv_loss = torch.zeros((), device=latents[names[0]].device)
        pair_count = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inv_loss = inv_loss + F.mse_loss(latents[names[i]], latents[names[j]])
                pair_count += 1
        if pair_count > 0:
            inv_loss = inv_loss / pair_count

        total = (
            self.inv_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )
        components = {
            "vicreg_inv": inv_loss.detach(),
            "vicreg_var": var_loss.detach(),
            "vicreg_cov": cov_loss.detach(),
        }
        return total, components
