"""Loss objectives for the Hyper-Latent Fusion training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _PredictorMLP(nn.Module):
    """Small MLP that predicts a target latent from a context latent."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAPredictiveLoss(nn.Module):
    """Joint-Embedding Predictive Architecture loss.

    Predicts the target-encoder latent from the context-encoder latent via a
    small MLP predictor. The target side is stop-gradient; the target encoder
    parameters are expected to be updated by EMA elsewhere.

    Attributes:
        predictor: MLP mapping context -> predicted target.
    """

    def __init__(self, dim: int, hidden: int = 512) -> None:
        """Build the predictor MLP.

        Args:
            dim: Latent dimension (shared by context and target).
            hidden: Hidden width of the predictor MLP.
        """
        super().__init__()
        self.predictor: nn.Module = _PredictorMLP(dim, hidden)

    def forward(
        self, context_latent: torch.Tensor, target_latent: torch.Tensor
    ) -> torch.Tensor:
        """Compute the JEPA prediction loss.

        Args:
            context_latent: Online-encoder latent, shape ``(B, D)``.
            target_latent: Target-encoder latent (will be detached), ``(B, D)``.

        Returns:
            Scalar loss: normalized MSE between prediction and target.
        """
        pred = self.predictor(context_latent)
        target = target_latent.detach()
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        return 2.0 - 2.0 * (pred_n * tgt_n).sum(dim=-1).mean()


class WorldModelLoss(nn.Module):
    """Latent-space world-model loss.

    MSE between the world-model's predicted next-latent and the observed
    next-latent, with an optional InfoNCE-style contrastive term comparing the
    prediction against negatives drawn from the replay buffer.

    Attributes:
        temperature: Softmax temperature for the contrastive term.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature: float = temperature

    def forward(
        self,
        predicted_next: torch.Tensor,
        observed_next: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reconstruction + optional contrastive loss.

        Args:
            predicted_next: World-model rollout latent, ``(B, D)``.
            observed_next: Ground-truth next latent from encoder, ``(B, D)``.
            negatives: Optional ``(K, D)`` negatives from the replay buffer.

        Returns:
            Tuple ``(total, mse_term, contrastive_term)``. The contrastive term
            is zero if ``negatives`` is None.
        """
        mse = F.mse_loss(predicted_next, observed_next)
        contrastive = predicted_next.new_zeros(())
        if negatives is not None and negatives.numel() > 0:
            p = F.normalize(predicted_next, dim=-1)
            pos = F.normalize(observed_next, dim=-1)
            neg = F.normalize(negatives.to(p.device), dim=-1)
            pos_logit = (p * pos).sum(dim=-1, keepdim=True) / self.temperature
            neg_logits = p @ neg.t() / self.temperature
            logits = torch.cat([pos_logit, neg_logits], dim=-1)
            labels = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
            contrastive = F.cross_entropy(logits, labels)
        return mse + contrastive, mse, contrastive


class MoEBalanceLoss(nn.Module):
    """Switch-transformer load-balancing loss.

    Computes ``N * sum_i f_i * P_i`` where ``N`` is the number of experts,
    ``f_i`` is the fraction of tokens routed to expert ``i`` (hard assignment),
    and ``P_i`` is the mean softmax probability assigned to expert ``i``.
    """

    def forward(
        self, router_logits: torch.Tensor, expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the load-balance penalty.

        Args:
            router_logits: ``(T, N)`` pre-softmax logits over experts.
            expert_indices: Optional hard routing indices ``(T,)``. When not
                given, uses argmax of ``router_logits``.

        Returns:
            Scalar balance loss.
        """
        if router_logits.dim() != 2:
            router_logits = router_logits.reshape(-1, router_logits.shape[-1])
        num_experts = router_logits.shape[-1]
        probs = F.softmax(router_logits, dim=-1)
        mean_prob = probs.mean(dim=0)
        if expert_indices is None:
            expert_indices = router_logits.argmax(dim=-1)
        one_hot = F.one_hot(expert_indices.reshape(-1), num_classes=num_experts).float()
        frac = one_hot.mean(dim=0)
        return num_experts * (frac * mean_prob).sum()


@dataclass
class ObjectiveWeights:
    """Coefficient container for :class:`TotalObjective`."""

    jepa: float = 1.0
    world_model: float = 0.5
    moe_balance: float = 0.01
    vicreg: float = 0.1
    contrastive: float = 0.1


class TotalObjective(nn.Module):
    """Aggregate objective: weighted sum of all component losses.

    The ``forward`` accepts a dictionary of already-computed component losses
    and returns both the aggregated scalar and a dictionary suitable for
    logging.

    Attributes:
        weights: :class:`ObjectiveWeights` with per-term coefficients.
    """

    def __init__(self, weights: Optional[ObjectiveWeights] = None) -> None:
        super().__init__()
        self.weights: ObjectiveWeights = weights or ObjectiveWeights()

    def forward(self, components: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combine per-term losses into a total objective.

        Args:
            components: Mapping with any subset of keys
                ``{"jepa", "world_model", "moe_balance", "vicreg",
                   "contrastive"}``. Extra keys are passed through to the log.

        Returns:
            ``(total_scalar, log_dict)`` where ``log_dict`` contains every
            component as a Python float plus the ``"total"`` key.
        """
        coefs = {
            "jepa": self.weights.jepa,
            "world_model": self.weights.world_model,
            "moe_balance": self.weights.moe_balance,
            "vicreg": self.weights.vicreg,
            "contrastive": self.weights.contrastive,
        }
        total: Optional[torch.Tensor] = None
        log: Dict[str, float] = {}
        for k, v in components.items():
            if not isinstance(v, torch.Tensor):
                log[k] = float(v)
                continue
            log[k] = float(v.detach().cpu().item())
            if k in coefs:
                contrib = coefs[k] * v
                total = contrib if total is None else total + contrib
        if total is None:
            total = torch.zeros((), device=next(iter(components.values())).device
                                if any(isinstance(v, torch.Tensor) for v in components.values())
                                else "cpu")
        log["total"] = float(total.detach().cpu().item())
        return total, log


def jepa_pairs(num_modalities: int) -> List[Tuple[int, int]]:
    """Enumerate ordered modality pairs ``(i, j)`` with ``i != j``."""
    return [(i, j) for i in range(num_modalities) for j in range(num_modalities) if i != j]
