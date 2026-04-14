"""Transition replay buffer for world-model training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class Transition:
    """A single stored transition ``(z_t, a_t, z_{t+1}, r_t)``."""

    z_t: torch.Tensor
    a_t: torch.Tensor
    z_tp1: torch.Tensor
    r_t: torch.Tensor


class TransitionReplayBuffer:
    """Ring buffer over latent-space transitions.

    Supports both uniform and proportional prioritized sampling with a simple
    TD-error-style priority scalar per transition.

    Attributes:
        capacity: Maximum number of transitions stored.
        alpha: Priority exponent for prioritized sampling.
        eps: Small constant to avoid zero priorities.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6) -> None:
        """Initialize an empty buffer.

        Args:
            capacity: Maximum transitions to retain.
            alpha: Priority exponent in ``[0, 1]``.
            eps: Minimum priority floor.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.eps: float = eps
        self._storage: List[Transition] = []
        self._priorities: List[float] = []
        self._pos: int = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_tp1: torch.Tensor,
        r_t: torch.Tensor,
        priority: Optional[float] = None,
    ) -> None:
        """Insert a transition, overwriting the oldest slot when full.

        Args:
            z_t: Latent at time ``t`` of shape ``(D,)`` or ``(B, D)``.
            a_t: Action at time ``t``.
            z_tp1: Latent at time ``t+1``.
            r_t: Scalar reward.
            priority: Optional priority; defaults to max priority seen.
        """
        trans = Transition(
            z_t=z_t.detach().cpu(),
            a_t=a_t.detach().cpu(),
            z_tp1=z_tp1.detach().cpu(),
            r_t=r_t.detach().cpu(),
        )
        prio = (
            priority
            if priority is not None
            else (max(self._priorities) if self._priorities else 1.0)
        )
        if len(self._storage) < self.capacity:
            self._storage.append(trans)
            self._priorities.append(float(prio))
        else:
            self._storage[self._pos] = trans
            self._priorities[self._pos] = float(prio)
            self._pos = (self._pos + 1) % self.capacity

    def add_batch(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_tp1: torch.Tensor,
        r_t: torch.Tensor,
    ) -> None:
        """Insert a batch of transitions along the leading dim."""
        b = z_t.shape[0]
        for i in range(b):
            self.add(z_t[i], a_t[i], z_tp1[i], r_t[i])

    def _stack(self, samples: List[Transition]) -> Tuple[torch.Tensor, ...]:
        z_t = torch.stack([s.z_t for s in samples], dim=0)
        a_t = torch.stack([s.a_t for s in samples], dim=0)
        z_tp1 = torch.stack([s.z_tp1 for s in samples], dim=0)
        r_t = torch.stack([s.r_t.reshape(-1) for s in samples], dim=0).squeeze(-1)
        return z_t, a_t, z_tp1, r_t

    def sample_uniform(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample ``batch_size`` transitions.

        Returns:
            Tuple of stacked tensors ``(z_t, a_t, z_{t+1}, r_t)``.
        """
        if len(self) == 0:
            raise RuntimeError("cannot sample from empty buffer")
        idx = torch.randint(0, len(self), (batch_size,)).tolist()
        samples = [self._storage[i] for i in idx]
        return self._stack(samples)

    def sample_prioritized(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Proportional prioritized sampling.

        Args:
            batch_size: Number of transitions to draw.
            beta: Importance-sampling exponent.

        Returns:
            ``(z_t, a_t, z_{t+1}, r_t, is_weights, indices)``.
        """
        if len(self) == 0:
            raise RuntimeError("cannot sample from empty buffer")
        prios = torch.tensor(self._priorities[: len(self)], dtype=torch.float64)
        probs = (prios + self.eps).pow(self.alpha)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        samples = [self._storage[i] for i in idx.tolist()]
        n = len(self)
        is_w = (n * probs[idx]).pow(-beta)
        is_w = is_w / is_w.max()
        z_t, a_t, z_tp1, r_t = self._stack(samples)
        return z_t, a_t, z_tp1, r_t, is_w.float(), idx

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Overwrite priorities at the given indices."""
        for i, p in zip(indices.tolist(), priorities.detach().cpu().tolist()):
            if 0 <= i < len(self._priorities):
                self._priorities[i] = float(max(p, self.eps))

    def sample_negatives(self, n: int) -> torch.Tensor:
        """Draw ``n`` next-latent negatives for contrastive losses."""
        if len(self) == 0:
            raise RuntimeError("cannot sample from empty buffer")
        idx = torch.randint(0, len(self), (n,)).tolist()
        return torch.stack([self._storage[i].z_tp1 for i in idx], dim=0)
