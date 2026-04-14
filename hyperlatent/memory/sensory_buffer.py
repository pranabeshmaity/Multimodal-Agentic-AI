"""Sensory buffer: short-term ring buffer of latent observations.

This module implements the first stage of the Hyper-Latent Fusion memory
hierarchy: a fixed-capacity ring buffer that holds the most recent latent
observations. When the buffer overflows, the evicted item is forwarded to an
optional consolidation hook so that episodic memory can absorb it.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch


class SensoryBuffer:
    """Fixed-capacity ring buffer for latent observations.

    Implements O(1) amortized push via a circular list and a monotonically
    increasing write head. On overflow, the oldest entry is handed to an
    optional `consolidation_hook` so that downstream episodic memory can be
    updated without blocking the sensory path.

    Attributes:
        capacity: Maximum number of latents the buffer can hold.
        latent_dim: Dimensionality of each latent vector.
        device: Torch device on which latents are stored.
    """

    def __init__(
        self,
        capacity: int,
        latent_dim: int,
        device: Optional[torch.device] = None,
        consolidation_hook: Optional[Callable[[torch.Tensor, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the sensory buffer.

        Args:
            capacity: Maximum number of latents to retain.
            latent_dim: Dimensionality of each latent vector.
            device: Optional torch device; defaults to CPU.
            consolidation_hook: Callable invoked on eviction with
                `(latent, metadata)`; used to push oldest entries to
                episodic memory.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.capacity: int = capacity
        self.latent_dim: int = latent_dim
        self.device: torch.device = device or torch.device("cpu")
        self._hook = consolidation_hook

        self._storage: torch.Tensor = torch.zeros(
            (capacity, latent_dim), dtype=torch.float32, device=self.device
        )
        self._meta: List[Optional[Dict[str, Any]]] = [None] * capacity
        self._head: int = 0  # next write index
        self._size: int = 0  # current populated count
        self._total_pushes: int = 0

    def push(self, z: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Push a latent observation into the buffer.

        Args:
            z: Latent tensor of shape `(latent_dim,)`.
            metadata: Optional per-observation metadata.
        """
        if z.shape != (self.latent_dim,):
            raise ValueError(
                f"expected latent of shape ({self.latent_dim},), got {tuple(z.shape)}"
            )

        meta: Dict[str, Any] = dict(metadata or {})
        meta.setdefault("t", time.time())
        meta.setdefault("idx", self._total_pushes)

        if self._size == self.capacity and self._hook is not None:
            evicted = self._storage[self._head].detach().clone()
            evicted_meta = self._meta[self._head] or {}
            try:
                self._hook(evicted, evicted_meta)
            except Exception:
                # Consolidation failures must not break the sensory path.
                pass

        self._storage[self._head] = z.detach().to(self.device, dtype=torch.float32)
        self._meta[self._head] = meta
        self._head = (self._head + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_pushes += 1

    def latest(self, n: int = 1) -> torch.Tensor:
        """Return the most recent `n` latents in chronological order.

        Args:
            n: Number of latents to return. Clamped to current size.

        Returns:
            Tensor of shape `(min(n, size), latent_dim)`.
        """
        n = max(0, min(n, self._size))
        if n == 0:
            return torch.empty((0, self.latent_dim), device=self.device)
        indices = [(self._head - 1 - i) % self.capacity for i in range(n)][::-1]
        return self._storage[indices].clone()

    def snapshot(self) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Return all populated entries in chronological order."""
        if self._size == 0:
            return torch.empty((0, self.latent_dim), device=self.device), []
        if self._size < self.capacity:
            idx = list(range(self._size))
        else:
            idx = [(self._head + i) % self.capacity for i in range(self.capacity)]
        metas = [self._meta[i] or {} for i in idx]
        return self._storage[idx].clone(), metas

    def drain_to_hook(self) -> int:
        """Flush all current contents through the consolidation hook.

        Returns:
            The number of items forwarded to the hook.
        """
        if self._hook is None or self._size == 0:
            count = self._size
            self._size = 0
            self._head = 0
            return 0
        tensors, metas = self.snapshot()
        for i in range(tensors.shape[0]):
            try:
                self._hook(tensors[i], metas[i])
            except Exception:
                pass
        count = self._size
        self._size = 0
        self._head = 0
        return count

    def set_consolidation_hook(
        self, hook: Callable[[torch.Tensor, Dict[str, Any]], None]
    ) -> None:
        """Register (or replace) the consolidation hook."""
        self._hook = hook

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        tensors, metas = self.snapshot()
        for i in range(tensors.shape[0]):
            yield tensors[i], metas[i]
