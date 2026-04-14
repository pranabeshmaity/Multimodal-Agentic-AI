"""Exponential moving average wrapper for target encoders (JEPA style)."""
from __future__ import annotations

import copy
from typing import Iterator

import torch
from torch import nn


class EMATargetEncoder(nn.Module):
    """Maintains a non-trainable EMA copy of a source encoder.

    The target module mirrors the source architecture and parameters but is
    updated with a Polyak average rather than gradient descent. Used as the
    stop-gradient target in self-predictive objectives (JEPA / BYOL).

    Attributes:
        target: Detached EMA copy of the wrapped encoder.
    """

    def __init__(self, source: nn.Module) -> None:
        """Initialize the target as a deep copy of ``source``.

        Args:
            source: The online/context encoder whose EMA is tracked.
        """
        super().__init__()
        self.target: nn.Module = copy.deepcopy(source)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module, tau: float) -> None:
        """Polyak update: ``theta_tgt <- tau * theta_tgt + (1-tau) * theta_src``.

        Args:
            source: The online encoder providing fresh parameters.
            tau: EMA decay in ``[0, 1]``. Higher values retain more of the
                previous target.
        """
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        src_params = dict(source.named_parameters())
        src_buffers = dict(source.named_buffers())
        for name, p_tgt in self.target.named_parameters():
            if name in src_params:
                p_tgt.data.mul_(tau).add_(src_params[name].data, alpha=1.0 - tau)
        for name, b_tgt in self.target.named_buffers():
            if name in src_buffers:
                b_tgt.data.copy_(src_buffers[name].data)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the target encoder in ``no_grad`` mode.

        Returns:
            The target encoder's output with gradients detached.
        """
        with torch.no_grad():
            out = self.target(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out.detach()
        return out

    def parameters_iter(self) -> Iterator[nn.Parameter]:
        """Iterate the target's parameters (all frozen)."""
        return self.target.parameters()
