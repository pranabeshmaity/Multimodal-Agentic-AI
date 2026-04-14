"""Fixed-step Euler Latent ODE.

We evolve a latent ``z`` under a learned vector field
``dz/dt = f_psi(z, c)`` conditioned on a control signal ``c``. A plain
fixed-step forward-Euler integrator keeps the dependency surface tiny
(no ``torchdiffeq``) while still giving the model a continuous-time
inductive bias for Phase 1 experiments.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    """Learned vector field ``f_psi(z, c) -> dz/dt``.

    Args:
        z_dim: Dimensionality of the latent.
        c_dim: Dimensionality of the (optional) control/context vector.
        hidden_dim: MLP inner dimension.
        dropout: Dropout probability.

    Shape:
        z: ``(B, z_dim)``
        c: ``(B, c_dim)`` or ``None``
        Output: ``(B, z_dim)``
    """

    def __init__(
        self,
        z_dim: int = 512,
        c_dim: int = 0,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        in_dim = z_dim + c_dim + 1  # +1 for the time coordinate.
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Zero-init the last layer so the ODE is initially an identity
        # flow (dz/dt ≈ 0), preventing early training blow-ups.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, t: torch.Tensor, z: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluate the vector field.

        Args:
            t: Scalar time tensor of shape ``()`` or ``(B,)``.
            z: Latent state of shape ``(B, z_dim)``.
            c: Optional control of shape ``(B, c_dim)``.

        Returns:
            Tensor of shape ``(B, z_dim)``.
        """
        b = z.shape[0]
        if t.dim() == 0:
            t_feat = t.expand(b).unsqueeze(-1)
        else:
            t_feat = t.view(b, 1)
        parts: List[torch.Tensor] = [z, t_feat]
        if self.c_dim > 0:
            if c is None:
                raise ValueError("control c required when c_dim > 0")
            if c.shape[-1] != self.c_dim:
                raise ValueError(f"c dim {c.shape[-1]} != expected {self.c_dim}")
            parts.append(c)
        elif c is not None:
            # Silently ignore extra control if ODE is unconditional.
            pass
        return self.net(torch.cat(parts, dim=-1))


class LatentODE(nn.Module):
    """Thin fixed-step Euler integrator over the learned field ``f_psi``.

    Args:
        z_dim: Latent dimensionality.
        c_dim: Control dimensionality (``0`` for unconditional).
        hidden_dim: MLP inner dimension of ``ODEFunc``.
        n_steps: Number of Euler steps between ``t0`` and ``t1``.
        t0: Initial time.
        t1: Final time.
        return_trajectory: If True, ``forward`` additionally returns all
            intermediate states.
        dropout: Dropout in the vector field.

    Shape:
        Input z: ``(B, z_dim)``
        Output z_T: ``(B, z_dim)``
        Output trajectory (optional): ``(n_steps + 1, B, z_dim)``.
    """

    def __init__(
        self,
        z_dim: int = 512,
        c_dim: int = 0,
        hidden_dim: int = 1024,
        n_steps: int = 8,
        t0: float = 0.0,
        t1: float = 1.0,
        return_trajectory: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        self.n_steps = n_steps
        self.t0 = t0
        self.t1 = t1
        self.return_trajectory = return_trajectory
        self.func = ODEFunc(
            z_dim=z_dim, c_dim=c_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(
        self, z0: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Integrate ``z0`` forward through time with Euler steps.

        Args:
            z0: Initial latent state of shape ``(B, z_dim)``.
            c: Optional control of shape ``(B, c_dim)``.

        Returns:
            Tuple ``(z_T, trajectory)`` where ``trajectory`` is ``None``
            unless ``return_trajectory`` is set.
        """
        dt = (self.t1 - self.t0) / self.n_steps
        z = z0
        traj: List[torch.Tensor] = [z0] if self.return_trajectory else []
        t = torch.tensor(self.t0, device=z0.device, dtype=z0.dtype)
        dt_tensor = torch.tensor(dt, device=z0.device, dtype=z0.dtype)
        for _ in range(self.n_steps):
            dz = self.func(t, z, c)
            z = z + dt_tensor * dz
            t = t + dt_tensor
            if self.return_trajectory:
                traj.append(z)
        trajectory = torch.stack(traj, dim=0) if self.return_trajectory else None
        return z, trajectory
