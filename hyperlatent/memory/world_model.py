"""Semantic world model: forward-dynamics predictor in latent space.

The world model learns `f_theta(z_t, a_t) -> z_{t+1}` and supports online
updates via a single SGD step per `update()` call. A running mean of the
prediction loss is maintained and exposed for meta-controllers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


class SemanticWorldModel(nn.Module):
    """MLP-based latent-space forward dynamics model.

    The model concatenates the latent and action vectors, passes them through
    a residual MLP, and predicts the next latent. An internal `Adam` optimizer
    enables `update(transitions)` to perform a single online step, with the
    running mean prediction loss updated in-place.

    Attributes:
        latent_dim: Dimensionality of latent states.
        action_dim: Dimensionality of action embeddings.
        hidden_dim: Width of the MLP hidden layers.
        running_mean_loss: Exponentially-weighted mean of recent losses.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        ema_alpha: float = 0.05,
    ) -> None:
        """Initialize the world model and its online optimizer.

        Args:
            latent_dim: Dimensionality of latents.
            action_dim: Dimensionality of action embeddings.
            hidden_dim: Width of the MLP.
            lr: Adam learning rate for online updates.
            ema_alpha: EMA coefficient for `running_mean_loss`.
        """
        super().__init__()
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self._ema_alpha = ema_alpha

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

        self.register_buffer(
            "running_mean_loss", torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer("_update_count", torch.tensor(0, dtype=torch.long))

    # --------------------------------------------------------------- forward
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict the next latent `z_{t+1}`.

        The model outputs a residual `delta` added to `z_t` to stabilize
        long-horizon rollouts.

        Args:
            z: Latent tensor of shape `(..., latent_dim)`.
            a: Action tensor of shape `(..., action_dim)`.

        Returns:
            Predicted next-latent tensor of shape `(..., latent_dim)`.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z last-dim {z.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if a.shape[-1] != self.action_dim:
            raise ValueError(
                f"a last-dim {a.shape[-1]} != action_dim {self.action_dim}"
            )
        h = self.trunk(torch.cat([z, a], dim=-1))
        return z + self.delta_head(h)

    # ---------------------------------------------------------------- online
    def update(
        self,
        transitions: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """Perform a single SGD step on a batch of `(z, a, z_next)` tuples.

        Args:
            transitions: Iterable of `(z_t, a_t, z_{t+1})` tensors. Each
                element may be a single sample of shape `(latent_dim,)` /
                `(action_dim,)` or a pre-batched tensor.

        Returns:
            The scalar prediction loss for this batch.
        """
        z_list, a_list, zn_list = [], [], []
        for z, a, zn in transitions:
            if z.dim() == 1:
                z = z.unsqueeze(0)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if zn.dim() == 1:
                zn = zn.unsqueeze(0)
            z_list.append(z)
            a_list.append(a)
            zn_list.append(zn)

        if not z_list:
            return float(self.running_mean_loss.item())

        device = next(self.parameters()).device
        z_batch = torch.cat(z_list, dim=0).to(device, dtype=torch.float32)
        a_batch = torch.cat(a_list, dim=0).to(device, dtype=torch.float32)
        zn_batch = torch.cat(zn_list, dim=0).to(device, dtype=torch.float32)

        self.train()
        self._optimizer.zero_grad(set_to_none=True)
        pred = self.forward(z_batch, a_batch)
        loss = self._loss_fn(pred, zn_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self._optimizer.step()

        loss_val = float(loss.detach().item())
        with torch.no_grad():
            if int(self._update_count.item()) == 0:
                self.running_mean_loss.fill_(loss_val)
            else:
                new = (1 - self._ema_alpha) * float(
                    self.running_mean_loss.item()
                ) + self._ema_alpha * loss_val
                self.running_mean_loss.fill_(new)
            self._update_count += 1
        return loss_val

    # -------------------------------------------------------------- rollouts
    @torch.no_grad()
    def rollout(
        self, z0: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Unroll the model for a sequence of actions.

        Args:
            z0: Initial latent of shape `(B, latent_dim)` or `(latent_dim,)`.
            actions: Action tensor of shape `(B, H, action_dim)` or
                `(H, action_dim)`.

        Returns:
            Tensor of predicted latents with shape `(B, H, latent_dim)`.
        """
        self.eval()
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != z0.shape[0]:
            if actions.shape[0] == 1:
                actions = actions.expand(z0.shape[0], -1, -1)
            elif z0.shape[0] == 1:
                z0 = z0.expand(actions.shape[0], -1)
            else:
                raise ValueError("z0 and actions batch sizes must match")

        horizon = actions.shape[1]
        z = z0
        out = []
        for t in range(horizon):
            z = self.forward(z, actions[:, t, :])
            out.append(z)
        return torch.stack(out, dim=1)

    def diagnostics(self) -> Dict[str, float]:
        """Return a small diagnostic dict."""
        return {
            "running_mean_loss": float(self.running_mean_loss.item()),
            "updates": int(self._update_count.item()),
        }
