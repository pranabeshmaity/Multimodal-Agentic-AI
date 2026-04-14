"""Speculative rollout engine for latent-space planning.

Given a current latent `z_t` and a discrete set of candidate action embeddings
`A`, the engine samples `K` Gaussian-perturbed rollouts of horizon `H` per
action using the world model, scores each rollout with a learned reward head,
and returns the argmax action along with per-action value estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hyperlatent.memory.world_model import SemanticWorldModel


class RewardHead(nn.Module):
    """MLP predicting a scalar reward from a latent."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        """Initialize the reward head.

        Args:
            latent_dim: Dimensionality of input latents.
            hidden_dim: Width of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return a `(..., 1)` reward tensor for latents `z`."""
        return self.net(z)


@dataclass
class RolloutResult:
    """Container for the output of a speculative rollout.

    Attributes:
        best_action_index: Index of the argmax action.
        values: `(num_actions,)` tensor of expected discounted returns.
        best_value: Scalar value of the chosen action.
    """

    best_action_index: int
    values: torch.Tensor
    best_value: float


class SpeculativeRolloutEngine:
    """Monte-Carlo planner in latent space.

    For each candidate action the engine unrolls the world model `K` times
    for `H` steps, injecting Gaussian noise into each predicted latent to
    approximate aleatoric uncertainty. The reward head scores every step, and
    the resulting returns are discounted and averaged over samples.
    """

    def __init__(
        self,
        world_model: SemanticWorldModel,
        reward_head: Optional[RewardHead] = None,
        horizon: int = 5,
        num_samples: int = 16,
        discount: float = 0.95,
        noise_scale: float = 0.05,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the rollout engine.

        Args:
            world_model: Trained/partially-trained world model.
            reward_head: Learned reward head; one is constructed if None.
            horizon: Number of steps `H` per rollout.
            num_samples: Number of rollouts `K` per candidate action.
            discount: Per-step discount factor `gamma`.
            noise_scale: Std of Gaussian noise injected per step.
            device: Optional torch device.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")

        self.world_model = world_model
        self.reward_head = reward_head or RewardHead(world_model.latent_dim)
        self.horizon = horizon
        self.num_samples = num_samples
        self.discount = discount
        self.noise_scale = noise_scale
        self.device = device or next(world_model.parameters()).device

        self.world_model.to(self.device)
        self.reward_head.to(self.device)

    @torch.no_grad()
    def plan(
        self, z: torch.Tensor, candidate_actions: torch.Tensor
    ) -> RolloutResult:
        """Select the best action via speculative rollouts.

        Args:
            z: Current latent of shape `(latent_dim,)`.
            candidate_actions: Tensor of shape `(A, action_dim)` listing the
                candidate action embeddings.

        Returns:
            A `RolloutResult` summarizing per-action values and the argmax.
        """
        if z.dim() != 1:
            raise ValueError("z must be a single latent of shape (latent_dim,)")
        if candidate_actions.dim() != 2:
            raise ValueError("candidate_actions must be shape (A, action_dim)")

        self.world_model.eval()
        self.reward_head.eval()

        z = z.to(self.device, dtype=torch.float32)
        actions = candidate_actions.to(self.device, dtype=torch.float32)
        num_actions, action_dim = actions.shape
        B = num_actions * self.num_samples

        # Expand: (A, K, action_dim) -> (B, action_dim)
        z_batch = z.unsqueeze(0).expand(B, -1).contiguous()
        first_action = actions.unsqueeze(1).expand(
            num_actions, self.num_samples, action_dim
        ).reshape(B, action_dim)

        discounts = torch.tensor(
            [self.discount ** t for t in range(self.horizon)],
            device=self.device,
            dtype=torch.float32,
        )

        returns = torch.zeros(B, device=self.device, dtype=torch.float32)
        cur_z = z_batch
        cur_a = first_action
        for t in range(self.horizon):
            next_z = self.world_model(cur_z, cur_a)
            if self.noise_scale > 0:
                next_z = next_z + torch.randn_like(next_z) * self.noise_scale
            r = self.reward_head(next_z).squeeze(-1)
            returns = returns + discounts[t] * r
            cur_z = next_z
            # Re-use the same committed action across the horizon as an
            # open-loop plan; downstream callers may substitute a policy.
            cur_a = first_action

        per_action = returns.view(num_actions, self.num_samples).mean(dim=1)
        best_idx = int(torch.argmax(per_action).item())
        return RolloutResult(
            best_action_index=best_idx,
            values=per_action.detach().cpu(),
            best_value=float(per_action[best_idx].item()),
        )

    def train_reward_head(
        self, latents: torch.Tensor, rewards: torch.Tensor, lr: float = 1e-3
    ) -> float:
        """Fit the reward head with a single SGD step.

        Args:
            latents: `(B, latent_dim)` tensor of observed latents.
            rewards: `(B,)` tensor of observed scalar rewards.
            lr: Learning rate for this step.

        Returns:
            The scalar MSE loss.
        """
        latents = latents.to(self.device, dtype=torch.float32)
        rewards = rewards.to(self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.reward_head.parameters(), lr=lr)
        self.reward_head.train()
        optimizer.zero_grad(set_to_none=True)
        pred = self.reward_head(latents).squeeze(-1)
        loss = nn.functional.mse_loss(pred, rewards)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())
