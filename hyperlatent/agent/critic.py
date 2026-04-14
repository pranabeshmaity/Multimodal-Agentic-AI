"""Self-correction critic scoring `(latent, action, observation)` triples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Critique:
    """Output of the self-correction critic.

    Attributes:
        score: Scalar quality score in `[0, 1]`.
        text: Natural-language critique suitable for prompt injection.
    """

    score: float
    text: str


class SelfCorrectionCritic(nn.Module):
    """MLP critic over `(z_t, a_t, o_{t+1})`.

    The critic ingests a concatenated triple and emits a sigmoid score. A
    small templated message is produced alongside to feed back into the
    agent's context when the score falls below threshold.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        obs_dim: Optional[int] = None,
        hidden_dim: int = 128,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the critic.

        Args:
            latent_dim: Dimensionality of latents `z_t`.
            action_dim: Dimensionality of action embeddings `a_t`.
            obs_dim: Dimensionality of observation embeddings; defaults to
                `latent_dim` when `None`.
            hidden_dim: Width of the hidden layer.
            threshold: Acceptance threshold on the sigmoid score.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim or latent_dim
        self.threshold = threshold

        in_dim = latent_dim + action_dim + self.obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, o: torch.Tensor
    ) -> torch.Tensor:
        """Return a `(B,)` sigmoid score in `[0, 1]`."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if o.dim() == 1:
            o = o.unsqueeze(0)
        x = torch.cat([z, a, o], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))

    @torch.no_grad()
    def critique(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        o: torch.Tensor,
        action_name: str = "action",
    ) -> Critique:
        """Score a single `(z, a, o)` triple and return a textual critique.

        Args:
            z: Latent tensor `(latent_dim,)`.
            a: Action embedding `(action_dim,)`.
            o: Observation embedding `(obs_dim,)`.
            action_name: Human-readable name of the action for the template.

        Returns:
            A `Critique` with score and natural-language text.
        """
        self.eval()
        score = float(self.forward(z, a, o).item())
        if score >= self.threshold:
            text = (
                f"Action '{action_name}' looks successful (confidence "
                f"{score:.2f}). Continue with the next step."
            )
        else:
            # Residual between predicted latent continuity and observation.
            if o.shape[-1] == z.shape[-1]:
                drift = float((o - z).norm().item())
                drift_note = f" Observation drifted by {drift:.3f} in latent norm."
            else:
                drift_note = ""
            text = (
                f"Action '{action_name}' produced a low-confidence outcome "
                f"(score {score:.2f} < threshold {self.threshold:.2f})."
                f"{drift_note} Re-plan: consider an alternative action, verify "
                "preconditions, or request clarifying information before "
                "proceeding."
            )
        return Critique(score=score, text=text)

    def is_acceptable(self, critique: Critique) -> bool:
        """Return whether a critique's score clears the threshold."""
        return critique.score >= self.threshold
