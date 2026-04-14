"""Typed container dataclasses for the Hyper-Latent Fusion pipeline.

This module defines the lightweight structures shuffled between the
modality-specific encoders, the cross-modal fusion stack and the latent
space projectors. Keeping these as dataclasses (rather than bare tuples)
improves readability inside training loops and allows easy extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ModalityBatch:
    """Container for a single multimodal training / inference batch.

    Each field is optional so partial batches (e.g. text-only) are legal.
    Tensors should already be moved to the correct device before being
    packed into this structure.

    Args:
        vision: Image tensor of shape ``(B, C, H, W)``.
        audio: Raw or mel-spectrogram audio of shape ``(B, C, T_audio)``.
        text: Integer token ids of shape ``(B, T_text)``.
        vision_mask: Optional attention mask for vision patches, shape
            ``(B, N_patches)``.
        audio_mask: Optional attention mask for audio frames.
        text_mask: Optional attention mask for text tokens.
        metadata: Free-form dictionary for auxiliary info (ids, labels).

    Shape:
        vision: ``(B, 3, H, W)``
        audio:  ``(B, 1, T_audio)``
        text:   ``(B, T_text)``
    """

    vision: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None
    vision_mask: Optional[torch.Tensor] = None
    audio_mask: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModalityBatch":
        """Move every tensor field to ``device`` in-place-ish.

        Args:
            device: Target torch device.

        Returns:
            A new ``ModalityBatch`` whose tensors live on ``device``.
        """
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if isinstance(t, torch.Tensor) else t

        return ModalityBatch(
            vision=_mv(self.vision),
            audio=_mv(self.audio),
            text=_mv(self.text),
            vision_mask=_mv(self.vision_mask),
            audio_mask=_mv(self.audio_mask),
            text_mask=_mv(self.text_mask),
            metadata={k: _mv(v) for k, v in self.metadata.items()},
        )

    def available_modalities(self) -> Dict[str, bool]:
        """Return a dict indicating which modalities are present.

        Returns:
            Mapping from modality name to a boolean availability flag.
        """
        return {
            "vision": self.vision is not None,
            "audio": self.audio is not None,
            "text": self.text is not None,
        }


@dataclass
class FusedLatent:
    """Output of the Hyper-Latent fusion stack.

    Args:
        z_shared: Fused latent in the shared space Z, shape ``(B, D_z)``.
        z_per_modality: Per-modality projected latents keyed by modality
            name, each of shape ``(B, D_z)``.
        aux_losses: Dictionary of scalar auxiliary losses (VICReg,
            load-balancing, etc.) keyed by name.
        router_weights: Optional soft routing weights, shape
            ``(B, n_experts)``.
        ode_trajectory: Optional list of intermediate latent states emitted
            by the Latent ODE integrator.

    Shape:
        z_shared: ``(B, D_z)``
        z_per_modality[m]: ``(B, D_z)``
    """

    z_shared: torch.Tensor
    z_per_modality: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    router_weights: Optional[torch.Tensor] = None
    ode_trajectory: Optional[torch.Tensor] = None

    def total_aux_loss(self) -> torch.Tensor:
        """Sum all auxiliary losses into a single scalar.

        Returns:
            Scalar tensor equal to the sum of ``aux_losses`` values, or a
            zero tensor on ``z_shared``'s device if no losses are set.
        """
        if not self.aux_losses:
            return torch.zeros((), device=self.z_shared.device)
        return torch.stack(list(self.aux_losses.values())).sum()
