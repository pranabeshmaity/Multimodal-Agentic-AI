"""Training configuration for the Hyper-Latent Fusion system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainingConfig:
    """Hyperparameter container for :class:`HyperLatentTrainer`.

    Attributes:
        lr: Base learning rate for AdamW.
        betas: AdamW betas.
        weight_decay: AdamW weight decay.
        ema_tau: EMA decay for target encoder updates (0 < tau < 1).
        accumulation_steps: Gradient accumulation step count.
        grad_clip: Maximum gradient norm. Set <= 0 to disable.
        amp: Whether to enable ``torch.cuda.amp`` mixed precision.
        jepa_coef: Weight for JEPA predictive loss.
        world_model_coef: Weight for world-model loss.
        moe_balance_coef: Weight for MoE load-balancing loss.
        vicreg_coef: Weight for VICReg regularization.
        contrastive_coef: Weight for the world-model contrastive term.
        num_modalities: Number of modality streams (vision, audio, text = 3).
        latent_dim: Hyper-latent dimensionality.
        predictor_hidden: Hidden size of the JEPA predictor MLP.
        action_dim: Dimensionality of actions in the world model.
        num_experts: Number of experts in the MoE router.
        batch_size: Per-step batch size for the mock loader.
        log_every: Print/log cadence in steps.
        checkpoint_every: Checkpointing cadence; 0 disables.
        ckpt_dir: Directory for checkpoint files.
        device: Torch device string.
        seed: RNG seed.
        replay_capacity: Transition replay buffer capacity.
        num_negatives: Negatives drawn from replay for contrastive term.
    """

    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    ema_tau: float = 0.996
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    amp: bool = False

    jepa_coef: float = 1.0
    world_model_coef: float = 0.5
    moe_balance_coef: float = 0.01
    vicreg_coef: float = 0.1
    contrastive_coef: float = 0.1

    num_modalities: int = 3
    latent_dim: int = 64
    predictor_hidden: int = 128
    action_dim: int = 8
    num_experts: int = 4

    batch_size: int = 4
    log_every: int = 1
    checkpoint_every: int = 0
    ckpt_dir: str = "./checkpoints"
    device: str = "cpu"
    seed: int = 42

    replay_capacity: int = 1024
    num_negatives: int = 8

    extras: dict = field(default_factory=dict)
