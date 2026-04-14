"""Phase-3 training package for Hyper-Latent Fusion."""
from __future__ import annotations

from .config import TrainingConfig
from .ema import EMATargetEncoder
from .objectives import (
    JEPAPredictiveLoss,
    MoEBalanceLoss,
    ObjectiveWeights,
    TotalObjective,
    WorldModelLoss,
)
from .replay import Transition, TransitionReplayBuffer
from .trainer import HyperLatentTrainer

__all__ = [
    "TrainingConfig",
    "EMATargetEncoder",
    "JEPAPredictiveLoss",
    "WorldModelLoss",
    "MoEBalanceLoss",
    "TotalObjective",
    "ObjectiveWeights",
    "Transition",
    "TransitionReplayBuffer",
    "HyperLatentTrainer",
]
