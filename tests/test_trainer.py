"""End-to-end trainer smoke test."""
from __future__ import annotations

from hyperlatent.training import HyperLatentTrainer, TrainingConfig


def test_trainer_runs_two_steps_and_loss_is_finite() -> None:
    cfg = TrainingConfig(
        num_modalities=3,
        latent_dim=64,
        predictor_hidden=128,
        action_dim=8,
        num_experts=4,
        batch_size=2,
        log_every=0,
        device="cpu",
        seed=0,
    )
    trainer = HyperLatentTrainer(cfg)
    history = trainer.fit(num_steps=2)
    assert len(history) == 2
    for log in history:
        for k in ("jepa", "world_model", "moe_balance", "vicreg", "total"):
            assert k in log
        assert log["total"] == log["total"]  # not NaN
        assert log["vicreg"] > 0  # regression guard for the bug we just fixed
