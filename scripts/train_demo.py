"""Smoke-test demo for the Hyper-Latent Fusion trainer.

Runs 5 steps on random tensors with tiny dimensions and prints the loss dict
for each step. Intended to verify that the training loop wires up correctly
without any real datasets or GPU.

Usage:
    python scripts/train_demo.py
"""
from __future__ import annotations
import os
import sys
import torch
import matplotlib.pyplot as plt

# Ensure the repo root is on sys.path when invoked directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # faiss is an optional dependency for downstream retrieval modules.
    import faiss  # type: ignore  # noqa: F401
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from hyperlatent.training import HyperLatentTrainer, TrainingConfig

def main() -> None:
    """Construct a tiny trainer and run 5 steps of random-tensor training."""
    cfg = TrainingConfig(
        lr=3e-4,
        ema_tau=0.99,
        accumulation_steps=1,
        grad_clip=1.0,
        amp=False,
        num_modalities=3,
        latent_dim=64,
        predictor_hidden=128,
        action_dim=8,
        num_experts=4,
        batch_size=4,
        log_every=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )
    print(f"[demo] faiss available: {_HAS_FAISS}")
    print(f"[demo] device: {cfg.device}")
    trainer = HyperLatentTrainer(cfg)
    history = trainer.fit(num_steps=100)
    print("[demo] training finished; final log:")
    print(history[-1])
    
    steps = list(range(len(history)))
    jepa = [h["jepa"] for h in history]
    vicreg = [h["vicreg"] for h in history]
    total = [h["total"] for h in history]

    plt.plot(steps, jepa, label = "JEPA")
    plt.plot(steps, vicreg, label = "VICReg")
    plt.plot(steps, total, label = "Total")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curves")
    plt.savefig("loss_plot.png")
    plt.show()    

if __name__ == "__main__":
    main()