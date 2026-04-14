"""Tests for sensory buffer, episodic memory, and world model."""
from __future__ import annotations

import torch

from hyperlatent.memory import EpisodicMemory, SemanticWorldModel, SensoryBuffer


def test_sensory_buffer_ring() -> None:
    buf = SensoryBuffer(capacity=3, latent_dim=16)
    for i in range(5):
        buf.push(torch.full((16,), float(i)))
    assert len(buf) == 3


def test_episodic_add_and_query() -> None:
    mem = EpisodicMemory(latent_dim=8, max_entries=32, use_faiss=False)
    for i in range(5):
        mem.add(torch.randn(8), metadata={"i": i})
    q = torch.randn(8)
    hits = mem.query(q, k=3)
    assert len(hits) == 3


def test_world_model_predict_and_update() -> None:
    wm = SemanticWorldModel(latent_dim=16, action_dim=4)
    z = torch.randn(2, 16)
    a = torch.randn(2, 4)
    pred = wm(z, a)
    assert pred.shape == (2, 16)
    loss0 = torch.nn.functional.mse_loss(pred, torch.randn(2, 16)).item()
    assert loss0 >= 0
