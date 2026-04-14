"""Tests for cross-modal attention, hyper-latent projection, and MoE routing."""
from __future__ import annotations

import torch

from hyperlatent.fusion import (
    CrossModalAttention,
    CrossModalMoERouter,
    HyperLatentProjector,
    VICRegLoss,
)


MODS = ("vision", "audio", "text")


def test_cross_modal_attention_shape_and_grad() -> None:
    cma = CrossModalAttention(d_model=64, n_heads=4, modalities=MODS)
    q = torch.randn(2, 1, 64, requires_grad=True)
    ctx = {"audio": torch.randn(2, 4, 64), "text": torch.randn(2, 8, 64)}
    out = cma(q, ctx, query_modality="vision")
    assert out.shape == (2, 1, 64)
    out.sum().backward()
    assert q.grad is not None


def test_hyper_latent_projector_dict() -> None:
    proj = HyperLatentProjector(
        modality_dims={m: 64 for m in MODS}, hidden_dim=128, z_dim=64
    )
    feats = {m: torch.randn(2, 64) for m in MODS}
    out = proj(feats)
    assert set(out.keys()) == set(MODS)
    for m in MODS:
        assert out[m].shape == (2, 64)


def test_vicreg_loss_positive() -> None:
    loss_fn = VICRegLoss()
    lats = {m: torch.randn(4, 64) for m in MODS}
    total, comps = loss_fn(lats)
    assert total.item() > 0
    assert {"vicreg_inv", "vicreg_var", "vicreg_cov"} <= set(comps.keys())


def test_moe_router_shapes_and_balance() -> None:
    router = CrossModalMoERouter(d_model=64, hidden_dim=128, top_k=2)
    x = torch.randn(8, 64)
    y, info = router(x)
    assert y.shape == (8, 64)
    assert "router_probs" in info
    assert info["router_probs"].shape[0] == 8
