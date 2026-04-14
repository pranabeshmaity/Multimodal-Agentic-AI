"""Shape and gradient tests for per-modality encoders."""
from __future__ import annotations

import torch

from hyperlatent.encoders import AudioEncoder, TextEncoder, VisionEncoder


def test_vision_encoder_shape() -> None:
    enc = VisionEncoder(image_size=32, patch_size=8, d_model=64, n_heads=4, depth=2)
    x = torch.randn(2, 3, 32, 32)
    out = enc(x)
    assert out.dim() == 3 and out.shape[0] == 2 and out.shape[-1] == 64


def test_audio_encoder_shape() -> None:
    enc = AudioEncoder(d_model=64, n_heads=4, depth=2, max_frames=64)
    x = torch.randn(2, 1, 2048)
    out = enc(x)
    assert out.dim() == 3 and out.shape[0] == 2 and out.shape[-1] == 64


def test_text_encoder_shape() -> None:
    enc = TextEncoder(vocab_size=128, d_model=64, n_heads=4, depth=2, max_seq_len=16)
    tok = torch.randint(0, 128, (2, 16))
    out = enc(tok)
    assert out.shape == (2, 16, 64)


def test_vision_encoder_backward() -> None:
    enc = VisionEncoder(image_size=32, patch_size=8, d_model=64, n_heads=4, depth=1)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
