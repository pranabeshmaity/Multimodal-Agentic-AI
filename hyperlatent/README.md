# Hyper-Latent Fusion — Phase 1

Phase 1 delivers the modality-specific encoders and the core fusion
primitives that underpin the Hyper-Latent Fusion architecture.

## Modules

### `encoders/`
- **`VisionEncoder`** — ViT-style patch-embedding + transformer stack
  (`d_model=512`, `depth=6`). Layer 1 accepts a `cross_modal_context`.
- **`AudioEncoder`** — Wave2Vec-style 1D conv feature extractor feeding a
  transformer stack. Same cross-modal hook at layer 1.
- **`TextEncoder`** — Token + learned positional embeddings feeding a
  transformer stack with the same cross-modal hook.

### `fusion/`
- **`CrossModalAttention`** *(novel)* — queries from one modality,
  keys/values from the concatenated others, learnable
  modality-type embeddings, optional RoPE, and a **sigmoid-gated
  residual** so cross-modal signal is injected adaptively.
- **`HyperLatentProjector` + `VICRegLoss`** — modality-specific
  near-isometric MLPs `phi_m` projecting into a shared space `Z`
  together with the VICReg invariance / variance / covariance loss.
- **`CrossModalMoERouter`** — noisy top-k gating over four experts
  (`v2t`, `t2a`, `a2v`, `tri`) with Switch-Transformer load-balancing
  auxiliary loss.
- **`LatentODE`** — lightweight fixed-step forward-Euler integrator of
  `dz/dt = f_psi(z, c)`; no `torchdiffeq` dependency.

### `utils/`
- **`ModalityBatch`**, **`FusedLatent`** — dataclasses used throughout
  the pipeline.

## Minimal usage

```python
import torch
from hyperlatent.encoders import VisionEncoder, AudioEncoder, TextEncoder
from hyperlatent.fusion import (
    CrossModalAttention, HyperLatentProjector, VICRegLoss,
    CrossModalMoERouter, LatentODE,
)

d_model = 512
vision = VisionEncoder(d_model=d_model)
audio = AudioEncoder(d_model=d_model)
text = TextEncoder(d_model=d_model)
xattn = CrossModalAttention(d_model=d_model, modalities=("vision", "audio", "text"))
projector = HyperLatentProjector(
    modality_dims={"vision": d_model, "audio": d_model, "text": d_model},
    z_dim=d_model,
)
vicreg = VICRegLoss()
router = CrossModalMoERouter(d_model=d_model)
ode = LatentODE(z_dim=d_model)

imgs = torch.randn(2, 3, 224, 224)
wav = torch.randn(2, 1, 16000)
tok = torch.randint(0, 32000, (2, 32))

# 1) encode without cross-modal context (first pass).
v = vision(imgs)
a = audio(wav)
t = text(tok)

# 2) re-encode vision with cross-modal context from audio + text.
ctx = torch.cat([a, t], dim=1)
v = vision(imgs, cross_modal_context=ctx)

# 3) alternative: explicit cross-modal attention layer.
v = xattn(query=v, contexts={"vision": v, "audio": a, "text": t},
          query_modality="vision")

# 4) project, regularise, route, evolve.
z = projector({"vision": v, "audio": a, "text": t})
vic_loss, _ = vicreg(z)
z_fused = sum(z.values()) / len(z)
z_routed, info = router(z_fused)
z_T, _ = ode(z_routed)
```

## Defaults

Modest defaults (`d_model=512`, `n_heads=8`, `depth=6`) so the stack fits
comfortably on a single mid-range GPU.
