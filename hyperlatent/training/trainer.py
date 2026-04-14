"""End-to-end training loop for the Hyper-Latent Fusion system."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import nn

from .config import TrainingConfig
from .ema import EMATargetEncoder
from .objectives import (
    JEPAPredictiveLoss,
    MoEBalanceLoss,
    ObjectiveWeights,
    TotalObjective,
    WorldModelLoss,
    jepa_pairs,
)
from .replay import TransitionReplayBuffer


class HyperLatentTrainer:
    """Phase-3 trainer tying together encoders, fusion, memory, and objectives.

    The trainer owns:
        * Per-modality online encoders and their EMA target copies.
        * A cross-modal attention block and hyper-latent projector.
        * A cross-modal MoE router.
        * A semantic world model predicting next-latents.
        * All loss modules, the optimizer, AMP scaler, and replay buffer.

    Expected ``batch`` format passed to :meth:`step`:
        ``{
            "modalities": List[Tensor],       # one tensor per modality
            "next_modalities": List[Tensor],  # observations at t+1
            "actions": Tensor,                # (B, action_dim)
            "rewards": Tensor,                # (B,)
        }``

    Attributes:
        cfg: :class:`TrainingConfig` instance.
        encoders: ``nn.ModuleList`` of online per-modality encoders.
        target_encoders: List of :class:`EMATargetEncoder` wrappers.
        fusion: Cross-modal attention block.
        projector: Hyper-latent projector.
        router: MoE router.
        world_model: Latent world model.
        memory: Episodic memory bank.
        vicreg: VICReg regularizer module.
        jepa_losses: Per-modality JEPA predictor losses.
        world_loss: :class:`WorldModelLoss`.
        moe_loss: :class:`MoEBalanceLoss`.
        total_objective: Aggregator.
        replay: :class:`TransitionReplayBuffer`.
        optimizer: AdamW optimizer.
        scaler: Optional :class:`torch.cuda.amp.GradScaler`.
        global_step: Monotonic step counter.
    """

    MODALITY_NAMES: Tuple[str, ...] = ("vision", "audio", "text")

    def __init__(self, cfg: TrainingConfig) -> None:
        """Construct all submodules and optimizer.

        Args:
            cfg: Fully-populated :class:`TrainingConfig`.
        """
        from hyperlatent.encoders import AudioEncoder, TextEncoder, VisionEncoder
        from hyperlatent.fusion import (
            CrossModalAttention,
            CrossModalMoERouter,
            HyperLatentProjector,
            VICRegLoss,
        )
        from hyperlatent.memory import EpisodicMemory, SemanticWorldModel

        torch.manual_seed(cfg.seed)
        self.cfg: TrainingConfig = cfg
        self.device: torch.device = torch.device(cfg.device)

        D = cfg.latent_dim
        n_heads = cfg.extras.get("n_heads", 4)
        if D % n_heads != 0:
            raise ValueError(f"latent_dim {D} must be divisible by n_heads {n_heads}")

        self.modality_names: Tuple[str, ...] = self.MODALITY_NAMES[: cfg.num_modalities]

        # Build tiny domain-appropriate encoders.
        vision = VisionEncoder(
            image_size=cfg.extras.get("image_size", 32),
            patch_size=cfg.extras.get("patch_size", 8),
            in_channels=3,
            d_model=D,
            n_heads=n_heads,
            depth=cfg.extras.get("encoder_depth", 2),
        )
        audio = AudioEncoder(
            in_channels=1,
            d_model=D,
            n_heads=n_heads,
            depth=cfg.extras.get("encoder_depth", 2),
            max_frames=cfg.extras.get("max_audio_frames", 256),
        )
        text = TextEncoder(
            vocab_size=cfg.extras.get("vocab_size", 1000),
            d_model=D,
            n_heads=n_heads,
            depth=cfg.extras.get("encoder_depth", 2),
            max_seq_len=cfg.extras.get("max_text_len", 16),
        )
        enc_pool = [vision, audio, text][: cfg.num_modalities]
        self.encoders: nn.ModuleList = nn.ModuleList(enc_pool).to(self.device)
        self.target_encoders: List[EMATargetEncoder] = [
            EMATargetEncoder(e).to(self.device) for e in self.encoders
        ]

        self.fusion: nn.Module = CrossModalAttention(
            d_model=D, n_heads=n_heads, modalities=self.modality_names
        ).to(self.device)
        self.projector: nn.Module = HyperLatentProjector(
            modality_dims={n: D for n in self.modality_names},
            hidden_dim=D * 2,
            z_dim=D,
        ).to(self.device)
        self.router: nn.Module = CrossModalMoERouter(
            d_model=D,
            hidden_dim=D * 2,
            top_k=min(2, cfg.num_experts),
        ).to(self.device)
        self.world_model: nn.Module = SemanticWorldModel(
            latent_dim=D, action_dim=cfg.action_dim
        ).to(self.device)
        self.memory = EpisodicMemory(latent_dim=D, max_entries=cfg.replay_capacity, use_faiss=False)
        self.vicreg: nn.Module = VICRegLoss().to(self.device)

        self.jepa_losses: nn.ModuleList = nn.ModuleList(
            [
                JEPAPredictiveLoss(dim=cfg.latent_dim, hidden=cfg.predictor_hidden)
                for _ in range(cfg.num_modalities)
            ]
        ).to(self.device)
        self.world_loss: WorldModelLoss = WorldModelLoss().to(self.device)
        self.moe_loss: MoEBalanceLoss = MoEBalanceLoss().to(self.device)
        self.total_objective: TotalObjective = TotalObjective(
            ObjectiveWeights(
                jepa=cfg.jepa_coef,
                world_model=cfg.world_model_coef,
                moe_balance=cfg.moe_balance_coef,
                vicreg=cfg.vicreg_coef,
                contrastive=cfg.contrastive_coef,
            )
        ).to(self.device)

        self.replay: TransitionReplayBuffer = TransitionReplayBuffer(cfg.replay_capacity)

        params = list(self._trainable_parameters())
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
        )
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if cfg.amp and torch.cuda.is_available() else None
        )
        self.global_step: int = 0
        self._accum_counter: int = 0

    def _trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters across owned submodules."""
        modules: List[nn.Module] = [
            self.encoders,
            self.fusion,
            self.projector,
            self.router,
            self.world_model,
            self.jepa_losses,
        ]
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    # --------------------------------------------------------------- forward
    def _encode_all(
        self, inputs: List[torch.Tensor], use_target: bool = False
    ) -> List[torch.Tensor]:
        """Run per-modality encoders, returning pooled ``(B, D)`` tensors.

        Args:
            inputs: Per-modality domain tensors (images, waveforms, tokens).
            use_target: If True, route through EMA target encoders instead.

        Returns:
            List of pooled latent tensors shaped ``(B, D)``.
        """
        latents: List[torch.Tensor] = []
        for i, x in enumerate(inputs):
            enc = self.target_encoders[i] if use_target else self.encoders[i]
            z = enc(x.to(self.device))
            if isinstance(z, (tuple, list)):
                z = z[0]
            if z.dim() == 3:  # (B, T, D) -> mean pool
                z = z.mean(dim=1)
            latents.append(z)
        return latents

    def _fuse(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """Project + cross-modal attention fuse into a single hyper-latent.

        Each modality queries the others via :class:`CrossModalAttention`,
        outputs are averaged across modalities.

        Args:
            latents: Per-modality pooled latents ``(B, D)``.

        Returns:
            Fused hyper-latent of shape ``(B, D)``.
        """
        feat_dict: Dict[str, torch.Tensor] = {
            name: latents[i] for i, name in enumerate(self.modality_names)
        }
        projected: Dict[str, torch.Tensor] = self.projector(feat_dict)

        attended: List[torch.Tensor] = []
        for name in self.modality_names:
            q = projected[name].unsqueeze(1)  # (B, 1, D)
            contexts = {
                other: projected[other].unsqueeze(1)
                for other in self.modality_names
                if other != name
            }
            out = self.fusion(q, contexts, query_modality=name)  # (B, 1, D)
            attended.append(out.squeeze(1))
        return torch.stack(attended, dim=1).mean(dim=1)  # (B, D)

    def _route(self, hyper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the MoE router.

        Args:
            hyper: Hyper-latent of shape ``(B, D)``.

        Returns:
            ``(routed_output, router_logits)`` where logits are ``(B, E)``.
        """
        routed, info = self.router(hyper)
        logits = info.get("router_logits")
        if logits is None:
            logits = info.get("router_probs")
        if logits is None:
            logits = hyper.new_zeros(hyper.shape[0], self.cfg.num_experts)
        return routed, logits

    def _world_forward(
        self, hyper: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next hyper-latent conditioned on actions.

        Args:
            hyper: Current hyper-latent ``(B, D)``.
            actions: Action batch ``(B, A)``.

        Returns:
            Predicted next hyper-latent ``(B, D)``.
        """
        return self.world_model(hyper, actions.to(self.device))

    # ----------------------------------------------------------------- step
    def step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        """Run one training step on a single batch.

        Args:
            batch: Dictionary with ``modalities``, ``next_modalities``,
                ``actions``, and ``rewards`` keys.

        Returns:
            Dictionary of scalar losses for logging, including ``"total"``.
        """
        cfg = self.cfg
        mods: List[torch.Tensor] = list(batch["modalities"])
        next_mods: List[torch.Tensor] = list(batch["next_modalities"])
        actions: torch.Tensor = batch["actions"].to(self.device)
        rewards: torch.Tensor = batch["rewards"].to(self.device)

        autocast_enabled = cfg.amp and torch.cuda.is_available()
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=autocast_enabled)
            if autocast_enabled
            else _NullCtx()
        )

        with amp_ctx:
            context_latents = self._encode_all(mods, use_target=False)
            with torch.no_grad():
                target_latents = self._encode_all(mods, use_target=True)

            # JEPA across modality pairs: predict target_j from context_i.
            jepa_terms: List[torch.Tensor] = []
            for i, j in jepa_pairs(cfg.num_modalities):
                loss_ij = self.jepa_losses[i](context_latents[i], target_latents[j])
                jepa_terms.append(loss_ij)
            jepa_loss = (
                torch.stack(jepa_terms).mean()
                if jepa_terms
                else context_latents[0].new_zeros(())
            )

            hyper = self._fuse(context_latents)
            routed, router_logits = self._route(hyper)
            moe_balance = self.moe_loss(router_logits)

            # VICReg across all per-modality context latents.
            vicreg_inputs = {
                name: context_latents[i] for i, name in enumerate(self.modality_names)
            }
            vicreg_val, _ = self.vicreg(vicreg_inputs)

            # World model: predict hyper_{t+1} from hyper_t, action_t.
            with torch.no_grad():
                next_latents = self._encode_all(next_mods, use_target=True)
                next_hyper = self._fuse(next_latents)
            predicted_next = self._world_forward(routed, actions)
            negatives: Optional[torch.Tensor] = None
            if len(self.replay) >= cfg.num_negatives:
                negatives = self.replay.sample_negatives(cfg.num_negatives).to(self.device)
            world_total, _, contrastive = self.world_loss(
                predicted_next, next_hyper, negatives
            )

            components: Dict[str, torch.Tensor] = {
                "jepa": jepa_loss,
                "world_model": world_total,
                "moe_balance": moe_balance,
                "vicreg": vicreg_val if isinstance(vicreg_val, torch.Tensor) else hyper.new_tensor(float(vicreg_val)),
                "contrastive": contrastive,
            }
            total, log = self.total_objective(components)
            total = total / max(cfg.accumulation_steps, 1)

        if self.scaler is not None:
            self.scaler.scale(total).backward()
        else:
            total.backward()
        self._accum_counter += 1

        if self._accum_counter >= cfg.accumulation_steps:
            if cfg.grad_clip and cfg.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self._trainable_parameters()), cfg.grad_clip
                )
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0

            # EMA update for target encoders.
            for enc, tgt in zip(self.encoders, self.target_encoders):
                tgt.update(enc, cfg.ema_tau)

        # Push transitions into replay for use in subsequent steps.
        self.replay.add_batch(
            hyper.detach(), actions.detach(), next_hyper.detach(), rewards.detach()
        )

        # Consolidate into episodic memory (one entry per batch item).
        for i in range(hyper.shape[0]):
            self.memory.add(hyper[i].detach(), metadata={"step": self.global_step})

        self.global_step += 1
        if cfg.checkpoint_every and self.global_step % cfg.checkpoint_every == 0:
            self.save_checkpoint(os.path.join(cfg.ckpt_dir, f"step_{self.global_step}.pt"))
        return log

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        num_steps: int,
        dataloader: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> List[Dict[str, float]]:
        """Run ``num_steps`` training iterations.

        Args:
            num_steps: Total steps to execute.
            dataloader: Iterable yielding batch dicts. If None, uses
                :meth:`mock_batches` with random tensors.

        Returns:
            A list of per-step log dictionaries.
        """
        it: Iterator[Mapping[str, Any]] = (
            iter(dataloader) if dataloader is not None else self.mock_batches()
        )
        history: List[Dict[str, float]] = []
        for step in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = (
                    iter(dataloader) if dataloader is not None else self.mock_batches()
                )
                batch = next(it)
            log = self.step(batch)
            history.append(log)
            if self.cfg.log_every and (step % self.cfg.log_every == 0):
                print(f"[step {self.global_step}] {log}")
        return history

    def mock_batches(self) -> Iterator[Dict[str, Any]]:
        """Infinite generator of random-tensor batches for smoke testing.

        Yields:
            Dictionaries matching the :meth:`step` contract.
        """
        cfg = self.cfg
        B = cfg.batch_size
        img_size = cfg.extras.get("image_size", 32)
        aud_len = cfg.extras.get("audio_len", 2048)
        txt_len = cfg.extras.get("max_text_len", 16)
        vocab = cfg.extras.get("vocab_size", 1000)

        def _sample_mods() -> List[torch.Tensor]:
            samples: List[torch.Tensor] = []
            if cfg.num_modalities >= 1:
                samples.append(torch.randn(B, 3, img_size, img_size))
            if cfg.num_modalities >= 2:
                samples.append(torch.randn(B, 1, aud_len))
            if cfg.num_modalities >= 3:
                samples.append(torch.randint(0, vocab, (B, txt_len)))
            return samples

        while True:
            mods = _sample_mods()
            next_mods = _sample_mods()
            yield {
                "modalities": mods,
                "next_modalities": next_mods,
                "actions": torch.randn(cfg.batch_size, cfg.action_dim),
                "rewards": torch.randn(cfg.batch_size),
            }

    # --------------------------------------------------------- checkpointing
    def save_checkpoint(self, path: str) -> None:
        """Serialize the trainer state to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "global_step": self.global_step,
            "encoders": self.encoders.state_dict(),
            "target_encoders": [t.state_dict() for t in self.target_encoders],
            "fusion": self.fusion.state_dict(),
            "projector": self.projector.state_dict(),
            "router": self.router.state_dict(),
            "world_model": self.world_model.state_dict(),
            "jepa_losses": self.jepa_losses.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore trainer state from ``path``."""
        state = torch.load(path, map_location=self.device)
        self.global_step = int(state.get("global_step", 0))
        self.encoders.load_state_dict(state["encoders"])
        for t, sd in zip(self.target_encoders, state["target_encoders"]):
            t.load_state_dict(sd)
        self.fusion.load_state_dict(state["fusion"])
        self.projector.load_state_dict(state["projector"])
        self.router.load_state_dict(state["router"])
        self.world_model.load_state_dict(state["world_model"])
        self.jepa_losses.load_state_dict(state["jepa_losses"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])


class _NullCtx:
    """A minimal no-op context manager used when AMP is disabled."""

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
