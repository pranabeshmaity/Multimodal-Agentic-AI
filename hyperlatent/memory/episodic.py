"""Episodic memory: vector store over consolidated latent observations.

Uses FAISS when available (inner-product over L2-normalized latents, which is
equivalent to cosine similarity). Otherwise falls back to a brute-force
torch-based cosine similarity search. Supports `consolidate()` that yields
batches of `(latents, metadata)` suitable for offline semantic distillation.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False


class EpisodicMemory:
    """Cosine-similarity vector store over latent episodes.

    Each entry is a `(latent, metadata)` pair. The store exposes `add`,
    `query`, and `consolidate` primitives. The consolidation routine yields
    deterministically-sized batches so that a semantic distiller can train
    against the accumulated experience.

    Attributes:
        latent_dim: Dimensionality of latents in the store.
        max_entries: Maximum number of retained entries (FIFO eviction).
        device: Torch device used for the fallback brute-force path.
        use_faiss: Whether FAISS is in use.
    """

    def __init__(
        self,
        latent_dim: int,
        max_entries: int = 100_000,
        device: Optional[torch.device] = None,
        use_faiss: bool = True,
    ) -> None:
        """Initialize the episodic store.

        Args:
            latent_dim: Dimensionality of latents.
            max_entries: Maximum number of retained entries; oldest evicted.
            device: Torch device for the fallback path.
            use_faiss: If True and FAISS is installed, use the FAISS index.
        """
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.latent_dim: int = latent_dim
        self.max_entries: int = max_entries
        self.device: torch.device = device or torch.device("cpu")
        self.use_faiss: bool = bool(use_faiss and _HAS_FAISS)

        self._metadata: List[Dict[str, Any]] = []
        self._consolidation_cursor: int = 0

        if self.use_faiss:
            # Inner product on L2-normalized vectors == cosine similarity.
            self._index = faiss.IndexFlatIP(latent_dim)  # type: ignore[attr-defined]
            self._tensor_store: Optional[torch.Tensor] = None
        else:
            self._index = None
            self._tensor_store = torch.empty(
                (0, latent_dim), dtype=torch.float32, device=self.device
            )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize(z: torch.Tensor) -> torch.Tensor:
        """L2-normalize a tensor along its last dimension with an eps guard."""
        return z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-12))

    def __len__(self) -> int:
        return len(self._metadata)

    # -------------------------------------------------------------------- api
    def add(self, z: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a single latent with metadata.

        Args:
            z: Latent tensor of shape `(latent_dim,)` or `(1, latent_dim)`.
            metadata: Optional metadata associated with this memory.

        Returns:
            The integer id of the inserted entry.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent last-dim {z.shape[-1]} != store dim {self.latent_dim}"
            )

        z_norm = self._normalize(z.detach().to(torch.float32))
        meta = dict(metadata or {})
        meta.setdefault("t", time.time())

        if self.use_faiss:
            self._index.add(z_norm.cpu().numpy())  # type: ignore[union-attr]
        else:
            assert self._tensor_store is not None
            self._tensor_store = torch.cat(
                [self._tensor_store, z_norm.to(self.device)], dim=0
            )

        self._metadata.append(meta)
        new_id = len(self._metadata) - 1

        if len(self._metadata) > self.max_entries:
            self._evict_oldest(len(self._metadata) - self.max_entries)
        return new_id

    def _evict_oldest(self, n: int) -> None:
        """Evict the oldest `n` entries (FIFO)."""
        n = min(n, len(self._metadata))
        if n <= 0:
            return
        self._metadata = self._metadata[n:]
        self._consolidation_cursor = max(0, self._consolidation_cursor - n)
        if self.use_faiss:
            # FAISS IndexFlatIP has no in-place remove; rebuild.
            remaining = self._index.reconstruct_n(n, self._index.ntotal - n)  # type: ignore[union-attr]
            self._index.reset()  # type: ignore[union-attr]
            self._index.add(remaining)  # type: ignore[union-attr]
        else:
            assert self._tensor_store is not None
            self._tensor_store = self._tensor_store[n:]

    def query(
        self, z: torch.Tensor, k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Return the top-`k` most similar entries by cosine similarity.

        Args:
            z: Query latent of shape `(latent_dim,)` or `(1, latent_dim)`.
            k: Number of nearest neighbors to return.

        Returns:
            A list of `(id, similarity, metadata)` tuples, highest first.
        """
        if len(self._metadata) == 0:
            return []
        if z.dim() == 1:
            z = z.unsqueeze(0)
        q = self._normalize(z.detach().to(torch.float32))
        k = min(k, len(self._metadata))

        if self.use_faiss:
            sims, ids = self._index.search(q.cpu().numpy(), k)  # type: ignore[union-attr]
            out: List[Tuple[int, float, Dict[str, Any]]] = []
            for sim, idx in zip(sims[0].tolist(), ids[0].tolist()):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                out.append((int(idx), float(sim), self._metadata[idx]))
            return out

        assert self._tensor_store is not None
        sims = (q.to(self.device) @ self._tensor_store.T).squeeze(0)
        topk = torch.topk(sims, k=k, largest=True)
        return [
            (int(i), float(s), self._metadata[int(i)])
            for s, i in zip(topk.values.tolist(), topk.indices.tolist())
        ]

    def consolidate(
        self, batch_size: int = 64, reset_cursor: bool = False
    ) -> Iterator[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """Iterate over un-consolidated entries in batches.

        The internal cursor advances past every yielded entry so that
        subsequent calls only surface new memories. Set `reset_cursor=True`
        to replay from the beginning.

        Args:
            batch_size: Number of entries per yielded batch.
            reset_cursor: If True, restart from the first entry.

        Yields:
            `(latents, metadatas)` where `latents` is a `(B, latent_dim)`
            tensor of L2-normalized entries.
        """
        if reset_cursor:
            self._consolidation_cursor = 0
        total = len(self._metadata)
        start = self._consolidation_cursor
        if start >= total:
            return
        for lo in range(start, total, batch_size):
            hi = min(lo + batch_size, total)
            tensors = self._reconstruct_range(lo, hi)
            metas = self._metadata[lo:hi]
            self._consolidation_cursor = hi
            yield tensors, metas

    def _reconstruct_range(self, lo: int, hi: int) -> torch.Tensor:
        """Reconstruct stored (normalized) vectors in `[lo, hi)`."""
        if self.use_faiss:
            vecs = self._index.reconstruct_n(lo, hi - lo)  # type: ignore[union-attr]
            return torch.from_numpy(vecs).to(self.device)
        assert self._tensor_store is not None
        return self._tensor_store[lo:hi].clone()

    def stats(self) -> Dict[str, Any]:
        """Return a diagnostic dict describing the store's state."""
        return {
            "size": len(self._metadata),
            "capacity": self.max_entries,
            "backend": "faiss" if self.use_faiss else "torch",
            "consolidation_cursor": self._consolidation_cursor,
            "latent_dim": self.latent_dim,
        }
