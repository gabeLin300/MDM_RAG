"""Batch embedding generation for chunked text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 64
    normalize: bool = True


class EmbeddingGenerator:
    """Generate embeddings from local sentence-transformers model."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.backend = "hash-fallback"
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.config.model_name)
            self.backend = "sentence-transformers"
        except Exception:
            self._model = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts and return shape [n, dim] float32 matrix."""
        values = [str(t).strip() for t in texts]
        if self._model is not None:
            vectors = self._model.encode(
                values,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.astype("float32")
        return self._hash_embed(values)

    def embed_in_batches(self, texts: Sequence[str]) -> Iterable[Tuple[int, np.ndarray]]:
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            yield start, self.embed_texts(batch)

    def _hash_embed(self, texts: Sequence[str]) -> np.ndarray:
        dim = self.config.dimensions
        output = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], byteorder="little") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                output[i, idx] += sign
        if self.config.normalize:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            output = output / norms
        return output
