"""Cross-encoder reranking for candidate lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .types import RetrievalResult


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_candidates: int = 50


class CrossEncoderReranker:
    """Optional cross-encoder reranker with graceful fallback."""

    def __init__(self, enabled: bool = False, config: RerankerConfig | None = None) -> None:
        self.enabled = enabled
        self.config = config or RerankerConfig()
        self.backend = "disabled"
        self._model = None
        if not self.enabled:
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.config.model_name)
            self.backend = "sentence-transformers-cross-encoder"
        except Exception:
            self._model = None
            self.backend = "unavailable"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if top_k <= 0:
            return []
        if not candidates:
            return []
        if not self.enabled or self._model is None:
            return candidates[:top_k]

        pool = candidates[: self.config.max_candidates]
        pairs = [(query, c.chunk_text) for c in pool]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        ranked = np.argsort(-scores)

        reranked: List[RetrievalResult] = []
        for idx in ranked[:top_k]:
            cand = pool[int(idx)]
            reranked.append(
                RetrievalResult(
                    chunk_id=cand.chunk_id,
                    score=float(scores[int(idx)]),
                    chunk_text=cand.chunk_text,
                    metadata=cand.metadata,
                )
            )
        return reranked
