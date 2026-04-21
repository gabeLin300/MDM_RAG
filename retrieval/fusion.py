"""Hybrid rank-fusion helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .types import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: Iterable[List[RetrievalResult]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """Combine ranked lists with Reciprocal Rank Fusion (RRF)."""
    combined_scores: Dict[str, float] = {}
    by_chunk: Dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            combined_scores[item.chunk_id] = combined_scores.get(item.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            if item.chunk_id not in by_chunk:
                by_chunk[item.chunk_id] = item

    fused = [
        RetrievalResult(
            chunk_id=chunk_id,
            score=float(score),
            chunk_text=by_chunk[chunk_id].chunk_text,
            metadata=by_chunk[chunk_id].metadata,
        )
        for chunk_id, score in combined_scores.items()
    ]
    fused.sort(key=lambda x: x.score, reverse=True)
    if top_k is not None:
        return fused[:top_k]
    return fused
