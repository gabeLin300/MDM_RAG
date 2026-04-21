"""Sparse lexical retrieval powered by BM25."""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np

from .types import RetrievalResult

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]


_INNER_PUNCT = {"-", "_", "/", ".", "+", "#"}


def _tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer tuned for technical product text.

    Keeps compound tokens like `12-28vdc`, `rs-485`, `li-ion`, `ip67`, `5ghz`
    while splitting surrounding punctuation and whitespace.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens: List[str] = []
    current: List[str] = []
    n = len(normalized)

    def flush() -> None:
        if not current:
            return
        token = "".join(current).strip("".join(_INNER_PUNCT))
        if token:
            tokens.append(token)
        current.clear()

    for i, ch in enumerate(normalized):
        if ch.isalnum():
            current.append(ch)
            continue

        if ch in _INNER_PUNCT:
            next_ch = normalized[i + 1] if i + 1 < n else ""
            if current and next_ch.isalnum():
                current.append(ch)
                continue

        flush()

    flush()
    return tokens


class BM25SparseRetriever:
    """Sparse retrieval with BM25Okapi and deterministic fallback scoring."""

    def __init__(self, metadata: List[Dict[str, Any]]) -> None:
        self.metadata = metadata
        self.corpus_tokens = [_tokenize(str(row.get("chunk_text", ""))) for row in metadata]
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None

    def _fallback_scores(self, query_tokens: List[str]) -> np.ndarray:
        qset = set(query_tokens)
        scores = np.zeros((len(self.corpus_tokens),), dtype=np.float32)
        for i, dtoks in enumerate(self.corpus_tokens):
            if not dtoks:
                continue
            overlap = sum(1 for tok in dtoks if tok in qset)
            # Light normalization to avoid over-favoring very long chunks.
            scores[i] = overlap / math.sqrt(max(1, len(dtoks)))
        return scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if self._bm25 is not None:
            scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        else:
            scores = self._fallback_scores(query_tokens)

        ranked = np.argsort(-scores)
        results: List[RetrievalResult] = []
        for idx in ranked[: max(top_k * 5, top_k)]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            record = self.metadata[idx]
            if product_id and product_id not in record.get("product_id", []):
                continue
            if document_type and record.get("document_type") != document_type:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=record["chunk_id"],
                    score=score,
                    chunk_text=record["chunk_text"],
                    metadata=record,
                )
            )
            if len(results) >= top_k:
                break
        return results
