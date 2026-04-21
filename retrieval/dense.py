"""Dense-vector retrieval over FAISS-like indexes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from embeddings import EmbeddingConfig, EmbeddingGenerator

from .types import RetrievalResult


class DenseRetriever:
    """Embedding + vector-index search adapter."""

    def __init__(
        self,
        index: Any,
        metadata: List[Dict[str, Any]],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.embedder = EmbeddingGenerator(
            EmbeddingConfig(model_name=embedding_model, dimensions=index.d)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if top_k <= 0:
            return []

        query_vec = self.embedder.embed_texts([query]).astype("float32")
        scores, ids = self.index.search(query_vec, top_k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = self.metadata[idx]
            if product_id and product_id not in record.get("product_id", []):
                continue
            if document_type and record.get("document_type") != document_type:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=record["chunk_id"],
                    score=float(score),
                    chunk_text=record["chunk_text"],
                    metadata=record,
                )
            )
            if len(results) >= top_k:
                break
        return results
