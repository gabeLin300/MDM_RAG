"""Deterministic local baseline RAG with extractive answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from embeddings import EmbeddingConfig, EmbeddingGenerator


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    chunk_text: str
    metadata: Dict[str, Any]


class BaselineRAG:
    """Simple retrieval + extractive composition chain."""

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
        query_vec = self.embedder.embed_texts([query]).astype("float32")
        scores, ids = self.index.search(query_vec, max(top_k * 5, top_k))

        filtered: List[RetrievalResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = self.metadata[idx]
            if product_id and product_id not in record.get("product_id", []):
                continue
            if document_type and record.get("document_type") != document_type:
                continue
            filtered.append(
                RetrievalResult(
                    chunk_id=record["chunk_id"],
                    score=float(score),
                    chunk_text=record["chunk_text"],
                    metadata=record,
                )
            )
            if len(filtered) >= top_k:
                break
        return filtered

    def answer(
        self,
        query: str,
        top_k: int = 5,
        product_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        hits = self.search(
            query=query,
            top_k=top_k,
            product_id=product_id,
            document_type=document_type,
        )
        if not hits:
            return {
                "answer": "No relevant chunks found for the query.",
                "citations": [],
                "scores": [],
            }

        snippets = [h.chunk_text[:280].strip() for h in hits]
        answer_text = " ".join(snippets)
        return {
            "answer": answer_text,
            "citations": [h.chunk_id for h in hits],
            "scores": [round(h.score, 5) for h in hits],
        }
