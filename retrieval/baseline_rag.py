"""Hybrid baseline RAG with dense+sparse retrieval and optional reranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .sparse import BM25SparseRetriever
from .types import RetrievalResult


class BaselineRAG:
    """Entry point orchestrating hybrid search + optional reranking."""

    def __init__(
        self,
        index: Any,
        metadata: List[Dict[str, Any]],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_sparse: bool = True,
        enable_reranker: bool = True,
    ) -> None:
        self.metadata = metadata
        self.dense_retriever = DenseRetriever(
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
        )
        self.sparse_retriever = BM25SparseRetriever(metadata=metadata) if enable_sparse else None
        self.reranker = CrossEncoderReranker(enabled=enable_reranker)

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: Optional[str] = None,
        document_type: Optional[str] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        if top_k <= 0:
            return []

        dense_results = self.dense_retriever.search(
            query=query,
            top_k=dense_top_k or max(top_k * 5, top_k),
            product_id=product_id,
            document_type=document_type,
        )
        result_lists: List[List[RetrievalResult]] = [dense_results]

        if self.sparse_retriever is not None:
            sparse_results = self.sparse_retriever.search(
                query=query,
                top_k=sparse_top_k or max(top_k * 5, top_k),
                product_id=product_id,
                document_type=document_type,
            )
            result_lists.append(sparse_results)

        fused = reciprocal_rank_fusion(
            result_lists=result_lists,
            rrf_k=rrf_k,
            top_k=max(top_k * 5, top_k),
        )
        if rerank:
            return self.reranker.rerank(query=query, candidates=fused, top_k=top_k)
        return fused[:top_k]

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
