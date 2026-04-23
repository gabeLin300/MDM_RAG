"""Hybrid baseline RAG with dense+sparse retrieval, metadata filtering, and query decomposition."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .sparse import BM25SparseRetriever
from .types import RetrievalResult



def _decompose_query(query: str, enable: bool = False) -> List[str]:
    """
    Decompose a multi-part query into smaller focused sub-queries.
    
    Detects common multi-part patterns like:
    - "Find X, Y, and Z for product P"
    - "Extract X and Y specs for model M"
    - "Get X, Y, Z"
    
    Returns the original query as a single-item list if decomposition is not
    beneficial or if enable=False.
    
    Args:
        query: The input query string.
        enable: Whether to enable query decomposition. If False, returns [query].
    
    Returns:
        List of sub-queries (or original query as single-item list).
    """
    if not enable:
        return [query]
    
    # Simple heuristic: if query contains multiple comma-separated items or
    # multiple "and" conjunctions, split and re-contextualize.
    query = query.strip()
    
    # Pattern: Look for comma-separated items, especially technical attributes
    comma_pattern = r'(\b(?:voltage|current|temperature|pressure|dimension|size|weight|frequency|power|impedance|resistance|capacitance|inductance|material|surface|finish|color|certification|standard|mounting|interface|protocol|connector|pin|thread|speed|torque|rating|specification|spec)s?\b[^,]*)'
    
    # If query has multiple commas or "and" keywords, consider decomposing
    comma_count = query.count(",")
    and_count = len(re.findall(r"\band\b", query, re.IGNORECASE))
    
    # Heuristic: decompose if we detect 2+ comma-separated segments or multiple "and"
    if comma_count >= 1 and and_count >= 0:
        # Try to split by commas first
        parts = re.split(r",\s*", query)
        if len(parts) > 1:
            # Extract entity references (e.g., product names, model numbers)
            entity_match = re.search(r"\b(?:product|model|device|unit|system)[\s:]*([A-Z0-9\-_]+)", query, re.IGNORECASE)
            entity_ref = entity_match.group(1) if entity_match else ""
            
            # Clean parts and re-add entity reference
            sub_queries = []
            for part in parts:
                part = part.strip()
                if part:
                    # Add entity reference to each sub-query if present
                    if entity_ref and entity_ref not in part:
                        part = f"{part} for {entity_ref}"
                    sub_queries.append(part)
            
            # Return decomposed queries if we have multiple valid sub-queries
            if len(sub_queries) > 1:
                return sub_queries
    
    # No beneficial decomposition detected
    return [query]


def _apply_metadata_filters(
    results: List[RetrievalResult],
    filters: Dict[str, Any],
) -> List[RetrievalResult]:
    """
    Apply metadata filters to a list of retrieval results.
    
    Supports flexible filtering by any metadata field. Examples:
    - filters = {"product_id": "MODEL_X"}
    - filters = {"document_type": "specification"}
    - filters = {"category": "electrical", "brand": "ACME"}
    
    Only results matching ALL filter conditions are retained.
    
    Args:
        results: List of RetrievalResult objects to filter.
        filters: Dict mapping metadata field names to values to match.
                 Supports string, int, list, and substring matching.
    
    Returns:
        Filtered list of results matching all filter conditions.
    """
    if not filters:
        return results
    
    filtered: List[RetrievalResult] = []
    
    for result in results:
        # Check if result matches all filter conditions
        match = True
        for field, filter_value in filters.items():
            metadata_value = result.metadata.get(field)
            
            # Handle list-type metadata fields
            if isinstance(metadata_value, list):
                if isinstance(filter_value, list):
                    # Both are lists: check for any intersection
                    if not any(v in metadata_value for v in filter_value):
                        match = False
                        break
                else:
                    # Metadata is list, filter is single value
                    if filter_value not in metadata_value:
                        match = False
                        break
            else:
                # Direct comparison (exact match or substring for strings)
                if isinstance(metadata_value, str) and isinstance(filter_value, str):
                    if filter_value.lower() not in metadata_value.lower():
                        match = False
                        break
                else:
                    if metadata_value != filter_value:
                        match = False
                        break
        
        if match:
            filtered.append(result)
    
    return filtered


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
        filters: Optional[Dict[str, Any]] = None,
        enable_decomposition: bool = False,
    ) -> List[RetrievalResult]:
        """
        Orchestrate hybrid search with optional query decomposition and metadata filtering.
        
        Args:
            query: User query string.
            top_k: Number of final results to return.
            product_id: (Legacy) Filter by product_id.
            document_type: (Legacy) Filter by document_type.
            dense_top_k: Optional k for dense retrieval.
            sparse_top_k: Optional k for sparse retrieval.
            rrf_k: RRF parameter for fusion.
            rerank: Whether to apply cross-encoder reranking.
            filters: Optional dict of metadata filters. Examples:
                     {"product_id": "X", "category": "electrical"}
            enable_decomposition: If True, decompose multi-part queries into
                                  sub-queries and merge results.
        
        Returns:
            List of RetrievalResult objects, deduplicated and optionally reranked.
        """
        if top_k <= 0:
            return []
        
        # Build composite filter dict from legacy parameters + new filters
        composite_filters: Dict[str, Any] = {}
        if product_id:
            composite_filters["product_id"] = product_id
        if document_type:
            composite_filters["document_type"] = document_type
        if filters:
            composite_filters.update(filters)
        
        # Decompose query if requested
        sub_queries = _decompose_query(query, enable=enable_decomposition)
        
        # Retrieve and merge results from all sub-queries
        all_results: Dict[str, RetrievalResult] = {}  # chunk_id -> RetrievalResult
        all_scores: Dict[str, float] = {}  # chunk_id -> max score seen
        
        for sub_query in sub_queries:
            # Dense retrieval
            dense_results = self.dense_retriever.search(
                query=sub_query,
                top_k=dense_top_k or max(top_k * 5, top_k),
                product_id=product_id,
                document_type=document_type,
            )
            result_lists: List[List[RetrievalResult]] = [dense_results]
            
            # Sparse retrieval
            if self.sparse_retriever is not None:
                sparse_results = self.sparse_retriever.search(
                    query=sub_query,
                    top_k=sparse_top_k or max(top_k * 5, top_k),
                    product_id=product_id,
                    document_type=document_type,
                )
                result_lists.append(sparse_results)
            
            # Fuse results from this sub-query
            fused = reciprocal_rank_fusion(
                result_lists=result_lists,
                rrf_k=rrf_k,
                top_k=max(top_k * 5, top_k),
            )
            
            # Merge into global results, keeping highest score per chunk
            for result in fused:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                    all_scores[result.chunk_id] = result.score
                else:
                    # Keep the higher score
                    if result.score > all_scores[result.chunk_id]:
                        all_results[result.chunk_id] = result
                        all_scores[result.chunk_id] = result.score
        
        # Convert back to list, sorted by score
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda r: all_scores[r.chunk_id], reverse=True)
        
        # Apply metadata filters if any
        if composite_filters:
            merged_results = _apply_metadata_filters(merged_results, composite_filters)
        
        # Rerank if requested
        if rerank:
            return self.reranker.rerank(query=query, candidates=merged_results, top_k=top_k)
        
        return merged_results[:top_k]

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
