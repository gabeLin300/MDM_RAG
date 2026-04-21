from retrieval.baseline_rag import BaselineRAG
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import _tokenize
from retrieval.types import RetrievalResult


def _hit(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        chunk_text=f"text-{chunk_id}",
        metadata={"chunk_id": chunk_id, "chunk_text": f"text-{chunk_id}"},
    )


def test_rrf_fusion_combines_dense_and_sparse():
    dense = [_hit("a", 0.9), _hit("b", 0.8)]
    sparse = [_hit("b", 10.0), _hit("c", 8.0)]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    assert [x.chunk_id for x in fused[:3]] == ["b", "a", "c"]


class _StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, **kwargs):
        top_k = kwargs.get("top_k", len(self.hits))
        return self.hits[:top_k]


class _StubReranker:
    def rerank(self, query, candidates, top_k):
        # Deterministic test behavior: reverse top candidates.
        return list(reversed(candidates[:top_k]))


def test_baseline_search_is_entrypoint_orchestration():
    rag = BaselineRAG.__new__(BaselineRAG)
    rag.metadata = []
    rag.dense_retriever = _StubRetriever([_hit("a", 0.9), _hit("b", 0.8)])
    rag.sparse_retriever = _StubRetriever([_hit("b", 10.0), _hit("c", 8.0)])
    rag.reranker = _StubReranker()

    fused_only = rag.search("query", top_k=2, rerank=False)
    assert [x.chunk_id for x in fused_only] == ["b", "a"]

    reranked = rag.search("query", top_k=2, rerank=True)
    assert [x.chunk_id for x in reranked] == ["a", "b"]


def test_sparse_tokenizer_preserves_technical_terms():
    tokens = _tokenize("Input power range 12-28 VDC, RS-485, Li-Ion, IP67, 5GHz.")
    assert "12-28" in tokens
    assert "vdc" in tokens
    assert "rs-485" in tokens
    assert "li-ion" in tokens
    assert "ip67" in tokens
    assert "5ghz" in tokens
