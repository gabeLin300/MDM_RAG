from datetime import datetime, timezone
from retrieval.baseline_rag import BaselineRAG
from agents.unified_agent import UnifiedAgent
from schemas.product_schema import AttributeValidator

class Orchestrator:
    def __init__(self, index, metadata):
        self.rag = BaselineRAG(index=index, metadata=metadata)
        self.agent = UnifiedAgent()
        self.validator = AttributeValidator()
        self.max_chunk_chars = 1200
        self.max_total_chars = 6000

    def _select_chunks(self, retrieved_results):
        texts, ids = [], []
        total = 0
        for result in retrieved_results:
            if total >= self.max_total_chars:
                break
            text = result.chunk_text[:self.max_chunk_chars] 
            remaining_chars = self.max_total_chars - total
            if len(text) <= remaining_chars:
                texts.append(text)
                ids.append(result.chunk_id)
                total += len(text)
            elif remaining_chars > 100: 
                texts.append(result.chunk_text[:remaining_chars])
                ids.append(result.chunk_id)
                break
            else:
                break
        return texts, ids
    
    def run_for_document(self, doc_chunks: list[dict]) -> dict:
        """Extract attributes from a document's chunks directly — no RAG search needed."""
        sorted_chunks = sorted(doc_chunks, key=lambda c: c.get("char_start", 0))

        texts, chunk_ids = [], []
        total = 0
        for chunk in sorted_chunks:
            text = chunk["chunk_text"][:self.max_chunk_chars]
            if total + len(text) > self.max_total_chars:
                break
            texts.append(text)
            chunk_ids.append(chunk["chunk_id"])
            total += len(text)

        if not texts:
            return {
                "source_chunk_ids": [],
                "attributes": {},
                "quality_flags": ["no_chunks"],
                "review_required": True,
            }

        raw = self.agent.extract(texts)
        flags, review_required = self.validator.validate(raw["attributes"])
        return {
            "source_chunk_ids": chunk_ids,
            "attributes": raw["attributes"],
            "quality_flags": flags,
            "review_required": review_required,
        }

    def run_for_product_batch(
        self,
        product_contexts: dict[str, str],
        source_chunk_ids_by_product: dict[str, list[str]],
    ) -> dict[str, dict]:
        raw = self.agent.extract_batch(product_contexts)
        output: dict[str, dict] = {}
        for product_id, result in raw.items():
            attrs = result.get("attributes", {})
            flags, review_required = self.validator.validate(attrs)
            output[product_id] = {
                "product_id": product_id,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "source_chunk_ids": source_chunk_ids_by_product.get(product_id, []),
                "attributes": attrs,
                "quality_flags": flags,
                "review_required": review_required,
            }
        return output

    def run(self, product_id: str) -> dict:
        retrieved = self.rag.search(
            "product technical specifications",
            product_id=product_id,
            top_k=3,
        )

        if not retrieved:
            return {
                "product_id": product_id,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "source_chunk_ids": [],
                "attributes": {},
                "quality_flags": ["no_chunks_retrieved"],
                "review_required": True,
            }

        texts, chunk_ids = self._select_chunks(retrieved)
        raw = self.agent.extract(texts)
        flags, review_required = self.validator.validate(raw["attributes"])

        return {
            "product_id": product_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "source_chunk_ids": chunk_ids,
            "attributes": raw["attributes"],
            "quality_flags": flags,
            "review_required": review_required,
        }
