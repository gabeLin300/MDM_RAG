from retrieval.baseline_rag import BaselineRAG
from agents.unified_agent import UnifiedAgent

class Orchestrator:
    def __init__(self, index, metadata):
        self.unified_agent = UnifiedAgent()
        self.rag = BaselineRAG(index=index, metadata=metadata)
        # Max context length per chunk (tokens ~= chars/4)
        self.max_chunk_length = 800

    def _limit_context_length(self, chunks: list[str], max_total_chars: int = 4000) -> list[str]:
        """Limit total context by truncating chunks if needed."""
        result = []
        total_chars = 0
        for chunk in chunks:
            if total_chars >= max_total_chars:
                break
            # Limit individual chunk to prevent excessive context
            truncated = chunk[:self.max_chunk_length]
            if total_chars + len(truncated) <= max_total_chars:
                result.append(truncated)
                total_chars += len(truncated)
            else:
                # Add partial chunk to fill up to limit
                remaining = max_total_chars - total_chars
                if remaining > 100:  # Only add if meaningful content
                    result.append(chunk[:remaining])
                break
        return result

    def serialize_output(self, product_id, attributes):
        """Convert flat attributes dict to wrapped format.

        Transform {"voltage": "24V", "current": "0.5A"}
        into {"product_id": [{"voltage": "24V"}, {"current": "0.5A"}]}
        """
        return {
            product_id: [
                {key: value} for key, value in attributes.items()
            ]
        }

    def run(self, product_id):
        # Retrieve with reduced top_k (2 instead of 5) to reduce token usage
        retrieved_chunks = self.rag.search(
            "product technical specifications",
            product_id=product_id,
            top_k=2
        )

        # CRITICAL: Skip LLM call if no chunks retrieved
        if not retrieved_chunks:
            # Return empty attributes without LLM call
            return self.serialize_output(product_id, {})

        # Extract chunk text and limit context length
        chunks = [result.chunk_text for result in retrieved_chunks]
        chunks = self._limit_context_length(chunks)

        # CRITICAL: Single unified LLM call instead of 5 separate calls
        unified_results = self.unified_agent.extract(chunks)

        return self.serialize_output(product_id, unified_results["attributes"])