import numpy as np

from embeddings import EmbeddingConfig, EmbeddingGenerator
from vector_store import build_faiss_index


def test_embedding_dimensions_are_consistent():
    generator = EmbeddingGenerator(EmbeddingConfig(dimensions=384, batch_size=8))
    vectors = generator.embed_texts(["hello world", "specification voltage 24V"])
    assert vectors.shape == (2, 384)


def test_faiss_index_count_matches_vectors():
    generator = EmbeddingGenerator(EmbeddingConfig(dimensions=384, batch_size=8))
    vectors = generator.embed_texts(["a", "b", "c"])
    index = build_faiss_index(vectors.astype(np.float32))
    assert index.ntotal == 3
