from chunking.chunker import ChunkingConfig, chunk_documents


def _doc(clean_text: str):
    return {
        "doc_id": "d1",
        "clean_text": clean_text,
        "sections": [{"title": "SPECIFICATIONS"}],
        "attributes_raw": {},
        "metadata": {
            "product_id": ["P1"],
            "document_type": "Product Documents",
            "source_file": "sample.csv",
        },
    }


def test_chunking_overlap_and_boundaries():
    text = "A" * 3000
    chunks = chunk_documents([_doc(text)], ChunkingConfig(chunk_size=1200, chunk_overlap=150))
    assert len(chunks) >= 3
    assert chunks[0]["char_start"] == 0
    assert chunks[1]["char_start"] == 1050
    assert chunks[0]["char_end"] == 1200


def test_chunking_preserves_metadata():
    chunks = chunk_documents([_doc("Useful content " * 200)])
    assert chunks
    c0 = chunks[0]
    assert c0["doc_id"] == "d1"
    assert c0["product_id"] == ["P1"]
    assert c0["document_type"] == "Product Documents"
    assert c0["section_title"] == "SPECIFICATIONS"


def test_chunking_skips_tiny_content():
    chunks = chunk_documents([_doc("small")], ChunkingConfig(min_chunk_chars=10))
    assert chunks == []
