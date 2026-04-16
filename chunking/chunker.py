"""Deterministic chunking for Week 1 baseline RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 40


def _iter_windows(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    start = 0
    step = chunk_size - chunk_overlap
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield start, end
        if end >= text_len:
            break
        start += step


def chunk_documents(documents: List[Dict[str, Any]], config: ChunkingConfig | None = None) -> List[Dict[str, Any]]:
    """Split cleaned document text into overlapping metadata-rich chunks."""
    cfg = config or ChunkingConfig()
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        text = str(doc.get("clean_text") or "").strip()
        if not text:
            continue

        metadata = doc.get("metadata", {})
        sections = doc.get("sections", [])
        default_section = sections[0]["title"] if sections else "General"
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue

        chunk_idx = 0
        for start, end in _iter_windows(text, cfg.chunk_size, cfg.chunk_overlap):
            chunk_text = text[start:end].strip()
            if len(chunk_text) < cfg.min_chunk_chars:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{chunk_idx:04d}",
                    "doc_id": doc_id,
                    "product_id": metadata.get("product_id", []),
                    "document_type": metadata.get("document_type", ""),
                    "source_file": metadata.get("source_file", ""),
                    "section_title": default_section,
                    "chunk_text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                }
            )
            chunk_idx += 1
    return chunks
