"""Shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    chunk_text: str
    metadata: Dict[str, Any]
