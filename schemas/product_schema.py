"""Week 1 baseline PIM schema definitions and validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

REQUIRED_TOP_LEVEL_FIELDS = (
    "doc_id",
    "product_id",
    "document_type",
    "title",
    "attributes",
    "source_trace",
    "quality_flags",
)


@dataclass
class SourceTrace:
    """Traceability record linking an extracted attribute to source text."""

    chunk_id: str
    source_file: str
    section_title: str
    score: float


@dataclass
class ProductRecordV0:
    """Baseline Week 1 record used by ingestion/retrieval handoff."""

    doc_id: str
    product_id: List[str]
    document_type: str
    title: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_trace: List[Dict[str, Any]] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_product_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate baseline record shape and required field types."""
    errors: List[str] = []
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in record]
    if missing:
        errors.append(f"missing_fields={missing}")

    if "doc_id" in record and not str(record.get("doc_id", "")).strip():
        errors.append("doc_id_empty")

    product_id = record.get("product_id")
    if "product_id" in record and not isinstance(product_id, list):
        errors.append("product_id_not_list")

    if "attributes" in record and not isinstance(record.get("attributes"), dict):
        errors.append("attributes_not_dict")

    source_trace = record.get("source_trace")
    if "source_trace" in record and not isinstance(source_trace, list):
        errors.append("source_trace_not_list")

    quality_flags = record.get("quality_flags")
    if "quality_flags" in record and not isinstance(quality_flags, list):
        errors.append("quality_flags_not_list")

    return len(errors) == 0, errors
