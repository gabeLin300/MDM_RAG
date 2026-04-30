"""PIM schema definitions and validators."""

from __future__ import annotations

import re
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


ATTRIBUTE_PATTERNS: Dict[str, str] = {
    "voltage_rating": r"\d+\.?\d*\s*[Vv]",
    "current_rating": r"\d+\.?\d*\s*[Aa]",
    "power_consumption": r"\d+\.?\d*\s*[Ww]",
    "frequency": r"\d+\.?\d*\s*[Hh][Zz]",
    "operating_temperature": r"-?\d+",
    "storage_temperature": r"-?\d+",
    "dimensions": r"\d+",
    "weight": r"\d+\.?\d*\s*(?:g|kg|lb|oz)",
}

_LOW_CONFIDENCE_THRESHOLD = 60
_MIN_FILLED_FOR_AUTO_APPROVE = 3


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _iter_attribute_entries(attributes: Any) -> List[Tuple[str, Dict[str, Any]]]:
    entries: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(attributes, dict):
        for name, entry in attributes.items():
            if isinstance(entry, dict):
                value = entry.get("value")
                confidence = entry.get("confidence", 0)
            else:
                value = entry
                confidence = 0
            entries.append((str(name), {"value": value, "confidence": confidence}))
        return entries

    if isinstance(attributes, list):
        for item in attributes:
            if not isinstance(item, dict):
                continue
            if "name" in item:
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                entries.append(
                    (
                        name,
                        {
                            "value": item.get("value"),
                            "confidence": item.get("confidence", 0),
                        },
                    )
                )
                continue

            # Fallback for [{"Supply Voltage": "24"}] shape.
            if len(item) == 1:
                (name, value), = item.items()
                entries.append((str(name), {"value": value, "confidence": 0}))

    return entries


class AttributeValidator:
    def validate(self, attributes: Any) -> Tuple[List[str], bool]:
        """Return (quality_flags, review_required) for extracted attributes."""
        flags: List[str] = []
        entries = _iter_attribute_entries(attributes)

        filled = [
            name
            for name, entry in entries
            if entry.get("value") is not None and str(entry.get("value")).strip() != ""
        ]

        for name, entry in entries:
            value = entry.get("value")
            if value is None or str(value).strip() == "":
                continue

            try:
                confidence = int(float(entry.get("confidence", 0) or 0))
            except (TypeError, ValueError):
                confidence = 0

            if confidence < _LOW_CONFIDENCE_THRESHOLD:
                flags.append(f"low_confidence:{name}")

            normalized = _normalize_key(name)
            pattern = ATTRIBUTE_PATTERNS.get(name) or ATTRIBUTE_PATTERNS.get(normalized)
            if pattern and not re.search(pattern, str(value), re.IGNORECASE):
                flags.append(f"format_mismatch:{name}")

        review_required = bool(flags) or len(filled) < _MIN_FILLED_FOR_AUTO_APPROVE
        return flags, review_required
