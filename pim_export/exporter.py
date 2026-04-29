"""Export extracted product attributes in PIM-compatible JSON and CSV formats."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

APPROVED_STATUSES = {"APPROVED", "EDITED"}
DEFAULT_STATUS = "APPROVED"
SCHEMA_VERSION = "pim_export_v1"


@dataclass
class PIMAttribute:
    """One importable product attribute value."""

    name: str
    value: Any
    status: str = DEFAULT_STATUS
    source_document_id: str = ""
    source_chunk_id: str = ""
    confidence: float | None = None
    reviewed_at: str = ""
    reviewer: str = ""
    notes: str = ""


@dataclass
class PIMRecord:
    """PIM record keyed by a product identifier."""

    product_id: str
    attributes: List[PIMAttribute] = field(default_factory=list)


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _json_safe_value(value: Any) -> Any:
    if _is_empty_value(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_status(value: Any) -> str:
    status = str(value or DEFAULT_STATUS).strip().upper()
    return status or DEFAULT_STATUS


def _coerce_attribute(name: Any, value: Any, metadata: Mapping[str, Any] | None = None) -> PIMAttribute | None:
    metadata = metadata or {}
    attr_name = str(name or "").strip()
    safe_value = _json_safe_value(value)
    if not attr_name or safe_value is None:
        return None

    confidence = metadata.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    return PIMAttribute(
        name=attr_name,
        value=safe_value,
        status=_normalize_status(metadata.get("status")),
        source_document_id=str(metadata.get("source_document_id") or metadata.get("doc_id") or ""),
        source_chunk_id=str(metadata.get("source_chunk_id") or metadata.get("chunk_id") or ""),
        confidence=confidence,
        reviewed_at=str(metadata.get("reviewed_at") or ""),
        reviewer=str(metadata.get("reviewer") or ""),
        notes=str(metadata.get("notes") or metadata.get("comment") or ""),
    )


def _attributes_from_value(value: Any) -> List[PIMAttribute]:
    attributes: List[PIMAttribute] = []

    if isinstance(value, Mapping):
        if "attributes" in value:
            attributes.extend(_attributes_from_value(value.get("attributes")))
            return attributes

        if "name" in value and "value" in value:
            attr = _coerce_attribute(value.get("name"), value.get("value"), value)
            if attr:
                attributes.append(attr)
            return attributes

        for name, attr_value in value.items():
            if isinstance(attr_value, Mapping) and "value" in attr_value:
                attr = _coerce_attribute(name, attr_value.get("value"), attr_value)
            else:
                attr = _coerce_attribute(name, attr_value)
            if attr:
                attributes.append(attr)
        return attributes

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            attributes.extend(_attributes_from_value(item))

    return attributes


def normalize_pim_records(
    extracted: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    approved_only: bool = True,
) -> List[PIMRecord]:
    """Normalize extractor or approval output into importable PIM records.

    Supported inputs:
    - Current orchestrator output: {"SKU": [{"voltage": "24V"}, ...]}
    - Approval records: [{"product_id": "SKU", "attributes": [...]}]
    - Flat product records: {"SKU": {"attributes": {"voltage": "24V"}}}
    """
    by_product: Dict[str, PIMRecord] = {}

    if isinstance(extracted, Mapping):
        items: Iterable[tuple[str, Any]] = extracted.items()
    else:
        items = (
            (str(record.get("product_id", "")).strip(), record)
            for record in extracted
            if isinstance(record, Mapping)
        )

    for product_id, raw_value in items:
        product_id = str(product_id or "").strip()
        if not product_id:
            continue

        record = by_product.setdefault(product_id, PIMRecord(product_id=product_id))
        for attribute in _attributes_from_value(raw_value):
            if approved_only and attribute.status not in APPROVED_STATUSES:
                continue
            record.attributes.append(attribute)

    return [record for record in by_product.values() if record.attributes]


def _record_to_dict(record: PIMRecord) -> Dict[str, Any]:
    return {
        "product_id": record.product_id,
        "attributes": [asdict(attribute) for attribute in record.attributes],
    }


def write_pim_json(
    records: Sequence[PIMRecord],
    output_path: str | Path,
    generated_at: str | None = None,
) -> Path:
    """Write canonical PIM JSON import payload."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "attribute_count": sum(len(record.attributes) for record in records),
        "records": [_record_to_dict(record) for record in records],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def write_pim_csv(records: Sequence[PIMRecord], output_path: str | Path) -> Path:
    """Write flat CSV suitable for PIM import mapping."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product_id",
        "attribute_name",
        "attribute_value",
        "status",
        "source_document_id",
        "source_chunk_id",
        "confidence",
        "reviewed_at",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for attribute in record.attributes:
                writer.writerow(
                    {
                        "product_id": record.product_id,
                        "attribute_name": attribute.name,
                        "attribute_value": attribute.value,
                        "status": attribute.status,
                        "source_document_id": attribute.source_document_id,
                        "source_chunk_id": attribute.source_chunk_id,
                        "confidence": "" if attribute.confidence is None else attribute.confidence,
                        "reviewed_at": attribute.reviewed_at,
                        "reviewer": attribute.reviewer,
                        "notes": attribute.notes,
                    }
                )
    return path


def export_pim_files(
    extracted: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    basename: str = "pim_export",
    approved_only: bool = True,
) -> Dict[str, Any]:
    """Normalize extracted attributes and write both JSON and CSV exports."""
    records = normalize_pim_records(extracted, approved_only=approved_only)
    out = Path(output_dir)
    json_path = write_pim_json(records, out / f"{basename}.json")
    csv_path = write_pim_csv(records, out / f"{basename}.csv")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "record_count": len(records),
        "attribute_count": sum(len(record.attributes) for record in records),
    }
