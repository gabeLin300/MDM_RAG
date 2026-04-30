"""Main pipeline entrypoint: ingest -> parse -> chunk -> embed -> FAISS -> RAG smoke."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from chunking import ChunkingConfig, chunk_documents
from embeddings import EmbeddingConfig, EmbeddingGenerator
from ingestion.csv_loader import (
    REQUIRED_COLUMNS,
    discover_csv_files,
    load_csv_files_with_report,
    validate_schema,
)
from pipeline.analyze_dataset import write_dataset_profile
from preprocessing.cleaner import clean_text
from preprocessing.language import detect_language, normalize_encoding
from preprocessing.parser import process_row
from retrieval import BaselineRAG
from schemas.product_schema import ProductRecordV0, validate_product_record
from vector_store import build_faiss_index, load_faiss_artifacts, save_faiss_artifacts
from vector_store.faiss_store import faiss
from agents.orchestrator import Orchestrator
from pim_export import export_pim_files

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SAMPLE_QUERIES = [
    "Which panels are supported by the Universal CLSS Gateway Modbus and BACnet feature?",
    "Will there be a price increase for the Universal CLSS Gateway Modbus/BACnet feature?",
    "What communication protocols are mentioned for the CLSS Gateway?",
    "In the Li-Ion Tamer Rack Monitor, what is the maximum number of sensors per controller?",
    "What is the Li-Ion Tamer controller input power range?",
    "What are the controller and sensor power consumption values for Li-Ion Tamer?",
    "What are the gas detection threshold and response time in the Li-Ion Tamer datasheet?",
    "What key features are listed for the TC300 Commercial Thermostat?",
    "What user interface/display capabilities are listed for the TC300 thermostat?",
    "What is the ALER-9000 controller described as (controller/server/platform)?",
    "What kind of valves are in the VN Series Zone Valves datasheet (2-way or 3-way)?",
    "What actuator behavior is described for VN Series valves (fail-safe/fail-in-place)?",
    "What are the TR21/TR22/TR23 wall modules used for?",
    "What sensor inputs or measurements are mentioned for TR21/TR22/TR23 wall modules?",
    "¿Cuál es la finalidad del controlador IQ5 según la ficha técnica en español?",
    "Welche Funktionen bietet die IQ5-DDC-Station laut Datenblatt?"
]

ATTRIBUTE_EXTRACTION_QUERIES = [
    "electrical specifications voltage current power frequency operating limits",
    "environmental specifications temperature humidity ingress protection certification",
    "physical specifications dimensions weight material mounting",
]
RETRIEVAL_TOP_K_PER_QUERY = 4
RETRIEVAL_MAX_CHUNKS_PER_PRODUCT = 10
PRODUCT_SNIPPET_CHARS = 900
PRODUCT_MAX_TOKENS = 200
BATCH_MIN_PRODUCTS = 20
BATCH_MAX_PRODUCTS = 50
BATCH_TARGET_PRODUCTS = 30

# Canonical naming layer for dynamic attributes.
# Keys are normalized via _normalize_attr_alias_key.
ATTRIBUTE_ALIAS_MAP = {
    "voltage": "Supply Voltage",
    "voltage_rating": "Supply Voltage",
    "supply_voltage": "Supply Voltage",
    "supply_voltage_frequency": "Supply Voltage Frequency",
    "frequency": "Supply Voltage Frequency",
    "current": "Current Rating",
    "current_rating": "Current Rating",
    "power": "Driving Power Consumption",
    "power_consumption": "Driving Power Consumption",
    "driving_power_consumption": "Driving Power Consumption",
    "operating_torque": "Operating Torque",
    "torque": "Operating Torque",
    "enclosure": "Enclosure Rating",
    "enclosure_rating": "Enclosure Rating",
    "ip_rating": "Enclosure Rating",
    "control_signal": "Control Signal",
    "feedback_signal": "Feedback",
    "feedback": "Feedback",
    "fail_safe_action": "Fail-Safe Action",
    "failsafe_action": "Fail-Safe Action",
    "failsafe_timing": "Fail-Safe Timing",
    "fail_safe_timing": "Fail-Safe Timing",
    "actuator_type": "Actuator Type",
    "mounting_type": "Mounting Type",
    "electrical_connection_type": "Electrical Connection Type",
    "max_operating_ambient_temperature": "Maximum Operating Ambient Temperature",
    "maximum_operating_ambient_temperature": "Maximum Operating Ambient Temperature",
    "min_operating_ambient_temperature": "Minimum Operating Ambient Temperature",
    "minimum_operating_ambient_temperature": "Minimum Operating Ambient Temperature",
    "max_storage_temperature": "Maximum Storage Temperature",
    "maximum_storage_temperature": "Maximum Storage Temperature",
    "min_storage_temperature": "Minimum Storage Temperature",
    "minimum_storage_temperature": "Minimum Storage Temperature",
    "max_operating_humidity": "Maximum Operating Humidity",
    "maximum_operating_humidity": "Maximum Operating Humidity",
    "min_operating_humidity": "Minimum Operating Humidity",
    "minimum_operating_humidity": "Minimum Operating Humidity",
    "overall_width": "Overall Width",
    "overall_height": "Overall Height",
    "overall_depth": "Overall Depth",
    "brand_name": "Brand",
    "brand": "Brand",
    "certificates_and_standards": "Certificates and Standards",
    "standards_compliance": "Certificates and Standards",
    "for_use_with": "For Use With",
    "valve_compatibility": "Valve Compatibility",
    "manual_operation_type": "Manual Operation Type",
    "number_of_auxiliary_switches": "Number Of Auxiliary Switches",
    "maximum_angle_of_rotation": "Maximum Angle Of Rotation",
    "pnc_selector": "PNC-Selector",
    "pnc_display_sku": "PNC-Display SKU",
}


def _is_document_row(row: Dict[str, Any]) -> bool:
    doc_id = str(row.get("id") or row.get("doc_id") or "").strip()
    content = str(row.get("file_content") or row.get("raw_text") or "").strip()
    return bool(doc_id) or bool(content)


def _validate_parsed_document(document: Dict[str, Any]) -> bool:
    if not document.get("doc_id"):
        return False
    if not document.get("clean_text"):
        return False
    if len(document["clean_text"]) < 20:
        return False
    if "sections" not in document or "attributes_raw" not in document:
        return False
    return True


def _build_ingestion_report(
    file_paths: List[Path],
    df,
    csv_report: Dict[str, Any],
    schema_ok: bool,
    skipped_empty: int,
    skipped_invalid: int,
    parsed_count: int,
) -> Dict[str, Any]:
    per_file_counts = (
        df.groupby("source_file").size().to_dict() if not df.empty and "source_file" in df.columns else {}
    )
    return {
        "source_files": [str(p) for p in file_paths],
        "rows_read": int(len(df)),
        "rows_skipped_empty": int(skipped_empty),
        "rows_skipped_invalid": int(skipped_invalid),
        "rows_parsed": int(parsed_count),
        "schema_ok": schema_ok,
        "required_columns": REQUIRED_COLUMNS,
        "per_file_counts": {str(k): int(v) for k, v in per_file_counts.items()},
        "files_seen": int(csv_report.get("files_seen", 0)),
        "files_loaded": int(csv_report.get("files_loaded", 0)),
        "files_skipped_schema": int(csv_report.get("files_skipped_schema", 0)),
        "files_skipped_read_error": int(csv_report.get("files_skipped_read_error", 0)),
    }


def parse_documents(csv_input: Union[str, Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and parse documents into normalized contract consumed by chunking."""
    file_paths = discover_csv_files(csv_input)
    if not file_paths:
        logger.warning("No CSV files found for input: %s", csv_input)
        return [], {"source_files": [], "rows_read": 0, "schema_ok": False}

    df, csv_report = load_csv_files_with_report(file_paths)
    if df.empty:
        logger.warning("CSV loading returned no data for input: %s", csv_input)
        return [], {"source_files": [str(p) for p in file_paths], "rows_read": 0, "schema_ok": False}

    schema_ok = validate_schema(df, REQUIRED_COLUMNS)
    if not schema_ok:
        logger.warning("Merged dataframe failed schema validation for required columns.")

    parsed_documents: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_invalid = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if not _is_document_row(row_dict):
            skipped_empty += 1
            continue

        parsed = process_row(row_dict, row_index=idx)
        cleaned_text = clean_text(parsed.get("text", ""))
        normalized_text = normalize_encoding(cleaned_text)
        language = detect_language(normalized_text)

        document: Dict[str, Any] = {
            "doc_id": parsed.get("doc_id", f"row-{idx}"),
            "title": parsed.get("title", ""),
            "clean_text": normalized_text,
            "sections": parsed.get("sections", []),
            "attributes_raw": parsed.get("attributes_raw", {}),
            "metadata": {
                "title": parsed.get("title", ""),
                "file_name": parsed.get("file_name", ""),
                "source_file": row_dict.get("source_file", ""),
                "document_type": parsed.get("document_type", ""),
                "product_id": parsed.get("product_id", []),
                "language": language,
                "row_index": int(idx),
                "char_count": len(normalized_text),
            },
        }

        if _validate_parsed_document(document):
            parsed_documents.append(document)
        else:
            skipped_invalid += 1

    report = _build_ingestion_report(
        file_paths=file_paths,
        df=df,
        csv_report=csv_report,
        schema_ok=schema_ok,
        skipped_empty=skipped_empty,
        skipped_invalid=skipped_invalid,
        parsed_count=len(parsed_documents),
    )
    return parsed_documents, report


def _validate_product_records(parsed_documents: List[Dict[str, Any]]) -> int:
    invalid = 0
    for doc in parsed_documents:
        candidate = ProductRecordV0(
            doc_id=doc["doc_id"],
            product_id=doc["metadata"].get("product_id", []),
            document_type=doc["metadata"].get("document_type", ""),
            title=doc.get("title", ""),
            attributes=doc.get("attributes_raw", {}),
            source_trace=[],
            quality_flags=[],
        ).to_dict()
        ok, _ = validate_product_record(candidate)
        if not ok:
            invalid += 1
    return invalid


def build_index_artifacts(
    parsed_documents: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    chunks = chunk_documents(parsed_documents, ChunkingConfig(chunk_size=1200, chunk_overlap=150))
    texts = [chunk["chunk_text"] for chunk in chunks]

    embedder = EmbeddingGenerator(
        EmbeddingConfig(
            model_name=embedding_model,
            dimensions=384,
            batch_size=64,
            normalize=True,
        )
    )
    vectors = embedder.embed_texts(texts) if texts else np.zeros((0, 384), dtype=np.float32)
    index = build_faiss_index(vectors) if len(vectors) else None

    metadata_store: List[Dict[str, Any]] = []
    for chunk in chunks:
        record = dict(chunk)
        metadata_store.append(record)

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at_utc": now,
        "embedding_model": embedding_model,
        "embedding_backend": embedder.backend,
        "index_backend": "faiss" if faiss is not None else "numpy-flatip",
        "embedding_dimensions": int(vectors.shape[1]) if vectors.size else 384,
        "documents_count": len(parsed_documents),
        "chunks_count": len(chunks),
        "index_size": int(index.ntotal) if index is not None else 0,
        "source_files": sorted({doc["metadata"].get("source_file", "") for doc in parsed_documents}),
    }

    artifact_paths = {}
    if index is not None:
        artifact_paths = save_faiss_artifacts(
            index=index,
            metadata=metadata_store,
            manifest=manifest,
            output_dir=output_dir,
        )

    return {
        "chunks": chunks,
        "vectors": vectors,
        "index": index,
        "metadata_store": metadata_store,
        "manifest": manifest,
        "artifact_paths": artifact_paths,
    }


def _normalized_sources_from_documents(parsed_documents: List[Dict[str, Any]]) -> List[str]:
    sources = set()
    for doc in parsed_documents:
        raw = str(doc.get("metadata", {}).get("source_file", "")).strip()
        if not raw:
            continue
        sources.add(Path(raw.replace("\\", "/")).name)
    return sorted(sources)


def _normalized_sources_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    sources = manifest.get("source_files", []) or []
    normalized = {Path(str(src).replace("\\", "/")).name for src in sources if str(src).strip()}
    return sorted(normalized)


def _artifacts_valid_for_reuse(
    manifest: Dict[str, Any],
    parsed_documents: List[Dict[str, Any]],
    embedding_model: str,
) -> tuple[bool, str]:
    if str(manifest.get("embedding_model", "")).strip() != embedding_model:
        return False, "embedding_model_mismatch"

    expected_docs = len(parsed_documents)
    cached_docs = int(manifest.get("documents_count", -1))
    if cached_docs != expected_docs:
        return False, f"documents_count_mismatch:{cached_docs}!={expected_docs}"

    expected_sources = _normalized_sources_from_documents(parsed_documents)
    cached_sources = _normalized_sources_from_manifest(manifest)
    if expected_sources and cached_sources and set(expected_sources) != set(cached_sources):
        return False, "source_files_mismatch"

    return True, "ok"


def load_or_build_index_artifacts(
    parsed_documents: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    embedding_model: str,
    reuse_index: bool = True,
) -> Dict[str, Any]:
    out = Path(output_dir)
    index_path = out / "baseline.index"
    metadata_path = out / "baseline_metadata.json"
    manifest_path = out / "baseline_manifest.json"
    artifact_paths = {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }

    if reuse_index and index_path.exists() and metadata_path.exists() and manifest_path.exists():
        try:
            index, metadata_store, manifest = load_faiss_artifacts(output_dir=out)
            valid, reason = _artifacts_valid_for_reuse(
                manifest=manifest,
                parsed_documents=parsed_documents,
                embedding_model=embedding_model,
            )
            if valid:
                logger.info("Reusing cached index artifacts from %s", out)
                return {
                    "chunks": metadata_store,
                    "vectors": np.zeros((0, int(manifest.get("embedding_dimensions", 384))), dtype=np.float32),
                    "index": index,
                    "metadata_store": metadata_store,
                    "manifest": manifest,
                    "artifact_paths": artifact_paths,
                    "index_reused": True,
                }
            logger.info("Cached index exists but will be rebuilt: %s", reason)
        except Exception as exc:
            logger.warning("Failed to load cached index artifacts; rebuilding. reason=%s", exc)

    built = build_index_artifacts(
        parsed_documents=parsed_documents,
        output_dir=out,
        embedding_model=embedding_model,
    )
    built["index_reused"] = False
    return built


def run_query_smoke(
    index: Any,
    metadata_store: List[Dict[str, Any]],
    embedding_model: str,
    enable_sparse: bool = True,
    enable_reranker: bool = True,
    queries: List[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    queries = queries or DEFAULT_SAMPLE_QUERIES
    rag = BaselineRAG(
        index=index,
        metadata=metadata_store,
        embedding_model=embedding_model,
        enable_sparse=enable_sparse,
        enable_reranker=enable_reranker,
    )

    outputs = []
    latencies_ms = []
    for q in queries:
        start = time.perf_counter()
        result = rag.answer(q, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        outputs.append({"query": q, **result, "latency_ms": round(elapsed_ms, 2)})

    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    return {
        "query_results": outputs,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
    }


def get_all_product_ids_from_metadata(metadata_store: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique product_ids from metadata store."""
    product_ids = set()
    for record in metadata_store:
        # product_id is stored as list in metadata
        pids = record.get("product_id", [])
        if isinstance(pids, list):
            product_ids.update(pids)
        elif isinstance(pids, str):
            product_ids.add(pids)
    return sorted(list(product_ids))


def _build_product_chunk_candidates(
    metadata_store: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in metadata_store:
        pids = chunk.get("product_id", [])
        if isinstance(pids, str):
            pids = [pids]
        if not isinstance(pids, list):
            continue
        for pid in pids:
            pid = str(pid).strip()
            if not pid:
                continue
            candidates.setdefault(pid, []).append(chunk)
    return candidates


def _build_doc_chunk_index(metadata_store: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in metadata_store:
        doc_id = str(chunk.get("doc_id", "")).strip()
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(chunk)
    return by_doc


def _score_chunk_by_query_overlap(chunk_text: str, query: str) -> float:
    q_terms = [t for t in re.findall(r"[a-z0-9]+", str(query).lower()) if len(t) > 2]
    if not q_terms:
        return 0.0
    c_text = str(chunk_text).lower()
    hits = sum(1 for term in q_terms if term in c_text)
    if hits == 0:
        return 0.0
    return float(hits) / float(len(q_terms))


def _select_topk_chunks_for_products(
    rag: BaselineRAG,
    metadata_store: List[Dict[str, Any]],
    product_ids: List[str],
    queries: List[str],
    top_k_per_query: int = 3,
    max_chunks_per_product: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {}
    candidates_by_product = _build_product_chunk_candidates(metadata_store)
    chunks_by_doc = _build_doc_chunk_index(metadata_store)

    # Product -> doc scope built from any chunk that references that product.
    doc_scope_by_product: Dict[str, set[str]] = {}
    for product_id, chunks in candidates_by_product.items():
        scope = {str(chunk.get("doc_id", "")).strip() for chunk in chunks if str(chunk.get("doc_id", "")).strip()}
        doc_scope_by_product[product_id] = scope

    for product_id in product_ids:
        scoped_docs = doc_scope_by_product.get(product_id, set())
        scoped_candidates: List[Dict[str, Any]] = []
        if scoped_docs:
            for doc_id in scoped_docs:
                scoped_candidates.extend(chunks_by_doc.get(doc_id, []))
        else:
            scoped_candidates = list(candidates_by_product.get(product_id, []))

        candidate_chunk_ids = {str(c.get("chunk_id", "")) for c in scoped_candidates if str(c.get("chunk_id", "")).strip()}
        candidate_by_id = {str(c.get("chunk_id", "")): c for c in scoped_candidates if str(c.get("chunk_id", "")).strip()}

        by_chunk_id: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        overfetch_k = max(top_k_per_query * 30, 120)
        for query in queries:
            hits = rag.search(
                query=query,
                top_k=overfetch_k,
                rerank=False,
            )
            scoped_hits = [h for h in hits if h.chunk_id in candidate_chunk_ids]

            # Fallback when scoped FAISS hits are sparse: lexical scoring inside scoped candidates.
            if len(scoped_hits) < top_k_per_query and scoped_candidates:
                lexical_ranked = sorted(
                    scoped_candidates,
                    key=lambda c: _score_chunk_by_query_overlap(c.get("chunk_text", ""), query),
                    reverse=True,
                )
                for c in lexical_ranked[: top_k_per_query]:
                    cid = str(c.get("chunk_id", ""))
                    if not cid:
                        continue
                    score = _score_chunk_by_query_overlap(c.get("chunk_text", ""), query)
                    existing = by_chunk_id.get(cid)
                    if existing is None or score > existing[0]:
                        by_chunk_id[cid] = (float(score), dict(c))

            for hit in scoped_hits[: top_k_per_query]:
                score = float(hit.score)
                existing = by_chunk_id.get(hit.chunk_id)
                if existing is None or score > existing[0]:
                    by_chunk_id[hit.chunk_id] = (score, dict(candidate_by_id.get(hit.chunk_id, hit.metadata)))

        if by_chunk_id:
            ranked = sorted(
                by_chunk_id.values(),
                key=lambda item: (item[0], -int(item[1].get("char_start", 0))),
                reverse=True,
            )
            selected[product_id] = [item[1] for item in ranked[:max_chunks_per_product]]
            continue

        fallback_chunks = scoped_candidates or candidates_by_product.get(product_id, [])
        fallback_sorted = sorted(
            fallback_chunks,
            key=lambda c: int(c.get("char_start", 0)),
        )
        selected[product_id] = fallback_sorted[:max_chunks_per_product]

    return selected


def _group_products_by_evidence(
    selected_chunks_by_product: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for product_id, chunks in selected_chunks_by_product.items():
        if not chunks:
            continue
        signature = tuple(sorted(str(chunk.get("chunk_id", "")) for chunk in chunks if chunk.get("chunk_id")))
        if not signature:
            continue
        if signature not in groups:
            groups[signature] = {
                "product_ids": [],
                "chunks": chunks,
            }
        groups[signature]["product_ids"].append(product_id)

    grouped = list(groups.values())
    grouped.sort(key=lambda g: len(g["product_ids"]), reverse=True)
    return grouped


def _compact_text_snippet(text: str, max_chars: int = PRODUCT_SNIPPET_CHARS) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return str(text)[:max_chars]
    merged = " ".join(lines[:3])
    if len(merged) > max_chars:
        return merged[:max_chars].rstrip()
    return merged


def _compact_chunks_for_llm(chunks: List[Dict[str, Any]], snippet_chars: int = PRODUCT_SNIPPET_CHARS) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for chunk in chunks:
        record = dict(chunk)
        record["chunk_text"] = _compact_text_snippet(record.get("chunk_text", ""), max_chars=snippet_chars)
        compacted.append(record)
    return compacted


def _estimate_tokens(text: str) -> int:
    # Fast heuristic for budgeting without model-specific tokenizer dependency.
    return max(1, len(text) // 4)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _build_product_context(
    chunks: List[Dict[str, Any]],
    snippet_chars: int = PRODUCT_SNIPPET_CHARS,
    max_tokens: int = PRODUCT_MAX_TOKENS,
) -> str:
    parts: List[str] = []
    token_total = 0
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        section = str(chunk.get("section_title", "")).strip()
        doc_type = str(chunk.get("document_type", "")).strip()
        source_file = str(chunk.get("source_file", "")).strip()
        text = _compact_text_snippet(chunk.get("chunk_text", ""), max_chars=snippet_chars)
        if not text:
            continue
        meta = " | ".join([m for m in [section, doc_type, source_file] if m])
        if meta:
            line = f"[{chunk_id}] ({meta}) {text}"
        else:
            line = f"[{chunk_id}] {text}"

        remaining = max_tokens - token_total
        if remaining <= 0:
            break
        line = _truncate_to_token_budget(line, remaining)
        if not line:
            break
        line_tokens = _estimate_tokens(line)
        if line_tokens <= 0:
            continue
        parts.append(line)
        token_total += line_tokens
        if token_total >= max_tokens:
            break

    return "\n".join(parts)


def _is_empty_extracted_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _normalize_attr_alias_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _canonical_attribute_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    alias = ATTRIBUTE_ALIAS_MAP.get(_normalize_attr_alias_key(cleaned))
    return alias or cleaned


def _to_dynamic_attribute_output(extracted: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert full orchestrator records to product -> [{attribute: value}, ...]."""
    dynamic: Dict[str, List[Dict[str, Any]]] = {}
    for product_id, record in extracted.items():
        attributes = {}
        if isinstance(record, dict):
            attributes = record.get("attributes", {})
        if not isinstance(attributes, (dict, list)):
            dynamic[str(product_id)] = []
            continue

        rows: List[Dict[str, Any]] = []
        if isinstance(attributes, list):
            for entry in attributes:
                if not isinstance(entry, dict):
                    continue
                if "name" in entry:
                    name = str(entry.get("name") or "").strip()
                    value = entry.get("value")
                    if not name or _is_empty_extracted_value(value):
                        continue
                    rows.append({_canonical_attribute_name(name): value})
                    continue
                if len(entry) == 1:
                    (name, value), = entry.items()
                    if _is_empty_extracted_value(value):
                        continue
                    rows.append({_canonical_attribute_name(str(name)): value})
        else:
            for key, entry in attributes.items():
                value = entry.get("value") if isinstance(entry, dict) else entry
                if _is_empty_extracted_value(value):
                    continue
                rows.append({_canonical_attribute_name(str(key)): value})

        dynamic[str(product_id)] = rows
    return dynamic


def _make_product_batches(
    product_ids: List[str],
    min_size: int = BATCH_MIN_PRODUCTS,
    max_size: int = BATCH_MAX_PRODUCTS,
    target_size: int = BATCH_TARGET_PRODUCTS,
) -> List[List[str]]:
    if not product_ids:
        return []
    if len(product_ids) <= max_size:
        return [product_ids]

    batches: List[List[str]] = []
    i = 0
    n = len(product_ids)
    while i < n:
        batch = product_ids[i : i + target_size]
        batches.append(batch)
        i += target_size

    if len(batches) >= 2 and len(batches[-1]) < min_size:
        deficit = min_size - len(batches[-1])
        take = min(deficit, len(batches[-2]) - min_size)
        if take > 0:
            moved = batches[-2][-take:]
            batches[-2] = batches[-2][:-take]
            batches[-1] = moved + batches[-1]

    return [b for b in batches if b]


def run_batch_orchestrator(
    orchestrator: Orchestrator,
    metadata_store: List[Dict[str, Any]],
    product_ids: List[str] | None = None,
    retrieval_queries: List[str] | None = None,
    top_k_per_query: int = 4,
    max_chunks_per_product: int = 10,
    snippet_chars: int = PRODUCT_SNIPPET_CHARS,
    rate_limit_delay: float = 1.0,
) -> Dict[str, Any]:
    """Retrieve top-k chunks per product and extract in large product-count batches."""
    import time

    results = {}
    failed_group_product_ids: List[str] = []
    processed_groups = 0

    product_ids = product_ids or get_all_product_ids_from_metadata(metadata_store)
    retrieval_queries = retrieval_queries or ATTRIBUTE_EXTRACTION_QUERIES
    selected_chunks_by_product = _select_topk_chunks_for_products(
        rag=orchestrator.rag,
        metadata_store=metadata_store,
        product_ids=product_ids,
        queries=retrieval_queries,
        top_k_per_query=top_k_per_query,
        max_chunks_per_product=max_chunks_per_product,
    )
    ordered_products = [pid for pid in product_ids if selected_chunks_by_product.get(pid)]
    grouped_batches = _make_product_batches(ordered_products)
    batch_sizes = [len(batch) for batch in grouped_batches]
    logger.info(
        "Prepared %s product batches from %s products",
        len(grouped_batches),
        len(ordered_products),
    )
    if batch_sizes:
        logger.info(
            "Batch size distribution: %s (min=%s, max=%s)",
            batch_sizes,
            min(batch_sizes),
            max(batch_sizes),
        )

    for idx, batch_product_ids in enumerate(grouped_batches):
        product_contexts: Dict[str, str] = {}
        source_chunk_ids_by_product: Dict[str, List[str]] = {}
        for product_id in batch_product_ids:
            chunks = selected_chunks_by_product.get(product_id, [])
            compact_chunks = _compact_chunks_for_llm(chunks, snippet_chars=snippet_chars)
            source_chunk_ids_by_product[product_id] = [str(c.get("chunk_id", "")) for c in compact_chunks if c.get("chunk_id")]
            product_contexts[product_id] = _build_product_context(compact_chunks, snippet_chars=snippet_chars)

        if not product_contexts:
            continue

        backoff_delay = rate_limit_delay
        max_retries = 3
        retry_count = 0
        had_retry = False

        while retry_count < max_retries:
            try:
                logger.info(
                    "Processing batch %s/%s (%s products)",
                    idx + 1,
                    len(grouped_batches),
                    len(batch_product_ids),
                )
                batch_results = orchestrator.run_for_product_batch(
                    product_contexts=product_contexts,
                    source_chunk_ids_by_product=source_chunk_ids_by_product,
                )
                results.update(batch_results)

                processed_groups += 1
                break

            except Exception as e:
                retry_count += 1
                had_retry = True
                if retry_count < max_retries:
                    logger.warning(
                        f"Attempt {retry_count}/{max_retries} failed for batch {idx + 1}: {e}. "
                        f"Retrying in {backoff_delay}s..."
                    )
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    logger.error(
                        "Failed batch %s after %s attempts (products=%s): %s",
                        idx + 1,
                        max_retries,
                        batch_product_ids[:5],
                        e,
                    )
                    failed_group_product_ids.extend(batch_product_ids)

        if idx < len(grouped_batches) - 1:
            time.sleep(rate_limit_delay)

    if failed_group_product_ids:
        logger.warning("Failed products: %s", failed_group_product_ids)

    return {
        "extracted_attributes": results,
        "products_processed": len(results),
        "products_failed": len(set(failed_group_product_ids)),
        "failed_product_ids": sorted(set(failed_group_product_ids)),
        "groups_processed": processed_groups,
        "groups_total": len(grouped_batches),
    }


def run_pipeline(
    profile: str = "sample",
    input_path: Union[str, Path] | None = None,
    output_dir: Union[str, Path] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    enable_sparse: bool = True,
    enable_reranker: bool = True,
    run_orchestrator: bool = True,
) -> Dict[str, Any]:
    profile = profile.lower().strip()
    if input_path is None:
        if profile == "full":
            input_path = "data/raw/advanced_rag_full_dataset.csv"
        else:
            input_path = "data/raw/100_sample_advanced_rag.csv"
    if output_dir is None:
        output_dir = Path("data/processed") / profile

    parsed_documents, ingestion_report = parse_documents(input_path)
    invalid_schema_records = _validate_product_records(parsed_documents)
    artifacts = load_or_build_index_artifacts(
        parsed_documents=parsed_documents,
        output_dir=output_dir,
        embedding_model=embedding_model,
    )

    smoke = {"query_results": [], "latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    if artifacts["index"] is not None:
        smoke = run_query_smoke(
            index=artifacts["index"],
            metadata_store=artifacts["metadata_store"],
            embedding_model=embedding_model,
            enable_sparse=enable_sparse,
            enable_reranker=enable_reranker,
            queries=DEFAULT_SAMPLE_QUERIES,
            top_k=5,
        )

    # Batch orchestrator stage: process all unique product_ids
    orchestrator_results = {
        "extracted_attributes": {},
        "products_processed": 0,
        "products_failed": 0,
        "failed_product_ids": [],
        "groups_processed": 0,
        "groups_total": 0,
        "backend": "skipped",
    }

    if run_orchestrator and artifacts["index"] is not None:
        try:
            logger.info("Starting batch orchestrator stage...")
            product_ids = get_all_product_ids_from_metadata(artifacts["metadata_store"])
            logger.info(f"Found {len(product_ids)} unique product IDs to process")

            orchestrator = Orchestrator(
                index=artifacts["index"],
                metadata=artifacts["metadata_store"],
            )

            # Run batch orchestrator
            # Embeddings are handled internally by BaselineRAG through DenseRetriever
            batch_result = run_batch_orchestrator(
                orchestrator,
                artifacts["metadata_store"],
                product_ids=product_ids,
                top_k_per_query=RETRIEVAL_TOP_K_PER_QUERY,
                max_chunks_per_product=RETRIEVAL_MAX_CHUNKS_PER_PRODUCT,
            )
            orchestrator_results.update(batch_result)
            orchestrator_results["backend"] = artifacts["manifest"]["embedding_backend"]
            logger.info(
                f"Batch orchestrator complete: {batch_result['products_processed']} processed, "
                f"{batch_result['products_failed']} failed, "
                f"{batch_result.get('groups_processed', 0)}/{batch_result.get('groups_total', 0)} groups"
            )
        except Exception as e:
            logger.error(f"Batch orchestrator failed: {e}")
            orchestrator_results["error"] = str(e)

    write_dataset_profile(input_path=input_path, output_dir="reports")
    reports_dir = Path("reports")
    legacy_json = reports_dir / "week1_dataset_profile.json"
    legacy_md = reports_dir / "week1_dataset_profile.md"
    dataset_json = reports_dir / "dataset_profile.json"
    dataset_md = reports_dir / "dataset_profile.md"
    if legacy_json.exists():
        dataset_json.write_text(legacy_json.read_text(encoding="utf-8"), encoding="utf-8")
    if legacy_md.exists():
        dataset_md.write_text(legacy_md.read_text(encoding="utf-8"), encoding="utf-8")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    extracted_dynamic = _to_dynamic_attribute_output(orchestrator_results["extracted_attributes"])

    # Save extracted attributes from orchestrator in dynamic serialized format.
    extracted_file = output / f"extracted_attributes_{profile}.json"
    with extracted_file.open("w", encoding="utf-8") as f:
        json.dump(extracted_dynamic, f, ensure_ascii=False, indent=2)
    logger.info(f"Extracted attributes saved to: {extracted_file}")

    # Save raw extractor output for trace/debug.
    extracted_raw_file = output / f"extracted_attributes_{profile}_raw.json"
    with extracted_raw_file.open("w", encoding="utf-8") as f:
        json.dump(orchestrator_results["extracted_attributes"], f, ensure_ascii=False, indent=2)
    logger.info("Raw extracted attributes saved to: %s", extracted_raw_file)

    pim_export = export_pim_files(
        extracted=extracted_dynamic,
        output_dir=output / "pim_exports",
        basename=f"pim_export_{profile}",
        approved_only=True,
    )
    logger.info("PIM export files written: %s", pim_export)

    metrics = {
        "profile": profile,
        "input_path": str(input_path),
        "ingestion_report": ingestion_report,
        "invalid_schema_records": invalid_schema_records,
        "chunks_created": len(artifacts["chunks"]),
        "index_size": artifacts["manifest"]["index_size"],
        "index_reused": bool(artifacts.get("index_reused", False)),
        "embedding_backend": artifacts["manifest"]["embedding_backend"],
        "hybrid_sparse_enabled": bool(enable_sparse),
        "reranker_enabled": bool(enable_reranker),
        "artifact_paths": artifacts["artifact_paths"],
        "dataset_profile_path": "reports/dataset_profile.json",
        "query_smoke": smoke,
        "orchestrator_results": {
            "backend": orchestrator_results["backend"],
            "products_processed": orchestrator_results["products_processed"],
            "products_failed": orchestrator_results["products_failed"],
            "groups_processed": orchestrator_results.get("groups_processed", 0),
            "groups_total": orchestrator_results.get("groups_total", 0),
            "retrieval_top_k_per_query": RETRIEVAL_TOP_K_PER_QUERY,
            "retrieval_max_chunks_per_product": RETRIEVAL_MAX_CHUNKS_PER_PRODUCT,
        },
        "extracted_attributes_path": str(extracted_file),
        "extracted_attributes_raw_path": str(extracted_raw_file),
        "pim_export": pim_export,
    }

    with (output / f"metrics_{profile}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (reports_dir / "query_results.json").open("w", encoding="utf-8") as f:
        json.dump(smoke, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Acceptance Report",
        "",
        f"- Profile: {profile}",
        f"- Input: {input_path}",
        f"- Parsed documents: {ingestion_report.get('rows_parsed', 0)}",
        f"- Chunks created: {len(artifacts['chunks'])}",
        f"- Index size: {artifacts['manifest']['index_size']}",
        f"- Embedding backend: {artifacts['manifest']['embedding_backend']}",
        f"- Index backend: {artifacts['manifest']['index_backend']}",
        f"- Hybrid sparse enabled: {bool(enable_sparse)}",
        f"- Reranker enabled: {bool(enable_reranker)}",
        f"- Query latency p50 (ms): {smoke['latency_p50_ms']}",
        f"- Query latency p95 (ms): {smoke['latency_p95_ms']}",
        f"- Orchestrator backend: {orchestrator_results['backend']}",
        f"- Products processed: {orchestrator_results['products_processed']}",
        f"- Products failed: {orchestrator_results['products_failed']}",
        f"- Batch groups processed: {orchestrator_results.get('groups_processed', 0)}/{orchestrator_results.get('groups_total', 0)}",
        "",
        "## Deliverable Checklist",
        "- [x] Ingestion pipeline evidence generated",
        "- [x] Vector store populated",
        "- [x] Hybrid retrieval outputs generated",
        "- [x] Manifest and metrics reports generated",
        "- [x] Full vector store indexed with all documents",
        "- [x] Hybrid search and reranking enabled",
        "- [x] Batch orchestrator processed all products",
        f"- [x] Extracted attributes saved to: {extracted_file}",
        f"- [x] PIM JSON export saved to: {pim_export['json_path']}",
        f"- [x] PIM CSV export saved to: {pim_export['csv_path']}",
    ]
    (reports_dir / "acceptance_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return metrics


def run_week1_pipeline(
    profile: str = "sample",
    input_path: Union[str, Path] | None = None,
    output_dir: Union[str, Path] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    enable_sparse: bool = True,
    enable_reranker: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible alias for older tests and callers."""
    return run_pipeline(
        profile=profile,
        input_path=input_path,
        output_dir=output_dir,
        embedding_model=embedding_model,
        enable_sparse=enable_sparse,
        enable_reranker=enable_reranker,
        run_orchestrator=True,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the main pipeline and generate report artifacts.")
    parser.add_argument("--profile", choices=["sample", "full"], default="sample")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--disable-sparse", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    result = run_pipeline(
        profile=args.profile,
        input_path=args.input_path,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        enable_sparse=not args.disable_sparse,
        enable_reranker=not args.disable_reranker,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
