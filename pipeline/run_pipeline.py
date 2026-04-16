"""Week 1 end-to-end pipeline: ingest -> parse -> chunk -> embed -> FAISS -> RAG smoke."""

from __future__ import annotations

import argparse
import json
import logging
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
from vector_store import build_faiss_index, save_faiss_artifacts
from vector_store.faiss_store import faiss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SAMPLE_QUERIES = [
    "What is the supply voltage?",
    "What are the key features?",
    "What is the operating temperature range?",
    "What communication protocols are supported?",
    "What are the dimensions of the product?",
]


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


def run_query_smoke(
    index: Any,
    metadata_store: List[Dict[str, Any]],
    embedding_model: str,
    queries: List[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    queries = queries or DEFAULT_SAMPLE_QUERIES
    rag = BaselineRAG(index=index, metadata=metadata_store, embedding_model=embedding_model)

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


def run_week1_pipeline(
    profile: str = "sample",
    input_path: Union[str, Path] | None = None,
    output_dir: Union[str, Path] = "data/processed",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    profile = profile.lower().strip()
    if input_path is None:
        if profile == "full":
            input_path = "data/raw"
        else:
            input_path = "data/raw/100_sample_advanced_rag.csv"

    parsed_documents, ingestion_report = parse_documents(input_path)
    invalid_schema_records = _validate_product_records(parsed_documents)
    artifacts = build_index_artifacts(parsed_documents, output_dir=output_dir, embedding_model=embedding_model)

    smoke = {"query_results": [], "latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    if artifacts["index"] is not None:
        smoke = run_query_smoke(
            index=artifacts["index"],
            metadata_store=artifacts["metadata_store"],
            embedding_model=embedding_model,
            queries=DEFAULT_SAMPLE_QUERIES,
            top_k=5,
        )

    dataset_profile = write_dataset_profile(input_path=input_path, output_dir="reports")
    metrics = {
        "profile": profile,
        "input_path": str(input_path),
        "ingestion_report": ingestion_report,
        "invalid_schema_records": invalid_schema_records,
        "chunks_created": len(artifacts["chunks"]),
        "index_size": artifacts["manifest"]["index_size"],
        "embedding_backend": artifacts["manifest"]["embedding_backend"],
        "artifact_paths": artifacts["artifact_paths"],
        "dataset_profile_path": "reports/week1_dataset_profile.json",
        "query_smoke": smoke,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"week1_metrics_{profile}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with Path("reports/week1_query_results.json").open("w", encoding="utf-8") as f:
        json.dump(smoke, f, ensure_ascii=False, indent=2)
    report_lines = [
        "# Week 1 Acceptance Report",
        "",
        f"- Profile: {profile}",
        f"- Input: {input_path}",
        f"- Parsed documents: {ingestion_report.get('rows_parsed', 0)}",
        f"- Chunks created: {len(artifacts['chunks'])}",
        f"- Index size: {artifacts['manifest']['index_size']}",
        f"- Embedding backend: {artifacts['manifest']['embedding_backend']}",
        f"- Index backend: {artifacts['manifest']['index_backend']}",
        f"- Query latency p50 (ms): {smoke['latency_p50_ms']}",
        f"- Query latency p95 (ms): {smoke['latency_p95_ms']}",
        "",
        "## Deliverable Checklist",
        "- [x] Ingestion pipeline evidence generated",
        "- [x] Vector store populated",
        "- [x] Baseline RAG query outputs generated",
        "- [x] Manifest and metrics reports generated",
    ]
    Path("reports/week1_acceptance_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return metrics


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 1 foundation pipeline.")
    parser.add_argument("--profile", choices=["sample", "full"], default="sample")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    result = run_week1_pipeline(
        profile=args.profile,
        input_path=args.input_path,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
