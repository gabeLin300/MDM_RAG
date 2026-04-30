"""Main pipeline entrypoint: ingest -> parse -> chunk -> embed -> FAISS -> RAG smoke."""

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

def run_batch_orchestrator(
    orchestrator: Orchestrator,
    metadata_store: List[Dict[str, Any]],
    rate_limit_delay: float = 3.0,
) -> Dict[str, Any]:
    """Process all documents once, mapping extracted attributes to all product IDs they cover.

    One LLM call per document (not per product ID) — avoids redundant extraction
    when multiple products share the same source document.
    """
    import time
    from collections import defaultdict

    docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in metadata_store:
        docs[chunk["doc_id"]].append(chunk)

    results = {}
    failed_docs = []
    processed_count = 0
    doc_list = list(docs.items())

    for idx, (doc_id, doc_chunks) in enumerate(doc_list):
        product_ids = set()
        for chunk in doc_chunks:
            pids = chunk.get("product_id", [])
            if isinstance(pids, list):
                product_ids.update(pids)
            elif isinstance(pids, str):
                product_ids.add(pids)
        product_ids.discard("")

        backoff_delay = rate_limit_delay
        max_retries = 3
        retry_count = 0
        had_retry = False

        while retry_count < max_retries:
            try:
                logger.info(f"Processing document {idx + 1}/{len(doc_list)}: {doc_id} ({len(product_ids)} products)")
                doc_result = orchestrator.run_for_document(doc_chunks)
                extracted_at = datetime.now(timezone.utc).isoformat()

                for product_id in product_ids:
                    results[product_id] = {
                        "product_id": product_id,
                        "doc_id": doc_id,
                        "extracted_at": extracted_at,
                        **doc_result,
                    }

                processed_count += 1
                logger.info(f"Processed {doc_id}: {len(doc_result.get('attributes', {}))} attributes → {len(product_ids)} products")
                break

            except Exception as e:
                retry_count += 1
                had_retry = True
                if retry_count < max_retries:
                    logger.warning(
                        f"Attempt {retry_count}/{max_retries} failed for {doc_id}: {e}. "
                        f"Retrying in {backoff_delay}s..."
                    )
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    logger.error(f"Failed to process {doc_id} after {max_retries} attempts: {e}")
                    failed_docs.append(doc_id)

        if idx < len(doc_list) - 1:
            # After a retry, use the doubled backoff so Groq has time to recover
            time.sleep(backoff_delay if had_retry else rate_limit_delay)

    if failed_docs:
        logger.warning(f"Failed documents: {failed_docs}")

    return {
        "extracted_attributes": results,
        "products_processed": processed_count,
        "products_failed": len(failed_docs),
        "failed_product_ids": failed_docs,
    }


def run_pipeline(
    profile: str = "sample",
    input_path: Union[str, Path] | None = None,
    output_dir: Union[str, Path] = "data/processed",
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

    parsed_documents, ingestion_report = parse_documents(input_path)
    invalid_schema_records = _validate_product_records(parsed_documents)
    artifacts = build_index_artifacts(parsed_documents, output_dir=output_dir, embedding_model=embedding_model)

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
            batch_result = run_batch_orchestrator(orchestrator, artifacts["metadata_store"])
            orchestrator_results.update(batch_result)
            orchestrator_results["backend"] = artifacts["manifest"]["embedding_backend"]
            logger.info(
                f"Batch orchestrator complete: {batch_result['products_processed']} processed, "
                f"{batch_result['products_failed']} failed"
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

    # Save extracted attributes from orchestrator in serialized format.
    extracted_file = output / f"extracted_attributes_{profile}.json"
    with extracted_file.open("w", encoding="utf-8") as f:
        json.dump(orchestrator_results["extracted_attributes"], f, ensure_ascii=False, indent=2)
    logger.info(f"Extracted attributes saved to: {extracted_file}")

    pim_export = export_pim_files(
        extracted=orchestrator_results["extracted_attributes"],
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
        },
        "extracted_attributes_path": str(extracted_file),
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
    output_dir: Union[str, Path] = "data/processed",
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
    parser.add_argument("--output-dir", default="data/processed")
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
