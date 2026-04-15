import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from ingestion.csv_loader import discover_csv_files, load_csv_files
from preprocessing.cleaner import clean_text
from preprocessing.language import detect_language, normalize_encoding
from preprocessing.parser import process_row


logger = logging.getLogger(__name__)


def _is_document_row(row: Dict[str, Any]) -> bool:
    """
    Return True only for rows that are genuine product documents.

    Skips rows where both the document id and the raw text content are empty –
    these are blank continuation rows that appear when multi-line CSV cells are
    exported from some tools and parsed naively.
    """
    doc_id = str(row.get("id") or row.get("doc_id") or "").strip()
    content = str(row.get("file_content") or row.get("raw_text") or "").strip()
    return bool(doc_id) or bool(content)


def _validate_parsed_document(document: Dict[str, Any]) -> bool:
    """Run lightweight validations required for downstream RAG readiness."""
    if not document.get("doc_id"):
        return False
    if not document.get("clean_text"):
        return False
    if len(document["clean_text"]) < 20:
        return False
    if "sections" not in document or "attributes_raw" not in document:
        return False
    return True


def run_pipeline(csv_input: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parse and normalise all product-document records from one CSV file or directory.

    Row filtering
    -------------
    Before parsing, rows that have no document id AND no file_content are skipped.
    This is important for the advanced-rag sample CSV which has ~3,500 blank
    continuation rows for every 100 real document rows.
    """
    file_paths = discover_csv_files(csv_input)
    if not file_paths:
        logger.warning("No CSV files found for input: %s", csv_input)
        return []

    df = load_csv_files(file_paths)
    if df.empty:
        logger.warning("CSV loading returned no data for input: %s", csv_input)
        return []

    total_rows = len(df)
    parsed_documents: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_invalid = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # ---- Skip empty / continuation rows early ----
        if not _is_document_row(row_dict):
            skipped_empty += 1
            continue

        parsed = process_row(row_dict, row_index=idx)

        cleaned_text = clean_text(parsed.get("text", ""))
        normalized_text = normalize_encoding(cleaned_text)
        language = detect_language(normalized_text)

        output: Dict[str, Any] = {
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
                "row_index": idx,
                "char_count": len(normalized_text),
                "section_count": len(parsed.get("sections", [])),
                "attribute_count": len(parsed.get("attributes_raw", {})),
            },
        }

        if _validate_parsed_document(output):
            parsed_documents.append(output)
        else:
            skipped_invalid += 1

    logger.info(
        "Parsing complete. total_rows=%s  empty_skipped=%s  parsed=%s  "
        "valid=%s  invalid=%s",
        total_rows,
        skipped_empty,
        total_rows - skipped_empty,
        len(parsed_documents),
        skipped_invalid,
    )
    return parsed_documents