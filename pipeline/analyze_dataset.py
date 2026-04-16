"""Dataset profiling for Week 1 structure sampling/classification evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ingestion.csv_loader import discover_csv_files, load_csv_files


def _column_as_str(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def build_dataset_profile(input_path: str | Path) -> Dict[str, Any]:
    files = discover_csv_files(input_path)
    df = load_csv_files(files)
    if df.empty:
        return {
            "source_files": [str(f) for f in files],
            "rows_total": 0,
            "document_type_counts": {},
            "document_subtype_counts": {},
            "language_guess_counts": {},
            "text_length_bins": {},
            "malformed_rows": {"empty_id_and_content": 0, "missing_required_fields": 0},
        }

    working = df.copy()
    working["id"] = _column_as_str(working, "id", "").str.strip()
    working["file_content"] = _column_as_str(working, "file_content", "")
    working["document_type"] = _column_as_str(working, "document_type", "unknown")
    working["document_subtype"] = _column_as_str(working, "document_subtype", "unknown")
    working["language"] = _column_as_str(working, "language", "")
    working["char_len"] = working["file_content"].str.len()

    empty_id_and_content = ((working["id"] == "") & (working["file_content"].str.strip() == "")).sum()
    missing_required = ((_column_as_str(working, "title", "").str.strip() == "") | (working["document_type"].str.strip() == "")).sum()

    bins = [0, 100, 500, 1000, 2000, 5000, 10000, 50000, 200000]
    labels = [
        "0-99",
        "100-499",
        "500-999",
        "1000-1999",
        "2000-4999",
        "5000-9999",
        "10000-49999",
        "50000+",
    ]
    working["length_bin"] = pd.cut(working["char_len"], bins=bins, labels=labels, right=False, include_lowest=True)
    length_counts = working["length_bin"].value_counts(dropna=False).to_dict()

    language_counts = working["language"].replace("", "unknown").value_counts().head(20).to_dict()

    return {
        "source_files": [str(f) for f in files],
        "rows_total": int(len(working)),
        "document_type_counts": {str(k): int(v) for k, v in working["document_type"].value_counts().to_dict().items()},
        "document_subtype_counts": {
            str(k): int(v) for k, v in working["document_subtype"].value_counts().to_dict().items()
        },
        "language_guess_counts": {str(k): int(v) for k, v in language_counts.items()},
        "text_length_bins": {str(k): int(v) for k, v in length_counts.items()},
        "malformed_rows": {
            "empty_id_and_content": int(empty_id_and_content),
            "missing_required_fields": int(missing_required),
        },
    }


def write_dataset_profile(input_path: str | Path, output_dir: str | Path = "reports") -> Dict[str, Any]:
    report = build_dataset_profile(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "week1_dataset_profile.json"
    md_path = out / "week1_dataset_profile.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines: List[str] = [
        "# Week 1 Dataset Profile",
        "",
        f"- Rows total: {report['rows_total']}",
        f"- Source files: {', '.join(report['source_files'])}",
        "",
        "## Document Type Counts",
    ]
    for key, value in report["document_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Document Subtype Counts"])
    for key, value in report["document_subtype_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Language Guess Counts"])
    for key, value in report["language_guess_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Text Length Bins"])
    for key, value in report["text_length_bins"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Malformed Row Patterns"])
    for key, value in report["malformed_rows"].items():
        lines.append(f"- {key}: {value}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report
