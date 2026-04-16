from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

REQUIRED_COLUMNS = [
    "id",
    "title",
    "file_name",
    "file_content",
    "document_type",
    "product_id",
]


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Return True if dataframe contains all required columns."""
    if df is None or df.empty:
        return False

    actual_columns = set(df.columns)
    missing_columns = set(expected_columns) - actual_columns
    return not missing_columns


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load one CSV file and return a DataFrame with source metadata."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return None

    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["source_file"] = str(path)
        return df
    except Exception:
        return None


def discover_csv_files(input_path: Union[str, Path]) -> List[Path]:
    """Return sorted CSV files from a single file path or a directory path."""
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return []


def load_csv_files(file_paths: Iterable[Union[str, Path]]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files into one DataFrame."""
    frames, _ = load_csv_files_with_report(file_paths)
    return frames


def load_csv_files_with_report(
    file_paths: Iterable[Union[str, Path]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load/concat CSV files and return ingestion quality report."""
    frames: List[pd.DataFrame] = []
    files_seen = 0
    files_loaded = 0
    files_skipped_schema = 0
    files_skipped_read_error = 0
    rows_read = 0
    per_file_counts: Dict[str, int] = {}

    for file_path in file_paths:
        files_seen += 1
        df = load_csv(file_path)
        if df is None:
            files_skipped_read_error += 1
            continue

        if not validate_schema(df, REQUIRED_COLUMNS):
            files_skipped_schema += 1
            continue

        if not df.empty:
            frames.append(df)
            files_loaded += 1
            rows_read += int(len(df))
            per_file_counts[str(file_path)] = int(len(df))

    if not frames:
        return pd.DataFrame(), {
            "files_seen": files_seen,
            "files_loaded": files_loaded,
            "files_skipped_schema": files_skipped_schema,
            "files_skipped_read_error": files_skipped_read_error,
            "rows_read": rows_read,
            "per_file_counts": per_file_counts,
        }

    merged = pd.concat(frames, ignore_index=True)
    return merged, {
        "files_seen": files_seen,
        "files_loaded": files_loaded,
        "files_skipped_schema": files_skipped_schema,
        "files_skipped_read_error": files_skipped_read_error,
        "rows_read": rows_read,
        "per_file_counts": per_file_counts,
    }
