from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load one CSV file and return a DataFrame with source metadata."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return None

    try:
        df = pd.read_csv(path)
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
    frames: List[pd.DataFrame] = []
    for file_path in file_paths:
        df = load_csv(file_path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)