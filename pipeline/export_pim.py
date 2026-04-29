"""CLI for exporting extracted attributes to PIM-compatible JSON and CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pim_export import export_pim_files


def load_extracted_attributes(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def export_from_file(
    input_path: str | Path,
    output_dir: str | Path = "data/processed/pim_exports",
    basename: str = "pim_export",
    approved_only: bool = True,
) -> dict[str, Any]:
    extracted = load_extracted_attributes(input_path)
    return export_pim_files(
        extracted=extracted,
        output_dir=output_dir,
        basename=basename,
        approved_only=approved_only,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export extracted product attributes for PIM import.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to extracted attributes JSON, e.g. data/processed/extracted_attributes_full.json",
    )
    parser.add_argument("--output-dir", default="data/processed/pim_exports")
    parser.add_argument("--basename", default="pim_export")
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Include attributes that are not APPROVED or EDITED.",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    result = export_from_file(
        input_path=args.input,
        output_dir=args.output_dir,
        basename=args.basename,
        approved_only=not args.include_pending,
    )
    print(json.dumps(result, indent=2))
