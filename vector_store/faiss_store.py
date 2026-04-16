"""FAISS index creation and persistence."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class NumpyIndexFlatIP:
    """Minimal FAISS-like fallback index for environments without faiss."""

    def __init__(self, dim: int) -> None:
        self.d = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.d:
            raise ValueError("vector dimension mismatch")
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            return (
                np.full((query.shape[0], k), -1.0, dtype=np.float32),
                np.full((query.shape[0], k), -1, dtype=np.int64),
            )
        scores = query @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, order, axis=1).astype(np.float32)
        return top_scores, order.astype(np.int64)


def build_faiss_index(vectors: np.ndarray) -> Any:
    """Build cosine-similarity FAISS index using normalized vectors."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be rank-2 array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    normalized = normalized.astype("float32")
    dim = normalized.shape[1]
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
    else:
        index = NumpyIndexFlatIP(dim)
    index.add(normalized)
    return index


def save_faiss_artifacts(
    index: Any,
    metadata: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    output_dir: Path | str,
) -> Dict[str, str]:
    """Persist index, metadata map, and manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / "baseline.index"
    metadata_path = out / "baseline_metadata.json"
    manifest_path = out / "baseline_manifest.json"

    if faiss is not None and hasattr(index, "__class__") and index.__class__.__module__.startswith("faiss"):
        faiss.write_index(index, str(index_path))
    else:
        with index_path.open("wb") as f:
            pickle.dump(index, f)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }


def load_faiss_artifacts(output_dir: Path | str) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    """Load persisted index + metadata + manifest."""
    out = Path(output_dir)
    index_path = out / "baseline.index"
    if faiss is not None:
        try:
            index = faiss.read_index(str(index_path))
        except Exception:
            with index_path.open("rb") as f:
                index = pickle.load(f)
    else:
        with index_path.open("rb") as f:
            index = pickle.load(f)
    with (out / "baseline_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with (out / "baseline_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, metadata, manifest
