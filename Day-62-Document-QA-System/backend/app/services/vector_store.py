import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "metadata.json"

        self.vectors: np.ndarray | None = None
        self.metadata: List[Dict[str, str]] = []

        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.vectors = np.load(self.index_path)
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save(self) -> None:
        if self.vectors is not None:
            np.save(self.index_path, self.vectors)
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=True, indent=2)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @property
    def is_empty(self) -> bool:
        return self.vectors is None or self.vectors.shape[0] == 0

    @property
    def total_chunks(self) -> int:
        return 0 if self.vectors is None else int(self.vectors.shape[0])

    def add(self, vectors: List[List[float]], metadata_rows: List[Dict[str, str]]) -> None:
        if not vectors:
            return

        np_vectors = np.asarray(vectors, dtype=np.float32, order="C")
        np_vectors = self._normalize(np_vectors)

        if self.vectors is None:
            self.vectors = np_vectors
        else:
            if self.vectors.shape[1] != np_vectors.shape[1]:
                raise ValueError("Embedding dimension mismatch with existing index.")
            self.vectors = np.vstack([self.vectors, np_vectors])

        self.metadata.extend(metadata_rows)
        self._save()

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, str]]:
        if self.is_empty:
            return []

        query = np.asarray([query_vector], dtype=np.float32, order="C")
        query = self._normalize(query)

        scores = np.matmul(self.vectors, query[0])
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = dict(self.metadata[idx])
            row["score"] = float(scores[idx])
            results.append(row)

        return results
