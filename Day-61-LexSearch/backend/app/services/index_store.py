import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import INDEX_DIR
from app.services.text_utils import tokenize_for_search


class IndexStore:
    """Persistent store for chunks + dense/sparse indexes."""

    def __init__(self) -> None:
        self.chunks_path = INDEX_DIR / "chunks.json"
        self.embeddings_path = INDEX_DIR / "embeddings.npy"

        self.chunks: List[Dict[str, Any]] = []
        self._chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

        self._faiss_index: faiss.IndexFlatIP | None = None
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: List[List[str]] = []
        self._dim: int | None = None

        self._load()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (vectors / norms).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        return tokenize_for_search(text)

    def _build_sparse_index(self) -> None:
        self._tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def _build_dense_index(self) -> None:
        if self._embeddings.size == 0:
            self._faiss_index = None
            self._dim = None
            return

        self._dim = int(self._embeddings.shape[1])
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._faiss_index.add(self._normalize(self._embeddings))

    def _load(self) -> None:
        if self.chunks_path.exists():
            self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self._chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        if self.embeddings_path.exists():
            loaded = np.load(self.embeddings_path)
            self._embeddings = loaded.astype(np.float32)
        else:
            self._embeddings = np.empty((0, 0), dtype=np.float32)

        if self.chunks and self._embeddings.size and len(self.chunks) == len(self._embeddings):
            self._build_dense_index()
        elif self.chunks and self._embeddings.size and len(self.chunks) != len(self._embeddings):
            min_len = min(len(self.chunks), len(self._embeddings))
            self.chunks = self.chunks[:min_len]
            self._embeddings = self._embeddings[:min_len]
            self._chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
            self._build_dense_index()
        else:
            self._faiss_index = None

        self._build_sparse_index()

    def _persist(self) -> None:
        self.chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._embeddings.size:
            np.save(self.embeddings_path, self._embeddings)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks count and embeddings count mismatch")

        embeddings = embeddings.astype(np.float32)

        if self._embeddings.size == 0:
            self._embeddings = embeddings
        else:
            if self._embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch")
            self._embeddings = np.vstack([self._embeddings, embeddings]).astype(np.float32)

        self.chunks.extend(chunks)
        for c in chunks:
            self._chunk_by_id[c["chunk_id"]] = c

        self._build_dense_index()
        self._build_sparse_index()
        self._persist()

    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self._faiss_index is None or not self.chunks:
            return []

        query_embedding = self._normalize(query_embedding.astype(np.float32))
        scores, indices = self._faiss_index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "file_name": chunk["file_name"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(score),
                    "rank": rank,
                }
            )
        return results

    def sparse_search(self, query_terms: List[str], top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self.chunks:
            return []

        scores = self._bm25.get_scores(query_terms)
        ranked = np.argsort(scores)[::-1]
        has_positive = bool(np.any(scores > 0))

        results: List[Dict[str, Any]] = []
        rank = 1
        for idx in ranked.tolist():
            score = float(scores[idx])
            if has_positive and score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "file_name": chunk["file_name"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": score,
                    "rank": rank,
                }
            )
            if len(results) >= top_k:
                break
            rank += 1
        return results

    def stats(self) -> Dict[str, int]:
        docs = {c["doc_id"] for c in self.chunks}
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
        }
