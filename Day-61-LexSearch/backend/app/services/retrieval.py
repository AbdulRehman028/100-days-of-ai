import re
from typing import Dict, List

from app.core.config import RRF_K
from app.services.embedding_service import EmbeddingService
from app.services.index_store import IndexStore
from app.services.text_utils import tokenize_for_search


class RetrievalService:
    def __init__(self, index_store: IndexStore, embedding_service: EmbeddingService) -> None:
        self.index_store = index_store
        self.embedding_service = embedding_service

    def _tokenize(self, text: str) -> List[str]:
        return tokenize_for_search(text)

    def _apply_file_filter(self, rows: List[Dict], file_name_contains: str | None) -> List[Dict]:
        if not file_name_contains:
            return rows
        needle = file_name_contains.strip().lower()
        if not needle:
            return rows
        return [row for row in rows if needle in row["file_name"].lower()]

    def _trim_dense_rows(self, rows: List[Dict], top_k: int) -> List[Dict]:
        if not rows:
            return rows
        top_score = float(rows[0]["score"])
        if top_score <= 0:
            return rows[:top_k]

        cutoff = max(top_score * 0.35, 0.08)
        trimmed = [row for row in rows if float(row["score"]) >= cutoff]
        return trimmed[:top_k]

    def _snippet(self, text: str, terms: List[str], span: int = 230) -> str:
        if not text:
            return ""
        lower = text.lower()
        pos = -1
        for t in terms:
            pos = lower.find(t)
            if pos != -1:
                break

        if pos == -1:
            return text[:span].strip() + ("..." if len(text) > span else "")

        start = max(0, pos - span // 2)
        end = min(len(text), start + span)
        out = text[start:end].strip()
        if start > 0:
            out = "..." + out
        if end < len(text):
            out = out + "..."
        return out

    def _highlight(self, snippet: str, terms: List[str]) -> str:
        if not snippet:
            return ""
        highlighted = snippet
        for term in sorted(set(terms), key=len, reverse=True):
            if len(term) < 2:
                continue
            highlighted = re.sub(
                rf"({re.escape(term)})",
                r"<mark>\1</mark>",
                highlighted,
                flags=re.IGNORECASE,
            )
        return highlighted

    def _format_results(self, rows: List[Dict], query_terms: List[str]) -> List[Dict]:
        formatted: List[Dict] = []
        for rank, row in enumerate(rows, start=1):
            snippet = self._snippet(row["text"], query_terms)
            highlighted = self._highlight(snippet, query_terms)
            formatted.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "file_name": row["file_name"],
                    "score": float(row["score"]),
                    "rank": rank,
                    "snippet": snippet,
                    "highlighted_snippet": highlighted,
                    "metadata": row["metadata"],
                }
            )
        return formatted

    def _rrf(self, dense_rows: List[Dict], sparse_rows: List[Dict], top_k: int) -> List[Dict]:
        scores: Dict[str, float] = {}
        payload: Dict[str, Dict] = {}

        for rank, row in enumerate(dense_rows, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            payload[cid] = row

        for rank, row in enumerate(sparse_rows, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in payload:
                payload[cid] = row

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                **payload[cid],
                "score": score,
            }
            for cid, score in ranked
        ]

    def search(self, query: str, top_k: int, mode: str, file_name_contains: str | None = None) -> Dict:
        query_terms = self._tokenize(query)

        dense_rows: List[Dict] = []
        sparse_rows: List[Dict] = []

        if mode in {"dense", "hybrid"}:
            q_emb = self.embedding_service.encode([query])
            if q_emb.size:
                dense_rows = self.index_store.dense_search(q_emb, top_k)
                dense_rows = self._trim_dense_rows(dense_rows, top_k)

        if mode in {"sparse", "hybrid"}:
            sparse_rows = self.index_store.sparse_search(query_terms, top_k)

        dense_rows = self._apply_file_filter(dense_rows, file_name_contains)
        sparse_rows = self._apply_file_filter(sparse_rows, file_name_contains)

        fused_rows = self._rrf(dense_rows, sparse_rows, top_k)

        return {
            "dense_results": self._format_results(dense_rows, query_terms),
            "sparse_results": self._format_results(sparse_rows, query_terms),
            "fused_results": self._format_results(fused_rows, query_terms),
        }
