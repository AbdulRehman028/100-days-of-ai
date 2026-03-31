import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import EMBED_MODEL_NAME


class EmbeddingService:
    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = None
        self.mode = "hash-fallback"

        try:
            self.model = SentenceTransformer(model_name)
            self.mode = "sentence-transformer"
        except Exception:
            # Keep service online even when model download/cache is unavailable.
            self.model = None

    def _hash_encode(self, texts: List[str], dim: int = 384) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        token_pattern = re.compile(r"[a-zA-Z0-9]+")

        for i, text in enumerate(texts):
            for token in token_pattern.findall(text.lower()):
                idx = hash(token) % dim
                vectors[i, idx] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.model is None:
            return self._hash_encode(texts)

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vectors.astype(np.float32)
