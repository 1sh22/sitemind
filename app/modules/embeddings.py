from __future__ import annotations

from typing import Any, List, Optional

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore[assignment]
import numpy as np
from sentence_transformers import SentenceTransformer


class Embeddings:
    """Embeds text chunks and builds an in-memory FAISS index."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Use huggingface hub path to avoid ambiguity
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype("float32")

    def build_index(self, chunks: List[str]) -> Any:
        if not chunks:
            # Return an empty, safe index
            return {"type": "bruteforce", "embeddings": np.zeros((0, 384), dtype="float32")}
        embeddings = self.encode(chunks)
        if faiss is not None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            return index
        # Fallback: store embeddings directly
        return {"type": "bruteforce", "embeddings": embeddings}

    def search(self, index: Any, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        if not chunks:
            return []
        qv = self.encode([query])  # shape (1, d)
        if faiss is not None and hasattr(index, "search"):
            scores, ids = index.search(qv, top_k)
            if ids.size == 0:
                return []
            id_list = ids[0]
            return [chunks[i] for i in id_list if 0 <= i < len(chunks)]

        # Fallback: brute-force cosine via dot (embeddings normalized)
        matrix = index.get("embeddings") if isinstance(index, dict) else None
        if matrix is None:
            return []
        if matrix.shape[0] == 0:
            return []
        sims = (matrix @ qv.T).reshape(-1)
        top_idx = np.argsort(-sims)[:top_k]
        return [chunks[i] for i in top_idx if 0 <= i < len(chunks)]


