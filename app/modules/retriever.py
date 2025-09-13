from __future__ import annotations

from typing import List

import faiss

from .embeddings import Embeddings


class Retriever:
    """Lightweight wrapper over FAISS index search."""

    def retrieve(
        self,
        query: str,
        index: faiss.Index,
        embedder: Embeddings,
        chunks: List[str],
        top_k: int = 5,
    ) -> List[str]:
        return embedder.search(index, query, chunks, top_k=top_k)


