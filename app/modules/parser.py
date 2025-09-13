from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Parser:
    """Clean and chunk raw texts into embedding-ready chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def parse(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        cleaned = [self._clean_text(t) for t in texts if t and t.strip()]
        concatenated = "\n\n".join(t for t in cleaned if t)
        if not concatenated:
            return []
        chunks = self.splitter.split_text(concatenated)
        return [c.strip() for c in chunks if c.strip()]

    def _clean_text(self, text: str) -> str:
        # Basic whitespace normalization and dedup of spaces
        text = " ".join(text.split())
        return text


