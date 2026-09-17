"""FastEmbed embedding generation."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from config import RetrievalConfig


class SentenceTransformerEmbedder:
    """Reusable wrapper around FastEmbed embedding model."""
    def __init__(self, config: RetrievalConfig) -> None:
        """Load the configured FastEmbed model once for batch embedding."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError("Install fastembed with pip install -r requirements.txt") from exc
        self.config = config
        print(f"[embedding] Loading model: {config.embedding_model}")
        self.model = TextEmbedding(model_name=config.embedding_model)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create normalized vector embeddings for a sequence of texts."""
        print(f"[embedding] Generating embeddings for {len(texts)} text(s).")
        vectors_gen = self.model.embed(list(texts), batch_size=self.config.embedding_batch_size)
        vectors = list(vectors_gen)
        return [v.tolist() for v in vectors]

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed chunk text only; ``Document.metadata`` is never embedded."""
        texts = [document.page_content for document in documents]
        print("[embedding] Embedding chunk text only; metadata is reserved for filters.")
        return self.embed_texts(texts)
