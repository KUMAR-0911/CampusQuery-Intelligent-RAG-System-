"""Hugging Face Sentence-Transformer embedding generation."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from config import RetrievalConfig


class SentenceTransformerEmbedder:
    """Reusable wrapper around a local Hugging Face embedding model."""
    def __init__(self, config: RetrievalConfig) -> None:
        """Load the configured Sentence Transformer once for batch embedding."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers with pip install -r requirements.txt") from exc
        self.config = config
        print(f"[embedding] Loading model: {config.embedding_model}")
        self.model = SentenceTransformer(config.embedding_model, token=config.hf_token)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create normalized vector embeddings for a sequence of texts."""
        print(f"[embedding] Generating embeddings for {len(texts)} text(s).")
        vectors = self.model.encode(list(texts), batch_size=self.config.embedding_batch_size, normalize_embeddings=self.config.normalize_embeddings, show_progress_bar=False)
        return vectors.tolist()

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed chunk text only; ``Document.metadata`` is never embedded."""
        texts = [document.page_content for document in documents]
        print("[embedding] Embedding chunk text only; metadata is reserved for filters.")
        return self.embed_texts(texts)
