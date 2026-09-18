"""HuggingFace Inference API embedding generation."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from config import RetrievalConfig


class SentenceTransformerEmbedder:
    """Wrapper around HuggingFace Inference API for embeddings."""
    def __init__(self, config: RetrievalConfig) -> None:
        """Initialize the embedder using HF Inference API."""
        self.config = config
        
        if not config.hf_token:
            raise ValueError("HF_TOKEN is required — all embeddings run via HF Inference API.")
        
        from huggingface_hub import InferenceClient
        
        print(f"[embedding] Using remote Hugging Face Inference API for model: {config.embedding_model}")
        client_kwargs = {"model": config.embedding_model, "token": config.hf_token}
        client_kwargs["provider"] = "hf-inference"
        self.client = InferenceClient(**client_kwargs)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create normalized vector embeddings for a sequence of texts."""
        if not texts:
            return []
            
        print(f"[embedding] Generating embeddings for {len(texts)} text(s).")
        vectors = self.client.feature_extraction(list(texts)).tolist()
        if len(texts) == 1 and isinstance(vectors[0], float):
             vectors = [vectors]
        return vectors

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed chunk text only; ``Document.metadata`` is never embedded."""
        texts = [document.page_content for document in documents]
        print("[embedding] Embedding chunk text only; metadata is reserved for filters.")
        return self.embed_texts(texts)
