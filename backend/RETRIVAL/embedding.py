"""FastEmbed or InferenceClient embedding generation."""

from __future__ import annotations

from typing import Sequence
import os

from langchain_core.documents import Document
from config import RetrievalConfig


class SentenceTransformerEmbedder:
    """Reusable wrapper around FastEmbed or HuggingFace API for embeddings."""
    def __init__(self, config: RetrievalConfig) -> None:
        """Initialize the embedder, using remote API if token is present, else local FastEmbed."""
        self.config = config
        self.is_remote = bool(config.hf_token)
        
        if self.is_remote:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise ImportError("Install huggingface_hub with pip install -r requirements.txt") from exc
            
            print(f"[embedding] Using remote Hugging Face Inference API for model: {config.embedding_model}")
            client_kwargs = {"model": config.embedding_model, "token": config.hf_token}
            provider = getattr(config, "hf_inference_provider", "featherless-ai")
            if provider:
                print(f"[embedding] Using inference provider: {provider}")
                client_kwargs["provider"] = provider
            self.client = InferenceClient(**client_kwargs)
        else:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError("Install fastembed with pip install -r requirements.txt") from exc
            
            print(f"[embedding] Loading local model: {config.embedding_model}")
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
            if cache_dir:
                self.model = TextEmbedding(model_name=config.embedding_model, cache_dir=cache_dir)
            else:
                self.model = TextEmbedding(model_name=config.embedding_model)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create normalized vector embeddings for a sequence of texts."""
        if not texts:
            return []
            
        print(f"[embedding] Generating embeddings for {len(texts)} text(s).")
        if self.is_remote:
            vectors = self.client.feature_extraction(list(texts)).tolist()
            if len(texts) == 1 and isinstance(vectors[0], float):
                 vectors = [vectors]
            return vectors
        else:
            vectors_gen = self.model.embed(list(texts), batch_size=self.config.embedding_batch_size)
            return [v.tolist() for v in vectors_gen]

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed chunk text only; ``Document.metadata`` is never embedded."""
        texts = [document.page_content for document in documents]
        print("[embedding] Embedding chunk text only; metadata is reserved for filters.")
        return self.embed_texts(texts)
