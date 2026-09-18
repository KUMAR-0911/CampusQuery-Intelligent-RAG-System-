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
        
        token = getattr(config, "hf_embedding_token", None) or config.hf_token
        if not token:
            raise ValueError("HF_EMBEDDING_TOKEN (or HF_TOKEN) is required — all embeddings run via HF Inference API.")
        
        from huggingface_hub import InferenceClient
        
        print(f"[embedding] Using remote Hugging Face Inference API for model: {config.embedding_model}")
        client_kwargs = {"model": config.embedding_model, "token": token}
        client_kwargs["provider"] = "hf-inference"
        self.client = InferenceClient(**client_kwargs)

    def preload(self) -> None:
        """Warm up the remote embedding client on startup."""
        try:
            _ = self.embed_texts(["ping"])
            print(f"[embedding] Remote embedding client preloaded & warmed up ({self.config.embedding_model}).")
        except Exception as exc:
            print(f"[embedding] Remote embedding preload notice: {exc}")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create normalized vector embeddings for a sequence of texts."""
        if not texts:
            return []
            
        text_list = list(texts)
        print(f"[embedding] Generating embeddings for {len(text_list)} text(s).")
        try:
            vectors = self.client.feature_extraction(text_list)
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if len(text_list) == 1 and isinstance(vectors[0], (float, int)):
                vectors = [vectors]
            return vectors
        except Exception as exc:
            # Fallback that bypasses numpy entirely using InferenceClient's direct response helper
            print(f"[embedding] Direct provider extraction fallback (reason: {exc})")
            try:
                from huggingface_hub.inference._providers import get_provider_helper
                helper = get_provider_helper(self.client.provider, task="feature-extraction", model=self.client.model)
                params = helper.prepare_request(
                    inputs=text_list,
                    parameters={},
                    headers=self.client.headers,
                    model=self.client.model,
                    api_key=self.client.token,
                )
                resp = self.client._inner_post(params)
                vectors = helper.get_response(resp)
                if len(text_list) == 1 and isinstance(vectors[0], (float, int)):
                    vectors = [vectors]
                return vectors
            except Exception as inner_exc:
                raise RuntimeError(f"Embedding feature extraction failed: {inner_exc}") from inner_exc

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed chunk text only; ``Document.metadata`` is never embedded."""
        texts = [document.page_content for document in documents]
        print("[embedding] Embedding chunk text only; metadata is reserved for filters.")
        return self.embed_texts(texts)
