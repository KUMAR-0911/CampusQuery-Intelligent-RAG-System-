"""
config.py — Central configuration for the CampusQuery RAG backend.

HOW IT WORKS
------------
1. Reads the `.env` file that sits next to this script.
2. Packs every setting into a single, immutable RetrievalConfig object.
3. The rest of the codebase imports `DEFAULT_CONFIG` and never touches
   `os.getenv` directly — so there is exactly ONE place to look when
   you need to change or debug a setting.

WHAT EACH SETTING CONTROLS
---------------------------
  qdrant_url / qdrant_api_key   →  Qdrant Cloud connection
  qdrant_collection             →  name of the Qdrant collection
  embedding_model               →  HuggingFace model used for embeddings
  groq_api_key / groq_model     →  Groq LLM used for answer generation
  chunk_size / chunk_overlap    →  how documents are split before embedding
    top_k                         →  how many chunks the retriever returns
    reranker_top_n                →  how many reranked chunks go to the LLM
    reranker_model / device       →  local CrossEncoder model and device
  system_prompt                 →  the instruction given to the LLM
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the .env file in this directory.
load_dotenv(Path(__file__).parent / ".env")

# Fallback system prompt used when none is set in .env.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful document question-answering assistant. "
    "Answer using the supplied document context. If it does not contain "
    "the answer, say that the documents do not provide enough information."
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Immutable bag of settings read once from environment variables.

    Frozen=True means nobody can accidentally overwrite a field at runtime,
    which prevents hard-to-trace bugs in a multi-module project.
    """

    # ── Qdrant Cloud ────────────────────────────────────────
    qdrant_url:        str = os.getenv("QDRANT_URL", "")
    qdrant_api_key:    str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "campusquery")

    # ── Embedding model ─────────────────────────────────────
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Groq LLM ────────────────────────────────────────────
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model:   str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Chunking ────────────────────────────────────────────
    chunk_size:    int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # ── Retrieval ───────────────────────────────────────────
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", os.getenv("TOP_K", "10")))
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "6"))
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")

    # ── LLM system prompt ──────────────────────────────────
    system_prompt: str = os.getenv(
        "CAMPUSQUERY_SYSTEM_PROMPT",
        os.getenv("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT),
    )

    # ── Validation ──────────────────────────────────────────
    def __post_init__(self) -> None:
        """Catch bad settings early, before they cause confusing errors later."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be a positive number.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
            )
        if self.top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be a positive number.")
        if self.reranker_top_n <= 0:
            raise ValueError("RERANKER_TOP_N must be a positive number.")
        if not self.reranker_model.strip():
            raise ValueError("RERANKER_MODEL must not be empty.")


# The single config instance used everywhere else.
DEFAULT_CONFIG = RetrievalConfig()
