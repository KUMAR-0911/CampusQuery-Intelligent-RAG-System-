"""
qdrantstore.py — Store and search document embeddings in Qdrant Cloud.

WHAT IS QDRANT?
---------------
Qdrant is a cloud-hosted vector database.  You upload vectors (lists of
numbers that represent the meaning of text), and Qdrant lets you quickly
find the ones most similar to a query vector.  This is the core of
"semantic search" in a RAG pipeline.

THIS MODULE PROVIDES
--------------------
QdrantVectorStore  — a class with two main methods:
    • upsert(documents)  → embed chunks and store them in Qdrant
    • search(query)      → embed the question and find the closest chunks

CONNECTION
----------
Set QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION in your `.env` file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

# Support imports when the backend directory is not the current working directory.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import RetrievalConfig
from langchain_core.documents import Document
from RETRIVAL.embedding import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Store document embeddings and search for similar chunks."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: SentenceTransformerEmbedder | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the embedder and Qdrant client."""
        if not config.qdrant_url or not config.qdrant_url.strip():
            raise ValueError("Qdrant URL is required.")
        if not config.qdrant_api_key or not config.qdrant_api_key.strip():
            raise ValueError("Qdrant API key is required.")

        self.config = config
        self.embedder = (
            embedder if embedder is not None else SentenceTransformerEmbedder(config)
        )
        self.client = client if client is not None else self._create_client(config)
        logger.info("Initialized Qdrant vector store for %s", config.qdrant_collection)

    @staticmethod
    def _create_client(config: RetrievalConfig) -> Any:
        """Create a Qdrant client using the configured credentials."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise ImportError(
                "The 'qdrant-client' package is required. Install it with "
                "'pip install qdrant-client'."
            ) from error

        return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)

    def preload(self) -> None:
        """Warm up the embedder and verify the Qdrant connection."""
        self.embedder.preload()
        self.client.get_collections()
        logger.info("Qdrant connection check passed")

    def upsert(self, documents: Sequence[Document]) -> None:
        """Embed and store chunks, replacing points with matching chunk IDs."""
        if not documents:
            logger.info("No document chunks to upsert")
            return

        self._validate_chunk_ids(documents)
        vectors = self.embedder.embed_documents(documents)
        self._validate_vectors(vectors, len(documents))
        self._ensure_collection(len(vectors[0]))

        from qdrant_client.http import models

        points = []
        for document, vector in zip(documents, vectors):
            metadata = dict(document.metadata)
            point_id = str(uuid5(NAMESPACE_URL, str(metadata["chunk_id"])))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": document.page_content,
                        "metadata": metadata,
                        "source_file": str(metadata.get("source_file", "")),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.config.qdrant_collection,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d document chunks", len(points))

    @staticmethod
    def _validate_chunk_ids(documents: Sequence[Document]) -> None:
        """Ensure every chunk has an ID for stable point replacement."""
        if any(not document.metadata.get("chunk_id") for document in documents):
            raise ValueError("Every document chunk must have metadata['chunk_id'].")

    @staticmethod
    def _validate_vectors(vectors: list[list[float]], expected_count: int) -> None:
        """Ensure embeddings contain one non-empty, consistent vector per chunk."""
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} vectors but got {len(vectors)}."
            )
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Embedder returned an empty vector.")

        vector_size = len(vectors[0])
        if any(len(vector) != vector_size for vector in vectors):
            raise RuntimeError("Embedder returned vectors with different sizes.")

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the collection with cosine distance if it is missing."""
        collection = self.config.qdrant_collection
        if self.client.collection_exists(collection):
            return

        from qdrant_client.http import models

        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection %s", collection)

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the closest chunks, optionally filtered by source file."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        limit = self.config.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be a positive number.")

        collection = self.config.qdrant_collection
        if not self.client.collection_exists(collection):
            logger.info("Qdrant collection %s does not exist", collection)
            return []

        query_vectors = self.embedder.embed_texts([query])
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise RuntimeError("Embedder returned an invalid query vector.")

        response = self.client.query_points(
            collection_name=collection,
            query=query_vectors[0],
            query_filter=self._build_filter(metadata_filters),
            limit=limit,
            with_payload=True,
        )
        return [self._format_result(point) for point in response.points]

    @staticmethod
    def _build_filter(metadata_filters: dict[str, Any] | None) -> Any | None:
        """Build a Qdrant filter for one or more source files."""
        if not metadata_filters or "source_file" not in metadata_filters:
            return None

        source_files = metadata_filters["source_file"]
        if source_files is None:
            return None
        values = source_files if isinstance(source_files, list) else [source_files]
        if not values:
            return None

        from qdrant_client.http import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchAny(any=[str(value) for value in values]),
                )
            ]
        )

    @staticmethod
    def _format_result(point: Any) -> dict[str, Any]:
        """Convert a Qdrant point into the public search-result shape."""
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        return {
            "id": str(point.id),
            "text": str(payload.get("text", "")),
            "metadata": metadata,
            "similarity_score": float(point.score or 0.0),
            "heading": metadata.get("heading", ""),
        }
