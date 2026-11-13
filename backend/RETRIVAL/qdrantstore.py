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

from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from config import RetrievalConfig
from RETRIVAL.embedding import SentenceTransformerEmbedder


class QdrantVectorStore:
    """Store and search document chunks in a Qdrant Cloud collection."""

    # ── Initialisation ──────────────────────────────────────────

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: SentenceTransformerEmbedder | None = None,
        client: Any | None = None,
    ) -> None:
        """Connect to Qdrant Cloud and prepare the embedder.

        Args:
            config:   Application settings (must include Qdrant credentials).
            embedder: Optional pre-built embedder (useful for testing).
            client:   Optional pre-built QdrantClient (useful for testing).

        Raises:
            ValueError:   If QDRANT_URL or QDRANT_API_KEY is missing.
            ImportError:  If the qdrant-client package is not installed.
        """
        if not config.qdrant_url:
            raise ValueError("Set QDRANT_URL in the backend .env file.")
        if not config.qdrant_api_key:
            raise ValueError("Set QDRANT_API_KEY in the backend .env file.")

        print(f"[qdrant] Connecting to collection '{config.qdrant_collection}' …")

        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise ImportError(
                "The 'qdrant-client' package is required.\n"
                "Install it with:  pip install qdrant-client"
            ) from error

        self.config = config

        # Build (or reuse) the embedding model and the Qdrant client.
        self.embedder = embedder or SentenceTransformerEmbedder(config)
        self.client = client or QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

        print("[qdrant] Vector store is ready.")

    # ── Health check ────────────────────────────────────────────

    def preload(self) -> None:
        """Verify both the embedder and the Qdrant connection work."""
        print("[qdrant] Running a connection check …")
        self.embedder.preload()
        self.client.get_collections()
        print("[qdrant] Connection check passed.")

    # ── Write: index document chunks ────────────────────────────

    def upsert(self, documents: Sequence[Document]) -> None:
        """Embed and upload document chunks to Qdrant.

        Each chunk must have a `chunk_id` in its metadata (added by
        `enrich_metadata`).  The chunk_id is hashed into a UUID so that
        re-indexing the same document updates existing points instead of
        creating duplicates.

        Args:
            documents: LangChain Document objects (already chunked + enriched).

        Raises:
            ValueError:    If any chunk is missing a chunk_id.
            RuntimeError:  If the embedder returns unexpected results.
        """
        if not documents:
            print("[qdrant] Nothing to index — empty document list.")
            return

        print(f"[qdrant] Preparing {len(documents)} chunk(s) for indexing …")
        from qdrant_client.http import models

        # ── Validate ────────────────────────────────────────
        if any(not doc.metadata.get("chunk_id") for doc in documents):
            raise ValueError(
                "Every document chunk must have a 'chunk_id' in its metadata. "
                "Run enrich_metadata() before calling upsert()."
            )

        # ── Embed ───────────────────────────────────────────
        vectors = self.embedder.embed_documents(documents)
        self._validate_vectors(vectors, expected_count=len(documents))

        # ── Create the collection if it doesn't exist yet ───
        vector_size = len(vectors[0])
        self._ensure_collection(vector_size)

        # ── Build Qdrant point objects ──────────────────────
        points = []
        for doc, vector in zip(documents, vectors):
            metadata = dict(doc.metadata)
            # Convert the chunk_id into a deterministic UUID.
            point_id = str(uuid5(NAMESPACE_URL, str(metadata["chunk_id"])))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text":        doc.page_content,
                        "metadata":    metadata,
                        "source_file": str(metadata.get("source_file", "")),
                    },
                )
            )

        # ── Upload ──────────────────────────────────────────
        self.client.upsert(
            collection_name=self.config.qdrant_collection,
            points=points,
            wait=True,  # Block until Qdrant confirms the write.
        )
        print(f"[qdrant] Indexed {len(points)} point(s) successfully.")

    # ── Read: semantic search ───────────────────────────────────

    def search(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find the most similar chunks to a natural-language query.

        Args:
            query:            The user's question as plain text.
            metadata_filters: Optional dict to filter by source_file, etc.
            top_k:            How many results to return (default from config).

        Returns:
            A list of dicts, each containing:
              id, text, metadata, similarity_score, heading
        """
        collection = self.config.qdrant_collection
        print(f"[qdrant] Searching collection '{collection}' …")

        # If the collection hasn't been created yet, there's nothing to search.
        if not self.client.collection_exists(collection):
            print("[qdrant] Collection does not exist yet — returning empty.")
            return []

        limit = top_k if top_k is not None else self.config.top_k
        if limit <= 0:
            raise ValueError("top_k must be a positive number.")

        # Embed the query into a vector.
        query_vector = self.embedder.embed_texts([query])
        if not query_vector or not query_vector[0]:
            raise RuntimeError("Embedder returned an empty query vector.")

        # Ask Qdrant for the nearest points.
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector[0],
            query_filter=self._build_filter(metadata_filters),
            limit=limit,
            with_payload=True,
        )

        print(f"[qdrant] Found {len(response.points)} matching point(s).")

        # ── Format results ──────────────────────────────────
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload    = point.payload or {}
            metadata   = payload.get("metadata") or {}
            similarity = float(point.score or 0.0)

            results.append({
                "id":               str(point.id),
                "text":             str(payload.get("text", "")),
                "metadata":         metadata,
                "similarity_score": similarity,
                "rerank_score":     similarity,   # kept for backward compat
                "raw_score":        similarity,   # kept for backward compat
                "heading":          metadata.get("heading", ""),
            })

        return results

    # ── Private helpers ─────────────────────────────────────────

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        from qdrant_client.http import models

        if self.client.collection_exists(self.config.qdrant_collection):
            print(f"[qdrant] Collection '{self.config.qdrant_collection}' "
                  f"already exists.")
            return

        print(
            f"[qdrant] Creating collection '{self.config.qdrant_collection}' "
            f"(vector size = {vector_size}) …"
        )
        self.client.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _build_filter(metadata_filters: dict[str, Any] | None):
        """Convert a simple metadata dict into a Qdrant Filter object.

        Currently supports filtering by `source_file` only.
        Returns None if no filters are provided.
        """
        if not metadata_filters:
            return None

        from qdrant_client.http import models

        conditions = []
        for key in ("source_file",):
            value = metadata_filters.get(key)
            if not value:
                continue

            # Normalise to a list so we can use MatchAny.
            values = list(value) if isinstance(value, (list, tuple, set)) else [value]
            if not values:
                continue

            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=[str(v) for v in values]),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]], expected_count: int
    ) -> None:
        """Raise RuntimeError if the vectors look wrong."""
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} vectors but got {len(vectors)}."
            )
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedder returned an empty vector.")
        size = len(vectors[0])
        if any(len(v) != size for v in vectors):
            raise RuntimeError("Embedder returned vectors with mixed sizes.")
