"""
embedding.py — Turn text into numeric vectors using Sentence Transformers.

HOW EMBEDDING WORKS (plain English)
------------------------------------
An "embedding" is a list of numbers (a vector) that captures the *meaning*
of a piece of text.  Two texts with similar meanings will have vectors
that are close together — which is how the retriever finds relevant chunks
when you ask a question.

We use the `sentence-transformers` library, which downloads a small
neural-network model and runs it locally on your CPU.  The default model
is `all-MiniLM-L6-v2` (≈80 MB, produces 384-dimensional vectors).

USAGE
-----
    from RETRIVAL.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(config)
    vectors  = embedder.embed_texts(["Hello world", "Another sentence"])
    # vectors → [[0.012, -0.34, …], [0.07, 0.21, …]]
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from config import RetrievalConfig


class SentenceTransformerEmbedder:
    """Create document and query embeddings with a local HuggingFace model."""

    def __init__(self, config: RetrievalConfig) -> None:
        """Download (if needed) and load the configured embedding model.

        Args:
            config: Application settings — only `embedding_model` is used.

        Raises:
            ImportError: If `sentence-transformers` is not installed.
        """
        self.config = config

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "The 'sentence-transformers' package is required.\n"
                "Install it with:  pip install sentence-transformers"
            ) from error

        print(f"[embedding] Loading model '{config.embedding_model}' …")
        self.model = SentenceTransformer(config.embedding_model)
        print(f"[embedding] Model '{config.embedding_model}' is ready.")

    # ── Warm-up ─────────────────────────────────────────────────

    def preload(self) -> None:
        """Encode a tiny dummy text so the first real call is fast."""
        print(f"[embedding] Warming up '{self.config.embedding_model}' …")
        self.embed_texts(["ping"])
        print(f"[embedding] Warm-up complete.")

    # ── Core: text → vectors ────────────────────────────────────

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Turn a list of plain-text strings into normalised float vectors.

        Args:
            texts: One or more strings to embed.

        Returns:
            A list of vectors (each vector is a list of floats).
            The list has the same length as the input.

        Raises:
            ValueError: If the model returns an unexpected number of
                        vectors, an empty vector, or mixed-size vectors.
        """
        if not texts:
            print("[embedding] Nothing to embed — empty input.")
            return []

        print(f"[embedding] Encoding {len(texts)} text(s) …")

        # `normalize_embeddings=True` makes cosine similarity work correctly.
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Convert numpy arrays → plain Python lists.
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        # When there is only one input text, some models return a flat
        # list instead of a list-of-lists.  Fix that here.
        if embeddings and isinstance(embeddings[0], (float, int)):
            embeddings = [embeddings]

        # ── Safety checks ───────────────────────────────────────
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} vectors but got {len(embeddings)}."
            )

        vectors = [[float(v) for v in vec] for vec in embeddings]

        if any(not vec for vec in vectors):
            raise ValueError("The model returned an empty vector.")
        if any(len(vec) != len(vectors[0]) for vec in vectors):
            raise ValueError("The model returned vectors with different sizes.")

        print(f"[embedding] Produced {len(vectors)} vector(s), "
              f"each {len(vectors[0])} dimensions.")
        return vectors

    # ── Convenience: Document list → vectors ────────────────────

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Embed only the text of each Document (metadata is NOT embedded).

        Args:
            documents: LangChain Document objects.

        Returns:
            One vector per document, in the same order.
        """
        print(f"[embedding] Preparing {len(documents)} document chunk(s) …")
        return self.embed_texts([doc.page_content for doc in documents])
