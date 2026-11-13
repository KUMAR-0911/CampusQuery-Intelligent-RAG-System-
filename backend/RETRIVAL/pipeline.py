"""
pipeline.py — The main orchestration module that ties everything together.

DATA FLOW (document ingestion)
──────────────────────────────
  File on disk
       │
       ▼
  DoclingDocumentLoader  →  raw text (Markdown)
       │
       ▼
  clean_documents()      →  normalised text
       │
       ▼
  chunk_text_documents() →  small text chunks (as Document objects)
       │
       ▼
  enrich_metadata()      →  chunks with IDs, headings, word counts
       │
       ▼
  QdrantVectorStore.upsert()  →  embedded and stored in Qdrant Cloud

DATA FLOW (question answering)
──────────────────────────────
  User question
       │
       ▼
  QdrantVectorStore.search()  →  top-K similar chunks
       │
       ▼
  RetrievalAgent.answer()     →  Groq LLM generates a grounded answer

SINGLETON PATTERN
-----------------
Heavy objects (the Docling loader, the text splitter, the Qdrant store)
are created once via `@lru_cache` and reused across requests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_CONFIG, RetrievalConfig
from RETRIVAL.chunking import build_text_splitter, chunk_text_documents
from RETRIVAL.cleaning import clean_documents
from RETRIVAL.document_loading import DoclingDocumentLoader
from RETRIVAL.enrich_metadata import enrich_metadata, extract_date_metadata
from RETRIVAL.qdrantstore import QdrantVectorStore
from retrieval_agent import RetrievalAgent

# Type alias for functions that accept one path or many.
DocumentPaths = str | Path | Iterable[str | Path]


# ═══════════════════════════════════════════════════════════════
#  Singletons — created once, reused for every request
# ═══════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def get_docling_extractor() -> DoclingDocumentLoader:
    """Return the shared Docling document loader (created once)."""
    print("[pipeline] Loading the Docling document extractor …")
    return DoclingDocumentLoader()


@lru_cache(maxsize=1)
def get_hybrid_chunker(
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> RecursiveCharacterTextSplitter:
    """Return the shared text splitter (created once)."""
    print("[pipeline] Loading the text chunker …")
    return build_text_splitter(config)


@lru_cache(maxsize=1)
def get_vector_store(
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> QdrantVectorStore:
    """Return the shared Qdrant Cloud vector store (created once)."""
    print("[pipeline] Connecting to Qdrant Cloud …")
    return QdrantVectorStore(config)


# ═══════════════════════════════════════════════════════════════
#  Document ingestion
# ═══════════════════════════════════════════════════════════════


def _prepare_documents(
    paths: DocumentPaths,
    config: RetrievalConfig,
) -> list[Document]:
    """Run the full extraction → cleaning → chunking → enrichment pipeline.

    Args:
        paths:  One file path, or a list / iterable of file paths.
        config: Application configuration.

    Returns:
        A list of enriched Document chunks, ready for embedding.
    """
    # Normalise input: always work with a list of paths.
    sources = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not sources:
        raise ValueError("You must provide at least one document path.")

    print(f"[pipeline] Starting ingestion for {len(sources)} file(s) …")

    loader   = get_docling_extractor()
    splitter = get_hybrid_chunker(config)
    all_chunks: list[Document] = []

    for source in sources:
        source_path = Path(source)
        print(f"[pipeline] Processing '{source_path.name}' …")

        # Step 1 — Extract text with Docling.
        raw_text = loader.extract_text(source_path)

        # Step 2 — Wrap as a Document for the rest of the pipeline.
        doc = Document(
            page_content=raw_text,
            metadata={
                "source":      str(source_path),
                "source_file": source_path.name,
                **extract_date_metadata(raw_text, source_path),
            },
        )

        # Step 3 — Clean whitespace, dividers, etc.
        cleaned = clean_documents([doc])

        # Step 4 — Split into small chunks.
        chunks = chunk_text_documents(cleaned, config, splitter=splitter)
        all_chunks.extend(chunks)

    # Step 5 — Add chunk IDs, headings, word counts.
    print(f"[pipeline] Enriching metadata for {len(all_chunks)} chunk(s) …")
    enriched = enrich_metadata(all_chunks)
    print("[pipeline] Document preparation complete.")
    return enriched


def ingest_to_qdrant(
    paths: DocumentPaths,
    config: RetrievalConfig = DEFAULT_CONFIG,
    store: QdrantVectorStore | None = None,
) -> QdrantVectorStore:
    """Prepare documents and upload their embeddings to Qdrant Cloud.

    This is the main entry point for document indexing.

    Args:
        paths:  File path(s) to ingest.
        config: Application configuration.
        store:  Optional pre-built vector store (uses the singleton if None).

    Returns:
        The QdrantVectorStore instance that was used.
    """
    print("[pipeline] Starting Qdrant document indexing …")

    documents = _prepare_documents(paths, config)
    vector_store = store if store is not None else get_vector_store(config)
    vector_store.upsert(documents)

    print(f"[pipeline] Indexed {len(documents)} chunk(s) in Qdrant.")
    return vector_store


# ═══════════════════════════════════════════════════════════════
#  Question answering
# ═══════════════════════════════════════════════════════════════


def answer_question(
    question: str,
    store: QdrantVectorStore,
    config: RetrievalConfig = DEFAULT_CONFIG,
    agent: RetrievalAgent | None = None,
    metadata_filters: dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer.

    Args:
        question:             The user's question in plain text.
        store:                The Qdrant vector store to search.
        config:               Application configuration.
        agent:                Optional pre-built RetrievalAgent.
        metadata_filters:     Optional filters (e.g. by source_file).
        conversation_context: Optional dict with recent chat messages.

    Returns:
        A dict with keys: answer, citations, retrieved, retrieved_chunks.
    """
    print("[pipeline] Starting question-answering …")

    # Lazily create an agent and cache it on the store object.
    if agent is None:
        agent = getattr(store, "_retrieval_agent", None)
        if agent is None:
            print("[pipeline] Creating the Groq retrieval agent …")
            agent = RetrievalAgent(config)
            store._retrieval_agent = agent  # type: ignore[attr-defined]

    result = agent.answer(
        question,
        store,
        metadata_filters=metadata_filters,
        conversation_context=conversation_context,
    )

    print("[pipeline] Question-answering complete.")
    return result
