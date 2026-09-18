"""High-level ingestion pipeline: Affinda Resume Parser → text → clean → hybrid chunk → enrich → embed → pgvector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from functools import lru_cache

from langchain_core.documents import Document

from config import DEFAULT_CONFIG, RetrievalConfig
from RETRIVAL.affinda_extractor import AffindaResumeParser
from RETRIVAL.chunking import build_text_splitter, chunk_text_documents
from RETRIVAL.cleaning import clean_documents
from RETRIVAL.embedding import SentenceTransformerEmbedder
from RETRIVAL.enrich_metadata import enrich_metadata
from RETRIVAL.pgvectorstore import PgVectorStore, make_vector_records
from retrieval_agent import RetrievalAgent


@lru_cache(maxsize=1)
def get_affinda_extractor(config: RetrievalConfig = DEFAULT_CONFIG) -> AffindaResumeParser:
    """Return singleton AffindaResumeParser."""
    print("[startup] Initializing Affinda Resume Parser...")
    parser = AffindaResumeParser(config)
    has_key = bool(parser.api_key and parser.api_key.strip())
    print(f"[startup] Affinda Resume Parser ready (API Key {'configured' if has_key else 'NOT set, will use local fallback'}).")
    return parser


@lru_cache(maxsize=1)
def get_hybrid_chunker(config: RetrievalConfig = DEFAULT_CONFIG):
    """Return singleton hybrid text splitter."""
    print("[startup] Pre-loading Hybrid Text Splitter...")
    splitter = build_text_splitter(config)
    print("[startup] Hybrid Text Splitter ready!")
    return splitter


@lru_cache(maxsize=1)
def get_vector_store(config: RetrievalConfig = DEFAULT_CONFIG) -> PgVectorStore:
    """Return singleton PgVectorStore with HF Inference API embedder and reranker."""
    print(f"[startup] Initializing PgVectorStore (reranker: {config.reranker_model} via HF API)...")
    return PgVectorStore(config)


def _prepare_documents(
    paths: str | Path | Iterable[str | Path], config: RetrievalConfig, user_id: str | None = None
) -> list[Document]:
    """Full ingestion flow per file:
      1. Affinda Resume Parser API → JSON → flat structured text
      2. Clean the text
      3. Hybrid chunk into Document objects
      4. Enrich metadata (heading, source, chunk_id, etc.)
    """
    sources = [paths] if isinstance(paths, (str, Path)) else list(paths)
    parser = get_affinda_extractor(config)
    splitter = get_hybrid_chunker(config)
    all_chunks: list[Document] = []

    print(f"[pipeline] Starting document ingestion for {len(sources)} file(s).")

    for source in sources:
        source_path = str(Path(source))
        source_name = Path(source).name
        print(f"\n[pipeline] ===== Processing: {source_name} =====")

        # Step 1: Affinda API → JSON → text (or local fallback → text)
        try:
            raw_text = parser.extract_text(source_path)
        except Exception as exc:
            print(f"[pipeline] Extraction failed for {source_name}: {exc}")
            continue

        if not raw_text or not raw_text.strip():
            print(f"[pipeline] Warning: No text extracted from {source_name}. Skipping.")
            continue

        print(f"[pipeline] Extracted {len(raw_text)} chars of text from '{source_name}'.")

        # Step 2: Wrap as a single Document for cleaning + chunking
        full_doc = Document(
            page_content=raw_text,
            metadata={
                "source": source_path,
                "source_file": source_name,
            },
        )

        # Step 3: Clean the text
        cleaned_docs = clean_documents([full_doc])
        if not cleaned_docs:
            print(f"[pipeline] Warning: Cleaning produced empty result for {source_name}. Skipping.")
            continue

        # Step 4: Hybrid chunk
        chunks = chunk_text_documents(cleaned_docs, config, splitter=splitter)
        print(f"[pipeline] Chunked '{source_name}' into {len(chunks)} chunk(s).")

        # Debug: print each chunk
        for i, chunk in enumerate(chunks):
            preview = chunk.page_content[:120].replace("\n", " ")
            print(f"[pipeline]   Chunk {i}: {len(chunk.page_content)} chars | Preview: {preview}...")

        all_chunks.extend(chunks)

    # Step 5: Tag with user_id
    if user_id:
        all_chunks = [
            Document(
                page_content=c.page_content,
                metadata={**c.metadata, "user_id": user_id},
            )
            for c in all_chunks
        ]

    # Step 6: Enrich metadata (heading, chunk_id, source_file, char/word counts)
    enriched = enrich_metadata(all_chunks)
    print(f"[pipeline] Total enriched chunks ready for embedding: {len(enriched)}")
    return enriched


def ingest_documents(
    paths: str | Path | Iterable[str | Path], config: RetrievalConfig = DEFAULT_CONFIG,
    user_id: str | None = None,
) -> list[dict]:
    """Convert, chunk, contextualise, enrich and embed input documents."""
    enriched_chunks = _prepare_documents(paths, config, user_id=user_id)
    embeddings = SentenceTransformerEmbedder(config).embed_documents(enriched_chunks)
    records = make_vector_records(enriched_chunks, embeddings)
    print("[pipeline] Ingestion completed successfully.")
    return records


def ingest_to_pgvector(
    paths: str | Path | Iterable[str | Path], config: RetrievalConfig = DEFAULT_CONFIG,
    user_id: str | None = None,
    store: PgVectorStore | None = None,
) -> PgVectorStore:
    """Ingest source files and persist their contextualised chunks in PostgreSQL pgvector."""
    print("[pipeline] Preparing PostgreSQL pgvector ingestion.")
    if not user_id or not user_id.strip():
        raise ValueError("A user ID is required before documents can be indexed.")
    documents = _prepare_documents(paths, config, user_id=user_id.strip())
    store = store or get_vector_store(config)
    store.upsert(documents)
    print("[pipeline] PostgreSQL pgvector ingestion completed.")
    return store


def answer_question(
    question: str,
    store: PgVectorStore,
    config: RetrievalConfig = DEFAULT_CONFIG,
    agent: RetrievalAgent | None = None,
    metadata_filters: dict | None = None,
    memory: object | None = None,
    user_id: str | None = None,
    conversation_context: dict | None = None,
) -> dict:
    """Let the cached LLM agent decide whether to retrieve, then answer."""
    print(f"[pipeline] Routing question: {question[:80]}")
    if agent is None:
        agent = getattr(store, "_retrieval_agent", None)
        if agent is None:
            agent = RetrievalAgent(config)
            store._retrieval_agent = agent
    answer = agent.answer(
        question,
        store,
        metadata_filters=metadata_filters,
        memory=memory,
        user_id=user_id,
        conversation_context=conversation_context,
    )
    print("[pipeline] Question answering completed.")
    return answer
