"""
chunking.py — Split cleaned documents into smaller pieces for embedding.

WHY CHUNK?
----------
LLMs and embedding models have limited context windows.  Splitting a
20-page PDF into ~500-token chunks lets us embed and retrieve only the
most relevant *part* of a document, instead of feeding the whole thing
to the model.

HOW IT WORKS
------------
We use LangChain's RecursiveCharacterTextSplitter, which tries to break
text at natural boundaries (double newlines → single newlines → sentences
→ words) before falling back to a hard character limit.

The `chunk_size` in RetrievalConfig is in *tokens*.  We approximate
4 characters per token, so a 500-token chunk ≈ 2 000 characters.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RetrievalConfig


def build_text_splitter(config: RetrievalConfig) -> RecursiveCharacterTextSplitter:
    """Create a text splitter configured with the project's chunk settings.

    Args:
        config: The application configuration object.

    Returns:
        A ready-to-use RecursiveCharacterTextSplitter.
    """
    # Approximate: 1 token ≈ 4 characters.
    char_chunk_size = config.chunk_size * 4
    char_overlap    = config.chunk_overlap * 4

    print(
        f"[chunking] Chunk size = {config.chunk_size} tokens "
        f"(≈{char_chunk_size} chars), "
        f"overlap = {config.chunk_overlap} tokens."
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text_documents(
    documents: list[Document],
    config: RetrievalConfig,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split a list of documents into smaller chunks.

    Each output chunk carries all of the original document's metadata
    plus a `split_index` field showing its position within the source.

    Args:
        documents: Cleaned documents to split.
        config:    Application configuration (used to build a default splitter).
        splitter:  Optional pre-built splitter (avoids re-creating one).

    Returns:
        A flat list of Document chunks.
    """
    text_splitter = splitter or build_text_splitter(config)
    chunks: list[Document] = []

    for doc in documents:
        pieces = text_splitter.split_text(doc.page_content)
        filename = doc.metadata.get("source_file", "document")
        print(f"[chunking] '{filename}' → {len(pieces)} chunk(s).")

        for index, piece in enumerate(pieces):
            metadata = {**doc.metadata, "split_index": index}
            chunks.append(Document(page_content=piece, metadata=metadata))

    print(f"[chunking] Total chunks created: {len(chunks)}.")
    return chunks
