"""Structure-aware text chunking with token/character limits for retrieval."""

from __future__ import annotations

from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RetrievalConfig
from RETRIVAL.cleaning import clean_text


def build_text_splitter(config: RetrievalConfig) -> RecursiveCharacterTextSplitter:
    """Build a recursive character text splitter scaled to embedding token chunk sizes."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size * 4,
        chunk_overlap=config.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text_documents(
    documents: list[Document],
    config: RetrievalConfig,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split documents into cleanly contextualized chunk pieces."""
    splitter = splitter or build_text_splitter(config)
    result: list[Document] = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        pieces = splitter.split_text(cleaned_content)
        for piece_idx, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta["split_index"] = piece_idx
            result.append(Document(page_content=piece, metadata=meta))
    return result
