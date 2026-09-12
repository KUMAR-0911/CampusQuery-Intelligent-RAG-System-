"""Metadata extraction for traceable retrieval citations."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def enrich_metadata(documents: list[Document]) -> list[Document]:
    """Add source, page, heading, IDs and content metrics to every chunk."""
    enriched: list[Document] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        source = str(metadata.get("source", "unknown"))
        nested = metadata.get("extractor_metadata", {}) or metadata.get("docling_metadata", {})
        page_number = metadata.get("page_number", metadata.get("page"))
        if page_number is None and isinstance(nested, dict):
            page_number = nested.get("page_number")
        headings = _HEADING.findall(document.page_content)
        metadata.update({"chunk_id": sha256(f"{source}:{index}:{document.page_content}".encode()).hexdigest()[:16], "source_file": Path(source).name, "page_number": page_number, "heading": headings[-1] if headings else metadata.get("heading", ""), "char_count": len(document.page_content), "word_count": len(document.page_content.split())})
        enriched.append(Document(page_content=document.page_content, metadata=metadata))
    print(f"[metadata] Enriched metadata for {len(enriched)} chunks.")
    return enriched


def format_citation(chunk: dict[str, Any], index: int) -> str:
    """Make a stable, human-readable citation label for a retrieved chunk."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source_file", metadata.get("source", "unknown source"))
    page = metadata.get("page_number")
    return f"[{index}] {source}" + (f", page {page}" if page is not None else "")

