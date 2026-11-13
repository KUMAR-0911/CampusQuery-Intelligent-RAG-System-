"""
enrich_metadata.py — Add useful fields to every document chunk.

After chunking, each chunk is a bare slab of text with minimal metadata.
This module walks through each chunk and adds:

  • chunk_id     — A short, stable hash that uniquely identifies the chunk.
                   If the same text from the same source is re-indexed, it
                   gets the same ID, which lets Qdrant's upsert de-duplicate.
  • source_file  — Just the filename (e.g. "resume.pdf"), handy for display.
  • page_number  — Extracted from the metadata if the loader set it.
  • heading      — The last Markdown heading found inside the chunk text.
  • char_count   — Number of characters in the chunk.
  • word_count   — Number of whitespace-separated words in the chunk.

It also provides a small `format_citation()` helper used when building
the answer's source list.
"""

from __future__ import annotations

import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


# ── Main enrichment function ────────────────────────────────────


def enrich_metadata(documents: list[Document]) -> list[Document]:
    """Add derived metadata fields to every chunk.

    Args:
        documents: A list of chunked Document objects.

    Returns:
        A new list of Documents with enriched metadata dicts.
    """
    print(f"[metadata] Enriching {len(documents)} chunk(s) …")
    enriched: list[Document] = []

    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata)

        # ── Source & page ───────────────────────────────────
        source = str(metadata.get("source", "unknown"))
        page_number = _find_page_number(metadata)

        # ── Heading detection ───────────────────────────────
        heading = _extract_last_heading(doc.page_content)
        if not heading:
            heading = metadata.get("heading", "")

        # ── Stable chunk ID ─────────────────────────────────
        identity_string = f"{source}:{index}:{doc.page_content}"
        chunk_id = sha256(identity_string.encode()).hexdigest()[:16]

        # ── Write everything back ───────────────────────────
        metadata.update({
            "chunk_id":    chunk_id,
            "source_file": Path(source).name,
            "page_number": page_number,
            "heading":     heading,
            "char_count":  len(doc.page_content),
            "word_count":  len(doc.page_content.split()),
        })

        enriched.append(
            Document(page_content=doc.page_content, metadata=metadata)
        )

    print(f"[metadata] Enriched {len(enriched)} chunk(s).")
    return enriched


def extract_date_metadata(text: str, source_path: str | Path) -> dict[str, Any]:
    """Extract document dates and fall back to the file modification date."""
    date_matches = _extract_dates(text)
    mentioned_dates = sorted({value for _, value in date_matches})

    notice_date = ""
    for line in text.splitlines():
        if re.search(
            r"\b(?:notice\s+date|date\s+of\s+(?:issue|publication)|"
            r"issued\s+on|published\s+on)\b|^\s*date\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            line_dates = _extract_dates(line)
            if line_dates:
                notice_date = line_dates[0][1]
                break

    if not notice_date and date_matches:
        notice_date = date_matches[0][1]

    try:
        file_modified_date = datetime.fromtimestamp(
            Path(source_path).stat().st_mtime
        ).date().isoformat()
    except (OSError, ValueError):
        file_modified_date = ""

    return {
        "notice_date": notice_date,
        "file_modified_date": file_modified_date,
        "document_date": notice_date or file_modified_date,
        "document_date_source": (
            "notice" if notice_date else
            "file_modified" if file_modified_date else
            "unknown"
        ),
        "dates_mentioned": mentioned_dates,
    }


# ── Citation formatting ─────────────────────────────────────────


def format_citation(chunk: dict[str, Any], index: int) -> str:
    """Build a human-readable citation string like '[1] resume.pdf, page 3'.

    Args:
        chunk: A retrieval result dict with a "metadata" sub-dict.
        index: The citation number (1-based).

    Returns:
        A formatted citation string.
    """
    metadata = chunk.get("metadata", {})
    source = metadata.get("source_file", metadata.get("source", "unknown"))
    page = metadata.get("page_number")
    page_text = f", page {page}" if page is not None else ""
    return f"[{index}] {source}{page_text}"


# ── Private helpers ─────────────────────────────────────────────


def _find_page_number(metadata: dict[str, Any]) -> int | None:
    """Look for a page number in common metadata locations."""
    page = metadata.get("page_number") or metadata.get("page")
    if page is not None:
        return page

    # Some extractors (like Docling) nest extra info in a sub-dict.
    nested = metadata.get("extractor_metadata") or metadata.get("docling_metadata") or {}
    if isinstance(nested, dict):
        return nested.get("page_number")

    return None


def _extract_last_heading(text: str) -> str:
    """Return the last Markdown heading (# … ######) found in the text.

    We take the *last* heading because it is most likely to describe
    the section the chunk belongs to.
    """
    headings: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        # Count leading '#' characters.
        without_hashes = stripped.lstrip("#")
        hash_count = len(stripped) - len(without_hashes)

        # Valid Markdown headings have 1–6 '#' characters followed by a space.
        if 1 <= hash_count <= 6 and without_hashes.startswith(" "):
            title = without_hashes.strip().rstrip("#").strip()
            if title:
                headings.append(title)

    return headings[-1] if headings else ""


def _extract_dates(text: str) -> list[tuple[int, str]]:
    """Return recognized dates as (text position, ISO date) pairs."""
    month = (
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember|t)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    )
    patterns = [
        r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b",
        r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2}\b",
        rf"\b(?:{month})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+(?:19|20)\d{{2}}\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month})\.?[,]?\s+(?:19|20)\d{{2}}\b",
        rf"\b(?:{month})\.?\s+(?:19|20)\d{{2}}\b",
    ]
    matches: list[tuple[int, int, str]] = []
    for pattern in patterns:
        matches.extend(
            (match.start(), match.end(), match.group(0))
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    dates: list[tuple[int, str]] = []
    occupied_until = -1
    seen_dates: set[str] = set()
    for start, end, value in matches:
        if start < occupied_until:
            continue
        normalized = _normalize_date(value)
        if normalized and normalized not in seen_dates:
            dates.append((start, normalized))
            seen_dates.add(normalized)
            occupied_until = end
    return sorted(dates, key=lambda item: item[0])


def _normalize_date(value: str) -> str:
    """Normalize common numeric and written dates to sortable ISO strings."""
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%d %B %Y", "%d %b %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(cleaned, date_format).date().isoformat()
        except ValueError:
            continue

    for date_format in ("%B %Y", "%b %Y"):
        try:
            return datetime.strptime(cleaned, date_format).strftime("%Y-%m")
        except ValueError:
            continue
    return ""
