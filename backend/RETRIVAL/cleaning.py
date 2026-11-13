"""
cleaning.py — Normalise and tidy raw document text before chunking.

WHAT IT DOES
------------
1. Converts Windows / old-Mac line endings to Unix newlines.
2. Collapses runs of whitespace inside each line.
3. Drops purely decorative separator lines (e.g. "======", "------").
4. Removes blank lines so the text is compact and chunk-friendly.

The cleaning is intentionally *light* — it never alters real content,
headings, or punctuation.
"""

from __future__ import annotations

from langchain_core.documents import Document

# Characters that, on their own, make up a decorative divider line.
_DIVIDER_CHARS = set("-_*#=")


def _is_divider_line(line: str) -> bool:
    """Return True if the line is just a decorative separator.

    Examples that match: "---", "===", "***", "###", "___"
    Examples that do NOT match: "# Heading", "- list item"
    """
    return len(line) >= 3 and all(char in _DIVIDER_CHARS for char in line)


def clean_text(text: str) -> str:
    """Clean a single text string.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Cleaned text with normalised whitespace and no divider lines.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("clean_text() expects a string.")

    # Normalise line endings and tabs.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        # Collapse multiple spaces into one and strip leading/trailing space.
        clean_line = " ".join(line.split())

        # Keep the line only if it has real content (not blank, not a divider).
        if clean_line and not _is_divider_line(clean_line):
            cleaned_lines.append(clean_line)

    return "\n".join(cleaned_lines)


def clean_documents(documents: list[Document]) -> list[Document]:
    """Apply `clean_text()` to every document, dropping any that become empty.

    Metadata is preserved unchanged.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        A new list containing only the documents with non-empty cleaned text.
    """
    print(f"[cleaning] Cleaning {len(documents)} document(s) …")

    cleaned: list[Document] = []
    for doc in documents:
        content = clean_text(doc.page_content)
        if content:
            cleaned.append(
                Document(page_content=content, metadata=dict(doc.metadata))
            )

    print(f"[cleaning] Kept {len(cleaned)} non-empty document(s).")
    return cleaned
