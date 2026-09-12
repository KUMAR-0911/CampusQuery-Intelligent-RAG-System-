"""Whitespace normalisation utilities used before indexing."""

import re

from langchain_core.documents import Document

_EXCESS_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise whitespace in *text* without removing or changing symbols and dedup sentence by sentence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _EXCESS_WHITESPACE.sub(" ", cleaned)
    
    lines = cleaned.split('\n')
    deduped_lines = []
    seen = set()
    
    for line in lines:
        # Split by sentence endings (.!?) followed by space
        sentences = re.split(r'(?<=[.!?])\s+', line)
        deduped_sentences = []
        for s in sentences:
            s_lower = s.strip().lower()
            if s_lower and s_lower not in seen:
                seen.add(s_lower)
                deduped_sentences.append(s)
        if deduped_sentences:
            deduped_lines.append(' '.join(deduped_sentences))
        elif not line.strip():
            # preserve empty lines for paragraph breaks
            deduped_lines.append('')
            
    cleaned = '\n'.join(deduped_lines)
    result = _EXCESS_NEWLINES.sub("\n\n", cleaned).strip()
    print(f"[cleaning] Cleaned text from {len(text)} to {len(result)} characters.")
    return result


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean LangChain documents without discarding their loader metadata."""
    cleaned: list[Document] = []
    for document in documents:
        text = clean_text(document.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=dict(document.metadata)))
    print(f"[cleaning] Cleaned {len(cleaned)} non-empty document/page record(s).")
    return cleaned
