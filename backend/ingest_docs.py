"""Index documents from the backend docs folder into Qdrant."""

from __future__ import annotations

from pathlib import Path

from RETRIVAL.pipeline import ingest_to_qdrant


SUPPORTED_FILE_TYPES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


def main() -> int:
    docs_directory = Path(__file__).parent / "docs"
    if not docs_directory.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {docs_directory}")

    documents = sorted(
        path
        for path in docs_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES
    )
    if not documents:
        print(f"No supported documents found in {docs_directory}.")
        print("Add documents there, then run this command again.")
        return 1

    print(f"Found {len(documents)} document(s) in {docs_directory}.")
    ingest_to_qdrant(documents)
    print(f"Successfully indexed {len(documents)} document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
