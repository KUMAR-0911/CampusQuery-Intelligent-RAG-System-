"""
document_loading.py — Extract text from uploaded documents using Docling.

WHY DOCLING?
------------
Docling (by IBM Research) uses deep-learning models to understand document
layouts — headings, tables, columns, reading order — so you get much
cleaner text than basic PDF parsers like PyPDF.

SUPPORTED FORMATS
-----------------
PDF, DOCX, PPTX, XLSX, HTML, Markdown, plain text, and images.

HOW TO USE
----------
    from RETRIVAL.document_loading import DoclingDocumentLoader

    loader = DoclingDocumentLoader()         # create once, reuse many times
    text   = loader.extract_text("cv.pdf")   # returns a plain-text string
"""

from __future__ import annotations

from pathlib import Path


class DoclingDocumentLoader:
    """Thin wrapper around Docling's DocumentConverter.

    Create one instance at startup and call `extract_text()` for each file.
    The converter is heavy to initialise (it loads ML models), so reusing
    the same instance saves a lot of time on the second file onwards.
    """

    def __init__(self) -> None:
        """Import Docling and create the converter.

        Raises ImportError with a clear message if the package is missing.
        """
        print("[document_loading] Initialising Docling converter …")

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise ImportError(
                "The 'docling' package is required for document extraction.\n"
                "Install it with:  pip install docling"
            ) from error

        self._converter = DocumentConverter()
        print("[document_loading] Docling converter is ready.")

    # ── Public method ───────────────────────────────────────────

    def extract_text(self, file_path: str | Path) -> str:
        """Convert a single document file into clean Markdown text.

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            The full document text exported as Markdown.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError:        If Docling could not extract any text.
        """
        path = Path(file_path)

        # Guard: make sure the file actually exists.
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        print(f"[document_loading] Extracting text from '{path.name}' …")

        # Docling returns a ConversionResult; we export to Markdown because
        # it preserves headings and table structure nicely.
        result = self._converter.convert(str(path))
        text = result.document.export_to_markdown().strip()

        if not text:
            raise ValueError(
                f"Docling extracted no text from '{path.name}'. "
                "The file may be empty or in an unsupported format."
            )

        print(
            f"[document_loading] Got {len(text):,} characters "
            f"from '{path.name}'."
        )
        return text
