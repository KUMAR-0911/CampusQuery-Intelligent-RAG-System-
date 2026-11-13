    """
main.py — FastAPI REST endpoints for the CampusQuery RAG backend.

ENDPOINTS
---------
  GET  /         →  Health check (returns {"status": "online"}).
  GET  /health   →  Simple health probe.
  POST /upload   →  Upload one or more documents; they are indexed in Qdrant.
  POST /chat     →  Ask a question; the answer is grounded in indexed docs.

HOW TO RUN
----------
    cd backend
    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from RETRIVAL.pipeline import answer_question, get_vector_store, ingest_to_qdrant

logger = logging.getLogger(__name__)

app = FastAPI(title="CampusQuery RAG API", version="1.0.0")


# ── Request / response models ──────────────────────────────────


class ChatRequest(BaseModel):
    """Body for the POST /chat endpoint."""
    question: str = Field(min_length=1)
    conversation_context: dict[str, Any] | None = None


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, str]:
    """Basic service info."""
    return {"status": "online", "service": "CampusQuery RAG API"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/upload")
def upload_documents(
    files: list[UploadFile] = File(...),
) -> dict[str, str]:
    """Accept uploaded documents, extract text, and index in Qdrant.

    The files are saved to a temporary directory, processed through the
    full ingestion pipeline (Docling → clean → chunk → enrich → embed
    → Qdrant), and then the temporary files are deleted.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")

    print(f"[api] Received {len(files)} file(s) for indexing.")

    try:
        with tempfile.TemporaryDirectory(prefix="campusquery_") as tmp_dir:
            # Save uploaded bytes to disk so Docling can read them.
            paths: list[Path] = []
            for index, uploaded_file in enumerate(files):
                filename = Path(uploaded_file.filename or f"document-{index}").name
                path = Path(tmp_dir) / f"{index}_{filename}"
                path.write_bytes(uploaded_file.file.read())
                paths.append(path)

            print("[api] Files saved — starting the indexing pipeline.")
            ingest_to_qdrant(paths, store=get_vector_store())

        print(f"[api] Successfully indexed {len(paths)} document(s).")
        return {"message": f"Successfully indexed {len(paths)} document(s)."}

    except Exception as error:
        print(f"[api] Upload failed: {type(error).__name__}")
        logger.exception("Document upload/indexing failed")
        raise HTTPException(
            status_code=500, detail=f"Indexing failed: {error}"
        ) from error


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer a question grounded in the indexed documents.

    Optionally pass `conversation_context` for multi-turn chat support.
    """
    print(f"[api] Chat question received ({len(request.question)} chars).")

    try:
        result = answer_question(
            request.question,
            get_vector_store(),
            conversation_context=request.conversation_context,
        )
        print("[api] Chat response ready.")
        return result

    except Exception as error:
        print(f"[api] Chat failed: {type(error).__name__}")
        logger.exception("RAG question answering failed")
        raise HTTPException(
            status_code=500, detail=f"Could not answer question: {error}"
        ) from error
