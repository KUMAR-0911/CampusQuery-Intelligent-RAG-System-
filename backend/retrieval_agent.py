"""
retrieval_agent.py — Retrieve relevant chunks, then ask Groq for an answer.

HOW THE RAG ANSWER IS BUILT
────────────────────────────
1. RETRIEVE  — Embed the user's question and find the closest chunks
               in Qdrant (semantic search).
2. CONTEXT   — Format those chunks into a block of text the LLM can read.
3. GENERATE  — Send the system prompt + context + question to the Groq
               LLM and get back a grounded answer.
4. PACKAGE   — Return the answer, citations, and raw chunks to the caller.

CONFIGURATION
─────────────
  GROQ_API_KEY  — Your Groq API key (set in .env).
  GROQ_MODEL    — Which model to use (default: llama-3.3-70b-versatile).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import RetrievalConfig
from RETRIVAL.qdrantstore import QdrantVectorStore

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Three-stage RAG agent: retrieve → build context → generate answer."""

    # ── Initialisation ──────────────────────────────────────────

    def __init__(self, config: RetrievalConfig) -> None:
        """Create the Groq LLM client.

        Args:
            config: Must include a valid groq_api_key and groq_model.

        Raises:
            ValueError:  If GROQ_API_KEY is not set.
            ImportError:  If langchain-groq is not installed.
        """
        if not config.groq_api_key:
            raise ValueError("Set GROQ_API_KEY in the backend .env file.")

        try:
            from langchain_groq import ChatGroq
        except ImportError as error:
            raise ImportError(
                "The 'langchain-groq' package is required.\n"
                "Install it with:  pip install langchain-groq"
            ) from error

        self.config = config
        self.llm = ChatGroq(
            model=config.groq_model,
            groq_api_key=config.groq_api_key,
            temperature=0.1,
        )
        self.reranker = None

        print(f"[agent] Ready — using Groq model '{config.groq_model}'.")
        logger.info("Groq model initialised: %s", config.groq_model)

    def preload(self) -> None:
        """No-op — the LLM client is ready after __init__."""
        print(f"[agent] Groq model '{self.config.groq_model}' is ready.")

    # ── Stage 1: Retrieve ───────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        store: QdrantVectorStore,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search Qdrant for chunks relevant to the query.

        Duplicates (by chunk ID) are removed.

        Args:
            query:            The user's question.
            store:            The Qdrant vector store to search.
            metadata_filters: Optional filters to narrow the search.

        Returns:
            A de-duplicated list of chunk dicts.
        """
        print(f"[agent] Searching for relevant chunks …")

        raw_chunks = store.search(
            query,
            metadata_filters=metadata_filters,
            top_k=self.config.top_k,
        )

        # De-duplicate by chunk ID.
        unique_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for chunk in raw_chunks:
            metadata = chunk.get("metadata") or {}
            chunk_id = str(
                chunk.get("id", metadata.get("id", chunk.get("text", "")))
            )
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        print(f"[agent] Retrieved {len(unique_chunks)} unique chunk(s).")
        logger.info("Retrieved %d unique chunks", len(unique_chunks))
        return unique_chunks

    # ── Stage 2 + 3: Build context and generate answer ──────────

    def answer(
        self,
        question: str,
        store: QdrantVectorStore,
        metadata_filters: dict[str, Any] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full RAG pipeline: retrieve → context → generate → package.

        Args:
            question:             The user's question.
            store:                Qdrant vector store.
            metadata_filters:     Optional search filters.
            conversation_context: Optional dict with 'summary' and
                                  'recent_messages' for multi-turn chat.

        Returns:
            A dict with: answer, citations, retrieved (bool),
            retrieved_chunks (list).
        """
        print("[agent] ── Stage 1/3: Retrieve ──")
        chunks = self.retrieve_documents(
            question, store=store, metadata_filters=metadata_filters,
        )
        self._print_chunk_previews("Retrieved chunks", chunks, "similarity_score")

        # ── Build the document context block ────────────────
        print("[agent] ── Stage 2/3: Build context ──")
        retrieved_context = self._format_context(chunks)

        # ── Assemble the LLM message list ───────────────────
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=self.config.system_prompt),
        ]

        # Add conversation history (multi-turn support).
        profile_context = ""
        if conversation_context:
            summary = (conversation_context.get("summary") or "").strip()
            if summary:
                profile_context = (
                    f"PERSISTENT USER PROFILE AND MEMORY:\n{summary}\n\n"
                )

            # Replay the last 4 messages so the LLM knows the conversation.
            recent = (conversation_context.get("recent_messages") or [])[-4:]
            for msg in recent:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Final user message: profile + context + question + instructions.
        user_message = (
            f"{profile_context}"
            f"{retrieved_context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Instruction for reasoning and answer generation:\n"
            "Directly and concisely answer the question using the retrieved "
            "context. Avoid a repeated template, include only relevant "
            "details, use no blank lines between paragraphs, and follow "
            "the system prompt's formatting rules."
        )
        messages.append(HumanMessage(content=user_message))

        # ── Call the Groq LLM ───────────────────────────────
        print(f"[agent] ── Stage 3/3: Generate answer via "
              f"'{self.config.groq_model}' ──")
        logger.info("Generating answer with %s", self.config.groq_model)

        try:
            response = self.llm.invoke(messages)
            answer = self._extract_text(response)
            print(f"[agent] Received answer ({len(answer)} chars).")
        except Exception as error:
            print(f"[agent] Groq generation failed: {error}")
            raise RuntimeError(
                f"Groq answer generation failed: {error}"
            ) from error

        # ── Package the result ──────────────────────────────
        result = {
            "answer":           answer,
            "citations":        self._build_citations(chunks),
            "retrieved":        bool(chunks),
            "retrieved_chunks": self._serialise_chunks(chunks),
        }
        print("[agent] All stages complete.")
        return result

    # ── Private helpers ─────────────────────────────────────────

    def _rerank_documents(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score candidate chunks locally and keep the highest-ranked ones."""
        if not chunks:
            return []

        if self.reranker is None:
            from sentence_transformers import CrossEncoder

            print(
                f"[agent] Loading local reranker '{self.config.reranker_model}' "
                f"on {self.config.reranker_device} …"
            )
            self.reranker = CrossEncoder(
                self.config.reranker_model,
                max_length=512,
                device=self.config.reranker_device,
            )

        pairs = [(query, str(chunk.get("text", ""))) for chunk in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked_chunks = []
        for chunk, score in zip(chunks, scores):
            ranked_chunk = dict(chunk)
            ranked_chunk["rerank_score"] = float(score)
            ranked_chunks.append(ranked_chunk)

        ranked_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )
        return ranked_chunks[:self.config.reranker_top_n]

    @staticmethod
    def _print_chunk_previews(
        stage: str,
        chunks: list[dict[str, Any]],
        score_key: str,
    ) -> None:
        """Print each chunk's rank and a short text preview to the console."""
        print(f"[agent] {stage}: {len(chunks)} chunk(s)")
        for index, chunk in enumerate(chunks, start=1):
            text = " ".join(str(chunk.get("text", "")).split())
            preview = text[:180] + ("..." if len(text) > 180 else "")
            print(
                f"  {index}. id={chunk.get('id', 'unknown')} "
                f"score={chunk.get(score_key, 0.0)}"
            )
            print(f"     {preview}")

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        """Turn a list of chunk dicts into a labelled text block."""
        if not chunks:
            return (
                "RETRIEVED DOCUMENT CONTEXT:\n"
                "[No relevant document chunks found in the vector database. "
                "Politely tell the user that no relevant content was found "
                "and invite them to upload a document.]"
            )

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata") or {}
            source = (
                meta.get("source_file")
                or meta.get("source")
                or "uploaded document"
            )
            score = chunk.get("similarity_score", 0.0)
            text = chunk.get("text", "").strip()
            document_date = meta.get("document_date")
            date_source = meta.get("document_date_source", "unknown")
            mentioned_dates = meta.get("dates_mentioned") or []
            date_context = (
                f"\nDocument date: {document_date} ({date_source})"
                if document_date else ""
            )
            if mentioned_dates:
                if isinstance(mentioned_dates, str):
                    mentioned_dates = [mentioned_dates]
                date_context += (
                    f"\nDates mentioned: {', '.join(mentioned_dates)}"
                )
            blocks.append(
                f"[Document Chunk {i} | Source: {source} | "
                f"Similarity: {score:.4f}]{date_context}\n{text}"
            )

        return "RETRIEVED DOCUMENT CONTEXT:\n" + "\n\n".join(blocks)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the plain-text answer out of a LangChain response object."""
        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        return str(content)

    @staticmethod
    def _build_citations(chunks: list[dict[str, Any]]) -> list[str]:
        """Extract a unique, ordered list of source filenames."""
        return list(dict.fromkeys(
            (chunk.get("metadata") or {}).get("source_file")
            or (chunk.get("metadata") or {}).get("source", "")
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("source_file")
            or (chunk.get("metadata") or {}).get("source")
        ))

    @staticmethod
    def _serialise_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert raw chunk dicts into a clean JSON-serialisable format."""
        serialised = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            score = float(chunk.get("similarity_score",
                                    chunk.get("rerank_score", 0.0)))
            rerank_score = float(chunk.get("rerank_score", score))
            heading = (
                chunk.get("heading")
                or meta.get("heading")
                or meta.get("title")
                or meta.get("section")
                or "Section"
            )
            serialised.append({
                "id":               str(chunk.get("id", "")),
                "text":             str(chunk.get("text", "")),
                "similarity_score": score,
                "rerank_score":     rerank_score,
                "raw_score":        score,
                "heading":          str(heading),
                "source":           str(
                    meta.get("source_file") or meta.get("source") or "document"
                ),
                "metadata": {
                    k: str(v) for k, v in meta.items() if k != "embedding"
                },
            })
        return serialised
