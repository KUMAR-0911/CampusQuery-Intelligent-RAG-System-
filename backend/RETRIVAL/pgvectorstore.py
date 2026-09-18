"""PostgreSQL (pgvector) dense retrieval and HF Inference API reranking."""

from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5
from typing import Any, Sequence

from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from config import RetrievalConfig
from RETRIVAL.embedding import SentenceTransformerEmbedder

from database import get_db_engine

def make_vector_records(documents: list[Document], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    """Pair chunks with vectors in a database-neutral record format."""
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must have the same length")
    records = [{"id": doc.metadata["chunk_id"], "text": doc.page_content, "metadata": doc.metadata, "vector": vector} for doc, vector in zip(documents, embeddings)]
    print(f"[vectorstore] Prepared {len(records)} vector records.")
    return records


class PgVectorStore:
    """Store and search chunks using pgvector and BGE reranking."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: SentenceTransformerEmbedder | None = None,
        engine: Engine | None = None,
    ) -> None:
        """Create PostgreSQL engine. Reranking uses HF Inference API (no local models)."""
        if not config.postgres_url:
            raise ValueError("Set POSTGRES_URL in .env to your PostgreSQL database.")

        self.config = config
        self.engine = engine or get_db_engine()
        self.embedder = embedder or SentenceTransformerEmbedder(config)
        self.reranker_client = None
        self._init_reranker_client()
        print(f"[vectorstore] PgVectorStore initialized. Reranker ({config.reranker_model}) via HF Inference API.")

    def _init_reranker_client(self):
        if self.reranker_client is not None:
            return self.reranker_client
        token = getattr(self.config, "hf_reranker_token", None) or self.config.hf_token
        if token:
            try:
                from huggingface_hub import InferenceClient
                client_kwargs = {"model": self.config.reranker_model, "token": token}
                if hasattr(self.config, "reranker_inference_provider") and self.config.reranker_inference_provider:
                    client_kwargs["provider"] = self.config.reranker_inference_provider
                self.reranker_client = InferenceClient(**client_kwargs)
                print(f"[vectorstore] Remote Reranker client initialized ({self.config.reranker_model}).")
            except Exception as exc:
                print(f"[vectorstore] Notice initializing reranker client: {exc}")
        return self.reranker_client

    def preload(self) -> None:
        """Eagerly initialize and prewarm embedder, reranker, and vector database schema."""
        if hasattr(self.embedder, "preload"):
            self.embedder.preload()
        self._init_reranker_client()
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[vectorstore] PostgreSQL pgvector connection preloaded successfully.")
        except Exception as exc:
            print(f"[vectorstore] Database ping notice during preload: {exc}")
        
    def ensure_collection(self, vector_size: int) -> None:
        """Create the pgvector extension and the chunks table if not existing."""
        table = self.config.pgvector_table
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text(f"SELECT to_regclass('public.{table}') IS NOT NULL AS exists")).scalar()
                if res:
                    return
        except Exception:
            pass

        with self.engine.begin() as conn:
            conn.execute(text(f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {table} (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR(255),
                    source_file VARCHAR(1024),
                    text TEXT,
                    metadata JSONB,
                    embedding vector({vector_size})
                );
                CREATE INDEX IF NOT EXISTS {table}_embedding_idx 
                ON {table} USING hnsw (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS {table}_user_id_idx ON {table} (user_id);
            """))
            
    def upsert(self, documents: Sequence[Document]) -> None:
        """Write text vectors and metadata to PostgreSQL."""
        if not documents:
            print("[vectorstore] No documents to index.")
            return
            
        dense_vectors = self.embedder.embed_documents(documents)
        self.ensure_collection(len(dense_vectors[0]))
        
        table = self.config.pgvector_table
        
        with self.engine.begin() as conn:
            for document, dense in zip(documents, dense_vectors):
                chunk_id = str(uuid5(NAMESPACE_URL, document.metadata["chunk_id"]))
                user_id = document.metadata.get("user_id", "")
                source_file = document.metadata.get("source_file", "")
                
                conn.execute(text(f"""
                    INSERT INTO {table} (id, user_id, source_file, text, metadata, embedding)
                    VALUES (:id, :user_id, :source_file, :text, CAST(:metadata AS JSONB), CAST(:embedding AS VECTOR))
                    ON CONFLICT (id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        source_file = EXCLUDED.source_file,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                """), {
                    "id": chunk_id,
                    "user_id": user_id,
                    "source_file": source_file,
                    "text": document.page_content,
                    "metadata": json.dumps(document.metadata),
                    "embedding": f"[{','.join(map(str, dense))}]"
                })
                
        print(f"[vectorstore] Indexed {len(documents)} chunks in PostgreSQL (pgvector).")

    def search(
        self, query: str, metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None, rerank_top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve and rerank the most relevant chunks for a query.
        
        Uses pgvector for cosine similarity, followed by HF Inference API reranking.
        """
        print(f"[vectorstore] Searching pgvector for: {query[:80]}")
        
        dense_query = self.embedder.embed_texts([query])[0]
        embedding_str = f"[{','.join(map(str, dense_query))}]"
        
        candidate_limit = rerank_top_n if rerank_top_n is not None else self.config.rerank_top_n
        final_limit = top_k if top_k is not None else self.config.top_k
        table = self.config.pgvector_table
        
        # Build filter query
        where_clauses = []
        params = {"embedding": embedding_str, "limit": candidate_limit}
        
        if metadata_filters:
            if "user_id" in metadata_filters and metadata_filters["user_id"]:
                users = metadata_filters["user_id"]
                if isinstance(users, (list, tuple, set)):
                    users = list(users)
                else:
                    users = [users]
                where_clauses.append("user_id = ANY(:user_ids)")
                params["user_ids"] = users
                
            if "source_file" in metadata_filters and metadata_filters["source_file"]:
                sources = metadata_filters["source_file"]
                if isinstance(sources, (list, tuple, set)):
                    sources = list(sources)
                else:
                    sources = [sources]
                where_clauses.append("source_file = ANY(:source_files)")
                params["source_files"] = sources

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        sql = f"""
            SELECT id, text, metadata, 1 - (embedding <=> CAST(:embedding AS VECTOR)) AS cosine_similarity
            FROM {table}
            {where_sql}
            ORDER BY embedding <=> CAST(:embedding AS VECTOR)
            LIMIT :limit
        """
        
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            
        print(f"\n==================== [CHUNKS BEFORE RERANKER] ====================")
        print(f"Total candidate chunks retrieved from pgvector: {len(rows)}")
        for idx, row in enumerate(rows, 1):
            meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
            source_file = meta.get("source_file", meta.get("source", "unknown"))
            sim = float(row.get("cosine_similarity") or 0.0)
            preview = row["text"].replace("\n", " ").strip()[:140]
            print(f"  [{idx}] Sim: {sim:.4f} | File: {source_file} | ID: {row['id']}")
            print(f"      Text: \"{preview}...\"\n")
        print("===================================================================\n")
        
        if not rows:
            return []
            
        # Rerank via HF Inference API (cloud-only, no local models)
        import math
        def sigmoid(logit: float) -> float:
            try:
                if logit >= 0:
                    return 1.0 / (1.0 + math.exp(-logit))
                else:
                    z = math.exp(logit)
                    return z / (1.0 + z)
            except OverflowError:
                return 0.0 if logit < 0 else 1.0

        rerank_pairs = [[query, row["text"]] for row in rows]
        rerank_scores = []

        print(f"[vectorstore] Calling HF Inference API for reranking with {self.config.reranker_model}...")
        try:
            client = self.reranker_client or self._init_reranker_client()
            if not client:
                raise ValueError("Reranker client could not be initialized. Check HF_RERANKER_TOKEN or HF_TOKEN.")
            payload = {
                "inputs": [{"text": pair[0], "text_pair": pair[1]} for pair in rerank_pairs]
            }
            
            response_bytes = client.post(json=payload)
            api_result = json.loads(response_bytes.decode("utf-8"))
            
            # HF API can return list of floats or list of dicts
            if isinstance(api_result, list):
                for item in api_result:
                    if isinstance(item, float) or isinstance(item, int):
                        rerank_scores.append(float(item))
                    elif isinstance(item, dict) and "score" in item:
                        rerank_scores.append(float(item["score"]))
                    elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict) and "score" in item[0]:
                        rerank_scores.append(float(item[0]["score"]))
                    else:
                        rerank_scores.append(0.0)
            else:
                print(f"[vectorstore] Unexpected HF API response format: {api_result}")
                rerank_scores = [0.0] * len(rows)
        except Exception as e:
            print(f"[vectorstore] HF InferenceClient request failed: {e}")
            rerank_scores = [0.0] * len(rows)

        
        print(f"\n==================== [CROSS-ENCODER RERANKING DEBUG] ====================")
        print(f"Query: \"{query}\"")
        print(f"Total candidate chunks evaluated: {len(rows)} | Score threshold: {self.config.rerank_score_threshold}")

        ranked_allowed = []
        ranked_all = []
        
        for idx, (row, score) in enumerate(zip(rows, rerank_scores), 1):
            raw_logit = float(score)
            sig_score = sigmoid(raw_logit)
            row_id = str(row["id"])
            chunk_text = row["text"].strip()
            meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
            source_file = meta.get("source_file", meta.get("source", "unknown"))
            heading = meta.get("heading") or meta.get("title") or meta.get("section") or "General Content"
            cos_sim = float(row.get("cosine_similarity") or 0.0)
            is_allowed = raw_logit >= self.config.rerank_score_threshold

            status_str = "ALLOWED" if is_allowed else "FILTERED (< threshold)"
            preview = chunk_text.replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:100] + "..."

            print(f"  [{idx}] {status_str} | Sigmoid: {sig_score:.4f} (Raw: {raw_logit:+.4f}) | Sim: {cos_sim:.4f} | File: {source_file}")
            print(f"      Heading: {heading} | Text: \"{preview}\"")

            chunk_entry = {
                "id": row_id,
                "text": chunk_text,
                "metadata": meta,
                "rerank_score": sig_score,
                "raw_score": raw_logit,
                "heading": heading,
            }
            ranked_all.append((chunk_entry, sig_score))
            if is_allowed:
                ranked_allowed.append((chunk_entry, sig_score))

        print("==========================================================================\n")

        # Use allowed chunks if available, otherwise fall back to top candidates
        final_pool = ranked_allowed if ranked_allowed else ranked_all
        final_pool.sort(key=lambda item: item[1], reverse=True)
        
        results = [entry for entry, _ in final_pool[: final_limit]]
        
        print(f"\n==================== [RETRIEVED DOCUMENT CHUNKS FOR ANSWER (Top {len(results)})] ====================")
        print(f"Full document chunks passed to reasoning model:")
        for idx, item in enumerate(results, 1):
            meta = item["metadata"]
            source_file = meta.get("source_file", meta.get("source", "unknown"))
            sig_score = float(item["rerank_score"])
            raw_logit = float(item.get("raw_score", sig_score))
            heading = item.get("heading", "Document Section")
            print(f"\n--- [Chunk {idx}] Heading: {heading} | Source: {source_file} (ID: {item['id']}) ---")
            print(f"    Rerank Score (Sigmoid): {sig_score:.4f} (Raw Logit: {raw_logit:+.4f})")
            print(f"    Full Content:")
            for line in item["text"].strip().splitlines():
                print(f"      {line}")
        print("\n============================================================================================\n")
        return results

    def get_all_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve all document chunks for a specific user, bypassing vector search."""
        table = self.config.pgvector_table
        sql = f"SELECT id, text, metadata FROM {table} WHERE user_id = :user_id"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"user_id": user_id}).mappings().all()
        
        results = [
            {
                "id": str(row["id"]), 
                "text": row["text"], 
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]), 
                "rerank_score": 1.0
            } 
            for row in rows
        ]
        print(f"[vectorstore] Fetched {len(results)} total chunks for user {user_id}.")
        return results

    def delete_user_chunks(self, user_id: str) -> None:
        """Delete all document chunks from the vector database for a specific user."""
        table = self.config.pgvector_table
        print(f"[vectorstore] Deleting ALL chunks for user '{user_id}' from table '{table}'...")
        sql = f"DELETE FROM {table} WHERE user_id = :user_id"
        with self.engine.begin() as conn:
            result = conn.execute(text(sql), {"user_id": user_id})
            deleted_count = result.rowcount
        print(f"[vectorstore] Deleted {deleted_count} chunk(s) for user '{user_id}'. Database is clean for new document upload.")
