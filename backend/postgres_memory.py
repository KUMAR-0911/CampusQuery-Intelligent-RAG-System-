"""PostgreSQL-backed persistent memory for per-user facts, summaries, and chat history.

Supports cloud PostgreSQL providers (Neon, Supabase, Railway, Render) out of the
box — just set POSTGRES_URL in your .env file.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class PostgresMemory:
    """Store and retrieve per-user durable facts, rolling summaries, and recent messages.

    Tables created automatically on first connection:
        - ``user_memories``  — one row per user with extracted facts + rolling summary.
        - ``conversation_messages`` — append-only chat log with compaction flag.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        engine: Engine,
        recent_messages: int = 5,
        summary_max_chars: int = 6000,
        cache_ttl_seconds: float = 60.0,
        cache_maxsize: int = 1024,
    ) -> None:
        if recent_messages <= 0:
            raise ValueError("recent_messages must be greater than zero.")

        self.recent_messages = recent_messages
        self.summary_max_chars = summary_max_chars
        self.engine = engine
        self._cache_ttl = cache_ttl_seconds
        self._cache_maxsize = cache_maxsize
        self._cache: collections.OrderedDict[str, tuple[float, dict[str, Any]]] = collections.OrderedDict()
        self._cache_lock = threading.Lock()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create the memory tables and indexes if they do not already exist."""
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id   VARCHAR(255) PRIMARY KEY,
                facts     TEXT NOT NULL DEFAULT '',
                summary   TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id          BIGSERIAL PRIMARY KEY,
                user_id     VARCHAR(255) NOT NULL,
                role        VARCHAR(50)  NOT NULL,
                content     TEXT         NOT NULL,
                is_compacted BOOLEAN     NOT NULL DEFAULT FALSE,
                created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_conv_user_id ON conversation_messages(user_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_conv_compacted ON conversation_messages(user_id, is_compacted)",
        ]

        with self.engine.begin() as conn:
            for stmt in ddl_statements:
                conn.execute(text(stmt.strip()))

        print("[postgres] Memory tables verified / created.")

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def context_for(self, user_id: str) -> dict[str, Any]:
        """Return durable facts, rolling summary, and the last N messages for *user_id* (O(1) LRU cached)."""
        uid = user_id.strip()
        now = time.time()

        with self._cache_lock:
            if uid in self._cache:
                ts, data = self._cache[uid]
                if now - ts < self._cache_ttl:
                    self._cache.move_to_end(uid)
                    return dict(data)

        with self.engine.connect() as conn:
            # Durable memory row
            row = (
                conn.execute(
                    text("SELECT facts, summary FROM user_memories WHERE user_id = :uid LIMIT 1"),
                    {"uid": uid},
                )
                .mappings()
                .first()
            )
            facts = row["facts"] if row else ""
            summary = row["summary"] if row else ""

            # Recent messages (oldest-first)
            recent = (
                conn.execute(
                    text(
                        """
                        SELECT role, content FROM (
                            SELECT id, role, content
                            FROM conversation_messages
                            WHERE user_id = :uid
                            ORDER BY id DESC
                            LIMIT :lim
                        ) sub
                        ORDER BY id ASC
                        """
                    ),
                    {"uid": uid, "lim": self.recent_messages},
                )
                .mappings()
                .all()
            )

        formatted = (
            f"Durable facts and preferences:\n{facts}\n\n"
            f"Rolling conversation summary:\n{summary}"
        ).strip()

        result = {
            "summary": formatted,
            "facts": facts,
            "rolling_summary": summary,
            "recent_messages": [{"role": r["role"], "content": r["content"]} for r in recent],
        }

        with self._cache_lock:
            if len(self._cache) >= self._cache_maxsize and uid not in self._cache:
                self._cache.popitem(last=False)
            self._cache[uid] = (now, dict(result))

        return result

    def message_count(self, user_id: str) -> int:
        """Return the number of *uncompacted* messages for a user."""
        uid = user_id.strip()
        with self.engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM conversation_messages "
                    "WHERE user_id = :uid AND is_compacted = FALSE"
                ),
                {"uid": uid},
            ).scalar()
        return int(count or 0)

    def get_all_messages(self, user_id: str) -> list[dict[str, str]]:
        """Return all conversation messages for a user in chronological order."""
        uid = user_id.strip()
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT role, content FROM conversation_messages "
                        "WHERE user_id = :uid ORDER BY id ASC"
                    ),
                    {"uid": uid},
                )
                .mappings()
                .all()
            )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_message(self, user_id: str, role: str, content: str) -> None:
        """Append a single chat message to persistent history and invalidate memory cache."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")

        uid = user_id.strip()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO conversation_messages (user_id, role, content, is_compacted, created_at)
                    VALUES (:uid, :role, :content, FALSE, :now)
                    """
                ),
                {"uid": uid, "role": role, "content": content, "now": datetime.now(timezone.utc)},
            )
        with self._cache_lock:
            self._cache.pop(uid, None)
            
    def clear_chat(self, user_id: str) -> None:
        """Delete all conversation messages and semantic memory for a user to start completely fresh."""
        uid = user_id.strip()
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM conversation_messages WHERE user_id = :uid"),
                {"uid": uid},
            )
            conn.execute(
                text("DELETE FROM user_memories WHERE user_id = :uid"),
                {"uid": uid},
            )
        with self._cache_lock:
            self._cache.pop(uid, None)
        print(f"[postgres] Cleared chat history and semantic memory for user '{uid}'.")


    # ------------------------------------------------------------------
    # Memory compaction
    # ------------------------------------------------------------------

    def compact_history(self, user_id: str, summarizer: Any) -> None:
        """Run the background summariser to extract durable facts and update the rolling summary.

        After extraction the older messages are flagged as *compacted* so only
        the most recent ``self.recent_messages`` remain active.
        """
        uid = user_id.strip()

        # 1. Load uncompacted messages (sliding window) for transcript
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT id, role, content FROM conversation_messages "
                        "WHERE user_id = :uid AND is_compacted = FALSE ORDER BY id ASC"
                    ),
                    {"uid": uid},
                )
                .mappings()
                .all()
            )

        if not rows:
            return

        current_ctx = self.context_for(uid)
        transcript = "\n".join(f"{r['role']}: {r['content']}" for r in rows)
        field_limit = max(1, self.summary_max_chars // 2)

        # 2. Summarise via the small background model
        if hasattr(summarizer, "summarize"):
            extracted = summarizer.summarize(transcript, existing_context=current_ctx["summary"])
        else:
            from memory_summarizer import MemorySummarizer

            prompt = MemorySummarizer.MEMORY_EXTRACTION_PROMPT.format(
                existing_context=current_ctx["summary"] or "None",
                transcript=transcript,
            )
            raw = summarizer.invoke(prompt)
            extracted = MemorySummarizer.parse_memory_json(getattr(raw, "content", str(raw)))

        facts = str(extracted.get("facts", "")).strip()[:field_limit]
        summary = str(extracted.get("summary", "")).strip()[:field_limit]
        now = datetime.now(timezone.utc)

        # 3. Upsert durable memory + mark old messages as compacted
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO user_memories (user_id, facts, summary, updated_at)
                    VALUES (:uid, :facts, :summary, :now)
                    ON CONFLICT (user_id) DO UPDATE
                        SET facts      = EXCLUDED.facts,
                            summary    = EXCLUDED.summary,
                            updated_at = EXCLUDED.updated_at
                    """
                ),
                {"uid": uid, "facts": facts, "summary": summary, "now": now},
            )
            conn.execute(
                text(
                    """
                    UPDATE conversation_messages
                    SET is_compacted = TRUE
                    WHERE user_id = :uid
                      AND id NOT IN (
                          SELECT id FROM conversation_messages
                          WHERE user_id = :uid
                          ORDER BY id DESC
                          LIMIT :keep
                      )
                    """
                ),
                {"uid": uid, "keep": self.recent_messages},
            )

        with self._cache_lock:
            self._cache.pop(uid, None)

        print(f"[postgres] Compacted memory for user '{uid}'.")

