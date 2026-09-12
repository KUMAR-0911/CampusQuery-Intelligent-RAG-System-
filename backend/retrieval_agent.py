"""Direct RAG pipeline for CampusQuery: Query Rewriting -> Direct Retrieval -> Reasoning Model Answer Generation."""

from __future__ import annotations

import json
import re
from typing import Any

from config import RetrievalConfig
from RETRIVAL.pgvectorstore import PgVectorStore


class RetrievalAgent:
    """Sequential RAG pipeline: Query Rewriter -> Direct Retrieval Function -> Reasoning Answer Model."""

    def __init__(self, config: RetrievalConfig) -> None:
        if not config.portkey_api_key:
            raise ValueError("Set PORTKEY_API_KEY in .env before using the retrieval agent.")

        if not config.portkey_config_slug:
            raise ValueError(
                "Set PORTKEY_CONFIG_SLUG to the saved Portkey config slug, "
                "for example pc-xxxxxxxx."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "Install LangChain OpenAI integration with pip install -r requirements.txt"
            ) from exc

        self.config = config
        self._last_retrieved_chunks: list[dict[str, Any]] = []

        raw_model = config.groq_model or "qwen3:32b-a3b"
        if "qwen3" in raw_model.lower() and ("32b" in raw_model.lower() or "30b" in raw_model.lower()):
            resolved_model = "qwen/qwen3-30b-a3b"
        else:
            resolved_model = raw_model

        # Reasoning & answer generation model via Portkey / Groq
        self.llm = ChatOpenAI(
            model=resolved_model,
            api_key=config.portkey_api_key,
            base_url="https://api.portkey.ai/v1",
            default_headers={
                "x-portkey-config": config.portkey_config_slug
            },
            temperature=0.1,
        )

        self.query_rewriter_model = getattr(config, "query_rewriter_model", "Qwen/Qwen3-0.6B")
        self.query_rewriter_provider = getattr(config, "query_rewriter_provider", "featherless-ai")
        self._hf_rewriter_client = None
        if config.hf_token:
            try:
                from huggingface_hub import InferenceClient
                client_kwargs = {"token": config.hf_token, "model": self.query_rewriter_model}
                provider_choice = self.query_rewriter_provider or getattr(config, "hf_inference_provider", None)
                if provider_choice:
                    client_kwargs["provider"] = provider_choice
                self._hf_rewriter_client = InferenceClient(**client_kwargs)
                print(f"[pipeline] Initialized Hugging Face query rewriter: {self.query_rewriter_model} (provider={client_kwargs.get('provider')})")
            except Exception as e:
                print(f"[pipeline] Notice: Could not initialize HF rewriter client: {e}")

        print(f"[pipeline] Direct Retrieval Pipeline initialized with reasoning model: {config.groq_model}")

    def rewrite_and_classify(
        self,
        question: str,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Classify user intent and rewrite the query into an evidence-seeking, context-aware query using Qwen3."""
        history_str = ""
        summary_str = ""
        recent_count = 0
        
        if conversation_context:
            summary = str(conversation_context.get("summary", "")).strip()
            if summary:
                summary_str = f"User Profile & Conversation Memory:\n{summary}\n\n"
            
            recent = conversation_context.get("recent_messages", [])[-6:]
            recent_count = len(recent)
            if recent:
                lines = []
                for m in recent:
                    role = str(m.get("role", "user")).capitalize()
                    content = str(m.get("content", "")).strip()
                    if content:
                        lines.append(f"{role}: {content}")
                if lines:
                    history_str = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            f"{self.config.query_rewriter_prompt}\n\n"
            f"{summary_str}"
            f"{history_str}"
            f"Current User Question: {question}\n\n"
            f"CONTEXT-AWARE INSTRUCTION: Analyze the current question in light of the conversation history and memory above. "
            f"Resolve any pronouns (it, they, that, my, those) and implicit references. "
            f"Produce a standalone, keyword-rich retrieval query for matching against resume sections.\n\n"
            f"Output JSON:"
        )

        print(f"[rewrite] Context-Aware Query Rewriting: {recent_count} recent turns, {len(summary_str)} chars profile memory.")

        raw = None
        # 1. Attempt query rewriting with Qwen/Qwen3-0.6B via Hugging Face
        if self._hf_rewriter_client is not None:
            try:
                hf_res = self._hf_rewriter_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.1,
                )
                msg = hf_res.choices[0].message
                content_text = (msg.content or "").strip()
                if not content_text and hasattr(msg, "reasoning") and msg.reasoning:
                    content_text = str(msg.reasoning).strip()
                if content_text:
                    raw = content_text
                    print(f"[rewrite] Successfully used {self.query_rewriter_model} for query rewrite.")
            except Exception as hf_err:
                print(f"[rewrite] Hugging Face {self.query_rewriter_model} attempt notice: {hf_err}. Falling back to primary LLM.")

        # 2. Fallback to primary LLM if Hugging Face was unavailable or returned empty
        if not raw:
            try:
                response = self.llm.invoke([{"role": "user", "content": prompt}])
                raw = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"[rewrite] Primary LLM rewrite failed: {e}")

        if raw:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    intent = data.get("intent", "TARGETED").strip().upper()
                    rewritten_query = data.get("rewritten_query", question).strip()
                    if intent not in ["TARGETED", "SKILL_GAP", "JOB_MATCH", "RESUME_SCORE"]:
                        intent = "TARGETED"
                    return {"intent": intent, "rewritten_query": rewritten_query}
                except Exception as parse_err:
                    print(f"[rewrite] JSON parse error: {parse_err}")

        return {"intent": "TARGETED", "rewritten_query": question}

    def retrieve_documents(
        self,
        query: str,
        store: PgVectorStore,
        intent: str = "TARGETED",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Direct retrieval function call to PgVectorStore with intent-tuned parameters."""
        print("[retrieval] ============================================")
        print(f"[retrieval] Direct function call with Query: {query}")
        print(f"[retrieval] Classified Intent: {intent}")

        intent_clean = (intent or "TARGETED").strip().upper()
        if intent_clean == "TARGETED":
            top_k = 5
            rerank_top_n = 10
        elif intent_clean == "SKILL_GAP":
            top_k = 10
            rerank_top_n = 20
        elif intent_clean == "JOB_MATCH":
            top_k = 15
            rerank_top_n = 30
        elif intent_clean == "RESUME_SCORE":
            top_k = 20
            rerank_top_n = 40
        else:
            top_k = self.config.top_k
            rerank_top_n = self.config.rerank_top_n

        print(f"[retrieval] Applied parameters: top_k={top_k}, rerank_top_n={rerank_top_n}")

        raw_chunks = store.search(
            query,
            metadata_filters=metadata_filters,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )

        # Deduplicate chunks
        unique_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for chunk in raw_chunks:
            metadata = chunk.get("metadata", {})
            chunk_id = str(chunk.get("id", metadata.get("id", chunk.get("text", ""))))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        print(f"\n[retrieval] ==================== [RETRIEVED DOCUMENT CHUNKS ({len(unique_chunks)})] ====================")
        for i, chunk in enumerate(unique_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source_file") or meta.get("source") or "document"
            sig_score = float(chunk.get("rerank_score", 0.0))
            raw_logit = float(chunk.get("raw_score", sig_score))
            heading = chunk.get("heading") or meta.get("heading") or meta.get("title") or meta.get("section") or f"Section {i}"
            print(f"--- [Chunk {i}] Heading: {heading} | Source: {src} | ID: {chunk.get('id')} ---")
            print(f"    Relevance (Sigmoid): {sig_score:.4f} | Raw Logit: {raw_logit:+.4f}")
            print(f"    Content:")
            for line in chunk.get("text", "").strip().splitlines():
                print(f"      {line}")
            print()
        print("[retrieval] =============================================================================\n")
        return unique_chunks

    @staticmethod
    def _clean_output_text(text: str) -> str:
        """Thoroughly strip all *, #, --, citations, unnecessary divider symbols, and excessive spacing from response text."""
        if not text:
            return ""
        # 0. Strip citation markers and references ([1], [Source: ...], [Document Chunk 1])
        cleaned = re.sub(r'\[\d+\]', '', text)
        cleaned = re.sub(r'\[Document Chunk\s*\d*.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[Source:?.*?\]', '', cleaned, flags=re.IGNORECASE)
        # Strip trailing citations or sources section
        cleaned = re.sub(r'(?i)(^|\n)(citations?|sources?):\s*[\s\S]*$', '', cleaned)
        # 1. Strip markdown headings (# Title -> Title)
        cleaned = re.sub(r'(?m)^[ \t]*#{1,6}[ \t]*', '', cleaned)
        # 2. Strip all # characters anywhere in the response
        cleaned = cleaned.replace('#', '')
        # 3. Strip bold / italic asterisks (**text** or *text* -> text)
        cleaned = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', cleaned)
        # 4. Strip bullet asterisks at line starts (* item -> item) and any remaining *
        cleaned = re.sub(r'(?m)^[ \t]*\*[ \t]+', '', cleaned)
        cleaned = cleaned.replace('*', '')
        # 5. Strip horizontal rules / dividers (---, ---, etc.)
        cleaned = re.sub(r'(?m)^[ \t]*-{2,}[ \t]*$', '', cleaned)
        # 6. Strip double hyphens (-- or --- or more) anywhere in the text
        cleaned = re.sub(r'-{2,}', '', cleaned)
        # 7. Replace unicode dashes (em-dash, en-dash) with a space
        cleaned = cleaned.replace('—', ' ').replace('–', ' ')
        # 8. Strip leading bullet dashes (- item -> item) if followed by space
        cleaned = re.sub(r'(?m)^[ \t]*-[ \t]+', '', cleaned)
        # 9. Clean horizontal spacing on each line (collapse 2+ spaces, trim ends)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.splitlines()]
        cleaned = '\n'.join(lines)
        # 10. Collapse multiple empty lines between list items (e.g., '1. Item\n\n2. Item' -> '1. Item\n2. Item')
        cleaned = re.sub(r'(\d+\..*?)\n\n+(?=\d+\.)', r'\1\n', cleaned)
        # 11. Clean up multiple blank lines resulting from stripped headers/dividers
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def answer(
        self,
        question: str,
        store: PgVectorStore,
        metadata_filters: dict[str, Any] | None = None,
        memory: Any | None = None,
        user_id: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute Direct RAG: Query Rewriting & Intent Detection -> Direct Retrieval Function -> Reasoning Model."""

        print("\n==================== [RAG PIPELINE START] ====================")
        print(f"[pipeline] Original user question: \"{question}\"")

        if conversation_context is None and memory is not None and user_id:
            conversation_context = memory.context_for(user_id)

        # Step 1: PRE-RETRIEVAL: Query Rewriting Model & Intent Detection
        print(f"[pipeline:step-1] PRE-RETRIEVAL: Invoking Query Rewriter & Intent Detector ({self.query_rewriter_model} on {self.query_rewriter_provider})...")
        analysis = self.rewrite_and_classify(question, conversation_context=conversation_context)
        intent = analysis["intent"]
        rewritten_query = analysis["rewritten_query"]
        print(f"[pipeline:step-1] PRE-RETRIEVAL: Detected Intent: {intent}")
        print(f"[pipeline:step-1] PRE-RETRIEVAL: Rewritten Query: \"{rewritten_query}\"")

        # Step 2: RETRIEVAL: Direct Retrieval Function Call with rewritten query & intent tuning
        print(f"[pipeline:step-2] RETRIEVAL: Executing direct retrieval & Cross-Encoder reranker...")
        chunks = self.retrieve_documents(
            rewritten_query,
            store=store,
            intent=intent,
            metadata_filters=metadata_filters,
        )
        self._last_retrieved_chunks = chunks
        print(f"[pipeline:step-2] RETRIEVAL: Selected top {len(chunks)} grounded chunk(s) for answer reasoning.")

        # Step 3: Reasoning Model (Answer Generation)
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        # Build retrieved document context
        if chunks:
            context_blocks = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                source_name = meta.get("source_file") or meta.get("source") or "uploaded document"
                score = chunk.get("rerank_score", 0.0)
                chunk_text = chunk.get("text", "").strip()
                context_blocks.append(f"[Document Chunk {i} | Source: {source_name} | Rerank Score: {score:.4f}]\n{chunk_text}")
            retrieved_context_str = "RETRIEVED DOCUMENT CONTEXT:\n" + "\n\n".join(context_blocks)
        else:
            retrieved_context_str = (
                "RETRIEVED DOCUMENT CONTEXT:\n"
                "[No relevant document chunks found in vector database. "
                "If the user is asking about their resume, job description, or document analysis, "
                "politely inform them that no document has been uploaded yet, and invite them to upload their resume.]"
            )

        messages: list[Any] = []

        # System prompt with strict constraint against *, #, and --
        messages.append(SystemMessage(content=self.config.system_prompt))

        # Persistent user memory / profile
        profile_context = ""
        if conversation_context:
            summary = conversation_context.get("summary", "").strip()
            if summary:
                profile_context = f"PERSISTENT USER PROFILE AND MEMORY:\n{summary}\n\n"
                print(f"[pipeline:step-3] Injected persistent memory profile ({len(summary)} chars)")

        # Conversation turns for multi-turn awareness
        if conversation_context:
            recent = conversation_context.get("recent_messages", [])
            for m in recent:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            if recent:
                print(f"[pipeline:step-3] Injected {len(recent)} recent conversational turn(s)")

        # Grounded reasoning user prompt
        user_message_content = (
            f"{profile_context}"
            f"{retrieved_context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"Instruction for reasoning and answer generation:\n"
            f"Generate a thorough, professional, and well-structured answer based strictly on the retrieved document context.\n"
            f"STRICT SYMBOL & SPACING CONSTRAINTS:\n"
            f"1. Absolutely DO NOT use any asterisks anywhere. No bold text, no italic text, and no bullet points.\n"
            f"2. Absolutely DO NOT use any hashtag, hash, or pound symbols anywhere. No markdown headers.\n"
            f"3. Absolutely DO NOT use any double hyphens, dashes, or horizontal line dividers anywhere.\n"
            f"4. Use plain UPPERCASE titles followed by a colon for section headings (for example: EXTRACTED COMPONENTS:).\n"
            f"5. For lists, use standard numbers (1., 2., 3.) or plain sentences without bullet symbols.\n"
            f"6. Keep output cleanly spaced and compact: do NOT insert unnecessary empty lines or multiple blank lines between list items.\n"
            f"7. Absolutely DO NOT output any citation markers, chunk references (like [Document Chunk 1]), or source citations in the answer."
        )
        messages.append(HumanMessage(content=user_message_content))

        print(f"[pipeline:step-3] Invoking Reasoning Model: {self.llm.model_name} via Portkey Gateway...")
        try:
            response = self.llm.invoke(messages)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                answer = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                answer = str(content)
            print(f"[pipeline:step-3] Raw answer received from {self.llm.model_name} ({len(answer)} chars).")
        except Exception as exc:
            print(f"[pipeline:step-3] Reasoning model generation error: {exc}")
            answer = "I encountered an error generating the answer. Please try again."

        # Clean all markdown headings (#), asterisks (*, **), double hyphens (--), and dividers
        answer = self._clean_output_text(answer)
        print(f"[pipeline:step-3] Post-processed answer (stripped #, *, --). Final length: {len(answer)} chars.")

        citations = list(dict.fromkeys(
            chunk.get("metadata", {}).get("source_file") or chunk.get("metadata", {}).get("source", "")
            for chunk in chunks
            if chunk.get("metadata", {}).get("source_file") or chunk.get("metadata", {}).get("source")
        ))

        print(f"[pipeline] Completed pipeline answer generation ({len(chunks)} chunks retrieved, {len(citations)} citations).")
        print("==================== [RAG PIPELINE END] ====================\n")

        serialized_chunks = []
        for c in chunks:
            sig_score = float(c.get("rerank_score", 0.0))
            raw_logit = float(c.get("raw_score", sig_score))
            meta = c.get("metadata", {})
            heading = c.get("heading") or meta.get("heading") or meta.get("title") or meta.get("section") or "Section"
            serialized_chunks.append({
                "id": str(c.get("id", "")),
                "text": str(c.get("text", "")),
                "rerank_score": sig_score,
                "raw_score": raw_logit,
                "heading": str(heading),
                "source": str(meta.get("source_file") or meta.get("source") or "document"),
                "metadata": {k: str(v) for k, v in meta.items() if k not in ["embedding"]},
            })

        return {
            "answer": answer,
            "citations": citations,
            "retrieved": bool(chunks),
            "retrieved_chunks": serialized_chunks,
        }
