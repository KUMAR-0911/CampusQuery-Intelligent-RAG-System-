"""Lightweight open-source background memory summarizer for CampusQuery."""

from __future__ import annotations

import json
import re
from typing import Any

from config import RetrievalConfig, DEFAULT_CONFIG


class MemorySummarizer:
    """Uses a small, fast open-source model (e.g., Llama-3.2-1B, Qwen-1.5B) for background memory summarization."""

    MEMORY_EXTRACTION_PROMPT = (
        "You are an efficient background memory extractor for an intelligent campus assistant.\n"
        "Analyze the conversation transcript below and extract memory.\n"
        "You MUST respond ONLY with a valid JSON object containing exactly two keys:\n"
        "  - \"facts\": Permanent user details (e.g. name, key skills, core goals). Retain existing facts and add any newly discovered ones.\n"
        "  - \"summary\": Exactly 2 to 3 lines of chaining context. Apply a rolling/sliding window: when you extract new context, you must drop the old, less relevant lines to strictly keep this to 2 to 3 lines total.\n\n"
        "Do NOT include greetings, system meta-instructions, or invented details.\n"
        "Do NOT output markdown fences like ```json or any other text outside the JSON object.\n\n"
        "Existing Known Context (Facts & Summary):\n{existing_context}\n\n"
        "Recent Conversation Window:\n{transcript}\n\n"
        "JSON Result:"
    )

    def __init__(self, config: RetrievalConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.provider = (config.summarizer_provider or "groq").lower()
        self.model = config.summarizer_model or "llama-3.2-1b-preview"
        self._llm = None
        self._hf_client = None

    def preload(self) -> None:
        """Eagerly load and initialize the background summarizer model client on application startup."""
        self._get_llm()
        print(f"[memory_summarizer] Background memory summarizer preloaded successfully ({self.provider}: {self.model}).")


    def _get_llm(self) -> Any:
        """Initialize or return the cached small model client without duplicate code."""
        if self._llm is not None:
            return self._llm

        if self.provider == "huggingface":
            token = getattr(self.config, "hf_memory_token", None) or self.config.hf_token
            if not token:
                raise ValueError("Set HF_MEMORY_TOKEN (or HF_TOKEN) for Hugging Face background summarizer.")
            from huggingface_hub import InferenceClient

            provider_choice = getattr(self.config, "summarizer_inference_provider", "nscale")
            self._hf_client = InferenceClient(
                model=self.model,
                token=token,
                provider=provider_choice,
            )
            print(f"[memory_summarizer] Hugging Face InferenceClient initialized: {self.model} (provider={provider_choice})")
            return self._hf_client

        # Configure OpenAI-compatible endpoint for Groq, Ollama, or custom host
        headers: dict[str, str] = {}
        base_url: str | None = None

        if self.provider == "groq":
            if self.config.groq_api_key:
                api_key = self.config.groq_api_key
                base_url = "https://api.groq.com/openai/v1"
            elif self.config.portkey_api_key:
                api_key = self.config.portkey_api_key
                base_url = "https://api.portkey.ai/v1"
                if self.config.portkey_config_slug:
                    headers["x-portkey-config"] = self.config.portkey_config_slug
            else:
                raise ValueError("Set GROQ_API_KEY (or PORTKEY_API_KEY) for the Groq summarizer.")

        elif self.provider == "ollama":
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"

        else:
            api_key = self.config.groq_api_key or self.config.portkey_api_key or "sk-dummy"

        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url=base_url,
            default_headers=headers if headers else None,
            temperature=self.config.summarizer_temperature,
            max_tokens=self.config.summarizer_max_tokens,
        )
        print(f"[memory_summarizer] {self.provider.capitalize()} small model initialized: {self.model}")
        return self._llm

    @staticmethod
    def parse_memory_json(raw_text: str) -> dict[str, str]:
        """Extract 'facts' and 'summary' from model text output."""
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {
                    "facts": str(data.get("facts", "")).strip(),
                    "summary": str(data.get("summary", "")).strip(),
                }
            except json.JSONDecodeError:
                pass
        return {"facts": "", "summary": raw_text[:1000].strip()}

    def summarize(self, transcript: str, existing_context: str = "") -> dict[str, str]:
        """Summarize conversation transcript into durable facts and a rolling summary."""
        if not transcript.strip():
            return {"facts": "", "summary": ""}

        prompt_template = getattr(self.config, "memory_extraction_prompt", None) or self.MEMORY_EXTRACTION_PROMPT
        prompt = prompt_template.format(
            existing_context=existing_context or "None",
            transcript=transcript,
        )

        print(f"\n[memory_summarizer] Starting memory extraction for transcript ({len(transcript)} chars)...")
        print(f"[memory_summarizer] Model: {self.model} | Provider: {self.provider} ({getattr(self.config, 'summarizer_inference_provider', '')})")

        try:
            if self.provider == "huggingface":
                client = self._get_llm()
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.summarizer_max_tokens,
                    temperature=self.config.summarizer_temperature,
                )
                raw = response.choices[0].message.content or ""
            else:
                llm = self._get_llm()
                resp = llm.invoke(prompt)
                raw = getattr(resp, "content", str(resp)).strip()

            print(f"[memory_summarizer] Raw output from {self.model}:\n{raw[:200]}...")
            parsed = self.parse_memory_json(raw)
            print(f"[memory_summarizer] Successfully extracted facts: {parsed.get('facts')}")
            print(f"[memory_summarizer] Rolling summary: {parsed.get('summary')}\n")
            return parsed

        except Exception as exc:
            print(f"[memory_summarizer] Summarization error with model {self.model}: {exc}")
            return {"facts": "", "summary": ""}
