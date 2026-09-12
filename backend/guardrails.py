"""Remote input and output guardrails for the CampusQuery chat flow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import RetrievalConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class GuardrailResult:
    """Decision returned by the input or output guardrail."""

    allowed: bool
    message: str = ""
    category: str = "allowed"
    confidence: float = 0.0


class CampusGuardrails:
    """Use Hugging Face model for safety decisions (supporting Llama Guard and classifier models)."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        model: str | None = None,
        token: str | None = None,
        provider: str | None = None,
        prompt: str | None = None,
        block_threshold: float | None = None,
    ) -> None:
        cfg = config or DEFAULT_CONFIG
        self.model = model or cfg.hf_guardrail_model
        self.token = token or cfg.hf_token
        self.provider = provider or getattr(cfg, "hf_inference_provider", "featherless-ai")
        self.prompt = prompt or cfg.resume_analyser_guardrail_prompt
        self.guard_block_threshold = block_threshold if block_threshold is not None else cfg.guard_block_threshold
        self.is_llama_guard = "llama-guard" in (self.model or "").lower()
        self._client = None

    def _get_client(self):
        if self._client is None and self.token:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise ImportError("Install huggingface_hub with pip install -r requirements.txt") from exc
            
            client_kwargs = {"model": self.model, "token": self.token}
            if self.provider:
                client_kwargs["provider"] = self.provider

            try:
                self._client = InferenceClient(**client_kwargs)
                print(
                    f"[guardrail] Hugging Face InferenceClient initialized "
                    f"for model={self.model}" + (f" with provider={self.provider}" if self.provider else "")
                )
            except Exception as e:
                print(f"[guardrail] Notice: InferenceClient init error: {e}")
        return self._client

    def preload(self) -> None:
        """Eagerly initialize the remote guardrail model client on application startup."""
        if self.token:
            self._get_client()

    def _remote_check(self, text: str, direction: str) -> GuardrailResult:
        if not self.token:
            print("[guardrail] HF_TOKEN not set. Allowing request.")
            return GuardrailResult(True, category="allowed", confidence=1.0)
        
        client = self._get_client()
        if client is None:
            print("[guardrail] Guardrail client could not be initialized. Allowing request as fallback.")
            return GuardrailResult(True, category="allowed", confidence=1.0)

        # Specialized handling for meta-llama/Llama-Guard models
        if self.is_llama_guard:
            role = "user" if direction == "input" else "assistant"
            messages = [{"role": role, "content": text}]
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=60,
                    temperature=0.0,
                )
                raw = (response.choices[0].message.content or "").strip().lower()
                print(f"[guardrail] Llama Guard response: {raw}")

                if raw.startswith("unsafe"):
                    parts = raw.split("\n")
                    violation = parts[1].strip().upper() if len(parts) > 1 else "UNSAFE_CONTENT"
                    return GuardrailResult(
                        allowed=False,
                        message=f"I cannot process this request because it was flagged by safety guardrails ({violation}). Please ask a relevant career or resume query.",
                        category=violation,
                        confidence=0.0,
                    )
                return GuardrailResult(True, category="allowed", confidence=1.0)
            except Exception as exc:
                exc_str = str(exc)
                if "403" in exc_str or "restricted" in exc_str or "gated" in exc_str:
                    print(
                        f"[guardrail] Notice: {self.model} requires accepting Meta's license at "
                        f"https://huggingface.co/{self.model}. Allowing request via fallback."
                    )
                else:
                    print(f"[guardrail] Notice: Guardrail request failed ({exc}). Allowing request via fallback.")
                return GuardrailResult(True, category="allowed", confidence=1.0)

        # Standard custom relevance prompt for generic LLMs
        output_instruction = """
This method is used as the only guardrail check in the application and may
receive either a user's question before retrieval or an assistant response.
Allow academic, campus, career, resume, technical, and document questions.
""" if direction == "output" else ""

        classifier_text = f"""{self.prompt}

{output_instruction}

Analyze the following {direction} text and determine how relevant it is to this application's domain (resumes, ATS, careers, academics).
Return ONLY a valid JSON object with a single key "score", containing a float between 0.0 and 1.0. 
1.0 means highly relevant/safe, 0.0 means completely irrelevant/unsafe. Do not include any other text or explanation.

Text to classify:
{text}"""

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": classifier_text}],
                max_tokens=80,
                temperature=0,
            )
        except Exception as exc:
            print(f"[guardrail] Guardrail request failed: {exc}. Allowing via fallback.")
            return GuardrailResult(True, category="allowed", confidence=1.0)

        raw = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return GuardrailResult(True, category="allowed", confidence=1.0)

        try:
            decision = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(decision.get("score", 1.0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            score = 1.0

        if score >= self.guard_block_threshold:
            return GuardrailResult(True, category="allowed", confidence=score)

        reason = f"This request was blocked because its relevance score ({score:.2f}) is below the threshold ({self.guard_block_threshold})."
        return GuardrailResult(
            False,
            "I can help build and analyse resumes, analyse job descriptions, match jobs, "
            "identify skill gaps, prepare cover letters and interviews, and answer career, "
            "education, campus, programming, and technical questions. " + reason,
            "blocked",
            score,
        )

    def check_input(self, text: str) -> GuardrailResult:
        """Ask the remote model to classify input, including greetings, PII, and injection."""
        return self._remote_check(text, "input")

    def check_output(self, text: str) -> GuardrailResult:
        """Ask the remote model to classify generated output for safety and relevance."""
        return self._remote_check(text, "output")
