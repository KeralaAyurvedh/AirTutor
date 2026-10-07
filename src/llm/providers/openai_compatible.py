"""OpenAI-Compatible LLM Provider Adapter for AirTutor (Roadmap Sec. 24 & 26).

Connects to any standard `/chat/completions` API endpoint:
- OpenAI (GPT-4o, GPT-4o-mini)
- Grok / xAI
- Local Ollama (`http://localhost:11434/v1`)
- vLLM / LM Studio / LocalAI

Uses standard library `urllib.request` to ensure zero additional dependencies
and maximum CPU efficiency.

CRITICAL INVARIANT (Roadmap Sec. 26):
Every call is wrapped in timeout protection and try/except.
Any failure falls back automatically to OfflineTemplateAdapter without crashing.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import Concept
from src.llm.provider import LLMProvider, TutorResponse
from src.llm.providers.offline import OfflineTemplateAdapter

logger = logging.getLogger("AirTutor.LLM.OpenAICompatible")

# Systematic action prompts
ACTION_INSTRUCTIONS: dict[str, str] = {
    "EXPLAIN": (
        "Provide a clear, pedagogical, intuitive explanation of the concept. "
        "Define terms cleanly, highlight the core mechanism, and explain why it matters."
    ),
    "EXAMPLE": (
        "Provide a concrete, annotated code example demonstrating this concept in action. "
        "Explain step-by-step what each line does and the expected output."
    ),
    "HINT": (
        "Provide a targeted scaffolding hint. Do NOT give away the complete solution. "
        "Guide the student's thinking toward identifying the correct approach or boundary case."
    ),
    "PRACTICE": (
        "Pose a targeted practice question or mini-exercise on this concept. "
        "Encourage the student to attempt it, specifying what input/output or behavior to target."
    ),
    "REVIEW_PREREQUISITE": (
        "Review the foundational prerequisite concept. Explain how mastering this prerequisite "
        "directly enables understanding the current concept."
    ),
}


class OpenAICompatibleAdapter(LLMProvider):
    """Adapter for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 15.0,
    ) -> None:
        self.api_key: str | None = api_key or os.getenv("LLM_API_KEY")
        self.model: str = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url: str = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout: float = float(timeout or os.getenv("LLM_TIMEOUT", "15"))
        self.provider_name: str = "openai_compatible"
        self.offline_fallback: OfflineTemplateAdapter = OfflineTemplateAdapter()

    def _post_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Make HTTP POST to /chat/completions endpoint with timeout."""
        endpoint = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AirTutor/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw_body = resp.read().decode("utf-8")
            data = json.loads(raw_body)
            content = data["choices"][0]["message"]["content"]
            return str(content).strip()

    def generate_tutoring_response(
        self,
        action: PedagogicalAction | str,
        concept: Concept | str,
        learner_state: Any = None,
        curriculum_context: Any = None,
    ) -> TutorResponse:
        """Render action into natural language via OpenAI-compatible endpoint."""
        act_name = action.name if isinstance(action, PedagogicalAction) else str(action).upper()
        cname = concept.name if isinstance(concept, Concept) else str(concept).replace("_", " ").title()
        diff = concept.difficulty if isinstance(concept, Concept) else 2
        cid = concept.concept_id if isinstance(concept, Concept) else str(concept)

        instruction = ACTION_INSTRUCTIONS.get(
            act_name,
            f"Deliver a pedagogical intervention of type {act_name} for {cname}."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are AirTutor, an adaptive educational AI tutor for computer science. "
                    "A tabular reinforcement learning policy has selected an optimal "
                    "pedagogical intervention for this step based on the student's observable learning state. "
                    "Your role is strictly to deliver that intervention in natural language with high educational clarity."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target Concept: {cname} (Difficulty: {diff}/5)\n"
                    f"Pedagogical Action Selected by RL: {act_name}\n"
                    f"Action Directive: {instruction}\n"
                    f"Student Observable State Context: {learner_state}\n\n"
                    "Deliver the tutoring response now."
                ),
            },
        ]

        start_time = time.time()
        try:
            # If endpoint is remote OpenAI and no key is configured, trigger fallback immediately
            if "openai.com" in self.base_url and not self.api_key:
                raise ValueError("No API key provided for remote OpenAI endpoint")

            text = self._post_chat_completion(messages, temperature=0.7, max_tokens=500)
            if not text:
                raise ValueError("Empty response received from OpenAI-compatible endpoint")

            latency_ms = (time.time() - start_time) * 1000.0
            return TutorResponse(
                text=text,
                success=True,
                meta={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "action": act_name,
                    "concept_id": cid,
                    "latency_ms": latency_ms,
                },
            )

        except Exception as e:
            logger.warning(
                f"OpenAI-compatible call failed ({type(e).__name__}: {e}). Triggering offline fallback."
            )
            fallback_resp = self.offline_fallback.generate_tutoring_response(
                action=action,
                concept=concept,
                learner_state=learner_state,
                curriculum_context=curriculum_context,
            )
            fallback_resp.meta["fallback_from"] = "openai_compatible"
            fallback_resp.meta["fallback_reason"] = f"{type(e).__name__}: {e}"
            fallback_resp.meta["error"] = str(e)
            return fallback_resp

    def generate_raw(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate raw text or JSON from prompt with fallback protection."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            if "openai.com" in self.base_url and not self.api_key:
                raise ValueError("No API key configured for remote endpoint")

            return self._post_chat_completion(messages, temperature=0.2, max_tokens=2000)

        except Exception as e:
            logger.warning(f"OpenAI-compatible generate_raw failed ({e}). Falling back to offline generator.")
            return self.offline_fallback.generate_raw(prompt, system_instruction)
