"""Google Gemini LLM Provider Adapter for AirTutor (Roadmap Sec. 24 & 26).

Connects to Google's official Gemini API using the modern `google-genai` SDK.
Requires GEMINI_API_KEY or LLM_API_KEY set in environment or passed via constructor.

CRITICAL INVARIANT (Roadmap Sec. 26):
Every external call is wrapped in try/except with timeout protection.
ANY error — missing API key, network timeout, rate limits, quota exhaustion,
or service downtime — automatically logs a warning and falls back to OfflineTemplateAdapter.
This provider NEVER crashes upstream RL or experiment code.
"""

import logging
import os
import time
from typing import Any

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import Concept
from src.llm.provider import LLMProvider, TutorResponse
from src.llm.providers.offline import OfflineTemplateAdapter

logger = logging.getLogger("AirTutor.LLM.Gemini")

# Systematic action prompts mapping pedagogical action to pedagogical instructions
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


class GeminiAdapter(LLMProvider):
    """Adapter for Google GenAI API (google-genai package)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        timeout: float = 15.0,
    ) -> None:
        self.api_key: str | None = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model: str = model or "gemini-2.5-flash"
        self.timeout: float = float(timeout)
        self.provider_name: str = "gemini"
        self.offline_fallback: OfflineTemplateAdapter = OfflineTemplateAdapter()

        self._client: Any = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize google.genai Client: {e}")
                self._client = None

    def _build_tutoring_prompt(
        self,
        action_name: str,
        concept_name: str,
        difficulty: int,
        learner_state: Any,
        curriculum_context: Any,
    ) -> tuple[str, str]:
        """Construct pedagogical system instruction and user prompt."""
        system_instruction = (
            "You are AirTutor, an adaptive educational AI tutor for computer science. "
            "A tabular reinforcement learning policy has selected an optimal "
            "pedagogical intervention for this step based on the student's observable learning state. "
            "Your role is strictly to deliver that intervention in natural language with high educational clarity."
        )

        instruction = ACTION_INSTRUCTIONS.get(
            action_name,
            f"Deliver a pedagogical intervention of type {action_name} for {concept_name}."
        )

        prompt_lines = [
            f"Target Concept: {concept_name} (Curriculum Difficulty: {difficulty}/5)",
            f"Pedagogical Action Selected by RL: {action_name}",
            f"Action Directive: {instruction}",
        ]

        if learner_state:
            prompt_lines.append(f"Student Observable State Context: {learner_state}")
        if curriculum_context:
            prompt_lines.append(f"Curriculum Context: {curriculum_context}")

        prompt_lines.append(
            "\nDeliver the response now. Keep it direct, educational, concise, and focused on the selected action."
        )

        return system_instruction, "\n".join(prompt_lines)

    def generate_tutoring_response(
        self,
        action: PedagogicalAction | str,
        concept: Concept | str,
        learner_state: Any = None,
        curriculum_context: Any = None,
    ) -> TutorResponse:
        """Render action into natural language using Gemini, falling back on error."""
        act_name = action.name if isinstance(action, PedagogicalAction) else str(action).upper()
        cname = concept.name if isinstance(concept, Concept) else str(concept).replace("_", " ").title()
        diff = concept.difficulty if isinstance(concept, Concept) else 2
        cid = concept.concept_id if isinstance(concept, Concept) else str(concept)

        # 1. Graceful offline fallback if client is unconfigured
        if not self._client or not self.api_key:
            logger.info("Gemini API key not configured; serving response via OfflineTemplateAdapter.")
            fallback_resp = self.offline_fallback.generate_tutoring_response(
                action=action,
                concept=concept,
                learner_state=learner_state,
                curriculum_context=curriculum_context,
            )
            fallback_resp.meta["fallback_from"] = "gemini"
            fallback_resp.meta["fallback_reason"] = "no_api_key_configured"
            return fallback_resp

        system_instruction, user_prompt = self._build_tutoring_prompt(
            action_name=act_name,
            concept_name=cname,
            difficulty=diff,
            learner_state=learner_state,
            curriculum_context=curriculum_context,
        )

        # 2. Execute live API call with exception isolation
        start_time = time.time()
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=500,
                ),
            )

            latency_ms = (time.time() - start_time) * 1000.0
            text = (response.text or "").strip()

            if not text:
                raise ValueError("Gemini returned empty response text")

            return TutorResponse(
                text=text,
                success=True,
                meta={
                    "provider": self.provider_name,
                    "model": self.model,
                    "action": act_name,
                    "concept_id": cid,
                    "latency_ms": latency_ms,
                },
            )

        except Exception as e:
            logger.warning(f"Gemini API invocation failed ({type(e).__name__}: {e}). Triggering offline fallback.")
            fallback_resp = self.offline_fallback.generate_tutoring_response(
                action=action,
                concept=concept,
                learner_state=learner_state,
                curriculum_context=curriculum_context,
            )
            fallback_resp.meta["fallback_from"] = "gemini"
            fallback_resp.meta["fallback_reason"] = f"{type(e).__name__}: {e}"
            fallback_resp.meta["error"] = str(e)
            return fallback_resp

    def generate_raw(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate raw text or JSON from a prompt, with fallback protection."""
        if not self._client or not self.api_key:
            return self.offline_fallback.generate_raw(prompt, system_instruction)

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=0.2,  # Lower temperature for structured generation
                max_output_tokens=2000,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                return self.offline_fallback.generate_raw(prompt, system_instruction)
            return text

        except Exception as e:
            logger.warning(f"Gemini generate_raw failed ({e}). Falling back to offline generator.")
            return self.offline_fallback.generate_raw(prompt, system_instruction)
