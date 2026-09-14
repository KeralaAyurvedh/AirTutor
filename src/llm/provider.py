"""Abstract LLM Provider Interface and Factory for AirTutor (Roadmap Sec. 24-26).

Decouples pedagogical decision-making (RL policy) from content delivery (LLM rendering).
Nothing outside src/llm/ ever imports a concrete provider adapter.

CRITICAL INVARIANTS:
1. Provider-agnostic: RL, environment, and simulator code NEVER import specific providers.
2. Config-driven: Provider selection is governed exclusively by environment variables via python-dotenv.
3. Offline viability: In the absence of an API key or internet connection, the system safely operates
   headlessly using OfflineTemplateAdapter without crashing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import Concept

# Load environment variables from .env if present
load_dotenv()


@dataclass
class TutorResponse:
    """Standardized response container returned by all LLM providers."""
    text: str
    success: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "success": self.success,
            "meta": dict(self.meta),
        }


class LLMProvider(ABC):
    """Abstract interface that all tutoring content providers must implement."""

    @abstractmethod
    def generate_tutoring_response(
        self,
        action: PedagogicalAction | str,
        concept: Concept | str,
        learner_state: Any = None,
        curriculum_context: Any = None,
    ) -> TutorResponse:
        """Render a pedagogical action into educational natural-language content."""
        pass

    @abstractmethod
    def generate_raw(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate raw text/JSON from a prompt (used for curriculum generation/extensibility)."""
        pass


def get_llm_provider(
    provider_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> LLMProvider:
    """Factory function: reads configuration and returns the appropriate LLMProvider instance.

    Provider selection precedence:
    1. Explicit `provider_type` argument
    2. `LLM_PROVIDER` environment variable
    3. Defaults to "offline" (zero network dependency, zero API key required)
    """
    prov = (provider_type or os.getenv("LLM_PROVIDER", "offline")).strip().lower()

    # Lazily import concrete adapters to preserve provider-agnostic encapsulation
    if prov == "gemini":
        from src.llm.providers.gemini import GeminiAdapter
        gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        gemini_model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        to = float(timeout or os.getenv("LLM_TIMEOUT", "15"))
        return GeminiAdapter(api_key=gemini_key, model=gemini_model, timeout=to)

    elif prov in {"openai_compatible", "openai", "ollama", "vllm"}:
        from src.llm.providers.openai_compatible import OpenAICompatibleAdapter
        oai_key = api_key or os.getenv("LLM_API_KEY")
        oai_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        oai_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        to = float(timeout or os.getenv("LLM_TIMEOUT", "15"))
        return OpenAICompatibleAdapter(api_key=oai_key, model=oai_model, base_url=oai_url, timeout=to)

    elif prov == "offline":
        from src.llm.providers.offline import OfflineTemplateAdapter
        return OfflineTemplateAdapter()

    else:
        # Unknown provider requested -> fall back gracefully to OfflineTemplateAdapter
        from src.llm.providers.offline import OfflineTemplateAdapter
        return OfflineTemplateAdapter()
