"""Unit tests for LLM Provider Abstraction & Adapters (Roadmap Sec. 24-26)."""

from unittest.mock import MagicMock, patch
import pytest

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import Concept
from src.llm.provider import LLMProvider, TutorResponse, get_llm_provider
from src.llm.providers.gemini import GeminiAdapter
from src.llm.providers.offline import OfflineTemplateAdapter
from src.llm.providers.openai_compatible import OpenAICompatibleAdapter


@pytest.fixture
def sample_concept() -> Concept:
    return Concept(
        concept_id="variables",
        name="Variables & Data Types",
        prerequisites=[],
        difficulty=1,
        templates={
            "EXPLAIN": "Variables store values in memory with dynamic typing.",
            "HINT": "Check integer division vs floor division behavior.",
        },
    )


def test_offline_adapter_distinct_actions(sample_concept):
    """Verify that EXPLAIN and HINT produce visibly different text for the same concept."""
    adapter = OfflineTemplateAdapter()

    resp_explain = adapter.generate_tutoring_response(
        action=PedagogicalAction.EXPLAIN,
        concept=sample_concept,
    )
    resp_hint = adapter.generate_tutoring_response(
        action=PedagogicalAction.HINT,
        concept=sample_concept,
    )

    assert resp_explain.success is True
    assert resp_hint.success is True
    assert resp_explain.text != resp_hint.text
    assert "Variables store values" in resp_explain.text
    assert "Check integer division" in resp_hint.text


def test_offline_adapter_generic_fallback():
    """Verify that concepts without pre-written templates use generic templates cleanly."""
    adapter = OfflineTemplateAdapter()
    new_concept = Concept(concept_id="recursion", name="Recursion", prerequisites=[], difficulty=3)

    resp = adapter.generate_tutoring_response(PedagogicalAction.EXPLAIN, new_concept)
    assert resp.success is True
    assert "Recursion" in resp.text


def test_gemini_adapter_mocked_success(sample_concept):
    """Verify GeminiAdapter handles successful API response properly."""
    adapter = GeminiAdapter(api_key="mock_key", model="gemini-2.5-flash")

    # Mock genai client and response
    mock_response = MagicMock()
    mock_response.text = "This is a live Gemini explanation of variables."

    adapter._client = MagicMock()
    adapter._client.models.generate_content.return_value = mock_response

    resp = adapter.generate_tutoring_response(PedagogicalAction.EXPLAIN, sample_concept)
    assert resp.success is True
    assert resp.text == "This is a live Gemini explanation of variables."
    assert resp.meta["provider"] == "gemini"


def test_gemini_adapter_fallback_on_api_error(sample_concept):
    """Verify GeminiAdapter falls back to OfflineTemplateAdapter without raising an exception."""
    adapter = GeminiAdapter(api_key="mock_key", model="gemini-2.5-flash")
    adapter._client = MagicMock()
    adapter._client.models.generate_content.side_effect = RuntimeError("API Quota Exceeded (429)")

    resp = adapter.generate_tutoring_response(PedagogicalAction.EXPLAIN, sample_concept)

    # Must NOT crash, and must return the offline template content
    assert resp is not None
    assert "Variables store values" in resp.text
    assert resp.meta["fallback_from"] == "gemini"
    assert "API Quota Exceeded" in resp.meta["error"]


def test_gemini_adapter_no_key_fallback(sample_concept):
    """Verify that with no key configured, GeminiAdapter immediately serves offline fallback."""
    adapter = GeminiAdapter(api_key=None)
    adapter._client = None

    resp = adapter.generate_tutoring_response(PedagogicalAction.HINT, sample_concept)
    assert resp is not None
    assert "Check integer division" in resp.text
    assert resp.meta["fallback_from"] == "gemini"


def test_openai_compatible_mocked_success(sample_concept):
    """Verify OpenAICompatibleAdapter parses standard /chat/completions response."""
    adapter = OpenAICompatibleAdapter(
        api_key="mock_key",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )

    fake_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is an OpenAI-compatible explanation of variables.",
                }
            }
        ]
    }

    with patch.object(adapter, "_post_chat_completion", return_value=fake_response["choices"][0]["message"]["content"]):
        resp = adapter.generate_tutoring_response(PedagogicalAction.EXPLAIN, sample_concept)
        assert resp.success is True
        assert resp.text == "This is an OpenAI-compatible explanation of variables."
        assert resp.meta["provider"] == "openai_compatible"


def test_openai_compatible_fallback_on_connection_error(sample_concept):
    """Verify OpenAICompatibleAdapter falls back to offline template on connection error."""
    adapter = OpenAICompatibleAdapter(
        api_key="mock_key",
        model="gpt-4o-mini",
        base_url="http://localhost:11434/v1",
    )

    with patch.object(adapter, "_post_chat_completion", side_effect=ConnectionRefusedError("Endpoint unavailable")):
        resp = adapter.generate_tutoring_response(PedagogicalAction.HINT, sample_concept)
        assert resp is not None
        assert "Check integer division" in resp.text
        assert resp.meta["fallback_from"] == "openai_compatible"


def test_get_llm_provider_factory():
    """Verify get_llm_provider returns requested adapter type."""
    p_offline = get_llm_provider("offline")
    assert isinstance(p_offline, OfflineTemplateAdapter)

    p_gemini = get_llm_provider("gemini")
    assert isinstance(p_gemini, GeminiAdapter)

    p_oai = get_llm_provider("openai_compatible")
    assert isinstance(p_oai, OpenAICompatibleAdapter)

    # Unknown defaults to offline
    p_unknown = get_llm_provider("non_existent_provider")
    assert isinstance(p_unknown, OfflineTemplateAdapter)
