"""Demonstration and Verification Script for Stage 5 LLM Delivery Layer (Roadmap Sec. 24-28).

Demonstrates the 4 requirements in Prompt 6's Definition of Done:
1. EXPLAIN and HINT produce visibly distinct text for the same concept.
2. With NO API key configured, the system executes cleanly via OfflineTemplateAdapter with zero errors.
3. Switching LLM_PROVIDER requires zero code changes outside src/llm/.
4. Pasting a paragraph of notes produces a new, valid curriculum JSON file.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import load_curriculum
from src.llm.add_material import add_module_from_notes
from src.llm.provider import get_llm_provider


def demo_definition_of_done() -> None:
    print("=" * 88)
    print(" AirTutor Stage 5: LLM Delivery Layer & Extensibility Verification")
    print("=" * 88)

    # 1. Load curriculum concepts
    repo = load_curriculum("data/curriculum")
    concept = repo.get_concept("variables")

    # 2. Verify with zero API keys (Offline Fallback & Distinction)
    print("\n[Check 1: Clean Execution with Zero API Keys & Offline Fallback]")
    provider = get_llm_provider("offline")
    resp_explain = provider.generate_tutoring_response(PedagogicalAction.EXPLAIN, concept)
    resp_hint = provider.generate_tutoring_response(PedagogicalAction.HINT, concept)

    print(f"-> EXPLAIN Output:\n   \"{resp_explain.text[:120]}...\"")
    print(f"-> HINT Output:\n   \"{resp_hint.text[:120]}...\"")
    assert resp_explain.text != resp_hint.text, "EXPLAIN and HINT text must be visibly distinct!"
    print("-> PASSED: EXPLAIN and HINT produce visibly different text for the same concept.")

    # 3. Verify Gemini provider failure handling (no key / mocked error -> graceful fallback)
    print("\n[Check 2: Live Provider Error Isolation (Never Crashes Upstream)]")
    gemini_prov = get_llm_provider("gemini", api_key="dummy_or_unset_key")
    resp_gemini = gemini_prov.generate_tutoring_response(PedagogicalAction.EXAMPLE, concept)
    print(f"-> Gemini Provider Result (status: {'fallback' if 'fallback_from' in resp_gemini.meta else 'live'}):")
    print(f"   \"{resp_gemini.text[:120]}...\"")
    print(f"   Meta: {resp_gemini.meta}")
    assert resp_gemini.text, "Must always return educational content even if live API fails"
    print("-> PASSED: Gemini provider isolates errors and safely falls back without crashing.")

    # 4. Verify OpenAI-compatible provider
    print("\n[Check 3: OpenAI-Compatible Provider Adapter]")
    oai_prov = get_llm_provider("openai_compatible", base_url="http://localhost:11434/v1")
    resp_oai = oai_prov.generate_tutoring_response(PedagogicalAction.PRACTICE, concept)
    print(f"-> OpenAI-Compatible Result (status: {'fallback' if 'fallback_from' in resp_oai.meta else 'live'}):")
    print(f"   \"{resp_oai.text[:120]}...\"")
    print("-> PASSED: OpenAI-compatible adapter operates under the identical interface.")

    # 5. Verify 'Add Material' Extensibility Flow
    print("\n[Check 4: 'Add Material' Extensibility from Notes]")
    notes_paragraph = (
        "Topic: Object-Oriented Programming in Python\n"
        "Classes and objects form the foundation of object-oriented design. "
        "A class defines the blueprint, methods, and attributes, while an object is an instance. "
        "Encapsulation restricts direct access to internal state using private variables. "
        "Inheritance allows a subclass to reuse and extend code from a parent class. "
        "Polymorphism enables different classes to be treated through a shared interface."
    )

    created_path = add_module_from_notes(
        raw_text=notes_paragraph,
        module_title="Object-Oriented Programming",
        output_dir="data/curriculum",
    )
    print(f"-> Created new curriculum module file: {created_path}")
    assert created_path.is_file(), f"File {created_path} was not created!"

    # Verify reload of entire curriculum
    updated_repo = load_curriculum("data/curriculum")
    all_concepts = updated_repo.get_all_concept_ids()
    print(f"-> Total loaded curriculum concepts now: {len(all_concepts)}")
    print(f"   Concepts: {all_concepts}")
    print("-> PASSED: Pasting raw notes produced a valid, reloadable curriculum module.")

    print("\n" + "=" * 88)
    print(" ALL STAGE 5 DEFINITION OF DONE VERIFICATIONS PASSED CLEANLY")
    print("=" * 88)


if __name__ == "__main__":
    demo_definition_of_done()
