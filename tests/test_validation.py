"""Unit tests for Curriculum Schema Validation and 'Add Material' flow (Roadmap Sec. 27)."""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest

from src.environment.curriculum import load_module_from_file
from src.llm.add_material import add_module_from_notes
from src.llm.providers.offline import OfflineTemplateAdapter
from src.llm.validation import validate_curriculum_json


@pytest.fixture
def valid_module_dict():
    return {
        "module_id": "test_module",
        "title": "Test Module Title",
        "concepts": [
            {
                "concept_id": "c1",
                "name": "Concept One",
                "prerequisites": [],
                "difficulty": 1,
                "questions": [
                    {
                        "question_id": "q1",
                        "prompt": "What is 1+1?",
                        "answer": "2",
                        "type": "short",
                    },
                    {
                        "question_id": "q2",
                        "prompt": "Select the correct option",
                        "answer": "Option A",
                        "type": "mcq",
                    },
                ],
            },
            {
                "concept_id": "c2",
                "name": "Concept Two",
                "prerequisites": ["c1"],
                "difficulty": 2,
                "questions": [
                    {
                        "question_id": "q3",
                        "prompt": "Write code to return True",
                        "answer": "return True",
                        "type": "code",
                    }
                ],
            },
        ],
    }


def test_validate_valid_dict(valid_module_dict):
    """Verify valid module passes validation with no errors."""
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_valid_json_string(valid_module_dict):
    """Verify valid JSON string and markdown code fence cleaning."""
    json_str = "```json\n" + json.dumps(valid_module_dict) + "\n```"
    is_valid, errors = validate_curriculum_json(json_str)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_missing_root_fields():
    """Verify detection of missing required root fields."""
    bad_data = {"module_id": "only_id"}
    is_valid, errors = validate_curriculum_json(bad_data)
    assert is_valid is False
    assert any("Missing required root field 'title'" in e for e in errors)
    assert any("Missing required root field 'concepts'" in e for e in errors)


def test_validate_duplicate_concept_ids(valid_module_dict):
    """Verify duplicate concept_ids are detected and rejected."""
    valid_module_dict["concepts"].append(
        {
            "concept_id": "c1",  # duplicate!
            "name": "Duplicate C1",
            "prerequisites": [],
            "difficulty": 3,
            "questions": [{"question_id": "q4", "prompt": "p", "answer": "a", "type": "short"}],
        }
    )
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is False
    assert any("Duplicate 'concept_id' detected: 'c1'" in e for e in errors)


def test_validate_invalid_question_type(valid_module_dict):
    """Verify question types outside ['mcq', 'short', 'code'] are rejected."""
    valid_module_dict["concepts"][0]["questions"][0]["type"] = "essay"
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is False
    assert any("invalid type 'essay'" in e for e in errors)


def test_validate_empty_prompt_or_answer(valid_module_dict):
    """Verify empty prompt or answer is rejected."""
    valid_module_dict["concepts"][0]["questions"][0]["prompt"] = "   "
    valid_module_dict["concepts"][0]["questions"][1]["answer"] = ""
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is False
    assert any("empty 'prompt'" in e for e in errors)
    assert any("empty 'answer'" in e for e in errors)


def test_validate_unknown_prerequisite(valid_module_dict):
    """Verify unknown prerequisite concept is flagged."""
    valid_module_dict["concepts"][0]["prerequisites"] = ["unknown_concept_99"]
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is False
    assert any("references unknown prerequisite 'unknown_concept_99'" in e for e in errors)


def test_validate_self_and_circular_prerequisites(valid_module_dict):
    """Verify self-referencing and circular prerequisite chains are flagged."""
    # Self reference
    valid_module_dict["concepts"][0]["prerequisites"] = ["c1"]
    is_valid, errors = validate_curriculum_json(valid_module_dict)
    assert is_valid is False
    assert any("references itself as a prerequisite" in e for e in errors)

    # Circular chain: c1 requires c2, and c2 requires c1
    valid_module_dict["concepts"][0]["prerequisites"] = ["c2"]
    valid_module_dict["concepts"][1]["prerequisites"] = ["c1"]
    is_valid_cycle, cycle_errors = validate_curriculum_json(valid_module_dict)
    assert is_valid_cycle is False
    assert any("Circular prerequisite dependency detected" in e for e in cycle_errors)


def test_add_module_from_notes_offline_end_to_end():
    """Verify add_module_from_notes generates a valid, loadable module file via OfflineTemplateAdapter."""
    notes = (
        "Topic: Object-Oriented Programming\n"
        "Key concepts include Classes and Objects, Encapsulation, Inheritance, and Polymorphism. "
        "A class defines the blueprint, while an object is an instance."
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_file = add_module_from_notes(
            raw_text=notes,
            module_title="Object-Oriented Programming",
            output_dir=tmp_dir,
            provider=OfflineTemplateAdapter(),
        )

        assert dest_file.is_file()
        # Verify it loads cleanly with our curriculum loader
        module = load_module_from_file(dest_file)
        assert module.module_id == "object_oriented_programming"
        assert len(module.concepts) >= 2


def test_add_module_from_notes_retry_success(valid_module_dict):
    """Verify retry behavior: initial malformed output is corrected on retry."""
    mock_provider = MagicMock()
    # First response is invalid JSON; second response is valid JSON
    mock_provider.generate_raw.side_effect = [
        "Sorry, I cannot format this right now.",
        json.dumps(valid_module_dict),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_file = add_module_from_notes(
            raw_text="Some notes here",
            module_title="Test Retry",
            output_dir=tmp_dir,
            provider=mock_provider,
        )
        assert dest_file.is_file()
        assert mock_provider.generate_raw.call_count == 2


def test_add_module_from_notes_double_failure_raises():
    """Verify that if output remains invalid after retry, ValueError is raised with error details."""
    mock_provider = MagicMock()
    mock_provider.generate_raw.return_value = '{"invalid": "schema"}'

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError) as excinfo:
            add_module_from_notes(
                raw_text="Some notes here",
                module_title="Fail Module",
                output_dir=tmp_dir,
                provider=mock_provider,
            )
        assert "Curriculum generation failed schema validation after retry" in str(excinfo.value)
        assert mock_provider.generate_raw.call_count == 2
