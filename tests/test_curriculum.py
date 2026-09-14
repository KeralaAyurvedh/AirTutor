"""Unit tests for Curriculum Architecture and Loader (Roadmap Sec. 28)."""

from pathlib import Path
import pytest

from src.environment.curriculum import (
    Concept,
    CurriculumModule,
    CurriculumRepository,
    load_curriculum,
    load_module_from_file,
)


def test_load_gate_curriculum():
    """Verify loading the seed GATE programming fundamentals curriculum."""
    repo = load_curriculum("data/curriculum")

    # Verify module existence
    assert "gate_programming_fundamentals" in repo.modules
    module = repo.get_module("gate_programming_fundamentals")
    assert len(module.concepts) == 5

    # Verify concepts
    concept_ids = repo.get_all_concept_ids()
    expected_ids = ["variables", "conditionals", "loops", "functions", "lists"]
    assert concept_ids == expected_ids

    # Verify each concept has 5-8 questions and valid difficulty
    for cid in expected_ids:
        concept = repo.get_concept(cid)
        assert 5 <= len(concept.questions) <= 8
        assert 1 <= concept.difficulty <= 5
        for q in concept.questions:
            assert q.question_id
            assert q.prompt
            assert q.answer
            assert q.type in ["mcq", "short", "code"]


def test_prerequisite_graph_validity():
    """Verify prerequisite dependencies are acyclic and resolvable."""
    repo = load_curriculum("data/curriculum")

    # 'variables' has no prerequisites
    assert repo.get_concept("variables").prerequisites == []

    # 'conditionals' depends on 'variables'
    cond_prereqs = repo.get_prerequisites("conditionals")
    assert [p.concept_id for p in cond_prereqs] == ["variables"]

    # 'lists' depends on 'loops' and 'functions'
    list_prereqs = repo.get_prerequisites("lists")
    assert sorted([p.concept_id for p in list_prereqs]) == ["functions", "loops"]


def test_unknown_prerequisite_raises_error():
    """Verify error raised if a concept references an unknown prerequisite."""
    invalid_concepts = [
        Concept(
            concept_id="c1",
            name="Concept 1",
            prerequisites=["non_existent"],
            difficulty=1,
            questions=[],
        )
    ]
    invalid_module = CurriculumModule("mod", "Mod", invalid_concepts)

    with pytest.raises(ValueError, match="references unknown prerequisite"):
        CurriculumRepository([invalid_module])
