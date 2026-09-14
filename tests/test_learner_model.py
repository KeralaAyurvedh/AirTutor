"""Unit tests for Observable Learner Model (Roadmap Sec. 8)."""

import pytest
from src.learner_model.mastery import LearnerModel


def test_learner_model_initialization():
    """Verify learner model initializes masteries to prior within [0, 1]."""
    concepts = ["variables", "loops"]
    model = LearnerModel(concepts=concepts, initial_mastery=0.15)

    assert model.get_mastery("variables") == pytest.approx(0.15)
    assert model.get_mastery("loops") == pytest.approx(0.15)


def test_correct_response_moves_mastery_up():
    """Verify that a correct answer strictly increases estimated mastery."""
    model = LearnerModel(concepts=["variables"], eta=0.20, delta=0.01, initial_mastery=0.30)
    # Target = 1.0, current = 0.30, steps_unpracticed = 0
    # Expected: 0.30 + 0.20 * (1.0 - 0.30) = 0.44
    new_m = model.update("variables", correct=True, steps_unpracticed=0)

    assert new_m > 0.30
    assert new_m == pytest.approx(0.44)
    assert model.get_mastery("variables") == pytest.approx(0.44)


def test_incorrect_response_moves_mastery_down():
    """Verify that an incorrect answer strictly decreases estimated mastery."""
    model = LearnerModel(concepts=["variables"], eta=0.20, delta=0.01, initial_mastery=0.60)
    # Target = 0.0, current = 0.60, steps_unpracticed = 0
    # Expected: 0.60 + 0.20 * (0.0 - 0.60) = 0.48
    new_m = model.update("variables", correct=False, steps_unpracticed=0)

    assert new_m < 0.60
    assert new_m == pytest.approx(0.48)


def test_unpracticed_concept_decays():
    """Verify unpracticed concepts decay over time."""
    model = LearnerModel(concepts=["variables", "loops"], eta=0.20, delta=0.05, initial_mastery=0.50)

    # Step decay on unpracticed concept
    model.step_decay_unpracticed(practiced_concept_id="variables")
    assert model.get_mastery("loops") == pytest.approx(0.45)
    assert model.get_mastery("variables") == pytest.approx(0.50)


def test_bounds_clipped_between_zero_and_one():
    """Verify estimated mastery never leaves [0.0, 1.0] under extreme updates."""
    model = LearnerModel(concepts=["variables"], eta=0.50, delta=0.10, initial_mastery=0.95)

    # Extreme correct updates
    for _ in range(10):
        model.update("variables", correct=True, steps_unpracticed=0)
    assert model.get_mastery("variables") <= 1.0

    # Extreme incorrect and decay updates
    for _ in range(20):
        model.update("variables", correct=False, steps_unpracticed=5)
    assert model.get_mastery("variables") >= 0.0
