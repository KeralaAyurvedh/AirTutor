"""Unit tests for Teacher Baselines (Roadmap Sec. 16)."""

import pytest

from src.environment.actions import ACTIONS, PedagogicalAction
from src.teachers.base_teacher import TeacherObservation
from src.teachers.fixed_teacher import FixedTeacher
from src.teachers.random_teacher import RandomTeacher
from src.teachers.rule_teacher import RuleTeacher


def make_obs(
    concept_id: str = "variables",
    estimated_mastery: float = 0.50,
    last_response_correct: bool | None = True,
    wrong_answer_streak: int = 0,
    weakest_prereq_id: str | None = None,
    weakest_prereq_mastery: float | None = None,
) -> TeacherObservation:
    return TeacherObservation(
        current_concept_id=concept_id,
        estimated_mastery=estimated_mastery,
        all_estimated_masteries={concept_id: estimated_mastery},
        weakest_prereq_id=weakest_prereq_id,
        weakest_prereq_mastery=weakest_prereq_mastery,
        last_response_correct=last_response_correct,
        wrong_answer_streak=wrong_answer_streak,
    )


def test_random_teacher_coverage():
    """Verify RandomTeacher outputs valid actions and covers the entire action space."""
    teacher = RandomTeacher(seed=123)
    obs = make_obs()
    chosen_actions = {teacher.select_action(obs) for _ in range(100)}

    assert len(chosen_actions) == 5
    assert chosen_actions == set(ACTIONS)


def test_fixed_teacher_alternation():
    """Verify FixedTeacher strictly alternates EXPLAIN -> PRACTICE on each concept."""
    teacher = FixedTeacher()

    # Concept A: Step 1 -> EXPLAIN, Step 2 -> PRACTICE
    obs_a = make_obs(concept_id="variables")
    assert teacher.select_action(obs_a) == PedagogicalAction.EXPLAIN
    assert not teacher.should_advance_concept(obs_a)

    assert teacher.select_action(obs_a) == PedagogicalAction.PRACTICE
    assert teacher.should_advance_concept(obs_a)

    # Concept B: Step 1 -> EXPLAIN, Step 2 -> PRACTICE
    obs_b = make_obs(concept_id="conditionals")
    assert teacher.select_action(obs_b) == PedagogicalAction.EXPLAIN
    assert teacher.select_action(obs_b) == PedagogicalAction.PRACTICE


def test_rule_teacher_decision_boundaries():
    """Verify RuleTeacher follows Roadmap Sec. 16 heuristic rules."""
    teacher = RuleTeacher()

    # Rule 1: Weak prerequisite (< 0.40) triggers REVIEW_PREREQUISITE
    obs_weak_prereq = make_obs(
        concept_id="conditionals",
        estimated_mastery=0.50,
        weakest_prereq_id="variables",
        weakest_prereq_mastery=0.30,
    )
    assert teacher.select_action(obs_weak_prereq) == PedagogicalAction.REVIEW_PREREQUISITE

    # Reset review counter for remaining checks
    teacher.reset()

    # Rule 2: Low mastery (< 0.40) triggers EXPLAIN
    obs_low = make_obs(estimated_mastery=0.25, last_response_correct=True)
    assert teacher.select_action(obs_low) == PedagogicalAction.EXPLAIN

    # Rule 3: Last answer wrong triggers HINT
    obs_wrong = make_obs(estimated_mastery=0.50, last_response_correct=False, wrong_answer_streak=1)
    assert teacher.select_action(obs_wrong) == PedagogicalAction.HINT

    # Rule 4: Moderate mastery [0.40, 0.70) triggers EXAMPLE
    obs_mid = make_obs(estimated_mastery=0.55, last_response_correct=True)
    assert teacher.select_action(obs_mid) == PedagogicalAction.EXAMPLE

    # Rule 5: High mastery (>= 0.70) triggers PRACTICE
    obs_high = make_obs(estimated_mastery=0.72, last_response_correct=True)
    assert teacher.select_action(obs_high) == PedagogicalAction.PRACTICE


def test_rule_teacher_advance_condition():
    """Verify RuleTeacher advances when concept mastery reaches threshold."""
    teacher = RuleTeacher(mastery_threshold=0.75)

    obs_unmastered = make_obs(estimated_mastery=0.60)
    assert not teacher.should_advance_concept(obs_unmastered)

    obs_mastered = make_obs(estimated_mastery=0.80)
    assert teacher.should_advance_concept(obs_mastered)
