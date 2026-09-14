"""UI and Application Integration Tests for AirTutor (Roadmap Sec. 31).

Verifies that the Streamlit application module loads, initializes its cached
resources (curriculum, policy, LLM provider), and executes an interaction step
without raising exceptions.
"""

from pathlib import Path
import pytest

from src.environment.actions import ACTIONS, PedagogicalAction
from src.environment.curriculum import load_curriculum
from src.environment.state_encoder import ObservableHistory, StateEncoderConfig, encode_state, state_tuple_to_index
from src.learner_model.mastery import LearnerModel
from src.llm.provider import LLMProvider, TutorResponse
from src.llm.providers.offline import OfflineTemplateAdapter
from src.rl.q_learning import TabularQAgent
from src.teachers.base_teacher import TeacherObservation


def test_app_imports_cleanly():
    """Verify that app.main imports with zero syntax or dependency errors."""
    import app.main
    assert hasattr(app.main, "main")
    assert hasattr(app.main, "get_cached_curriculum")
    assert hasattr(app.main, "get_cached_policy")


def test_app_cached_resources():
    """Verify cached resource loaders instantiate expected types."""
    from app.main import get_cached_curriculum, get_cached_policy, get_cached_provider

    curriculum = get_cached_curriculum()
    assert len(curriculum.list_modules()) >= 1

    policy = get_cached_policy()
    assert isinstance(policy, TabularQAgent)
    assert policy.q_table.shape == (320, 5)

    provider = get_cached_provider()
    assert provider is not None


def test_end_to_end_app_interaction_headless():
    """Verify that an end-to-end interaction step works without crashing."""
    curriculum = load_curriculum()
    module = curriculum.get_module(curriculum.list_modules()[0])
    concept = module.concepts[0]
    question = concept.questions[0]

    learner_model = LearnerModel(concepts=[c.concept_id for c in module.concepts])
    initial_m = learner_model.get_mastery(concept.concept_id)

    # 1. Update learner model on correct response
    new_m = learner_model.update(concept.concept_id, correct=True)
    assert new_m > initial_m

    # 2. Form observation
    obs = TeacherObservation(
        current_concept_id=concept.concept_id,
        estimated_mastery=new_m,
        all_estimated_masteries=learner_model.get_all_masteries(),
        prerequisite_ids=concept.prerequisites,
        weakest_prereq_id=None,
        weakest_prereq_mastery=None,
        last_response_correct=True,
        recent_responses=[True],
        wrong_answer_streak=0,
        hint_used_last_step=False,
        step_count=1,
    )

    # 3. Policy decision
    policy = TabularQAgent(seed=42)
    action = policy.select_action(obs, greedy=True)
    assert isinstance(action, PedagogicalAction)

    # 4. LLM provider rendering
    provider = OfflineTemplateAdapter()
    response = provider.generate_tutoring_response(
        action=action,
        concept=concept,
        learner_state=obs,
        curriculum_context={"module_title": module.title},
    )
    assert isinstance(response, TutorResponse)
    assert response.success is True
    assert len(response.text) > 0
