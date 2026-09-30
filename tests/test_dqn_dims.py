"""Unit tests for fixed-size 14-D cognitive state encoder and DQN agent dimensionality."""

import numpy as np
import pytest
import torch

from src.environment import state_encoder
from src.environment.state_encoder import CognitiveStateEncoder, encode_cognitive_state
from src.learner_model.bkt_tracker import BKTTracker
from src.rl.dqn_agent import DQNAgent, QNetwork


def test_state_encoder_curriculum_invariance():
    """Assert state encoder strictly outputs shape (14,) regardless of curriculum size."""
    # Small curriculum: 3 concepts
    tracker_small = BKTTracker(concepts=["var", "cond", "loops"])
    tracker_small.update("var", is_correct=True)

    state_small = state_encoder.encode(
        bkt_tracker=tracker_small,
        current_concept_id="cond",
        prerequisite_concept_id="var",
        all_concept_ids=["var", "cond", "loops"],
        consecutive_failures=1,
        session_steps=5,
        previous_action="EXPLAIN",
        last_response_correct=True,
    )
    assert isinstance(state_small, np.ndarray)
    assert state_small.shape == (14,)
    assert state_small.dtype == np.float32

    # Large curriculum: 25 concepts
    large_concepts = [f"concept_{i}" for i in range(25)]
    tracker_large = BKTTracker(concepts=large_concepts)
    for c in large_concepts[:10]:
        tracker_large.update(c, is_correct=True)

    encoder = CognitiveStateEncoder(max_session_steps=50)
    state_large = encoder.encode(
        bkt_tracker=tracker_large,
        current_concept_id=large_concepts[15],
        prerequisite_concept_id=large_concepts[14],
        all_concept_ids=large_concepts,
        consecutive_failures=3,
        session_steps=20,
        previous_action="PRACTICE",
        last_response_correct=False,
    )
    assert isinstance(state_large, np.ndarray)
    assert state_large.shape == (14,)

    # Huge curriculum: 100 concepts
    huge_concepts = [f"c_{i}" for i in range(100)]
    tracker_huge = BKTTracker(concepts=huge_concepts)
    state_huge = encode_cognitive_state(
        bkt_tracker=tracker_huge,
        current_concept_id=huge_concepts[0],
        all_concept_ids=huge_concepts,
    )
    assert state_huge.shape == (14,)


def test_dqn_action_selection_and_dimensions():
    """Assert dqn_agent.select_action(state) accepts state and outputs valid action in [0, 4]."""
    agent = DQNAgent(state_dim=14, num_actions=5, seed=42)

    # 1D array of shape (14,)
    dummy_state = np.random.uniform(0.0, 1.0, size=(14,)).astype(np.float32)
    action = agent.select_action(dummy_state, greedy=True)
    assert isinstance(action, int)
    assert 0 <= action < 5

    # Torch tensor
    tensor_state = torch.tensor(dummy_state)
    action_t = agent.select_action(tensor_state, greedy=False)
    assert isinstance(action_t, int)
    assert 0 <= action_t < 5


def test_dqn_training_and_serialization(tmp_path):
    """Test replay buffer transition storage, train step, and checkpoint save/load."""
    agent = DQNAgent(state_dim=14, num_actions=5, batch_size=4, buffer_capacity=100, seed=123)

    # Populate replay buffer with dummy transitions
    for i in range(10):
        s = np.random.uniform(0.0, 1.0, size=(14,)).astype(np.float32)
        a = i % 5
        r = 0.5
        next_s = np.random.uniform(0.0, 1.0, size=(14,)).astype(np.float32)
        done = (i == 9)
        agent.store_transition(s, a, r, next_s, done)

    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0

    # Serialization test
    ckpt_path = tmp_path / "test_dqn.pt"
    agent.save(ckpt_path)
    assert ckpt_path.exists()

    loaded_agent = DQNAgent(state_dim=14, num_actions=5, seed=999)
    loaded_agent.load(ckpt_path)

    # Policy weights should match exactly
    s = np.random.uniform(0.0, 1.0, size=(14,)).astype(np.float32)
    act_orig = agent.select_action(s, greedy=True)
    act_loaded = loaded_agent.select_action(s, greedy=True)
    assert act_orig == act_loaded


def test_time_to_mastery_calculation():
    """Regression test: verify steps_to_mastery correctly detects when all concepts reach BKT >= 0.85."""
    from src.environment.curriculum import load_module_from_file, CurriculumRepository
    from src.evaluation.run_dqn_benchmark import run_episode
    from src.simulator.population import FastLearner
    from src.teachers.rule_teacher import RuleTeacher

    mod = load_module_from_file("data/curriculum/gate_programming_fundamentals.json")
    curriculum = CurriculumRepository([mod])
    concept_ids = curriculum.get_all_concept_ids()

    # Fast learner with high mastery capacity
    steps_list = []
    for s in range(10):
        learner = FastLearner(seed=s, concepts=concept_ids)
        teacher = RuleTeacher()
        res = run_episode(
            policy_or_teacher=teacher,
            learner=learner,
            curriculum=curriculum,
            max_steps=30,
        )
        assert 1 <= res["steps_to_mastery"] <= 30
        assert 0.0 <= res["learning_gain"] <= 1.0
        steps_list.append(res["steps_to_mastery"])

    # Verify that not all episodes collapse to degenerate 30.00
    assert any(st < 30 for st in steps_list), "At least one FastLearner should achieve mastery before step 30"

