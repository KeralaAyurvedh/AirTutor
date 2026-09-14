"""Unit tests for Tabular Q-Learning Agent (Roadmap Sec. 7 & 13)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.environment.actions import ACTIONS, PedagogicalAction
from src.rl.q_learning import TabularQAgent
from src.teachers.base_teacher import TeacherObservation


@pytest.fixture
def dummy_obs() -> TeacherObservation:
    return TeacherObservation(
        current_concept_id="variables",
        estimated_mastery=0.45,
        all_estimated_masteries={"variables": 0.45},
        last_response_correct=True,
        recent_responses=[True, True],
        wrong_answer_streak=0,
        hint_used_last_step=False,
    )


def test_agent_initialization():
    """Verify Q-table dimensions and initial zero values."""
    agent = TabularQAgent(seed=42)
    assert agent.num_states == 320
    assert agent.num_actions == 5
    assert agent.q_table.shape == (320, 5)
    assert np.all(agent.q_table == 0.0)


def test_agent_state_mapping(dummy_obs):
    """Verify observable TeacherObservation maps to valid integer state index."""
    agent = TabularQAgent(seed=42)
    s_idx = agent.get_state_index(dummy_obs)
    assert 0 <= s_idx < 320


def test_bellman_update():
    """Verify standard temporal-difference Bellman update equation."""
    agent = TabularQAgent(learning_rate=0.10, discount_factor=0.90, seed=1)

    s = 10
    a = 2
    r = 1.0
    s_next = 20

    # Non-terminal update
    # Expected target = r + gamma * max_a' Q(s', a') = 1.0 + 0.9 * 0 = 1.0
    # Expected new Q = 0 + 0.1 * (1.0 - 0) = 0.10
    td_err = agent.update(s, a, r, s_next, done=False)
    assert td_err == pytest.approx(1.0)
    assert agent.q_table[s, a] == pytest.approx(0.10)

    # Terminal update
    agent.q_table[s_next, 0] = 5.0
    td_term = agent.update(s, a, 2.0, s_next, done=True)
    # Target = 2.0 (ignores next state Q)
    # Expected new Q = 0.10 + 0.1 * (2.0 - 0.10) = 0.29
    assert agent.q_table[s, a] == pytest.approx(0.29)


def test_epsilon_decay():
    """Verify exponential exploration decay down to epsilon_min."""
    agent = TabularQAgent(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.25)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.125)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.10)  # hit floor
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.10)


def test_save_and_load(dummy_obs):
    """Verify persistence of learned Q-values."""
    agent = TabularQAgent(seed=1)
    s_idx = agent.get_state_index(dummy_obs)
    agent.q_table[s_idx, 3] = 4.25

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_q_table.npy"
        agent.save(save_path)

        loaded_agent = TabularQAgent(seed=2)
        loaded_agent.load(save_path)

        assert loaded_agent.q_table[s_idx, 3] == pytest.approx(4.25)
        # Greedy selection should now pick action index 3
        chosen = loaded_agent.select_action(dummy_obs, greedy=True)
        assert chosen == ACTIONS[3]
