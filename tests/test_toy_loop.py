"""Unit tests verifying the mechanical tabular Q-learning loop for Stage 0."""

import pytest
from src.rl.toy_loop_demo import TabularQLearningAgent, ToyChainEnv


def test_q_table_updates_after_step():
    """Verify that a step update correctly modifies the Q-table via the Bellman equation."""
    agent = TabularQLearningAgent(
        num_states=5,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.0,  # Pure greedy
        seed=1,
    )

    # Initially, all Q-values must be zero
    assert agent.q_table[1, 1] == 0.0

    # Execute non-terminal update: s=1, a=1, r=2.0, next_s=2, done=False
    # Expected target: r + gamma * max_a Q(next_s, a) = 2.0 + 0.9 * 0.0 = 2.0
    # Expected new Q: 0.0 + 0.1 * (2.0 - 0.0) = 0.20
    td_err = agent.update(state=1, action=1, reward=2.0, next_state=2, done=False)

    assert td_err == pytest.approx(2.0)
    assert agent.q_table[1, 1] == pytest.approx(0.20)
    assert agent.q_table[1, 0] == 0.0  # Untouched action remains 0


def test_q_table_terminal_update():
    """Verify terminal Bellman update ignores bootstrap value."""
    agent = TabularQLearningAgent(
        num_states=5,
        num_actions=2,
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.0,
        seed=1,
    )
    # Give next_state some non-zero value that should be ignored if done=True
    agent.q_table[4, 0] = 5.0
    agent.q_table[4, 1] = 8.0

    td_err = agent.update(state=3, action=1, reward=10.0, next_state=4, done=True)
    # Terminal target = reward = 10.0
    # Expected new Q: 0.0 + 0.2 * (10.0 - 0.0) = 2.0
    assert td_err == pytest.approx(10.0)
    assert agent.q_table[3, 1] == pytest.approx(2.0)


def test_toy_chain_env_mechanics():
    """Verify ToyChainEnv step and reset transitions."""
    env = ToyChainEnv(length=5, max_steps=10)
    init_state = env.reset()
    assert init_state == 0

    next_state, reward, done = env.step(1)
    assert next_state == 1
    assert reward == pytest.approx(-0.1)
    assert not done
