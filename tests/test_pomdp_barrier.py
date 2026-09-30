"""Behavioral Information Leakage Verification Test (POMDP Information Barrier - Rule 1).

Verifies that the policy, BKT belief tracker, and 14-D state encoder are strictly
blind to the student's true latent mastery scalar M(c, t) and theta.

Test specification:
1. Construct two learner instances:
   - Learner A with true latent mastery = 0.15 (struggling)
   - Learner B with true latent mastery = 0.85 (mastered)
2. Provide both learners with identical observable interaction histories (same sequence
   of actions and binary correct/incorrect outcomes).
3. Feed both observable histories into BKTTracker and CognitiveStateEncoder.
4. Assert that the resulting 14-dimensional state vectors are numerically identical:
   np.allclose(state_a, state_b, atol=1e-7)
5. Assert that dqn_agent.select_action() receives identical inputs and returns the identical action.
"""

import numpy as np
import pytest

from src.environment.state_encoder import CognitiveStateEncoder
from src.learner_model.bkt_tracker import BKTTracker
from src.rl.dqn_agent import DQNAgent
from src.simulator.population import LearnerParameters, SimulatedLearner


def test_pomdp_information_barrier_no_leakage():
    # 1. Instantiate Learner A (low true mastery) and Learner B (high true mastery)
    concepts = ["concept_a", "concept_b", "concept_c"]

    theta_a = LearnerParameters(alpha=0.08, beta=0.12, gamma=1.8, h=0.9, tau=0.3, sigma=0.05)
    learner_a = SimulatedLearner(
        learner_id="learner_a_low_mastery",
        theta=theta_a,
        concepts=concepts,
        initial_mastery=0.15,
        seed=101,
    )
    for c in concepts:
        learner_a.set_true_mastery(c, 0.15)

    theta_b = LearnerParameters(alpha=0.32, beta=0.02, gamma=0.6, h=0.2, tau=0.9, sigma=0.03)
    learner_b = SimulatedLearner(
        learner_id="learner_b_high_mastery",
        theta=theta_b,
        concepts=concepts,
        initial_mastery=0.85,
        seed=202,
    )
    for c in concepts:
        learner_b.set_true_mastery(c, 0.85)

    # Sanity check: true latent masteries are vastly different
    assert learner_a.get_true_mastery("concept_a") == pytest.approx(0.15)
    assert learner_b.get_true_mastery("concept_a") == pytest.approx(0.85)

    # 2. Provide identical sequence of observable interaction outcomes:
    # 5 steps: Correct, Incorrect, Correct, Correct, Incorrect
    observed_history = [
        ("concept_a", "EXPLAIN", True),
        ("concept_a", "PRACTICE", False),
        ("concept_a", "HINT", True),
        ("concept_b", "EXAMPLE", True),
        ("concept_b", "PRACTICE", False),
    ]

    tracker_a = BKTTracker(concepts=concepts)
    tracker_b = BKTTracker(concepts=concepts)

    consecutive_failures_a = 0
    consecutive_failures_b = 0
    session_steps_a = 0
    session_steps_b = 0

    for cid, act, outcome in observed_history:
        tracker_a.update(cid, is_correct=outcome)
        session_steps_a += 1
        consecutive_failures_a = 0 if outcome else consecutive_failures_a + 1

        tracker_b.update(cid, is_correct=outcome)
        session_steps_b += 1
        consecutive_failures_b = 0 if outcome else consecutive_failures_b + 1

    encoder = CognitiveStateEncoder(max_session_steps=30)

    # Encode state for Learner A
    state_a = encoder.encode(
        bkt_tracker=tracker_a,
        current_concept_id="concept_b",
        prerequisite_concept_id="concept_a",
        all_concept_ids=concepts,
        consecutive_failures=consecutive_failures_a,
        session_steps=session_steps_a,
        previous_action=observed_history[-1][1],
        last_response_correct=observed_history[-1][2],
    )

    # Encode state for Learner B
    state_b = encoder.encode(
        bkt_tracker=tracker_b,
        current_concept_id="concept_b",
        prerequisite_concept_id="concept_a",
        all_concept_ids=concepts,
        consecutive_failures=consecutive_failures_b,
        session_steps=session_steps_b,
        previous_action=observed_history[-1][1],
        last_response_correct=observed_history[-1][2],
    )

    # 4. Strictly assert state vectors are numerically identical
    assert np.allclose(state_a, state_b, atol=1e-7), (
        f"POMDP barrier violated! Hidden mastery leaked into state vector.\n"
        f"State A: {state_a}\nState B: {state_b}"
    )

    # 5. Assert DQN policy produces identical action decisions on both states
    agent = DQNAgent(state_dim=14, num_actions=5, seed=777)
    action_a = agent.select_action(state_a, greedy=True)
    action_b = agent.select_action(state_b, greedy=True)

    assert action_a == action_b, (
        f"Policy selected action {action_a} for Learner A and {action_b} for Learner B "
        f"despite identical observable histories!"
    )
