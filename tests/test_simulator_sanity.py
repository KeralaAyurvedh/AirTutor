"""The Five Required Simulator Sanity Checks (Roadmap Sec. 12).

Automated unit tests proving the simulator obeys fundamental pedagogical and
statistical invariants before any RL training is conducted:
1. Monotonic in mastery: Higher M(c, t) -> Higher P(correct)
2. Monotonic in difficulty: Higher difficulty -> Lower P(correct)
3. Monotonic in alpha: Higher alpha -> Faster mastery growth under fixed actions
4. Monotonic in beta: Higher beta -> Faster mastery decay when unpracticed
5. Monotonic in sigma: Higher sigma -> Higher variance in observed response probabilities
"""

import numpy as np
import pytest

from src.environment.actions import PedagogicalAction
from src.simulator.dynamics import (
    compute_correctness_probability,
    step_simulation,
)
from src.simulator.population import LearnerParameters, SimulatedLearner


@pytest.fixture
def base_theta() -> LearnerParameters:
    return LearnerParameters(
        alpha=0.20,
        beta=0.05,
        gamma=1.0,
        h=0.50,
        tau=0.50,
        sigma=0.05,
    )


def test_sanity_check_1_monotonic_in_mastery(base_theta):
    """Sanity Check 1: Higher M(c, t) -> higher P(correct), holding difficulty fixed."""
    difficulty = 3
    action = PedagogicalAction.PRACTICE
    mastery_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    probs = [
        compute_correctness_probability(
            current_mastery=m,
            difficulty=difficulty,
            action=action,
            theta=base_theta,
            apply_noise=False,
        )
        for m in mastery_levels
    ]

    # Verify strictly monotonic increasing
    for i in range(len(probs) - 1):
        assert probs[i + 1] > probs[i], (
            f"Expected P(correct|M={mastery_levels[i+1]}) > P(correct|M={mastery_levels[i]}), "
            f"got {probs[i+1]:.4f} <= {probs[i]:.4f}"
        )


def test_sanity_check_2_monotonic_in_difficulty(base_theta):
    """Sanity Check 2: Higher difficulty -> lower P(correct), holding mastery fixed."""
    mastery = 0.50
    action = PedagogicalAction.PRACTICE
    difficulties = [1, 2, 3, 4, 5]

    probs = [
        compute_correctness_probability(
            current_mastery=mastery,
            difficulty=d,
            action=action,
            theta=base_theta,
            apply_noise=False,
        )
        for d in difficulties
    ]

    # Verify strictly monotonic decreasing
    for i in range(len(probs) - 1):
        assert probs[i + 1] < probs[i], (
            f"Expected P(correct|diff={difficulties[i+1]}) < P(correct|diff={difficulties[i]}), "
            f"got {probs[i+1]:.4f} >= {probs[i]:.4f}"
        )


def test_sanity_check_3_monotonic_in_alpha():
    """Sanity Check 3: Higher alpha -> faster mastery growth under a fixed action sequence."""
    # Fast learner vs. slow learner
    theta_slow = LearnerParameters(alpha=0.08, beta=0.03, gamma=1.0, h=0.5, tau=0.5, sigma=0.02)
    theta_fast = LearnerParameters(alpha=0.32, beta=0.03, gamma=1.0, h=0.5, tau=0.5, sigma=0.02)

    learner_slow = SimulatedLearner("slow", theta_slow, concepts=["variables"], initial_mastery=0.10, seed=10)
    learner_fast = SimulatedLearner("fast", theta_fast, concepts=["variables"], initial_mastery=0.10, seed=10)

    # Apply identical action sequence with forced correct answers to isolate alpha's effect
    actions = [
        PedagogicalAction.EXPLAIN,
        PedagogicalAction.PRACTICE,
        PedagogicalAction.EXAMPLE,
        PedagogicalAction.PRACTICE,
        PedagogicalAction.PRACTICE,
    ]

    for act in actions:
        step_simulation(learner_slow, action=act, concept_id="variables", question_difficulty=2, force_correct=True)
        step_simulation(learner_fast, action=act, concept_id="variables", question_difficulty=2, force_correct=True)

    m_slow = learner_slow.get_true_mastery("variables")
    m_fast = learner_fast.get_true_mastery("variables")

    assert m_fast > m_slow, f"Expected fast learner ({m_fast:.4f}) > slow learner ({m_slow:.4f})"


def test_sanity_check_4_monotonic_in_beta():
    """Sanity Check 4: Higher beta -> faster mastery decay when unpracticed."""
    # Low forgetting vs. high forgetting
    theta_low_decay = LearnerParameters(alpha=0.20, beta=0.02, gamma=1.0, h=0.5, tau=0.5, sigma=0.02)
    theta_high_decay = LearnerParameters(alpha=0.20, beta=0.14, gamma=1.0, h=0.5, tau=0.5, sigma=0.02)

    learner_low = SimulatedLearner("low_decay", theta_low_decay, concepts=["variables", "loops"], initial_mastery=0.80, seed=20)
    learner_high = SimulatedLearner("high_decay", theta_high_decay, concepts=["variables", "loops"], initial_mastery=0.80, seed=20)

    # 4 steps practicing 'loops', leaving 'variables' unpracticed
    for _ in range(4):
        step_simulation(learner_low, action=PedagogicalAction.PRACTICE, concept_id="loops", question_difficulty=2)
        step_simulation(learner_high, action=PedagogicalAction.PRACTICE, concept_id="loops", question_difficulty=2)

    decayed_low = learner_low.get_true_mastery("variables")
    decayed_high = learner_high.get_true_mastery("variables")

    assert decayed_high < decayed_low, (
        f"Expected higher beta learner ({decayed_high:.4f}) to decay faster than lower beta ({decayed_low:.4f})"
    )


def test_sanity_check_5_monotonic_in_sigma():
    """Sanity Check 5: Higher noise sigma -> higher variance in observed response probabilities."""
    theta_low_noise = LearnerParameters(alpha=0.20, beta=0.05, gamma=1.0, h=0.5, tau=0.5, sigma=0.02)
    theta_high_noise = LearnerParameters(alpha=0.20, beta=0.05, gamma=1.0, h=0.5, tau=0.5, sigma=0.10)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    n_samples = 1500
    probs_low = [
        compute_correctness_probability(
            current_mastery=0.50,
            difficulty=3,
            action=PedagogicalAction.PRACTICE,
            theta=theta_low_noise,
            apply_noise=True,
            rng=rng1,
        )
        for _ in range(n_samples)
    ]

    probs_high = [
        compute_correctness_probability(
            current_mastery=0.50,
            difficulty=3,
            action=PedagogicalAction.PRACTICE,
            theta=theta_high_noise,
            apply_noise=True,
            rng=rng2,
        )
        for _ in range(n_samples)
    ]

    var_low = float(np.var(probs_low))
    var_high = float(np.var(probs_high))

    assert var_high > var_low * 3.0, (
        f"Expected high-sigma variance ({var_high:.6f}) to be significantly greater than low-sigma ({var_low:.6f})"
    )
