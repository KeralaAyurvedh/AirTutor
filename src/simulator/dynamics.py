"""Simulator Dynamics for AirTutor (Roadmap Sec. 11).

CRITICAL ARCHITECTURAL DESIGN RULE:
Action effects must emerge from smooth functions of latent parameters theta and
current continuous mastery M(c, t) — NEVER from a lookup table keyed on a learner 'type'.

Implements:
1. Question-level correctness probability P(correct) modulated by:
   - Mastery-difficulty gap
   - Difficulty sensitivity gamma
   - Action boost (e.g. Hint scaled by h)
   - Stochastic noise sigma
2. Mastery update:
   M(c, t+1) = clip(M(c, t) + alpha * A_effect(action, M(c, t), theta) * correct
                    - beta * steps_unpracticed(c), 0, 1)
"""

from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from src.environment.actions import PedagogicalAction
from src.simulator.population import LearnerParameters, SimulatedLearner

# ============================================================================
# --- Tunable Pedagogical Action Dynamics Constants (Roadmap Sec. 11) ---
# All constants below are tunable hyperparameters of the research simulator.
# ============================================================================

# EXPLAIN: Conceptual clarity contribution, largely independent of theta.
EXPLAIN_BASE_EFFECT: float = 0.55

# EXAMPLE: Base multiplier for concrete examples, scaled smoothly by (1 - alpha).
# Slower learners (lower alpha) benefit relatively more from concrete illustration.
EXAMPLE_BASE_EFFECT: float = 0.85

# HINT: Immediate correctness probability boost factor (scaled by hint dependence h).
HINT_ACCURACY_BOOST: float = 0.40

# HINT: Durable mastery contribution (comparatively small; makes over-hinting detectable).
HINT_DURABLE_EFFECT: float = 0.10

# PRACTICE: Peak multiplier when current mastery is at moderate "desirable difficulty" (~0.5).
PRACTICE_PEAK_EFFECT: float = 0.70
PRACTICE_BASE_EFFECT: float = 0.05

# REVIEW_PREREQUISITE: Base mastery gain on the prerequisite concept before transfer.
REVIEW_PREREQ_BASE_GAIN: float = 0.45

# Forgetting scale factor per interaction step:
# Converts theta.beta in [0.01, 0.15] to per-step unpracticed decay in [0.001, 0.015],
# aligning with observable learner model delta (0.01) and ensuring multi-concept retention.
FORGETTING_SCALE: float = 0.08

# Steepness multiplier for the logistic difficulty-mastery response curve.
SIGMOID_STEEPNESS: float = 4.0


@dataclass
class SimulationStepResult:
    """Outcome of a single simulated tutoring interaction step."""
    concept_id: str
    action: PedagogicalAction
    correct: bool
    p_correct: float
    previous_true_mastery: float
    new_true_mastery: float
    prerequisite_id: str | None = None
    prerequisite_gain: float = 0.0


def compute_action_effect(
    action: PedagogicalAction,
    current_mastery: float,
    theta: LearnerParameters,
    action_effect_multiplier: float = 1.0,
) -> float:
    """Compute smooth pedagogical action multiplier A_effect(action, M, theta).

    Follows Roadmap Sec. 11 specifications:
    - EXPLAIN: Constant conceptual-clarity contribution.
    - EXAMPLE: Scaled by (1 - alpha) so slower learners benefit relatively more.
    - HINT: Small durable gain (immediate boost happens in P(correct)).
    - PRACTICE: Peaks at moderate mastery (M ~ 0.5) via smooth parabolic curve.
    - REVIEW_PREREQUISITE: Base effect applied to prerequisite before transfer.
    """
    m = max(0.0, min(1.0, current_mastery))

    if action == PedagogicalAction.EXPLAIN:
        base = EXPLAIN_BASE_EFFECT

    elif action == PedagogicalAction.EXAMPLE:
        # Slower learners (lower alpha in [0.05, 0.35]) get a higher multiplier
        base = EXAMPLE_BASE_EFFECT * (1.0 - theta.alpha)

    elif action == PedagogicalAction.HINT:
        # Contributes little to durable mastery
        base = HINT_DURABLE_EFFECT

    elif action == PedagogicalAction.PRACTICE:
        # Desirable difficulty: peaks at M = 0.5 where 4 * M * (1 - M) = 1.0
        parabola = 4.0 * m * (1.0 - m)
        base = PRACTICE_BASE_EFFECT + (PRACTICE_PEAK_EFFECT * parabola)

    elif action == PedagogicalAction.REVIEW_PREREQUISITE:
        base = REVIEW_PREREQ_BASE_GAIN

    else:
        raise ValueError(f"Unknown action: {action}")

    return base * action_effect_multiplier


def compute_correctness_probability(
    current_mastery: float,
    difficulty: float | int,
    action: PedagogicalAction,
    theta: LearnerParameters,
    apply_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute P(correct) smoothly from latent mastery, question difficulty, and theta.

    - Question difficulty is normalized to [0.1, 0.9].
    - Response probability follows a logistic sigmoid over (mastery - difficulty),
      modulated by difficulty sensitivity gamma.
    - HINT action boosts immediate correctness scaled by hint-dependence h.
    - Response noise sigma introduces stochastic perturbation when apply_noise=True.
    """
    m = max(0.0, min(1.0, current_mastery))

    # Normalize integer difficulty (1-5) or continuous difficulty into [0.1, 0.9]
    if isinstance(difficulty, int) or difficulty > 1.0:
        d_norm = 0.10 + 0.16 * max(1, min(5, float(difficulty)))
    else:
        d_norm = max(0.10, min(0.90, float(difficulty)))

    # Gap between current mastery and question difficulty
    gap = m - d_norm

    # Base probability via logistic sigmoid modulated by gamma
    z = SIGMOID_STEEPNESS * theta.gamma * gap
    # Safe sigmoid computation
    if z >= 30.0:
        p_base = 0.999
    elif z <= -30.0:
        p_base = 0.001
    else:
        p_base = 1.0 / (1.0 + math.exp(-z))

    # Action-specific adjustments
    if action == PedagogicalAction.HINT:
        # HINT strongly boosts immediate correctness scaled by h
        p_action = p_base + (theta.h * HINT_ACCURACY_BOOST * (1.0 - p_base))
    elif action == PedagogicalAction.EXPLAIN:
        # Explanation provides an immediate conceptual clarification boost
        p_action = p_base + (0.35 * (1.0 - p_base))
    elif action == PedagogicalAction.EXAMPLE:
        # Example provides scaffolding boost, higher for lower alpha
        p_action = p_base + (0.30 * (1.0 - theta.alpha) * (1.0 - p_base))
    else:
        p_action = p_base

    # Bound probability before noise
    p_clamped = max(0.01, min(0.99, p_action))

    if apply_noise and rng is not None:
        # Add stochastic noise perturbation Uniform(-sigma, sigma)
        noise = float(rng.uniform(-theta.sigma, theta.sigma))
        p_clamped = max(0.01, min(0.99, p_clamped + noise))

    return p_clamped


def step_simulation(
    learner: SimulatedLearner,
    action: PedagogicalAction,
    concept_id: str,
    question_difficulty: float | int = 2,
    prerequisite_id: str | None = None,
    steps_unpracticed: int | None = None,
    force_correct: bool | None = None,
    action_effect_multiplier: float = 1.0,
) -> SimulationStepResult:
    """Execute a single simulated pedagogical interaction step on a simulated learner.

    Updates internal true continuous mastery M(c, t) according to Roadmap Sec. 11.
    Decays unpracticed concepts by beta * FORGETTING_SCALE.
    Propagates transfer ability tau if reviewing a prerequisite.
    """
    theta = learner.theta
    m_prev = learner.get_true_mastery(concept_id)

    # Determine steps unpracticed for concept_id
    if steps_unpracticed is None:
        last_step = learner.get_last_practiced_step(concept_id)
        steps_unpracticed = max(0, learner.step_count - last_step)

    # 1. Compute response probability
    p_correct = compute_correctness_probability(
        current_mastery=m_prev,
        difficulty=question_difficulty,
        action=action,
        theta=theta,
        apply_noise=True,
        rng=learner.rng,
    )

    # 2. Sample response correctness
    if force_correct is not None:
        correct = bool(force_correct)
    else:
        correct = bool(learner.rng.random() < p_correct)

    prereq_gain = 0.0

    # 3. Handle action dynamics & mastery update
    if action == PedagogicalAction.REVIEW_PREREQUISITE and prerequisite_id is not None:
        # Reviewing prerequisite concept p
        m_prereq_prev = learner.get_true_mastery(prerequisite_id)
        a_eff = compute_action_effect(action, m_prereq_prev, theta, action_effect_multiplier=action_effect_multiplier)

        # Prerequisite gain
        if correct:
            prereq_gain = theta.alpha * a_eff
        else:
            prereq_gain = theta.alpha * a_eff * 0.30  # Partial exposure gain

        new_m_prereq = max(0.0, min(1.0, m_prereq_prev + prereq_gain))
        learner.set_true_mastery(prerequisite_id, new_m_prereq)

        # Transfer a fraction tau of the prerequisite gain to dependent concept c
        transfer_gain = theta.tau * prereq_gain
        new_m_target = max(0.0, min(1.0, m_prev + transfer_gain))
        learner.set_true_mastery(concept_id, new_m_target)

    else:
        # Standard intervention on concept_id
        a_eff = compute_action_effect(action, m_prev, theta, action_effect_multiplier=action_effect_multiplier)

        if correct:
            learning_gain = theta.alpha * a_eff
        else:
            # Slower gain if incorrect (student attempted but missed)
            learning_gain = theta.alpha * a_eff * 0.35

        new_m_target = max(0.0, min(1.0, m_prev + learning_gain))
        learner.set_true_mastery(concept_id, new_m_target)

    # 4. Apply 1-step forgetting decay to all other unpracticed concepts
    per_step_decay = theta.beta * FORGETTING_SCALE
    for other_c in learner.get_all_true_masteries():
        if other_c != concept_id and (action != PedagogicalAction.REVIEW_PREREQUISITE or other_c != prerequisite_id):
            curr = learner.get_true_mastery(other_c)
            learner.set_true_mastery(other_c, max(0.0, curr - per_step_decay))

    learner.record_practice(concept_id)

    return SimulationStepResult(
        concept_id=concept_id,
        action=action,
        correct=correct,
        p_correct=p_correct,
        previous_true_mastery=m_prev,
        new_true_mastery=learner.get_true_mastery(concept_id),
        prerequisite_id=prerequisite_id,
        prerequisite_gain=prereq_gain,
    )


# Alias for backward and benchmark compatibility
simulate_tutoring_step = step_simulation

