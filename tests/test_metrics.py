"""Unit tests for Evaluation Metrics (Roadmap Sec. 19)."""

import math
import pytest

from src.environment.actions import ACTIONS, PedagogicalAction
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_action_entropy,
    compute_final_mastery,
    compute_hint_usage_rate,
    compute_intervention_efficiency,
    compute_learning_gain,
    compute_steps_to_mastery,
    evaluate_episode,
)


def test_compute_learning_gain():
    """Verify mean learning gain across concepts."""
    init_m = {"c1": 0.10, "c2": 0.20}
    final_m = {"c1": 0.60, "c2": 0.80}
    # Gains: (0.50 + 0.60) / 2 = 0.55
    gain = compute_learning_gain(init_m, final_m)
    assert gain == pytest.approx(0.55)


def test_compute_final_mastery():
    """Verify mean final mastery."""
    final_m = {"c1": 0.70, "c2": 0.90}
    assert compute_final_mastery(final_m) == pytest.approx(0.80)


def test_compute_steps_to_mastery():
    """Verify steps to mastery calculation with threshold."""
    history = [
        {"c1": 0.30, "c2": 0.30},  # mean 0.30
        {"c1": 0.50, "c2": 0.60},  # mean 0.55
        {"c1": 0.80, "c2": 0.80},  # mean 0.80 -> reaches 0.75 at step 3
        {"c1": 0.90, "c2": 0.90},
    ]
    assert compute_steps_to_mastery(history, threshold=0.75, max_steps=10) == 3

    # If threshold never reached, returns max_steps
    assert compute_steps_to_mastery(history, threshold=0.95, max_steps=10) == 10


def test_compute_intervention_efficiency():
    """Verify efficiency = gain / steps."""
    assert compute_intervention_efficiency(0.50, 10) == pytest.approx(0.05)
    assert compute_intervention_efficiency(0.50, 0) == 0.0


def test_compute_hint_usage_rate():
    """Verify hint rate calculation."""
    actions = [
        PedagogicalAction.EXPLAIN,
        PedagogicalAction.HINT,
        PedagogicalAction.PRACTICE,
        PedagogicalAction.HINT,
    ]
    assert compute_hint_usage_rate(actions) == pytest.approx(0.50)


def test_compute_action_entropy():
    """Verify Shannon entropy computation in bits."""
    # 1. Single action -> entropy = 0.0
    single_action = [PedagogicalAction.PRACTICE] * 20
    assert compute_action_entropy(single_action) == pytest.approx(0.0)

    # 2. Perfectly uniform distribution across 5 actions -> log2(5) ~ 2.3219 bits
    uniform_actions = list(ACTIONS) * 10
    expected_entropy = math.log2(5)
    assert compute_action_entropy(uniform_actions) == pytest.approx(expected_entropy, abs=1e-4)


def test_evaluate_and_aggregate_metrics():
    """Verify end-to-end episode metric evaluation and batch aggregation."""
    init_m = {"c1": 0.10}
    final_m1 = {"c1": 0.60}
    final_m2 = {"c1": 0.80}

    actions = [PedagogicalAction.EXPLAIN, PedagogicalAction.PRACTICE]

    m1 = evaluate_episode(init_m, final_m1, actions)
    m2 = evaluate_episode(init_m, final_m2, actions)

    agg = aggregate_metrics([m1, m2])

    assert agg["learning_gain"][0] == pytest.approx((0.50 + 0.70) / 2)
    assert agg["final_mastery"][0] == pytest.approx((0.60 + 0.80) / 2)
    assert "action_entropy" in agg
    assert "hint_usage_rate" in agg
    assert "intervention_efficiency" in agg
