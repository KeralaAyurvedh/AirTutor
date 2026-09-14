"""Unit tests for Reward Functions v1, v2, v3 (Roadmap Sec. 15)."""

import pytest

from src.environment.rewards import (
    compute_reward,
    compute_reward_v1,
    compute_reward_v2,
    compute_reward_v3,
)


def test_compute_reward_v1():
    """Verify Reward v1 computes delta M_hat accurately."""
    r_pos = compute_reward_v1(prev_concept_mastery=0.20, new_concept_mastery=0.45)
    assert r_pos == pytest.approx(0.25)

    r_zero = compute_reward_v1(prev_concept_mastery=0.50, new_concept_mastery=0.50)
    assert r_zero == pytest.approx(0.0)


def test_compute_reward_v2():
    """Verify Reward v2 includes retention penalty on unpracticed concepts and step cost."""
    prev_all = {"c1": 0.20, "c2": 0.50}
    new_all = {"c1": 0.35, "c2": 0.48}  # c1 gained +0.15, c2 decayed -0.02

    # gain = 0.15, retention = -0.02, cost = 1.0 * 0.015 = 0.015
    # r = 0.15 + 0.50 * (-0.02) - 0.015 = 0.15 - 0.01 - 0.015 = 0.125
    r = compute_reward_v2(
        prev_concept_mastery=0.20,
        new_concept_mastery=0.35,
        prev_all_masteries=prev_all,
        new_all_masteries=new_all,
        concept_id="c1",
        lambda_ret=0.50,
        lambda_cost=0.015,
    )
    assert r == pytest.approx(0.125)


def test_compute_reward_v3():
    """Verify Reward v3 penalizes consecutive hints and wasted steps on mastered concepts."""
    # Case 1: Normal step with prerequisite transfer credit
    r_normal = compute_reward_v3(
        prev_concept_mastery=0.20,
        new_concept_mastery=0.30,
        action_name="PRACTICE",
        hint_used_last_step=False,
        prerequisite_gain=0.10,  # transfer credit = 0.10 * 0.50 = +0.05
    )
    # r_v2 approx = 0.10 - 0.015 = 0.085; + transfer (0.05) = 0.135
    assert r_normal == pytest.approx(0.135)

    # Case 2: Excessive hint penalty
    r_excessive_hint = compute_reward_v3(
        prev_concept_mastery=0.20,
        new_concept_mastery=0.25,
        action_name="HINT",
        hint_used_last_step=True,  # triggers 0.05 penalty
    )
    r_single_hint = compute_reward_v3(
        prev_concept_mastery=0.20,
        new_concept_mastery=0.25,
        action_name="HINT",
        hint_used_last_step=False,
    )
    assert r_single_hint - r_excessive_hint == pytest.approx(0.05)

    # Case 3: Wasted step penalty on already-mastered concept (>= 0.85)
    r_mastered = compute_reward_v3(
        prev_concept_mastery=0.90,
        new_concept_mastery=0.92,
        action_name="PRACTICE",
    )
    r_unmastered = compute_reward_v3(
        prev_concept_mastery=0.70,
        new_concept_mastery=0.72,
        action_name="PRACTICE",
    )
    assert r_unmastered - r_mastered == pytest.approx(0.05)


def test_compute_reward_dispatcher():
    """Verify unified dispatcher routes to appropriate reward function."""
    assert compute_reward("v1", 0.1, 0.3) == pytest.approx(0.2)
    assert compute_reward("v2", 0.1, 0.3) < 0.2  # penalized by step cost
