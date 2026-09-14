"""Unit tests for Reward Functions (Roadmap Sec. 15)."""

import pytest

from src.environment.rewards import compute_reward_v1


def test_compute_reward_v1():
    """Verify Reward v1 computes delta M_hat accurately."""
    # Positive mastery gain
    r_pos = compute_reward_v1(prev_estimated_mastery=0.20, new_estimated_mastery=0.45)
    assert r_pos == pytest.approx(0.25)

    # Zero gain
    r_zero = compute_reward_v1(prev_estimated_mastery=0.50, new_estimated_mastery=0.50)
    assert r_zero == pytest.approx(0.0)

    # Negative gain (e.g. from decay)
    r_neg = compute_reward_v1(prev_estimated_mastery=0.60, new_estimated_mastery=0.55)
    assert r_neg == pytest.approx(-0.05)
