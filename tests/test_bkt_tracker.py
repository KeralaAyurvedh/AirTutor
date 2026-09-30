"""Unit tests for Bayesian Knowledge Tracing (BKT) tracker."""

import pytest
from src.learner_model.bkt_tracker import BKTTracker, BKTConceptParams, DEFAULT_P_L0, DEFAULT_P_T


def test_bkt_prior_initialization():
    tracker = BKTTracker(concepts=["c1", "c2"])
    assert tracker.get_belief("c1") == pytest.approx(DEFAULT_P_L0)
    assert tracker.get_belief("c2") == pytest.approx(DEFAULT_P_L0)
    # Untracked concept initializes on demand to P_L0
    assert tracker.get_belief("c3") == pytest.approx(DEFAULT_P_L0)


def test_bkt_update_correct():
    tracker = BKTTracker(concepts=["c1"])
    initial_b = tracker.get_belief("c1")
    updated_b = tracker.update("c1", is_correct=True)
    # On correct answer, belief should strictly increase
    assert updated_b > initial_b
    assert 0.0 <= updated_b <= 1.0


def test_bkt_update_incorrect():
    tracker = BKTTracker(concepts=["c1"])
    # After a correct answer, an incorrect answer should depress belief relative to consecutive correct
    tracker.update("c1", is_correct=True)
    b_after_correct = tracker.get_belief("c1")
    b_after_incorrect = tracker.update("c1", is_correct=False)
    assert b_after_incorrect < b_after_correct
    assert 0.0 <= b_after_incorrect <= 1.0


def test_bkt_bounds_under_repeated_evidence():
    tracker = BKTTracker(concepts=["c1", "c2"])
    # 20 consecutive correct answers
    for _ in range(20):
        b = tracker.update("c1", is_correct=True)
        assert 0.0 <= b <= 1.0
    assert tracker.get_belief("c1") > 0.95

    # 20 consecutive incorrect answers
    for _ in range(20):
        b = tracker.update("c2", is_correct=False)
        assert 0.0 <= b <= 1.0
    # Note: because P(T)=0.15, floor is >= P(T)
    assert tracker.get_belief("c2") < 0.20


def test_bkt_reset():
    tracker = BKTTracker(concepts=["c1", "c2"])
    tracker.update("c1", is_correct=True)
    tracker.update("c2", is_correct=False)
    assert tracker.get_belief("c1") != DEFAULT_P_L0
    tracker.reset()
    assert tracker.get_belief("c1") == pytest.approx(DEFAULT_P_L0)
    assert tracker.get_belief("c2") == pytest.approx(DEFAULT_P_L0)


def test_bkt_independent_concepts():
    tracker = BKTTracker(concepts=["c1", "c2"])
    tracker.update("c1", is_correct=True)
    assert tracker.get_belief("c1") > tracker.get_belief("c2")
    assert tracker.get_belief("c2") == pytest.approx(DEFAULT_P_L0)
