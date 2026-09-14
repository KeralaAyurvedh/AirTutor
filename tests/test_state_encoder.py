"""Unit tests for Observable State Encoder (Roadmap Sec. 13)."""

import pytest

from src.environment.state_encoder import (
    ObservableHistory,
    StateEncoderConfig,
    discretize_hint_used,
    discretize_last_correct,
    discretize_mastery,
    discretize_recent_accuracy,
    discretize_wrong_streak,
    encode_state,
    get_total_state_count,
    index_to_state_tuple,
    state_tuple_to_index,
)


def test_default_state_count():
    """Verify default state count is exactly 5 x 2 x 4 x 4 x 2 = 320."""
    config = StateEncoderConfig()
    total = get_total_state_count(config)
    assert total == 320


def test_discretization_functions():
    """Verify each individual discretization function maps into expected bucket ranges."""
    # Mastery: 5 buckets
    assert discretize_mastery(0.0) == 0
    assert discretize_mastery(0.19) == 0
    assert discretize_mastery(0.20) == 1
    assert discretize_mastery(0.39) == 1
    assert discretize_mastery(0.40) == 2
    assert discretize_mastery(0.59) == 2
    assert discretize_mastery(0.60) == 3
    assert discretize_mastery(0.79) == 3
    assert discretize_mastery(0.80) == 4
    assert discretize_mastery(1.00) == 4

    # Last correct: 2 buckets
    assert discretize_last_correct(None) == 0
    assert discretize_last_correct(False) == 0
    assert discretize_last_correct(True) == 1

    # Recent accuracy (last 3): 4 buckets
    assert discretize_recent_accuracy([]) == 0
    assert discretize_recent_accuracy([False]) == 0
    assert discretize_recent_accuracy([True]) == 1
    assert discretize_recent_accuracy([True, False, True]) == 2
    assert discretize_recent_accuracy([True, True, True]) == 3
    # Handles longer history by taking last 3
    assert discretize_recent_accuracy([False, False, True, True, True]) == 3

    # Wrong streak: 4 buckets (0, 1, 2, 3+)
    assert discretize_wrong_streak(0) == 0
    assert discretize_wrong_streak(1) == 1
    assert discretize_wrong_streak(2) == 2
    assert discretize_wrong_streak(3) == 3
    assert discretize_wrong_streak(10) == 3

    # Hint used: 2 buckets
    assert discretize_hint_used(False) == 0
    assert discretize_hint_used(True) == 1


def test_state_encoder_round_trip():
    """Verify bidirectional mapping between feature tuple and scalar state index."""
    config = StateEncoderConfig()
    total_states = get_total_state_count(config)

    for idx in [0, 1, 50, 100, 200, 319]:
        state_tuple = index_to_state_tuple(idx, config)
        recovered_idx = state_tuple_to_index(state_tuple, config)
        assert recovered_idx == idx, f"Mismatch at index {idx}: tuple={state_tuple}"


def test_encode_state_from_history():
    """Verify encode_state creates correct tuple from ObservableHistory."""
    history = ObservableHistory(
        concept_id="variables",
        estimated_mastery=0.45,  # bucket 2
        last_response_correct=True,  # bucket 1
        recent_responses=[True, False, True],  # bucket 2
        wrong_answer_streak=0,  # bucket 0
        hint_used_last_step=False,  # bucket 0
    )
    state = encode_state(history)
    assert state == (2, 1, 2, 0, 0)
    idx = state_tuple_to_index(state)
    assert 0 <= idx < 320


def test_modular_feature_ablation():
    """Verify that toggling features reduces the state dimension appropriately."""
    # Ablate hint_used: 5 x 2 x 4 x 4 = 160 states
    ablated_config = StateEncoderConfig(use_hint_used=False)
    assert get_total_state_count(ablated_config) == 160

    history = ObservableHistory(
        concept_id="loops",
        estimated_mastery=0.75,
        last_response_correct=False,
        recent_responses=[False, False],
        wrong_answer_streak=2,
        hint_used_last_step=True,
    )
    ablated_tuple = encode_state(history, config=ablated_config)
    assert len(ablated_tuple) == 4
    assert state_tuple_to_index(ablated_tuple, config=ablated_config) < 160
