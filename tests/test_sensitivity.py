"""Unit tests for Stage 4 Sensitivity Analysis (Roadmap Sec. 21)."""

import pytest

from src.environment.curriculum import load_curriculum
from src.evaluation.sensitivity import get_perturbed_param_ranges, run_sensitivity_sweep
from src.rl.q_learning import TabularQAgent
from src.simulator.population import DEFAULT_PARAM_RANGES


def test_perturbed_param_ranges():
    """Verify bounds are scaled correctly by factor."""
    base = dict(DEFAULT_PARAM_RANGES)
    orig_low, orig_high = base["alpha"]
    perturbed = get_perturbed_param_ranges(base, "alpha", 0.8)

    assert perturbed["alpha"][0] == pytest.approx(orig_low * 0.8)
    assert perturbed["alpha"][1] == pytest.approx(orig_high * 0.8)
    # Other keys remain unchanged
    assert perturbed["beta"] == base["beta"]


def test_sensitivity_sweep_fast():
    """Verify sensitivity sweep runs across conditions and categorizes advantage."""
    curr = load_curriculum("data/curriculum")
    agent = TabularQAgent(seed=42)

    results = run_sensitivity_sweep(
        agents=[agent],
        n_eval=4,
        eval_seed_start=10000,
        curriculum=curr,
        verbose=False,
    )

    assert len(results) >= 10
    assert "Baseline (Default)" in results
    assert "Alpha -20%" in results
    assert "High-h Skew [0.5, 1.0]" in results

    for cond, data in results.items():
        assert "rl_gain" in data
        assert "rule_gain" in data
        assert "advantage" in data
        assert data["status"] in {"Holds", "Shrinks", "Vanishes"}
