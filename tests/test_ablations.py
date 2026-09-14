"""Unit tests for Stage 4 Ablation studies (Roadmap Sec. 15 & 22)."""

import pytest

from src.environment.curriculum import load_curriculum
from src.evaluation.ablations import (
    run_population_diversity_ablation,
    run_reward_ablation,
    run_state_ablation,
)


@pytest.fixture
def curriculum():
    return load_curriculum("data/curriculum")


def test_reward_ablation_fast(curriculum):
    """Verify reward ablation runs and generates valid metrics and action distributions for v1, v2, v3."""
    results = run_reward_ablation(
        rl_seeds=[42],
        n_train=10,
        n_eval=5,
        train_seed_start=0,
        eval_seed_start=10000,
        curriculum=curriculum,
        verbose=False,
    )

    assert set(results.keys()) == {"v1", "v2", "v3"}
    for v in ["v1", "v2", "v3"]:
        assert "learning_gain" in results[v]
        assert "true_learning_gain" in results[v]
        assert "action_distribution" in results[v]
        dist = results[v]["action_distribution"]
        total_prop = sum(dist.values())
        assert total_prop == pytest.approx(1.0, abs=1e-3)


def test_state_ablation_fast(curriculum):
    """Verify state representation ablation runs across the 3 configurations."""
    results = run_state_ablation(
        rl_seeds=[42],
        n_train=10,
        n_eval=5,
        train_seed_start=0,
        eval_seed_start=10000,
        reward_version="v1",
        curriculum=curriculum,
        verbose=False,
    )

    assert set(results.keys()) == {"Full State (320)", "No-Hint (160)", "No-Accuracy (80)"}
    for k in results:
        assert "learning_gain" in results[k]
        assert "true_learning_gain" in results[k]


def test_population_diversity_ablation_fast(curriculum):
    """Verify population diversity ablation compares wide vs narrow training."""
    results = run_population_diversity_ablation(
        rl_seeds=[42],
        n_train=10,
        n_eval=5,
        train_seed_start=0,
        eval_seed_start=10000,
        curriculum=curriculum,
        verbose=False,
    )

    assert "Wide Population (Default)" in results
    assert "Narrow Population (+/-10%)" in results
