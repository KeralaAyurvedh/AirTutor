"""Robustness and Sensitivity Analysis for AirTutor (Roadmap Sec. 21).

Tests whether the RL pedagogical advantage over the competitive Rule baseline
is stable, or whether it shrinks/vanishes under parameter perturbations:
1. Learning rate bounds alpha (+/- 20%)
2. Forgetting rate bounds beta (+/- 20%)
3. Response noise sigma (+/- 50%)
4. Action dynamics multiplier A_effect (+/- 20%)
5. Population composition skewed toward high-h (hint-dependent) learners
6. Curriculum difficulty shifted (+/- 1.0)
"""

import copy
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from src.environment.curriculum import CurriculumRepository, load_curriculum
from src.evaluation.baseline_comparison import run_tutoring_episode
from src.evaluation.compare_teachers import evaluate_agent_on_held_out
from src.evaluation.metrics import EpisodeMetrics
from src.rl.q_learning import TabularQAgent
from src.simulator.population import DEFAULT_PARAM_RANGES, sample_learner
from src.teachers.rule_teacher import RuleTeacher


def get_perturbed_param_ranges(
    base_ranges: dict[str, tuple[float, float]],
    param_name: str,
    factor: float,
) -> dict[str, tuple[float, float]]:
    """Return a copy of parameter ranges with one parameter bounds scaled by factor."""
    ranges = copy.deepcopy(base_ranges)
    low, high = ranges[param_name]
    ranges[param_name] = (max(0.001, low * factor), high * factor)
    return ranges


def run_sensitivity_sweep(
    agents: Sequence[TabularQAgent],
    n_eval: int = 100,
    eval_seed_start: int = 10000,
    curriculum: CurriculumRepository | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Execute robustness sweep across all 10+ perturbation conditions."""
    curr = curriculum or load_curriculum("data/curriculum")
    concept_ids = curr.get_all_concept_ids()
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))

    base_ranges = copy.deepcopy(DEFAULT_PARAM_RANGES)

    # Define all perturbation specifications
    # (condition_label, param_ranges, action_effect_multiplier, difficulty_offset)
    conditions: list[tuple[str, dict[str, tuple[float, float]], float, float]] = [
        ("Baseline (Default)", base_ranges, 1.0, 0.0),
        ("Alpha -20%", get_perturbed_param_ranges(base_ranges, "alpha", 0.8), 1.0, 0.0),
        ("Alpha +20%", get_perturbed_param_ranges(base_ranges, "alpha", 1.2), 1.0, 0.0),
        ("Beta -20%", get_perturbed_param_ranges(base_ranges, "beta", 0.8), 1.0, 0.0),
        ("Beta +20%", get_perturbed_param_ranges(base_ranges, "beta", 1.2), 1.0, 0.0),
        ("Noise Sigma -50%", get_perturbed_param_ranges(base_ranges, "sigma", 0.5), 1.0, 0.0),
        ("Noise Sigma +50%", get_perturbed_param_ranges(base_ranges, "sigma", 1.5), 1.0, 0.0),
        ("A_effect -20%", base_ranges, 0.80, 0.0),
        ("A_effect +20%", base_ranges, 1.20, 0.0),
        ("High-h Skew [0.5, 1.0]", {**base_ranges, "h": (0.50, 1.00)}, 1.0, 0.0),
        ("Curriculum Harder (+1)", base_ranges, 1.0, +1.0),
        ("Curriculum Easier (-1)", base_ranges, 1.0, -1.0),
    ]

    results: dict[str, dict[str, Any]] = {}

    if verbose:
        print("\n" + "=" * 90)
        print(f" AirTutor Robustness & Sensitivity Sweep across {len(conditions)} Conditions")
        print(f" Evaluating {len(agents)} Trained RL Seeds vs. Rule Baseline on {n_eval} Held-Out Learners")
        print("=" * 90)

    for label, p_ranges, a_mult, diff_off in conditions:
        # 1. Evaluate Rule baseline under this condition
        rule_teacher = RuleTeacher(mastery_threshold=0.75, review_interval=5, max_steps_per_concept=6)
        rule_gains: list[float] = []

        for seed in eval_seeds:
            learner = sample_learner(
                seed=seed,
                concepts=concept_ids,
                initial_mastery=0.10,
                param_ranges=p_ranges,
            )
            ep_m, _ = run_tutoring_episode(
                teacher=rule_teacher,
                learner=learner,
                curriculum=curr,
                max_steps=30,
                initial_mastery=0.10,
                action_effect_multiplier=a_mult,
                difficulty_offset=diff_off,
            )
            rule_gains.append(ep_m.learning_gain)

        rule_mean = float(np.mean(rule_gains))
        rule_std = float(np.std(rule_gains))

        # 2. Evaluate RL agents under this condition
        rl_gains: list[float] = []
        for agent in agents:
            ag_m, _ = evaluate_agent_on_held_out(
                agent=agent,
                eval_seeds=eval_seeds,
                curriculum=curr,
                max_steps=30,
                param_ranges=p_ranges,
                action_effect_multiplier=a_mult,
                difficulty_offset=diff_off,
            )
            rl_gains.extend([m.learning_gain for m in ag_m])

        rl_mean = float(np.mean(rl_gains))
        rl_std = float(np.std(rl_gains))

        # 3. Compute RL Advantage Delta = RL - Rule
        adv_delta = rl_mean - rule_mean
        adv_std = float(np.sqrt((rl_std ** 2) / len(rl_gains) + (rule_std ** 2) / len(rule_gains)))

        # Categorize status
        if adv_delta >= 0.020:
            status = "Holds"
        elif adv_delta >= 0.005:
            status = "Shrinks"
        else:
            status = "Vanishes"

        results[label] = {
            "rl_gain": (rl_mean, rl_std),
            "rule_gain": (rule_mean, rule_std),
            "advantage": (adv_delta, adv_std),
            "status": status,
            "action_effect_multiplier": a_mult,
            "difficulty_offset": diff_off,
        }

        if verbose:
            print(
                f" {label:24} | RL: {rl_mean:+.3f} | Rule: {rule_mean:+.3f} | "
                f"Delta: {adv_delta:+.3f} | Status: [{status:^8}]"
            )

    return results
