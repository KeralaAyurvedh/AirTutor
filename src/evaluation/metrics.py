"""Evaluation Metrics Suite for AirTutor (Roadmap Sec. 19).

Computes core research metrics across simulated tutoring episodes:
1. Learning gain: Mean mastery increase over an episode across curriculum concepts.
2. Final mastery: Mean final mastery achieved at the end of the episode.
3. Steps to mastery: Total interaction steps required to reach mastery threshold.
4. Intervention efficiency: Learning gain per interaction step.
5. Hint-usage rate: Fraction of interventions that were HINT.
6. Action-distribution entropy: Shannon entropy of the chosen action distribution
   (evaluating policy diversity vs. degenerate collapse).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.environment.actions import ACTIONS, PedagogicalAction


@dataclass
class EpisodeMetrics:
    """Metrics evaluated for a single simulated tutoring episode."""
    learning_gain: float
    final_mastery: float
    steps_to_mastery: int
    intervention_efficiency: float
    hint_usage_rate: float
    action_entropy: float
    action_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_gain": self.learning_gain,
            "final_mastery": self.final_mastery,
            "steps_to_mastery": float(self.steps_to_mastery),
            "intervention_efficiency": self.intervention_efficiency,
            "hint_usage_rate": self.hint_usage_rate,
            "action_entropy": self.action_entropy,
            "action_counts": dict(self.action_counts),
        }


def compute_learning_gain(
    initial_mastery: Mapping[str, float],
    final_mastery: Mapping[str, float],
) -> float:
    """Compute mean mastery increase across all concepts over an episode."""
    common_concepts = set(initial_mastery.keys()).intersection(final_mastery.keys())
    if not common_concepts:
        return 0.0
    gains = [final_mastery[c] - initial_mastery[c] for c in common_concepts]
    return float(np.mean(gains))


def compute_final_mastery(final_mastery: Mapping[str, float]) -> float:
    """Compute mean final mastery across all concepts."""
    if not final_mastery:
        return 0.0
    return float(np.mean(list(final_mastery.values())))


def compute_steps_to_mastery(
    mastery_trajectory: Sequence[Mapping[str, float]],
    threshold: float = 0.75,
    max_steps: int = 30,
) -> int:
    """Compute steps taken until mean concept mastery reaches or exceeds threshold.

    Returns max_steps if threshold is never reached.
    """
    for step_idx, masteries in enumerate(mastery_trajectory, start=1):
        if masteries and np.mean(list(masteries.values())) >= threshold:
            return step_idx
    return max_steps


def compute_intervention_efficiency(learning_gain: float, total_steps: int) -> float:
    """Compute learning gain per interaction step."""
    if total_steps <= 0:
        return 0.0
    return float(learning_gain / total_steps)


def compute_hint_usage_rate(actions: Sequence[PedagogicalAction | str]) -> float:
    """Compute fraction of actions that were HINT."""
    if not actions:
        return 0.0
    hint_count = sum(
        1 for a in actions if (a == PedagogicalAction.HINT or a == "HINT")
    )
    return float(hint_count / len(actions))


def compute_action_entropy(actions: Sequence[PedagogicalAction | str]) -> float:
    """Compute Shannon entropy H(A) of the action distribution in bits.

    H(A) = - sum_{a} p(a) * log2(p(a)).
    Maximum entropy for 5 actions is log2(5) ~ 2.322 bits.
    """
    if not actions:
        return 0.0

    counts: dict[str, int] = {}
    for a in actions:
        key = a.value if isinstance(a, PedagogicalAction) else str(a)
        counts[key] = counts.get(key, 0) + 1

    total = len(actions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return float(entropy)


def evaluate_episode(
    initial_mastery: Mapping[str, float],
    final_mastery: Mapping[str, float],
    actions: Sequence[PedagogicalAction | str],
    mastery_history: Sequence[Mapping[str, float]] | None = None,
    mastery_threshold: float = 0.75,
    max_steps: int = 30,
) -> EpisodeMetrics:
    """Compute all 6 roadmap metrics for an episode."""
    gain = compute_learning_gain(initial_mastery, final_mastery)
    final_m = compute_final_mastery(final_mastery)
    total_steps = len(actions)

    traj = mastery_history if mastery_history is not None else [final_mastery]
    steps_to_m = compute_steps_to_mastery(traj, threshold=mastery_threshold, max_steps=max_steps)

    efficiency = compute_intervention_efficiency(gain, total_steps)
    hint_rate = compute_hint_usage_rate(actions)
    entropy = compute_action_entropy(actions)

    act_counts: dict[str, int] = {a.name: 0 for a in ACTIONS}
    for a in actions:
        key = a.name if isinstance(a, PedagogicalAction) else str(a)
        act_counts[key] = act_counts.get(key, 0) + 1

    return EpisodeMetrics(
        learning_gain=gain,
        final_mastery=final_m,
        steps_to_mastery=steps_to_m,
        intervention_efficiency=efficiency,
        hint_usage_rate=hint_rate,
        action_entropy=entropy,
        action_counts=act_counts,
    )


def aggregate_metrics(metrics_list: Sequence[EpisodeMetrics]) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) for each metric across a collection of episodes."""
    if not metrics_list:
        return {}

    keys = [
        "learning_gain",
        "final_mastery",
        "steps_to_mastery",
        "intervention_efficiency",
        "hint_usage_rate",
        "action_entropy",
    ]

    result: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics_list]
        result[k] = (float(np.mean(vals)), float(np.std(vals)))

    return result
