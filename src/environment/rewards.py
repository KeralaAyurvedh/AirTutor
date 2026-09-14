"""Reward Function Architecture for AirTutor (Roadmap Sec. 15).

Reward is treated as a research variable.
- Reward v1 (Stage 3 default):
    r_t = M_hat(c, t+1) - M_hat(c, t)
  Mastery gain on the concept just taught, computed strictly from observable
  estimated mastery (LearnerModel), not simulator-internal ground truth.
"""

from typing import Callable, Mapping


def compute_reward_v1(
    prev_estimated_mastery: float,
    new_estimated_mastery: float,
) -> float:
    """Reward v1: Change in estimated mastery on the concept taught at step t.

    Formula:
        r_t = M_hat(c, t+1) - M_hat(c, t)
    """
    return float(new_estimated_mastery - prev_estimated_mastery)


# Reward dispatch table for versioned reward experimentation
REWARD_FUNCTIONS: dict[str, Callable[..., float]] = {
    "v1": compute_reward_v1,
}
