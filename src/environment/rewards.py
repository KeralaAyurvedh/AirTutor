"""Reward Function Architecture for AirTutor (Roadmap Sec. 15).

Reward is treated as a research variable across three versions:
- Reward v1 (Stage 3):
    r_t = M_hat(c, t+1) - M_hat(c, t)
  Mastery gain on current concept only. Known risk: short-term optimization.

- Reward v2 (Stage 4):
    r_t = (M_hat(c, t+1) - M_hat(c, t)) + lambda_ret * retention_signal - lambda_cost * intervention_cost
  Penalizes multi-concept forgetting/stall and introduces step costs.

- Reward v3 (Stage 4):
    r_t = r_v2 + transfer_term - excessive_hint_penalty - wasted_step_penalty
  Directly penalizes repeated consecutive hints and redundant drilling on already-mastered concepts,
  while crediting prerequisite transfer.
"""

from typing import Any, Callable, Mapping

# --- Hyperparameters for Reward v2 & v3 (Roadmap Sec. 15) ---
LAMBDA_RET_V2: float = 0.50          # Weight for unpracticed concept retention signal
LAMBDA_COST_V2: float = 0.015        # Step intervention cost penalty
EXCESSIVE_HINT_PENALTY_V3: float = 0.05  # Penalty if HINT is chosen consecutively
WASTED_STEP_PENALTY_V3: float = 0.05    # Penalty for drilling already-mastered concepts (>= 0.85)


def compute_reward_v1(
    prev_concept_mastery: float,
    new_concept_mastery: float,
    **kwargs: Any,
) -> float:
    """Reward v1: Mastery gain on the concept taught at step t."""
    return float(new_concept_mastery - prev_concept_mastery)


def compute_reward_v2(
    prev_concept_mastery: float,
    new_concept_mastery: float,
    prev_all_masteries: Mapping[str, float] | None = None,
    new_all_masteries: Mapping[str, float] | None = None,
    concept_id: str | None = None,
    lambda_ret: float = LAMBDA_RET_V2,
    lambda_cost: float = LAMBDA_COST_V2,
    **kwargs: Any,
) -> float:
    """Reward v2: Mastery gain + retention signal on other concepts - intervention cost."""
    gain = new_concept_mastery - prev_concept_mastery

    # Retention signal: net change across other concepts (typically negative or zero from decay)
    retention_signal = 0.0
    if prev_all_masteries is not None and new_all_masteries is not None and concept_id is not None:
        other_deltas = [
            new_all_masteries[c] - prev_all_masteries[c]
            for c in prev_all_masteries
            if c != concept_id and c in new_all_masteries
        ]
        if other_deltas:
            retention_signal = float(sum(other_deltas))

    intervention_cost = 1.0
    return float(gain + (lambda_ret * retention_signal) - (lambda_cost * intervention_cost))


def compute_reward_v3(
    prev_concept_mastery: float,
    new_concept_mastery: float,
    action_name: str,
    hint_used_last_step: bool = False,
    prerequisite_gain: float = 0.0,
    prev_all_masteries: Mapping[str, float] | None = None,
    new_all_masteries: Mapping[str, float] | None = None,
    concept_id: str | None = None,
    lambda_ret: float = LAMBDA_RET_V2,
    lambda_cost: float = LAMBDA_COST_V2,
    **kwargs: Any,
) -> float:
    """Reward v3: Reward v2 + transfer credit - excessive hint penalty - wasted step penalty."""
    r_v2 = compute_reward_v2(
        prev_concept_mastery=prev_concept_mastery,
        new_concept_mastery=new_concept_mastery,
        prev_all_masteries=prev_all_masteries,
        new_all_masteries=new_all_masteries,
        concept_id=concept_id,
        lambda_ret=lambda_ret,
        lambda_cost=lambda_cost,
    )

    # 1. Transfer credit: reward gain propagated from reviewing prerequisite
    transfer_term = float(prerequisite_gain * 0.50)

    # 2. Excessive hint penalty: penalize selecting HINT when hint was already used on last step
    excessive_hint_penalty = 0.0
    if action_name == "HINT" and hint_used_last_step:
        excessive_hint_penalty = EXCESSIVE_HINT_PENALTY_V3

    # 3. Wasted step penalty: penalize re-teaching a concept that is already mastered (>= 0.85)
    wasted_step_penalty = 0.0
    if prev_concept_mastery >= 0.85 and action_name != "REVIEW_PREREQUISITE":
        wasted_step_penalty = WASTED_STEP_PENALTY_V3

    return float(r_v2 + transfer_term - excessive_hint_penalty - wasted_step_penalty)


REWARD_FUNCTIONS: dict[str, Callable[..., float]] = {
    "v1": compute_reward_v1,
    "v2": compute_reward_v2,
    "v3": compute_reward_v3,
}


def compute_reward(
    reward_version: str,
    prev_concept_mastery: float | None = None,
    new_concept_mastery: float | None = None,
    action_name: str | None = None,
    hint_used_last_step: bool = False,
    prerequisite_gain: float = 0.0,
    prev_all_masteries: Mapping[str, float] | None = None,
    new_all_masteries: Mapping[str, float] | None = None,
    concept_id: str | None = None,
    **kwargs: Any,
) -> float:
    """Unified reward dispatcher accepting standard or keyword aliases."""
    prev_m = float(prev_concept_mastery if prev_concept_mastery is not None else kwargs.get("prev_mastery", 0.0))
    new_m = float(new_concept_mastery if new_concept_mastery is not None else kwargs.get("new_mastery", 0.0))

    if action_name is not None:
        act = action_name
    elif "action" in kwargs:
        a_val = kwargs["action"]
        act = a_val.name if hasattr(a_val, "name") else str(a_val)
    else:
        act = "PRACTICE"

    cid = concept_id or kwargs.get("current_concept_id")
    prev_all = prev_all_masteries or kwargs.get("prev_all_masteries")
    new_all = new_all_masteries or kwargs.get("all_masteries") or kwargs.get("new_all_masteries")

    fn = REWARD_FUNCTIONS.get(reward_version, compute_reward_v1)
    return fn(
        prev_concept_mastery=prev_m,
        new_concept_mastery=new_m,
        action_name=act,
        hint_used_last_step=hint_used_last_step,
        prerequisite_gain=prerequisite_gain,
        prev_all_masteries=prev_all,
        new_all_masteries=new_all,
        concept_id=cid,
    )
