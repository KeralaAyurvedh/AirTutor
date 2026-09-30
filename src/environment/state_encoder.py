"""Observable State Representation & Encoder for AirTutor (Roadmap Sec. 13).

Maps observable learner interaction history into a discrete feature tuple and flat
state index for tabular Q-learning.

Features and bucket counts (default v1 configuration):
1. Current concept's estimated mastery bucket: 5 (0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
2. Last response correct: 2 (0: False/None, 1: True)
3. Accuracy over last 3 responses: 4 (0, 1, 2, 3 correct)
4. Consecutive wrong-answer streak: 4 (0, 1, 2, 3+)
5. Hint used on last step: 2 (0: No, 1: Yes)

Total state count: 5 x 2 x 4 x 4 x 2 = 320 states (1,600 Q-values across 5 actions).
Each feature is independently toggleable via StateEncoderConfig to support modular
ablation experiments (Roadmap Sec. 22).
"""

from dataclasses import dataclass, field
import functools
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class StateEncoderConfig:
    """Configures which observable features are included in the state representation."""
    use_mastery_bucket: bool = True       # 5 buckets
    use_last_correct: bool = True         # 2 buckets
    use_recent_accuracy: bool = True      # 4 buckets
    use_wrong_streak: bool = True         # 4 buckets
    use_hint_used: bool = True            # 2 buckets
    use_weakest_prereq: bool = False      # 5 buckets (optional v1.1 extension)


@dataclass
class ObservableHistory:
    """Encapsulates observable student interaction history (strictly no latent theta)."""
    concept_id: str
    estimated_mastery: float
    last_response_correct: bool | None = None
    recent_responses: list[bool] = field(default_factory=list)  # tracks up to last 3
    wrong_answer_streak: int = 0
    hint_used_last_step: bool = False
    weakest_prereq_mastery: float | None = None

    def record_step(self, correct: bool, action_name: str) -> None:
        """Update history given the latest interaction outcome."""
        self.last_response_correct = correct
        self.recent_responses.append(correct)
        if len(self.recent_responses) > 3:
            self.recent_responses.pop(0)

        if correct:
            self.wrong_answer_streak = 0
        else:
            self.wrong_answer_streak += 1

        self.hint_used_last_step = (action_name == "HINT")


def discretize_mastery(mastery: float) -> int:
    """Discretize continuous estimated mastery in [0, 1] into 5 buckets."""
    m = max(0.0, min(1.0, float(mastery)))
    if m < 0.20:
        return 0
    elif m < 0.40:
        return 1
    elif m < 0.60:
        return 2
    elif m < 0.80:
        return 3
    else:
        return 4


def discretize_last_correct(last_correct: bool | None) -> int:
    """Discretize last response correctness into 2 buckets (0: False/None, 1: True)."""
    return 1 if (last_correct is True) else 0


def discretize_recent_accuracy(recent_responses: Sequence[bool]) -> int:
    """Discretize last 3 responses into 4 buckets (0, 1, 2, 3 correct)."""
    last_three = recent_responses[-3:] if len(recent_responses) >= 3 else recent_responses
    correct_count = sum(1 for r in last_three if r)
    return min(3, correct_count)


def discretize_wrong_streak(wrong_streak: int) -> int:
    """Discretize consecutive wrong answers into 4 buckets (0, 1, 2, 3+)."""
    return min(3, max(0, int(wrong_streak)))


def discretize_hint_used(hint_used: bool) -> int:
    """Discretize whether hint was used on last step into 2 buckets (0: No, 1: Yes)."""
    return 1 if hint_used else 0


def get_feature_dimensions(config: StateEncoderConfig = StateEncoderConfig()) -> list[int]:
    """Return the cardinality (bucket count) of each active feature in order."""
    dims: list[int] = []
    if config.use_mastery_bucket:
        dims.append(5)
    if config.use_last_correct:
        dims.append(2)
    if config.use_recent_accuracy:
        dims.append(4)
    if config.use_wrong_streak:
        dims.append(4)
    if config.use_hint_used:
        dims.append(2)
    if config.use_weakest_prereq:
        dims.append(5)
    return dims


def get_total_state_count(config: StateEncoderConfig = StateEncoderConfig()) -> int:
    """Return total number of discrete states under the current encoder configuration."""
    dims = get_feature_dimensions(config)
    if not dims:
        return 1
    return functools.reduce(lambda x, y: x * y, dims, 1)


def encode_state(
    history: ObservableHistory,
    config: StateEncoderConfig = StateEncoderConfig(),
) -> tuple[int, ...]:
    """Map observable interaction history to a discrete feature tuple.

    Under default config, returns a 5-tuple:
        (mastery_bucket, last_correct, recent_accuracy, wrong_streak, hint_used)
    """
    features: list[int] = []

    if config.use_mastery_bucket:
        features.append(discretize_mastery(history.estimated_mastery))

    if config.use_last_correct:
        features.append(discretize_last_correct(history.last_response_correct))

    if config.use_recent_accuracy:
        features.append(discretize_recent_accuracy(history.recent_responses))

    if config.use_wrong_streak:
        features.append(discretize_wrong_streak(history.wrong_answer_streak))

    if config.use_hint_used:
        features.append(discretize_hint_used(history.hint_used_last_step))

    if config.use_weakest_prereq:
        prereq_m = history.weakest_prereq_mastery if history.weakest_prereq_mastery is not None else 1.0
        features.append(discretize_mastery(prereq_m))

    return tuple(features)


def state_tuple_to_index(
    state_tuple: tuple[int, ...],
    config: StateEncoderConfig = StateEncoderConfig(),
) -> int:
    """Convert an encoded feature tuple to a flat scalar state index in [0, N-1]."""
    dims = get_feature_dimensions(config)
    if len(state_tuple) != len(dims):
        raise ValueError(
            f"State tuple length {len(state_tuple)} does not match config dimensions {len(dims)}"
        )

    index = 0
    multiplier = 1
    # Mixed-radix row-major indexing (from rightmost to leftmost)
    for val, dim in zip(reversed(state_tuple), reversed(dims)):
        if not (0 <= val < dim):
            raise ValueError(f"Feature value {val} out of bounds for dimension size {dim}")
        index += val * multiplier
        multiplier *= dim

    return index


def index_to_state_tuple(
    state_index: int,
    config: StateEncoderConfig = StateEncoderConfig(),
) -> tuple[int, ...]:
    """Convert a flat scalar state index in [0, N-1] back to an encoded feature tuple."""
    dims = get_feature_dimensions(config)
    total_states = get_total_state_count(config)

    if not (0 <= state_index < total_states):
        raise ValueError(f"State index {state_index} out of bounds for total states {total_states}")

    tuple_vals: list[int] = []
    curr = state_index
    for dim in reversed(dims):
        tuple_vals.append(curr % dim)
        curr //= dim

    return tuple(reversed(tuple_vals))


# ============================================================================
# --- Fixed-Size 14-Dimensional Cognitive State Encoder (DQN Upgrade) ---
# Strictly invariant to curriculum length (works for 3 or 100 concepts).
# ============================================================================

COGNITIVE_STATE_DIM: int = 14

PEDAGOGICAL_ACTION_MAP: dict[str, int] = {
    "EXPLAIN": 0,
    "EXAMPLE": 1,
    "HINT": 2,
    "PRACTICE": 3,
    "QUIZ": 3,
    "REVIEW_PREREQUISITE": 4,
    "ADVANCE": 4,
}


def _resolve_action_index(action: Any) -> int | None:
    """Resolve an action enum, string, or integer index to an action index in [0, 4]."""
    if action is None:
        return None
    if isinstance(action, int):
        return action if (0 <= action < 5) else None
    if hasattr(action, "name"):
        name = action.name.upper()
    else:
        name = str(action).upper()
    return PEDAGOGICAL_ACTION_MAP.get(name, None)


def encode_cognitive_state(
    bkt_tracker: Any,
    current_concept_id: str,
    prerequisite_concept_id: str | None = None,
    all_concept_ids: Sequence[str] | None = None,
    consecutive_failures: int = 0,
    session_steps: int = 0,
    max_session_steps: int = 30,
    previous_action: Any = None,
    last_response_correct: bool | None = None,
    as_tensor: bool = False,
) -> np.ndarray:
    """Construct the fixed 14-dimensional continuous cognitive state vector.

    Feature Layout:
    [0]:  BKT belief of current active concept b(curr) in [0, 1]
    [1]:  BKT belief of prerequisite concept b(prereq) in [0, 1] (1.0 if no prereq)
    [2]:  Mean BKT belief across all concepts in curriculum in [0, 1]
    [3]:  Minimum BKT belief across all concepts in [0, 1]
    [4]:  Variance of BKT beliefs across all concepts in [0, 1]
    [5]:  Curriculum mastery ratio: fraction of concepts with belief >= 0.85
    [6]:  Normalized failure streak: min(consecutive_failures, 5) / 5.0 in [0, 1]
    [7]:  Fatigue indicator: min(session_steps / max_session_steps, 1.0) in [0, 1]
    [8]:  One-hot action: EXPLAIN
    [9]:  One-hot action: EXAMPLE
    [10]: One-hot action: HINT
    [11]: One-hot action: PRACTICE / QUIZ
    [12]: One-hot action: REVIEW_PREREQUISITE / ADVANCE
    [13]: Binary indicator of last observed response (1.0 = correct, 0.0 = incorrect/unobserved)
    """
    # 1. Extract beliefs dictionary from BKTTracker or dict
    if hasattr(bkt_tracker, "get_all_beliefs") and hasattr(bkt_tracker, "get_belief"):
        all_beliefs_dict = bkt_tracker.get_all_beliefs()
        b_curr = float(bkt_tracker.get_belief(current_concept_id))
        if prerequisite_concept_id:
            b_prereq = float(bkt_tracker.get_belief(prerequisite_concept_id))
        else:
            b_prereq = 1.0
    elif isinstance(bkt_tracker, dict):
        all_beliefs_dict = bkt_tracker
        b_curr = float(bkt_tracker.get(current_concept_id, 0.10))
        b_prereq = float(bkt_tracker.get(prerequisite_concept_id, 1.0)) if prerequisite_concept_id else 1.0
    else:
        all_beliefs_dict = {current_concept_id: 0.10}
        b_curr = 0.10
        b_prereq = 1.0

    # 2. Gather list of concept IDs for curriculum-level statistics
    if all_concept_ids is not None and len(all_concept_ids) > 0:
        c_ids = list(all_concept_ids)
    elif len(all_beliefs_dict) > 0:
        c_ids = list(all_beliefs_dict.keys())
    else:
        c_ids = [current_concept_id]

    # Collect beliefs for all curriculum concepts
    if hasattr(bkt_tracker, "get_belief"):
        belief_values = [float(bkt_tracker.get_belief(cid)) for cid in c_ids]
    else:
        belief_values = [float(all_beliefs_dict.get(cid, 0.10)) for cid in c_ids]

    # Calculate curriculum aggregate statistics
    mean_belief = float(np.mean(belief_values)) if belief_values else b_curr
    min_belief = float(np.min(belief_values)) if belief_values else b_curr
    var_belief = float(np.var(belief_values)) if belief_values else 0.0
    mastery_ratio = (
        float(sum(1 for b in belief_values if b >= 0.85) / len(belief_values))
        if belief_values
        else 0.0
    )

    # 3. Behavioral and temporal indicators
    norm_failures = float(min(max(0, consecutive_failures), 5) / 5.0)
    max_steps = max(1, max_session_steps)
    fatigue = float(min(max(0, session_steps) / max_steps, 1.0))

    # 4. One-hot action encoding (size 5)
    action_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0]
    act_idx = _resolve_action_index(previous_action)
    if act_idx is not None and 0 <= act_idx < 5:
        action_one_hot[act_idx] = 1.0

    # 5. Last response indicator
    last_resp = 1.0 if (last_response_correct is True) else 0.0

    # Construct complete 14-D vector
    state_vector = np.array(
        [
            b_curr,
            b_prereq,
            mean_belief,
            min_belief,
            var_belief,
            mastery_ratio,
            norm_failures,
            fatigue,
            action_one_hot[0],
            action_one_hot[1],
            action_one_hot[2],
            action_one_hot[3],
            action_one_hot[4],
            last_resp,
        ],
        dtype=np.float32,
    )

    if as_tensor:
        import torch
        return torch.from_numpy(state_vector)

    return state_vector


class CognitiveStateEncoder:
    """Encoder producing fixed 14-dimensional cognitive state vectors."""

    def __init__(self, max_session_steps: int = 30):
        self.max_session_steps = max_session_steps

    def encode(
        self,
        bkt_tracker: Any,
        current_concept_id: str,
        prerequisite_concept_id: str | None = None,
        all_concept_ids: Sequence[str] | None = None,
        consecutive_failures: int = 0,
        session_steps: int = 0,
        previous_action: Any = None,
        last_response_correct: bool | None = None,
        as_tensor: bool = False,
    ) -> np.ndarray:
        return encode_cognitive_state(
            bkt_tracker=bkt_tracker,
            current_concept_id=current_concept_id,
            prerequisite_concept_id=prerequisite_concept_id,
            all_concept_ids=all_concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=session_steps,
            max_session_steps=self.max_session_steps,
            previous_action=previous_action,
            last_response_correct=last_response_correct,
            as_tensor=as_tensor,
        )


# Module-level convenience function / alias
encode = encode_cognitive_state

