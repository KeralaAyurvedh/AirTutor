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
from typing import Sequence


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
