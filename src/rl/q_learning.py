"""Tabular Q-Learning Agent for AirTutor (Roadmap Sec. 7 & 13).

Learns an adaptive pedagogical policy mapping observable student state indices
to discrete pedagogical actions using temporal-difference (TD) Bellman updates.

CRITICAL INVARIANT (GEMINI.md):
The policy conditions EXCLUSIVELY on the observable state feature vector produced
by src/environment/state_encoder.py. Latent theta and true latent mastery M(c, t)
are NEVER accessed by this agent.
"""

from pathlib import Path
from typing import Any
import numpy as np

from src.environment.actions import ACTIONS, NUM_ACTIONS, PedagogicalAction
from src.environment.state_encoder import (
    ObservableHistory,
    StateEncoderConfig,
    encode_state,
    get_total_state_count,
    state_tuple_to_index,
)
from src.teachers.base_teacher import BaseTeacher, TeacherObservation

# ============================================================================
# --- Named Tabular Q-Learning Hyperparameters (Roadmap Sec. 7) ---
# ============================================================================

# Learning rate (alpha): Determines the weight given to new TD error information.
DEFAULT_LEARNING_RATE: float = 0.15

# Discount factor (gamma): Balances immediate rewards against future expected gains.
DEFAULT_DISCOUNT_FACTOR: float = 0.90

# Exploration schedule: Initial epsilon probability of choosing a random exploratory action.
DEFAULT_EPSILON_START: float = 1.0

# Minimum exploration rate floor: Retains a modest level of exploration.
DEFAULT_EPSILON_MIN: float = 0.05

# Multiplicative exploration decay factor applied at the conclusion of each episode.
DEFAULT_EPSILON_DECAY: float = 0.992

# Default patience steps on a single concept before advancing in evaluation
DEFAULT_MAX_STEPS_PER_CONCEPT: int = 6


class TabularQAgent(BaseTeacher):
    """Tabular Q-learning agent implementing epsilon-greedy policy over discrete states."""

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON_START,
        epsilon_min: float = DEFAULT_EPSILON_MIN,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        state_config: StateEncoderConfig | None = None,
        seed: int = 42,
        name: str = "RL_Q_Learning",
        max_steps_per_concept: int = DEFAULT_MAX_STEPS_PER_CONCEPT,
    ):
        super().__init__(name=name)
        self.lr: float = learning_rate
        self.gamma: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.state_config: StateEncoderConfig = state_config or StateEncoderConfig()
        self.max_steps_per_concept: int = max_steps_per_concept

        self.num_states: int = get_total_state_count(self.state_config)
        self.num_actions: int = NUM_ACTIONS
        self.rng = np.random.default_rng(seed)

        # Tabular Q-table initialized to zero: shape (num_states, num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        self._steps_on_concept: dict[str, int] = {}
        self._last_concept: str | None = None

    def get_state_index(self, obs: TeacherObservation) -> int:
        """Map purely observable TeacherObservation to flat state index in [0, num_states-1]."""
        history = ObservableHistory(
            concept_id=obs.current_concept_id,
            estimated_mastery=obs.estimated_mastery,
            last_response_correct=obs.last_response_correct,
            recent_responses=obs.recent_responses,
            wrong_answer_streak=obs.wrong_answer_streak,
            hint_used_last_step=obs.hint_used_last_step,
            weakest_prereq_mastery=obs.weakest_prereq_mastery,
        )
        state_tuple = encode_state(history, self.state_config)
        return state_tuple_to_index(state_tuple, self.state_config)

    def select_action(
        self,
        obs: TeacherObservation,
        greedy: bool = False,
    ) -> PedagogicalAction:
        """Select discrete pedagogical action using epsilon-greedy policy over Q-values."""
        concept = obs.current_concept_id
        if concept != self._last_concept:
            self._steps_on_concept[concept] = 0
            self._last_concept = concept
        self._steps_on_concept[concept] = self._steps_on_concept.get(concept, 0) + 1

        state_idx = self.get_state_index(obs)

        # Exploration branch (only if not greedy evaluation)
        if (not greedy) and (self.rng.random() < self.epsilon):
            action_idx = int(self.rng.integers(0, self.num_actions))
            return ACTIONS[action_idx]

        # Exploitation branch: break ties uniformly at random
        q_row = self.q_table[state_idx]
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(q_row == max_q)
        chosen_idx = int(self.rng.choice(best_actions))
        return ACTIONS[chosen_idx]

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """Perform standard Bellman temporal-difference update.

        Formula:
            Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.q_table[state_idx, action_idx]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        td_error = target - current_q
        self.q_table[state_idx, action_idx] += self.lr * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay exploration rate epsilon by decay factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def should_advance_concept(self, obs: TeacherObservation) -> bool:
        """Advance when concept is mastered (>= 0.75) or max concept patience reached."""
        steps = self._steps_on_concept.get(obs.current_concept_id, 0)
        mastered = obs.estimated_mastery >= 0.75
        timeout = steps >= self.max_steps_per_concept
        return mastered or timeout

    def reset(self) -> None:
        self._steps_on_concept.clear()
        self._last_concept = None

    def save(self, filepath: Path | str) -> None:
        """Save Q-table to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, filepath: Path | str) -> None:
        """Load Q-table from disk."""
        path = Path(filepath)
        self.q_table = np.load(path)
