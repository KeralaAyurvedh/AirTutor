"""Observable Learner Model (Roadmap Sec. 8).

Maintains the tutor's own estimate of learner mastery M_hat(c) in [0, 1] per concept.
Deliberately distinct from the simulator's true latent mastery M(c, t).
The observable estimator updates based only on empirical interaction history.
"""

from typing import Mapping

# --- Tunable Estimator Hyperparameters (Roadmap Sec. 8) ---
# Fixed estimator learning rate constant controlling how rapidly M_hat adjusts
# toward observed response correctness (1.0 for correct, 0.0 for incorrect).
DEFAULT_ETA: float = 0.20

# Per-step time decay applied when a concept goes unpracticed.
DEFAULT_DELTA: float = 0.01

# Initial uninformative mastery prior for a concept before observations.
DEFAULT_INITIAL_MASTERY: float = 0.10


class LearnerModel:
    """Estimates concept mastery from observable student response evidence."""

    def __init__(
        self,
        concepts: list[str],
        eta: float = DEFAULT_ETA,
        delta: float = DEFAULT_DELTA,
        initial_mastery: float = DEFAULT_INITIAL_MASTERY,
    ):
        self.eta: float = eta
        self.delta: float = delta
        self.initial_mastery: float = initial_mastery

        self._mastery: dict[str, float] = {c: initial_mastery for c in concepts}
        self._last_practiced_step: dict[str, int] = {c: 0 for c in concepts}
        self._current_step: int = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    def get_mastery(self, concept_id: str) -> float:
        """Return estimated mastery M_hat for the given concept."""
        if concept_id not in self._mastery:
            raise KeyError(f"Concept '{concept_id}' not tracked by learner model")
        return self._mastery[concept_id]

    def get_all_masteries(self) -> dict[str, float]:
        """Return a copy of all estimated masteries."""
        return dict(self._mastery)

    def update(
        self,
        concept_id: str,
        correct: bool,
        steps_unpracticed: int | None = None,
    ) -> float:
        """Update estimated mastery M_hat(c) after an observed response.

        Formula (Roadmap Sec. 8):
            M_hat(c) <- M_hat(c) + eta * (correct - M_hat(c)) - delta * steps_unpracticed(c)
        clipped strictly to [0.0, 1.0].
        """
        if concept_id not in self._mastery:
            raise KeyError(f"Concept '{concept_id}' not tracked by learner model")

        self._current_step += 1

        if steps_unpracticed is None:
            # Derived from turn tracking
            last_step = self._last_practiced_step[concept_id]
            steps_unpracticed = max(0, (self._current_step - 1) - last_step)

        current_m = self._mastery[concept_id]
        target = 1.0 if correct else 0.0

        # Update rule
        delta_m = self.eta * (target - current_m) - (self.delta * steps_unpracticed)
        updated_m = current_m + delta_m

        # Clip strictly to [0.0, 1.0]
        clipped_m = max(0.0, min(1.0, updated_m))
        self._mastery[concept_id] = clipped_m
        self._last_practiced_step[concept_id] = self._current_step

        return clipped_m

    def apply_decay(self, concept_id: str, elapsed_steps: int = 1) -> float:
        """Explicitly decay an unpracticed concept's estimated mastery."""
        if concept_id not in self._mastery:
            raise KeyError(f"Concept '{concept_id}' not tracked by learner model")
        current_m = self._mastery[concept_id]
        decayed = max(0.0, min(1.0, current_m - (self.delta * elapsed_steps)))
        self._mastery[concept_id] = decayed
        return decayed

    def step_decay_unpracticed(self, practiced_concept_id: str) -> None:
        """Decay all concepts other than the one currently practiced by one step."""
        for c in self._mastery:
            if c != practiced_concept_id:
                self._mastery[c] = max(0.0, min(1.0, self._mastery[c] - self.delta))
