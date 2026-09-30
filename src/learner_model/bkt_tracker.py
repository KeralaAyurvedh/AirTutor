"""Bayesian Knowledge Tracing (BKT) Belief Tracker for AirTutor.

Maintains belief distributions over latent concept mastery based strictly on
observable student responses (binary correct/incorrect outcomes).

CRITICAL INVARIANT (Rule 1):
The BKT tracker conditions EXCLUSIVELY on observable interaction outcomes.
Neither the simulator's true latent parameters theta nor ground-truth continuous
mastery M(c, t) is ever accessible or passed to this tracker.
"""

from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_P_L0: float = 0.10  # Initial mastery prior P(L_0)
DEFAULT_P_T: float = 0.15   # Transition / learning probability P(T)
DEFAULT_P_S: float = 0.10   # Slip probability P(S)
DEFAULT_P_G: float = 0.20   # Guess probability P(G)


@dataclass
class BKTConceptParams:
    """Configurable BKT parameters for a single concept."""
    p_l0: float = DEFAULT_P_L0
    p_t: float = DEFAULT_P_T
    p_s: float = DEFAULT_P_S
    p_g: float = DEFAULT_P_G


class BKTTracker:
    """Bayesian Knowledge Tracing engine tracking concept mastery belief states."""

    def __init__(
        self,
        concepts: list[str] | None = None,
        default_params: BKTConceptParams | None = None,
        concept_params: dict[str, BKTConceptParams] | None = None,
    ):
        self.default_params: BKTConceptParams = default_params or BKTConceptParams()
        self.concept_params: dict[str, BKTConceptParams] = dict(concept_params or {})
        self._initial_concepts: list[str] = list(concepts) if concepts else []
        self._beliefs: dict[str, float] = {}
        self.reset()

    def _get_params(self, concept_id: str) -> BKTConceptParams:
        return self.concept_params.get(concept_id, self.default_params)

    def get_belief(self, concept_id: str) -> float:
        """Return the current mastery belief P(L_t) for the given concept."""
        if concept_id not in self._beliefs:
            params = self._get_params(concept_id)
            self._beliefs[concept_id] = params.p_l0
        return self._beliefs[concept_id]

    def get_all_beliefs(self) -> dict[str, float]:
        """Return a copy of all tracked concept beliefs."""
        return dict(self._beliefs)

    def update(self, concept_id: str, is_correct: bool) -> float:
        """Update mastery belief given an observed response and transition to next state.

        Formula:
        Observation update:
            If correct (o_t = 1):
                P(L_t | o_t=1) = [b_t * (1 - P(S))] / [b_t * (1 - P(S)) + (1 - b_t) * P(G)]
            If incorrect (o_t = 0):
                P(L_t | o_t=0) = [b_t * P(S)] / [b_t * P(S) + (1 - b_t) * (1 - P(G))]

        Posterior state transition:
            b_{t+1} = P(L_t | o_t) + (1 - P(L_t | o_t)) * P(T)
        """
        b_t = self.get_belief(concept_id)
        params = self._get_params(concept_id)
        p_s = params.p_s
        p_g = params.p_g
        p_t = params.p_t

        eps = 1e-12

        if is_correct:
            numerator = b_t * (1.0 - p_s)
            denominator = numerator + (1.0 - b_t) * p_g
        else:
            numerator = b_t * p_s
            denominator = numerator + (1.0 - b_t) * (1.0 - p_g)

        p_obs = numerator / max(eps, denominator)

        # Transition to next step
        b_next = p_obs + (1.0 - p_obs) * p_t

        # Strictly clamp within [0.0, 1.0]
        clamped_b = max(0.0, min(1.0, float(b_next)))
        self._beliefs[concept_id] = clamped_b
        return clamped_b

    def reset(self) -> None:
        """Reset all tracked concept beliefs to their initial priors P(L_0)."""
        self._beliefs.clear()
        for c in self._initial_concepts:
            params = self._get_params(c)
            self._beliefs[c] = params.p_l0
