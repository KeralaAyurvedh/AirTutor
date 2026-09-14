"""Parametric Simulated Learner Population (Roadmap Sec. 9).

Each simulated learner draws a latent parameter vector:
    theta = [alpha, beta, gamma, h, tau, sigma]
independently from parametric distributions.

WARNING: theta and true latent mastery M(c, t) are strictly simulator-internal.
They must NEVER be exposed to the observable learner model or RL policy inputs.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np

# Default parameter bounds if configs/learner_population.json is not provided
DEFAULT_PARAM_RANGES = {
    "alpha": (0.05, 0.35),  # Learning rate
    "beta": (0.01, 0.15),   # Forgetting rate
    "gamma": (0.5, 2.0),    # Difficulty sensitivity
    "h": (0.0, 1.0),        # Hint dependence
    "tau": (0.2, 1.0),      # Transfer ability
    "sigma": (0.02, 0.10),  # Response noise
}


def load_population_config(config_path: str | Path = "configs/learner_population.json") -> dict[str, tuple[float, float]]:
    """Load parameter ranges from configuration file."""
    path = Path(config_path)
    if not path.is_file():
        return dict(DEFAULT_PARAM_RANGES)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranges = {}
        for k, v in data.get("parameter_ranges", {}).items():
            ranges[k] = (float(v["min"]), float(v["max"]))
        return ranges
    except Exception:
        return dict(DEFAULT_PARAM_RANGES)


@dataclass(frozen=True)
class LearnerParameters:
    """Simulator-internal latent learner parameter vector theta."""
    alpha: float  # Learning rate: mastery gain per intervention
    beta: float   # Forgetting rate: decay per unpracticed step
    gamma: float  # Difficulty sensitivity: impact of difficulty mismatch
    h: float      # Hint dependence: short-term boost vs durable gain
    tau: float    # Transfer ability: prerequisite propagation fraction
    sigma: float  # Response noise: stochasticity in correctness

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "h": self.h,
            "tau": self.tau,
            "sigma": self.sigma,
        }


class SimulatedLearner:
    """A simulated student instance holding latent traits and ground-truth mastery.

    CRITICAL ARCHITECTURAL BARRIER:
    Neither `theta` nor `_true_mastery` may be accessed by the RL policy or
    observable learner model.
    """

    def __init__(
        self,
        learner_id: str,
        theta: LearnerParameters,
        concepts: list[str] | None = None,
        initial_mastery: float = 0.0,
        seed: int = 42,
    ):
        self.learner_id: str = learner_id
        self._theta: LearnerParameters = theta
        self.rng = np.random.default_rng(seed)

        concept_list = concepts or []
        self._true_mastery: dict[str, float] = {c: float(initial_mastery) for c in concept_list}
        self._last_practiced: dict[str, int] = {c: 0 for c in concept_list}
        self._step_count: int = 0

    @property
    def theta(self) -> LearnerParameters:
        """Access internal theta (Simulator dynamics use only)."""
        return self._theta

    @property
    def step_count(self) -> int:
        return self._step_count

    def get_true_mastery(self, concept_id: str) -> float:
        """Access ground-truth continuous mastery M(c, t) (Simulator/evaluation use only)."""
        if concept_id not in self._true_mastery:
            self._true_mastery[concept_id] = 0.0
            self._last_practiced[concept_id] = self._step_count
        return self._true_mastery[concept_id]

    def set_true_mastery(self, concept_id: str, mastery: float) -> None:
        """Update ground-truth mastery strictly within [0.0, 1.0]."""
        self._true_mastery[concept_id] = max(0.0, min(1.0, float(mastery)))

    def get_all_true_masteries(self) -> dict[str, float]:
        """Return full ground-truth mastery profile (Simulator/evaluation use only)."""
        return dict(self._true_mastery)

    def get_last_practiced_step(self, concept_id: str) -> int:
        return self._last_practiced.get(concept_id, 0)

    def record_practice(self, concept_id: str) -> None:
        self._step_count += 1
        self._last_practiced[concept_id] = self._step_count


def sample_learner(
    seed: int,
    learner_id: str | None = None,
    concepts: list[str] | None = None,
    initial_mastery: float = 0.0,
    param_ranges: dict[str, tuple[float, float]] | None = None,
) -> SimulatedLearner:
    """Sample a single simulated learner from parameter distributions using a fixed seed."""
    ranges = param_ranges or load_population_config()
    rng = np.random.default_rng(seed)

    alpha = float(rng.uniform(*ranges.get("alpha", DEFAULT_PARAM_RANGES["alpha"])))
    beta = float(rng.uniform(*ranges.get("beta", DEFAULT_PARAM_RANGES["beta"])))
    gamma = float(rng.uniform(*ranges.get("gamma", DEFAULT_PARAM_RANGES["gamma"])))
    h = float(rng.uniform(*ranges.get("h", DEFAULT_PARAM_RANGES["h"])))
    tau = float(rng.uniform(*ranges.get("tau", DEFAULT_PARAM_RANGES["tau"])))
    sigma = float(rng.uniform(*ranges.get("sigma", DEFAULT_PARAM_RANGES["sigma"])))

    theta = LearnerParameters(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        h=h,
        tau=tau,
        sigma=sigma,
    )

    lid = learner_id or f"sim_learner_seed_{seed}"
    return SimulatedLearner(
        learner_id=lid,
        theta=theta,
        concepts=concepts,
        initial_mastery=initial_mastery,
        seed=seed,
    )


def sample_population(
    seed_start: int,
    num_learners: int,
    concepts: list[str] | None = None,
    initial_mastery: float = 0.0,
    param_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[SimulatedLearner]:
    """Sample a cohort of learners across a contiguous seed range [seed_start, seed_start + num_learners).

    Enables disjoint seed ranges for training (e.g. 0-499) vs. held-out evaluation (e.g. 10000-10099).
    """
    ranges = param_ranges or load_population_config()
    population: list[SimulatedLearner] = []

    for i in range(num_learners):
        seed = seed_start + i
        learner = sample_learner(
            seed=seed,
            learner_id=f"learner_seed_{seed}",
            concepts=concepts,
            initial_mastery=initial_mastery,
            param_ranges=ranges,
        )
        population.append(learner)

    return population
