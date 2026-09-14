"""Random Baseline Teacher for AirTutor (Roadmap Sec. 16).

Selects uniformly at random among all 5 discrete pedagogical actions.
Serves as an empirical lower bound: any principled policy must beat this baseline.
"""

import numpy as np

from src.environment.actions import ACTIONS, PedagogicalAction
from src.teachers.base_teacher import BaseTeacher, TeacherObservation


class RandomTeacher(BaseTeacher):
    """Uniform random teacher baseline."""

    def __init__(self, seed: int = 42, name: str = "Random", steps_per_concept: int = 6):
        super().__init__(name=name)
        self.rng = np.random.default_rng(seed)
        self.steps_per_concept: int = steps_per_concept
        self._steps_on_concept: int = 0

    def select_action(self, obs: TeacherObservation) -> PedagogicalAction:
        action_idx = int(self.rng.integers(0, len(ACTIONS)))
        return ACTIONS[action_idx]

    def should_advance_concept(self, obs: TeacherObservation) -> bool:
        self._steps_on_concept += 1
        if self._steps_on_concept >= self.steps_per_concept:
            self._steps_on_concept = 0
            return True
        return False

    def reset(self) -> None:
        self._steps_on_concept = 0
