"""Fixed Curriculum Teacher for AirTutor (Roadmap Sec. 16).

Strictly follows curriculum order:
Always alternates EXPLAIN -> PRACTICE on each concept, then advances to the next concept.
Completely ignores student performance, responses, and estimated mastery.
Represents a non-adaptive baseline ("one size fits all" lecturing).
"""

from src.environment.actions import PedagogicalAction
from src.teachers.base_teacher import BaseTeacher, TeacherObservation


class FixedTeacher(BaseTeacher):
    """Non-adaptive fixed curriculum baseline."""

    def __init__(self, name: str = "Fixed"):
        super().__init__(name=name)
        self._concept_steps: dict[str, int] = {}
        self._last_concept: str | None = None

    def select_action(self, obs: TeacherObservation) -> PedagogicalAction:
        concept = obs.current_concept_id

        if concept != self._last_concept:
            self._concept_steps[concept] = 0
            self._last_concept = concept

        step_on_concept = self._concept_steps.get(concept, 0)
        self._concept_steps[concept] = step_on_concept + 1

        # Strict alternating pattern: EXPLAIN on even turns, PRACTICE on odd turns
        if step_on_concept % 2 == 0:
            return PedagogicalAction.EXPLAIN
        else:
            return PedagogicalAction.PRACTICE

    def should_advance_concept(self, obs: TeacherObservation) -> bool:
        """Fixed teacher advances to the next concept immediately after practicing."""
        step_on_concept = self._concept_steps.get(obs.current_concept_id, 0)
        # Advance after every EXPLAIN -> PRACTICE pair (2 steps)
        return (step_on_concept >= 2)

    def reset(self) -> None:
        self._concept_steps.clear()
        self._last_concept = None
