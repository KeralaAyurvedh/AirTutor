"""Rule-Based Adaptive Teacher for AirTutor (Roadmap Sec. 16).

A genuinely competitive, transparent hand-designed pedagogical heuristic:
- If weakest prerequisite is weak (< 0.40) or unreviewed for several steps: REVIEW_PREREQUISITE.
- Elif current concept estimated mastery < 0.40: EXPLAIN.
- Elif last response was incorrect (struggle detected): HINT.
- Elif current concept estimated mastery < 0.70: EXAMPLE.
- Else (mastery >= 0.70): PRACTICE.

Advances to the next curriculum concept when mastery threshold (>= 0.75) is achieved
or concept patience limit is reached.
"""

from src.environment.actions import PedagogicalAction
from src.teachers.base_teacher import BaseTeacher, TeacherObservation


class RuleTeacher(BaseTeacher):
    """Adaptive heuristic teacher baseline."""

    def __init__(
        self,
        name: str = "Rule",
        mastery_threshold: float = 0.75,
        prereq_review_threshold: float = 0.40,
        review_interval: int = 5,
        max_steps_per_concept: int = 8,
    ):
        super().__init__(name=name)
        self.mastery_threshold: float = mastery_threshold
        self.prereq_review_threshold: float = prereq_review_threshold
        self.review_interval: int = review_interval
        self.max_steps_per_concept: int = max_steps_per_concept

        self._steps_on_concept: dict[str, int] = {}
        self._steps_since_review: int = 0
        self._last_concept: str | None = None

    def select_action(self, obs: TeacherObservation) -> PedagogicalAction:
        concept = obs.current_concept_id
        if concept != self._last_concept:
            self._steps_on_concept[concept] = 0
            self._last_concept = concept

        self._steps_on_concept[concept] = self._steps_on_concept.get(concept, 0) + 1
        self._steps_since_review += 1

        # 1. Periodically check and review weak prerequisite if available
        if obs.weakest_prereq_id is not None and obs.weakest_prereq_mastery is not None:
            weak_prereq = obs.weakest_prereq_mastery < self.prereq_review_threshold
            periodic_check = (
                self._steps_since_review >= self.review_interval
                and obs.weakest_prereq_mastery < 0.65
            )
            if weak_prereq or periodic_check:
                self._steps_since_review = 0
                return PedagogicalAction.REVIEW_PREREQUISITE

        m = obs.estimated_mastery

        # 2. Rescue student with a hint if sustained struggle (2+ wrong answers)
        if obs.wrong_answer_streak >= 2:
            return PedagogicalAction.HINT

        # 3. Foundational explanation if low mastery
        if m < 0.40:
            return PedagogicalAction.EXPLAIN

        # 4. Scaffold with hint if student struggled on the last response
        if obs.last_response_correct is False or obs.wrong_answer_streak >= 1:
            return PedagogicalAction.HINT

        # 5. Exemplify for intermediate understanding
        if m < 0.70:
            return PedagogicalAction.EXAMPLE

        # 6. Practice/drill when mastery is strong (peaks around desirable difficulty)
        return PedagogicalAction.PRACTICE

    def should_advance_concept(self, obs: TeacherObservation) -> bool:
        """Advance when concept is mastered or max patience reached."""
        steps = self._steps_on_concept.get(obs.current_concept_id, 0)
        mastered = obs.estimated_mastery >= self.mastery_threshold
        timeout = steps >= self.max_steps_per_concept
        return mastered or timeout

    def reset(self) -> None:
        self._steps_on_concept.clear()
        self._steps_since_review = 0
        self._last_concept = None
