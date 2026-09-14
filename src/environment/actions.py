"""Pedagogical Action Space for AirTutor (Roadmap Sec. 14).

Defines the discrete set of pedagogical interventions the tutor can select:
- EXPLAIN: Conceptual explanation.
- EXAMPLE: Concrete code or application example.
- HINT: Scaffolding hint boosting immediate response without high durable gain.
- PRACTICE: Assessment/drill question testing current concept.
- REVIEW_PREREQUISITE: Revisits prerequisite concept to reinforce foundational mastery.
"""

from enum import Enum


class PedagogicalAction(str, Enum):
    EXPLAIN = "EXPLAIN"
    EXAMPLE = "EXAMPLE"
    HINT = "HINT"
    PRACTICE = "PRACTICE"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"

    @classmethod
    def from_index(cls, index: int) -> "PedagogicalAction":
        actions = list(cls)
        if 0 <= index < len(actions):
            return actions[index]
        raise ValueError(f"Action index {index} out of bounds for {cls.__name__}")

    def to_index(self) -> int:
        return list(PedagogicalAction).index(self)


ACTIONS = list(PedagogicalAction)
NUM_ACTIONS = len(ACTIONS)
