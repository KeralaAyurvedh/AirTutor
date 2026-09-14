"""Base Teacher Interface for AirTutor (Roadmap Sec. 16).

Defines the teacher abstraction for all pedagogical agents (Random, Fixed, Rule, RL).
Teachers condition ONLY on observable student signals; latent theta and true continuous
mastery M(c, t) are strictly inaccessible.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.environment.actions import PedagogicalAction


@dataclass
class TeacherObservation:
    """Observable context provided to the teacher at each decision step."""
    current_concept_id: str
    estimated_mastery: float
    all_estimated_masteries: dict[str, float]
    prerequisite_ids: list[str] = field(default_factory=list)
    weakest_prereq_id: str | None = None
    weakest_prereq_mastery: float | None = None
    last_response_correct: bool | None = None
    recent_responses: list[bool] = field(default_factory=list)
    wrong_answer_streak: int = 0
    hint_used_last_step: bool = False
    step_count: int = 0


class BaseTeacher(ABC):
    """Abstract interface for all baseline and learned teachers."""

    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    def select_action(self, obs: TeacherObservation) -> PedagogicalAction:
        """Select a discrete pedagogical action given the observable state."""
        pass

    def reset(self) -> None:
        """Reset internal episodic state (e.g. curriculum pointers)."""
        pass
