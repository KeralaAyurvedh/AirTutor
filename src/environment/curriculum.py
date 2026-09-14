"""Curriculum architecture and loader for AirTutor (Roadmap Sec. 28).

Decouples curriculum content from the RL engine via a generic schema.
Loads all JSON modules from data/curriculum/ without hard-coded content.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal

QuestionType = Literal["mcq", "short", "code"]


@dataclass(frozen=True)
class Question:
    question_id: str
    prompt: str
    answer: str
    type: QuestionType


@dataclass(frozen=True)
class Concept:
    concept_id: str
    name: str
    prerequisites: list[str]
    difficulty: int
    questions: list[Question] = field(default_factory=list)

    def get_question(self, question_id: str) -> Question | None:
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None


@dataclass
class CurriculumModule:
    module_id: str
    title: str
    concepts: list[Concept]


class CurriculumRepository:
    """In-memory repository containing all loaded curriculum modules and concepts."""

    def __init__(self, modules: list[CurriculumModule]):
        self.modules: dict[str, CurriculumModule] = {m.module_id: m for m in modules}
        self.concepts: dict[str, Concept] = {}

        for m in modules:
            for c in m.concepts:
                if c.concept_id in self.concepts:
                    raise ValueError(f"Duplicate concept_id '{c.concept_id}' across modules")
                self.concepts[c.concept_id] = c

        self._validate_prerequisites()

    def _validate_prerequisites(self) -> None:
        """Ensure all prerequisite concept IDs exist and detect simple self-referencing."""
        for c_id, concept in self.concepts.items():
            for prereq_id in concept.prerequisites:
                if prereq_id not in self.concepts:
                    raise ValueError(
                        f"Concept '{c_id}' references unknown prerequisite '{prereq_id}'"
                    )
                if prereq_id == c_id:
                    raise ValueError(f"Concept '{c_id}' cannot be a prerequisite of itself")

    def get_concept(self, concept_id: str) -> Concept:
        if concept_id not in self.concepts:
            raise KeyError(f"Concept '{concept_id}' not found in curriculum")
        return self.concepts[concept_id]

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        concept = self.get_concept(concept_id)
        return [self.get_concept(p_id) for p_id in concept.prerequisites]

    def get_module(self, module_id: str) -> CurriculumModule:
        if module_id not in self.modules:
            raise KeyError(f"Module '{module_id}' not found in curriculum")
        return self.modules[module_id]

    def get_all_concept_ids(self) -> list[str]:
        return list(self.concepts.keys())

    def get_all_concepts(self) -> list[Concept]:
        return list(self.concepts.values())


def load_module_from_file(file_path: Path | str) -> CurriculumModule:
    """Parse and validate a single curriculum JSON file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    module_id = data["module_id"]
    title = data["title"]
    raw_concepts = data["concepts"]

    concepts: list[Concept] = []
    for rc in raw_concepts:
        questions: list[Question] = []
        for rq in rc.get("questions", []):
            questions.append(
                Question(
                    question_id=rq["question_id"],
                    prompt=rq["prompt"],
                    answer=rq["answer"],
                    type=rq["type"],
                )
            )
        concepts.append(
            Concept(
                concept_id=rc["concept_id"],
                name=rc["name"],
                prerequisites=list(rc.get("prerequisites", [])),
                difficulty=int(rc.get("difficulty", 1)),
                questions=questions,
            )
        )

    return CurriculumModule(module_id=module_id, title=title, concepts=concepts)


def load_curriculum(directory: Path | str = "data/curriculum") -> CurriculumRepository:
    """Load all JSON curriculum modules from the given directory.

    Adding a new curriculum module only requires placing a new JSON file into
    data/curriculum/; no code modifications are needed.
    """
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Curriculum directory '{directory}' does not exist")

    json_files = sorted(list(path.glob("*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON curriculum modules found in '{directory}'")

    modules = [load_module_from_file(f) for f in json_files]
    return CurriculumRepository(modules)
