"""Offline Template Adapter for AirTutor (Roadmap Sec. 24 & 26).

Pulls pre-written pedagogical template text keyed by action + concept from
curriculum JSON or deterministic educational templates. Zero network dependency.
Acts as the universal fallback adapter for all live LLM providers on failure.
"""

import json
import re
from typing import Any

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import Concept
from src.llm.provider import LLMProvider, TutorResponse

# Fallback generic pedagogical templates if concept JSON lacks pre-written templates
GENERIC_TEMPLATES: dict[str, str] = {
    "EXPLAIN": (
        "Let's review the core concepts of {name}. {name} is a fundamental topic in computer science. "
        "Understanding its mechanics, state transformations, and operational rules provides the foundation "
        "for subsequent algorithms and system design."
    ),
    "EXAMPLE": (
        "Here is a concrete illustrative example for {name}:\n"
        "Consider how input values are initialized, transformed step-by-step, and evaluated. "
        "Tracing execution line-by-line clarifies how the concept operates in practical scenarios."
    ),
    "HINT": (
        "Helpful Hint for {name}: Break the problem down into its elementary operations. "
        "Check variable boundary conditions, edge cases, and ensure type expectations match the operation."
    ),
    "PRACTICE": (
        "Practice Challenge for {name}: Apply what you've learned to solve a targeted problem. "
        "Consider the inputs, expected return values, and write out your logic carefully."
    ),
    "REVIEW_PREREQUISITE": (
        "Foundational Review for {name}: Before advancing, let's revisit the prerequisite principles. "
        "Solidifying foundational knowledge ensures a smooth transition to this concept."
    ),
}


class OfflineTemplateAdapter(LLMProvider):
    """Zero-network template adapter that renders tutoring content from curriculum templates."""

    def __init__(self) -> None:
        self.provider_name = "offline"

    def generate_tutoring_response(
        self,
        action: PedagogicalAction | str,
        concept: Concept | str,
        learner_state: Any = None,
        curriculum_context: Any = None,
    ) -> TutorResponse:
        """Render action into text using curriculum-stored templates or generic fallbacks."""
        act_name = action.name if isinstance(action, PedagogicalAction) else str(action).upper()
        cid = concept.concept_id if isinstance(concept, Concept) else str(concept)
        cname = concept.name if isinstance(concept, Concept) else str(concept).replace("_", " ").title()

        # 1. Try concept-stored templates from curriculum JSON
        text: str | None = None
        if isinstance(concept, Concept) and hasattr(concept, "templates"):
            text = concept.templates.get(act_name)

        # 2. Fall back to generic template formatted with concept name
        if not text:
            template = GENERIC_TEMPLATES.get(act_name, GENERIC_TEMPLATES["EXPLAIN"])
            text = template.format(name=cname, concept_id=cid)

        return TutorResponse(
            text=text,
            success=True,
            meta={
                "provider": self.provider_name,
                "model": "template-engine",
                "action": act_name,
                "concept_id": cid,
                "cached": True,
            },
        )

    def generate_raw(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate deterministic text/JSON for prompt completions when offline."""
        # Detect if asking for curriculum generation from notes
        if "curriculum module" in prompt.lower() or "module_id" in prompt.lower():
            # Extract possible title or topic
            title_match = re.search(r"title[:\s]+([^\n\r]+)", prompt, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Generated Module"
            slug = re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_") or "new_topic_module"

            sample_module = {
                "module_id": slug,
                "title": title,
                "concepts": [
                    {
                        "concept_id": f"{slug}_fundamentals",
                        "name": f"{title} Fundamentals",
                        "prerequisites": [],
                        "difficulty": 2,
                        "questions": [
                            {
                                "question_id": f"{slug}_q1",
                                "prompt": f"What is the primary purpose of {title} in software systems?",
                                "answer": "To organize and modularize program behavior and data.",
                                "type": "short",
                            },
                            {
                                "question_id": f"{slug}_q2",
                                "prompt": f"Which of the following best characterizes {title}?",
                                "answer": "Structured abstraction",
                                "type": "mcq",
                            },
                        ],
                        "templates": {
                            "EXPLAIN": f"An overview of {title} fundamentals and core principles.",
                            "EXAMPLE": f"An illustrative code example demonstrating {title}.",
                            "HINT": f"Think about how {title} manages abstraction and scope.",
                            "PRACTICE": f"Write a brief definition or implementation of {title}.",
                            "REVIEW_PREREQUISITE": "Review foundational concepts preceding this topic.",
                        },
                    },
                    {
                        "concept_id": f"{slug}_applications",
                        "name": f"{title} Applications",
                        "prerequisites": [f"{slug}_fundamentals"],
                        "difficulty": 3,
                        "questions": [
                            {
                                "question_id": f"{slug}_q3",
                                "prompt": f"How do applications of {title} handle edge cases?",
                                "answer": "By validating input constraints and handling errors gracefully.",
                                "type": "short",
                            },
                            {
                                "question_id": f"{slug}_q4",
                                "prompt": f"Write code applying {title}.",
                                "answer": "pass",
                                "type": "code",
                            },
                        ],
                        "templates": {
                            "EXPLAIN": f"Advanced practical applications of {title}.",
                            "EXAMPLE": f"Application code example demonstrating {title}.",
                            "HINT": f"Check prerequisite relationships in {title}.",
                            "PRACTICE": f"Implement a complete routine using {title}.",
                            "REVIEW_PREREQUISITE": f"Review {title} Fundamentals before applying them.",
                        },
                    },
                ],
            }
            return json.dumps(sample_module, indent=2)

        return (
            "Offline template response: The system is operating in offline mode. "
            "Pedagogical interventions and structured templates are served locally."
        )
