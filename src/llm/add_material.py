"""'Add Material' Extensibility Flow for AirTutor (Roadmap Sec. 27 & 28).

Transforms unstructured notes or syllabus excerpts into a schema-validated
curriculum JSON module without modifying any RL or core code.

Workflow:
1. Formulates a structured schema extraction prompt from raw notes.
2. Calls the configured LLMProvider (or OfflineTemplateAdapter).
3. Validates output against the strict curriculum schema via validate_curriculum_json.
4. On failure, retries ONCE with concrete validation errors sent back to the model.
5. On a second failure, rejects output and raises ValueError surfacing all errors.
6. On success, writes to data/curriculum/{module_id}.json and returns its Path.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any

from src.llm.provider import LLMProvider, get_llm_provider
from src.llm.validation import validate_curriculum_json

logger = logging.getLogger("AirTutor.LLM.AddMaterial")

CURRICULUM_EXTRACTION_SYSTEM_PROMPT = """You are an expert educational curriculum architect for Computer Science.
Your task is to transform raw educational notes into a complete, structured curriculum JSON module.
You MUST output strictly valid JSON matching this schema:
{
  "module_id": "snake_case_string",
  "title": "Clean Human Readable Title",
  "concepts": [
    {
      "concept_id": "unique_concept_slug",
      "name": "Concept Name",
      "prerequisites": ["prerequisite_concept_id_from_this_module"],
      "difficulty": 1-5,
      "questions": [
        {
          "question_id": "unique_q_id",
          "prompt": "Question text here",
          "answer": "Expected answer text",
          "type": "mcq" | "short" | "code"
        }
      ],
      "templates": {
        "EXPLAIN": "Detailed explanation of concept...",
        "EXAMPLE": "Annotated code example...",
        "HINT": "Helpful scaffolding hint...",
        "PRACTICE": "Practice challenge prompt...",
        "REVIEW_PREREQUISITE": "Prerequisite review text..."
      }
    }
  ]
}

CRITICAL RULES:
1. Provide 2 to 5 coherent concepts derived from the notes.
2. Provide at least 2 questions per concept with distinct question types ('mcq', 'short', 'code').
3. Ensure prerequisites are ordered hierarchically with no self-references and no cycles.
4. Return ONLY valid, parseable JSON. Do not wrap in conversational markdown except optional json code fences.
"""


def _extract_json_text(text: str) -> str:
    """Extract clean JSON text from LLM response, stripping markdown fences."""
    t = text.strip()
    # Match ```json ... ``` or ``` ... ```
    fence_pattern = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if fence_pattern:
        return fence_pattern.group(1).strip()
    return t


def add_module_from_notes(
    raw_text: str,
    module_title: str,
    output_dir: str | Path = "data/curriculum",
    provider: LLMProvider | None = None,
) -> Path:
    """Create and validate a new curriculum module from raw text notes.

    Args:
        raw_text: Freeform syllabus notes, lecture summaries, or textbook excerpts.
        module_title: Human-readable title for the new module.
        output_dir: Directory to save the resulting JSON file (default 'data/curriculum').
        provider: Optional LLMProvider instance (defaults to get_llm_provider()).

    Returns:
        Path to the saved curriculum JSON file.

    Raises:
        ValueError: If curriculum generation fails validation after retry.
    """
    llm = provider or get_llm_provider()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_prompt = (
        f"Create a curriculum module from the following notes.\n"
        f"Module Title: {module_title}\n\n"
        f"Raw Notes:\n{raw_text}\n\n"
        f"Return strictly valid JSON conforming to the curriculum schema."
    )

    logger.info(f"Generating curriculum module '{module_title}' using {llm.__class__.__name__}...")

    # Attempt 1
    raw_output = llm.generate_raw(
        prompt=user_prompt,
        system_instruction=CURRICULUM_EXTRACTION_SYSTEM_PROMPT,
    )
    json_candidate = _extract_json_text(raw_output)
    is_valid, errors = validate_curriculum_json(json_candidate)

    # Attempt 2 (Retry on failure)
    if not is_valid:
        logger.warning(
            f"Curriculum generation attempt 1 failed validation ({len(errors)} errors). "
            f"Retrying with error feedback..."
        )
        retry_prompt = (
            f"The previous output had the following schema validation errors:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nPlease fix all errors and return ONLY the corrected, valid JSON object for the module."
        )
        retry_output = llm.generate_raw(
            prompt=retry_prompt,
            system_instruction=CURRICULUM_EXTRACTION_SYSTEM_PROMPT,
        )
        json_candidate = _extract_json_text(retry_output)
        is_valid, errors = validate_curriculum_json(json_candidate)

    # If still invalid after retry, reject with detailed errors
    if not is_valid:
        error_msg = (
            f"Curriculum generation failed schema validation after retry for module '{module_title}':\n"
            + "\n".join(f"  * {e}" for e in errors)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Parse and write to disk
    module_dict = json.loads(json_candidate)
    mod_id = module_dict.get("module_id", "new_module")
    # Sanitize module_id for filename
    file_slug = re.sub(r"[^a-z0-9_]+", "_", mod_id.lower()).strip("_") or "curriculum_module"
    module_dict["module_id"] = file_slug
    dest_path = out_dir / f"{file_slug}.json"

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(module_dict, f, indent=2)

    logger.info(f"Successfully created new curriculum module at: {dest_path}")
    return dest_path
