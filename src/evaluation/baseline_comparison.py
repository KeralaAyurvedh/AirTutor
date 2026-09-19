"""Baseline Teacher Comparison Experiment for AirTutor (Roadmap Sec. 16 & 19).

Evaluates the three baseline teachers across 50 simulated tutoring episodes:
1. Random: Uniform random action choice.
2. Fixed: Non-adaptive curriculum order (EXPLAIN -> PRACTICE -> next).
3. Rule: Competitive adaptive pedagogical heuristic.

Definition of Done for Stage 2:
Computes and prints all six core evaluation metrics for each teacher and verifies
that the Rule baseline visibly outperforms Random and Fixed on learning gain.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from typing import Sequence
import numpy as np

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import CurriculumRepository, load_curriculum
from src.evaluation.metrics import EpisodeMetrics, aggregate_metrics, evaluate_episode
from src.learner_model.mastery import LearnerModel
from src.simulator.dynamics import step_simulation
from src.simulator.population import SimulatedLearner, sample_learner
from src.teachers.base_teacher import BaseTeacher, TeacherObservation
from src.teachers.fixed_teacher import FixedTeacher
from src.teachers.random_teacher import RandomTeacher
from src.teachers.rule_teacher import RuleTeacher


def run_tutoring_episode(
    teacher: BaseTeacher,
    learner: SimulatedLearner,
    curriculum: CurriculumRepository,
    max_steps: int = 30,
    initial_mastery: float = 0.10,
    action_effect_multiplier: float = 1.0,
    difficulty_offset: float = 0.0,
) -> tuple[EpisodeMetrics, float]:
    """Execute a complete simulated tutoring episode using observable teacher decisions.

    Returns:
        (EpisodeMetrics, true_learning_gain)
    """
    teacher.reset()
    concept_ids = curriculum.get_all_concept_ids()
    model = LearnerModel(concepts=concept_ids, initial_mastery=initial_mastery)

    # Initial observable and ground-truth masteries
    initial_obs_m = model.get_all_masteries()
    initial_true_m = learner.get_all_true_masteries()

    concept_idx = 0
    current_concept_id = concept_ids[concept_idx]

    action_history: list[PedagogicalAction] = []
    obs_mastery_history: list[dict[str, float]] = []
    engaged_concepts: set[str] = set()

    last_response_correct: bool | None = None
    recent_responses: list[bool] = []
    wrong_streak: int = 0
    hint_used_last: bool = False

    for step in range(1, max_steps + 1):
        concept = curriculum.get_concept(current_concept_id)

        # Identify weakest prerequisite if any
        prereqs = concept.prerequisites
        weakest_prereq_id = None
        weakest_prereq_m = None
        if prereqs:
            prereq_masteries = [(p, model.get_mastery(p)) for p in prereqs]
            weakest_prereq_id, weakest_prereq_m = min(prereq_masteries, key=lambda x: x[1])

        # 1. Build purely observable observation
        obs = TeacherObservation(
            current_concept_id=current_concept_id,
            estimated_mastery=model.get_mastery(current_concept_id),
            all_estimated_masteries=model.get_all_masteries(),
            prerequisite_ids=prereqs,
            weakest_prereq_id=weakest_prereq_id,
            weakest_prereq_mastery=weakest_prereq_m,
            last_response_correct=last_response_correct,
            recent_responses=list(recent_responses),
            wrong_answer_streak=wrong_streak,
            hint_used_last_step=hint_used_last,
            step_count=step,
        )

        # Check concept progression
        if hasattr(teacher, "should_advance_concept") and teacher.should_advance_concept(obs):
            concept_idx = (concept_idx + 1) % len(concept_ids)
            current_concept_id = concept_ids[concept_idx]
            concept = curriculum.get_concept(current_concept_id)
            # Re-evaluate weakest prerequisite for new concept
            prereqs = concept.prerequisites
            weakest_prereq_id = None
            weakest_prereq_m = None
            if prereqs:
                prereq_masteries = [(p, model.get_mastery(p)) for p in prereqs]
                weakest_prereq_id, weakest_prereq_m = min(prereq_masteries, key=lambda x: x[1])

            obs = TeacherObservation(
                current_concept_id=current_concept_id,
                estimated_mastery=model.get_mastery(current_concept_id),
                all_estimated_masteries=model.get_all_masteries(),
                prerequisite_ids=prereqs,
                weakest_prereq_id=weakest_prereq_id,
                weakest_prereq_mastery=weakest_prereq_m,
                last_response_correct=last_response_correct,
                recent_responses=list(recent_responses),
                wrong_answer_streak=wrong_streak,
                hint_used_last_step=hint_used_last,
                step_count=step,
            )

        # 2. Teacher selects action
        action = teacher.select_action(obs)
        action_history.append(action)

        # Determine target concept and simulation parameters
        if action == PedagogicalAction.REVIEW_PREREQUISITE and weakest_prereq_id is not None:
            active_cid = weakest_prereq_id
            active_diff = float(curriculum.get_concept(weakest_prereq_id).difficulty)
            sim_cid = weakest_prereq_id
            sim_prereq = None
        else:
            active_cid = current_concept_id
            active_diff = float(concept.difficulty)
            sim_cid = current_concept_id
            sim_prereq = weakest_prereq_id

        engaged_concepts.add(active_cid)
        diff_val = max(1.0, min(5.0, active_diff + difficulty_offset))

        # 3. Step simulator (updates true continuous mastery M)
        sim_res = step_simulation(
            learner=learner,
            action=action,
            concept_id=sim_cid,
            question_difficulty=diff_val,
            prerequisite_id=sim_prereq,
            action_effect_multiplier=action_effect_multiplier,
        )

        # 4. Update observable learner model (strictly from response correctness)
        model.update(active_cid, correct=sim_res.correct)
        model.step_decay_unpracticed(active_cid, only_engaged=True)

        # 5. Update history features
        last_response_correct = sim_res.correct
        recent_responses.append(sim_res.correct)
        if len(recent_responses) > 3:
            recent_responses.pop(0)

        if sim_res.correct:
            wrong_streak = 0
        else:
            wrong_streak += 1

        hint_used_last = (action == PedagogicalAction.HINT)
        obs_mastery_history.append(model.get_all_masteries())

    # Final metrics evaluation
    final_obs_m = model.get_all_masteries()
    final_true_m = learner.get_all_true_masteries()

    metrics = evaluate_episode(
        initial_mastery=initial_obs_m,
        final_mastery=final_obs_m,
        actions=action_history,
        mastery_history=obs_mastery_history,
        mastery_threshold=0.75,
        max_steps=max_steps,
        engaged_concepts=engaged_concepts,
    )

    engaged_list = list(engaged_concepts) if engaged_concepts else concept_ids
    true_gain = float(np.mean([final_true_m[c] - initial_true_m[c] for c in engaged_list]))
    return metrics, true_gain


def evaluate_teachers(
    num_episodes: int = 50,
    seed_start: int = 1000,
    max_steps: int = 30,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Run comparative evaluation of Random, Fixed, and Rule teachers across identical learners."""
    curriculum = load_curriculum("data/curriculum")
    concept_ids = curriculum.get_all_concept_ids()

    teachers = [
        RandomTeacher(seed=seed_start),
        FixedTeacher(),
        RuleTeacher(mastery_threshold=0.75, review_interval=5, max_steps_per_concept=6),
    ]

    all_results: dict[str, list[EpisodeMetrics]] = {t.name: [] for t in teachers}
    true_gains: dict[str, list[float]] = {t.name: [] for t in teachers}

    for ep in range(num_episodes):
        learner_seed = seed_start + ep

        for teacher in teachers:
            # Each teacher faces an identical clone of the simulated learner
            learner = sample_learner(
                seed=learner_seed,
                concepts=concept_ids,
                initial_mastery=0.10,
            )

            ep_metrics, true_gain = run_tutoring_episode(
                teacher=teacher,
                learner=learner,
                curriculum=curriculum,
                max_steps=max_steps,
                initial_mastery=0.10,
            )
            all_results[teacher.name].append(ep_metrics)
            true_gains[teacher.name].append(true_gain)

    # Compute aggregates
    aggregated: dict[str, dict[str, tuple[float, float]]] = {}
    for name, m_list in all_results.items():
        agg = aggregate_metrics(m_list)
        agg["true_learning_gain"] = (float(np.mean(true_gains[name])), float(np.std(true_gains[name])))
        aggregated[name] = agg

    return aggregated


def print_baseline_table(aggregated: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Print clean comparative metrics table across baseline teachers."""
    metric_labels = [
        ("Learning Gain (Obs)", "learning_gain", "{:.3f} +/- {:.3f}"),
        ("True Learning Gain", "true_learning_gain", "{:.3f} +/- {:.3f}"),
        ("Final Mastery", "final_mastery", "{:.3f} +/- {:.3f}"),
        ("Steps to Mastery", "steps_to_mastery", "{:.1f} +/- {:.1f}"),
        ("Efficiency (Gain/Step)", "intervention_efficiency", "{:.4f} +/- {:.4f}"),
        ("Hint Usage Rate", "hint_usage_rate", "{:.3f} +/- {:.3f}"),
        ("Action Entropy (bits)", "action_entropy", "{:.3f} +/- {:.3f}"),
    ]

    teachers = list(aggregated.keys())
    col_w = 26

    print("\n" + "=" * 90)
    print(" AirTutor Stage 2: Baseline Teacher Evaluation (50 Episodes Each)")
    print("=" * 90)
    header = f"{'Metric':25} | " + " | ".join(f"{t:^{col_w}}" for t in teachers)
    print(header)
    print("-" * 90)

    for label, key, fmt in metric_labels:
        row = f"{label:25} | "
        cols = []
        for t in teachers:
            mean, std = aggregated[t][key]
            cols.append(f"{fmt.format(mean, std):^{col_w}}")
        row += " | ".join(cols)
        print(row)

    print("=" * 90)

    # Sanity verification
    rule_gain = aggregated["Rule"]["learning_gain"][0]
    random_gain = aggregated["Random"]["learning_gain"][0]
    fixed_gain = aggregated["Fixed"]["learning_gain"][0]

    print("\n[Verification Check]")
    print(f"  Rule Learning Gain ({rule_gain:.3f}) vs. Random ({random_gain:.3f}) -> Delta: {rule_gain - random_gain:+.3f}")
    print(f"  Rule Learning Gain ({rule_gain:.3f}) vs. Fixed  ({fixed_gain:.3f}) -> Delta: {rule_gain - fixed_gain:+.3f}")

    if rule_gain > random_gain and rule_gain > fixed_gain:
        print("  -> PASSED: Rule teacher visibly outperforms Random and Fixed teachers on learning gain.")
    else:
        print("  -> FAILED: Rule teacher did not outperform baselines. Tuning needed.")


def main() -> None:
    results = evaluate_teachers(num_episodes=50, seed_start=1000, max_steps=30)
    print_baseline_table(results)


if __name__ == "__main__":
    main()
