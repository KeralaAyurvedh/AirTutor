"""Ablation Studies for AirTutor (Roadmap Sec. 15 & 22).

Executes formal ablations across three core architectural dimensions:
1. Reward Formulation Ablation:
   - v1: r_t = M_hat(c, t+1) - M_hat(c, t)
   - v2: v1 + lambda_ret * retention - lambda_cost * intervention_cost
   - v3: v2 + transfer - excessive_hint_penalty - wasted_step_penalty
   Measures metric deltas AND qualitative action preference shifts.
2. State Representation Ablation:
   - Full state: 320 states (mastery, last_correct, recent_accuracy, wrong_streak, hint_used)
   - No-Hint: 160 states (use_hint_used=False)
   - No-Recent-Accuracy: 80 states (use_recent_accuracy=False)
   Identifies which observable state features are load-bearing.
3. Learner Population Diversity Ablation:
   - Wide bounds (standard default diverse population)
   - Narrow bounds (tight +/- 10% parameter interval around midpoints)
   Tests whether training on narrow distributions breaks held-out generalization on wide distributions.
"""

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from src.environment.actions import ACTIONS, NUM_ACTIONS, PedagogicalAction
from src.environment.curriculum import CurriculumRepository, load_curriculum
from src.environment.state_encoder import StateEncoderConfig
from src.evaluation.baseline_comparison import run_tutoring_episode
from src.evaluation.compare_teachers import compute_confidence_intervals_95, evaluate_agent_on_held_out
from src.evaluation.metrics import EpisodeMetrics
from src.rl.q_learning import TabularQAgent
from src.rl.train import train_agent_single_seed
from src.simulator.population import DEFAULT_PARAM_RANGES, sample_learner

# Tight parameter bounds (~ +/- 10% around default distribution midpoints)
NARROW_PARAM_RANGES = {
    "alpha": (0.18, 0.22),
    "beta": (0.07, 0.09),
    "gamma": (1.10, 1.40),
    "h": (0.45, 0.55),
    "tau": (0.55, 0.65),
    "sigma": (0.05, 0.07),
}


def run_reward_ablation(
    rl_seeds: Sequence[int] = (42, 101, 2024),
    n_train: int = 300,
    n_eval: int = 100,
    train_seed_start: int = 0,
    eval_seed_start: int = 10000,
    curriculum: CurriculumRepository | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare RL agents trained under Reward v1, v2, and v3."""
    curr = curriculum or load_curriculum("data/curriculum")
    concept_ids = curr.get_all_concept_ids()
    learner_seeds = list(range(train_seed_start, train_seed_start + n_train))
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))

    versions = ["v1", "v2", "v3"]
    results: dict[str, Any] = {}

    for version in versions:
        if verbose:
            print(f"\n--- [Reward Ablation: {version.upper()}] Training across {len(rl_seeds)} seeds ---")

        agents: list[TabularQAgent] = []
        for seed in rl_seeds:
            agent, _ = train_agent_single_seed(
                rl_seed=seed,
                learner_seeds=learner_seeds,
                curriculum=curr,
                num_episodes=n_train,
                reward_version=version,
            )
            agents.append(agent)

        # Evaluate greedily on held-out learners and track action counts
        all_metrics: list[EpisodeMetrics] = []
        all_true_gains: list[float] = []
        action_counts = {a.name: 0 for a in PedagogicalAction}
        total_actions = 0

        for agent in agents:
            # Temporarily force epsilon to 0 for pure greedy evaluation
            saved_eps = agent.epsilon
            agent.epsilon = 0.0

            for eseed in eval_seeds:
                learner = sample_learner(seed=eseed, concepts=concept_ids, initial_mastery=0.10)
                ep_metrics, true_gain = run_tutoring_episode(
                    teacher=agent,
                    learner=learner,
                    curriculum=curr,
                    max_steps=30,
                    initial_mastery=0.10,
                )
                all_metrics.append(ep_metrics)
                all_true_gains.append(true_gain)

                # Record actions taken in greedy evaluation
                for act_name, count in ep_metrics.action_counts.items():
                    action_counts[act_name] = action_counts.get(act_name, 0) + count
                    total_actions += count

            agent.epsilon = saved_eps

        # Statistical summary
        stats = compute_confidence_intervals_95(all_metrics)
        tg_mean = float(np.mean(all_true_gains))
        tg_std = float(np.std(all_true_gains))
        tg_ci95 = float(1.96 * (tg_std / np.sqrt(len(all_true_gains))))
        stats["true_learning_gain"] = (tg_mean, tg_std, tg_ci95)

        # Action preference distribution (proportions)
        action_distribution = {
            a: float(count / total_actions) if total_actions > 0 else 0.0
            for a, count in action_counts.items()
        }
        stats["action_distribution"] = action_distribution
        results[version] = stats

        if verbose:
            obs_m = stats["learning_gain"][0]
            true_m = stats["true_learning_gain"][0]
            print(f"   Reward {version.upper()} -> Obs Gain: {obs_m:+.3f} | True Gain: {true_m:+.3f}")
            print(f"   Action Distribution: " + ", ".join(f"{a}: {p*100:.1f}%" for a, p in action_distribution.items()))

    return results


def run_state_ablation(
    rl_seeds: Sequence[int] = (42, 101, 2024),
    n_train: int = 300,
    n_eval: int = 100,
    train_seed_start: int = 0,
    eval_seed_start: int = 10000,
    reward_version: str = "v1",
    curriculum: CurriculumRepository | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare Full state (320) vs No-Hint (160) vs No-Accuracy (80)."""
    curr = curriculum or load_curriculum("data/curriculum")
    concept_ids = curr.get_all_concept_ids()
    learner_seeds = list(range(train_seed_start, train_seed_start + n_train))
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))

    configs = {
        "Full State (320)": StateEncoderConfig(use_hint_used=True, use_recent_accuracy=True),
        "No-Hint (160)": StateEncoderConfig(use_hint_used=False, use_recent_accuracy=True),
        "No-Accuracy (80)": StateEncoderConfig(use_hint_used=True, use_recent_accuracy=False),
    }

    results: dict[str, Any] = {}

    for name, s_cfg in configs.items():
        if verbose:
            print(f"\n--- [State Ablation: {name}] Training across {len(rl_seeds)} seeds ---")

        agents: list[TabularQAgent] = []
        for seed in rl_seeds:
            agent, _ = train_agent_single_seed(
                rl_seed=seed,
                learner_seeds=learner_seeds,
                curriculum=curr,
                num_episodes=n_train,
                reward_version=reward_version,
                state_config=s_cfg,
            )
            agents.append(agent)

        all_metrics: list[EpisodeMetrics] = []
        all_true_gains: list[float] = []

        for agent in agents:
            ag_m, ag_tg = evaluate_agent_on_held_out(
                agent=agent,
                eval_seeds=eval_seeds,
                curriculum=curr,
                max_steps=30,
            )
            all_metrics.extend(ag_m)
            all_true_gains.extend(ag_tg)

        stats = compute_confidence_intervals_95(all_metrics)
        tg_mean = float(np.mean(all_true_gains))
        tg_std = float(np.std(all_true_gains))
        tg_ci95 = float(1.96 * (tg_std / np.sqrt(len(all_true_gains))))
        stats["true_learning_gain"] = (tg_mean, tg_std, tg_ci95)
        results[name] = stats

        if verbose:
            obs_m = stats["learning_gain"][0]
            true_m = stats["true_learning_gain"][0]
            eff = stats["intervention_efficiency"][0]
            print(f"   {name} -> Obs Gain: {obs_m:+.3f} | True Gain: {true_m:+.3f} | Efficiency: {eff:.4f}")

    return results


def run_population_diversity_ablation(
    rl_seeds: Sequence[int] = (42, 101, 2024),
    n_train: int = 300,
    n_eval: int = 100,
    train_seed_start: int = 0,
    eval_seed_start: int = 10000,
    reward_version: str = "v1",
    curriculum: CurriculumRepository | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare policy trained on Wide bounds vs Narrow bounds when tested on Wide held-out population."""
    curr = curriculum or load_curriculum("data/curriculum")
    concept_ids = curr.get_all_concept_ids()
    learner_seeds = list(range(train_seed_start, train_seed_start + n_train))
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))

    conditions = {
        "Wide Population (Default)": DEFAULT_PARAM_RANGES,
        "Narrow Population (+/-10%)": NARROW_PARAM_RANGES,
    }

    results: dict[str, Any] = {}

    for name, ranges in conditions.items():
        if verbose:
            print(f"\n--- [Population Diversity Ablation: {name}] Training across {len(rl_seeds)} seeds ---")

        agents: list[TabularQAgent] = []
        for seed in rl_seeds:
            # Train on designated population
            agent = TabularQAgent(seed=seed, name=f"Agent_{seed}")
            # Train episode by episode with specified param_ranges
            for ep in range(n_train):
                agent.reset()
                l_seed = learner_seeds[ep % len(learner_seeds)]
                learner = sample_learner(
                    seed=l_seed,
                    concepts=concept_ids,
                    initial_mastery=0.10,
                    param_ranges=ranges,
                )
                from src.learner_model.mastery import LearnerModel
                from src.simulator.dynamics import step_simulation
                from src.environment.rewards import compute_reward
                from src.teachers.base_teacher import TeacherObservation

                model = LearnerModel(concepts=concept_ids, initial_mastery=0.10)
                concept_idx = 0
                current_concept_id = concept_ids[concept_idx]
                last_correct = None
                recent_responses: list[bool] = []
                wrong_streak = 0
                hint_used_last = False
                ep_actions: list[PedagogicalAction] = []

                for step in range(1, 31):
                    concept = curr.get_concept(current_concept_id)
                    prereqs = concept.prerequisites
                    weakest_prereq_id, weakest_prereq_m = None, None
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
                        last_response_correct=last_correct,
                        recent_responses=list(recent_responses),
                        wrong_answer_streak=wrong_streak,
                        hint_used_last_step=hint_used_last,
                        step_count=step,
                    )

                    if agent.should_advance_concept(obs):
                        concept_idx = (concept_idx + 1) % len(concept_ids)
                        current_concept_id = concept_ids[concept_idx]
                        concept = curr.get_concept(current_concept_id)
                        prereqs = concept.prerequisites
                        weakest_prereq_id, weakest_prereq_m = None, None
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
                            last_response_correct=last_correct,
                            recent_responses=list(recent_responses),
                            wrong_answer_streak=wrong_streak,
                            hint_used_last_step=hint_used_last,
                            step_count=step,
                        )

                    state_idx = agent.get_state_index(obs)
                    action = agent.select_action(obs, greedy=False)
                    action_idx = action.to_index()
                    ep_actions.append(action)
                    recent_act_window = ep_actions[-6:]
                    rolling_hint_rate = float(sum(1 for a in recent_act_window if a == PedagogicalAction.HINT) / len(recent_act_window))

                    prev_m_hat = model.get_mastery(current_concept_id)
                    prev_all_m = model.get_all_masteries()

                    sim_res = step_simulation(
                        learner=learner,
                        action=action,
                        concept_id=current_concept_id,
                        question_difficulty=concept.difficulty,
                        prerequisite_id=weakest_prereq_id,
                    )
                    model.update(current_concept_id, correct=sim_res.correct)
                    model.step_decay_unpracticed(current_concept_id, only_engaged=True)
                    new_m_hat = model.get_mastery(current_concept_id)
                    new_all_m = model.get_all_masteries()

                    reward_t = compute_reward(
                        reward_version=reward_version,
                        prev_mastery=prev_m_hat,
                        new_mastery=new_m_hat,
                        prev_all_masteries=prev_all_m,
                        new_all_masteries=new_all_m,
                        current_concept_id=current_concept_id,
                        action=action,
                        hint_used_last_step=hint_used_last,
                        rolling_hint_rate=rolling_hint_rate,
                        is_correct=sim_res.correct,
                    )

                    last_correct = sim_res.correct
                    recent_responses.append(sim_res.correct)
                    if len(recent_responses) > 3:
                        recent_responses.pop(0)
                    wrong_streak = 0 if sim_res.correct else wrong_streak + 1
                    hint_used_last = (action == PedagogicalAction.HINT)

                    next_obs = TeacherObservation(
                        current_concept_id=current_concept_id,
                        estimated_mastery=new_m_hat,
                        all_estimated_masteries=model.get_all_masteries(),
                        prerequisite_ids=prereqs,
                        weakest_prereq_id=weakest_prereq_id,
                        weakest_prereq_mastery=weakest_prereq_m,
                        last_response_correct=last_correct,
                        recent_responses=list(recent_responses),
                        wrong_answer_streak=wrong_streak,
                        hint_used_last_step=hint_used_last,
                        step_count=step + 1,
                    )
                    next_state_idx = agent.get_state_index(next_obs)
                    done = (step == 30)
                    agent.update(state_idx, action_idx, reward_t, next_state_idx, done)

                agent.decay_epsilon()

            agents.append(agent)

        # Evaluate both on the FULL WIDE held-out population (param_ranges=None uses DEFAULT_PARAM_RANGES)
        all_metrics = []
        all_true_gains = []
        for agent in agents:
            ag_m, ag_tg = evaluate_agent_on_held_out(
                agent=agent,
                eval_seeds=eval_seeds,
                curriculum=curr,
                max_steps=30,
                param_ranges=None,  # Always evaluate on full wide distribution!
            )
            all_metrics.extend(ag_m)
            all_true_gains.extend(ag_tg)

        stats = compute_confidence_intervals_95(all_metrics)
        tg_mean = float(np.mean(all_true_gains))
        tg_std = float(np.std(all_true_gains))
        tg_ci95 = float(1.96 * (tg_std / np.sqrt(len(all_true_gains))))
        stats["true_learning_gain"] = (tg_mean, tg_std, tg_ci95)
        results[name] = stats

        if verbose:
            obs_m = stats["learning_gain"][0]
            true_m = stats["true_learning_gain"][0]
            print(f"   {name} evaluated on Wide Held-Out -> Obs Gain: {obs_m:+.3f} | True Gain: {true_m:+.3f}")

    return results
