"""RL Training Pipeline for AirTutor (Roadmap Sec. 17).

Trains tabular Q-learning agents against the parametric learner simulator
across multiple independent random seeds to report distributions rather than single runs.

Training protocol:
- Training population sampled from seed range [0, N_train) with N_train ~ 300.
- Evaluates across 5 independent RL training seeds.
- Reward v1: r_t = M_hat(c, t+1) - M_hat(c, t).
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
from src.environment.rewards import compute_reward_v1
from src.learner_model.mastery import LearnerModel
from src.rl.q_learning import (
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN,
    DEFAULT_EPSILON_START,
    DEFAULT_LEARNING_RATE,
    TabularQAgent,
)
from src.simulator.dynamics import step_simulation
from src.simulator.population import sample_learner
from src.teachers.base_teacher import TeacherObservation


def train_agent_single_seed(
    rl_seed: int,
    learner_seeds: Sequence[int],
    curriculum: CurriculumRepository,
    num_episodes: int = 300,
    max_steps_per_episode: int = 30,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
    epsilon_start: float = DEFAULT_EPSILON_START,
    epsilon_min: float = DEFAULT_EPSILON_MIN,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    initial_mastery: float = 0.10,
) -> tuple[TabularQAgent, list[float]]:
    """Train a single TabularQAgent instance across training episodes."""
    agent = TabularQAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=rl_seed,
        name=f"RL_Seed_{rl_seed}",
    )

    concept_ids = curriculum.get_all_concept_ids()
    episode_rewards: list[float] = []

    for ep in range(num_episodes):
        agent.reset()
        # Draw a simulated student from the designated training seed population
        learner_seed = learner_seeds[ep % len(learner_seeds)]
        learner = sample_learner(
            seed=learner_seed,
            concepts=concept_ids,
            initial_mastery=initial_mastery,
        )

        model = LearnerModel(concepts=concept_ids, initial_mastery=initial_mastery)

        concept_idx = 0
        current_concept_id = concept_ids[concept_idx]

        last_correct: bool | None = None
        recent_responses: list[bool] = []
        wrong_streak: int = 0
        hint_used_last: bool = False

        total_ep_reward = 0.0

        for step in range(1, max_steps_per_episode + 1):
            concept = curriculum.get_concept(current_concept_id)
            prereqs = concept.prerequisites
            weakest_prereq_id = None
            weakest_prereq_m = None
            if prereqs:
                prereq_masteries = [(p, model.get_mastery(p)) for p in prereqs]
                weakest_prereq_id, weakest_prereq_m = min(prereq_masteries, key=lambda x: x[1])

            # 1. Construct observable observation
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

            # Check concept progression
            if agent.should_advance_concept(obs):
                concept_idx = (concept_idx + 1) % len(concept_ids)
                current_concept_id = concept_ids[concept_idx]
                concept = curriculum.get_concept(current_concept_id)
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
                    last_response_correct=last_correct,
                    recent_responses=list(recent_responses),
                    wrong_answer_streak=wrong_streak,
                    hint_used_last_step=hint_used_last,
                    step_count=step,
                )

            state_idx = agent.get_state_index(obs)

            # 2. Select action via epsilon-greedy policy
            action = agent.select_action(obs, greedy=False)
            action_idx = action.to_index()

            # Record pre-intervention estimated mastery on concept
            prev_m_hat = model.get_mastery(current_concept_id)

            # 3. Step simulator (updates true continuous mastery M)
            sim_res = step_simulation(
                learner=learner,
                action=action,
                concept_id=current_concept_id,
                question_difficulty=concept.difficulty,
                prerequisite_id=weakest_prereq_id,
            )

            # 4. Update observable learner model
            model.update(current_concept_id, correct=sim_res.correct)
            model.step_decay_unpracticed(current_concept_id)

            # Post-intervention estimated mastery on concept
            new_m_hat = model.get_mastery(current_concept_id)

            # 5. Compute Reward v1: r_t = M_hat(c, t+1) - M_hat(c, t)
            reward_t = compute_reward_v1(prev_m_hat, new_m_hat)
            total_ep_reward += reward_t

            # 6. Update observable history for next state
            last_correct = sim_res.correct
            recent_responses.append(sim_res.correct)
            if len(recent_responses) > 3:
                recent_responses.pop(0)

            if sim_res.correct:
                wrong_streak = 0
            else:
                wrong_streak += 1
            hint_used_last = (action == PedagogicalAction.HINT)

            # Construct next observation
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
            done = (step == max_steps_per_episode)

            # 7. Perform Bellman Q-table update
            agent.update(state_idx, action_idx, reward_t, next_state_idx, done)

        agent.decay_epsilon()
        episode_rewards.append(total_ep_reward)

    return agent, episode_rewards


def train_multi_seed(
    rl_seeds: Sequence[int] = (42, 101, 2024, 777, 999),
    num_train_learners: int = 300,
    train_seed_start: int = 0,
    num_episodes: int = 300,
    curriculum: CurriculumRepository | None = None,
    verbose: bool = True,
) -> tuple[list[TabularQAgent], dict[int, list[float]]]:
    """Train agents across multiple independent RL seeds."""
    curr = curriculum or load_curriculum("data/curriculum")
    learner_seeds = list(range(train_seed_start, train_seed_start + num_train_learners))

    agents: list[TabularQAgent] = []
    reward_trajectories: dict[int, list[float]] = {}

    for idx, seed in enumerate(rl_seeds, start=1):
        if verbose:
            print(f"-> Training RL agent {idx}/{len(rl_seeds)} (seed={seed}) across {num_episodes} episodes...")

        agent, ep_rewards = train_agent_single_seed(
            rl_seed=seed,
            learner_seeds=learner_seeds,
            curriculum=curr,
            num_episodes=num_episodes,
        )
        agents.append(agent)
        reward_trajectories[seed] = ep_rewards

        if verbose:
            recent_reward = np.mean(ep_rewards[-50:])
            print(f"   Done. Mean reward (last 50 episodes): {recent_reward:+.3f} | Epsilon: {agent.epsilon:.3f}")

    return agents, reward_trajectories
