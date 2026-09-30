"""Deep Q-Network (DQN) Comparative Benchmark and Ablation Suite.

Executes:
1. Multi-seed training (seeds 42, 123, 456) across heterogeneous learner archetypes
   (FastLearner, SlowLearner, InattentiveLearner).
2. Comparative baseline evaluation against RuleTeacher, FixedTeacher, and RandomTeacher
   across 100 evaluation episodes per learner archetype (disjoint evaluation seeds).
3. Ablation studies comparing:
   - Condition A: Full DQN + BKT Belief State (14-D)
   - Condition B: DQN without BKT (raw accuracy history replacing BKT beliefs)
   - Condition C: DQN without Action History (zeroed action indices [8-12])
4. Artifact generation:
   - experiments/dqn_benchmark/metrics.csv
   - experiments/dqn_benchmark/learning_gain_comparison.png
   - experiments/dqn_benchmark/training_reward_loss.png
   - experiments/dqn_benchmark/ablation_study_deltas.png
"""

import os
from pathlib import Path
import random
import time
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.environment.actions import ACTIONS, NUM_ACTIONS, PedagogicalAction
from src.environment.curriculum import Concept, CurriculumRepository, load_curriculum
from src.environment.state_encoder import (
    CognitiveStateEncoder,
    encode_cognitive_state,
)
from src.learner_model.bkt_tracker import BKTTracker
from src.rl.dqn_agent import DQNAgent
from src.simulator.dynamics import simulate_tutoring_step
from src.simulator.population import (
    FastLearner,
    InattentiveLearner,
    SimulatedLearner,
    SlowLearner,
    sample_archetype_learner,
)
from src.teachers.base_teacher import BaseTeacher, TeacherObservation
from src.teachers.fixed_teacher import FixedTeacher
from src.teachers.random_teacher import RandomTeacher
from src.teachers.rule_teacher import RuleTeacher

EXPERIMENT_DIR = Path("experiments/dqn_benchmark")
TRAIN_SEEDS = [42, 123, 456]
NUM_TRAIN_EPISODES = 800
NUM_EVAL_EPISODES_PER_TYPE = 100
MAX_SESSION_STEPS = 25
MASTERY_THRESHOLD = 0.85


def build_ablation_state(
    condition: str,
    bkt_tracker: BKTTracker,
    raw_accuracies: dict[str, float],
    current_concept_id: str,
    prerequisite_concept_id: str | None,
    all_concept_ids: list[str],
    consecutive_failures: int,
    session_steps: int,
    previous_action: Any,
    last_response_correct: bool | None,
    encoder: CognitiveStateEncoder,
) -> np.ndarray:
    """Construct state vector conforming to Condition A, B, or C."""
    if condition == "Condition_A_Full":
        # Full 14-D state
        return encoder.encode(
            bkt_tracker=bkt_tracker,
            current_concept_id=current_concept_id,
            prerequisite_concept_id=prerequisite_concept_id,
            all_concept_ids=all_concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=session_steps,
            previous_action=previous_action,
            last_response_correct=last_response_correct,
        )

    elif condition == "Condition_B_No_BKT":
        # Replace BKT beliefs with raw observable concept accuracy history
        return encoder.encode(
            bkt_tracker=raw_accuracies,
            current_concept_id=current_concept_id,
            prerequisite_concept_id=prerequisite_concept_id,
            all_concept_ids=all_concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=session_steps,
            previous_action=previous_action,
            last_response_correct=last_response_correct,
        )

    elif condition == "Condition_C_No_Action_History":
        # Zero out previous action one-hot indices [8-12]
        s = encoder.encode(
            bkt_tracker=bkt_tracker,
            current_concept_id=current_concept_id,
            prerequisite_concept_id=prerequisite_concept_id,
            all_concept_ids=all_concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=session_steps,
            previous_action=None,  # Zeros out action indices [8-12]
            last_response_correct=last_response_correct,
        )
        return s

    raise ValueError(f"Unknown condition: {condition}")


def run_episode(
    policy_or_teacher: Any,
    learner: SimulatedLearner,
    curriculum: CurriculumRepository,
    condition: str = "Condition_A_Full",
    encoder: CognitiveStateEncoder | None = None,
    max_steps: int = MAX_SESSION_STEPS,
    is_training: bool = False,
) -> dict[str, Any]:
    """Execute a single tutoring episode for a learner using a designated policy."""
    encoder = encoder or CognitiveStateEncoder(max_session_steps=max_steps)
    concept_ids = curriculum.get_all_concept_ids()
    bkt_tracker = BKTTracker(concepts=concept_ids)

    # Tracking for Condition B (raw accuracy history)
    concept_attempts: dict[str, int] = {c: 0 for c in concept_ids}
    concept_corrects: dict[str, int] = {c: 0 for c in concept_ids}
    raw_accuracies: dict[str, float] = {c: 0.0 for c in concept_ids}

    # Record initial ground-truth mastery
    pre_score = float(np.mean([learner.get_true_mastery(c) for c in concept_ids]))

    # Reset policy/teacher state if applicable
    if hasattr(policy_or_teacher, "reset"):
        policy_or_teacher.reset()

    concept_idx = 0
    steps_on_concept = 0
    consecutive_failures = 0
    recent_responses: list[bool] = []
    previous_action: Any = None
    last_response_correct: bool | None = None
    cumulative_reward = 0.0
    steps_to_mastery = max_steps
    episode_losses: list[float] = []

    for step in range(max_steps):
        curr_concept_id = concept_ids[concept_idx]
        curr_concept = curriculum.get_concept(curr_concept_id)
        prereq_ids = curr_concept.prerequisites
        prereq_id = prereq_ids[0] if prereq_ids else None

        # Build current state vector
        state = build_ablation_state(
            condition=condition,
            bkt_tracker=bkt_tracker,
            raw_accuracies=raw_accuracies,
            current_concept_id=curr_concept_id,
            prerequisite_concept_id=prereq_id,
            all_concept_ids=concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=step,
            previous_action=previous_action,
            last_response_correct=last_response_correct,
            encoder=encoder,
        )

        # Select action
        if isinstance(policy_or_teacher, DQNAgent):
            action_idx = policy_or_teacher.select_action(state, greedy=not is_training)
            action = ACTIONS[action_idx]
        elif isinstance(policy_or_teacher, BaseTeacher):
            # Construct TeacherObservation for baseline teachers
            obs = TeacherObservation(
                current_concept_id=curr_concept_id,
                estimated_mastery=float(bkt_tracker.get_belief(curr_concept_id)),
                all_estimated_masteries=bkt_tracker.get_all_beliefs(),
                prerequisite_ids=prereq_ids,
                weakest_prereq_id=prereq_id,
                weakest_prereq_mastery=float(bkt_tracker.get_belief(prereq_id)) if prereq_id else 1.0,
                last_response_correct=last_response_correct,
                recent_responses=recent_responses[-3:],
                wrong_answer_streak=consecutive_failures,
                hint_used_last_step=(previous_action == PedagogicalAction.HINT),
                step_count=step,
            )
            action = policy_or_teacher.select_action(obs)
            action_idx = action.to_index()
        else:
            action_idx = 0
            action = ACTIONS[0]

        # Simulator step
        sim_res = simulate_tutoring_step(
            learner=learner,
            action=action,
            concept_id=curr_concept_id,
            question_difficulty=curr_concept.difficulty,
            prerequisite_id=prereq_id,
        )

        is_correct = sim_res.correct
        prev_belief = float(bkt_tracker.get_belief(curr_concept_id))
        new_belief = bkt_tracker.update(curr_concept_id, is_correct=is_correct)

        # Update observable history
        concept_attempts[curr_concept_id] += 1
        if is_correct:
            concept_corrects[curr_concept_id] += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        raw_accuracies[curr_concept_id] = (
            concept_corrects[curr_concept_id] / max(1, concept_attempts[curr_concept_id])
        )

        recent_responses.append(is_correct)
        last_response_correct = is_correct
        previous_action = action
        steps_on_concept += 1

        # Reward computation: belief gain - step cost - hint penalty + transfer
        gain = new_belief - prev_belief
        step_cost = 0.015
        hint_penalty = 0.04 if (action == PedagogicalAction.HINT) else 0.0
        prereq_bonus = 0.10 * sim_res.prerequisite_gain
        reward = float(gain - step_cost - hint_penalty + prereq_bonus)
        cumulative_reward += reward

        # Check curriculum mastery across all concepts
        all_mastered = all(bkt_tracker.get_belief(c) >= MASTERY_THRESHOLD for c in concept_ids)
        if all_mastered and steps_to_mastery == max_steps:
            steps_to_mastery = step + 1

        done = (step == max_steps - 1) or all_mastered

        # Next state
        next_state = build_ablation_state(
            condition=condition,
            bkt_tracker=bkt_tracker,
            raw_accuracies=raw_accuracies,
            current_concept_id=curr_concept_id,
            prerequisite_concept_id=prereq_id,
            all_concept_ids=concept_ids,
            consecutive_failures=consecutive_failures,
            session_steps=step + 1,
            previous_action=action,
            last_response_correct=last_response_correct,
            encoder=encoder,
        )

        # Store transition and train if DQN (update every 4 steps per Mnih et al. 2015)
        if is_training and isinstance(policy_or_teacher, DQNAgent):
            policy_or_teacher.store_transition(state, action_idx, reward, next_state, done)
            if step % 4 == 0 or done:
                loss = policy_or_teacher.train_step()
                if loss is not None:
                    episode_losses.append(loss)

        if done:
            break

        # Concept progression logic
        concept_mastered = new_belief >= MASTERY_THRESHOLD
        concept_timeout = steps_on_concept >= 6
        if (concept_mastered or concept_timeout) and concept_idx < len(concept_ids) - 1:
            concept_idx += 1
            steps_on_concept = 0

    if is_training and isinstance(policy_or_teacher, DQNAgent):
        policy_or_teacher.decay_epsilon()

    # Final post-test ground-truth score
    post_score = float(np.mean([learner.get_true_mastery(c) for c in concept_ids]))
    if (1.0 - pre_score) > 1e-6:
        learning_gain = float(max(0.0, min(1.0, (post_score - pre_score) / (1.0 - pre_score))))
    else:
        learning_gain = 0.0

    mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

    return {
        "pre_score": pre_score,
        "post_score": post_score,
        "learning_gain": learning_gain,
        "steps_to_mastery": steps_to_mastery,
        "cumulative_reward": cumulative_reward,
        "mean_loss": mean_loss,
    }


def train_dqn_agent(
    seed: int,
    curriculum: CurriculumRepository,
    condition: str = "Condition_A_Full",
    num_episodes: int = NUM_TRAIN_EPISODES,
) -> tuple[DQNAgent, list[float], list[float]]:
    """Train a DQN agent over heterogeneous learner archetypes."""
    print(f"--> Training DQN ({condition}) [Seed {seed}] for {num_episodes} episodes...", flush=True)
    agent = DQNAgent(state_dim=14, num_actions=5, seed=seed)
    encoder = CognitiveStateEncoder(max_session_steps=MAX_SESSION_STEPS)
    concept_ids = curriculum.get_all_concept_ids()

    archetype_names = ["FastLearner", "SlowLearner", "InattentiveLearner"]
    episode_rewards: list[float] = []
    episode_losses: list[float] = []

    for ep in range(num_episodes):
        arch = archetype_names[ep % len(archetype_names)]
        ep_seed = seed * 10000 + ep
        learner = sample_archetype_learner(arch, seed=ep_seed, concepts=concept_ids)

        res = run_episode(
            policy_or_teacher=agent,
            learner=learner,
            curriculum=curriculum,
            condition=condition,
            encoder=encoder,
            is_training=True,
        )

        episode_rewards.append(res["cumulative_reward"])
        if res["mean_loss"] > 0:
            episode_losses.append(res["mean_loss"])

        if (ep + 1) % 200 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            print(f"    Episode {ep + 1}/{num_episodes} | Avg Reward (last 50): {avg_r:.3f} | Epsilon: {agent.epsilon:.3f}", flush=True)

    return agent, episode_rewards, episode_losses


def run_benchmark():
    """Main benchmark and ablation orchestrator."""
    start_time = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    curriculum = load_curriculum("data/curriculum")
    concept_ids = curriculum.get_all_concept_ids()

    records: list[dict[str, Any]] = []
    training_curves: dict[str, dict[str, list[float]]] = {}

    # =========================================================================
    # Step 1: Multi-Seed Training (Seeds 42, 123, 456 for Condition A; 400 eps for ablations)
    # =========================================================================
    trained_agents: dict[str, dict[int, DQNAgent]] = {
        "Condition_A_Full": {},
        "Condition_B_No_BKT": {},
        "Condition_C_No_Action_History": {},
    }

    # 1. Condition A: Multi-seed training across seeds 42, 123, 456 for 800 episodes
    training_curves["Condition_A_Full"] = {"rewards": [], "losses": []}
    for seed in TRAIN_SEEDS:
        agent, rewards, losses = train_dqn_agent(
            seed=seed,
            curriculum=curriculum,
            condition="Condition_A_Full",
            num_episodes=NUM_TRAIN_EPISODES,
        )
        trained_agents["Condition_A_Full"][seed] = agent
        ckpt_path = EXPERIMENT_DIR / f"dqn_Condition_A_Full_seed_{seed}.pt"
        agent.save(ckpt_path)

        if not training_curves["Condition_A_Full"]["rewards"]:
            training_curves["Condition_A_Full"]["rewards"] = rewards
            training_curves["Condition_A_Full"]["losses"] = losses

    # Save primary agent model checkpoint
    trained_agents["Condition_A_Full"][TRAIN_SEEDS[0]].save(EXPERIMENT_DIR / "dqn_policy_net.pt")

    # 2. Ablation conditions: Condition B and Condition C (400 episodes)
    for cond in ["Condition_B_No_BKT", "Condition_C_No_Action_History"]:
        training_curves[cond] = {"rewards": [], "losses": []}
        agent, rewards, losses = train_dqn_agent(
            seed=TRAIN_SEEDS[0],
            curriculum=curriculum,
            condition=cond,
            num_episodes=400,
        )
        trained_agents[cond][TRAIN_SEEDS[0]] = agent
        ckpt_path = EXPERIMENT_DIR / f"dqn_{cond}_seed_{TRAIN_SEEDS[0]}.pt"
        agent.save(ckpt_path)
        training_curves[cond]["rewards"] = rewards
        training_curves[cond]["losses"] = losses


    # =========================================================================
    # Step 2: Comparative Baseline Evaluation (DQN vs Rule vs Fixed vs Random)
    # 100 evaluation episodes per learner archetype across disjoint seeds
    # =========================================================================
    print("\n--> Running comparative baseline evaluation on held-out learners...")
    archetypes = ["FastLearner", "SlowLearner", "InattentiveLearner"]

    eval_seed_base = 50000

    # Policies to compare
    policies = {
        "DQN": ("Condition_A_Full", trained_agents["Condition_A_Full"][TRAIN_SEEDS[0]]),
        "DQN_No_BKT": ("Condition_B_No_BKT", trained_agents["Condition_B_No_BKT"][TRAIN_SEEDS[0]]),
        "DQN_No_Action_History": ("Condition_C_No_Action_History", trained_agents["Condition_C_No_Action_History"][TRAIN_SEEDS[0]]),
        "RuleTeacher": ("Condition_A_Full", RuleTeacher(name="Rule")),
        "FixedTeacher": ("Condition_A_Full", FixedTeacher(name="Fixed")),
        "RandomTeacher": ("Condition_A_Full", RandomTeacher(name="Random")),
    }

    for policy_name, (cond, pol_obj) in policies.items():
        print(f"    Evaluating {policy_name} across archetypes...")
        for arch in archetypes:
            for ep_idx in range(NUM_EVAL_EPISODES_PER_TYPE):
                eval_seed = eval_seed_base + (hash((policy_name, arch)) % 10000) + ep_idx
                learner = sample_archetype_learner(arch, seed=eval_seed, concepts=concept_ids)

                res = run_episode(
                    policy_or_teacher=pol_obj,
                    learner=learner,
                    curriculum=curriculum,
                    condition=cond,
                    is_training=False,
                )

                records.append(
                    {
                        "seed": TRAIN_SEEDS[0],
                        "policy": policy_name,
                        "condition": cond,
                        "learner_type": arch,
                        "episode": ep_idx,
                        "learning_gain": res["learning_gain"],
                        "steps_to_mastery": res["steps_to_mastery"],
                        "cumulative_reward": res["cumulative_reward"],
                    }
                )

    # Save to metrics.csv
    df = pd.DataFrame(records)
    csv_path = EXPERIMENT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"--> Saved {len(df)} evaluation records to {csv_path}")

    # =========================================================================
    # Step 3: Publication-Grade Visualizations
    # =========================================================================
    print("--> Generating publication-quality figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: learning_gain_comparison.png (Boxplot)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    primary_policies = ["DQN", "RuleTeacher", "FixedTeacher", "RandomTeacher"]
    sub_df = df[df["policy"].isin(primary_policies)]

    # Group by learner_type and policy
    plot_data = []
    labels = []
    colors = ["#2563EB", "#059669", "#D97706", "#DC2626"]
    
    positions = []
    pos = 1
    for arch in archetypes:
        for p_idx, pol in enumerate(primary_policies):
            vals = sub_df[(sub_df["learner_type"] == arch) & (sub_df["policy"] == pol)]["learning_gain"].values
            plot_data.append(vals)
            positions.append(pos)
            pos += 1
        pos += 1  # gap between archetypes

    bp = ax1.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.7)
    for i, box in enumerate(bp["boxes"]):
        c_idx = i % len(primary_policies)
        box.set_facecolor(colors[c_idx])
        box.set_alpha(0.7)

    # Group labels on x-axis
    group_ticks = [2.5, 7.5, 12.5]
    ax1.set_xticks(group_ticks)
    ax1.set_xticklabels(archetypes, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Normalized Learning Gain (g)", fontsize=12, fontweight="bold")
    ax1.set_title("Normalized Learning Gain Across Learner Archetypes (DQN vs. Baselines)", fontsize=13, fontweight="bold")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.7) for i in range(len(primary_policies))]
    ax1.legend(handles, primary_policies, loc="upper right", frameon=True)
    plt.tight_layout()
    fig1_path = EXPERIMENT_DIR / "learning_gain_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"    Saved {fig1_path}")

    # Figure 2: training_reward_loss.png (Dual-axis convergence)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    rewards_series = training_curves["Condition_A_Full"]["rewards"]
    losses_series = training_curves["Condition_A_Full"]["losses"]

    # Smooth curves with rolling average
    window = 30
    if len(rewards_series) >= window:
        smooth_rewards = pd.Series(rewards_series).rolling(window, min_periods=1).mean()
    else:
        smooth_rewards = pd.Series(rewards_series)

    color_r = "#2563EB"
    ax2.set_xlabel("Training Episode", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Episode Reward", color=color_r, fontsize=12, fontweight="bold")
    ax2.plot(smooth_rewards, color=color_r, linewidth=2, label="Episode Reward (Moving Avg)")
    ax2.tick_params(axis="y", labelcolor=color_r)

    if losses_series:
        ax2_loss = ax2.twinx()
        color_l = "#DC2626"
        smooth_losses = pd.Series(losses_series).rolling(window, min_periods=1).mean()
        ax2_loss.set_ylabel("Huber Loss (Smooth L1)", color=color_l, fontsize=12, fontweight="bold")
        ax2_loss.plot(smooth_losses, color=color_l, linewidth=1.5, linestyle="--", label="Huber Loss")
        ax2_loss.tick_params(axis="y", labelcolor=color_l)
        ax2_loss.grid(False)

    plt.title("DQN Policy Training Dynamics: Convergence of Episode Reward & Huber Loss", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_path = EXPERIMENT_DIR / "training_reward_loss.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"    Saved {fig2_path}")

    # Figure 3: ablation_study_deltas.png (Ablations Bar Chart)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ablation_policies = ["DQN", "DQN_No_BKT", "DQN_No_Action_History"]
    ablation_labels = ["Full DQN + BKT (14-D)", "No BKT (Raw Acc)", "No Action History"]
    means = [df[df["policy"] == p]["learning_gain"].mean() for p in ablation_policies]
    stds = [df[df["policy"] == p]["learning_gain"].std() for p in ablation_policies]

    bar_colors = ["#2563EB", "#7C3AED", "#9333EA"]
    bars = ax3.bar(ablation_labels, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.8, width=0.55)
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, m / 2.0, f"{m:.3f}", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax3.set_ylabel("Mean Normalized Learning Gain", fontsize=12, fontweight="bold")
    ax3.set_title("Ablation Study: Impact of BKT Beliefs and Action History on Learning Gain", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3_path = EXPERIMENT_DIR / "ablation_study_deltas.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"    Saved {fig3_path}")

    elapsed = time.time() - start_time
    print(f"\n==========================================================================")
    print(f"DQN Benchmark & Ablation Suite Completed Successfully in {elapsed:.1f}s!")
    print(f"Artifacts generated in: {EXPERIMENT_DIR}")
    print(f"==========================================================================")


if __name__ == "__main__":
    run_benchmark()
