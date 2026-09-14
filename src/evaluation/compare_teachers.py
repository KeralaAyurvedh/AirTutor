"""Comparative Evaluation & Benchmark Protocol for AirTutor (Roadmap Sec. 17, 20, 23).

Executes the Stage-3 core research benchmark:
1. Trains tabular Q-learning agents across multiple independent seeds on [0, N_train).
2. Evaluates Random, Fixed, Rule, and RL teachers on a strictly HELD-OUT population
   of N_eval=100 simulated learners sampled from a DISJOINT seed range [10000, 10100).
3. Computes statistical distributions (mean, std, 95% CI) for all 6 core metrics.
4. Generates both required publication plots:
   - Training reward curve (RL)
   - Learning gain by teacher with error bars
5. Logs all experimental artifacts under experiments/<run_id>/ per Roadmap Sec. 23.
"""

import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from src.environment.actions import PedagogicalAction
from src.environment.curriculum import CurriculumRepository, load_curriculum
from src.evaluation.baseline_comparison import run_tutoring_episode
from src.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from src.evaluation.plotting import plot_learning_gain_comparison, plot_training_rewards
from src.rl.q_learning import TabularQAgent
from src.rl.train import train_multi_seed
from src.simulator.population import sample_learner
from src.teachers.fixed_teacher import FixedTeacher
from src.teachers.random_teacher import RandomTeacher
from src.teachers.rule_teacher import RuleTeacher


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash for experiment reproducibility logging."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "uncommitted_or_git_unavailable"


def evaluate_agent_on_held_out(
    agent: TabularQAgent,
    eval_seeds: Sequence[int],
    curriculum: CurriculumRepository,
    max_steps: int = 30,
    param_ranges: dict[str, tuple[float, float]] | None = None,
    action_effect_multiplier: float = 1.0,
    difficulty_offset: float = 0.0,
) -> tuple[list[EpisodeMetrics], list[float]]:
    """Evaluate a trained RL agent greedily on the held-out learner population."""
    concept_ids = curriculum.get_all_concept_ids()
    metrics_list: list[EpisodeMetrics] = []
    true_gains: list[float] = []

    for seed in eval_seeds:
        learner = sample_learner(
            seed=seed,
            concepts=concept_ids,
            initial_mastery=0.10,
            param_ranges=param_ranges,
        )
        # Note: TabularQAgent inherits from BaseTeacher and select_action has greedy=False by default.
        # Temporarily force epsilon to 0 for pure greedy policy evaluation
        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0

        ep_metrics, true_gain = run_tutoring_episode(
            teacher=agent,
            learner=learner,
            curriculum=curriculum,
            max_steps=max_steps,
            initial_mastery=0.10,
            action_effect_multiplier=action_effect_multiplier,
            difficulty_offset=difficulty_offset,
        )
        agent.epsilon = saved_epsilon

        metrics_list.append(ep_metrics)
        true_gains.append(true_gain)

    return metrics_list, true_gains


def compute_confidence_intervals_95(
    metrics_list: Sequence[EpisodeMetrics],
) -> dict[str, tuple[float, float, float]]:
    """Compute (mean, std, 95% CI half-width) for each metric across episodes."""
    keys = [
        "learning_gain",
        "final_mastery",
        "steps_to_mastery",
        "intervention_efficiency",
        "hint_usage_rate",
        "action_entropy",
    ]
    n = len(metrics_list)
    results: dict[str, tuple[float, float, float]] = {}

    for k in keys:
        vals = [getattr(m, k) for m in metrics_list]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        # 95% CI: +/- 1.96 * (std / sqrt(n))
        ci95 = float(1.96 * (std / np.sqrt(n))) if n > 1 else 0.0
        results[k] = (mean, std, ci95)

    return results


def run_full_benchmark(
    n_train: int = 300,
    n_eval: int = 100,
    train_seed_start: int = 0,
    eval_seed_start: int = 10000,
    rl_seeds: Sequence[int] = (42, 101, 2024, 777, 999),
    max_steps_per_episode: int = 30,
    output_dir: Path | str = "experiments",
) -> Path:
    """Execute complete Stage 3 benchmark and export all logging and visualization artifacts."""
    # 1. Assert disjoint training and evaluation populations (Roadmap Sec. 17)
    train_seeds = list(range(train_seed_start, train_seed_start + n_train))
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))

    assert max(train_seeds) < min(eval_seeds), (
        f"Data leakage detected! Training seed range [{min(train_seeds)}, {max(train_seeds)}] "
        f"overlaps with evaluation seed range [{min(eval_seeds)}, {max(eval_seeds)}]."
    )

    curriculum = load_curriculum("data/curriculum")
    concept_ids = curriculum.get_all_concept_ids()

    # Create timestamped experiment run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"stage3_benchmark_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_commit_hash()
    print("=" * 88)
    print(f" AirTutor Stage 3 Benchmark: Tabular Q-Learning vs. Baselines")
    print(f" Run ID: {run_id} | Git Commit: {git_hash[:8]}")
    print(f" Training Population: {n_train} learners [seeds {train_seed_start}-{train_seed_start + n_train - 1}]")
    print(f" Held-Out Evaluation: {n_eval} learners [seeds {eval_seed_start}-{eval_seed_start + n_eval - 1}] (DISJOINT)")
    print(f" RL Training Seeds:   {list(rl_seeds)} ({len(rl_seeds)} independent runs)")
    print("=" * 88)

    # 2. Train RL agents across multiple seeds
    print("\n[Phase 1/3: Multi-Seed RL Training]")
    trained_agents, training_reward_trajectories = train_multi_seed(
        rl_seeds=rl_seeds,
        num_train_learners=n_train,
        train_seed_start=train_seed_start,
        num_episodes=n_train,
        curriculum=curriculum,
        verbose=True,
    )

    # 3. Evaluate all teachers on the held-out population
    print(f"\n[Phase 2/3: Held-Out Evaluation on {n_eval} Unseen Learners]")
    baseline_teachers = [
        RandomTeacher(seed=eval_seed_start),
        FixedTeacher(),
        RuleTeacher(mastery_threshold=0.75, review_interval=5, max_steps_per_concept=6),
    ]

    all_eval_metrics: dict[str, list[EpisodeMetrics]] = {}
    all_true_gains: dict[str, list[float]] = {}

    # Evaluate non-learning baselines
    for teacher in baseline_teachers:
        print(f"-> Evaluating {teacher.name} Teacher on {n_eval} held-out learners...")
        t_metrics: list[EpisodeMetrics] = []
        t_true_gains: list[float] = []

        for seed in eval_seeds:
            learner = sample_learner(seed=seed, concepts=concept_ids, initial_mastery=0.10)
            ep_m, tg = run_tutoring_episode(
                teacher=teacher,
                learner=learner,
                curriculum=curriculum,
                max_steps=max_steps_per_episode,
                initial_mastery=0.10,
            )
            t_metrics.append(ep_m)
            t_true_gains.append(tg)

        all_eval_metrics[teacher.name] = t_metrics
        all_true_gains[teacher.name] = t_true_gains

    # Evaluate RL agents (evaluate each trained seed greedily across held-out learners)
    print(f"-> Evaluating RL Policy across {len(trained_agents)} trained seeds on {n_eval} held-out learners...")
    rl_eval_metrics: list[EpisodeMetrics] = []
    rl_true_gains: list[float] = []

    for agent in trained_agents:
        ag_m, ag_tg = evaluate_agent_on_held_out(
            agent=agent,
            eval_seeds=eval_seeds,
            curriculum=curriculum,
            max_steps=max_steps_per_episode,
        )
        rl_eval_metrics.extend(ag_m)
        rl_true_gains.extend(ag_tg)

    all_eval_metrics["RL"] = rl_eval_metrics
    all_true_gains["RL"] = rl_true_gains

    # 4. Compute statistical summaries
    print("\n[Phase 3/3: Statistical Aggregation & Publication Artifacts]")
    summary_report: dict[str, dict[str, Any]] = {}

    for teacher_name, m_list in all_eval_metrics.items():
        stats = compute_confidence_intervals_95(m_list)
        tg_mean = float(np.mean(all_true_gains[teacher_name]))
        tg_std = float(np.std(all_true_gains[teacher_name]))
        tg_ci95 = float(1.96 * (tg_std / np.sqrt(len(all_true_gains[teacher_name]))))

        stats["true_learning_gain"] = (tg_mean, tg_std, tg_ci95)
        summary_report[teacher_name] = stats

    # 5. Generate and export plots
    reward_curve_path = run_dir / "training_reward_curve.png"
    gain_comparison_path = run_dir / "learning_gain_comparison.png"

    plot_training_rewards(training_reward_trajectories, reward_curve_path)
    # Also save formatted input for plot_learning_gain_comparison
    plot_summary = {
        t: {k: (v[0], v[1]) for k, v in summary_report[t].items()} for t in summary_report
    }
    plot_learning_gain_comparison(plot_summary, gain_comparison_path)

    # Also copy plots directly to experiments/ for immediate discovery
    root_exp = Path(output_dir)
    plot_training_rewards(training_reward_trajectories, root_exp / "training_reward_curve.png")
    plot_learning_gain_comparison(plot_summary, root_exp / "learning_gain_comparison.png")

    # 6. Save experiment metadata and JSON logs (Roadmap Sec. 23)
    config_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_hash,
        "n_train_learners": n_train,
        "train_seed_range": [train_seed_start, train_seed_start + n_train],
        "n_eval_learners": n_eval,
        "eval_seed_range": [eval_seed_start, eval_seed_start + n_eval],
        "rl_seeds": list(rl_seeds),
        "reward_version": "v1 (r_t = delta M_hat)",
        "state_encoder": "default_v1 (320 states, 1600 Q-values)",
        "max_steps_per_episode": max_steps_per_episode,
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Save raw episode records
    raw_records = {
        t: [m.to_dict() for m in m_list] for t, m_list in all_eval_metrics.items()
    }
    with open(run_dir / "results_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)

    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(git_hash + "\n")

    # Print summary table with 95% CI
    print_benchmark_table(summary_report)

    print(f"\nAll benchmark artifacts saved to: {run_dir}")
    print(f"Plots saved to: {reward_curve_path} and {gain_comparison_path}")

    return run_dir


def print_benchmark_table(summary: dict[str, dict[str, tuple[float, float, float]]]) -> None:
    """Format and print full statistical benchmark table."""
    teachers = list(summary.keys())
    col_w = 26

    metric_labels = [
        ("Learning Gain (Obs)", "learning_gain"),
        ("True Learning Gain", "true_learning_gain"),
        ("Final Mastery", "final_mastery"),
        ("Steps to Mastery", "steps_to_mastery"),
        ("Efficiency (Gain/Step)", "intervention_efficiency"),
        ("Hint Usage Rate", "hint_usage_rate"),
        ("Action Entropy (bits)", "action_entropy"),
    ]

    print("\n" + "=" * 94)
    print(" Held-Out Benchmark Results: Mean +/- Std [95% CI] (Disjoint Population)")
    print("=" * 94)
    header = f"{'Metric':24} | " + " | ".join(f"{t:^{col_w}}" for t in teachers)
    print(header)
    print("-" * 94)

    for label, key in metric_labels:
        row = f"{label:24} | "
        cols = []
        for t in teachers:
            mean, std, ci95 = summary[t][key]
            if key == "steps_to_mastery":
                formatted = f"{mean:.1f} +/- {std:.1f} [+/-{ci95:.1f}]"
            elif key == "intervention_efficiency":
                formatted = f"{mean:.4f} [+/-{ci95:.4f}]"
            else:
                formatted = f"{mean:.3f} +/- {std:.3f} [+/-{ci95:.3f}]"
            cols.append(f"{formatted:^{col_w}}")
        row += " | ".join(cols)
        print(row)

    print("=" * 94)

    # Honest assessment per GEMINI.md standing rule
    rl_gain = summary["RL"]["learning_gain"][0]
    rule_gain = summary["Rule"]["learning_gain"][0]
    fixed_gain = summary["Fixed"]["learning_gain"][0]
    random_gain = summary["Random"]["learning_gain"][0]

    print("\n[Standing Rule Research Assessment (GEMINI.md)]")
    print(f"  RL Gain:     {rl_gain:+.3f}")
    print(f"  Rule Gain:   {rule_gain:+.3f}  (Delta vs Rule: {rl_gain - rule_gain:+.3f})")
    print(f"  Fixed Gain:  {fixed_gain:+.3f}  (Delta vs Fixed: {rl_gain - fixed_gain:+.3f})")
    print(f"  Random Gain: {random_gain:+.3f} (Delta vs Random: {rl_gain - random_gain:+.3f})")

    if rl_gain > rule_gain:
        print("  -> Finding: RL outperforms Rule baseline on held-out learners.")
    else:
        print(
            "  -> Finding: RL does NOT outperform the strong Rule baseline under Reward v1 on held-out learners.\n"
            "     Per GEMINI.md, this result is retained and reported as-is without altering simulator or baselines."
        )


def main() -> None:
    run_full_benchmark(
        n_train=300,
        n_eval=100,
        train_seed_start=0,
        eval_seed_start=10000,
        rl_seeds=(42, 101, 2024, 777, 999),
        max_steps_per_episode=30,
    )


if __name__ == "__main__":
    main()
