"""Master Experiment Runner for AirTutor Stage 4 (Roadmap Sec. 15, 21, 22).

Executes the complete Stage 4 research battery in a single, reproducible, headless run:
1. Reward Ablation (v1 vs. v2 vs. v3) and qualitative action preference shift analysis.
2. State Representation Ablation (Full 320 vs. No-Hint 160 vs. No-Accuracy 80).
3. Population Diversity Ablation (Wide bounds vs. Narrow bounds training generalization).
4. Robustness & Sensitivity Sweep across 12 simulator parameter axes.
5. Saves all publication plots to experiments/ and timestamped run directory:
   - experiments/reward_ablation_comparison.png
   - experiments/state_ablation_deltas.png
   - experiments/sensitivity_sweep.png
6. Dumps structured JSON summaries and git provenance metadata.
"""

import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.environment.curriculum import load_curriculum
from src.evaluation.ablations import (
    run_population_diversity_ablation,
    run_reward_ablation,
    run_state_ablation,
)
from src.evaluation.plotting import (
    plot_reward_ablation_comparison,
    plot_sensitivity_sweep,
    plot_state_ablation_deltas,
)
from src.evaluation.sensitivity import run_sensitivity_sweep
from src.rl.train import train_multi_seed


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "uncommitted_or_git_unavailable"


def run_stage4_experiments(
    n_train: int = 300,
    n_eval: int = 100,
    train_seed_start: int = 0,
    eval_seed_start: int = 10000,
    rl_seeds: Sequence[int] = (42, 101, 2024),
    output_dir: Path | str = "experiments",
) -> Path:
    """Run all Stage 4 experiments and output structured artifacts."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"stage4_experiments_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    git_hash = get_git_commit_hash()

    print("=" * 92)
    print(" AirTutor Stage 4: Comprehensive Ablations & Robustness Sensitivity Sweep")
    print(f" Run ID: {run_id} | Git Commit: {git_hash[:8]}")
    print(f" RL Training Seeds: {list(rl_seeds)} | Train Learners: {n_train} | Held-Out Eval: {n_eval}")
    print("=" * 92)

    curriculum = load_curriculum("data/curriculum")

    # -------------------------------------------------------------------------
    # 1. Base RL Model Training (Reward v1) for Sensitivity Analysis
    # -------------------------------------------------------------------------
    print("\n[Phase 1/5: Training Base RL Agents for Sensitivity Sweep]")
    base_agents, _ = train_multi_seed(
        rl_seeds=rl_seeds,
        num_train_learners=n_train,
        train_seed_start=train_seed_start,
        num_episodes=n_train,
        curriculum=curriculum,
        reward_version="v1",
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # 2. Reward Ablation (v1 vs v2 vs v3)
    # -------------------------------------------------------------------------
    print("\n[Phase 2/5: Reward Formulation Ablation (v1 vs. v2 vs. v3)]")
    reward_results = run_reward_ablation(
        rl_seeds=rl_seeds,
        n_train=n_train,
        n_eval=n_eval,
        train_seed_start=train_seed_start,
        eval_seed_start=eval_seed_start,
        curriculum=curriculum,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # 3. State Representation Ablation
    # -------------------------------------------------------------------------
    print("\n[Phase 3/5: State Representation Ablation (Full vs. No-Hint vs. No-Accuracy)]")
    state_results = run_state_ablation(
        rl_seeds=rl_seeds,
        n_train=n_train,
        n_eval=n_eval,
        train_seed_start=train_seed_start,
        eval_seed_start=eval_seed_start,
        reward_version="v1",
        curriculum=curriculum,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # 4. Population Diversity Ablation
    # -------------------------------------------------------------------------
    print("\n[Phase 4/5: Population Diversity Ablation (Wide vs. Narrow Training Bounds)]")
    pop_results = run_population_diversity_ablation(
        rl_seeds=rl_seeds,
        n_train=n_train,
        n_eval=n_eval,
        train_seed_start=train_seed_start,
        eval_seed_start=eval_seed_start,
        reward_version="v1",
        curriculum=curriculum,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # 5. Robustness & Sensitivity Sweep (12 axes)
    # -------------------------------------------------------------------------
    print("\n[Phase 5/5: Robustness & Sensitivity Sweep across 12 Perturbation Axes]")
    sensitivity_results = run_sensitivity_sweep(
        agents=base_agents,
        n_eval=n_eval,
        eval_seed_start=eval_seed_start,
        curriculum=curriculum,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # 6. Generate and Save Publication Plots
    # -------------------------------------------------------------------------
    print("\n[Generating Publication Plots]")
    p_reward = run_dir / "reward_ablation_comparison.png"
    p_state = run_dir / "state_ablation_deltas.png"
    p_sens = run_dir / "sensitivity_sweep.png"

    plot_reward_ablation_comparison(reward_results, p_reward)
    plot_state_ablation_deltas(state_results, p_state)
    plot_sensitivity_sweep(sensitivity_results, p_sens)

    # Copy directly into experiments/ for immediate discovery per prompt requirements
    root_exp = Path(output_dir)
    shutil.copyfile(p_reward, root_exp / "reward_ablation_comparison.png")
    shutil.copyfile(p_state, root_exp / "state_ablation_deltas.png")
    shutil.copyfile(p_sens, root_exp / "sensitivity_sweep.png")

    print(f" -> Saved {root_exp / 'reward_ablation_comparison.png'}")
    print(f" -> Saved {root_exp / 'state_ablation_deltas.png'}")
    print(f" -> Saved {root_exp / 'sensitivity_sweep.png'}")

    # -------------------------------------------------------------------------
    # 7. Save JSON Metadata and Summary Dumps
    # -------------------------------------------------------------------------
    master_summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_hash,
        "n_train": n_train,
        "n_eval": n_eval,
        "rl_seeds": list(rl_seeds),
        "reward_ablation": {
            v: {
                "learning_gain": reward_results[v]["learning_gain"],
                "true_learning_gain": reward_results[v]["true_learning_gain"],
                "intervention_efficiency": reward_results[v]["intervention_efficiency"],
                "action_distribution": reward_results[v]["action_distribution"],
            }
            for v in reward_results
        },
        "state_ablation": {
            k: {
                "learning_gain": state_results[k]["learning_gain"],
                "true_learning_gain": state_results[k]["true_learning_gain"],
                "intervention_efficiency": state_results[k]["intervention_efficiency"],
            }
            for k in state_results
        },
        "population_diversity_ablation": {
            k: {
                "learning_gain": pop_results[k]["learning_gain"],
                "true_learning_gain": pop_results[k]["true_learning_gain"],
                "intervention_efficiency": pop_results[k]["intervention_efficiency"],
            }
            for k in pop_results
        },
        "sensitivity_sweep": {
            k: {
                "rl_gain": sensitivity_results[k]["rl_gain"],
                "rule_gain": sensitivity_results[k]["rule_gain"],
                "advantage": sensitivity_results[k]["advantage"],
                "status": sensitivity_results[k]["status"],
            }
            for k in sensitivity_results
        },
    }

    with open(run_dir / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    with open(root_exp / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(git_hash + "\n")

    print(f"\nAll Stage 4 experimental artifacts saved cleanly to: {run_dir}")
    return run_dir


def main() -> None:
    run_stage4_experiments(
        n_train=300,
        n_eval=100,
        train_seed_start=0,
        eval_seed_start=10000,
        rl_seeds=(42, 101, 2024),
    )


if __name__ == "__main__":
    main()
