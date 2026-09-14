"""Plotting utilities for AirTutor experiments (Roadmap Sec. 23 & 35).

Generates publication-quality figures:
1. Training reward curve (averaged across training seeds with std error band).
2. Learning gain by teacher with standard error bars.
"""

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_training_rewards(
    seed_rewards: Mapping[int, Sequence[float]],
    output_path: Path | str,
    window: int = 15,
) -> None:
    """Plot mean training reward curve across independent RL seeds with shaded std band."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(seed_rewards.keys())
    all_runs = np.array([seed_rewards[s] for s in seeds])  # shape: (n_seeds, n_episodes)
    n_episodes = all_runs.shape[1]
    episodes = np.arange(1, n_episodes + 1)

    mean_rewards = np.mean(all_runs, axis=0)
    std_rewards = np.std(all_runs, axis=0)

    plt.figure(figsize=(9, 4.8), dpi=150)

    # Plot individual runs faintly
    for s_idx, s in enumerate(seeds):
        plt.plot(episodes, all_runs[s_idx], color="#B0BEC5", alpha=0.35, linewidth=0.8)

    # Shaded +/- 1 std region
    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="#BBDEFB",
        alpha=0.5,
        label=r"$\pm 1\,\sigma$ across seeds",
    )

    # Smoothed rolling average of the mean
    if n_episodes >= window:
        rolling_mean = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        rolling_episodes = np.arange(window, n_episodes + 1)
        plt.plot(
            rolling_episodes,
            rolling_mean,
            color="#0D47A1",
            linewidth=2.2,
            label=f"Mean ({window}-ep moving avg)",
        )
    else:
        plt.plot(episodes, mean_rewards, color="#0D47A1", linewidth=2.0, label="Mean Reward")

    plt.title(f"RL Training Reward Curve ({len(seeds)} Seeds, Reward v1)", fontsize=12, pad=12)
    plt.xlabel("Training Episode", fontsize=10)
    plt.ylabel(r"Episode Reward ($\sum \Delta \hat{M}$)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def plot_learning_gain_comparison(
    teacher_metrics: Mapping[str, Mapping[str, tuple[float, float]]],
    output_path: Path | str,
    metric_key: str = "learning_gain",
    title: str = "Held-Out Learning Gain by Teacher",
) -> None:
    """Plot learning gain bar chart comparing teachers with standard deviation error bars."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    teachers = list(teacher_metrics.keys())
    means = [teacher_metrics[t][metric_key][0] for t in teachers]
    stds = [teacher_metrics[t][metric_key][1] for t in teachers]

    # Palette: distinct modern hues
    colors = ["#78909C", "#FFA726", "#42A5F5", "#2E7D32"]
    if len(teachers) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(teachers)))

    plt.figure(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(teachers))
    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(teachers)],
        edgecolor="#37474F",
        linewidth=0.8,
        alpha=0.88,
        error_kw={"elinewidth": 1.5, "ecolor": "#263238"},
    )

    # Label values on top of bars
    for bar, m in zip(bars, means):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        plt.annotate(
            f"{m:+.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.xticks(x, teachers, fontsize=10, fontweight="bold")
    plt.ylabel("Mean Learning Gain", fontsize=10)
    plt.title(title, fontsize=12, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()
