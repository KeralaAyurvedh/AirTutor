"""Plotting utilities for AirTutor experiments (Roadmap Sec. 23 & 35).

Generates publication-quality figures:
1. Training reward curve (averaged across training seeds with std error band).
2. Learning gain by teacher with standard error bars.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

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


def plot_reward_ablation_comparison(
    reward_results: Mapping[str, Any],
    output_path: Path | str,
) -> None:
    """Plot metrics and qualitative action preference shifts across Reward v1, v2, v3 (Roadmap Sec. 15)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    versions = list(reward_results.keys())  # ["v1", "v2", "v3"]
    actions = ["EXPLAIN", "EXAMPLE", "HINT", "PRACTICE", "REVIEW_PREREQUISITE"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Subplot 1: Learning Gain (Obs & True)
    x = np.arange(len(versions))
    width = 0.35

    obs_gains = [reward_results[v]["learning_gain"][0] for v in versions]
    obs_errs = [reward_results[v]["learning_gain"][1] for v in versions]
    true_gains = [reward_results[v]["true_learning_gain"][0] for v in versions]
    true_errs = [reward_results[v]["true_learning_gain"][1] for v in versions]

    rects1 = ax1.bar(x - width / 2, obs_gains, width, yerr=obs_errs, capsize=4,
                     label="Observable Gain", color="#1E88E5", alpha=0.85, edgecolor="#1565C0")
    rects2 = ax1.bar(x + width / 2, true_gains, width, yerr=true_errs, capsize=4,
                     label="True Latent Gain", color="#43A047", alpha=0.85, edgecolor="#2E7D32")

    ax1.set_ylabel("Learning Gain", fontsize=10)
    ax1.set_title("Learning Gain by Reward Formulation", fontsize=11, pad=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Reward {v.upper()}" for v in versions], fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:+.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:+.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Subplot 2: Action Preference Distribution (%)
    action_colors = ["#5C6BC0", "#26A69A", "#FFA726", "#42A5F5", "#AB47BC"]
    x_act = np.arange(len(actions))
    w_act = 0.25

    for i, v in enumerate(versions):
        dist = reward_results[v]["action_distribution"]
        props = [dist.get(a, 0.0) * 100.0 for a in actions]
        offset = (i - 1) * w_act
        bars = ax2.bar(x_act + offset, props, w_act, label=f"Reward {v.upper()}",
                       alpha=0.85, edgecolor="#37474F")

    ax2.set_ylabel("Action Selection Frequency (%)", fontsize=10)
    ax2.set_title("Qualitative Action Preference Shift", fontsize=11, pad=10, fontweight="bold")
    ax2.set_xticks(x_act)
    ax2.set_xticklabels(actions, fontsize=8, rotation=20, ha="right")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_state_ablation_deltas(
    state_results: Mapping[str, Mapping[str, tuple[float, float]]],
    output_path: Path | str,
    baseline_key: str = "Full State (320)",
) -> None:
    """Plot metric deltas relative to full state representation (Roadmap Sec. 22)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_obs = state_results[baseline_key]["learning_gain"][0]
    base_true = state_results[baseline_key]["true_learning_gain"][0]

    variants = [k for k in state_results.keys() if k != baseline_key]
    delta_obs = [state_results[k]["learning_gain"][0] - base_obs for k in variants]
    delta_true = [state_results[k]["true_learning_gain"][0] - base_true for k in variants]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width / 2, delta_obs, width, label=r"$\Delta$ Obs Learning Gain",
                   color="#1E88E5", alpha=0.85, edgecolor="#1565C0")
    bars2 = ax.bar(x + width / 2, delta_true, width, label=r"$\Delta$ True Learning Gain",
                   color="#FB8C00", alpha=0.85, edgecolor="#EF6C00")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"$\Delta$ vs. Full State", fontsize=10)
    ax.set_title("State Representation Ablation: Impact on Learning Gain", fontsize=11, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate deltas
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 3 if h >= 0 else -10
        ax.annotate(f"{h:+.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, offset), textcoords="offset points", ha="center", va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sensitivity_sweep(
    sensitivity_results: Mapping[str, Any],
    output_path: Path | str,
) -> None:
    """Plot RL Advantage (RL Gain - Rule Gain) across perturbation axes (Roadmap Sec. 21)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conditions = list(sensitivity_results.keys())
    advantages = [sensitivity_results[c]["advantage"][0] for c in conditions]
    adv_errs = [sensitivity_results[c]["advantage"][1] for c in conditions]
    statuses = [sensitivity_results[c]["status"] for c in conditions]

    # Color code by robustness status
    status_colors = {
        "Holds": "#2E7D32",     # Green: advantage reliably maintained
        "Shrinks": "#F57C00",   # Orange: advantage decreased but positive
        "Vanishes": "#C62828",  # Red: advantage eliminated or inverted
    }
    colors = [status_colors.get(s, "#546E7A") for s in statuses]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    y = np.arange(len(conditions))

    bars = ax.barh(y, advantages, xerr=adv_errs, capsize=4, color=colors,
                   alpha=0.85, edgecolor="#263238", height=0.6,
                   error_kw={"elinewidth": 1.2, "ecolor": "#263238"})

    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.axvline(0.02, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="Holds threshold (+0.02)")

    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontsize=9, fontweight="bold")
    ax.set_xlabel(r"RL Advantage Delta ($\text{Gain}_{\mathrm{RL}} - \text{Gain}_{\mathrm{Rule}}$)", fontsize=10)
    fig.suptitle("Robustness & Sensitivity Sweep: RL Advantage Over Rule Baseline",
                 fontsize=12, fontweight="bold", y=0.98)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    max_val = max(advantages) if advantages else 0.15
    ax.set_xlim(0, max(0.15, max_val * 1.25))

    # Value annotations on bars
    for b, adv, st in zip(bars, advantages, statuses):
        w = b.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 5 if w >= 0 else -5
        ax.annotate(f"{adv:+.3f} [{st}]", xy=(w, b.get_y() + b.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points", ha=ha, va="center",
                    fontsize=8, fontweight="bold")

    # Legend for statuses placed cleanly above plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", label="Holds (Advantage > +0.02)"),
        Patch(facecolor="#F57C00", label="Shrinks (0 <= Advantage <= +0.02)"),
        Patch(facecolor="#C62828", label="Vanishes (Advantage < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, framealpha=0.9, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path)
    plt.close()

