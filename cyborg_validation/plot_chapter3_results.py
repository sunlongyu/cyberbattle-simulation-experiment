from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_PATH = RESULTS_DIR / "chapter3_formal_experiments.json"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results() -> dict:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.facecolor": "#faf8f3",
            "figure.facecolor": "#f4f1e8",
            "savefig.facecolor": "#f4f1e8",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_primary_blue_rewards(data: dict) -> None:
    primary = data["primary_comparisons"]
    scenarios = [
        ("Scenario A", primary["scenario_a_high_prior_theta1"]),
        ("Scenario B", primary["scenario_b_low_prior_theta1"]),
    ]

    truthful = [item["truthful_aggregate"]["blue_reward_mean"] for _, item in scenarios]
    alternative = [item["alternative_aggregate"]["blue_reward_mean"] for _, item in scenarios]
    labels = [label for label, _ in scenarios]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], truthful, width=width, color="#b85c38", label="Truthful")
    ax.bar([i + width / 2 for i in x], alternative, width=width, color="#2f6f5e", label="PBNE")
    ax.set_title("Effectiveness Comparison: Blue Reward")
    ax.set_ylabel("Average Blue Reward")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)

    for i, value in enumerate(truthful):
        ax.text(i - width / 2, value + 6, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(alternative):
        ax.text(i + width / 2, value + 6, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save(fig, "chapter3_primary_blue_rewards.png")


def plot_primary_security_metrics(data: dict) -> None:
    primary = data["primary_comparisons"]
    metrics = [
        ("Attack-like\nActions", "attack_like_count_mean"),
        ("Production\nCompromise", "production_compromise_rate"),
        ("Critical Host\nCompromise", "critical_host_compromise_rate"),
    ]
    scenarios = [
        ("Scenario A", primary["scenario_a_high_prior_theta1"]),
        ("Scenario B", primary["scenario_b_low_prior_theta1"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, key) in zip(axes, metrics):
        truthful = [item["truthful_aggregate"][key] for _, item in scenarios]
        alternative = [item["alternative_aggregate"][key] for _, item in scenarios]
        x = range(len(scenarios))
        width = 0.34
        ax.bar([i - width / 2 for i in x], truthful, width=width, color="#d9a441", label="Truthful")
        ax.bar([i + width / 2 for i in x], alternative, width=width, color="#3a7ca5", label="PBNE")
        ax.set_title(title)
        ax.set_xticks(list(x), [label for label, _ in scenarios])
        if "Compromise" in title:
            ax.set_ylim(0, 1.05)
        for i, value in enumerate(truthful):
            ax.text(i - width / 2, value + 0.03 if value <= 1.05 else value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        for i, value in enumerate(alternative):
            ax.text(i + width / 2, value + 0.03 if value <= 1.05 else value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Effectiveness Comparison: Security Metrics", y=1.03, fontsize=13)
    save(fig, "chapter3_primary_security_metrics.png")


def plot_compromise_breakdown(data: dict) -> None:
    primary = data["primary_comparisons"]
    scenarios = [
        ("Scenario A\nTruthful", primary["scenario_a_high_prior_theta1"]["truthful_aggregate"]),
        ("Scenario A\nPBNE-1", primary["scenario_a_high_prior_theta1"]["alternative_aggregate"]),
        ("Scenario B\nTruthful", primary["scenario_b_low_prior_theta1"]["truthful_aggregate"]),
        ("Scenario B\nPBNE-2", primary["scenario_b_low_prior_theta1"]["alternative_aggregate"]),
    ]
    production = [item["production_compromise_rate"] for _, item in scenarios]
    honeypot = [item["honeypot_compromise_rate"] for _, item in scenarios]
    critical = [item["critical_host_compromise_rate"] for _, item in scenarios]

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = range(len(scenarios))
    width = 0.22
    ax.bar([i - width for i in x], production, width=width, color="#a23b72", label="Production")
    ax.bar(list(x), honeypot, width=width, color="#f18f01", label="Honeypot")
    ax.bar([i + width for i in x], critical, width=width, color="#2e86ab", label="Critical")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Compromise Rate")
    ax.set_title("Compromise Structure Under Different Policies")
    ax.set_xticks(list(x), [label for label, _ in scenarios])
    ax.legend(frameon=False, ncol=3, loc="upper center")
    for idx, values in enumerate([production, honeypot, critical]):
        shift = (-width, 0, width)[idx]
        for i, value in enumerate(values):
            ax.text(i + shift, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    save(fig, "chapter3_compromise_breakdown.png")


def plot_theory_environment_alignment(data: dict) -> None:
    theory_path = Path("/Users/SL/Documents/expproject/sg_deception_simulation/results/feasible_scenarios.json")
    theory = json.loads(theory_path.read_text(encoding="utf-8"))
    env_primary = data["primary_comparisons"]

    labels = ["Scenario A", "Scenario B"]
    theory_gain = [
        theory["scenario_a_high_prior_theta1"]["results"]["pbne_production_camouflage"]["defender_expected_utility"]
        - theory["scenario_a_high_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"],
        theory["scenario_b_low_prior_theta1"]["results"]["pbne_honeypot_camouflage"]["defender_expected_utility"]
        - theory["scenario_b_low_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"],
    ]
    env_gain = [
        env_primary["scenario_a_high_prior_theta1"]["comparison"]["blue_reward_lift"],
        env_primary["scenario_b_low_prior_theta1"]["comparison"]["blue_reward_lift"],
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], theory_gain, width=width, color="#4f5d75", label="Theory Utility Gain")
    ax.bar([i + width / 2 for i in x], env_gain, width=width, color="#ef8354", label="Environment Reward Gain")
    ax.set_title("Directional Alignment Between Theory and Environment")
    ax.set_ylabel("Gain Relative to Truthful Baseline")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    for i, value in enumerate(theory_gain):
        ax.text(i - width / 2, value + 1.5, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(env_gain):
        ax.text(i + width / 2, value + 1.5, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "chapter3_theory_environment_alignment.png")


def plot_sensitivity(data: dict, sweep_key: str, title: str, filename: str) -> None:
    rows = data["sensitivity_sweeps"][sweep_key]
    xs = [row["sweep_value"] for row in rows]
    reward_lift = [row["comparison"]["blue_reward_lift"] for row in rows]
    attack_reduction = [row["comparison"]["attack_like_count_reduction"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax1.plot(xs, reward_lift, color="#8f2d56", marker="o", linewidth=2.2, label="Blue Reward Lift")
    ax1.set_xlabel(sweep_key)
    ax1.set_ylabel("Blue Reward Lift", color="#8f2d56")
    ax1.tick_params(axis="y", labelcolor="#8f2d56")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(xs, attack_reduction, color="#1f6e8c", marker="s", linewidth=2.0, label="Attack-like Reduction")
    ax2.set_ylabel("Attack-like Count Reduction", color="#1f6e8c")
    ax2.tick_params(axis="y", labelcolor="#1f6e8c")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    save(fig, filename)


def plot_prior_effect(data: dict) -> None:
    plot_sensitivity(
        data,
        sweep_key="prior_theta1",
        title="Sensitivity: Effect of Prior Probability",
        filename="chapter3_sensitivity_prior_theta1.png",
    )


def plot_beta_effect(data: dict) -> None:
    plot_sensitivity(
        data,
        sweep_key="beta",
        title="Sensitivity: Effect of Belief Discount Beta",
        filename="chapter3_sensitivity_beta.png",
    )


def main() -> None:
    style()
    data = load_results()
    plot_primary_blue_rewards(data)
    plot_primary_security_metrics(data)
    plot_compromise_breakdown(data)
    plot_theory_environment_alignment(data)
    plot_prior_effect(data)
    plot_beta_effect(data)
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
