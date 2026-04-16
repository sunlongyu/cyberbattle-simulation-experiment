from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((BASE_DIR / ".mplconfig").resolve()))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch


RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
COLOR_BASELINE = "#5077B8"
COLOR_PBNE = "#D97941"
COLOR_TEXT = "#1F1F1F"
COLOR_GRID = "#D9D9D9"
COLOR_CERTAIN_HONEYPOT = "#5B7FA3"
COLOR_CRITICAL_UNCERTAINTY = "#E39A1F"
COLOR_CERTAIN_REAL = "#C56565"
COLOR_OTHER_STATE = "#D6DCE5"
FONT_MIXED = None
FONT_EN = None


def configure_matplotlib() -> None:
    global FONT_MIXED, FONT_EN
    available = {font.name for font in font_manager.fontManager.ttflist}
    cn_font = "Songti SC" if "Songti SC" in available else "STSong"
    en_font = "Times New Roman" if "Times New Roman" in available else "Times"
    symbol_font = "DejaVu Sans" if "DejaVu Sans" in available else en_font

    FONT_MIXED = FontProperties(family=[en_font, cn_font, symbol_font], size=10.5)
    FONT_EN = FontProperties(family=[en_font], size=10.5)

    matplotlib.rcParams["font.family"] = [en_font, cn_font, symbol_font]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 160
    matplotlib.rcParams["savefig.dpi"] = 320
    matplotlib.rcParams["axes.edgecolor"] = COLOR_TEXT
    matplotlib.rcParams["axes.linewidth"] = 0.9
    matplotlib.rcParams["grid.color"] = COLOR_GRID
    matplotlib.rcParams["grid.linestyle"] = "-"
    matplotlib.rcParams["grid.linewidth"] = 0.45
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()


def apply_axes_style(ax, *, x_font=None, y_font=None) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="out", length=4.0, width=0.85, colors=COLOR_TEXT)
    x_font = x_font or FONT_EN
    y_font = y_font or FONT_EN
    for label in ax.get_xticklabels():
        label.set_fontproperties(x_font)
        label.set_fontsize(10.5)
    for label in ax.get_yticklabels():
        label.set_fontproperties(y_font)
        label.set_fontsize(10.5)


def plot_experiment_one(experiment_result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    scenario_order = ["scenario_a", "scenario_b"]
    for axis, scenario_key in zip(axes, scenario_order):
        scenario = experiment_result["scenarios"][scenario_key]
        pbne_rows = scenario["horizon_sweep"]["recursive_pbne"]
        baseline_rows = scenario["horizon_sweep"]["truthful_baseline"]
        stages = np.array([row["horizon"] for row in pbne_rows], dtype=float)
        pbne_values = np.array([row["discounted_cumulative_defender_utility"] for row in pbne_rows], dtype=float)
        baseline_values = np.array([row["discounted_cumulative_defender_utility"] for row in baseline_rows], dtype=float)
        advantage_values = pbne_values - baseline_values

        axis.axhline(0.0, color="#BFBFBF", linewidth=0.8, linestyle=":")
        axis.plot(
            stages,
            baseline_values,
            color=COLOR_BASELINE,
            linewidth=1.6,
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgecolor=COLOR_BASELINE,
            label="真实披露基线",
        )
        axis.plot(
            stages,
            pbne_values,
            color=COLOR_PBNE,
            linewidth=1.8,
            linestyle="-",
            marker="s",
            markersize=4.6,
            markerfacecolor=COLOR_PBNE,
            markeredgecolor="white",
            label="递归 PBNE 策略",
        )
        axis.fill_between(stages, baseline_values, pbne_values, color="#F4D7C3", alpha=0.28)
        axis.text(
            0.97,
            0.92,
            f"终局增益: {advantage_values[-1]:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontproperties=FONT_MIXED,
            fontsize=10.0,
            color=COLOR_TEXT,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#D9D9D9",
                "linewidth": 0.8,
                "alpha": 0.94,
            },
        )
        axis.set_title(scenario["label"], fontproperties=FONT_MIXED, fontsize=10.5, pad=8)
        axis.set_xlabel("终止时域 T", fontproperties=FONT_MIXED, fontsize=10.5)
        axis.set_xticks(stages)
        axis.grid(True, axis="y")
        apply_axes_style(axis, x_font=FONT_EN, y_font=FONT_EN)

    axes[0].set_ylabel("防御方折扣累计期望收益", fontproperties=FONT_MIXED, fontsize=10.5)
    axes[1].legend(loc="best", prop=FONT_MIXED, fontsize=9.8)
    fig.suptitle("不同场景下防御方折扣累计期望收益对比", fontproperties=FONT_MIXED, fontsize=10.5, y=0.98)
    save_figure("fig3_2_cumulative_defender_utility_comparison.png")


def plot_stage_average_payoff_convergence(experiment_result: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True)
    scenario_order = ["scenario_a", "scenario_b"]
    row_specs = [
        ("discounted_cumulative_defender_utility", "防御方折扣累计期望收益"),
        ("discounted_cumulative_attacker_utility", "攻击方折扣累计期望收益"),
    ]

    for col_index, scenario_key in enumerate(scenario_order):
        scenario = experiment_result["scenarios"][scenario_key]
        pbne_rows = scenario["horizon_sweep"]["recursive_pbne"]
        baseline_rows = scenario["horizon_sweep"]["truthful_baseline"]
        stages = np.array([row["horizon"] for row in pbne_rows], dtype=float)

        for row_index, (metric_key, ylabel) in enumerate(row_specs):
            axis = axes[row_index, col_index]
            pbne_values = np.array([row[metric_key] for row in pbne_rows], dtype=float)
            baseline_values = np.array([row[metric_key] for row in baseline_rows], dtype=float)
            axis.axhline(0.0, color="#BFBFBF", linewidth=0.8, linestyle=":")
            axis.plot(
                stages,
                baseline_values,
                color=COLOR_BASELINE,
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=COLOR_BASELINE,
                label="真实披露基线",
            )
            axis.plot(
                stages,
                pbne_values,
                color=COLOR_PBNE,
                linewidth=1.8,
                marker="s",
                markersize=4.2,
                markerfacecolor=COLOR_PBNE,
                markeredgecolor="white",
                label="递归 PBNE 策略",
            )
            axis.set_xticks(stages)
            axis.grid(True, axis="y")
            if row_index == 0:
                axis.set_title(scenario["label"], fontproperties=FONT_MIXED, fontsize=10.5, pad=8)
                axis.text(
                    0.97,
                    0.90,
                    f"末期均值: {pbne_values[-1]:.2f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontproperties=FONT_MIXED,
                    fontsize=9.8,
                    color=COLOR_TEXT,
                    bbox={
                        "boxstyle": "round,pad=0.20",
                        "facecolor": "white",
                        "edgecolor": "#D9D9D9",
                        "linewidth": 0.8,
                        "alpha": 0.94,
                    },
                )
            if col_index == 0:
                axis.set_ylabel(ylabel, fontproperties=FONT_MIXED, fontsize=10.5)
            if row_index == 1:
                axis.set_xlabel("终止时域 T", fontproperties=FONT_MIXED, fontsize=10.5)
            apply_axes_style(axis, x_font=FONT_EN, y_font=FONT_EN)

    axes[0, 1].legend(loc="best", prop=FONT_MIXED, fontsize=9.8)
    fig.suptitle("攻防双方递归期望收益随终止时域变化", fontproperties=FONT_MIXED, fontsize=10.5, y=0.98)
    save_figure("fig3_2b_stage_average_payoff_convergence.png")


def plot_terminal_belief_distribution(experiment_result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharex=True, sharey=True)
    scenario_order = ["scenario_a", "scenario_b"]
    category_specs = [
        ("certain_honeypot", "确信蜜罐", COLOR_CERTAIN_HONEYPOT),
        ("critical_uncertainty", "保持不确定", COLOR_CRITICAL_UNCERTAINTY),
        ("certain_real", "确信真实", COLOR_CERTAIN_REAL),
    ]

    for axis, scenario_key in zip(axes, scenario_order):
        scenario = experiment_result["scenarios"][scenario_key]
        baseline_mass = scenario["results"]["truthful_baseline"]["terminal_probability_mass"]
        pbne_mass = scenario["results"]["recursive_pbne"]["terminal_probability_mass"]
        threshold_key = next(
            key for key in pbne_mass.keys() if key not in {"0.000000", "1.000000"}
        ) if any(key not in {"0.000000", "1.000000"} for key in pbne_mass.keys()) else None
        distributions = {
            "真实披露基线": {
                "certain_honeypot": baseline_mass.get("0.000000", 0.0),
                "critical_uncertainty": baseline_mass.get(threshold_key, 0.0) if threshold_key else 0.0,
                "certain_real": baseline_mass.get("1.000000", 0.0),
            },
            "递归 PBNE 策略": {
                "certain_honeypot": pbne_mass.get("0.000000", 0.0),
                "critical_uncertainty": pbne_mass.get(threshold_key, 0.0) if threshold_key else 0.0,
                "certain_real": pbne_mass.get("1.000000", 0.0),
            },
        }

        y_positions = np.arange(len(distributions))
        axis.grid(axis="x")
        for index, (label, values) in enumerate(distributions.items()):
            left = 0.0
            for category_key, category_label, color in category_specs:
                width = values[category_key]
                axis.barh(index, width, left=left, height=0.46, color=color, edgecolor="white", linewidth=0.8)
                if width >= 0.08:
                    axis.text(
                        left + width / 2.0,
                        index,
                        f"{width:.1%}",
                        ha="center",
                        va="center",
                        fontproperties=FONT_EN,
                        fontsize=9.8,
                        color="white" if category_key != "critical_uncertainty" else COLOR_TEXT,
                    )
                left += width

        axis.set_xlim(0.0, 1.0)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(list(distributions.keys()), fontproperties=FONT_MIXED, fontsize=10.5)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_xticklabels([f"{int(tick * 100)}%" for tick in np.linspace(0.0, 1.0, 6)], fontproperties=FONT_EN, fontsize=10.5)
        axis.invert_yaxis()
        axis.set_title(scenario["label"], fontproperties=FONT_MIXED, fontsize=10.5, pad=8)
        apply_axes_style(axis, x_font=FONT_EN, y_font=FONT_MIXED)

    axes[0].set_ylabel("策略", fontproperties=FONT_MIXED, fontsize=10.5)
    axes[0].set_xlabel("终局识别状态占比", fontproperties=FONT_MIXED, fontsize=10.5)
    axes[1].set_xlabel("终局识别状态占比", fontproperties=FONT_MIXED, fontsize=10.5)
    legend_handles = [Patch(facecolor=color, edgecolor="none") for _, _, color in category_specs]
    legend_labels = [label for _, label, _ in category_specs]
    axes[1].legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, prop=FONT_MIXED, fontsize=9.8)
    fig.suptitle("终局公共信念分布对比", fontproperties=FONT_MIXED, fontsize=10.5, y=0.98)
    save_figure("fig3_4_terminal_public_belief_distribution.png")


def plot_experiment_three(experiment_result: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    markers = ["o", "s", "^", "D"]
    title_labels = {
        ("prior_theta1", "scenario_a"): "初始公共信念变化（场景 A）",
        ("defender_loss", "scenario_a"): "真实系统受攻击损失变化（场景 A）",
        ("c_theta1", "scenario_a"): "真实系统伪装成本变化（场景 A）",
        ("c_theta2", "scenario_b"): "蜜罐系统伪装成本变化（场景 B）",
    }
    legend_labels = {
        "prior_theta1": r"$p_1$",
        "defender_loss": r"$l_d$",
        "c_theta1": r"$c_{\theta_1}$",
        "c_theta2": r"$c_{\theta_2}$",
    }

    def format_parameter_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    for axis, panel in zip(axes.flat, experiment_result["plot_panels"]):
        handles = []
        labels = []
        for index, series in enumerate(panel["series"]):
            color = palette[index % len(palette)]
            marker = markers[index % len(markers)]
            horizon_rows = series["horizon_rows"]
            stages = np.array([row["horizon"] for row in horizon_rows], dtype=float)
            cumulative_values = np.array([row["discounted_cumulative_defender_utility"] for row in horizon_rows], dtype=float)
            line, = axis.plot(
                stages,
                cumulative_values,
                color=color,
                linewidth=1.8,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
            )
            handles.append(line)
            value_text = format_parameter_value(float(series["parameter_value"]))
            labels.append(f"{legend_labels[panel['parameter_name']]}={value_text}")

        axis.axhline(0.0, color="#BFBFBF", linewidth=0.8, linestyle=":")
        axis.set_title(
            title_labels[(panel["parameter_name"], panel["scenario_key"])],
            fontproperties=FONT_MIXED,
            fontsize=10.5,
            pad=8,
        )
        axis.set_xlabel("终止时域 T", fontproperties=FONT_MIXED, fontsize=10.5)
        axis.set_ylabel("防御方折扣累计期望收益", fontproperties=FONT_MIXED, fontsize=10.5)
        axis.grid(True, axis="y")
        if panel["series"]:
            base_stages = np.array([row["horizon"] for row in panel["series"][0]["horizon_rows"]], dtype=float)
            axis.set_xticks(base_stages)
        apply_axes_style(axis, x_font=FONT_EN, y_font=FONT_EN)
        axis.legend(handles, labels, loc="best", prop=FONT_MIXED, fontsize=9.4)

    fig.suptitle("关键参数变化下递归 PBNE 防御方折扣累计期望收益对比", fontproperties=FONT_MIXED, fontsize=10.5, y=0.99)
    save_figure("fig3_5_parameter_sensitivity_analysis.png")


def main() -> None:
    configure_matplotlib()
    experiment_one = load_json(RESULTS_DIR / "experiment1_payoff_comparison.json")
    experiment_two = load_json(RESULTS_DIR / "experiment2_belief_dynamics.json")
    experiment_three = load_json(RESULTS_DIR / "experiment3_sensitivity_analysis.json")
    plot_experiment_one(experiment_one)
    plot_stage_average_payoff_convergence(experiment_one)
    plot_terminal_belief_distribution(experiment_two)
    plot_experiment_three(experiment_three)
    print("Figure written to", FIGURES_DIR / "fig3_2_cumulative_defender_utility_comparison.png")
    print("Figure written to", FIGURES_DIR / "fig3_2b_stage_average_payoff_convergence.png")
    print("Figure written to", FIGURES_DIR / "fig3_4_terminal_public_belief_distribution.png")
    print("Figure written to", FIGURES_DIR / "fig3_5_parameter_sensitivity_analysis.png")


if __name__ == "__main__":
    main()
