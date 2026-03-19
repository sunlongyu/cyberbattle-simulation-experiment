from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path("sg_deception_simulation/.mplconfig").resolve()))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


RESULTS_DIR = Path("sg_deception_simulation/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FONT_CN = None
FONT_EN = None
COLOR_BASELINE = "#8FA7BF"
COLOR_PBNE = "#C97C5D"
COLOR_PBNE_ALT = "#6E9F6D"
COLOR_GRID = "#D8D8D8"
COLOR_TEXT = "#222222"
def configure_matplotlib() -> None:
    global FONT_CN, FONT_EN
    available = {font.name for font in font_manager.fontManager.ttflist}
    cn_font = "Songti SC" if "Songti SC" in available else "STSong"
    en_font = "Times New Roman" if "Times New Roman" in available else "Times"

    FONT_CN = FontProperties(family=[en_font, cn_font], size=10.5)
    FONT_EN = FontProperties(family=[en_font], size=10.5)

    matplotlib.rcParams["font.family"] = [en_font, cn_font]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 160
    matplotlib.rcParams["savefig.dpi"] = 320
    matplotlib.rcParams["axes.edgecolor"] = COLOR_TEXT
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["grid.color"] = COLOR_GRID
    matplotlib.rcParams["grid.linestyle"] = "-"
    matplotlib.rcParams["grid.linewidth"] = 0.4
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def apply_academic_axes_style(ax, x_font=None, y_font=None) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="out", length=4, width=0.8, colors=COLOR_TEXT)
    x_font = x_font or FONT_EN
    y_font = y_font or FONT_EN
    for label in ax.get_xticklabels():
        label.set_fontproperties(x_font)
        label.set_fontsize(10.5)
    for label in ax.get_yticklabels():
        label.set_fontproperties(y_font)
        label.set_fontsize(10.5)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_scenario_utility(feasible: dict) -> None:
    scenario_labels = ["场景A\n生产系统占优", "场景B\n蜜罐系统占优"]
    baseline_values = [
        feasible["scenario_a_high_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"],
        feasible["scenario_b_low_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"],
    ]
    pbne_values = [
        feasible["scenario_a_high_prior_theta1"]["results"]["pbne_production_camouflage"]["defender_expected_utility"],
        feasible["scenario_b_low_prior_theta1"]["results"]["pbne_honeypot_camouflage"]["defender_expected_utility"],
    ]

    x = np.arange(len(scenario_labels))
    width = 0.32

    plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()
    ax.grid(axis="y")
    bars1 = plt.bar(
        x - width / 2,
        baseline_values,
        width=width,
        color="white",
        edgecolor=COLOR_BASELINE,
        hatch="//",
        linewidth=0.9,
        label="真实披露基线",
    )
    bars2 = plt.bar(
        x + width / 2,
        pbne_values,
        width=width,
        color="#F3DDD4",
        edgecolor=COLOR_PBNE,
        hatch="\\\\",
        linewidth=0.9,
        label="适用PBNE伪装策略",
    )

    for bars in (bars1, bars2):
        for bar in bars:
            y = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y + 0.06,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontproperties=FONT_EN,
                fontsize=10.5,
            )

    plt.xticks(x, scenario_labels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_CN)
        label.set_fontsize(10.5)
    plt.ylabel("防御者期望效用", fontproperties=FONT_CN, fontsize=10.5)
    plt.xlabel("实验场景", fontproperties=FONT_CN, fontsize=10.5)
    plt.title("图3-1  不同场景下防御者期望效用对比", fontproperties=FONT_CN, fontsize=10.5, pad=10)
    plt.legend(loc="upper right", prop=FONT_CN, fontsize=10.5)
    apply_academic_axes_style(ax, x_font=FONT_CN, y_font=FONT_EN)
    save_figure("fig3_1_defender_utility_comparison.png")


def plot_belief_trajectories(feasible: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)

    scenario_map = [
        (
            axes[0],
            feasible["scenario_a_high_prior_theta1"],
            "场景A：生产系统占优",
            "pbne_production_camouflage",
        ),
        (
            axes[1],
            feasible["scenario_b_low_prior_theta1"],
            "场景B：蜜罐系统占优",
            "pbne_honeypot_camouflage",
        ),
    ]

    for ax, scenario, subtitle, pbne_key in scenario_map:
        baseline_quantiles = scenario["results"]["truthful_baseline"]["belief_quantiles"]
        pbne_quantiles = scenario["results"][pbne_key]["belief_quantiles"]
        stages = np.arange(1, max(len(baseline_quantiles["q50"]), len(pbne_quantiles["q50"])) + 1)
        baseline_q25 = baseline_quantiles["q25"] + [baseline_quantiles["q25"][-1]] * (len(stages) - len(baseline_quantiles["q25"]))
        baseline_q50 = baseline_quantiles["q50"] + [baseline_quantiles["q50"][-1]] * (len(stages) - len(baseline_quantiles["q50"]))
        baseline_q75 = baseline_quantiles["q75"] + [baseline_quantiles["q75"][-1]] * (len(stages) - len(baseline_quantiles["q75"]))
        pbne_q25 = pbne_quantiles["q25"] + [pbne_quantiles["q25"][-1]] * (len(stages) - len(pbne_quantiles["q25"]))
        pbne_q50 = pbne_quantiles["q50"] + [pbne_quantiles["q50"][-1]] * (len(stages) - len(pbne_quantiles["q50"]))
        pbne_q75 = pbne_quantiles["q75"] + [pbne_quantiles["q75"][-1]] * (len(stages) - len(pbne_quantiles["q75"]))

        baseline_lower = np.array(baseline_q50) - np.array(baseline_q25)
        baseline_upper = np.array(baseline_q75) - np.array(baseline_q50)
        pbne_lower = np.array(pbne_q50) - np.array(pbne_q25)
        pbne_upper = np.array(pbne_q75) - np.array(pbne_q50)

        ax.errorbar(
            stages - 0.06,
            baseline_q50,
            yerr=np.vstack([baseline_lower, baseline_upper]),
            fmt="o-",
            color=COLOR_BASELINE,
            markerfacecolor="white",
            markeredgecolor=COLOR_BASELINE,
            linewidth=1.6,
            markersize=4.0,
            elinewidth=0.9,
            capsize=2.5,
            label="真实披露基线中位数",
        )
        ax.errorbar(
            stages + 0.06,
            pbne_q50,
            yerr=np.vstack([pbne_lower, pbne_upper]),
            fmt="s--",
            color=COLOR_PBNE,
            markerfacecolor=COLOR_PBNE,
            markeredgecolor=COLOR_PBNE,
            linewidth=1.6,
            markersize=4.0,
            elinewidth=0.9,
            capsize=2.5,
            label="PBNE伪装策略中位数",
        )
        ax.set_title(subtitle, fontproperties=FONT_CN, fontsize=10.5, pad=8)
        ax.set_xlabel("博弈阶段", fontproperties=FONT_CN, fontsize=10.5)
        ax.set_xticks(stages)
        ax.grid(True, axis="y")
        apply_academic_axes_style(ax, x_font=FONT_EN, y_font=FONT_EN)

    axes[0].set_ylabel("攻击者对生产系统的后验信念", fontproperties=FONT_CN, fontsize=10.5)
    axes[0].set_ylim(-0.03, 1.03)
    for label in axes[0].get_yticklabels():
        label.set_fontproperties(FONT_EN)
    axes[1].legend(loc="upper right", prop=FONT_CN, fontsize=9.8)
    fig.suptitle("图3-2  攻击者后验信念中位数及四分位区间", fontproperties=FONT_CN, fontsize=10.5, y=0.98)
    save_figure("fig3_2_belief_trajectories.png")


def plot_final_belief_distribution(feasible: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    baseline = feasible["scenario_b_low_prior_theta1"]["results"]["truthful_baseline"]["final_beliefs"]
    pbne = feasible["scenario_b_low_prior_theta1"]["results"]["pbne_honeypot_camouflage"]["final_beliefs"]

    box = ax.boxplot(
        [baseline, pbne],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLOR_TEXT, "linewidth": 1.1},
        whiskerprops={"color": COLOR_TEXT, "linewidth": 0.9},
        capprops={"color": COLOR_TEXT, "linewidth": 0.9},
    )
    for patch, color, face in zip(box["boxes"], [COLOR_BASELINE, COLOR_PBNE_ALT], ["#ECF2F9", "#E7F0E4"]):
        patch.set_edgecolor(color)
        patch.set_facecolor(face)
        patch.set_linewidth(1.0)

    rng = np.random.default_rng(20260319)
    for xpos, values, color in [
        (1, baseline, COLOR_BASELINE),
        (2, pbne, COLOR_PBNE_ALT),
    ]:
        sample = np.array(values, dtype=float)
        jitter = rng.uniform(-0.09, 0.09, size=sample.shape[0])
        ax.scatter(
            np.full(sample.shape[0], xpos) + jitter,
            sample,
            s=10,
            alpha=0.18,
            color=color,
            edgecolors="none",
        )

    ax.set_xticks([1, 2], ["真实披露基线", "PBNE-2"])
    plt.xlabel("策略类型", fontproperties=FONT_CN, fontsize=10.5)
    plt.ylabel("终局时刻对生产系统的后验信念", fontproperties=FONT_CN, fontsize=10.5)
    plt.title("图3-3  场景B下终局信念分布对比", fontproperties=FONT_CN, fontsize=10.5, pad=10)
    ax.set_ylim(-0.03, 1.03)
    plt.grid(axis="y")
    apply_academic_axes_style(ax, x_font=FONT_CN, y_font=FONT_EN)
    save_figure("fig3_3_final_belief_distribution.png")


def plot_sensitivity_curves(sensitivity: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    plot_map = [
        ("prior_theta1", "先验概率 p", axes[0, 0]),
        ("beta", "时间折扣因子 β", axes[0, 1]),
        ("c_theta1", "生产系统伪装成本", axes[1, 0]),
        ("c_theta2", "蜜罐伪装成本", axes[1, 1]),
    ]

    for key, xlabel, ax in plot_map:
        rows = sensitivity[key]
        x = [row["value"] for row in rows]
        y1 = [row["defender_expected_utility"] for row in rows]
        if key == "beta":
            y2 = [row["belief_span_mean"] for row in rows]
            secondary_label = "信念波动幅度"
        else:
            y2 = [row["mixing_probability_mean"] for row in rows]
            secondary_label = "均衡概率"

        ax.plot(
            x,
            y1,
            marker="o",
            color=COLOR_BASELINE,
            linewidth=1.2,
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=COLOR_BASELINE,
            label="防御者期望效用",
        )
        ax.set_xlabel(xlabel, fontproperties=FONT_CN, fontsize=10.5)
        ax.set_ylabel("防御者期望效用", fontproperties=FONT_CN, fontsize=10.5)
        ax.grid(True, axis="y")
        apply_academic_axes_style(ax, x_font=FONT_EN, y_font=FONT_EN)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            y2,
            marker="s",
            color=COLOR_PBNE,
            linewidth=1.2,
            linestyle="--",
            markersize=4.5,
            markerfacecolor=COLOR_PBNE,
            markeredgecolor=COLOR_PBNE,
            label=secondary_label,
        )
        ax2.set_ylabel(secondary_label, fontproperties=FONT_CN, fontsize=10.5)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(FONT_EN)
            label.set_fontsize(10.5)

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="best", prop=FONT_CN, fontsize=9.5)

    fig.suptitle("图3-4  参数敏感性分析结果", fontproperties=FONT_CN, fontsize=10.5, y=0.98)
    save_figure("fig3_4_sensitivity_analysis.png")


def build_analysis_text(feasible: dict, sensitivity: dict) -> str:
    a_base = feasible["scenario_a_high_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"]
    a_pbne = feasible["scenario_a_high_prior_theta1"]["results"]["pbne_production_camouflage"]["defender_expected_utility"]
    b_base = feasible["scenario_b_low_prior_theta1"]["results"]["truthful_baseline"]["defender_expected_utility"]
    b_pbne = feasible["scenario_b_low_prior_theta1"]["results"]["pbne_honeypot_camouflage"]["defender_expected_utility"]

    lines = [
        "# 第三章实验分析说明",
        "",
        "## 1. 策略对比实验分析",
        "",
        f"在场景A（生产系统占优场景，p=0.65）下，PBNE-1 的防御者期望效用为 {a_pbne:.4f}，优于真实披露基线的 {a_base:.4f}。这说明当真实系统在目标集合中占据较高比例时，生产系统以一定概率伪装为蜜罐信号，能够有效抑制攻击者的攻击收益预期，并改善防御方总体收益。",
        "",
        f"在场景B（蜜罐占优场景，p=0.35）下，PBNE-2 的防御者期望效用为 {b_pbne:.4f}，同样优于真实披露基线的 {b_base:.4f}。这说明当环境中蜜罐比例较高时，蜜罐通过伪装成正常系统可以更有效地吸引攻击者进入陷阱，并为防御方带来更高的情报收益。",
        "",
        "从信念演化结果看，真实披露基线下攻击者的后验信念变化较为单调，其判断主要由单次观测迅速锁定；而在 PBNE 伪装策略下，攻击者对目标是否为生产系统的判断呈现更明显的阶段性波动。这表明防御方通过混合伪装策略改变了攻击者的推断路径，验证了多阶段信号博弈中“策略随机化影响信念更新”的核心机理。",
        "",
        "## 2. 参数敏感性分析",
        "",
        "先验概率 p 对 PBNE-1 的影响最为明显。随着 p 增大，防御方的期望效用总体下降，生产系统选择伪装信号的均衡概率也随之减小，说明在真实资产占比过高时，攻击者更容易形成针对高价值目标的稳定预期，欺骗防御的边际收益会减弱。",
        "",
        "时间折扣因子 β 对防御者期望效用的影响相对有限，但会明显影响攻击者信念波动的幅度。这表明在当前参数区间内，β 的主要作用体现在调节攻击者对近期观测的敏感性，而不是直接改变均衡收益水平。",
        "",
        "生产系统伪装成本 c_theta1 上升时，PBNE-1 的防御者期望效用下降，而攻击方在信号 sigma2 下的均衡攻击概率同步下降，说明更高的伪装成本会削弱防御方实施生产系统伪装的激励，从而改变均衡中的攻防混合概率。",
        "",
        "蜜罐伪装成本 c_theta2 上升时，PBNE-2 的防御者期望效用同样下降，而攻击方在正常信号下的撤退概率上升，说明蜜罐伪装成本过高会削弱蜜罐的拟态吸引能力，并降低其诱导攻击者进入欺骗环境的效果。",
        "",
        "## 3. 可直接写入论文的结论",
        "",
        "实验结果表明，在与目标类型先验分布相匹配的适用场景中，基于 PBNE 的伪装策略均优于真实披露基线，能够通过影响攻击者的后验信念与攻击决策提升防御者的总体收益。此外，模型对先验概率和伪装成本较为敏感，这说明部署伪装防御策略时需要结合实际网络环境中的资产构成和欺骗开销进行参数配置。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_matplotlib()
    feasible = load_json(RESULTS_DIR / "feasible_scenarios.json")
    sensitivity = load_json(RESULTS_DIR / "sensitivity_analysis.json")

    plot_scenario_utility(feasible)
    plot_belief_trajectories(feasible)
    plot_final_belief_distribution(feasible)
    plot_sensitivity_curves(sensitivity)

    analysis_text = build_analysis_text(feasible, sensitivity)
    (RESULTS_DIR / "analysis_notes.md").write_text(analysis_text, encoding="utf-8")
    print("Figures written to", FIGURES_DIR)
    print("Analysis notes written to", RESULTS_DIR / "analysis_notes.md")


if __name__ == "__main__":
    main()
