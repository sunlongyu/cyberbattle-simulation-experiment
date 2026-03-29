from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


ROOT = Path("/Users/SL/Documents/expproject/MARL/results/ch4/module2_policy")
TABLE_PATH = ROOT / "tables" / "tab_4_3_parameter_sensitivity_overview.csv"
FIG_PATH_PNG = ROOT / "figures" / "fig_4_5_parameter_sensitivity_overview.png"
FIG_PATH_PDF = ROOT / "figures" / "fig_4_5_parameter_sensitivity_overview.pdf"

rcParams["font.family"] = ["Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


def load_rows() -> list[dict]:
    with TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["parameter_name"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row["parameter_value"]))
    return grouped


def plot() -> None:
    rows = load_rows()
    grouped = group_rows(rows)

    panels = [
        ("signal_effect", "Signal Effect"),
        ("real_host_attack_rate", "真实主机攻击率"),
        ("honeypot_hit_rate", "蜜罐命中率"),
        ("deception_success_rate", "欺骗成功率"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = {
        "T": "#184E77",
        "N": "#2A9D8F",
        "psi0": "#E76F51",
        "c_theta1": "#8D6A9F",
        "c_theta2": "#BC4749",
    }
    labels = {
        "T": "T",
        "N": "N",
        "psi0": "psi0",
        "c_theta1": "cθ1",
        "c_theta2": "cθ2",
    }

    for ax, (metric_key, title) in zip(axes, panels):
        for parameter_name, values in grouped.items():
            x = [str(row["parameter_value"]).rstrip("0").rstrip(".") for row in values]
            y = [float(row[metric_key]) for row in values]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                label=labels.get(parameter_name, parameter_name),
                color=colors.get(parameter_name),
            )
        ax.set_title(title)
        ax.set_xlabel("参数取值")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("数值")
    axes[2].set_ylabel("数值")
    plt.tight_layout()
    plt.savefig(FIG_PATH_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_PATH_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
