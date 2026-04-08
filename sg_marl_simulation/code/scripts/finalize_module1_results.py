from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from marl_core.io import write_csv
from marl_core.metrics import compute_convergence_step, compute_reward_volatility


ROOT = Path("/Users/SL/Documents/expproject/MARL/results/ch4/module1_convergence")
CSV_ROOT = ROOT / "csv"
FIG_ROOT = ROOT / "figures"
TABLE_ROOT = ROOT / "tables"

LR_PREFIX = "episode_log_module1_tuned_lr200_seed0_lr_sweep_"
SEED_PREFIX = "episode_log_module1_tuned_seed200_seed"

CONVERGENCE_RULE = {
    "threshold_ratio": 0.95,
    "stability_window": 20,
    "final_window": 30,
}

rcParams["font.family"] = ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_episode_csv(path: Path, run_name: str, experiment_group: str) -> dict:
    rows = load_rows(path)
    rewards = [float(row["episode_reward_mean"]) for row in rows]
    defender = [float(row["defender_reward"]) for row in rows]
    attacker = [float(row["attacker_reward"]) for row in rows]
    final_window = min(CONVERGENCE_RULE["final_window"], len(rewards))
    return {
        "run_name": run_name,
        "seed": rows[0]["seed"],
        "learning_rate": float(rows[0]["learning_rate"]),
        "mean_reward": statistics.mean(rewards),
        "std_reward": compute_reward_volatility(rewards),
        "mean_defender_reward": statistics.mean(defender),
        "std_defender_reward": compute_reward_volatility(defender),
        "mean_attacker_reward": statistics.mean(attacker),
        "std_attacker_reward": compute_reward_volatility(attacker),
        "final_avg_reward": statistics.mean(rewards[-final_window:]),
        "convergence_step": compute_convergence_step(rewards, **CONVERGENCE_RULE),
        "training_time_s": float(rows[-1]["time_total_s"]),
        "reward_volatility": compute_reward_volatility(rewards),
        "checkpoint_path": str(ROOT / "logs" / run_name),
        "experiment_group": experiment_group,
    }


def plot_learning_rate(rows_by_lr: dict[float, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ordered = sorted(rows_by_lr.items(), key=lambda item: item[0], reverse=True)
    for learning_rate, rows in ordered:
        x = [int(row["training_iteration"]) for row in rows]
        label = f"学习率={learning_rate:.1e}"
        axes[0, 0].plot(x, [float(row["defender_policy_loss"]) for row in rows], linewidth=1.8, label=label)
        axes[0, 1].plot(x, [float(row["attacker_policy_loss"]) for row in rows], linewidth=1.8, label=label)
        axes[1, 0].plot(x, [float(row["defender_reward"]) for row in rows], linewidth=2.0, label=label)
        axes[1, 1].plot(x, [float(row["attacker_reward"]) for row in rows], linewidth=2.0, label=label)

    axes[0, 0].set_title("防御方策略损失")
    axes[0, 1].set_title("攻击方策略损失")
    axes[1, 0].set_title("防御方回合收益")
    axes[1, 1].set_title("攻击方回合收益")
    for axis in axes.flat:
        axis.set_xlabel("训练迭代次数")
        axis.grid(True, alpha=0.28, linestyle="--")
        axis.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("损失值")
    axes[1, 0].set_ylabel("收益值")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "fig_4_3_learning_rate.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_ROOT / "fig_4_3_learning_rate.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_seed_stability(seed_rows: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    curves = []
    for seed, rows in sorted(seed_rows.items()):
        rewards = [float(row["episode_reward_mean"]) for row in rows]
        curves.append(rewards)
        ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.22, linewidth=1.4, label=f"Seed {seed}")

    max_len = max(len(curve) for curve in curves)
    mean_curve = []
    std_curve = []
    for idx in range(max_len):
        bucket = [curve[idx] for curve in curves if idx < len(curve)]
        mean_curve.append(statistics.mean(bucket))
        std_curve.append(statistics.pstdev(bucket) if len(bucket) > 1 else 0.0)

    x = list(range(1, max_len + 1))
    lower = [m - s for m, s in zip(mean_curve, std_curve)]
    upper = [m + s for m, s in zip(mean_curve, std_curve)]
    ax.plot(x, mean_curve, color="#111111", linewidth=2.4, label="均值")
    ax.fill_between(x, lower, upper, color="#9AA5B1", alpha=0.28, label="均值±标准差")
    ax.set_title("5次随机种子训练稳定性")
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("平均回合收益")
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "fig_4_4_seed_stability.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_ROOT / "fig_4_4_seed_stability.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    lr_files = sorted(CSV_ROOT.glob(f"{LR_PREFIX}*.csv"))
    seed_files = sorted(CSV_ROOT.glob(f"{SEED_PREFIX}*_seed_stability_*.csv"))

    lr_summary = []
    lr_rows_for_plot: dict[float, list[dict]] = {}
    for path in lr_files:
        rows = load_rows(path)
        lr = float(rows[0]["learning_rate"])
        run_name = path.stem.replace("episode_log_module1_tuned_lr200_seed0_", "")
        lr_rows_for_plot[lr] = rows
        lr_summary.append(summarize_episode_csv(path, run_name, "learning_rate_sweep"))

    seed_summary = []
    seed_rows_for_plot: dict[int, list[dict]] = {}
    for path in seed_files:
        rows = load_rows(path)
        seed = int(rows[0]["seed"])
        run_name = path.stem.replace(f"episode_log_module1_tuned_seed200_seed{seed}_", "")
        seed_rows_for_plot[seed] = rows
        seed_summary.append(summarize_episode_csv(path, run_name, "seed_stability"))

    aggregate = {
        "run_name": "seed_stability_aggregate",
        "seed": "aggregate",
        "learning_rate": seed_summary[0]["learning_rate"],
        "mean_reward": statistics.mean(row["mean_reward"] for row in seed_summary),
        "std_reward": statistics.mean(row["std_reward"] for row in seed_summary),
        "mean_defender_reward": statistics.mean(row["mean_defender_reward"] for row in seed_summary),
        "std_defender_reward": statistics.mean(row["std_defender_reward"] for row in seed_summary),
        "mean_attacker_reward": statistics.mean(row["mean_attacker_reward"] for row in seed_summary),
        "std_attacker_reward": statistics.mean(row["std_attacker_reward"] for row in seed_summary),
        "final_avg_reward": statistics.mean(row["final_avg_reward"] for row in seed_summary),
        "convergence_step": statistics.mean(row["convergence_step"] for row in seed_summary if row["convergence_step"] is not None),
        "training_time_s": statistics.mean(row["training_time_s"] for row in seed_summary),
        "reward_volatility": statistics.mean(row["reward_volatility"] for row in seed_summary),
        "checkpoint_path": "",
        "experiment_group": "seed_stability_aggregate",
    }

    final_rows = sorted(lr_summary, key=lambda row: row["learning_rate"], reverse=True) + seed_summary + [aggregate]
    write_csv(ROOT / "results_summary.csv", final_rows)
    write_csv(TABLE_ROOT / "tab_4_2_convergence_stats.csv", final_rows)
    plot_learning_rate(lr_rows_for_plot)
    plot_seed_stability(seed_rows_for_plot)


if __name__ == "__main__":
    main()
