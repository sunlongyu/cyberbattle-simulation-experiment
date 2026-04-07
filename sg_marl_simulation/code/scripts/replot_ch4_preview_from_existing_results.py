from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams["font.family"] = ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


ROOT = Path(__file__).resolve().parents[2]
MODULE1_ROOT = ROOT / "module1_convergence"
MODULE3_ROOT = ROOT / "module3_effectiveness"
PREVIEW_ROOT = ROOT / "preview_figures"
MODULE1_FIG_ROOT = MODULE1_ROOT / "figures"
MODULE3_FIG_ROOT = MODULE3_ROOT / "figures"


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _rolling_mean(values: List[float], window: int = 15) -> List[float]:
    if not values:
        return []
    rolled: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start : idx + 1]
        rolled.append(sum(bucket) / len(bucket))
    return rolled


def _bootstrap_ci(samples: List[float], n_boot: int = 320, seed: int = 20260407) -> Tuple[float, float]:
    if len(samples) < 2:
        point = samples[0] if samples else 0.0
        return point, point
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        resampled = [samples[rng.randrange(len(samples))] for _ in range(len(samples))]
        boot.append(_safe_mean(resampled))
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def _bootstrap_rows_metric(
    by_episode: Dict[int, List[dict]],
    metric_key: str,
    n_boot: int = 320,
    seed: int = 20260407,
) -> Tuple[float, float]:
    episodes = sorted(by_episode)
    if not episodes:
        return 0.0, 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sampled_rows: List[dict] = []
        for _ in episodes:
            sampled_rows.extend(by_episode[rng.choice(episodes)])
        boot.append(_metric_from_eval_rows(sampled_rows)[metric_key])
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float_rows(path: Path) -> List[dict]:
    rows = []
    for row in _read_csv_rows(path):
        parsed = {}
        for key, value in row.items():
            if value in ("", "nan", "NaN", None):
                parsed[key] = value
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        rows.append(parsed)
    return rows


def _metric_from_eval_rows(rows: List[dict]) -> Dict[str, float]:
    real_exposures = 0
    honeypot_exposures = 0
    real_attacks = 0
    honeypot_attacks = 0
    total_attacks = 0
    attack_probability = {
        "real": {"sN": [], "sH": []},
        "honeypot": {"sN": [], "sH": []},
    }
    for row in rows:
        theta_label = "real" if int(row["theta"]) == 0 else "honeypot"
        signal_label = "sN" if int(row["signal"]) == 0 else "sH"
        attacked = int(row["attacker_action"]) == 1
        attack_probability[theta_label][signal_label].append(float(row["attacker_prob_attack"]))
        if theta_label == "real":
            real_exposures += 1
            if attacked:
                real_attacks += 1
        else:
            honeypot_exposures += 1
            if attacked:
                honeypot_attacks += 1
        if attacked:
            total_attacks += 1
    return {
        "real_host_attack_rate": real_attacks / max(real_exposures, 1),
        "honeypot_hit_rate": honeypot_attacks / max(honeypot_exposures, 1),
        "deception_success_rate": honeypot_attacks / max(total_attacks, 1),
        "signal_effect": 0.5
        * (
            abs(_safe_mean(attack_probability["real"]["sN"]) - _safe_mean(attack_probability["real"]["sH"]))
            + abs(_safe_mean(attack_probability["honeypot"]["sN"]) - _safe_mean(attack_probability["honeypot"]["sH"]))
        ),
    }


def _episode_level_eval_metric(eval_path: Path, metric_key: str) -> Tuple[float, Tuple[float, float]]:
    rows = _read_csv_rows(eval_path)
    by_episode: Dict[int, List[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    point = _metric_from_eval_rows(rows)[metric_key]
    return point, _bootstrap_rows_metric(by_episode, metric_key)


def _training_mean_metric(log_path: Path, metric_key: str) -> Tuple[float, Tuple[float, float]]:
    rows = _parse_float_rows(log_path)
    samples = [float(row[metric_key]) for row in rows if isinstance(row.get(metric_key), float)]
    point = _safe_mean(samples)
    return point, _bootstrap_ci(samples)


def _training_final_window_metric(log_path: Path, metric_key: str, final_window: int) -> Tuple[float, Tuple[float, float]]:
    rows = _parse_float_rows(log_path)
    samples = [float(row[metric_key]) for row in rows if isinstance(row.get(metric_key), float)]
    point = _safe_mean(samples[-min(final_window, len(samples)) :])
    if len(samples) < final_window:
        return point, (point, point)
    trailing_means = []
    for idx in range(final_window - 1, len(samples)):
        trailing_means.append(_safe_mean(samples[idx - final_window + 1 : idx + 1]))
    return point, _bootstrap_ci(trailing_means)


def _save_figure(fig: plt.Figure, preview_name: str, target_dir: Path, target_name: str) -> None:
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_ROOT / f"{preview_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PREVIEW_ROOT / f"{preview_name}.pdf", bbox_inches="tight")
    fig.savefig(target_dir / f"{target_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(target_dir / f"{target_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig_4_4() -> None:
    seed_paths = sorted((MODULE1_ROOT / "csv").glob("episode_log_server_module1_formal_1000_seed*_seed_stability_*.csv"))
    seed_curves: List[List[float]] = []
    for path in seed_paths:
        rows = _parse_float_rows(path)
        seed_curves.append([float(row["episode_reward_mean"]) for row in rows])

    max_len = max(len(curve) for curve in seed_curves)
    mean_curve = []
    ci_curve = []
    for idx in range(max_len):
        bucket = [curve[idx] for curve in seed_curves if idx < len(curve)]
        mean_curve.append(statistics.mean(bucket))
        if len(bucket) > 1:
            std = statistics.stdev(bucket)
            ci_curve.append(1.96 * std / math.sqrt(len(bucket)))
        else:
            ci_curve.append(0.0)

    smoothed_mean = _rolling_mean(mean_curve, window=15)
    smoothed_ci = _rolling_mean(ci_curve, window=15)
    x = list(range(1, max_len + 1))

    fig, ax = plt.subplots(figsize=(12, 8))
    for curve in seed_curves:
        ax.plot(
            range(1, len(curve) + 1),
            _rolling_mean(curve, window=15),
            color="#B8C1CC",
            alpha=0.42,
            linewidth=1.15,
            zorder=1,
        )

    ax.plot(x, smoothed_mean, color="#1F4E79", linewidth=2.8, zorder=3, label="5种子均值")
    lower = [m - c for m, c in zip(smoothed_mean, smoothed_ci)]
    upper = [m + c for m, c in zip(smoothed_mean, smoothed_ci)]
    ax.fill_between(x, lower, upper, color="#9BB8D3", alpha=0.26, zorder=2, label="95%置信带")

    ax.set_title("不同随机种子下 SG-MAPPO 的平均回合收益轨迹", pad=10)
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("平均回合收益")
    ax.grid(True, alpha=0.24, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    _save_figure(
        fig,
        "fig_4_4_seed_stability_preview",
        MODULE1_FIG_ROOT,
        "fig_4_4_seed_stability",
    )


def _point_interval_panel(
    axis: plt.Axes,
    labels: List[str],
    values: List[float],
    intervals: List[Tuple[float, float] | None],
    colors: List[str],
    title: str,
    formatter: str = "{:.3f}",
) -> None:
    y_positions = list(range(len(labels)))[::-1]
    for y, label, value, interval, color in zip(y_positions, labels, values, intervals, colors):
        if interval is not None:
            low, high = interval
            axis.hlines(y, low, high, color=color, linewidth=2.4, zorder=2)
        axis.scatter(value, y, color=color, s=48, zorder=3)
        axis.annotate(
            formatter.format(value),
            xy=(value, y),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_title(title)
    axis.grid(True, axis="x", alpha=0.22, linestyle="--")
    axis.tick_params(axis="y", length=0)
    axis.margins(y=0.22, x=0.08)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_fig_4_6() -> None:
    table_path = MODULE3_ROOT / "tables/tab_4_4_baseline_metrics.csv"
    rows = {row["variant_name"]: row for row in _read_csv_rows(table_path)}
    variants = ["SG-MAPPO", "Plain-MAPPO"]
    labels = ["SG-MAPPO", "Plain-MAPPO"]
    colors = ["#4E79A7", "#D3725B"]
    eval_files = {
        "SG-MAPPO": MODULE3_ROOT / "csv/evaluation_baseline_SG-MAPPO.csv",
        "Plain-MAPPO": MODULE3_ROOT / "csv/evaluation_baseline_Plain-MAPPO.csv",
    }
    log_files = {
        "SG-MAPPO": MODULE3_ROOT / "csv/episode_log_baseline_SG-MAPPO.csv",
        "Plain-MAPPO": MODULE3_ROOT / "csv/episode_log_baseline_Plain-MAPPO.csv",
    }

    metrics = {
        "defender_reward": [],
        "signal_effect": [],
        "real_host_attack_rate": [],
        "honeypot_hit_rate": [],
    }
    intervals = {key: [] for key in metrics}
    for variant in variants:
        point, interval = _training_mean_metric(log_files[variant], "defender_reward")
        metrics["defender_reward"].append(float(rows[variant]["defender_reward"]))
        intervals["defender_reward"].append(interval)
        for metric_key in ("signal_effect", "real_host_attack_rate", "honeypot_hit_rate"):
            point, interval = _episode_level_eval_metric(eval_files[variant], metric_key)
            metrics[metric_key].append(float(rows[variant][metric_key]))
            intervals[metric_key].append(interval)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4))
    axes = axes.flatten()
    _point_interval_panel(
        axes[0],
        labels,
        metrics["defender_reward"],
        intervals["defender_reward"],
        colors,
        "防御方平均回合收益",
        formatter="{:.1f}",
    )
    _point_interval_panel(
        axes[1],
        labels,
        metrics["signal_effect"],
        intervals["signal_effect"],
        colors,
        "信号效应",
    )
    _point_interval_panel(
        axes[2],
        labels,
        metrics["real_host_attack_rate"],
        intervals["real_host_attack_rate"],
        colors,
        "真实主机攻击率",
    )
    _point_interval_panel(
        axes[3],
        labels,
        metrics["honeypot_hit_rate"],
        intervals["honeypot_hit_rate"],
        colors,
        "蜜罐命中率",
    )
    fig.suptitle("SG-MAPPO 与 Plain-MAPPO 的关键指标点估计与区间对比", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(
        fig,
        "fig_4_6_baseline_compare_preview",
        MODULE3_FIG_ROOT,
        "fig_4_6_baseline_compare",
    )


def plot_fig_4_7() -> None:
    table_path = MODULE3_ROOT / "tables/tab_4_5_ablation_metrics.csv"
    rows = {row["variant_name"]: row for row in _read_csv_rows(table_path)}
    variants = ["完整SG-MAPPO", "去除信念输入", "去除LSTM"]
    labels = ["SG-MAPPO", "No-Belief", "No-LSTM"]
    colors = ["#0066CC", "#8B4513", "#00AADE"]
    eval_files = {
        "完整SG-MAPPO": MODULE3_ROOT / "csv/evaluation_ablation_完整SG-MAPPO.csv",
        "去除信念输入": MODULE3_ROOT / "csv/evaluation_ablation_去除信念输入.csv",
        "去除LSTM": MODULE3_ROOT / "csv/evaluation_ablation_去除LSTM.csv",
    }
    log_files = {
        "完整SG-MAPPO": MODULE3_ROOT / "csv/episode_log_ablation_完整SG-MAPPO.csv",
        "去除信念输入": MODULE3_ROOT / "csv/episode_log_ablation_去除信念输入.csv",
        "去除LSTM": MODULE3_ROOT / "csv/episode_log_ablation_去除LSTM.csv",
    }

    metrics = {
        "defender_final_avg_reward": [],
        "real_host_attack_rate": [],
        "deception_success_rate": [],
        "signal_effect": [],
    }
    intervals = {key: [] for key in metrics}
    for variant in variants:
        point, interval = _training_final_window_metric(log_files[variant], "defender_reward", final_window=20)
        metrics["defender_final_avg_reward"].append(float(rows[variant]["defender_final_avg_reward"]))
        intervals["defender_final_avg_reward"].append(None)
        for metric_key in ("real_host_attack_rate", "deception_success_rate", "signal_effect"):
            point, interval = _episode_level_eval_metric(eval_files[variant], metric_key)
            metrics[metric_key].append(float(rows[variant][metric_key]))
            intervals[metric_key].append(interval)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4))
    axes = axes.flatten()
    _point_interval_panel(
        axes[0],
        labels,
        metrics["defender_final_avg_reward"],
        intervals["defender_final_avg_reward"],
        colors,
        "防御方最终平均回报",
        formatter="{:.1f}",
    )
    _point_interval_panel(
        axes[1],
        labels,
        metrics["real_host_attack_rate"],
        intervals["real_host_attack_rate"],
        colors,
        "真实主机攻击率",
    )
    _point_interval_panel(
        axes[2],
        labels,
        metrics["deception_success_rate"],
        intervals["deception_success_rate"],
        colors,
        "欺骗成功率",
    )
    _point_interval_panel(
        axes[3],
        labels,
        metrics["signal_effect"],
        intervals["signal_effect"],
        colors,
        "信号效应",
    )
    fig.suptitle("SG-MAPPO 消融实验的点估计与区间对比", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(
        fig,
        "fig_4_7_ablation_compare_preview",
        MODULE3_FIG_ROOT,
        "fig_4_7_ablation_compare",
    )


def main() -> None:
    plot_fig_4_4()
    plot_fig_4_6()
    plot_fig_4_7()
    print(PREVIEW_ROOT)


if __name__ == "__main__":
    main()
