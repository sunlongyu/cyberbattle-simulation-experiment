from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE1_ROOT = REPO_ROOT / "module1_convergence"
MODULE3_ROOT = REPO_ROOT / "module3_effectiveness"
RNG = random.Random(20260407)


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(q * (len(sorted_values) - 1))
    return sorted_values[index]


def rolling_mean(values: list[float], window: int) -> list[float]:
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        smoothed.append(safe_mean(segment))
    return smoothed


def write_text(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = "#222222",
    anchor: str = "middle",
    weight: str = "normal",
) -> None:
    parts.append(
        "<text "
        f"x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
        f"text-anchor='{anchor}' font-weight='{weight}' "
        "font-family='PingFang SC, Microsoft YaHei, Helvetica, Arial'>"
        f"{text}</text>"
    )


def metric_from_evaluation_rows(rows: list[dict[str, str]]) -> dict[str, float]:
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
        theta = "real" if int(row["theta"]) == 0 else "honeypot"
        signal = "sN" if int(row["signal"]) == 0 else "sH"
        attacked = int(row["attacker_action"]) == 1
        attack_probability[theta][signal].append(float(row["attacker_prob_attack"]))
        if theta == "real":
            real_exposures += 1
            if attacked:
                real_attacks += 1
        else:
            honeypot_exposures += 1
            if attacked:
                honeypot_attacks += 1
        if attacked:
            total_attacks += 1

    signal_effect = 0.5 * (
        abs(safe_mean(attack_probability["real"]["sN"]) - safe_mean(attack_probability["real"]["sH"]))
        + abs(
            safe_mean(attack_probability["honeypot"]["sN"])
            - safe_mean(attack_probability["honeypot"]["sH"])
        )
    )
    return {
        "real_host_attack_rate": real_attacks / max(real_exposures, 1),
        "honeypot_hit_rate": honeypot_attacks / max(honeypot_exposures, 1),
        "deception_success_rate": honeypot_attacks / max(total_attacks, 1),
        "signal_effect": signal_effect,
    }


def bootstrap_ci(
    eval_path: Path,
    metric_name: str,
    *,
    samples: int = 260,
) -> tuple[float, float, float]:
    rows = list(csv.DictReader(eval_path.open(encoding="utf-8-sig", newline="")))
    rows_by_episode: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode"]), []).append(row)

    episodes = sorted(rows_by_episode)
    all_rows = [row for episode in episodes for row in rows_by_episode[episode]]
    point_estimate = metric_from_evaluation_rows(all_rows)[metric_name]

    bootstrap_values: list[float] = []
    for _ in range(samples):
        sampled_rows: list[dict[str, str]] = []
        for _episode in episodes:
            episode = RNG.choice(episodes)
            sampled_rows.extend(rows_by_episode[episode])
        bootstrap_values.append(metric_from_evaluation_rows(sampled_rows)[metric_name])
    bootstrap_values.sort()
    return (
        point_estimate,
        quantile(bootstrap_values, 0.025),
        quantile(bootstrap_values, 0.975),
    )


def render_module1_seed_stability() -> Path:
    csv_root = MODULE1_ROOT / "csv"
    figure_root = MODULE1_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    seed_paths = sorted(csv_root.glob("episode_log_server_module1_formal_1000_seed*_seed_stability_*.csv"))
    seed_series: list[list[float]] = []
    for path in seed_paths:
        rewards = [
            float(row["episode_reward_mean"])
            for row in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))
        ]
        seed_series.append(rewards)

    length = min(len(series) for series in seed_series)
    seed_series = [series[:length] for series in seed_series]
    mean_curve = [safe_mean(list(values)) for values in zip(*seed_series)]
    std_curve = [sample_std(list(values)) for values in zip(*seed_series)]
    ci_curve = [1.96 * value / math.sqrt(len(seed_series)) for value in std_curve]
    smooth_mean = rolling_mean(mean_curve, window=15)
    smooth_ci = rolling_mean(ci_curve, window=15)

    width = 1120
    height = 680
    margin_left = 96
    margin_right = 36
    margin_top = 62
    margin_bottom = 74
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_y = []
    for series in seed_series:
        all_y.extend(series)
    all_y.extend([value + ci for value, ci in zip(smooth_mean, smooth_ci)])
    all_y.extend([value - ci for value, ci in zip(smooth_mean, smooth_ci)])
    ymin = min(all_y)
    ymax = max(all_y)
    padding = 0.08 * (ymax - ymin)
    ymin -= padding
    ymax += padding

    def x_pos(index: int) -> float:
        return margin_left + (index / max(length - 1, 1)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + plot_height - ((value - ymin) / max(ymax - ymin, 1e-6)) * plot_height

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
    ]

    for tick in range(6):
        y = margin_top + plot_height * tick / 5
        parts.append(
            f"<line x1='{margin_left}' y1='{y:.1f}' x2='{width - margin_right}' y2='{y:.1f}' "
            "stroke='#D9DFE7' stroke-dasharray='4 4' stroke-width='1'/>"
        )
        tick_value = ymax - (y - margin_top) / plot_height * (ymax - ymin)
        write_text(parts, margin_left - 12, y + 4, f"{tick_value:.0f}", size=11, anchor="end", fill="#556677")

    for tick in range(6):
        x = margin_left + plot_width * tick / 5
        parts.append(
            f"<line x1='{x:.1f}' y1='{margin_top}' x2='{x:.1f}' y2='{height - margin_bottom}' "
            "stroke='#EEF2F6' stroke-width='1'/>"
        )
        tick_value = int(1 + (length - 1) * tick / 5)
        write_text(parts, x, height - margin_bottom + 28, str(tick_value), size=11, fill="#556677")

    parts.append(
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' "
        "stroke='#334455' stroke-width='1.5'/>"
    )
    parts.append(
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' "
        "stroke='#334455' stroke-width='1.5'/>"
    )

    upper = [(x_pos(idx), y_pos(value + ci)) for idx, (value, ci) in enumerate(zip(smooth_mean, smooth_ci))]
    lower = [
        (x_pos(idx), y_pos(value - ci))
        for idx, (value, ci) in reversed(list(enumerate(zip(smooth_mean, smooth_ci))))
    ]
    polygon = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower) + " Z"
    parts.append("<path d='" + polygon + "' fill='#7FA7D8' fill-opacity='0.24' stroke='none'/>")

    for series in seed_series:
        path = "M " + " L ".join(
            f"{x_pos(idx):.1f},{y_pos(value):.1f}" for idx, value in enumerate(series)
        )
        parts.append(
            "<path d='" + path + "' fill='none' stroke='#B8C1CC' "
            "stroke-opacity='0.58' stroke-width='1.15'/>"
        )

    mean_path = "M " + " L ".join(
        f"{x_pos(idx):.1f},{y_pos(value):.1f}" for idx, value in enumerate(smooth_mean)
    )
    parts.append("<path d='" + mean_path + "' fill='none' stroke='#1F5AA6' stroke-width='3.1'/>")

    write_text(parts, width / 2, 30, "图4-4 5组随机种子的平均回合收益轨迹", size=18, weight="bold")
    write_text(parts, width / 2, height - 22, "训练迭代步", size=13)
    parts.append(
        f"<text x='28' y='{height / 2:.1f}' font-size='13' fill='#222222' text-anchor='middle' "
        "font-family='PingFang SC, Microsoft YaHei, Helvetica, Arial' "
        f"transform='rotate(-90 28 {height / 2:.1f})'>平均回合收益</text>"
    )

    parts.append("<rect x='784' y='84' width='258' height='62' rx='8' fill='white' fill-opacity='0.94' stroke='#D8E0EA'/>")
    parts.append("<line x1='804' y1='106' x2='846' y2='106' stroke='#1F5AA6' stroke-width='3'/>")
    write_text(parts, 918, 111, "5种子均值", size=12)
    parts.append("<rect x='804' y='119' width='42' height='12' fill='#7FA7D8' fill-opacity='0.24'/>")
    write_text(parts, 928, 130, "95%置信带", size=12)
    parts.append("</svg>")

    output_path = figure_root / "fig_4_4_seed_stability_formal.svg"
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def render_point_interval_svg(
    *,
    title: str,
    metrics: list[tuple[str, str]],
    variants: list[str],
    labels: list[str],
    colors: dict[str, str],
    point_lookup: dict[str, dict[str, float]],
    ci_lookup: dict[str, dict[str, tuple[float, float]]],
    output_path: Path,
) -> Path:
    width = 1280
    height = 828
    margin = 40
    panel_gap_x = 42
    panel_gap_y = 44
    panel_width = (width - margin * 2 - panel_gap_x) / 2
    panel_height = (height - 92 - margin - panel_gap_y) / 2

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
    ]
    write_text(parts, width / 2, 32, title, size=18, weight="bold")

    for metric_index, (metric_name, metric_title) in enumerate(metrics):
        row_idx = metric_index // 2
        col_idx = metric_index % 2
        x0 = margin + col_idx * (panel_width + panel_gap_x)
        y0 = 60 + row_idx * (panel_height + panel_gap_y)
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        parts.append(
            f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{panel_width:.1f}' height='{panel_height:.1f}' "
            "fill='white' stroke='#E3E9F0'/>"
        )
        write_text(parts, (x0 + x1) / 2, y0 + 24, metric_title, size=14, weight="bold")

        lows: list[float] = []
        highs: list[float] = []
        for variant in variants:
            point = point_lookup[variant][metric_name]
            ci = ci_lookup[variant].get(metric_name, (point, point))
            lows.append(min(ci[0], point))
            highs.append(max(ci[1], point))
        axis_min = min(lows)
        axis_max = max(highs)
        if axis_min == axis_max:
            axis_min -= 1.0
            axis_max += 1.0
        padding = 0.12 * (axis_max - axis_min)
        axis_min -= padding
        axis_max += padding

        plot_left = x0 + 122
        plot_right = x1 - 20
        plot_top = y0 + 46
        plot_bottom = y1 - 28
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        def px(value: float) -> float:
            return plot_left + (value - axis_min) / max(axis_max - axis_min, 1e-6) * plot_width

        for tick in range(5):
            tick_value = axis_min + (axis_max - axis_min) * tick / 4
            x = px(tick_value)
            parts.append(
                f"<line x1='{x:.1f}' y1='{plot_top}' x2='{x:.1f}' y2='{plot_bottom}' "
                "stroke='#E8EDF3' stroke-dasharray='4 4'/>"
            )
            label = f"{tick_value:.2f}" if abs(tick_value) < 10 else f"{tick_value:.0f}"
            write_text(parts, x, plot_bottom + 18, label, size=10, fill="#556677")

        for idx, (variant, label) in enumerate(zip(variants, labels)):
            y = plot_top + (idx + 0.5) * plot_height / len(labels)
            write_text(parts, plot_left - 10, y + 4, label, size=12, anchor="end")
            point = point_lookup[variant][metric_name]
            ci = ci_lookup[variant].get(metric_name, (point, point))
            parts.append(
                f"<line x1='{px(ci[0]):.1f}' y1='{y:.1f}' x2='{px(ci[1]):.1f}' y2='{y:.1f}' "
                f"stroke='{colors[variant]}' stroke-width='3' stroke-linecap='round'/>"
            )
            parts.append(f"<circle cx='{px(point):.1f}' cy='{y:.1f}' r='5.6' fill='{colors[variant]}'/>")
            label_text = f"{point:.3f}" if abs(point) < 10 else f"{point:.1f}"
            write_text(parts, px(point), y - 10, label_text, size=10, fill=colors[variant])

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def render_module3_baseline() -> Path:
    table_path = MODULE3_ROOT / "tables" / "tab_4_4_baseline_metrics.csv"
    rows = {
        row["variant_name"]: row
        for row in csv.DictReader(table_path.open(encoding="utf-8-sig", newline=""))
    }
    variants = ["SG-MAPPO", "Plain-MAPPO"]
    labels = ["SG-MAPPO", "Plain-MAPPO"]
    colors = {"SG-MAPPO": "#1F5AA6", "Plain-MAPPO": "#C95F3C"}
    point_lookup = {
        variant: {
            "defender_reward": float(rows[variant]["defender_reward"]),
            "signal_effect": float(rows[variant]["signal_effect"]),
            "real_host_attack_rate": float(rows[variant]["real_host_attack_rate"]),
            "honeypot_hit_rate": float(rows[variant]["honeypot_hit_rate"]),
        }
        for variant in variants
    }
    ci_lookup = {variant: {} for variant in variants}
    for variant in variants:
        eval_path = MODULE3_ROOT / "csv" / f"evaluation_baseline_{variant}.csv"
        for metric_name in ("signal_effect", "real_host_attack_rate", "honeypot_hit_rate"):
            _point, low, high = bootstrap_ci(eval_path, metric_name)
            ci_lookup[variant][metric_name] = (low, high)
        reward_value = point_lookup[variant]["defender_reward"]
        ci_lookup[variant]["defender_reward"] = (reward_value, reward_value)

    return render_point_interval_svg(
        title="图4-6 SG-MAPPO与Plain-MAPPO的点估计与评估区间",
        metrics=[
            ("defender_reward", "防御方平均回合收益"),
            ("signal_effect", "信号效应"),
            ("real_host_attack_rate", "真实主机攻击率"),
            ("honeypot_hit_rate", "蜜罐命中率"),
        ],
        variants=variants,
        labels=labels,
        colors=colors,
        point_lookup=point_lookup,
        ci_lookup=ci_lookup,
        output_path=MODULE3_ROOT / "figures" / "fig_4_6_baseline_compare_formal.svg",
    )


def render_module3_ablation() -> Path:
    table_path = MODULE3_ROOT / "tables" / "tab_4_5_ablation_metrics.csv"
    rows = {row["variant_name"]: row for row in csv.DictReader(table_path.open(newline=""))}
    variants = ["完整SG-MAPPO", "去除信念输入", "去除LSTM"]
    labels = ["SG-MAPPO", "No-Belief", "No-LSTM"]
    colors = {
        "完整SG-MAPPO": "#1F5AA6",
        "去除信念输入": "#9B5A2E",
        "去除LSTM": "#1E9BA7",
    }
    evaluation_file = {
        "完整SG-MAPPO": "evaluation_ablation_完整SG-MAPPO.csv",
        "去除信念输入": "evaluation_ablation_去除信念输入.csv",
        "去除LSTM": "evaluation_ablation_去除LSTM.csv",
    }
    point_lookup = {
        variant: {
            "defender_final_avg_reward": float(rows[variant]["defender_final_avg_reward"]),
            "real_host_attack_rate": float(rows[variant]["real_host_attack_rate"]),
            "deception_success_rate": float(rows[variant]["deception_success_rate"]),
            "signal_effect": float(rows[variant]["signal_effect"]),
        }
        for variant in variants
    }
    ci_lookup = {variant: {} for variant in variants}
    for variant in variants:
        eval_path = MODULE3_ROOT / "csv" / evaluation_file[variant]
        for metric_name in ("real_host_attack_rate", "deception_success_rate", "signal_effect"):
            _point, low, high = bootstrap_ci(eval_path, metric_name)
            ci_lookup[variant][metric_name] = (low, high)
        reward_value = point_lookup[variant]["defender_final_avg_reward"]
        ci_lookup[variant]["defender_final_avg_reward"] = (reward_value, reward_value)

    return render_point_interval_svg(
        title="图4-7 消融实验的点估计与评估区间",
        metrics=[
            ("defender_final_avg_reward", "防御方最终平均回报"),
            ("real_host_attack_rate", "真实主机攻击率"),
            ("deception_success_rate", "欺骗成功率"),
            ("signal_effect", "信号效应"),
        ],
        variants=variants,
        labels=labels,
        colors=colors,
        point_lookup=point_lookup,
        ci_lookup=ci_lookup,
        output_path=MODULE3_ROOT / "figures" / "fig_4_7_ablation_compare_formal.svg",
    )


def main() -> None:
    outputs = [
        render_module1_seed_stability(),
        render_module3_baseline(),
        render_module3_ablation(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
