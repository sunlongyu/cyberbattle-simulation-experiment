from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParams
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from marl_core import (
    ATTACKER,
    DEFENDER,
    CHAPTER4_MODULE_IDS,
    ExperimentArtifacts,
    ExperimentConfig,
    compute_convergence_step,
    compute_reward_volatility,
    create_rllib_env,
    figure_name,
    policy_mapping_fn,
    register_chapter4_env,
    table_name,
)
from marl_core.io import EpisodeLogger, SummaryWriter, write_csv
from experiments.module2.pipeline import evaluate_checkpoint

from .config import (
    MODULE3_ABLATIONS,
    MODULE3_ABLATION_ENV,
    MODULE3_ABLATION_EVAL_EPISODES,
    MODULE3_ABLATION_FINAL_WINDOW,
    MODULE3_ABLATION_TRAINING,
    MODULE3_BASELINE_ENV,
    MODULE3_BASELINE_EVAL_EPISODES,
    MODULE3_BASELINE_TRAINING,
    MODULE3_BASELINES,
    MODULE3_CONVERGENCE,
    MODULE3_DEFAULT_SEED,
    MODULE3_EVAL_EPISODES,
    MODULE3_ENV,
    MODULE3_TRAINING,
)

rcParams["font.family"] = ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


@dataclass
class Module3RunResult:
    variant_name: str
    family: str
    checkpoint_dir: str
    episode_rows: List[dict]
    summary_row: dict


def _slugify(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def _merge_env(overrides: Optional[dict] = None) -> dict:
    env_cfg = dict(MODULE3_ENV)
    env_cfg["payoffs"] = dict(MODULE3_ENV["payoffs"])
    if overrides:
        if "payoffs" in overrides:
            env_cfg["payoffs"].update(overrides["payoffs"])
        for key, value in overrides.items():
            if key != "payoffs":
                env_cfg[key] = value
    return env_cfg


def _merge_training(overrides: Optional[dict] = None) -> dict:
    cfg = dict(MODULE3_TRAINING)
    if overrides:
        cfg.update(overrides)
    return cfg


def _safe_metric(container: dict, *keys, default=float("nan")):
    current = container
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _extract_episode_row(result: dict, variant_name: str) -> dict:
    env_metrics = result.get("env_runners", {})
    learner = _safe_metric(result, "info", "learner", default={})
    return {
        "variant_name": variant_name,
        "training_iteration": result.get("training_iteration"),
        "time_total_s": result.get("time_total_s"),
        "timesteps_total": result.get("timesteps_total"),
        "episode_reward_mean": env_metrics.get("episode_reward_mean"),
        "episode_len_mean": env_metrics.get("episode_len_mean"),
        "defender_reward": _safe_metric(env_metrics, "policy_reward_mean", "defender_policy"),
        "attacker_reward": _safe_metric(env_metrics, "policy_reward_mean", "attacker_policy"),
        "defender_policy_loss": _safe_metric(
            learner, "defender_policy", "learner_stats", "policy_loss"
        ),
        "attacker_policy_loss": _safe_metric(
            learner, "attacker_policy", "learner_stats", "policy_loss"
        ),
        "defender_entropy": _safe_metric(learner, "defender_policy", "learner_stats", "entropy"),
        "attacker_entropy": _safe_metric(learner, "attacker_policy", "learner_stats", "entropy"),
    }


def _metric_values(rows: List[dict], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row.get(key) == row.get(key)]


def _rolling_mean(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        smoothed.append(statistics.mean(segment))
    return smoothed


def _rolling_std(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    deviations: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        deviations.append(statistics.pstdev(segment) if len(segment) > 1 else 0.0)
    return deviations


def _final_window_mean(rows: List[dict], key: str) -> float:
    values = _metric_values(rows, key)
    if not values:
        return float("nan")
    final_window = MODULE3_CONVERGENCE["final_window"]
    return statistics.mean(values[-min(final_window, len(values)) :])


def _final_window_mean_with_size(rows: List[dict], key: str, final_window: int) -> float:
    values = _metric_values(rows, key)
    if not values:
        return float("nan")
    return statistics.mean(values[-min(final_window, len(values)) :])


def _build_algo(env_cfg: dict, train_cfg: dict, seed: int):
    env_name = register_chapter4_env()
    temp_env = create_rllib_env(env_cfg)
    obs_spaces = temp_env.observation_space
    act_spaces = temp_env.action_space
    temp_env.close()

    use_lstm = bool(train_cfg.get("use_lstm", True))
    model_cfg = {
        "fcnet_hiddens": train_cfg["actor_critic_hidden_dims"],
        "use_lstm": use_lstm,
    }
    if use_lstm:
        model_cfg["lstm_cell_size"] = train_cfg["lstm_cell_size"]

    config = (
        PPOConfig()
        .resources(num_gpus=train_cfg["num_gpus"])
        .environment(env=env_name, env_config=env_cfg)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={
                "defender_policy": PolicySpec(
                    observation_space=obs_spaces[DEFENDER],
                    action_space=act_spaces[DEFENDER],
                    config={"agent_id": DEFENDER},
                ),
                "attacker_policy": PolicySpec(
                    observation_space=obs_spaces[ATTACKER],
                    action_space=act_spaces[ATTACKER],
                    config={"agent_id": ATTACKER},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["defender_policy", "attacker_policy"],
        )
        .training(
            model=model_cfg,
            gamma=env_cfg["discount_gamma"],
            lambda_=train_cfg["gae_lambda"],
            num_sgd_iter=train_cfg["num_sgd_iter"],
            clip_param=train_cfg["clip_param"],
            lr=train_cfg["learning_rate"],
            train_batch_size=train_cfg["batch_size"],
            minibatch_size=min(128, train_cfg["batch_size"]),
            entropy_coeff=train_cfg["entropy_coeff"],
            vf_loss_coeff=train_cfg["vf_loss_coeff"],
            vf_clip_param=10.0,
        )
        .env_runners(
            num_env_runners=train_cfg["num_env_runners"],
            batch_mode="truncate_episodes",
            rollout_fragment_length=train_cfg["rollout_fragment_length"],
        )
        .debugging(log_level="ERROR")
    )
    config.seed = seed
    return config.build()


def run_variant(
    *,
    family: str,
    variant_name: str,
    env_overrides: Optional[dict],
    training_overrides: Optional[dict],
    artifacts: ExperimentArtifacts,
    seed: int = MODULE3_DEFAULT_SEED,
    evaluation_episodes: int = MODULE3_EVAL_EPISODES,
) -> Module3RunResult:
    env_cfg = _merge_env(env_overrides)
    train_cfg = _merge_training(training_overrides)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=train_cfg["num_gpus"])

    algo = _build_algo(env_cfg, train_cfg, seed)
    log_name = _slugify(f"{family}_{variant_name}")
    checkpoint_dir = artifacts.log_path(log_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(
        artifacts.csv_path(f"episode_log_{log_name}.csv"),
        fieldnames=[
            "variant_name",
            "training_iteration",
            "time_total_s",
            "timesteps_total",
            "episode_reward_mean",
            "episode_len_mean",
            "defender_reward",
            "attacker_reward",
            "defender_policy_loss",
            "attacker_policy_loss",
            "defender_entropy",
            "attacker_entropy",
        ],
    )

    episode_rows: List[dict] = []
    start_time = time.time()
    for _ in range(train_cfg["episodes"]):
        result = algo.train()
        row = _extract_episode_row(result, variant_name)
        episode_rows.append(row)
        logger.log(row)

    checkpoint = algo.save(str(checkpoint_dir))
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()

    reward_series = _metric_values(episode_rows, "episode_reward_mean")
    defender_series = _metric_values(episode_rows, "defender_reward")
    attacker_series = _metric_values(episode_rows, "attacker_reward")
    defender_entropy = _final_window_mean(episode_rows, "defender_entropy")
    attacker_entropy = _final_window_mean(episode_rows, "attacker_entropy")

    eval_rows, eval_metrics = evaluate_checkpoint(
        checkpoint_dir=checkpoint.checkpoint.path if hasattr(checkpoint, "checkpoint") else str(checkpoint),
        env_overrides=env_overrides,
        n_eval_episodes=evaluation_episodes,
        seed=201,
    )
    write_csv(artifacts.csv_path(f"evaluation_{log_name}.csv"), eval_rows)

    summary_row = {
        "family": family,
        "variant_name": variant_name,
        "learning_rate": train_cfg["learning_rate"],
        "episodes": train_cfg["episodes"],
        "use_lstm": bool(train_cfg.get("use_lstm", True)),
        "clip_param": train_cfg["clip_param"],
        "num_sgd_iter": train_cfg["num_sgd_iter"],
        "disable_belief_input": bool(env_cfg.get("disable_belief_input", False)),
        "mean_reward": statistics.mean(reward_series),
        "final_avg_reward": _final_window_mean(episode_rows, "episode_reward_mean"),
        "defender_reward": statistics.mean(defender_series),
        "defender_final_avg_reward": _final_window_mean_with_size(
            episode_rows, "defender_reward", MODULE3_ABLATION_FINAL_WINDOW
        ),
        "attacker_reward": statistics.mean(attacker_series),
        "policy_entropy": 0.5 * (defender_entropy + attacker_entropy),
        "convergence_step": compute_convergence_step(
            reward_series,
            threshold_ratio=MODULE3_CONVERGENCE["threshold_ratio"],
            stability_window=MODULE3_CONVERGENCE["stability_window"],
            final_window=MODULE3_CONVERGENCE["final_window"],
        ),
        "training_time_s": time.time() - start_time,
        "reward_volatility": compute_reward_volatility(reward_series),
        **eval_metrics,
        "checkpoint_path": checkpoint.checkpoint.path if hasattr(checkpoint, "checkpoint") else str(checkpoint),
    }
    return Module3RunResult(
        variant_name=variant_name,
        family=family,
        checkpoint_dir=summary_row["checkpoint_path"],
        episode_rows=episode_rows,
        summary_row=summary_row,
    )


def _plot_comparison(rows: List[dict], title: str, filename_stub: str, artifacts: ExperimentArtifacts) -> None:
    metric_names = [
        ("final_avg_reward", "最终平均回报"),
        ("defender_reward", "防御方平均回报"),
        ("attacker_reward", "攻击方平均回报"),
        ("real_host_attack_rate", "真实主机攻击率"),
        ("honeypot_hit_rate", "蜜罐命中率"),
        ("policy_entropy", "策略熵"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    labels = [row["variant_name"] for row in rows]
    for axis, (metric_key, metric_label) in zip(axes, metric_names):
        values = [row.get(metric_key, float("nan")) for row in rows]
        bars = axis.bar(labels, values, color=["#0B6E4F", "#2C7DA0", "#C1666B", "#B08968"][: len(labels)])
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.25, linestyle="--")
        for bar, value in zip(bars, values):
            if value == value:
                axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(artifacts.figure_path(figure_name("4_6" if filename_stub == "baseline_compare" else "4_7", filename_stub, "png")), dpi=300, bbox_inches="tight")
    plt.savefig(artifacts.figure_path(figure_name("4_6" if filename_stub == "baseline_compare" else "4_7", filename_stub, "pdf")), bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_trajectory_and_cost(
    results: List[Module3RunResult],
    artifacts: ExperimentArtifacts,
) -> None:
    focus_variants = ["SG-MAPPO", "Plain-MAPPO"]
    results = [result for result in results if result.variant_name in focus_variants]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6))
    axes = axes.flatten()
    palette = {
        "SG-MAPPO": "#4E79A7",
        "SG-MATRPO": "#F28E2B",
        "SG-MAA2C": "#60BD68",
        "Plain-MAPPO": "#D3725B",
    }
    metric_keys = [
        ("defender_reward", "防御方平均回合收益"),
        ("signal_effect", "信号效应"),
        ("real_host_attack_rate", "真实主机攻击率"),
        ("honeypot_hit_rate", "蜜罐命中率"),
    ]
    labels = [result.variant_name for result in results]
    colors = [palette.get(label, "#999999") for label in labels]
    for axis, (metric_key, metric_label) in zip(axes, metric_keys):
        values = [float(result.summary_row[metric_key]) for result in results]
        bars = axis.bar(labels, values, color=colors, width=0.34)
        axis.set_title(metric_label)
        axis.grid(True, axis="y", alpha=0.22, linestyle="--")
        axis.tick_params(axis="x", rotation=0)
        axis.margins(x=0.18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}" if abs(value) < 100 else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        artifacts.figure_path(figure_name("4_6", "baseline_compare", "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        artifacts.figure_path(figure_name("4_6", "baseline_compare", "pdf")),
        bbox_inches="tight",
    )
    plt.close(fig)


def replot_module3_baseline_from_logs() -> dict:
    module_id = CHAPTER4_MODULE_IDS["module3"]
    config = ExperimentConfig(
        module_id=module_id,
        run_name="module3_baseline_refined_v1",
        description="Regenerated baseline plots from existing logs.",
        seeds=[MODULE3_DEFAULT_SEED],
        environment=MODULE3_BASELINE_ENV,
        training=MODULE3_BASELINE_TRAINING,
        analysis={"baselines": list(MODULE3_BASELINES.keys())},
        tags=["chapter4", "module3", "baseline", "replot"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    baseline_table = artifacts.table_path(table_name("4_4", "baseline_metrics"))
    summary_lookup: Dict[str, dict] = {}
    with baseline_table.open("r", encoding="utf-8-sig", newline="") as handle:
        import csv

        def _coerce_csv_value(key: str, value: str):
            if key in {"variant_name", "family", "checkpoint_dir"}:
                return value
            if value in ("", "nan", "NaN"):
                return float("nan")
            if value in {"True", "False"}:
                return value == "True"
            try:
                return float(value)
            except ValueError:
                return value

        for row in csv.DictReader(handle):
            summary_lookup[row["variant_name"]] = {
                key: _coerce_csv_value(key, value) for key, value in row.items()
            }

    baseline_results: List[Module3RunResult] = []
    for variant_name in MODULE3_BASELINES.keys():
        csv_path = artifacts.csv_path(f"episode_log_baseline_{variant_name}.csv")
        episode_rows: List[dict] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            import csv

            for row in csv.DictReader(handle):
                episode_rows.append(
                    {
                        key: (
                            float(value)
                            if key != "variant_name" and value not in ("", "nan", "NaN")
                            else value
                        )
                        for key, value in row.items()
                    }
                )
        baseline_results.append(
            Module3RunResult(
                variant_name=variant_name,
                family="baseline",
                checkpoint_dir=str(artifacts.log_path(f"baseline_{variant_name}")),
                episode_rows=episode_rows,
                summary_row=summary_lookup[variant_name],
            )
        )

    _plot_baseline_trajectory_and_cost(baseline_results, artifacts)
    return {
        "baseline_table": str(baseline_table),
        "baseline_figure": str(artifacts.figure_path(figure_name("4_6", "baseline_compare", "png"))),
    }


def _plot_ablation_compare(rows: List[dict], artifacts: ExperimentArtifacts) -> None:
    metric_names = [
        ("defender_final_avg_reward", "防御方最终平均回报"),
        ("real_host_attack_rate", "真实主机攻击率"),
        ("deception_success_rate", "欺骗成功率"),
        ("signal_effect", "信号效应"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6))
    axes = axes.flatten()
    label_map = {"完整SG-MAPPO": "SG-MAPPO"}
    labels = [label_map.get(row["variant_name"], row["variant_name"]) for row in rows]
    colors = ["#0066CC", "#8B4513", "#00AADE"]

    for axis, (metric_key, metric_label) in zip(axes, metric_names):
        values = [row.get(metric_key, float("nan")) for row in rows]
        bars = axis.bar(labels, values, color=colors[: len(labels)], width=0.34)
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(True, axis="y", alpha=0.22, linestyle="--")
        axis.margins(x=0.18)
        for bar, value in zip(bars, values):
            if value == value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(
        artifacts.figure_path(figure_name("4_7", "ablation_compare", "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        artifacts.figure_path(figure_name("4_7", "ablation_compare", "pdf")),
        bbox_inches="tight",
    )
    plt.close(fig)


def run_module3_ablation_refined(run_name: str = "module3_ablation_refined_v1") -> dict:
    module_id = CHAPTER4_MODULE_IDS["module3"]
    config = ExperimentConfig(
        module_id=module_id,
        run_name=run_name,
        description="Chapter 4 module 3 refined ablation study focused on belief and temporal modeling.",
        seeds=[MODULE3_DEFAULT_SEED],
        environment=MODULE3_ABLATION_ENV,
        training=MODULE3_ABLATION_TRAINING,
        analysis={
            "ablations": list(MODULE3_ABLATIONS.keys()),
            "evaluation_episodes": MODULE3_ABLATION_EVAL_EPISODES,
            "final_window": MODULE3_ABLATION_FINAL_WINDOW,
        },
        tags=["chapter4", "module3", "ablation", "refined"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    ablation_rows: List[dict] = []
    for variant_name, variant_cfg in MODULE3_ABLATIONS.items():
        training_overrides = dict(MODULE3_ABLATION_TRAINING)
        training_overrides.update(variant_cfg["training"])
        result = run_variant(
            family="ablation",
            variant_name=variant_name,
            env_overrides={**MODULE3_ABLATION_ENV, **variant_cfg["env"]},
            training_overrides=training_overrides,
            artifacts=artifacts,
            evaluation_episodes=MODULE3_ABLATION_EVAL_EPISODES,
        )
        ablation_rows.append(result.summary_row)

    write_csv(artifacts.root / "results_summary.csv", ablation_rows)
    write_csv(artifacts.table_path(table_name("4_5", "ablation_metrics")), ablation_rows)
    _plot_ablation_compare(ablation_rows, artifacts)
    return {
        "module_root": str(artifacts.root),
        "results_summary": str(artifacts.root / "results_summary.csv"),
        "ablation_table": str(artifacts.table_path(table_name("4_5", "ablation_metrics"))),
    }


def run_module3_baseline_refined(run_name: str = "module3_baseline_refined_v1") -> dict:
    module_id = CHAPTER4_MODULE_IDS["module3"]
    config = ExperimentConfig(
        module_id=module_id,
        run_name=run_name,
        description="Chapter 4 module 3 refined baseline comparison under a longer-horizon setting.",
        seeds=[MODULE3_DEFAULT_SEED],
        environment=MODULE3_BASELINE_ENV,
        training=MODULE3_BASELINE_TRAINING,
        analysis={
            "baselines": list(MODULE3_BASELINES.keys()),
            "evaluation_episodes": MODULE3_BASELINE_EVAL_EPISODES,
        },
        tags=["chapter4", "module3", "baseline", "refined"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    baseline_rows: List[dict] = []
    baseline_results: List[Module3RunResult] = []
    for variant_name, variant_cfg in MODULE3_BASELINES.items():
        training_overrides = dict(MODULE3_BASELINE_TRAINING)
        training_overrides.update(variant_cfg["training"])
        result = run_variant(
            family="baseline",
            variant_name=variant_name,
            env_overrides={**MODULE3_BASELINE_ENV, **variant_cfg["env"]},
            training_overrides=training_overrides,
            artifacts=artifacts,
            evaluation_episodes=MODULE3_BASELINE_EVAL_EPISODES,
        )
        baseline_rows.append(result.summary_row)
        baseline_results.append(result)

    write_csv(artifacts.root / "results_summary.csv", baseline_rows)
    write_csv(artifacts.table_path(table_name("4_4", "baseline_metrics")), baseline_rows)
    _plot_baseline_trajectory_and_cost(baseline_results, artifacts)
    return {
        "module_root": str(artifacts.root),
        "results_summary": str(artifacts.root / "results_summary.csv"),
        "baseline_table": str(artifacts.table_path(table_name("4_4", "baseline_metrics"))),
    }


def run_module3(run_name: str = "module3_formal_v1") -> dict:
    module_id = CHAPTER4_MODULE_IDS["module3"]
    config = ExperimentConfig(
        module_id=module_id,
        run_name=run_name,
        description="Chapter 4 module 3 baseline comparison and ablation study.",
        seeds=[MODULE3_DEFAULT_SEED],
        environment=MODULE3_ENV,
        training=MODULE3_TRAINING,
        analysis={
            "baselines": list(MODULE3_BASELINES.keys()),
            "ablations": list(MODULE3_ABLATIONS.keys()),
            "evaluation_episodes": MODULE3_EVAL_EPISODES,
        },
        tags=["chapter4", "module3", "effectiveness", "ablation"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    summary = SummaryWriter()
    baseline_rows: List[dict] = []
    ablation_rows: List[dict] = []
    baseline_results: List[Module3RunResult] = []

    for variant_name, variant_cfg in MODULE3_BASELINES.items():
        result = run_variant(
            family="baseline",
            variant_name=variant_name,
            env_overrides=variant_cfg["env"],
            training_overrides=variant_cfg["training"],
            artifacts=artifacts,
        )
        summary.add(result.summary_row)
        baseline_rows.append(result.summary_row)
        baseline_results.append(result)

    for variant_name, variant_cfg in MODULE3_ABLATIONS.items():
        result = run_variant(
            family="ablation",
            variant_name=variant_name,
            env_overrides=variant_cfg["env"],
            training_overrides=variant_cfg["training"],
            artifacts=artifacts,
        )
        summary.add(result.summary_row)
        ablation_rows.append(result.summary_row)

    summary.write(artifacts.root / "results_summary.csv")
    write_csv(artifacts.table_path(table_name("4_4", "baseline_metrics")), baseline_rows)
    write_csv(artifacts.table_path(table_name("4_5", "ablation_metrics")), ablation_rows)
    _plot_baseline_trajectory_and_cost(baseline_results, artifacts)
    _plot_ablation_compare(ablation_rows, artifacts)
    return {
        "module_root": str(artifacts.root),
        "results_summary": str(artifacts.root / "results_summary.csv"),
        "baseline_table": str(artifacts.table_path(table_name("4_4", "baseline_metrics"))),
        "ablation_table": str(artifacts.table_path(table_name("4_5", "ablation_metrics"))),
    }
