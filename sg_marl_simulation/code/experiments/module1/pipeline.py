from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import math
import statistics
import time

import matplotlib.pyplot as plt
from matplotlib import rcParams
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import ray

from marl_core import (
    ATTACKER,
    DEFENDER,
    CHAPTER4_MODULE_IDS,
    ExperimentArtifacts,
    ExperimentConfig,
    SummaryWriter,
    compute_convergence_step,
    compute_reward_volatility,
    create_rllib_env,
    figure_name,
    policy_mapping_fn,
    register_chapter4_env,
    table_name,
)
from marl_core.io import EpisodeLogger, write_csv
from .config import (
    MODULE1_CONVERGENCE,
    MODULE1_DEFAULT_SEEDS,
    MODULE1_ENV,
    MODULE1_LEARNING_RATES,
    MODULE1_TRAINING,
)

rcParams["font.family"] = ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


@dataclass
class TrainingRunResult:
    episode_rows: List[dict]
    summary_row: dict
    checkpoint_dir: Optional[str]


def _rolling_mean(values: List[float], window: int = 15) -> List[float]:
    if not values:
        return []
    rolled: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start : idx + 1]
        rolled.append(sum(bucket) / len(bucket))
    return rolled


def _safe_metric(container: dict, *keys, default=float("nan")):
    current = container
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _extract_episode_row(result: dict, seed: int, learning_rate: float) -> dict:
    env_metrics = result.get("env_runners", {})
    learner = _safe_metric(result, "info", "learner", default={})
    return {
        "seed": seed,
        "learning_rate": learning_rate,
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
        "defender_vf_loss": _safe_metric(learner, "defender_policy", "learner_stats", "vf_loss"),
        "attacker_vf_loss": _safe_metric(learner, "attacker_policy", "learner_stats", "vf_loss"),
    }


def _final_window_mean(rows: List[dict], key: str, final_window: int) -> float:
    values = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
    if not values:
        return float("nan")
    window = values[-min(final_window, len(values)) :]
    return statistics.mean(window)


def _metric_values(rows: List[dict], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row.get(key) == row.get(key)]


def _run_single_training(
    *,
    seed: int,
    learning_rate: float,
    run_name: str,
    batch_run_name: str,
    env_overrides: Optional[dict] = None,
    training_overrides: Optional[dict] = None,
    artifacts: ExperimentArtifacts,
) -> TrainingRunResult:
    env_cfg = dict(MODULE1_ENV)
    if env_overrides:
        if "payoffs" in env_overrides:
            env_cfg["payoffs"] = dict(env_cfg["payoffs"])
            env_cfg["payoffs"].update(env_overrides["payoffs"])
        for key, value in env_overrides.items():
            if key != "payoffs":
                env_cfg[key] = value

    train_cfg = dict(MODULE1_TRAINING)
    if training_overrides:
        train_cfg.update(training_overrides)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=train_cfg["num_gpus"])

    env_name = register_chapter4_env()
    temp_env = create_rllib_env(env_cfg)
    obs_spaces = temp_env.observation_space
    act_spaces = temp_env.action_space
    temp_env.close()

    config = (
        PPOConfig()
        .resources(num_gpus=train_cfg["num_gpus"])
        .environment(env=env_name, env_config=env_cfg)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={
                "defender_policy": PolicySpec(
                    observation_space=obs_spaces[DEFENDER],
                    action_space=act_spaces[DEFENDER],
                    config={"agent_id": DEFENDER, "lr": learning_rate},
                ),
                "attacker_policy": PolicySpec(
                    observation_space=obs_spaces[ATTACKER],
                    action_space=act_spaces[ATTACKER],
                    config={"agent_id": ATTACKER, "lr": learning_rate},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["defender_policy", "attacker_policy"],
        )
        .training(
            model={
                "fcnet_hiddens": train_cfg["actor_critic_hidden_dims"],
                "use_lstm": True,
                "lstm_cell_size": train_cfg["lstm_cell_size"],
            },
            gamma=env_cfg["discount_gamma"],
            lambda_=train_cfg["gae_lambda"],
            num_sgd_iter=train_cfg["num_sgd_iter"],
            clip_param=train_cfg["clip_param"],
            lr=learning_rate,
            train_batch_size=train_cfg["batch_size"],
            minibatch_size=min(128, train_cfg["batch_size"]),
            entropy_coeff=0.005,
            vf_loss_coeff=1.0,
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
    algo = config.build()

    checkpoint_dir = artifacts.log_path(f"{batch_run_name}_{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episode_log = EpisodeLogger(
        artifacts.csv_path(f"episode_log_{batch_run_name}_seed{seed}_{run_name}.csv"),
        fieldnames=[
            "seed",
            "learning_rate",
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
            "defender_vf_loss",
            "attacker_vf_loss",
        ],
    )

    episode_rows: List[dict] = []
    start_time = time.time()
    for _ in range(train_cfg["episodes"]):
        result = algo.train()
        row = _extract_episode_row(result, seed=seed, learning_rate=learning_rate)
        episode_rows.append(row)
        episode_log.log(row)

    checkpoint = algo.save(str(checkpoint_dir))
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()

    episode_rewards = _metric_values(episode_rows, "episode_reward_mean")
    defender_rewards = _metric_values(episode_rows, "defender_reward")
    attacker_rewards = _metric_values(episode_rows, "attacker_reward")

    convergence_step = compute_convergence_step(
        episode_rewards,
        threshold_ratio=MODULE1_CONVERGENCE["threshold_ratio"],
        stability_window=MODULE1_CONVERGENCE["stability_window"],
        final_window=MODULE1_CONVERGENCE["final_window"],
    )
    summary_row = {
        "run_name": run_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "mean_reward": statistics.mean(episode_rewards),
        "std_reward": compute_reward_volatility(episode_rewards),
        "mean_defender_reward": statistics.mean(defender_rewards),
        "std_defender_reward": compute_reward_volatility(defender_rewards),
        "mean_attacker_reward": statistics.mean(attacker_rewards),
        "std_attacker_reward": compute_reward_volatility(attacker_rewards),
        "final_avg_reward": _final_window_mean(
            episode_rows, "episode_reward_mean", MODULE1_CONVERGENCE["final_window"]
        ),
        "convergence_step": convergence_step,
        "training_time_s": time.time() - start_time,
        "reward_volatility": compute_reward_volatility(episode_rewards),
        "checkpoint_path": checkpoint.checkpoint.path if hasattr(checkpoint, "checkpoint") else str(checkpoint),
    }
    return TrainingRunResult(episode_rows=episode_rows, summary_row=summary_row, checkpoint_dir=str(checkpoint_dir))


def _plot_learning_rate_curves(rows_by_lr: Dict[float, List[dict]], artifacts: ExperimentArtifacts) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ordered = sorted(rows_by_lr.items(), key=lambda item: item[0], reverse=True)
    lr_to_color = {
        2e-4: "#5DA5DA",   # blue
        1e-4: "#F28E2B",   # orange
        7e-5: "#60BD68",   # green
        5e-5: "#E15759",   # red
    }
    for learning_rate, rows in ordered:
        color = lr_to_color.get(learning_rate, "#4E79A7")
        label = f"lr={learning_rate:.1e}"
        x = [row["training_iteration"] for row in rows]
        defender_loss = _rolling_mean(_metric_values(rows, "defender_policy_loss"), window=12)
        attacker_loss = _rolling_mean(_metric_values(rows, "attacker_policy_loss"), window=12)
        defender_reward = _rolling_mean(_metric_values(rows, "defender_reward"), window=12)
        attacker_reward = _rolling_mean(_metric_values(rows, "attacker_reward"), window=12)
        axes[0, 0].plot(x, defender_loss, label=label, color=color, linewidth=2.0)
        axes[0, 1].plot(x, attacker_loss, label=label, color=color, linewidth=2.0)
        axes[1, 0].plot(x, defender_reward, label=label, color=color, linewidth=2.0)
        axes[1, 1].plot(x, attacker_reward, label=label, color=color, linewidth=2.0)

    axes[0, 0].set_title("防御方策略损失")
    axes[0, 1].set_title("攻击方策略损失")
    axes[1, 0].set_title("防御方回合收益")
    axes[1, 1].set_title("攻击方回合收益")
    for axis in axes.flat:
        axis.set_xlabel("训练迭代次数")
        axis.grid(True, alpha=0.25, linestyle="--")
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("损失值")
    axes[1, 0].set_ylabel("收益值")
    plt.tight_layout()
    png = artifacts.figure_path(figure_name("4_3", "learning_rate", "png"))
    pdf = artifacts.figure_path(figure_name("4_3", "learning_rate", "pdf"))
    plt.savefig(png, dpi=artifacts.config.output.image_dpi)
    plt.savefig(pdf)
    plt.close(fig)


def _plot_seed_stability(seed_rows: Dict[int, List[dict]], artifacts: ExperimentArtifacts) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = sorted(seed_rows.items(), key=lambda item: item[0])
    max_len = max(len(rows) for _, rows in ordered)
    defender_matrix: List[List[float]] = []

    for seed, rows in ordered:
        values = _metric_values(rows, "episode_reward_mean")
        defender_matrix.append(values)
        ax.plot(
            range(1, len(values) + 1),
            _rolling_mean(values, window=15),
            alpha=0.18,
            linewidth=1.1,
            color="#7A8CA5",
            label=f"seed={seed}",
        )

    mean_curve = []
    ci_curve = []
    for idx in range(max_len):
        bucket = [curve[idx] for curve in defender_matrix if idx < len(curve)]
        mean_curve.append(statistics.mean(bucket))
        if len(bucket) > 1:
            std = statistics.stdev(bucket)
            ci_curve.append(1.96 * std / math.sqrt(len(bucket)))
        else:
            ci_curve.append(0.0)

    x = list(range(1, max_len + 1))
    smoothed_mean = _rolling_mean(mean_curve, window=15)
    smoothed_ci = _rolling_mean(ci_curve, window=15)
    ax.plot(x, smoothed_mean, color="#1F4E79", linewidth=2.8, label="均值")
    lower = [mean - ci for mean, ci in zip(smoothed_mean, smoothed_ci)]
    upper = [mean + ci for mean, ci in zip(smoothed_mean, smoothed_ci)]
    ax.fill_between(x, lower, upper, color="#9BB8D3", alpha=0.28)
    ax.set_title("5次随机种子训练稳定性")
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("平均回合收益")
    ax.grid(True, alpha=0.25, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l.startswith("seed=")]
    if filtered:
        ax.legend(
            [item[0] for item in filtered],
            [item[1] for item in filtered],
            frameon=False,
            loc="upper left",
        )
    plt.tight_layout()
    png = artifacts.figure_path(figure_name("4_4", "seed_stability", "png"))
    pdf = artifacts.figure_path(figure_name("4_4", "seed_stability", "pdf"))
    plt.savefig(png, dpi=artifacts.config.output.image_dpi)
    plt.savefig(pdf)
    plt.close(fig)


def _aggregate_seed_summaries(rows: List[dict], learning_rate: float) -> dict:
    aggregate_rows = [row for row in rows if row.get("experiment_group") == "seed_stability"]
    return {
        "run_name": "seed_stability_aggregate",
        "seed": "aggregate",
        "learning_rate": learning_rate,
        "mean_reward": statistics.mean(float(row["mean_reward"]) for row in aggregate_rows),
        "std_reward": statistics.mean(float(row["std_reward"]) for row in aggregate_rows),
        "mean_defender_reward": statistics.mean(float(row["mean_defender_reward"]) for row in aggregate_rows),
        "std_defender_reward": statistics.mean(float(row["std_defender_reward"]) for row in aggregate_rows),
        "mean_attacker_reward": statistics.mean(float(row["mean_attacker_reward"]) for row in aggregate_rows),
        "std_attacker_reward": statistics.mean(float(row["std_attacker_reward"]) for row in aggregate_rows),
        "final_avg_reward": statistics.mean(float(row["final_avg_reward"]) for row in aggregate_rows),
        "convergence_step": statistics.mean(
            float(row["convergence_step"]) for row in aggregate_rows if row.get("convergence_step") not in ("", None)
        ),
        "training_time_s": statistics.mean(float(row["training_time_s"]) for row in aggregate_rows),
        "reward_volatility": statistics.mean(float(row["reward_volatility"]) for row in aggregate_rows),
        "checkpoint_path": "",
        "experiment_group": "seed_stability_aggregate",
    }


def run_module1(
    *,
    run_name: str = "module1_default_protocol",
    learning_rates: Optional[Iterable[float]] = None,
    stability_seeds: Optional[Iterable[int]] = None,
    stability_learning_rate: float = 5e-4,
    env_overrides: Optional[dict] = None,
    training_overrides: Optional[dict] = None,
) -> dict:
    module_id = CHAPTER4_MODULE_IDS["module1"]
    config = ExperimentConfig(
        module_id=module_id,
        run_name=run_name,
        description="Chapter 4 module 1: learning-rate sensitivity and 5-seed stability.",
        seeds=list(stability_seeds or MODULE1_DEFAULT_SEEDS),
        environment={
            **MODULE1_ENV,
            **(env_overrides or {}),
            "payoffs": {
                **MODULE1_ENV["payoffs"],
                **((env_overrides or {}).get("payoffs", {})),
            },
        },
        training={**MODULE1_TRAINING, **(training_overrides or {})},
        analysis={
            "learning_rates": list(learning_rates or MODULE1_LEARNING_RATES),
            "stability_learning_rate": stability_learning_rate,
            "convergence_rule": dict(MODULE1_CONVERGENCE),
        },
        tags=["chapter4", "module1", "convergence", "stability"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    summary = SummaryWriter()
    results_summary_rows: List[dict] = []

    lr_runs: Dict[float, List[dict]] = {}
    for learning_rate in learning_rates or MODULE1_LEARNING_RATES:
        run = _run_single_training(
            seed=0,
            learning_rate=learning_rate,
            run_name=f"lr_sweep_{learning_rate:.1e}",
            batch_run_name=run_name,
            env_overrides=env_overrides,
            training_overrides=training_overrides,
            artifacts=artifacts,
        )
        lr_runs[learning_rate] = run.episode_rows
        summary.add({**run.summary_row, "experiment_group": "learning_rate_sweep"})
        results_summary_rows.append({**run.summary_row, "experiment_group": "learning_rate_sweep"})

    seed_runs: Dict[int, List[dict]] = {}
    for seed in stability_seeds or MODULE1_DEFAULT_SEEDS:
        run = _run_single_training(
            seed=seed,
            learning_rate=stability_learning_rate,
            run_name=f"seed_stability_{seed}",
            batch_run_name=run_name,
            env_overrides=env_overrides,
            training_overrides=training_overrides,
            artifacts=artifacts,
        )
        seed_runs[seed] = run.episode_rows
        summary.add({**run.summary_row, "experiment_group": "seed_stability"})
        results_summary_rows.append({**run.summary_row, "experiment_group": "seed_stability"})

    if seed_runs:
        aggregate_row = _aggregate_seed_summaries(results_summary_rows, stability_learning_rate)
        summary.add(aggregate_row)
        results_summary_rows.append(aggregate_row)

    summary.write(artifacts.root / "results_summary.csv")
    write_csv(artifacts.table_path(table_name("4_2", "convergence_stats")), results_summary_rows)
    _plot_learning_rate_curves(lr_runs, artifacts)
    _plot_seed_stability(seed_runs, artifacts)

    return {
        "module_root": str(artifacts.root),
        "summary_csv": str(artifacts.root / "results_summary.csv"),
        "table_csv": str(artifacts.table_path(table_name("4_2", "convergence_stats"))),
    }
