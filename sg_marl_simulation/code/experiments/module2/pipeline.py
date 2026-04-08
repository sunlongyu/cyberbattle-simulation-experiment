from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv
import statistics
import time

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec

from marl_core import (
    ATTACKER,
    DEFENDER,
    CHAPTER4_MODULE_IDS,
    ExperimentArtifacts,
    ExperimentConfig,
    create_rllib_env,
    figure_name,
    policy_mapping_fn,
    register_chapter4_env,
    table_name,
)
from marl_core.chapter4_env import ACTION_ATTACK, ACTION_RETREAT, SIGNAL_HONEYPOT, SIGNAL_NORMAL, THETA_HONEYPOT, THETA_REAL
from marl_core.io import EpisodeLogger, write_csv
from .config import MODULE2_BASE_ENV, MODULE2_BASE_TRAINING, MODULE2_PAYOFFS, MODULE2_SWEEPS

rcParams["font.family"] = ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5


@dataclass
class TrainingArtifact:
    checkpoint_dir: str
    episode_log_csv: str
    training_time_s: float


def _merge_env(overrides: Optional[dict] = None) -> dict:
    env_cfg = dict(MODULE2_BASE_ENV)
    env_cfg["payoffs"] = dict(MODULE2_PAYOFFS)
    if overrides:
        if "payoffs" in overrides:
            env_cfg["payoffs"].update(overrides["payoffs"])
        for key, value in overrides.items():
            if key != "payoffs":
                env_cfg[key] = value
    return env_cfg


def _merge_training(overrides: Optional[dict] = None) -> dict:
    cfg = dict(MODULE2_BASE_TRAINING)
    if overrides:
        cfg.update(overrides)
    return cfg


def _build_algo(seed: int, learning_rate: float, env_cfg: dict, train_cfg: dict):
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
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
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
    return config.build()


def train_setting(
    *,
    seed: int,
    run_name: str,
    env_overrides: Optional[dict],
    training_overrides: Optional[dict],
    artifacts: ExperimentArtifacts,
) -> TrainingArtifact:
    env_cfg = _merge_env(env_overrides)
    train_cfg = _merge_training(training_overrides)
    learning_rate = float(train_cfg["learning_rate"])

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=train_cfg["num_gpus"])

    algo = _build_algo(seed, learning_rate, env_cfg, train_cfg)
    checkpoint_dir = artifacts.log_path(run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = artifacts.csv_path(f"episode_log_{run_name}.csv")
    logger = EpisodeLogger(
        episode_log_path,
        fieldnames=[
            "seed",
            "training_iteration",
            "episode_reward_mean",
            "defender_reward",
            "attacker_reward",
            "defender_policy_loss",
            "attacker_policy_loss",
            "time_total_s",
        ],
    )

    start = time.time()
    for _ in range(train_cfg["episodes"]):
        result = algo.train()
        env_metrics = result.get("env_runners", {})
        learner = result.get("info", {}).get("learner", {})
        logger.log(
            {
                "seed": seed,
                "training_iteration": result.get("training_iteration"),
                "episode_reward_mean": env_metrics.get("episode_reward_mean"),
                "defender_reward": env_metrics.get("policy_reward_mean", {}).get("defender_policy"),
                "attacker_reward": env_metrics.get("policy_reward_mean", {}).get("attacker_policy"),
                "defender_policy_loss": learner.get("defender_policy", {}).get("learner_stats", {}).get("policy_loss"),
                "attacker_policy_loss": learner.get("attacker_policy", {}).get("learner_stats", {}).get("policy_loss"),
                "time_total_s": result.get("time_total_s"),
            }
        )

    checkpoint = algo.save(str(checkpoint_dir))
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()
    return TrainingArtifact(
        checkpoint_dir=checkpoint.checkpoint.path if hasattr(checkpoint, "checkpoint") else str(checkpoint),
        episode_log_csv=str(episode_log_path),
        training_time_s=time.time() - start,
    )


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def _record_probability(bucket: dict, outer: str, inner: str, value: float) -> None:
    bucket.setdefault(outer, {}).setdefault(inner, []).append(value)


def _multidiscrete_binary_probs(flat_logits: np.ndarray, n_systems: int) -> np.ndarray:
    logits = np.asarray(flat_logits, dtype=np.float32).reshape(n_systems, 2)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def evaluate_checkpoint(
    *,
    checkpoint_dir: str,
    env_overrides: Optional[dict],
    n_eval_episodes: int,
    seed: int,
) -> tuple[list[dict], dict]:
    env_cfg = _merge_env(env_overrides)
    register_chapter4_env()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=0)
    algo = PPO.from_checkpoint(checkpoint_dir)
    env = create_rllib_env(env_cfg)
    defender_policy = algo.get_policy("defender_policy")
    attacker_policy = algo.get_policy("attacker_policy")

    evaluation_rows: List[dict] = []
    signal_distribution = {"real": {"sN": [], "sH": []}, "honeypot": {"sN": [], "sH": []}}
    attack_probability = {
        "real": {"sN": [], "sH": []},
        "honeypot": {"sN": [], "sH": []},
    }
    belief_by_step: Dict[int, List[float]] = {}

    real_exposures = 0
    honeypot_exposures = 0
    real_attacks = 0
    honeypot_attacks = 0
    total_attacks = 0

    for episode_idx in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        defender_state = defender_policy.get_initial_state()
        attacker_state = attacker_policy.get_initial_state()

        while not done:
            defender_obs = {
                key: np.expand_dims(obs[DEFENDER][key].astype(np.float32), axis=0)
                for key in obs[DEFENDER]
            }
            attacker_obs = {
                key: np.expand_dims(obs[ATTACKER][key].astype(np.float32), axis=0)
                for key in obs[ATTACKER]
            }

            defender_action, defender_state, defender_info = defender_policy.compute_single_action(
                defender_obs, state=defender_state, explore=False, full_fetch=True
            )
            attacker_action, attacker_state, attacker_info = attacker_policy.compute_single_action(
                attacker_obs, state=attacker_state, explore=False, full_fetch=True
            )
            defender_action = np.asarray(defender_action, dtype=np.int8)
            attacker_action = np.asarray(attacker_action, dtype=np.int8)

            true_types = obs[DEFENDER]["system_types"]
            current_signals = defender_action
            current_beliefs = obs[ATTACKER]["belief_real"]

            defender_logits = np.asarray(defender_info.get("action_dist_inputs", np.zeros(env_cfg["n_systems"] * 2)))
            attacker_logits = np.asarray(attacker_info.get("action_dist_inputs", np.zeros(env_cfg["n_systems"] * 2)))
            defender_prob_h = _multidiscrete_binary_probs(defender_logits, env_cfg["n_systems"])
            attacker_prob_attack = _multidiscrete_binary_probs(attacker_logits, env_cfg["n_systems"])

            step_idx = int(round(float(obs[DEFENDER]["current_step"][0]) * env_cfg["max_steps"]))

            for idx, theta in enumerate(true_types):
                theta_label = "real" if int(theta) == THETA_REAL else "honeypot"
                signal_label = "sN" if int(current_signals[idx]) == SIGNAL_NORMAL else "sH"

                _record_probability(
                    signal_distribution,
                    theta_label,
                    "sH",
                    float(defender_prob_h[idx]),
                )
                _record_probability(
                    signal_distribution,
                    theta_label,
                    "sN",
                    float(1.0 - defender_prob_h[idx]),
                )
                _record_probability(
                    attack_probability,
                    theta_label,
                    signal_label,
                    float(attacker_prob_attack[idx]),
                )
                belief_by_step.setdefault(step_idx, []).append(float(current_beliefs[idx]))

                attacked = int(attacker_action[idx]) == ACTION_ATTACK
                if int(theta) == THETA_REAL:
                    real_exposures += 1
                    if attacked:
                        real_attacks += 1
                else:
                    honeypot_exposures += 1
                    if attacked:
                        honeypot_attacks += 1
                if attacked:
                    total_attacks += 1

                evaluation_rows.append(
                    {
                        "episode": episode_idx,
                        "step": step_idx,
                        "system_idx": idx,
                        "theta": int(theta),
                        "signal": int(current_signals[idx]),
                        "attacker_action": int(attacker_action[idx]),
                        "belief_real": float(current_beliefs[idx]),
                        "defender_prob_signal_h": float(defender_prob_h[idx]),
                        "attacker_prob_attack": float(attacker_prob_attack[idx]),
                    }
                )

            obs, _, terminations, truncations, _ = env.step({DEFENDER: defender_action, ATTACKER: attacker_action})
            done = all(terminations.values()) or all(truncations.values())

    algo.stop()
    env.close()

    metric_row = {
        "real_host_attack_rate": real_attacks / max(real_exposures, 1),
        "honeypot_hit_rate": honeypot_attacks / max(honeypot_exposures, 1),
        "deception_success_rate": honeypot_attacks / max(total_attacks, 1),
        "signal_effect": 0.5
        * (
            abs(_safe_mean(attack_probability["real"]["sN"]) - _safe_mean(attack_probability["real"]["sH"]))
            + abs(_safe_mean(attack_probability["honeypot"]["sN"]) - _safe_mean(attack_probability["honeypot"]["sH"]))
        ),
        "mean_belief_last_step": _safe_mean(belief_by_step[max(belief_by_step)] if belief_by_step else []),
        "defender_signal_real_sN": _safe_mean(signal_distribution["real"]["sN"]),
        "defender_signal_real_sH": _safe_mean(signal_distribution["real"]["sH"]),
        "defender_signal_honeypot_sN": _safe_mean(signal_distribution["honeypot"]["sN"]),
        "defender_signal_honeypot_sH": _safe_mean(signal_distribution["honeypot"]["sH"]),
        "attacker_real_sN": _safe_mean(attack_probability["real"]["sN"]),
        "attacker_real_sH": _safe_mean(attack_probability["real"]["sH"]),
        "attacker_honeypot_sN": _safe_mean(attack_probability["honeypot"]["sN"]),
        "attacker_honeypot_sH": _safe_mean(attack_probability["honeypot"]["sH"]),
    }
    return evaluation_rows, metric_row


def _focus_parameter_value(parameter_name: str, metric_rows: List[dict]):
    values = [row["parameter_value"] for row in metric_rows]
    if parameter_name == "T" and 50 in values:
        return 50
    return values[len(values) // 2]


def _plot_violin_distributions(
    evaluation_rows: List[dict],
    metric_rows: List[dict],
    parameter_name: str,
    artifacts: ExperimentArtifacts,
) -> None:
    focus_value = _focus_parameter_value(parameter_name, metric_rows)
    rows = [row for row in evaluation_rows if row["parameter_value"] == focus_value]

    defender_real = [1.0 - row["defender_prob_signal_h"] for row in rows if row["theta"] == THETA_REAL], [
        row["defender_prob_signal_h"] for row in rows if row["theta"] == THETA_REAL
    ]
    defender_honeypot = [1.0 - row["defender_prob_signal_h"] for row in rows if row["theta"] == THETA_HONEYPOT], [
        row["defender_prob_signal_h"] for row in rows if row["theta"] == THETA_HONEYPOT
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["普通信号", "蜜罐信号"]
    for axis, data, title in [
        (axes[0], defender_real, "真实系统下防御方信号分布"),
        (axes[1], defender_honeypot, "蜜罐系统下防御方信号分布"),
    ]:
        violin = axis.violinplot(data, showmedians=True)
        violin["cmedians"].set_color("green")
        violin["cmedians"].set_linewidth(2)
        axis.boxplot(data, showfliers=False)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(labels)
        axis.set_ylabel("信号选择概率")
        axis.set_title(title)
        axis.grid(True, alpha=0.25, linestyle="--")
        means = [statistics.mean(series) if series else 0.0 for series in data]
        axis.plot([1, 2], means, "r*", label="均值")
        axis.legend(frameon=False)
    plt.tight_layout()
    suffix = f"{parameter_name}{focus_value}"
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"policy_distribution_{suffix}", "png")), dpi=300, bbox_inches="tight")
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"policy_distribution_{suffix}", "pdf")), bbox_inches="tight")
    plt.close(fig)

    attacker_real = [
        [row["attacker_prob_attack"] for row in rows if row["theta"] == THETA_REAL and row["signal"] == SIGNAL_NORMAL],
        [row["attacker_prob_attack"] for row in rows if row["theta"] == THETA_REAL and row["signal"] == SIGNAL_HONEYPOT],
    ]
    attacker_honeypot = [
        [row["attacker_prob_attack"] for row in rows if row["theta"] == THETA_HONEYPOT and row["signal"] == SIGNAL_NORMAL],
        [row["attacker_prob_attack"] for row in rows if row["theta"] == THETA_HONEYPOT and row["signal"] == SIGNAL_HONEYPOT],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, data, title in [
        (axes[0], attacker_real, "真实系统下攻击方攻击概率"),
        (axes[1], attacker_honeypot, "蜜罐系统下攻击方攻击概率"),
    ]:
        violin = axis.violinplot(data, showmedians=True)
        violin["cmedians"].set_color("green")
        violin["cmedians"].set_linewidth(2)
        axis.boxplot(data, showfliers=False)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(labels)
        axis.set_ylabel("攻击概率")
        axis.set_title(title)
        axis.grid(True, alpha=0.25, linestyle="--")
        means = [statistics.mean(series) if series else 0.0 for series in data]
        axis.plot([1, 2], means, "r*", label="均值")
        axis.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"attack_probability_{suffix}", "png")), dpi=300, bbox_inches="tight")
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"attack_probability_{suffix}", "pdf")), bbox_inches="tight")
    plt.close(fig)


def _plot_signal_effect(metric_rows: List[dict], parameter_name: str, artifacts: ExperimentArtifacts) -> None:
    x = [str(row["parameter_value"]) for row in metric_rows]
    y = [row["signal_effect"] for row in metric_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", linewidth=2.0, color="#184E77")
    ax.set_title("信号效应随参数变化")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Signal Effect")
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"{parameter_name}_signal_effect", "png")), dpi=300, bbox_inches="tight")
    plt.savefig(artifacts.figure_path(figure_name("4_5", f"{parameter_name}_signal_effect", "pdf")), bbox_inches="tight")
    plt.close(fig)


def run_module2_sweep(
    *,
    parameter_name: str,
    parameter_values: Optional[Iterable[float]] = None,
    run_name: str,
    env_base_overrides: Optional[dict] = None,
    training_overrides: Optional[dict] = None,
    evaluation_episodes: int = 40,
) -> dict:
    module_id = CHAPTER4_MODULE_IDS["module2"]
    env_cfg = _merge_env(env_base_overrides)
    train_cfg = _merge_training(training_overrides)
    values = list(parameter_values or MODULE2_SWEEPS[parameter_name])
    config = ExperimentConfig(
        module_id=module_id,
        run_name=run_name,
        description=f"Chapter 4 module 2 sweep on {parameter_name}.",
        seeds=[0],
        environment=env_cfg,
        training=train_cfg,
        analysis={"parameter_name": parameter_name, "parameter_values": values, "evaluation_episodes": evaluation_episodes},
        tags=["chapter4", "module2", "policy", parameter_name],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    summary_rows = []
    all_eval_rows = []
    for value in values:
        env_overrides = {}
        env_overrides = dict(env_base_overrides or {})
        if parameter_name == "T":
            env_overrides["max_steps"] = int(value)
        elif parameter_name == "N":
            env_overrides["n_systems"] = int(value)
        elif parameter_name == "psi0":
            env_overrides["prior_belief_real"] = float(value)
        else:
            env_overrides["payoffs"] = {**(env_overrides.get("payoffs", {})), parameter_name: float(value)}

        suffix = f"{run_name}_{parameter_name}_{str(value).replace('.', '_')}"
        artifact = train_setting(
            seed=0,
            run_name=suffix,
            env_overrides=env_overrides,
            training_overrides=train_cfg,
            artifacts=artifacts,
        )
        eval_rows, metric_row = evaluate_checkpoint(
            checkpoint_dir=artifact.checkpoint_dir,
            env_overrides=env_overrides,
            n_eval_episodes=evaluation_episodes,
            seed=101,
        )
        metric_row.update({"parameter_name": parameter_name, "parameter_value": value, "training_time_s": artifact.training_time_s})
        summary_rows.append(metric_row)
        for row in eval_rows:
            row.update({"parameter_name": parameter_name, "parameter_value": value})
        all_eval_rows.extend(eval_rows)

    write_csv(artifacts.csv_path(f"{run_name}_evaluation_records.csv"), all_eval_rows)
    write_csv(artifacts.root / "results_summary.csv", summary_rows)
    write_csv(artifacts.csv_path(f"{run_name}_results_summary.csv"), summary_rows)
    write_csv(artifacts.table_path(table_name("4_3", f"{parameter_name}_policy_metrics")), summary_rows)
    _plot_violin_distributions(all_eval_rows, summary_rows, parameter_name, artifacts)
    _plot_signal_effect(summary_rows, parameter_name, artifacts)
    return {
        "module_root": str(artifacts.root),
        "results_summary": str(artifacts.root / "results_summary.csv"),
        "evaluation_csv": str(artifacts.csv_path(f"{run_name}_evaluation_records.csv")),
    }
