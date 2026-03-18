from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict
from math import sqrt
from typing import Dict, List

from .config import GameConfig
from .model import AttackerAction, DefenderType, Signal, StageOutcome, discounted_belief, utility
from .strategies import STRATEGY_BUILDERS, StrategyState


def _sample_signal(signal_probs: Dict[Signal, float], rng: random.Random) -> Signal:
    threshold = rng.random()
    cumulative = 0.0
    for signal, probability in signal_probs.items():
        cumulative += probability
        if threshold <= cumulative:
            return signal
    return Signal.SIGMA2


def run_episode(strategy_name: str, config: GameConfig, rng: random.Random) -> List[StageOutcome]:
    defender_type = DefenderType.THETA1 if rng.random() < config.prior_theta1 else DefenderType.THETA2
    belief_theta1 = config.prior_theta1
    signal_history: List[Signal] = []
    outcomes: List[StageOutcome] = []

    for stage in range(1, config.horizon + 1):
        regime = STRATEGY_BUILDERS[strategy_name](StrategyState(belief_theta1=belief_theta1, config=config))
        signal = _sample_signal(regime.defender_signal_probs[defender_type], rng)
        signal_history.append(signal)
        posterior = discounted_belief(
            prior_theta1=config.prior_theta1,
            signal_history=signal_history,
            type_signal_probs=regime.defender_signal_probs,
            beta=config.beta,
        )
        attack_probability = regime.attack_prob_by_signal[signal]
        action = AttackerAction.ATTACK if rng.random() < attack_probability else AttackerAction.RETREAT
        defender_value, attacker_value = utility(defender_type, signal, action, config)

        outcomes.append(
            StageOutcome(
                stage=stage,
                belief_theta1=posterior,
                signal=signal,
                attack_probability=attack_probability,
                action=action,
                defender_utility=defender_value,
                attacker_utility=attacker_value,
                regime_metrics=regime.mixing_metrics,
            )
        )

        belief_theta1 = posterior
        if action == AttackerAction.ATTACK:
            break

    return outcomes


def run_strategy_comparison(config: GameConfig) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    strategy_names = list(STRATEGY_BUILDERS.keys())

    for offset, strategy_name in enumerate(strategy_names):
        rng = random.Random(20260310 + offset)
        defender_total = 0.0
        attacker_total = 0.0
        belief_paths = []
        final_beliefs = []
        attack_probability_sum = 0.0
        stage_count = 0
        action_counter = Counter()
        signal_counter = Counter()
        metric_accumulator = defaultdict(float)

        for _ in range(config.monte_carlo_runs):
            episode = run_episode(strategy_name, config, rng)
            defender_total += sum(item.defender_utility for item in episode)
            attacker_total += sum(item.attacker_utility for item in episode)
            belief_paths.append([item.belief_theta1 for item in episode])
            final_beliefs.append(episode[-1].belief_theta1)
            action_counter.update(item.action.value for item in episode)
            signal_counter.update(item.signal.value for item in episode)
            attack_probability_sum += sum(item.attack_probability for item in episode)
            stage_count += len(episode)

            for item in episode:
                for key, value in item.regime_metrics.items():
                    metric_accumulator[key] += value

        max_len = max(len(path) for path in belief_paths)
        padded = []
        for path in belief_paths:
            tail = path[-1]
            padded.append(path + [tail] * (max_len - len(path)))

        avg_belief_path = [
            sum(path[index] for path in padded) / len(padded)
            for index in range(max_len)
        ]
        final_belief_mean = sum(final_beliefs) / len(final_beliefs)
        final_belief_std = sqrt(
            sum((belief - final_belief_mean) ** 2 for belief in final_beliefs) / len(final_beliefs)
        )

        results[strategy_name] = {
            "defender_expected_utility": defender_total / config.monte_carlo_runs,
            "attacker_expected_utility": attacker_total / config.monte_carlo_runs,
            "avg_belief_path": avg_belief_path,
            "final_beliefs": final_beliefs,
            "final_belief_mean": final_belief_mean,
            "final_belief_std": final_belief_std,
            "belief_span_mean": max(avg_belief_path) - min(avg_belief_path),
            "attack_probability_mean": attack_probability_sum / stage_count if stage_count else 0.0,
            "action_counts": dict(action_counter),
            "signal_counts": dict(signal_counter),
            "equilibrium_metrics_mean": {
                key: value / stage_count if stage_count else 0.0
                for key, value in metric_accumulator.items()
            },
        }
    return results


def run_sensitivity_analysis(config: GameConfig) -> Dict[str, List[Dict[str, float]]]:
    sweeps = {
        "prior_theta1": [0.3, 0.4, 0.5, 0.6, 0.7],
        "beta": [0.2, 0.5, 0.8, 1.1, 1.4],
        "c_theta1": [0.5, 1.0, 1.5, 2.0, 2.5],
        "c_theta2": [0.5, 1.0, 1.5, 2.0, 2.5],
        "defender_pressure_cost": [0.5, 1.0, 1.5, 2.0, 2.5],
        "defender_protection_gain": [2.0, 4.0, 6.0, 8.0, 10.0],
        "attacker_deception_penalty": [0.5, 1.0, 1.5, 2.0, 2.5],
    }
    strategy_focus = {
        "prior_theta1": "pbne_production_camouflage",
        "beta": "pbne_honeypot_camouflage",
        "c_theta1": "pbne_production_camouflage",
        "c_theta2": "pbne_honeypot_camouflage",
        "defender_pressure_cost": "pbne_production_camouflage",
        "defender_protection_gain": "pbne_production_camouflage",
        "attacker_deception_penalty": "pbne_honeypot_camouflage",
    }
    metric_key = {
        "prior_theta1": "lambda_d_star",
        "beta": "lambda_d_prime",
        "c_theta1": "lambda_a_star",
        "c_theta2": "lambda_a_prime",
        "defender_pressure_cost": "lambda_a_star",
        "defender_protection_gain": "lambda_a_star",
        "attacker_deception_penalty": "lambda_d_prime",
    }
    metric_label = {
        "prior_theta1": "lambda_d_star",
        "beta": "lambda_d_prime",
        "c_theta1": "lambda_a_star",
        "c_theta2": "lambda_a_prime",
        "defender_pressure_cost": "lambda_a_star",
        "defender_protection_gain": "lambda_a_star",
        "attacker_deception_penalty": "lambda_d_prime",
    }

    results: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for parameter, values in sweeps.items():
        for value in values:
            variant_kwargs = asdict(config)
            variant_kwargs[parameter] = value
            if parameter == "beta":
                variant_kwargs["prior_theta1"] = 0.35
            variant = GameConfig(**variant_kwargs)
            strategy_name = strategy_focus[parameter]
            summary = run_strategy_comparison(variant)[strategy_name]
            action_total = sum(summary["action_counts"].values())
            attack_ratio = summary["action_counts"].get("attack", 0) / action_total if action_total else 0.0
            results[parameter].append(
                {
                    "value": value,
                    "strategy": strategy_name,
                    "defender_expected_utility": summary["defender_expected_utility"],
                    "attacker_expected_utility": summary["attacker_expected_utility"],
                    "final_belief_mean": summary["final_belief_mean"],
                    "final_belief_std": summary["final_belief_std"],
                    "belief_span_mean": summary["belief_span_mean"],
                    "attack_ratio": attack_ratio,
                    "mixing_probability_mean": summary["equilibrium_metrics_mean"].get(metric_key[parameter], 0.0),
                    "mixing_metric_label": metric_label[parameter],
                }
            )
    return dict(results)


def run_feasible_comparison_scenarios(config: GameConfig) -> Dict[str, object]:
    high_prior_config = GameConfig(
        horizon=config.horizon,
        prior_theta1=0.65,
        beta=config.beta,
        epsilon=config.epsilon,
        attack_cost=config.attack_cost,
        attack_gain=config.attack_gain,
        defender_loss=config.defender_loss,
        defender_pressure_cost=config.defender_pressure_cost,
        defender_protection_gain=config.defender_protection_gain,
        intel_gain=config.intel_gain,
        intel_loss=config.intel_loss,
        defender_deception_bonus=config.defender_deception_bonus,
        attacker_deception_penalty=config.attacker_deception_penalty,
        c_theta1=config.c_theta1,
        c_theta2=config.c_theta2,
        monte_carlo_runs=config.monte_carlo_runs,
    )
    low_prior_config = GameConfig(
        horizon=config.horizon,
        prior_theta1=0.35,
        beta=config.beta,
        epsilon=config.epsilon,
        attack_cost=config.attack_cost,
        attack_gain=config.attack_gain,
        defender_loss=config.defender_loss,
        defender_pressure_cost=config.defender_pressure_cost,
        defender_protection_gain=config.defender_protection_gain,
        intel_gain=config.intel_gain,
        intel_loss=config.intel_loss,
        defender_deception_bonus=config.defender_deception_bonus,
        attacker_deception_penalty=config.attacker_deception_penalty,
        c_theta1=config.c_theta1,
        c_theta2=config.c_theta2,
        monte_carlo_runs=config.monte_carlo_runs,
    )

    return {
        "scenario_a_high_prior_theta1": {
            "description": "Production-system-dominant environment; compare truthful baseline with PBNE-1.",
            "config": {
                "prior_theta1": high_prior_config.prior_theta1,
                "beta": high_prior_config.beta,
            },
            "results": {
                key: value
                for key, value in run_strategy_comparison(high_prior_config).items()
                if key in {"truthful_baseline", "pbne_production_camouflage"}
            },
        },
        "scenario_b_low_prior_theta1": {
            "description": "Honeypot-dominant environment; compare truthful baseline with PBNE-2.",
            "config": {
                "prior_theta1": low_prior_config.prior_theta1,
                "beta": low_prior_config.beta,
            },
            "results": {
                key: value
                for key, value in run_strategy_comparison(low_prior_config).items()
                if key in {"truthful_baseline", "pbne_honeypot_camouflage"}
            },
        },
    }
