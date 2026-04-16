from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

from .config import GameConfig
from .model import (
    AttackerAction,
    DefenderType,
    Signal,
    attacker_expected_utility,
    attacker_indifference_threshold,
    defender_expected_utility,
    utility,
)
from .strategies import (
    RECURSIVE_PBNE_HONEYPOT,
    RECURSIVE_PBNE_PRODUCTION,
    TRUTHFUL_BASELINE,
    honeypot_camouflage_stage_policy,
    production_camouflage_stage_policy,
    truthful_stage_policy,
)

SCENARIO_A_PRIOR = 0.65
SCENARIO_B_PRIOR = 0.35
SCENARIO_B_C_THETA2 = 1.35
BELIEF_MONTE_CARLO_RUNS = 4000
BELIEF_MONTE_CARLO_SEED = 20260414
HORIZON_SWEEP_MAX = 15
SENSITIVITY_GRID_MAP: Dict[str, Dict[str, List[float]]] = {
    "prior_theta1": {
        "scenario_a": [0.56, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84],
        "scenario_b": [0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44],
    },
    "defender_loss": {
        "scenario_a": [8.0, 10.0, 12.0, 14.0],
        "scenario_b": [8.0, 10.0, 12.0, 14.0],
    },
    "c_theta1": {
        "scenario_a": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "scenario_b": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    },
    "c_theta2": {
        "scenario_a": [0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "scenario_b": [0.9, 1.1, 1.35, 1.5, 1.7, 1.9],
    },
}
SENSITIVITY_PARAMETER_LABELS = {
    "prior_theta1": "初始公共信念 p1",
    "defender_loss": "真实系统受攻击损失 l_d",
    "c_theta1": "真实系统伪装成本 c_theta1",
    "c_theta2": "蜜罐系统伪装成本 c_theta2",
}
SENSITIVITY_PLOT_SPECS = [
    {
        "panel_key": "prior_theta1_scenario_a",
        "parameter_name": "prior_theta1",
        "scenario_key": "scenario_a",
        "strategy_name": RECURSIVE_PBNE_PRODUCTION,
        "title": "初始公共信念 p1（场景 A）",
        "grid_values": [0.56, 0.65, 0.74, 0.83],
    },
    {
        "panel_key": "defender_loss_scenario_a",
        "parameter_name": "defender_loss",
        "scenario_key": "scenario_a",
        "strategy_name": RECURSIVE_PBNE_PRODUCTION,
        "title": "真实系统受攻击损失 l_d（场景 A）",
        "grid_values": [8.0, 10.0, 12.0, 14.0],
    },
    {
        "panel_key": "c_theta1_scenario_a",
        "parameter_name": "c_theta1",
        "scenario_key": "scenario_a",
        "strategy_name": RECURSIVE_PBNE_PRODUCTION,
        "title": "真实系统伪装成本 c_theta1（场景 A）",
        "grid_values": [1.5, 2.0, 2.5, 3.0],
    },
    {
        "panel_key": "c_theta2_scenario_b",
        "parameter_name": "c_theta2",
        "scenario_key": "scenario_b",
        "strategy_name": RECURSIVE_PBNE_HONEYPOT,
        "title": "蜜罐系统伪装成本 c_theta2（场景 B）",
        "grid_values": [0.9, 1.1, 1.35, 1.7],
    },
]


@dataclass(frozen=True)
class BackwardInductionResult:
    strategy_name: str
    initial_belief: float
    threshold_belief: float
    stage_records: List[Dict[str, object]]


def _belief_key(value: float) -> str:
    return f"{value:.6f}"


def _ordered_unique(values: Iterable[float]) -> List[float]:
    ordered: List[float] = []
    for value in values:
        numeric = float(value)
        if not any(abs(existing - numeric) <= 1e-12 for existing in ordered):
            ordered.append(numeric)
    return ordered


def _canonical_belief(value: float, belief_states: Iterable[float]) -> float:
    for candidate in belief_states:
        if abs(candidate - value) <= 1e-12:
            return candidate
    raise KeyError(f"Belief state {value:.16f} is not in the solved state set.")


def _build_experiment_scenarios(config: GameConfig) -> Dict[str, Dict[str, object]]:
    return {
        "scenario_a": {
            "label": "场景 A",
            "description": "高初始信念 p_1，偏向第一类半分离结构“以真乱假”。",
            "initial_belief": SCENARIO_A_PRIOR,
            "pbne_strategy": RECURSIVE_PBNE_PRODUCTION,
            "pbne_label": "递归 PBNE 策略（以真乱假）",
            "config": replace(config, prior_theta1=SCENARIO_A_PRIOR),
        },
        "scenario_b": {
            "label": "场景 B",
            "description": "低初始信念 p_1，偏向第二类半分离结构“以假乱真”；为满足第二类半分离 PBNE 的不偏离约束，场景内使用较低的蜜罐伪装成本。",
            "initial_belief": SCENARIO_B_PRIOR,
            "pbne_strategy": RECURSIVE_PBNE_HONEYPOT,
            "pbne_label": "递归 PBNE 策略（以假乱真）",
            "config": replace(config, prior_theta1=SCENARIO_B_PRIOR, c_theta2=SCENARIO_B_C_THETA2),
        },
    }


def _select_pbne_strategy(initial_belief: float, config: GameConfig) -> str:
    threshold = attacker_indifference_threshold(config)
    if initial_belief > threshold:
        return RECURSIVE_PBNE_PRODUCTION
    if initial_belief < threshold:
        return RECURSIVE_PBNE_HONEYPOT
    raise ValueError("Initial belief must stay away from the indifference threshold p^dagger.")


def _sample_signal(signal_probs: Dict[Signal, float], rng: random.Random) -> Signal:
    draw = rng.random()
    cumulative = 0.0
    for signal, probability in signal_probs.items():
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return signal
    return Signal.SIGMA2


def backward_induction(strategy_name: str, config: GameConfig, initial_belief: float) -> BackwardInductionResult:
    threshold = attacker_indifference_threshold(config)
    if strategy_name == RECURSIVE_PBNE_PRODUCTION and initial_belief <= threshold:
        raise ValueError("Scenario A must satisfy p_1 > p^dagger.")
    if strategy_name == RECURSIVE_PBNE_HONEYPOT and initial_belief >= threshold:
        raise ValueError("Scenario B must satisfy p_1 < p^dagger.")

    if strategy_name in {RECURSIVE_PBNE_PRODUCTION, RECURSIVE_PBNE_HONEYPOT}:
        belief_states = _ordered_unique([initial_belief, threshold, 0.0, 1.0])
    else:
        raise ValueError(f"Unsupported strategy for backward induction: {strategy_name}")

    next_values: Dict[DefenderType, Dict[float, float]] = {
        DefenderType.THETA1: {belief: 0.0 for belief in belief_states},
        DefenderType.THETA2: {belief: 0.0 for belief in belief_states},
    }
    stage_records: List[Dict[str, object]] = []

    for stage in range(config.horizon, 0, -1):
        current_values: Dict[DefenderType, Dict[float, float]] = {
            DefenderType.THETA1: {},
            DefenderType.THETA2: {},
        }
        belief_records: Dict[str, Dict[str, float]] = {}

        if strategy_name == RECURSIVE_PBNE_PRODUCTION:
            attack_after_sigma2 = 1.0 - (
                config.c_theta1
                + config.gamma * (next_values[DefenderType.THETA1][1.0] - next_values[DefenderType.THETA1][threshold])
            ) / (config.defender_loss + config.defender_pressure_cost + config.defender_protection_gain)
            for belief in belief_states:
                policy = production_camouflage_stage_policy(stage, belief, attack_after_sigma2, config)
                x_t = policy.mixing_metrics["x_t"]
                next_after_sigma1 = _canonical_belief(policy.next_belief_by_signal[Signal.SIGMA1], belief_states)
                next_after_sigma2 = _canonical_belief(policy.next_belief_by_signal[Signal.SIGMA2], belief_states)
                theta1_sigma1_value = utility(
                    DefenderType.THETA1,
                    Signal.SIGMA1,
                    action=AttackerAction.ATTACK,
                    config=config,
                )[0] + config.gamma * next_values[DefenderType.THETA1][next_after_sigma1]
                theta1_sigma2_value = defender_expected_utility(
                    DefenderType.THETA1,
                    Signal.SIGMA2,
                    attack_after_sigma2,
                    config,
                ) + config.gamma * next_values[DefenderType.THETA1][next_after_sigma2]
                theta2_on_path_value = defender_expected_utility(
                    DefenderType.THETA2,
                    Signal.SIGMA2,
                    attack_after_sigma2,
                    config,
                ) + config.gamma * next_values[DefenderType.THETA2][next_after_sigma2]

                current_values[DefenderType.THETA1][belief] = (1.0 - x_t) * theta1_sigma1_value + x_t * theta1_sigma2_value
                current_values[DefenderType.THETA2][belief] = theta2_on_path_value
                belief_records[_belief_key(belief)] = {
                    "x_t": x_t,
                    "V_theta1": current_values[DefenderType.THETA1][belief],
                    "V_theta2": current_values[DefenderType.THETA2][belief],
                }

            theta1_sigma1 = utility(
                DefenderType.THETA1,
                Signal.SIGMA1,
                action=AttackerAction.ATTACK,
                config=config,
            )[0] + config.gamma * next_values[DefenderType.THETA1][1.0]
            theta1_sigma2 = defender_expected_utility(
                DefenderType.THETA1,
                Signal.SIGMA2,
                attack_after_sigma2,
                config,
            ) + config.gamma * next_values[DefenderType.THETA1][threshold]
            theta2_on_path = defender_expected_utility(
                DefenderType.THETA2,
                Signal.SIGMA2,
                attack_after_sigma2,
                config,
            ) + config.gamma * next_values[DefenderType.THETA2][threshold]
            theta2_deviation = utility(
                DefenderType.THETA2,
                Signal.SIGMA1,
                action=AttackerAction.ATTACK,
                config=config,
            )[0] + config.gamma * next_values[DefenderType.THETA2][1.0]

            stage_records.append(
                {
                    "stage": stage,
                    "attack_after_sigma2": attack_after_sigma2,
                    "theta1_indifference_gap": theta1_sigma1 - theta1_sigma2,
                    "theta2_no_deviation_gap": theta2_on_path - theta2_deviation,
                    "belief_records": belief_records,
                }
            )
        else:
            attack_after_sigma1 = (
                config.c_theta2
                + config.gamma * (next_values[DefenderType.THETA2][0.0] - next_values[DefenderType.THETA2][threshold])
            ) / (config.intel_gain + config.defender_deception_bonus)
            for belief in belief_states:
                policy = honeypot_camouflage_stage_policy(stage, belief, attack_after_sigma1, config)
                z_t = policy.mixing_metrics["z_t"]
                next_after_sigma1 = _canonical_belief(policy.next_belief_by_signal[Signal.SIGMA1], belief_states)
                next_after_sigma2 = _canonical_belief(policy.next_belief_by_signal[Signal.SIGMA2], belief_states)
                theta1_on_path_value = defender_expected_utility(
                    DefenderType.THETA1,
                    Signal.SIGMA1,
                    attack_after_sigma1,
                    config,
                ) + config.gamma * next_values[DefenderType.THETA1][next_after_sigma1]
                theta2_sigma1_value = defender_expected_utility(
                    DefenderType.THETA2,
                    Signal.SIGMA1,
                    attack_after_sigma1,
                    config,
                ) + config.gamma * next_values[DefenderType.THETA2][next_after_sigma1]
                theta2_sigma2_value = utility(
                    DefenderType.THETA2,
                    Signal.SIGMA2,
                    action=AttackerAction.RETREAT,
                    config=config,
                )[0] + config.gamma * next_values[DefenderType.THETA2][next_after_sigma2]

                current_values[DefenderType.THETA1][belief] = theta1_on_path_value
                current_values[DefenderType.THETA2][belief] = z_t * theta2_sigma1_value + (1.0 - z_t) * theta2_sigma2_value
                belief_records[_belief_key(belief)] = {
                    "z_t": z_t,
                    "V_theta1": current_values[DefenderType.THETA1][belief],
                    "V_theta2": current_values[DefenderType.THETA2][belief],
                }

            theta1_on_path = defender_expected_utility(
                DefenderType.THETA1,
                Signal.SIGMA1,
                attack_after_sigma1,
                config,
            ) + config.gamma * next_values[DefenderType.THETA1][threshold]
            theta1_deviation = utility(
                DefenderType.THETA1,
                Signal.SIGMA2,
                action=AttackerAction.RETREAT,
                config=config,
            )[0] + config.gamma * next_values[DefenderType.THETA1][0.0]
            theta2_sigma1 = defender_expected_utility(
                DefenderType.THETA2,
                Signal.SIGMA1,
                attack_after_sigma1,
                config,
            ) + config.gamma * next_values[DefenderType.THETA2][threshold]
            theta2_sigma2 = utility(
                DefenderType.THETA2,
                Signal.SIGMA2,
                action=AttackerAction.RETREAT,
                config=config,
            )[0] + config.gamma * next_values[DefenderType.THETA2][0.0]

            stage_records.append(
                {
                    "stage": stage,
                    "attack_after_sigma1": attack_after_sigma1,
                    "theta2_indifference_gap": theta2_sigma1 - theta2_sigma2,
                    "theta1_no_deviation_gap": theta1_on_path - theta1_deviation,
                    "belief_records": belief_records,
                }
            )

        next_values = current_values

    stage_records.reverse()
    return BackwardInductionResult(
        strategy_name=strategy_name,
        initial_belief=initial_belief,
        threshold_belief=threshold,
        stage_records=stage_records,
    )


def _build_stage_policy(
    strategy_name: str,
    stage: int,
    belief_theta1: float,
    config: GameConfig,
    solution: BackwardInductionResult | None,
):
    if strategy_name == TRUTHFUL_BASELINE:
        return truthful_stage_policy(stage, belief_theta1, config)
    if solution is None:
        raise ValueError(f"{strategy_name} requires a backward-induction solution.")
    stage_record = solution.stage_records[stage - 1]
    if strategy_name == RECURSIVE_PBNE_PRODUCTION:
        return production_camouflage_stage_policy(
            stage,
            belief_theta1,
            stage_record["attack_after_sigma2"],
            config,
        )
    if strategy_name == RECURSIVE_PBNE_HONEYPOT:
        return honeypot_camouflage_stage_policy(
            stage,
            belief_theta1,
            stage_record["attack_after_sigma1"],
            config,
        )
    raise ValueError(f"Unknown strategy: {strategy_name}")


def forward_simulation(
    strategy_name: str,
    config: GameConfig,
    initial_belief: float,
    solution: BackwardInductionResult | None = None,
) -> Dict[str, object]:
    nodes: Dict[DefenderType, Dict[float, float]] = {
        DefenderType.THETA1: {initial_belief: initial_belief},
        DefenderType.THETA2: {initial_belief: 1.0 - initial_belief},
    }
    stage_rows: List[Dict[str, object]] = []
    cumulative_defender_utility = 0.0
    cumulative_attacker_utility = 0.0
    discounted_cumulative_defender_utility = 0.0
    discounted_cumulative_attacker_utility = 0.0

    for stage in range(1, config.horizon + 1):
        next_nodes: Dict[DefenderType, Dict[float, float]] = {
            DefenderType.THETA1: defaultdict(float),
            DefenderType.THETA2: defaultdict(float),
        }
        stage_defender_utility = 0.0
        stage_attacker_utility = 0.0
        stage_attack_probability = 0.0
        next_belief_distribution: Dict[float, float] = defaultdict(float)

        for defender_type, belief_weights in nodes.items():
            for belief_theta1, node_probability in belief_weights.items():
                if node_probability == 0.0:
                    continue
                policy = _build_stage_policy(strategy_name, stage, belief_theta1, config, solution)
                for signal, signal_probability in policy.defender_signal_probs[defender_type].items():
                    if signal_probability == 0.0:
                        continue
                    branch_probability = node_probability * signal_probability
                    attack_probability = policy.attack_prob_by_signal[signal]
                    next_belief = policy.next_belief_by_signal[signal]
                    stage_defender_utility += branch_probability * defender_expected_utility(
                        defender_type,
                        signal,
                        attack_probability,
                        config,
                    )
                    stage_attacker_utility += branch_probability * attacker_expected_utility(
                        defender_type,
                        signal,
                        attack_probability,
                        config,
                    )
                    stage_attack_probability += branch_probability * attack_probability
                    next_nodes[defender_type][next_belief] += branch_probability
                    next_belief_distribution[next_belief] += branch_probability

        cumulative_defender_utility += stage_defender_utility
        cumulative_attacker_utility += stage_attacker_utility
        discount_weight = config.gamma ** (stage - 1)
        discounted_stage_defender_utility = discount_weight * stage_defender_utility
        discounted_stage_attacker_utility = discount_weight * stage_attacker_utility
        discounted_cumulative_defender_utility += discounted_stage_defender_utility
        discounted_cumulative_attacker_utility += discounted_stage_attacker_utility
        stage_rows.append(
            {
                "stage": stage,
                "expected_defender_utility": stage_defender_utility,
                "expected_attacker_utility": stage_attacker_utility,
                "discounted_expected_defender_utility": discounted_stage_defender_utility,
                "discounted_expected_attacker_utility": discounted_stage_attacker_utility,
                "cumulative_defender_utility": cumulative_defender_utility,
                "cumulative_attacker_utility": cumulative_attacker_utility,
                "discounted_cumulative_defender_utility": discounted_cumulative_defender_utility,
                "discounted_cumulative_attacker_utility": discounted_cumulative_attacker_utility,
                "average_defender_utility": cumulative_defender_utility / stage,
                "average_attacker_utility": cumulative_attacker_utility / stage,
                "expected_attack_probability": stage_attack_probability,
                "expected_belief_theta1": sum(belief * weight for belief, weight in next_belief_distribution.items()),
                "belief_distribution": {
                    _belief_key(belief): probability
                    for belief, probability in sorted(next_belief_distribution.items())
                },
            }
        )
        nodes = {
            DefenderType.THETA1: dict(next_nodes[DefenderType.THETA1]),
            DefenderType.THETA2: dict(next_nodes[DefenderType.THETA2]),
        }

    final_belief_distribution = defaultdict(float)
    for belief_weights in nodes.values():
        for belief_theta1, probability in belief_weights.items():
            final_belief_distribution[belief_theta1] += probability

    return {
        "strategy_name": strategy_name,
        "initial_belief": initial_belief,
        "cumulative_defender_utility": cumulative_defender_utility,
        "cumulative_attacker_utility": cumulative_attacker_utility,
        "discounted_cumulative_defender_utility": discounted_cumulative_defender_utility,
        "discounted_cumulative_attacker_utility": discounted_cumulative_attacker_utility,
        "stage_rows": stage_rows,
        "final_belief_distribution": {
            _belief_key(belief): probability
            for belief, probability in sorted(final_belief_distribution.items())
        },
    }


def _simulate_belief_path(
    strategy_name: str,
    config: GameConfig,
    initial_belief: float,
    rng: random.Random,
    solution: BackwardInductionResult | None = None,
) -> Dict[str, object]:
    defender_type = DefenderType.THETA1 if rng.random() < initial_belief else DefenderType.THETA2
    current_belief = initial_belief
    belief_path = [current_belief]
    stage_records: List[Dict[str, object]] = []

    for stage in range(1, config.horizon + 1):
        policy = _build_stage_policy(strategy_name, stage, current_belief, config, solution)
        signal = _sample_signal(policy.defender_signal_probs[defender_type], rng)
        next_belief = policy.next_belief_by_signal[signal]
        stage_records.append(
            {
                "stage": stage,
                "belief_theta1": current_belief,
                "signal": signal.value,
                "next_belief_theta1": next_belief,
                "defender_type": defender_type.value,
            }
        )
        belief_path.append(next_belief)
        current_belief = next_belief

    return {
        "defender_type": defender_type.value,
        "belief_path": belief_path,
        "stage_records": stage_records,
        "terminal_belief": current_belief,
    }


def belief_trajectory(
    strategy_name: str,
    config: GameConfig,
    initial_belief: float,
    solution: BackwardInductionResult | None = None,
    *,
    num_runs: int = BELIEF_MONTE_CARLO_RUNS,
    seed: int = BELIEF_MONTE_CARLO_SEED,
) -> Dict[str, object]:
    rng = random.Random(seed)
    simulations: List[Dict[str, object]] = []
    path_length = config.horizon + 1
    path_sum = [0.0] * path_length

    for _ in range(num_runs):
        simulation = _simulate_belief_path(strategy_name, config, initial_belief, rng, solution)
        simulations.append(simulation)
        for index, belief in enumerate(simulation["belief_path"]):
            path_sum[index] += belief

    average_path = [value / num_runs for value in path_sum] if num_runs else []
    selected_path = simulations[0]["belief_path"] if simulations else []
    selected_stage_records = simulations[0]["stage_records"] if simulations else []

    return {
        "seed": seed,
        "num_runs": num_runs,
        "selected_trajectory": selected_path,
        "average_trajectory": average_path,
        "selected_stage_records": selected_stage_records,
        "simulations": simulations,
    }


def terminal_belief_distribution(
    strategy_name: str,
    config: GameConfig,
    initial_belief: float,
    solution: BackwardInductionResult | None = None,
    *,
    num_runs: int = BELIEF_MONTE_CARLO_RUNS,
    seed: int = BELIEF_MONTE_CARLO_SEED,
    simulations: List[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    if simulations is None:
        trajectory_result = belief_trajectory(
            strategy_name,
            config,
            initial_belief,
            solution,
            num_runs=num_runs,
            seed=seed,
        )
        simulations = trajectory_result["simulations"]

    probability_mass: Dict[str, float] = defaultdict(float)
    terminal_beliefs: List[float] = []

    for simulation in simulations:
        terminal_belief = simulation["terminal_belief"]
        terminal_beliefs.append(terminal_belief)
        probability_mass[_belief_key(terminal_belief)] += 1.0 / len(simulations)

    return {
        "seed": seed,
        "num_runs": len(simulations),
        "terminal_beliefs": terminal_beliefs,
        "probability_mass": dict(sorted(probability_mass.items())),
    }


def _build_experiment_two_seed_map() -> Dict[str, Dict[str, int]]:
    return {
        "scenario_a": {
            "recursive_pbne": BELIEF_MONTE_CARLO_SEED + 11,
            "truthful_baseline": BELIEF_MONTE_CARLO_SEED + 12,
        },
        "scenario_b": {
            "recursive_pbne": BELIEF_MONTE_CARLO_SEED + 21,
            "truthful_baseline": BELIEF_MONTE_CARLO_SEED + 22,
        },
    }


def _categorize_belief_state(belief: float, threshold: float) -> str:
    rounded_belief = round(belief, 6)
    rounded_threshold = round(threshold, 6)
    if abs(rounded_belief - 0.0) <= 1e-6:
        return "certain_honeypot"
    if abs(rounded_belief - rounded_threshold) <= 1e-6:
        return "critical_uncertainty"
    if abs(rounded_belief - 1.0) <= 1e-6:
        return "certain_real"
    return "other"


def _state_probability_evolution(
    initial_belief: float,
    threshold: float,
    forward_result: Dict[str, object],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    initial_state_mass = {
        "certain_honeypot": 0.0,
        "critical_uncertainty": 0.0,
        "certain_real": 0.0,
        "other": 0.0,
    }
    initial_state_mass[_categorize_belief_state(initial_belief, threshold)] = 1.0
    rows.append({"stage": 1, **initial_state_mass})

    for offset, stage_row in enumerate(forward_result["stage_rows"], start=2):
        state_mass = {
            "certain_honeypot": 0.0,
            "critical_uncertainty": 0.0,
            "certain_real": 0.0,
            "other": 0.0,
        }
        for belief_key, probability in stage_row["belief_distribution"].items():
            category = _categorize_belief_state(float(belief_key), threshold)
            state_mass[category] += probability
        rows.append({"stage": offset, **state_mass})

    return rows


def _terminal_belief_mean_from_distribution(belief_distribution: Dict[str, float]) -> float:
    return sum(float(belief_key) * probability for belief_key, probability in belief_distribution.items())


def _terminal_uncertainty_probability_from_distribution(
    belief_distribution: Dict[str, float],
    threshold: float,
) -> float:
    return belief_distribution.get(_belief_key(threshold), 0.0)


def _mean_attack_probability(stage_rows: List[Dict[str, object]]) -> float:
    if not stage_rows:
        return 0.0
    return sum(float(row["expected_attack_probability"]) for row in stage_rows) / len(stage_rows)


def horizon_sweep(
    strategy_name: str,
    config: GameConfig,
    initial_belief: float,
    *,
    max_horizon: int = HORIZON_SWEEP_MAX,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for horizon in range(1, max_horizon + 1):
        horizon_config = replace(config, horizon=horizon)
        solution = None
        if strategy_name != TRUTHFUL_BASELINE:
            solution = backward_induction(strategy_name, horizon_config, initial_belief)
        forward_result = forward_simulation(strategy_name, horizon_config, initial_belief, solution)
        rows.append(
            {
                "horizon": horizon,
                "discounted_cumulative_defender_utility": forward_result["discounted_cumulative_defender_utility"],
                "discounted_cumulative_attacker_utility": forward_result["discounted_cumulative_attacker_utility"],
                "undiscounted_cumulative_defender_utility": forward_result["cumulative_defender_utility"],
                "undiscounted_cumulative_attacker_utility": forward_result["cumulative_attacker_utility"],
                "mean_attack_probability": _mean_attack_probability(forward_result["stage_rows"]),
                "terminal_public_belief_mean": _terminal_belief_mean_from_distribution(
                    forward_result["final_belief_distribution"]
                ),
                "terminal_uncertainty_probability": _terminal_uncertainty_probability_from_distribution(
                    forward_result["final_belief_distribution"],
                    attacker_indifference_threshold(horizon_config),
                ),
            }
        )
    return rows


def run_experiment_one(config: GameConfig) -> Dict[str, object]:
    threshold = attacker_indifference_threshold(config)
    scenarios = _build_experiment_scenarios(config)

    results: Dict[str, object] = {
        "experiment_name": "实验一：收益对比实验",
        "figure_title": "不同场景下防御方折扣累计期望收益对比",
        "threshold_belief": threshold,
        "discount_factor": config.gamma,
        "horizon_sweep_max": HORIZON_SWEEP_MAX,
        "scenarios": {},
    }

    for scenario_key, scenario in scenarios.items():
        initial_belief = scenario["initial_belief"]
        scenario_config = scenario["config"]
        solution = backward_induction(scenario["pbne_strategy"], scenario_config, initial_belief)
        pbne_result = forward_simulation(scenario["pbne_strategy"], scenario_config, initial_belief, solution)
        truthful_result = forward_simulation(TRUTHFUL_BASELINE, scenario_config, initial_belief)
        pbne_horizon_rows = horizon_sweep(scenario["pbne_strategy"], scenario_config, initial_belief, max_horizon=HORIZON_SWEEP_MAX)
        truthful_horizon_rows = horizon_sweep(TRUTHFUL_BASELINE, scenario_config, initial_belief, max_horizon=HORIZON_SWEEP_MAX)

        results["scenarios"][scenario_key] = {
            "label": scenario["label"],
            "description": scenario["description"],
            "initial_belief": initial_belief,
            "config": {
                "horizon": scenario_config.horizon,
                "gamma": scenario_config.gamma,
                "c_theta1": scenario_config.c_theta1,
                "c_theta2": scenario_config.c_theta2,
            },
            "pbne_strategy_name": scenario["pbne_strategy"],
            "pbne_label": scenario["pbne_label"],
            "truthful_label": "真实披露基线",
            "results": {
                "recursive_pbne": pbne_result,
                "truthful_baseline": truthful_result,
            },
            "horizon_sweep": {
                "recursive_pbne": pbne_horizon_rows,
                "truthful_baseline": truthful_horizon_rows,
            },
            "backward_induction": {
                "strategy_name": solution.strategy_name,
                "initial_belief": solution.initial_belief,
                "threshold_belief": solution.threshold_belief,
                "stage_records": solution.stage_records,
            },
        }

    return results


def run_experiment_two(config: GameConfig) -> Dict[str, object]:
    threshold = attacker_indifference_threshold(config)
    scenarios = _build_experiment_scenarios(config)
    seed_map = _build_experiment_two_seed_map()
    results: Dict[str, object] = {
        "experiment_name": "实验二：信念演化与终局分布实验",
        "trajectory_figure_title": "公共信念状态概率演化",
        "terminal_figure_title": "终局不确定信念概率对比",
        "threshold_belief": threshold,
        "monte_carlo_runs": BELIEF_MONTE_CARLO_RUNS,
        "base_seed": BELIEF_MONTE_CARLO_SEED,
        "scenarios": {},
    }

    for scenario_key, scenario in scenarios.items():
        initial_belief = scenario["initial_belief"]
        scenario_config = scenario["config"]
        solution = backward_induction(scenario["pbne_strategy"], scenario_config, initial_belief)
        scenario_results: Dict[str, Dict[str, object]] = {}

        strategy_map = {
            "recursive_pbne": {
                "strategy_name": scenario["pbne_strategy"],
                "strategy_label": scenario["pbne_label"],
                "solution": solution,
            },
            "truthful_baseline": {
                "strategy_name": TRUTHFUL_BASELINE,
                "strategy_label": "真实披露基线",
                "solution": None,
            },
        }

        for strategy_key, strategy_spec in strategy_map.items():
            seed = seed_map[scenario_key][strategy_key]
            exact_forward_result = forward_simulation(
                strategy_spec["strategy_name"],
                scenario_config,
                initial_belief,
                strategy_spec["solution"],
            )
            trajectory_result = belief_trajectory(
                strategy_spec["strategy_name"],
                scenario_config,
                initial_belief,
                strategy_spec["solution"],
                num_runs=BELIEF_MONTE_CARLO_RUNS,
                seed=seed,
            )
            terminal_result = terminal_belief_distribution(
                strategy_spec["strategy_name"],
                scenario_config,
                initial_belief,
                strategy_spec["solution"],
                num_runs=BELIEF_MONTE_CARLO_RUNS,
                seed=seed,
                simulations=trajectory_result["simulations"],
            )
            raw_trajectory_rows: List[Dict[str, object]] = []
            for run_id, simulation in enumerate(trajectory_result["simulations"], start=1):
                belief_path = simulation["belief_path"]
                raw_trajectory_rows.append(
                    {
                        "run_id": run_id,
                        "stage": 1,
                        "defender_type": simulation["defender_type"],
                        "signal": "initial",
                        "public_belief": belief_path[0],
                    }
                )
                for stage_record in simulation["stage_records"]:
                    raw_trajectory_rows.append(
                        {
                            "run_id": run_id,
                            "stage": stage_record["stage"] + 1,
                            "defender_type": stage_record["defender_type"],
                            "signal": stage_record["signal"],
                            "public_belief": stage_record["next_belief_theta1"],
                        }
                    )

            scenario_results[strategy_key] = {
                "strategy_name": strategy_spec["strategy_name"],
                "strategy_label": strategy_spec["strategy_label"],
                "seed": seed,
                "selected_trajectory": trajectory_result["selected_trajectory"],
                "average_trajectory": trajectory_result["average_trajectory"],
                "selected_stage_records": trajectory_result["selected_stage_records"],
                "state_probability_evolution": _state_probability_evolution(
                    initial_belief,
                    threshold,
                    exact_forward_result,
                ),
                "terminal_probability_mass": terminal_result["probability_mass"],
                "terminal_belief_mean": sum(terminal_result["terminal_beliefs"]) / terminal_result["num_runs"],
                "terminal_uncertainty_probability": terminal_result["probability_mass"].get(_belief_key(threshold), 0.0),
                "raw_trajectory_rows": raw_trajectory_rows,
                "terminal_belief_samples": terminal_result["terminal_beliefs"],
            }

        results["scenarios"][scenario_key] = {
            "label": scenario["label"],
            "description": scenario["description"],
            "initial_belief": initial_belief,
            "config": {
                "horizon": scenario_config.horizon,
                "gamma": scenario_config.gamma,
                "c_theta1": scenario_config.c_theta1,
                "c_theta2": scenario_config.c_theta2,
            },
            "pbne_label": scenario["pbne_label"],
            "truthful_label": "真实披露基线",
            "results": scenario_results,
        }

    return results


def sensitivity_analysis(
    parameter_name: str,
    grid_values: List[float],
    fixed_config: GameConfig,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    for value in grid_values:
        perturbed_config = replace(fixed_config, **{parameter_name: value})
        initial_belief = value if parameter_name == "prior_theta1" else perturbed_config.prior_theta1
        threshold = attacker_indifference_threshold(perturbed_config)

        try:
            strategy_name = _select_pbne_strategy(initial_belief, perturbed_config)
            solution = backward_induction(strategy_name, perturbed_config, initial_belief)
            forward_result = forward_simulation(strategy_name, perturbed_config, initial_belief, solution)
            final_belief_distribution = forward_result["final_belief_distribution"]
            rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_label": SENSITIVITY_PARAMETER_LABELS[parameter_name],
                    "parameter_value": value,
                    "initial_belief": initial_belief,
                    "threshold_belief": threshold,
                    "strategy_name": strategy_name,
                    "cumulative_defender_utility": forward_result["cumulative_defender_utility"],
                    "cumulative_attacker_utility": forward_result["cumulative_attacker_utility"],
                    "mean_attack_probability": _mean_attack_probability(forward_result["stage_rows"]),
                    "terminal_attack_probability": forward_result["stage_rows"][-1]["expected_attack_probability"],
                    "terminal_public_belief_mean": _terminal_belief_mean_from_distribution(final_belief_distribution),
                    "terminal_uncertainty_probability": _terminal_uncertainty_probability_from_distribution(
                        final_belief_distribution,
                        threshold,
                    ),
                    "feasible": True,
                    "error_message": "",
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_label": SENSITIVITY_PARAMETER_LABELS[parameter_name],
                    "parameter_value": value,
                    "initial_belief": initial_belief,
                    "threshold_belief": threshold,
                    "strategy_name": "",
                    "cumulative_defender_utility": None,
                    "cumulative_attacker_utility": None,
                    "mean_attack_probability": None,
                    "terminal_attack_probability": None,
                    "terminal_public_belief_mean": None,
                    "terminal_uncertainty_probability": None,
                    "feasible": False,
                    "error_message": str(exc),
                }
            )

    return {
        "parameter_name": parameter_name,
        "parameter_label": SENSITIVITY_PARAMETER_LABELS[parameter_name],
        "rows": rows,
    }


def _build_sensitivity_plot_panel(
    panel_spec: Dict[str, object],
    base_config: GameConfig,
) -> Dict[str, object]:
    parameter_name = panel_spec["parameter_name"]
    strategy_name = panel_spec["strategy_name"]
    series: List[Dict[str, object]] = []

    for parameter_value in panel_spec["grid_values"]:
        perturbed_config = replace(base_config, **{parameter_name: parameter_value})
        initial_belief = parameter_value if parameter_name == "prior_theta1" else perturbed_config.prior_theta1
        horizon_rows = horizon_sweep(strategy_name, perturbed_config, initial_belief, max_horizon=HORIZON_SWEEP_MAX)
        final_row = horizon_rows[-1]
        series.append(
            {
                "parameter_value": parameter_value,
                "strategy_name": strategy_name,
                "initial_belief": initial_belief,
                "horizon_rows": horizon_rows,
                "discounted_cumulative_defender_utility": final_row["discounted_cumulative_defender_utility"],
                "mean_attack_probability": final_row["mean_attack_probability"],
                "terminal_public_belief_mean": final_row["terminal_public_belief_mean"],
            }
        )

    return {
        "panel_key": panel_spec["panel_key"],
        "parameter_name": parameter_name,
        "parameter_label": SENSITIVITY_PARAMETER_LABELS[parameter_name],
        "scenario_key": panel_spec["scenario_key"],
        "title": panel_spec["title"],
        "series": series,
    }


def run_experiment_three(config: GameConfig) -> Dict[str, object]:
    scenarios = _build_experiment_scenarios(config)
    results: Dict[str, object] = {
        "experiment_name": "实验三：参数敏感性分析",
        "figure_title": "关键参数变化下递归 PBNE 防御方折扣累计期望收益敏感性分析",
        "parameters": {},
        "plot_panels": [],
    }

    for parameter_name, grid_map in SENSITIVITY_GRID_MAP.items():
        parameter_result = {
            "parameter_name": parameter_name,
            "parameter_label": SENSITIVITY_PARAMETER_LABELS[parameter_name],
            "scenarios": {},
        }
        for scenario_key, scenario in scenarios.items():
            fixed_config = scenario["config"]
            sensitivity_result = sensitivity_analysis(parameter_name, grid_map[scenario_key], fixed_config)
            parameter_result["scenarios"][scenario_key] = {
                "label": scenario["label"],
                "description": scenario["description"],
                "strategy_label": scenario["pbne_label"],
                "fixed_config": {
                    "prior_theta1": fixed_config.prior_theta1,
                    "gamma": fixed_config.gamma,
                    "c_theta1": fixed_config.c_theta1,
                    "c_theta2": fixed_config.c_theta2,
                },
                "rows": sensitivity_result["rows"],
            }
        results["parameters"][parameter_name] = parameter_result

    for panel_spec in SENSITIVITY_PLOT_SPECS:
        scenario_config = scenarios[panel_spec["scenario_key"]]["config"]
        results["plot_panels"].append(_build_sensitivity_plot_panel(panel_spec, scenario_config))

    return results


def build_summary_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            final_horizon_row = scenario["horizon_sweep"][strategy_key][-1]
            rows.append(
                {
                    "scenario": scenario_key,
                    "scenario_label": scenario["label"],
                    "initial_belief": scenario["initial_belief"],
                    "gamma": experiment_result["discount_factor"],
                    "c_theta1": scenario["config"]["c_theta1"],
                    "c_theta2": scenario["config"]["c_theta2"],
                    "strategy": strategy_key,
                    "strategy_label": scenario["pbne_label"] if strategy_key == "recursive_pbne" else scenario["truthful_label"],
                    "cumulative_defender_utility": strategy_result["cumulative_defender_utility"],
                    "cumulative_attacker_utility": strategy_result["cumulative_attacker_utility"],
                    "discounted_cumulative_defender_utility": final_horizon_row["discounted_cumulative_defender_utility"],
                    "discounted_cumulative_attacker_utility": final_horizon_row["discounted_cumulative_attacker_utility"],
                }
            )
    return rows


def build_stage_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, horizon_rows in scenario["horizon_sweep"].items():
            strategy_label = scenario["pbne_label"] if strategy_key == "recursive_pbne" else scenario["truthful_label"]
            for stage_row in horizon_rows:
                rows.append(
                    {
                        "scenario": scenario_key,
                        "scenario_label": scenario["label"],
                        "initial_belief": scenario["initial_belief"],
                        "gamma": experiment_result["discount_factor"],
                        "c_theta1": scenario["config"]["c_theta1"],
                        "c_theta2": scenario["config"]["c_theta2"],
                        "strategy": strategy_key,
                        "strategy_label": strategy_label,
                        "horizon": stage_row["horizon"],
                        "discounted_cumulative_defender_utility": stage_row["discounted_cumulative_defender_utility"],
                        "discounted_cumulative_attacker_utility": stage_row["discounted_cumulative_attacker_utility"],
                        "undiscounted_cumulative_defender_utility": stage_row["undiscounted_cumulative_defender_utility"],
                        "undiscounted_cumulative_attacker_utility": stage_row["undiscounted_cumulative_attacker_utility"],
                        "mean_attack_probability": stage_row["mean_attack_probability"],
                        "terminal_public_belief_mean": stage_row["terminal_public_belief_mean"],
                        "terminal_uncertainty_probability": stage_row["terminal_uncertainty_probability"],
                    }
                )
    return rows


def build_belief_trajectory_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            for row in strategy_result["raw_trajectory_rows"]:
                rows.append(
                    {
                        "scenario": scenario_key,
                        "scenario_label": scenario["label"],
                        "initial_belief": scenario["initial_belief"],
                        "c_theta1": scenario["config"]["c_theta1"],
                        "c_theta2": scenario["config"]["c_theta2"],
                        "strategy": strategy_key,
                        "strategy_label": strategy_result["strategy_label"],
                        "seed": strategy_result["seed"],
                        "run_id": row["run_id"],
                        "stage": row["stage"],
                        "defender_type": row["defender_type"],
                        "signal": row["signal"],
                        "public_belief": row["public_belief"],
                    }
                )
    return rows


def build_terminal_belief_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            for run_id, terminal_belief in enumerate(strategy_result["terminal_belief_samples"], start=1):
                rows.append(
                    {
                        "scenario": scenario_key,
                        "scenario_label": scenario["label"],
                        "initial_belief": scenario["initial_belief"],
                        "c_theta1": scenario["config"]["c_theta1"],
                        "c_theta2": scenario["config"]["c_theta2"],
                        "strategy": strategy_key,
                        "strategy_label": strategy_result["strategy_label"],
                        "seed": strategy_result["seed"],
                        "run_id": run_id,
                        "terminal_public_belief": terminal_belief,
                    }
                )
    return rows


def build_state_probability_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            for row in strategy_result["state_probability_evolution"]:
                rows.append(
                    {
                        "scenario": scenario_key,
                        "scenario_label": scenario["label"],
                        "initial_belief": scenario["initial_belief"],
                        "strategy": strategy_key,
                        "strategy_label": strategy_result["strategy_label"],
                        "stage": row["stage"],
                        "certain_honeypot_probability": row["certain_honeypot"],
                        "critical_uncertainty_probability": row["critical_uncertainty"],
                        "certain_real_probability": row["certain_real"],
                        "other_probability": row["other"],
                    }
                )
    return rows


def build_terminal_uncertainty_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            rows.append(
                {
                    "scenario": scenario_key,
                    "scenario_label": scenario["label"],
                    "initial_belief": scenario["initial_belief"],
                    "strategy": strategy_key,
                    "strategy_label": strategy_result["strategy_label"],
                    "terminal_uncertainty_probability": strategy_result["terminal_uncertainty_probability"],
                    "terminal_belief_mean": strategy_result["terminal_belief_mean"],
                }
            )
    return rows


def build_terminal_state_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    threshold_key = _belief_key(experiment_result["threshold_belief"])
    for scenario_key, scenario in experiment_result["scenarios"].items():
        for strategy_key, strategy_result in scenario["results"].items():
            probability_mass = strategy_result["terminal_probability_mass"]
            rows.append(
                {
                    "scenario": scenario_key,
                    "scenario_label": scenario["label"],
                    "initial_belief": scenario["initial_belief"],
                    "strategy": strategy_key,
                    "strategy_label": strategy_result["strategy_label"],
                    "certain_honeypot_probability": probability_mass.get("0.000000", 0.0),
                    "critical_uncertainty_probability": probability_mass.get(threshold_key, 0.0),
                    "certain_real_probability": probability_mass.get("1.000000", 0.0),
                }
            )
    return rows


def build_experiment_two_summary(experiment_result: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "experiment_name": experiment_result["experiment_name"],
        "trajectory_figure_title": "公共信念随阶段变化曲线",
        "terminal_figure_title": "终局公共信念分布对比",
        "threshold_belief": experiment_result["threshold_belief"],
        "monte_carlo_runs": experiment_result["monte_carlo_runs"],
        "base_seed": experiment_result["base_seed"],
        "scenarios": {},
    }
    for scenario_key, scenario in experiment_result["scenarios"].items():
        summary["scenarios"][scenario_key] = {
            "label": scenario["label"],
            "description": scenario["description"],
            "initial_belief": scenario["initial_belief"],
            "config": scenario["config"],
            "pbne_label": scenario["pbne_label"],
            "truthful_label": scenario["truthful_label"],
            "results": {},
        }
        for strategy_key, strategy_result in scenario["results"].items():
            summary["scenarios"][scenario_key]["results"][strategy_key] = {
                "strategy_name": strategy_result["strategy_name"],
                "strategy_label": strategy_result["strategy_label"],
                "seed": strategy_result["seed"],
                "selected_trajectory": strategy_result["selected_trajectory"],
                "average_trajectory": strategy_result["average_trajectory"],
                "selected_stage_records": strategy_result["selected_stage_records"],
                "state_probability_evolution": strategy_result["state_probability_evolution"],
                "terminal_probability_mass": strategy_result["terminal_probability_mass"],
                "terminal_belief_mean": strategy_result["terminal_belief_mean"],
                "terminal_uncertainty_probability": strategy_result["terminal_uncertainty_probability"],
            }
    return summary


def build_sensitivity_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for parameter_name, parameter_result in experiment_result["parameters"].items():
        for scenario_key, scenario_result in parameter_result["scenarios"].items():
            for row in scenario_result["rows"]:
                rows.append(
                    {
                        "parameter_name": parameter_name,
                        "parameter_label": parameter_result["parameter_label"],
                        "scenario": scenario_key,
                        "scenario_label": scenario_result["label"],
                        "strategy_label": scenario_result["strategy_label"],
                        "parameter_value": row["parameter_value"],
                        "initial_belief": row["initial_belief"],
                        "threshold_belief": row["threshold_belief"],
                        "strategy_name": row["strategy_name"],
                        "cumulative_defender_utility": row["cumulative_defender_utility"],
                        "cumulative_attacker_utility": row["cumulative_attacker_utility"],
                        "mean_attack_probability": row["mean_attack_probability"],
                        "terminal_attack_probability": row["terminal_attack_probability"],
                        "terminal_public_belief_mean": row["terminal_public_belief_mean"],
                        "terminal_uncertainty_probability": row["terminal_uncertainty_probability"],
                        "feasible": row["feasible"],
                        "error_message": row["error_message"],
                    }
                )
    return rows


def build_sensitivity_trajectory_rows(experiment_result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for panel in experiment_result["plot_panels"]:
        for series in panel["series"]:
            for stage_row in series["horizon_rows"]:
                rows.append(
                    {
                        "panel_key": panel["panel_key"],
                        "panel_title": panel["title"],
                        "parameter_name": panel["parameter_name"],
                        "parameter_label": panel["parameter_label"],
                        "scenario": panel["scenario_key"],
                        "parameter_value": series["parameter_value"],
                        "horizon": stage_row["horizon"],
                        "discounted_cumulative_defender_utility": stage_row["discounted_cumulative_defender_utility"],
                        "discounted_cumulative_attacker_utility": stage_row["discounted_cumulative_attacker_utility"],
                        "undiscounted_cumulative_defender_utility": stage_row["undiscounted_cumulative_defender_utility"],
                        "mean_attack_probability": stage_row["mean_attack_probability"],
                        "terminal_public_belief_mean": stage_row["terminal_public_belief_mean"],
                    }
                )
    return rows
