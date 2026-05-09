from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

from .config import GameConfig


class DefenderType(str, Enum):
    THETA1 = "theta1"
    THETA2 = "theta2"


class Signal(str, Enum):
    SIGMA1 = "sigma1"
    SIGMA2 = "sigma2"


class AttackerAction(str, Enum):
    ATTACK = "attack"
    RETREAT = "retreat"


DEFENDER_TYPES = (DefenderType.THETA1, DefenderType.THETA2)
SIGNALS = (Signal.SIGMA1, Signal.SIGMA2)


def camouflage_cost(defender_type: DefenderType, signal: Signal, config: GameConfig) -> float:
    if defender_type == DefenderType.THETA1 and signal == Signal.SIGMA2:
        return config.c_theta1
    if defender_type == DefenderType.THETA2 and signal == Signal.SIGMA1:
        return config.c_theta2
    return 0.0


def utility(
    defender_type: DefenderType,
    signal: Signal,
    action: AttackerAction,
    config: GameConfig,
) -> Tuple[float, float]:
    disguise_cost = camouflage_cost(defender_type, signal, config)

    if action == AttackerAction.RETREAT:
        if defender_type == DefenderType.THETA1:
            return config.defender_protection_gain - disguise_cost, 0.0
        return -disguise_cost, 0.0

    if defender_type == DefenderType.THETA1:
        return (
            -(config.defender_loss + config.defender_pressure_cost + disguise_cost),
            config.attack_gain - config.attack_cost,
        )

    return (
        config.intel_gain + config.defender_deception_bonus - disguise_cost,
        -(config.attack_cost + config.intel_loss + config.attacker_deception_penalty),
    )


def defender_expected_utility(
    defender_type: DefenderType,
    signal: Signal,
    attack_probability: float,
    config: GameConfig,
) -> float:
    attack_value, _ = utility(defender_type, signal, AttackerAction.ATTACK, config)
    retreat_value, _ = utility(defender_type, signal, AttackerAction.RETREAT, config)
    return attack_probability * attack_value + (1.0 - attack_probability) * retreat_value


def attacker_expected_utility(
    defender_type: DefenderType,
    signal: Signal,
    attack_probability: float,
    config: GameConfig,
) -> float:
    _, attack_value = utility(defender_type, signal, AttackerAction.ATTACK, config)
    _, retreat_value = utility(defender_type, signal, AttackerAction.RETREAT, config)
    return attack_probability * attack_value + (1.0 - attack_probability) * retreat_value


def attacker_indifference_threshold(config: GameConfig) -> float:
    delta_r = config.attack_gain - config.attack_cost
    delta_h = config.attack_cost + config.intel_loss + config.attacker_deception_penalty
    return delta_h / (delta_r + delta_h)


def bayes_update(
    prior_theta1: float,
    signal: Signal,
    type_signal_probs: Dict[DefenderType, Dict[Signal, float]],
) -> float:
    lambda_theta1 = type_signal_probs[DefenderType.THETA1][signal]
    lambda_theta2 = type_signal_probs[DefenderType.THETA2][signal]
    denominator = prior_theta1 * lambda_theta1 + (1.0 - prior_theta1) * lambda_theta2
    if denominator <= 0.0:
        raise ValueError(f"Bayesian update is undefined for prior={prior_theta1:.6f}, signal={signal.value}.")
    return (prior_theta1 * lambda_theta1) / denominator
