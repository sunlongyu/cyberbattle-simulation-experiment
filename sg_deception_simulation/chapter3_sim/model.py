from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

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


@dataclass
class StageOutcome:
    stage: int
    belief_theta1: float
    signal: Signal
    attack_probability: float
    action: AttackerAction
    defender_utility: float
    attacker_utility: float
    regime_metrics: Dict[str, float]


def clip_mixed_probability(value: float, epsilon: float) -> float:
    return min(max(value, epsilon), 1.0 - epsilon)


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


def discounted_belief(
    prior_theta1: float,
    signal_history: List[Signal],
    type_signal_probs: Dict[DefenderType, Dict[Signal, float]],
    beta: float,
) -> float:
    prior_theta1 = min(max(prior_theta1, 1e-9), 1.0 - 1e-9)
    prior_theta2 = 1.0 - prior_theta1

    score_theta1 = math.log(prior_theta1)
    score_theta2 = math.log(prior_theta2)
    horizon = len(signal_history)

    for index, signal in enumerate(signal_history, start=1):
        weight = math.exp(-beta * (horizon - index))
        score_theta1 += weight * math.log(max(type_signal_probs[DefenderType.THETA1][signal], 1e-9))
        score_theta2 += weight * math.log(max(type_signal_probs[DefenderType.THETA2][signal], 1e-9))

    max_score = max(score_theta1, score_theta2)
    exp_theta1 = math.exp(score_theta1 - max_score)
    exp_theta2 = math.exp(score_theta2 - max_score)
    return exp_theta1 / (exp_theta1 + exp_theta2)
