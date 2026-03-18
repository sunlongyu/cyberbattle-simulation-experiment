from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import GameConfig
from .model import AttackerAction, DefenderType, Signal, clip_mixed_probability


@dataclass(frozen=True)
class StrategyState:
    belief_theta1: float
    config: GameConfig


@dataclass(frozen=True)
class Regime:
    name: str
    defender_signal_probs: Dict[DefenderType, Dict[Signal, float]]
    attack_prob_by_signal: Dict[Signal, float]
    mixing_metrics: Dict[str, float]


def truthful_baseline(config: GameConfig) -> Regime:
    return Regime(
        name="truthful_baseline",
        defender_signal_probs={
            DefenderType.THETA1: {Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
            DefenderType.THETA2: {Signal.SIGMA1: 0.0, Signal.SIGMA2: 1.0},
        },
        attack_prob_by_signal={Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
        mixing_metrics={},
    )


def pbne_production_camouflage(state: StrategyState) -> Regime:
    p = clip_mixed_probability(state.belief_theta1, state.config.epsilon)
    c = state.config
    lambda_d_star = (
        (1.0 - p) * (c.attack_cost + c.intel_loss + c.attacker_deception_penalty)
    ) / (p * (c.attack_gain - c.attack_cost))
    lambda_d_star = clip_mixed_probability(lambda_d_star, c.epsilon)
    lambda_a_star = 1.0 - c.c_theta1 / (
        c.defender_loss + c.defender_pressure_cost + c.defender_protection_gain
    )
    lambda_a_star = clip_mixed_probability(lambda_a_star, c.epsilon)

    return Regime(
        name="pbne_production_camouflage",
        defender_signal_probs={
            DefenderType.THETA1: {Signal.SIGMA1: 1.0 - lambda_d_star, Signal.SIGMA2: lambda_d_star},
            DefenderType.THETA2: {Signal.SIGMA1: 0.0, Signal.SIGMA2: 1.0},
        },
        attack_prob_by_signal={Signal.SIGMA1: 1.0, Signal.SIGMA2: lambda_a_star},
        mixing_metrics={
            "lambda_d_star": lambda_d_star,
            "lambda_a_star": lambda_a_star,
        },
    )


def pbne_honeypot_camouflage(state: StrategyState) -> Regime:
    p = clip_mixed_probability(state.belief_theta1, state.config.epsilon)
    c = state.config
    lambda_d_prime = (
        p * (c.attack_gain - c.attack_cost)
    ) / ((1.0 - p) * (c.attack_cost + c.intel_loss + c.attacker_deception_penalty))
    lambda_d_prime = clip_mixed_probability(lambda_d_prime, c.epsilon)
    lambda_a_prime = (
        c.intel_gain + c.defender_deception_bonus - c.c_theta2
    ) / (c.intel_gain + c.defender_deception_bonus)
    lambda_a_prime = clip_mixed_probability(lambda_a_prime, c.epsilon)
    attack_after_sigma1 = 1.0 - lambda_a_prime

    return Regime(
        name="pbne_honeypot_camouflage",
        defender_signal_probs={
            DefenderType.THETA1: {Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
            DefenderType.THETA2: {Signal.SIGMA1: lambda_d_prime, Signal.SIGMA2: 1.0 - lambda_d_prime},
        },
        attack_prob_by_signal={Signal.SIGMA1: attack_after_sigma1, Signal.SIGMA2: 0.0},
        mixing_metrics={
            "lambda_d_prime": lambda_d_prime,
            "lambda_a_prime": lambda_a_prime,
            "attack_after_sigma1": attack_after_sigma1,
        },
    )


STRATEGY_BUILDERS = {
    "truthful_baseline": lambda state: truthful_baseline(state.config),
    "pbne_production_camouflage": pbne_production_camouflage,
    "pbne_honeypot_camouflage": pbne_honeypot_camouflage,
}
