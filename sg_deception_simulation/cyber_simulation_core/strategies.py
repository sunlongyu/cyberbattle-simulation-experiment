from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import GameConfig
from .model import DefenderType, Signal, attacker_indifference_threshold, bayes_update


TRUTHFUL_BASELINE = "truthful_baseline"
RECURSIVE_PBNE_PRODUCTION = "recursive_pbne_production"
RECURSIVE_PBNE_HONEYPOT = "recursive_pbne_honeypot"


@dataclass(frozen=True)
class StagePolicy:
    stage: int
    strategy_name: str
    belief_theta1: float
    defender_signal_probs: Dict[DefenderType, Dict[Signal, float]]
    attack_prob_by_signal: Dict[Signal, float]
    next_belief_by_signal: Dict[Signal, float]
    mixing_metrics: Dict[str, float]


def _next_belief_map(
    belief_theta1: float,
    defender_signal_probs: Dict[DefenderType, Dict[Signal, float]],
) -> Dict[Signal, float]:
    next_beliefs: Dict[Signal, float] = {}
    for signal in (Signal.SIGMA1, Signal.SIGMA2):
        lambda_theta1 = defender_signal_probs[DefenderType.THETA1][signal]
        lambda_theta2 = defender_signal_probs[DefenderType.THETA2][signal]
        denominator = belief_theta1 * lambda_theta1 + (1.0 - belief_theta1) * lambda_theta2

        if denominator > 0.0:
            next_beliefs[signal] = bayes_update(belief_theta1, signal, defender_signal_probs)
        elif lambda_theta1 > 0.0 and lambda_theta2 == 0.0:
            next_beliefs[signal] = 1.0
        elif lambda_theta1 == 0.0 and lambda_theta2 > 0.0:
            next_beliefs[signal] = 0.0
        else:
            next_beliefs[signal] = belief_theta1
    return next_beliefs


def _normalize_probability(value: float, *, label: str) -> float:
    tolerance = 1e-9
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"{label}={value:.6f} is outside [0, 1].")
    return min(max(value, 0.0), 1.0)


def truthful_stage_policy(stage: int, belief_theta1: float, config: GameConfig) -> StagePolicy:
    del config
    defender_signal_probs = {
        DefenderType.THETA1: {Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
        DefenderType.THETA2: {Signal.SIGMA1: 0.0, Signal.SIGMA2: 1.0},
    }
    return StagePolicy(
        stage=stage,
        strategy_name=TRUTHFUL_BASELINE,
        belief_theta1=belief_theta1,
        defender_signal_probs=defender_signal_probs,
        attack_prob_by_signal={Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
        next_belief_by_signal=_next_belief_map(belief_theta1, defender_signal_probs),
        mixing_metrics={},
    )


def production_mix_probability(belief_theta1: float, config: GameConfig) -> float:
    delta_r = config.attack_gain - config.attack_cost
    delta_h = config.attack_cost + config.intel_loss + config.attacker_deception_penalty
    if belief_theta1 <= 0.0:
        return 1.0
    value = ((1.0 - belief_theta1) * delta_h) / (belief_theta1 * delta_r)
    return _normalize_probability(value, label="x_t")


def honeypot_mix_probability(belief_theta1: float, config: GameConfig) -> float:
    delta_r = config.attack_gain - config.attack_cost
    delta_h = config.attack_cost + config.intel_loss + config.attacker_deception_penalty
    if belief_theta1 >= 1.0:
        return 1.0
    value = (belief_theta1 * delta_r) / ((1.0 - belief_theta1) * delta_h)
    return _normalize_probability(value, label="z_t")


def production_camouflage_stage_policy(
    stage: int,
    belief_theta1: float,
    attack_after_sigma2: float,
    config: GameConfig,
) -> StagePolicy:
    x_t = production_mix_probability(belief_theta1, config)
    attack_after_sigma2 = _normalize_probability(attack_after_sigma2, label="y_t")
    defender_signal_probs = {
        DefenderType.THETA1: {Signal.SIGMA1: 1.0 - x_t, Signal.SIGMA2: x_t},
        DefenderType.THETA2: {Signal.SIGMA1: 0.0, Signal.SIGMA2: 1.0},
    }
    return StagePolicy(
        stage=stage,
        strategy_name=RECURSIVE_PBNE_PRODUCTION,
        belief_theta1=belief_theta1,
        defender_signal_probs=defender_signal_probs,
        attack_prob_by_signal={Signal.SIGMA1: 1.0, Signal.SIGMA2: attack_after_sigma2},
        next_belief_by_signal=_next_belief_map(belief_theta1, defender_signal_probs),
        mixing_metrics={
            "x_t": x_t,
            "y_t": attack_after_sigma2,
        },
    )


def honeypot_camouflage_stage_policy(
    stage: int,
    belief_theta1: float,
    attack_after_sigma1: float,
    config: GameConfig,
) -> StagePolicy:
    z_t = honeypot_mix_probability(belief_theta1, config)
    attack_after_sigma1 = _normalize_probability(attack_after_sigma1, label="q_t")
    defender_signal_probs = {
        DefenderType.THETA1: {Signal.SIGMA1: 1.0, Signal.SIGMA2: 0.0},
        DefenderType.THETA2: {Signal.SIGMA1: z_t, Signal.SIGMA2: 1.0 - z_t},
    }
    return StagePolicy(
        stage=stage,
        strategy_name=RECURSIVE_PBNE_HONEYPOT,
        belief_theta1=belief_theta1,
        defender_signal_probs=defender_signal_probs,
        attack_prob_by_signal={Signal.SIGMA1: attack_after_sigma1, Signal.SIGMA2: 0.0},
        next_belief_by_signal=_next_belief_map(belief_theta1, defender_signal_probs),
        mixing_metrics={
            "z_t": z_t,
            "q_t": attack_after_sigma1,
        },
    )
