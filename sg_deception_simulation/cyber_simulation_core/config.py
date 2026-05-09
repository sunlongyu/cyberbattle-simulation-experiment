from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    horizon: int = 10
    prior_theta1: float = 0.5
    gamma: float = 0.85

    attack_cost: float = 2.0
    attack_gain: float = 12.0
    defender_loss: float = 10.0
    defender_pressure_cost: float = 2.0
    defender_protection_gain: float = 6.0

    intel_gain: float = 8.0
    intel_loss: float = 8.0
    defender_deception_bonus: float = 2.0
    attacker_deception_penalty: float = 2.0

    c_theta1: float = 2.5
    c_theta2: float = 1.5


DEFAULT_CONFIG = GameConfig()
