from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    horizon: int = 10
    prior_theta1: float = 0.5
    beta: float = 1.0
    epsilon: float = 0.02

    attack_cost: float = 2.0
    attack_gain: float = 12.0
    defender_loss: float = 10.0

    intel_gain: float = 8.0
    intel_loss: float = 8.0

    c_theta1: float = 2.5
    c_theta2: float = 1.5

    monte_carlo_runs: int = 4000


DEFAULT_CONFIG = GameConfig()
