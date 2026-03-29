"""Shared experiment infrastructure for Chapter 4 MARL runs."""

from .chapter4_env import (
    ATTACKER,
    DEFENDER,
    create_rllib_env,
    policy_mapping_fn,
    register_chapter4_env,
)
from .config import ExperimentConfig, OutputConfig
from .defaults import (
    CHAPTER4_ENV_DEFAULTS,
    CHAPTER4_MODULE_IDS,
    CHAPTER4_TRAINING_DEFAULTS,
    CONVERGENCE_RULE_DEFAULTS,
)
from .io import ExperimentArtifacts, EpisodeLogger, SummaryWriter
from .metrics import compute_convergence_step, compute_reward_volatility
from .naming import figure_name, table_name
from .paths import ensure_module_layout, module_root

__all__ = [
    "EpisodeLogger",
    "ExperimentArtifacts",
    "ExperimentConfig",
    "OutputConfig",
    "SummaryWriter",
    "CHAPTER4_ENV_DEFAULTS",
    "CHAPTER4_MODULE_IDS",
    "CHAPTER4_TRAINING_DEFAULTS",
    "CONVERGENCE_RULE_DEFAULTS",
    "ATTACKER",
    "DEFENDER",
    "compute_convergence_step",
    "compute_reward_volatility",
    "create_rllib_env",
    "ensure_module_layout",
    "figure_name",
    "module_root",
    "policy_mapping_fn",
    "register_chapter4_env",
    "table_name",
]
