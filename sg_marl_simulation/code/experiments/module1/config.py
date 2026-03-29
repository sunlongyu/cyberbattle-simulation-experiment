from marl_core.defaults import (
    CHAPTER4_ENV_DEFAULTS,
    CHAPTER4_TRAINING_DEFAULTS,
    CONVERGENCE_RULE_DEFAULTS,
)

MODULE1_LEARNING_RATES = [2e-4, 1e-4, 7e-5, 5e-5]
MODULE1_DEFAULT_SEEDS = [0, 1, 2, 3, 4]

MODULE1_ENV = dict(CHAPTER4_ENV_DEFAULTS)
MODULE1_ENV["history_length"] = 10

MODULE1_TRAINING = dict(CHAPTER4_TRAINING_DEFAULTS)
MODULE1_TRAINING.update(
    {
        "num_gpus": 0,
        "num_env_runners": 1,
        "rollout_fragment_length": 256,
    }
)

MODULE1_CONVERGENCE = dict(CONVERGENCE_RULE_DEFAULTS)
