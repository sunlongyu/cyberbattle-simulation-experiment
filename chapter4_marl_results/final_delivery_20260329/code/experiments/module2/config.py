from marl_core.defaults import CHAPTER4_ENV_DEFAULTS, CHAPTER4_TRAINING_DEFAULTS


MODULE2_BASE_ENV = dict(CHAPTER4_ENV_DEFAULTS)
MODULE2_BASE_ENV["history_length"] = 10

MODULE2_BASE_TRAINING = dict(CHAPTER4_TRAINING_DEFAULTS)
MODULE2_BASE_TRAINING.update(
    {
        "episodes": 200,
        "batch_size": 256,
        "learning_rate": 7e-5,
        "num_gpus": 0,
        "num_env_runners": 1,
        "rollout_fragment_length": 128,
        "lstm_cell_size": 128,
        "actor_critic_hidden_dims": [128, 128],
        "num_sgd_iter": 4,
    }
)

MODULE2_PAYOFFS = {
    "g_a": 2.0,
    "c_a": 1.5,
    "l_a": 3.0,
    "l_i": 4.0,
    "g_i": 4.0,
    "g_c": 2.0,
    "eta_c": 0.5,
    "kappa_d": 1.5,
    "kappa_a": 1.5,
    "c_theta1": 1.0,
    "c_theta2": 1.0,
}

MODULE2_SWEEPS = {
    "T": [10, 25, 50, 100],
    "N": [5, 7, 10],
    "psi0": [0.3, 0.5, 0.7],
    "c_theta1": [0.6, 1.0, 1.4],
    "c_theta2": [0.6, 1.0, 1.4],
}
