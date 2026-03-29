"""Canonical Chapter 4 defaults derived from the thesis text."""

CHAPTER4_ENV_DEFAULTS = {
    "n_systems": 5,
    "max_steps": 50,
    "discount_gamma": 0.99,
    "belief_beta": 0.50,
    "prior_belief_real": 0.50,
    "payoffs": {
        "g_a": 2.0,
        "c_a": 1.5,
        "l_a": 3.0,
        "l_i": 4.0,
        "g_i": 4.0,
        "c_theta1": 1.5,
        "c_theta2": 2.0,
    },
}


CHAPTER4_TRAINING_DEFAULTS = {
    "episodes": 400,
    "batch_size": 1024,
    "learning_rate": 5e-4,
    "gae_lambda": 0.97,
    "clip_param": 0.2,
    "num_sgd_iter": 10,
    "lstm_cell_size": 256,
    "actor_critic_hidden_dims": [256, 256],
}


CONVERGENCE_RULE_DEFAULTS = {
    "threshold_ratio": 0.95,
    "stability_window": 20,
    "final_window": 30,
}


CHAPTER4_MODULE_IDS = {
    "module1": "module1_convergence",
    "module2": "module2_policy",
    "module3": "module3_effectiveness",
}
