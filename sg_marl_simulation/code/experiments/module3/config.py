from __future__ import annotations

from marl_core import CHAPTER4_ENV_DEFAULTS, CONVERGENCE_RULE_DEFAULTS

MODULE3_PAYOFFS = {
    "g_a": 5.0,
    "c_a": 0.5,
    "l_a": 3.0,
    "l_i": 5.0,
    "g_i": 4.0,
    "g_c": 2.0,
    "eta_c": 0.5,
    "kappa_d": 1.5,
    "kappa_a": 2.5,
    "c_theta1": 1.0,
    "c_theta2": 1.0,
}

MODULE3_ENV = {
    **CHAPTER4_ENV_DEFAULTS,
    "payoffs": MODULE3_PAYOFFS,
}

MODULE3_TRAINING = {
    "episodes": 30,
    "batch_size": 1024,
    "learning_rate": 7e-5,
    "gae_lambda": 0.97,
    "clip_param": 0.2,
    "num_sgd_iter": 10,
    "lstm_cell_size": 256,
    "actor_critic_hidden_dims": [256, 256],
    "num_env_runners": 1,
    "rollout_fragment_length": 50,
    "num_gpus": 0,
    "entropy_coeff": 0.005,
    "vf_loss_coeff": 1.0,
}

MODULE3_EVAL_EPISODES = 20

MODULE3_BASELINE_ENV = {
    **MODULE3_ENV,
    "max_steps": 100,
}

MODULE3_BASELINE_TRAINING = {
    **MODULE3_TRAINING,
    "episodes": 400,
    "rollout_fragment_length": 100,
}

MODULE3_BASELINE_EVAL_EPISODES = 40

MODULE3_ABLATION_ENV = {
    **MODULE3_ENV,
    "max_steps": 100,
}

MODULE3_ABLATION_TRAINING = {
    **MODULE3_TRAINING,
    "episodes": 100,
    "rollout_fragment_length": 100,
}

MODULE3_ABLATION_EVAL_EPISODES = 40

MODULE3_ABLATION_FINAL_WINDOW = 20

MODULE3_CONVERGENCE = {
    **CONVERGENCE_RULE_DEFAULTS,
}

MODULE3_BASELINES = {
    "SG-MAPPO": {
        "training": {
            "learning_rate": 1e-4,
            "num_sgd_iter": 8,
        },
        "env": {},
        "description": "完整模型，含 Bayes belief、LSTM 与 PPO clipping。",
    },
    "SG-MATRPO": {
        "training": {
            "learning_rate": 7e-5,
            "clip_param": 0.08,
            "num_sgd_iter": 12,
        },
        "env": {},
        "description": "更保守的 trust-region 风格更新近似基线。",
    },
    "SG-MAA2C": {
        "training": {
            "learning_rate": 1e-4,
            "clip_param": 1.0,
            "num_sgd_iter": 1,
            "entropy_coeff": 0.002,
        },
        "env": {},
        "description": "普通 actor-critic 风格更新近似基线。",
    },
    "Plain-MAPPO": {
        "training": {
            "learning_rate": 1e-4,
            "num_sgd_iter": 8,
        },
        "env": {"signal_mode": "truthful"},
        "description": "无欺骗伪装的普通 MAPPO 基线，防御方只能发送真实类型信号。",
    },
}

MODULE3_ABLATIONS = {
    "完整SG-MAPPO": {
        "training": {},
        "env": {},
        "description": "完整模型。",
    },
    "去除信念输入": {
        "training": {},
        "env": {"disable_belief_input": True},
        "description": "去除 Bayesian belief 输入。",
    },
    "去除LSTM": {
        "training": {"use_lstm": False},
        "env": {},
        "description": "去除时序建模。",
    },
}

MODULE3_DEFAULT_SEED = 1
