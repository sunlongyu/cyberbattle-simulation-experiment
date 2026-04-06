from __future__ import annotations

import functools
from typing import Dict as TypeDict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDictSpace, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from .defaults import CHAPTER4_ENV_DEFAULTS

DEFENDER = "defender"
ATTACKER = "attacker"

THETA_REAL = 0
THETA_HONEYPOT = 1
SIGNAL_NORMAL = 0
SIGNAL_HONEYPOT = 1
ACTION_RETREAT = 0
ACTION_ATTACK = 1


def _payoff_value(payoffs: dict, key: str, default: float = 0.0) -> float:
    return float(payoffs.get(key, default))


def _env_defaults(overrides: Optional[dict] = None) -> dict:
    merged = dict(CHAPTER4_ENV_DEFAULTS)
    merged["payoffs"] = dict(CHAPTER4_ENV_DEFAULTS["payoffs"])
    if overrides:
        for key, value in overrides.items():
            if key == "payoffs":
                merged["payoffs"].update(value)
            else:
                merged[key] = value
    return merged


def env(render_mode=None, **kwargs):
    wrapped = raw_env(render_mode=render_mode, **kwargs)
    return wrappers.OrderEnforcingWrapper(wrapped)


def parallel_env(**kwargs):
    return parallel_wrapper_fn(lambda: raw_env(**kwargs))()


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "chapter4_signaling_env",
        "is_parallelizable": True,
        "has_manual_policy": False,
    }

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        cfg = _env_defaults(kwargs)
        payoffs = cfg["payoffs"]

        self.N = int(cfg["n_systems"])
        self.T = int(cfg["max_steps"])
        self.history_length = int(cfg.get("history_length", 10))
        self.render_mode = render_mode

        self.g_a = _payoff_value(payoffs, "g_a")
        self.c_a = _payoff_value(payoffs, "c_a")
        self.l_a = _payoff_value(payoffs, "l_a")
        self.l_i = _payoff_value(payoffs, "l_i")
        self.g_i = _payoff_value(payoffs, "g_i")
        self.c_theta1 = _payoff_value(payoffs, "c_theta1")
        self.c_theta2 = _payoff_value(payoffs, "c_theta2")
        self.eta_c = _payoff_value(payoffs, "eta_c")
        self.g_c = _payoff_value(payoffs, "g_c")
        self.kappa_d = _payoff_value(payoffs, "kappa_d")
        self.kappa_a = _payoff_value(payoffs, "kappa_a")

        prior = float(cfg.get("prior_belief_real", 0.5))
        self.P_theta = [prior, 1.0 - prior]
        signal_matrix = cfg.get("signal_likelihood", [[0.8, 0.2], [0.2, 0.8]])
        self.lambda_D = signal_matrix
        self.beta = float(cfg.get("belief_beta", 0.5))
        self.disable_belief_input = bool(cfg.get("disable_belief_input", False))
        self.disable_deception_signal = bool(cfg.get("disable_deception_signal", False))
        self.signal_mode = str(cfg.get("signal_mode", "learned"))

        self.possible_agents = [DEFENDER, ATTACKER]
        self.agent_ids = self.possible_agents
        self._agent_ids = set(self.possible_agents)

        self._defender_obs_space = GymDictSpace(
            {
                "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "system_types": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
                "last_attacker_actions": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
                "attacker_action_history": Box(
                    low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8
                ),
                "defender_signal_history": Box(
                    low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8
                ),
            }
        )
        self._attacker_obs_space = GymDictSpace(
            {
                "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "current_defender_signals": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
                "attacker_action_history": Box(
                    low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8
                ),
                "defender_signal_history": Box(
                    low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8
                ),
                "belief_real": Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            }
        )
        self._action_space = MultiDiscrete([2] * self.N)

        self.system_types = np.zeros(self.N, dtype=np.int8)
        self.current_step = 0
        self._current_defender_signals = np.zeros(self.N, dtype=np.int8)
        self._current_attacker_actions = np.zeros(self.N, dtype=np.int8)
        self._last_attacker_actions = np.zeros(self.N, dtype=np.int8)
        self._defender_signal_history = np.zeros((self.history_length, self.N), dtype=np.int8)
        self._attacker_action_history = np.zeros((self.history_length, self.N), dtype=np.int8)
        self._full_signal_history: List[List[int]] = [[] for _ in range(self.N)]

        self.agents: List[str] = []
        self.rewards: TypeDict[str, float] = {}
        self._cumulative_rewards: TypeDict[str, float] = {}
        self.terminations: TypeDict[str, bool] = {}
        self.truncations: TypeDict[str, bool] = {}
        self.infos: TypeDict[str, TypeDict] = {}
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = ""

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.spaces.Space:
        if agent == DEFENDER:
            return self._defender_obs_space
        if agent == ATTACKER:
            return self._attacker_obs_space
        raise ValueError(f"Unknown agent: {agent}")

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.system_types = np.random.randint(0, 2, size=self.N, dtype=np.int8)
        self.current_step = 0
        self._current_defender_signals.fill(0)
        self._current_attacker_actions.fill(0)
        self._last_attacker_actions.fill(0)
        self._defender_signal_history.fill(0)
        self._attacker_action_history.fill(0)
        self._full_signal_history = [[] for _ in range(self.N)]

        self.agents = list(self.possible_agents)
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"step": self.current_step} for agent in self.agents}
        self.infos[DEFENDER]["true_types"] = self.system_types.copy()
        self.has_reset = True

    def observe(self, agent: str):
        if agent not in self.agents:
            return self.observation_space(agent).sample() * 0

        normalized_step = np.array([self.current_step / max(self.T, 1)], dtype=np.float32)
        if agent == DEFENDER:
            return {
                "current_step": normalized_step,
                "system_types": self.system_types.copy(),
                "last_attacker_actions": self._last_attacker_actions.copy(),
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
            }
        if agent == ATTACKER:
            if self.disable_belief_input:
                belief_real = np.zeros(self.N, dtype=np.float32)
            else:
                belief_real = np.array([self._compute_belief(i) for i in range(self.N)], dtype=np.float32)
            return {
                "current_step": normalized_step,
                "current_defender_signals": self._current_defender_signals.copy(),
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
                "belief_real": belief_real,
            }
        raise ValueError(f"Unknown agent: {agent}")

    def state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.system_types.astype(np.float32),
                np.array([self.current_step / max(self.T, 1)], dtype=np.float32),
                self._current_defender_signals.astype(np.float32),
                self._last_attacker_actions.astype(np.float32),
                self._defender_signal_history.flatten().astype(np.float32),
                self._attacker_action_history.flatten().astype(np.float32),
            ]
        )

    def _compute_belief(self, system_idx: int) -> float:
        signals = self._full_signal_history[system_idx]
        if not signals:
            return float(self.P_theta[0])
        t = len(signals)
        log_scores = np.zeros(2)
        for theta in [0, 1]:
            log_scores[theta] = np.log(self.P_theta[theta])
            for t_prime, signal in enumerate(signals, start=1):
                weight = np.exp(-self.beta * (t - t_prime))
                log_scores[theta] += np.log(self.lambda_D[theta][signal]) * weight
        max_log = np.max(log_scores)
        probs = np.exp(log_scores - max_log)
        probs = probs / np.sum(probs)
        return float(probs[0])

    def step(self, action: np.ndarray) -> None:
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(
            self.agent_selection, False
        ):
            self._was_dead_step(action)
            return

        acting_agent = self.agent_selection
        action = np.asarray(action, dtype=self._action_space.dtype)
        if acting_agent == DEFENDER:
            if self.disable_deception_signal:
                # Keep backward compatibility for earlier no-signal baselines.
                self._current_defender_signals = np.zeros_like(action, dtype=self._action_space.dtype)
            elif self.signal_mode == "truthful":
                # No deception: the signal directly reveals the underlying system type.
                self._current_defender_signals = self.system_types.astype(self._action_space.dtype, copy=True)
            elif self.signal_mode == "constant_normal":
                self._current_defender_signals = np.zeros_like(action, dtype=self._action_space.dtype)
            else:
                self._current_defender_signals = action
            self._defender_signal_history = np.roll(self._defender_signal_history, shift=1, axis=0)
            self._defender_signal_history[0, :] = self._current_defender_signals
            for idx in range(self.N):
                self._full_signal_history[idx].append(int(self._current_defender_signals[idx]))
        elif acting_agent == ATTACKER:
            self._current_attacker_actions = action
            self._attacker_action_history = np.roll(self._attacker_action_history, shift=1, axis=0)
            self._attacker_action_history[0, :] = action

        if self._agent_selector.is_last():
            rewards = self._calculate_rewards(self._current_defender_signals, self._current_attacker_actions)
            self.rewards[DEFENDER] = rewards[DEFENDER]
            self.rewards[ATTACKER] = rewards[ATTACKER]
            self._cumulative_rewards[DEFENDER] += rewards[DEFENDER]
            self._cumulative_rewards[ATTACKER] += rewards[ATTACKER]
            self.current_step += 1
            self._last_attacker_actions = self._current_attacker_actions.copy()

            done = self.current_step >= self.T
            self.truncations = {agent: done for agent in self.agents}
            self.terminations = {agent: done for agent in self.agents}
            self.infos = {agent: {"step": self.current_step} for agent in self.agents}
            self.infos[DEFENDER]["true_types"] = self.system_types.copy()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        if all(
            self.terminations.get(agent, False) or self.truncations.get(agent, False)
            for agent in self.agents
        ):
            self.agents = []
            self.infos = {}

    def _calculate_rewards(
        self, defender_signals_t: np.ndarray, attacker_actions_t: np.ndarray
    ) -> TypeDict[str, float]:
        total_reward_d = 0.0
        total_reward_a = 0.0
        for idx in range(self.N):
            theta = int(self.system_types[idx])
            signal = int(defender_signals_t[idx])
            action = int(attacker_actions_t[idx])
            reward_d = 0.0
            reward_a = 0.0
            if theta == THETA_REAL:
                if signal == SIGNAL_NORMAL:
                    if action == ACTION_ATTACK:
                        reward_d = -(self.l_a + self.eta_c)
                        reward_a = self.g_a - self.c_a
                    else:
                        reward_d = self.g_c
                else:
                    if action == ACTION_ATTACK:
                        reward_d = -(self.l_a + self.eta_c + self.c_theta1)
                        reward_a = self.g_a - self.c_a
                    else:
                        reward_d = self.g_c - self.c_theta1
            else:
                if signal == SIGNAL_NORMAL:
                    if action == ACTION_ATTACK:
                        reward_d = self.g_i + self.kappa_d - self.c_theta2
                        reward_a = -(self.c_a + self.l_i + self.kappa_a)
                    else:
                        reward_d = -self.c_theta2
                else:
                    if action == ACTION_ATTACK:
                        reward_d = self.g_i + self.kappa_d
                        reward_a = -(self.c_a + self.l_i + self.kappa_a)
                    else:
                        reward_d = 0.0
            total_reward_d += reward_d
            total_reward_a += reward_a
        return {DEFENDER: total_reward_d, ATTACKER: total_reward_a}

    def last(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        agent = self.agent_selection
        return (
            self.observe(agent),
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def close(self):
        return None


def create_rllib_env(env_config: dict):
    return ParallelPettingZooEnv(parallel_env(**env_config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id == DEFENDER:
        return "defender_policy"
    if agent_id == ATTACKER:
        return "attacker_policy"
    raise ValueError(f"Unknown agent: {agent_id}")


def register_chapter4_env(env_name: str = "chapter4_signaling_env") -> str:
    register_env(env_name, lambda cfg: create_rllib_env(cfg))
    return env_name
