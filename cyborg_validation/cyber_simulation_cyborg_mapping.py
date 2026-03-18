from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
from CybORG.Agents import RedMeanderAgent
from CybORG.Simulator.Actions import Misinform, Sleep
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.env import CybORG

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sg_deception_simulation"))

from cyber_simulation_core.config import GameConfig
from cyber_simulation_core.model import AttackerAction, DefenderType, Signal, discounted_belief
from cyber_simulation_core.strategies import StrategyState, pbne_production_camouflage, truthful_baseline


DEFAULT_PROTECTED_HOSTS = [
    "User1",
    "User2",
    "User3",
    "Enterprise0",
    "Enterprise1",
    "Op_Server0",
]
ATTACK_LIKE_ACTIONS = {"ExploitRemoteService", "PrivilegeEscalate", "Impact"}
CRITICAL_HOST_PREFIXES = ("Op_",)


@dataclass(frozen=True)
class CybORGMappingConfig:
    scenario_path: Path
    simulation_config: GameConfig
    protected_hosts: List[str]
    max_steps: int = 25


def default_scenario_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "CybORG" / "CybORG" / "Simulator" / "Scenarios" / "scenario_files" / "Scenario2.yaml"


def build_mapping_config(max_steps: int) -> CybORGMappingConfig:
    simulation_config = GameConfig(
        horizon=max_steps,
        prior_theta1=0.65,
        beta=1.0,
        epsilon=0.02,
        attack_cost=2.0,
        attack_gain=12.0,
        defender_loss=10.0,
        intel_gain=8.0,
        intel_loss=8.0,
        c_theta1=2.5,
        c_theta2=1.5,
        monte_carlo_runs=0,
    )
    return CybORGMappingConfig(
        scenario_path=default_scenario_path(),
        simulation_config=simulation_config,
        protected_hosts=DEFAULT_PROTECTED_HOSTS,
        max_steps=max_steps,
    )


def summarize_red_sessions(cyborg: CybORG) -> Dict[str, object]:
    sessions = cyborg.environment_controller.state.sessions["Red"]
    active_hosts = sorted({session.hostname for session in sessions.values() if session.active})
    return {
        "active_red_hosts": active_hosts,
        "critical_host_compromised": any(host.startswith(CRITICAL_HOST_PREFIXES) for host in active_hosts),
        "red_session_count": len([session for session in sessions.values() if session.active]),
    }


def choose_truthful_action(step: int, config: CybORGMappingConfig):
    del step, config
    return Sleep(), Signal.SIGMA1, {}


def choose_pbne1_action(
    step: int,
    belief_theta1: float,
    signal_history: List[Signal],
    config: CybORGMappingConfig,
    rng: random.Random,
):
    del signal_history
    regime = pbne_production_camouflage(
        StrategyState(
            belief_theta1=belief_theta1,
            config=config.simulation_config,
        )
    )
    host = config.protected_hosts[(step - 1) % len(config.protected_hosts)]
    sigma2_probability = regime.defender_signal_probs[DefenderType.THETA1][Signal.SIGMA2]
    if rng.random() < sigma2_probability:
        return (
            Misinform(session=0, agent="Blue", hostname=host),
            Signal.SIGMA2,
            {
                "target_host": host,
                "lambda_d_star": sigma2_probability,
                "lambda_a_star": regime.mixing_metrics["lambda_a_star"],
            },
        )
    return (
        Sleep(),
        Signal.SIGMA1,
        {
            "target_host": host,
            "lambda_d_star": sigma2_probability,
            "lambda_a_star": regime.mixing_metrics["lambda_a_star"],
        },
    )


def run_mapped_episode(
    mapping: CybORGMappingConfig,
    policy_name: str,
    seed: int,
) -> Dict[str, object]:
    sg = FileReaderScenarioGenerator(str(mapping.scenario_path))
    cyborg = CybORG(
        sg,
        "sim",
        agents={"Red": RedMeanderAgent()},
        seed=np.random.RandomState(seed),
    )
    cyborg.reset(agent="Blue")

    belief_theta1 = mapping.simulation_config.prior_theta1
    signal_history: List[Signal] = []
    belief_path: List[float] = []
    lambda_d_star_path: List[float] = []
    lambda_a_star_path: List[float] = []
    red_action_counts: Counter[str] = Counter()
    blue_action_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    blue_reward_total = 0.0
    red_reward_total = 0.0
    blue_success_count = 0
    attack_like_count = 0
    attack_like_steps: List[int] = []

    local_rng = random.Random(seed)

    for step in range(1, mapping.max_steps + 1):
        if policy_name == "truthful_baseline":
            action, signal, metrics = choose_truthful_action(step, mapping)
            regime = truthful_baseline(mapping.simulation_config)
        elif policy_name == "pbne_production_camouflage":
            action, signal, metrics = choose_pbne1_action(
                step=step,
                belief_theta1=belief_theta1,
                signal_history=signal_history,
                config=mapping,
                rng=local_rng,
            )
            regime = pbne_production_camouflage(
                StrategyState(belief_theta1=belief_theta1, config=mapping.simulation_config)
            )
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        result = cyborg.step("Blue", action)
        blue_action_counts[type(action).__name__] += 1
        signal_counts[signal.value] += 1

        blue_reward_total += cyborg.get_rewards()["Blue"]["HybridAvailabilityConfidentiality"]
        red_reward_total += cyborg.get_rewards()["Red"]["HybridImpactPwn"]
        if str(result.observation.get("success")).upper() == "TRUE":
            blue_success_count += 1

        signal_history.append(signal)
        belief_theta1 = discounted_belief(
            prior_theta1=mapping.simulation_config.prior_theta1,
            signal_history=signal_history,
            type_signal_probs=regime.defender_signal_probs,
            beta=mapping.simulation_config.beta,
        )
        belief_path.append(belief_theta1)

        if "lambda_d_star" in metrics:
            lambda_d_star_path.append(metrics["lambda_d_star"])
        if "lambda_a_star" in metrics:
            lambda_a_star_path.append(metrics["lambda_a_star"])

        last_red = cyborg.get_last_action("Red")
        if last_red is not None:
            red_action_name = type(last_red).__name__
            red_action_counts[red_action_name] += 1
            if red_action_name in ATTACK_LIKE_ACTIONS:
                attack_like_count += 1
                attack_like_steps.append(step)

        if result.done:
            break

    session_summary = summarize_red_sessions(cyborg)
    return {
        "seed": seed,
        "policy": policy_name,
        "blue_reward_total": blue_reward_total,
        "red_reward_total": red_reward_total,
        "blue_success_count": blue_success_count,
        "attack_like_count": attack_like_count,
        "attack_like_steps": attack_like_steps,
        "belief_path": belief_path,
        "final_belief_theta1": belief_theta1,
        "lambda_d_star_path": lambda_d_star_path,
        "lambda_a_star_path": lambda_a_star_path,
        "signal_counts": dict(signal_counts),
        "blue_action_counts": dict(blue_action_counts),
        "red_action_counts": dict(red_action_counts),
        **session_summary,
    }


def aggregate_episode_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    max_belief_len = max(len(row["belief_path"]) for row in rows)
    padded_beliefs = []
    for row in rows:
        tail = row["belief_path"][-1] if row["belief_path"] else 0.0
        padded_beliefs.append(row["belief_path"] + [tail] * (max_belief_len - len(row["belief_path"])))

    avg_belief_path = [
        mean(path[index] for path in padded_beliefs)
        for index in range(max_belief_len)
    ]

    return {
        "episode_count": len(rows),
        "blue_reward_mean": mean(row["blue_reward_total"] for row in rows),
        "red_reward_mean": mean(row["red_reward_total"] for row in rows),
        "blue_success_mean": mean(row["blue_success_count"] for row in rows),
        "attack_like_count_mean": mean(row["attack_like_count"] for row in rows),
        "critical_host_compromise_rate": mean(
            1.0 if row["critical_host_compromised"] else 0.0 for row in rows
        ),
        "red_session_count_mean": mean(row["red_session_count"] for row in rows),
        "final_belief_theta1_mean": mean(row["final_belief_theta1"] for row in rows),
        "avg_belief_path": avg_belief_path,
        "lambda_d_star_mean": mean(
            mean(row["lambda_d_star_path"]) for row in rows if row["lambda_d_star_path"]
        ) if any(row["lambda_d_star_path"] for row in rows) else 0.0,
        "lambda_a_star_mean": mean(
            mean(row["lambda_a_star_path"]) for row in rows if row["lambda_a_star_path"]
        ) if any(row["lambda_a_star_path"] for row in rows) else 0.0,
    }


def run_validation(episodes: int, max_steps: int) -> Dict[str, object]:
    mapping = build_mapping_config(max_steps=max_steps)
    policies = ["truthful_baseline", "pbne_production_camouflage"]
    output: Dict[str, object] = {
        "scenario_path": str(mapping.scenario_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "mapping": {
            "protected_hosts": mapping.protected_hosts,
            "prior_theta1": mapping.simulation_config.prior_theta1,
            "beta": mapping.simulation_config.beta,
            "c_theta1": mapping.simulation_config.c_theta1,
            "attack_cost": mapping.simulation_config.attack_cost,
            "attack_gain": mapping.simulation_config.attack_gain,
            "defender_loss": mapping.simulation_config.defender_loss,
            "intel_gain": mapping.simulation_config.intel_gain,
            "intel_loss": mapping.simulation_config.intel_loss,
        },
        "policies": {},
    }

    for policy_name in policies:
        rows = [
            run_mapped_episode(mapping=mapping, policy_name=policy_name, seed=20260315 + episode_seed)
            for episode_seed in range(episodes)
        ]
        output["policies"][policy_name] = {
            "aggregate": aggregate_episode_rows(rows),
            "episodes": rows,
        }

    truthful = output["policies"]["truthful_baseline"]["aggregate"]
    pbne1 = output["policies"]["pbne_production_camouflage"]["aggregate"]
    output["comparison"] = {
        "blue_reward_lift": pbne1["blue_reward_mean"] - truthful["blue_reward_mean"],
        "red_reward_reduction": truthful["red_reward_mean"] - pbne1["red_reward_mean"],
        "attack_like_count_reduction": truthful["attack_like_count_mean"] - pbne1["attack_like_count_mean"],
        "final_belief_shift": pbne1["final_belief_theta1_mean"] - truthful["final_belief_theta1_mean"],
        "interpretation": (
            "This comparison maps Chapter 3 truthful disclosure and PBNE-1-style production camouflage into CybORG "
            "Scenario2. Sigma1 is approximated by no deception action, while sigma2 is approximated by Misinform on "
            "a protected production host."
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Map the Chapter 3 truthful and PBNE-1 policies into CybORG.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cyber_simulation_mapped_results.json",
    )
    args = parser.parse_args()

    results = run_validation(episodes=args.episodes, max_steps=args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    truthful = results["policies"]["truthful_baseline"]["aggregate"]
    pbne1 = results["policies"]["pbne_production_camouflage"]["aggregate"]
    print("Truthful blue reward mean:", round(truthful["blue_reward_mean"], 4))
    print("PBNE-1 blue reward mean:", round(pbne1["blue_reward_mean"], 4))
    print("Blue reward lift:", round(results["comparison"]["blue_reward_lift"], 4))
    print("PBNE-1 lambda_d_star mean:", round(pbne1["lambda_d_star_mean"], 4))
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
