from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from CybORG.Agents import RedMeanderAgent
from CybORG.Simulator.Actions import Misinform, Sleep
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.env import CybORG

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sg_deception_simulation"))

from cyber_simulation_core.config import GameConfig
from cyber_simulation_core.model import DefenderType, Signal, discounted_belief
from cyber_simulation_core.strategies import (
    StrategyState,
    pbne_honeypot_camouflage,
    pbne_production_camouflage,
    truthful_baseline,
)


ATTACK_LIKE_ACTIONS = {"ExploitRemoteService", "PrivilegeEscalate", "Impact"}
CRITICAL_HOST_PREFIXES = ("Op_",)


@dataclass(frozen=True)
class HostTypeMapping:
    theta1_hosts: List[str]
    theta2_hosts: List[str]
    notes: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    prior_theta1: float
    policies: Tuple[str, str]


def default_cyborg_scenario_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "CybORG" / "CybORG" / "Simulator" / "Scenarios" / "scenario_files" / "Scenario2.yaml"


def load_mapping() -> HostTypeMapping:
    mapping_path = Path(__file__).resolve().parent / "cyber_simulation_host_type_mapping.json"
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    return HostTypeMapping(
        theta1_hosts=data["theta1_production_hosts"],
        theta2_hosts=data["theta2_honeypot_surrogate_hosts"],
        notes=data["notes"],
    )


def build_game_config(prior_theta1: float, horizon: int) -> GameConfig:
    return GameConfig(
        horizon=horizon,
        prior_theta1=prior_theta1,
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


def choose_host(defender_type: DefenderType, mapping: HostTypeMapping, step: int) -> str:
    if defender_type == DefenderType.THETA1:
        return mapping.theta1_hosts[(step - 1) % len(mapping.theta1_hosts)]
    return mapping.theta2_hosts[(step - 1) % len(mapping.theta2_hosts)]


def choose_regime(policy_name: str, belief_theta1: float, config: GameConfig):
    state = StrategyState(belief_theta1=belief_theta1, config=config)
    if policy_name == "truthful_baseline":
        return truthful_baseline(config)
    if policy_name == "pbne_production_camouflage":
        return pbne_production_camouflage(state)
    if policy_name == "pbne_honeypot_camouflage":
        return pbne_honeypot_camouflage(state)
    raise ValueError(f"Unknown policy {policy_name}")


def sample_signal(
    signal_probs: Dict[Signal, float],
    rng: random.Random,
) -> Signal:
    threshold = rng.random()
    cumulative = 0.0
    for signal, probability in signal_probs.items():
        cumulative += probability
        if threshold <= cumulative:
            return signal
    return Signal.SIGMA2


def signal_to_action(signal: Signal, hostname: str):
    if signal == Signal.SIGMA2:
        return Misinform(session=0, agent="Blue", hostname=hostname)
    return Sleep()


def summarize_red_sessions(cyborg: CybORG, mapping: HostTypeMapping) -> Dict[str, object]:
    sessions = cyborg.environment_controller.state.sessions["Red"]
    active_hosts = sorted({session.hostname for session in sessions.values() if session.active})
    production_compromised = sorted(host for host in active_hosts if host in mapping.theta1_hosts)
    honeypot_compromised = sorted(host for host in active_hosts if host in mapping.theta2_hosts)
    return {
        "active_red_hosts": active_hosts,
        "production_hosts_compromised": production_compromised,
        "honeypot_hosts_compromised": honeypot_compromised,
        "critical_host_compromised": any(host.startswith(CRITICAL_HOST_PREFIXES) for host in active_hosts),
        "red_session_count": len([session for session in sessions.values() if session.active]),
    }


def run_episode(
    scenario_path: Path,
    mapping: HostTypeMapping,
    scenario_spec: ScenarioSpec,
    policy_name: str,
    seed: int,
    max_steps: int,
) -> Dict[str, object]:
    game_config = build_game_config(prior_theta1=scenario_spec.prior_theta1, horizon=max_steps)
    sg = FileReaderScenarioGenerator(str(scenario_path))
    cyborg = CybORG(
        sg,
        "sim",
        agents={"Red": RedMeanderAgent()},
        seed=np.random.RandomState(seed),
    )
    cyborg.reset(agent="Blue")

    local_rng = random.Random(seed)
    belief_theta1 = game_config.prior_theta1
    signal_history: List[Signal] = []
    belief_path: List[float] = []
    lambda_d_metrics: List[float] = []
    lambda_a_metrics: List[float] = []
    blue_reward_total = 0.0
    red_reward_total = 0.0
    blue_success_count = 0
    attack_like_count = 0
    attack_like_steps: List[int] = []
    blue_action_counts: Counter[str] = Counter()
    red_action_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    targeted_hosts: List[str] = []

    for step in range(1, max_steps + 1):
        defender_type = DefenderType.THETA1 if local_rng.random() < game_config.prior_theta1 else DefenderType.THETA2
        type_counts[defender_type.value] += 1
        hostname = choose_host(defender_type, mapping, step)
        targeted_hosts.append(hostname)

        regime = choose_regime(policy_name=policy_name, belief_theta1=belief_theta1, config=game_config)
        signal = sample_signal(regime.defender_signal_probs[defender_type], local_rng)
        action = signal_to_action(signal, hostname)

        result = cyborg.step("Blue", action)
        blue_action_counts[type(action).__name__] += 1
        signal_counts[signal.value] += 1
        blue_reward_total += cyborg.get_rewards()["Blue"]["HybridAvailabilityConfidentiality"]
        red_reward_total += cyborg.get_rewards()["Red"]["HybridImpactPwn"]
        if str(result.observation.get("success")).upper() == "TRUE":
            blue_success_count += 1

        signal_history.append(signal)
        belief_theta1 = discounted_belief(
            prior_theta1=game_config.prior_theta1,
            signal_history=signal_history,
            type_signal_probs=regime.defender_signal_probs,
            beta=game_config.beta,
        )
        belief_path.append(belief_theta1)

        for key in ("lambda_d_star", "lambda_d_prime"):
            if key in regime.mixing_metrics:
                lambda_d_metrics.append(regime.mixing_metrics[key])
        for key in ("lambda_a_star", "lambda_a_prime"):
            if key in regime.mixing_metrics:
                lambda_a_metrics.append(regime.mixing_metrics[key])

        last_red = cyborg.get_last_action("Red")
        if last_red is not None:
            red_action_name = type(last_red).__name__
            red_action_counts[red_action_name] += 1
            if red_action_name in ATTACK_LIKE_ACTIONS:
                attack_like_count += 1
                attack_like_steps.append(step)

        if result.done:
            break

    session_summary = summarize_red_sessions(cyborg, mapping)
    return {
        "seed": seed,
        "scenario": scenario_spec.name,
        "policy": policy_name,
        "blue_reward_total": blue_reward_total,
        "red_reward_total": red_reward_total,
        "blue_success_count": blue_success_count,
        "attack_like_count": attack_like_count,
        "attack_like_steps": attack_like_steps,
        "belief_path": belief_path,
        "final_belief_theta1": belief_theta1,
        "lambda_d_mean_episode": mean(lambda_d_metrics) if lambda_d_metrics else 0.0,
        "lambda_a_mean_episode": mean(lambda_a_metrics) if lambda_a_metrics else 0.0,
        "signal_counts": dict(signal_counts),
        "type_counts": dict(type_counts),
        "blue_action_counts": dict(blue_action_counts),
        "red_action_counts": dict(red_action_counts),
        "targeted_hosts": targeted_hosts,
        **session_summary,
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(len(row["belief_path"]) for row in rows)
    padded = []
    for row in rows:
        tail = row["belief_path"][-1] if row["belief_path"] else 0.0
        padded.append(row["belief_path"] + [tail] * (max_len - len(row["belief_path"])))

    return {
        "episode_count": len(rows),
        "blue_reward_mean": mean(row["blue_reward_total"] for row in rows),
        "red_reward_mean": mean(row["red_reward_total"] for row in rows),
        "blue_success_mean": mean(row["blue_success_count"] for row in rows),
        "attack_like_count_mean": mean(row["attack_like_count"] for row in rows),
        "critical_host_compromise_rate": mean(
            1.0 if row["critical_host_compromised"] else 0.0 for row in rows
        ),
        "production_compromise_rate": mean(
            1.0 if len(row["production_hosts_compromised"]) > 0 else 0.0 for row in rows
        ),
        "honeypot_compromise_rate": mean(
            1.0 if len(row["honeypot_hosts_compromised"]) > 0 else 0.0 for row in rows
        ),
        "red_session_count_mean": mean(row["red_session_count"] for row in rows),
        "final_belief_theta1_mean": mean(row["final_belief_theta1"] for row in rows),
        "lambda_d_mean": mean(row["lambda_d_mean_episode"] for row in rows),
        "lambda_a_mean": mean(row["lambda_a_mean_episode"] for row in rows),
        "avg_belief_path": [
            mean(path[index] for path in padded)
            for index in range(max_len)
        ],
    }


def run_validation(episodes: int, max_steps: int) -> Dict[str, object]:
    mapping = load_mapping()
    scenario_path = default_cyborg_scenario_path()
    scenarios = [
        ScenarioSpec(
            name="scenario_a_high_prior_theta1",
            prior_theta1=0.65,
            policies=("truthful_baseline", "pbne_production_camouflage"),
        ),
        ScenarioSpec(
            name="scenario_b_low_prior_theta1",
            prior_theta1=0.35,
            policies=("truthful_baseline", "pbne_honeypot_camouflage"),
        ),
    ]

    output: Dict[str, object] = {
        "scenario_path": str(scenario_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "host_type_mapping": {
            "theta1_production_hosts": mapping.theta1_hosts,
            "theta2_honeypot_surrogate_hosts": mapping.theta2_hosts,
            "notes": mapping.notes,
        },
        "scenarios": {},
    }

    for scenario_spec in scenarios:
        scenario_output = {
            "prior_theta1": scenario_spec.prior_theta1,
            "policies": {},
        }
        for policy_name in scenario_spec.policies:
            rows = [
                run_episode(
                    scenario_path=scenario_path,
                    mapping=mapping,
                    scenario_spec=scenario_spec,
                    policy_name=policy_name,
                    seed=20260315 + idx,
                    max_steps=max_steps,
                )
                for idx in range(episodes)
            ]
            scenario_output["policies"][policy_name] = {
                "aggregate": aggregate_rows(rows),
                "episodes": rows,
            }

        truthful = scenario_output["policies"]["truthful_baseline"]["aggregate"]
        compared_policy = scenario_spec.policies[1]
        mapped = scenario_output["policies"][compared_policy]["aggregate"]
        scenario_output["comparison"] = {
            "compared_policy": compared_policy,
            "blue_reward_lift": mapped["blue_reward_mean"] - truthful["blue_reward_mean"],
            "red_reward_reduction": truthful["red_reward_mean"] - mapped["red_reward_mean"],
            "attack_like_count_reduction": truthful["attack_like_count_mean"] - mapped["attack_like_count_mean"],
            "final_belief_shift": mapped["final_belief_theta1_mean"] - truthful["final_belief_theta1_mean"],
        }
        output["scenarios"][scenario_spec.name] = scenario_output

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a host-typed Chapter 3 to CybORG validation.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cyber_simulation_host_typed_results.json",
    )
    args = parser.parse_args()

    results = run_validation(episodes=args.episodes, max_steps=args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    scenario_a = results["scenarios"]["scenario_a_high_prior_theta1"]["comparison"]
    scenario_b = results["scenarios"]["scenario_b_low_prior_theta1"]["comparison"]
    print("Scenario A blue reward lift:", round(scenario_a["blue_reward_lift"], 4))
    print("Scenario B blue reward lift:", round(scenario_b["blue_reward_lift"], 4))
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
