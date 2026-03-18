from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from CybORG.Simulator.Actions import DecoyApache, DecoySSHD, DecoyTomcat, Misinform, Sleep
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.env import CybORG

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sg_deception_simulation"))

from chapter3_sim.config import GameConfig
from chapter3_sim.model import DefenderType, Signal, camouflage_cost, discounted_belief
from chapter3_sim.strategies import (
    StrategyState,
    pbne_honeypot_camouflage,
    pbne_production_camouflage,
    truthful_baseline,
)
from thesis_red_agent import ThesisRedAgent


ATTACK_LIKE_ACTIONS = {"ExploitRemoteService", "PrivilegeEscalate", "Impact"}
CRITICAL_HOST_PREFIXES = ("Prod_Operational",)
UTILITY_STAGE_DISCOUNT = 0.95


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


def default_scenario_path() -> Path:
    return Path(__file__).resolve().parent / "scenarios" / "thesis_scenario.yaml"


def default_mapping_path() -> Path:
    return Path(__file__).resolve().parent / "thesis_scenario_mapping.json"


def load_mapping(mapping_path: Path) -> HostTypeMapping:
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
        defender_pressure_cost=2.0,
        defender_protection_gain=6.0,
        intel_gain=8.0,
        intel_loss=8.0,
        defender_deception_bonus=2.0,
        attacker_deception_penalty=2.0,
        c_theta1=2.5,
        c_theta2=1.5,
        monte_carlo_runs=0,
    )


def with_overrides(config: GameConfig, **kwargs) -> GameConfig:
    payload = asdict(config)
    payload.update(kwargs)
    return GameConfig(**payload)


def choose_host(defender_type: DefenderType, mapping: HostTypeMapping, step: int) -> str:
    hosts = mapping.theta1_hosts if defender_type == DefenderType.THETA1 else mapping.theta2_hosts
    return hosts[(step - 1) % len(hosts)]


def choose_regime(policy_name: str, belief_theta1: float, config: GameConfig):
    state = StrategyState(belief_theta1=belief_theta1, config=config)
    if policy_name == "truthful_baseline":
        return truthful_baseline(config)
    if policy_name == "pbne_production_camouflage":
        return pbne_production_camouflage(state)
    if policy_name == "pbne_honeypot_camouflage":
        return pbne_honeypot_camouflage(state)
    raise ValueError(f"Unknown policy {policy_name}")


def sample_signal(signal_probs: Dict[Signal, float], rng: random.Random) -> Signal:
    threshold = rng.random()
    cumulative = 0.0
    for signal, probability in signal_probs.items():
        cumulative += probability
        if threshold <= cumulative:
            return signal
    return Signal.SIGMA2


def signal_to_action(signal: Signal, defender_type: DefenderType, hostname: str, step: int):
    if signal == Signal.SIGMA2:
        return Misinform(session=0, agent="Blue", hostname=hostname)
    return Sleep()


def signal_to_action_with_decoys(signal: Signal, defender_type: DefenderType, hostname: str, step: int):
    if defender_type == DefenderType.THETA1:
        if signal == Signal.SIGMA2:
            return Misinform(session=0, agent="Blue", hostname=hostname)
        return Sleep()

    if signal == Signal.SIGMA1:
        decoy_cycle = (DecoyApache, DecoyTomcat, DecoySSHD)
        decoy_action = decoy_cycle[(step - 1) % len(decoy_cycle)]
        return decoy_action(session=0, agent="Blue", hostname=hostname)
    return Sleep()


def action_from_mapping_style(
    signal: Signal,
    defender_type: DefenderType,
    hostname: str,
    step: int,
    mapping_style: str,
):
    if mapping_style == "simple":
        return signal_to_action(signal, defender_type, hostname, step)
    if mapping_style == "decoy_honeypot":
        return signal_to_action_with_decoys(signal, defender_type, hostname, step)
    raise ValueError(f"Unknown mapping style: {mapping_style}")


def summarize_red_sessions(cyborg: CybORG, mapping: HostTypeMapping) -> Dict[str, object]:
    sessions = cyborg.environment_controller.state.sessions["Red"]
    active_hosts = sorted({session.hostname for session in sessions.values() if session.active})
    production_compromised = sorted(host for host in active_hosts if host in mapping.theta1_hosts)
    honeypot_compromised = sorted(host for host in active_hosts if host in mapping.theta2_hosts)
    return {
        "active_red_hosts": active_hosts,
        "production_hosts_compromised": production_compromised,
        "honeypot_hosts_compromised": honeypot_compromised,
        "critical_host_compromised": any(host in CRITICAL_HOST_PREFIXES for host in active_hosts),
        "red_session_count": len([session for session in sessions.values() if session.active]),
    }


def thesis_proxy_utilities(
    game_config: GameConfig,
    production_compromised_count: int,
    honeypot_compromised_count: int,
    disguise_cost_total: float,
) -> Dict[str, float]:
    defender_proxy = (
        honeypot_compromised_count * game_config.intel_gain
        - production_compromised_count * game_config.defender_loss
        - disguise_cost_total
    )
    attacker_proxy = (
        production_compromised_count * (game_config.attack_gain - game_config.attack_cost)
        - honeypot_compromised_count * (game_config.attack_cost + game_config.intel_loss)
    )
    return {
        "thesis_proxy_defender_utility": defender_proxy,
        "thesis_proxy_attacker_utility": attacker_proxy,
    }


def stage_discount(step: int) -> float:
    return math.pow(UTILITY_STAGE_DISCOUNT, step - 1)


def run_episode(
    scenario_path: Path,
    mapping: HostTypeMapping,
    scenario_spec: ScenarioSpec,
    policy_name: str,
    seed: int,
    max_steps: int,
    mapping_style: str = "simple",
    base_config: GameConfig | None = None,
) -> Dict[str, object]:
    game_config = build_game_config(prior_theta1=scenario_spec.prior_theta1, horizon=max_steps) if base_config is None else with_overrides(
        base_config,
        prior_theta1=scenario_spec.prior_theta1,
        horizon=max_steps,
        monte_carlo_runs=0,
    )
    sg = FileReaderScenarioGenerator(str(scenario_path))
    cyborg = CybORG(
        sg,
        "sim",
        agents={"Red": ThesisRedAgent()},
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
    blue_action_counts: Counter[str] = Counter()
    red_action_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    disguise_cost_total = 0.0
    refined_defender_utility_total = 0.0
    refined_attacker_utility_total = 0.0
    production_compromised_seen: set[str] = set()
    honeypot_compromised_seen: set[str] = set()

    for step in range(1, max_steps + 1):
        defender_type = DefenderType.THETA1 if local_rng.random() < game_config.prior_theta1 else DefenderType.THETA2
        type_counts[defender_type.value] += 1
        hostname = choose_host(defender_type, mapping, step)
        regime = choose_regime(policy_name=policy_name, belief_theta1=belief_theta1, config=game_config)
        signal = sample_signal(regime.defender_signal_probs[defender_type], local_rng)
        stage_cost = camouflage_cost(defender_type, signal, game_config)
        disguise_cost_total += stage_cost
        action = action_from_mapping_style(signal, defender_type, hostname, step, mapping_style)

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

        attack_like_this_step = 0
        last_red = cyborg.get_last_action("Red")
        if last_red is not None:
            red_action_name = type(last_red).__name__
            red_action_counts[red_action_name] += 1
            if red_action_name in ATTACK_LIKE_ACTIONS:
                attack_like_count += 1
                attack_like_this_step = 1

        step_summary = summarize_red_sessions(cyborg, mapping)
        new_production = set(step_summary["production_hosts_compromised"]) - production_compromised_seen
        new_honeypot = set(step_summary["honeypot_hosts_compromised"]) - honeypot_compromised_seen
        weight = stage_discount(step)
        refined_defender_utility_total += weight * (
            len(new_honeypot) * game_config.intel_gain
            - len(new_production) * game_config.defender_loss
            - attack_like_this_step * 0.5 * game_config.attack_cost
        ) - stage_cost
        refined_attacker_utility_total += weight * (
            len(new_production) * game_config.attack_gain
            - len(new_honeypot) * game_config.intel_loss
            - attack_like_this_step * game_config.attack_cost
        )
        production_compromised_seen.update(new_production)
        honeypot_compromised_seen.update(new_honeypot)

        if result.done:
            break

    session_summary = summarize_red_sessions(cyborg, mapping)
    production_compromised_count = len(session_summary["production_hosts_compromised"])
    honeypot_compromised_count = len(session_summary["honeypot_hosts_compromised"])
    proxy_metrics = thesis_proxy_utilities(
        game_config=game_config,
        production_compromised_count=production_compromised_count,
        honeypot_compromised_count=honeypot_compromised_count,
        disguise_cost_total=disguise_cost_total,
    )
    return {
        "seed": seed,
        "scenario": scenario_spec.name,
        "policy": policy_name,
        "blue_reward_total": blue_reward_total,
        "red_reward_total": red_reward_total,
        "blue_reward_net_disguise_cost": blue_reward_total - disguise_cost_total,
        "blue_success_count": blue_success_count,
        "attack_like_count": attack_like_count,
        "belief_path": belief_path,
        "final_belief_theta1": belief_theta1,
        "lambda_d_mean_episode": mean(lambda_d_metrics) if lambda_d_metrics else 0.0,
        "lambda_a_mean_episode": mean(lambda_a_metrics) if lambda_a_metrics else 0.0,
        "disguise_cost_total": disguise_cost_total,
        "refined_defender_utility_total": refined_defender_utility_total,
        "refined_attacker_utility_total": refined_attacker_utility_total,
        "signal_counts": dict(signal_counts),
        "type_counts": dict(type_counts),
        "blue_action_counts": dict(blue_action_counts),
        "red_action_counts": dict(red_action_counts),
        **session_summary,
        **proxy_metrics,
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
        "blue_reward_net_disguise_cost_mean": mean(row["blue_reward_net_disguise_cost"] for row in rows),
        "blue_success_mean": mean(row["blue_success_count"] for row in rows),
        "attack_like_count_mean": mean(row["attack_like_count"] for row in rows),
        "critical_host_compromise_rate": mean(1.0 if row["critical_host_compromised"] else 0.0 for row in rows),
        "production_compromise_rate": mean(1.0 if row["production_hosts_compromised"] else 0.0 for row in rows),
        "honeypot_compromise_rate": mean(1.0 if row["honeypot_hosts_compromised"] else 0.0 for row in rows),
        "red_session_count_mean": mean(row["red_session_count"] for row in rows),
        "final_belief_theta1_mean": mean(row["final_belief_theta1"] for row in rows),
        "lambda_d_mean": mean(row["lambda_d_mean_episode"] for row in rows),
        "lambda_a_mean": mean(row["lambda_a_mean_episode"] for row in rows),
        "disguise_cost_mean": mean(row["disguise_cost_total"] for row in rows),
        "thesis_proxy_defender_utility_mean": mean(row["thesis_proxy_defender_utility"] for row in rows),
        "thesis_proxy_attacker_utility_mean": mean(row["thesis_proxy_attacker_utility"] for row in rows),
        "refined_defender_utility_mean": mean(row["refined_defender_utility_total"] for row in rows),
        "refined_attacker_utility_mean": mean(row["refined_attacker_utility_total"] for row in rows),
        "avg_belief_path": [mean(path[index] for path in padded) for index in range(max_len)],
    }


def run_validation(
    episodes: int,
    max_steps: int,
    scenario_path: Path,
    mapping_path: Path,
    base_config: GameConfig | None = None,
    scenario_specs: List[ScenarioSpec] | None = None,
    mapping_style: str = "simple",
) -> Dict[str, object]:
    mapping = load_mapping(mapping_path)
    scenarios = scenario_specs or [
        ScenarioSpec("scenario_a_high_prior_theta1", 0.65, ("truthful_baseline", "pbne_production_camouflage")),
        ScenarioSpec("scenario_b_low_prior_theta1", 0.35, ("truthful_baseline", "pbne_honeypot_camouflage")),
    ]
    effective_config = build_game_config(prior_theta1=0.5, horizon=max_steps) if base_config is None else with_overrides(
        base_config,
        horizon=max_steps,
        monte_carlo_runs=0,
    )
    output = {
        "scenario_path": str(scenario_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "mapping_style": mapping_style,
        "base_config": asdict(effective_config),
        "host_type_mapping": {
            "theta1_production_hosts": mapping.theta1_hosts,
            "theta2_honeypot_surrogate_hosts": mapping.theta2_hosts,
            "notes": mapping.notes,
        },
        "scenarios": {},
    }
    for scenario_spec in scenarios:
        scenario_output = {"prior_theta1": scenario_spec.prior_theta1, "policies": {}}
        for policy_name in scenario_spec.policies:
            rows = [
                run_episode(
                    scenario_path,
                    mapping,
                    scenario_spec,
                    policy_name,
                    20260315 + idx,
                    max_steps,
                    mapping_style=mapping_style,
                    base_config=effective_config,
                )
                for idx in range(episodes)
            ]
            scenario_output["policies"][policy_name] = {
                "aggregate": aggregate_rows(rows),
                "episodes": rows,
            }
        truthful = scenario_output["policies"]["truthful_baseline"]["aggregate"]
        alt_name = scenario_spec.policies[1]
        alt = scenario_output["policies"][alt_name]["aggregate"]
        scenario_output["comparison"] = {
            "compared_policy": alt_name,
            "blue_reward_lift": alt["blue_reward_mean"] - truthful["blue_reward_mean"],
            "blue_reward_net_disguise_cost_lift": alt["blue_reward_net_disguise_cost_mean"] - truthful["blue_reward_net_disguise_cost_mean"],
            "red_reward_reduction": truthful["red_reward_mean"] - alt["red_reward_mean"],
            "attack_like_count_reduction": truthful["attack_like_count_mean"] - alt["attack_like_count_mean"],
            "final_belief_shift": alt["final_belief_theta1_mean"] - truthful["final_belief_theta1_mean"],
            "thesis_proxy_defender_utility_lift": alt["thesis_proxy_defender_utility_mean"] - truthful["thesis_proxy_defender_utility_mean"],
            "thesis_proxy_attacker_utility_reduction": truthful["thesis_proxy_attacker_utility_mean"] - alt["thesis_proxy_attacker_utility_mean"],
            "refined_defender_utility_lift": alt["refined_defender_utility_mean"] - truthful["refined_defender_utility_mean"],
            "refined_attacker_utility_reduction": truthful["refined_attacker_utility_mean"] - alt["refined_attacker_utility_mean"],
        }
        output["scenarios"][scenario_spec.name] = scenario_output
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation on the thesis-specific CybORG scenario.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--scenario-path", type=Path, default=default_scenario_path())
    parser.add_argument("--mapping-path", type=Path, default=default_mapping_path())
    parser.add_argument(
        "--mapping-style",
        choices=("simple", "decoy_honeypot"),
        default="simple",
        help="Environment action mapping used to realize the signaling-game policy in CybORG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "thesis_scenario_results.json",
    )
    args = parser.parse_args()
    results = run_validation(
        args.episodes,
        args.max_steps,
        args.scenario_path,
        args.mapping_path,
        mapping_style=args.mapping_style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    a = results["scenarios"]["scenario_a_high_prior_theta1"]["comparison"]["blue_reward_lift"]
    b = results["scenarios"]["scenario_b_low_prior_theta1"]["comparison"]["blue_reward_lift"]
    print("Scenario A blue reward lift:", round(a, 4))
    print("Scenario B blue reward lift:", round(b, 4))
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
