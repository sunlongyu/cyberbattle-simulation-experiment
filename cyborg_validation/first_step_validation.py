from __future__ import annotations

import argparse
import json
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


DEFAULT_HOST_CYCLE = [
    "User1",
    "User2",
    "User3",
    "Enterprise0",
    "Enterprise1",
    "Op_Server0",
]
CRITICAL_HOST_PREFIXES = ("Op_",)


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    camouflage_steps: int
    host_cycle: List[str]


def default_scenario_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "CybORG" / "CybORG" / "Simulator" / "Scenarios" / "scenario_files" / "Scenario2.yaml"


def choose_blue_action(policy: PolicyConfig, step: int):
    if step > policy.camouflage_steps:
        return Sleep()
    hostname = policy.host_cycle[(step - 1) % len(policy.host_cycle)]
    return Misinform(session=0, agent="Blue", hostname=hostname)


def summarize_red_sessions(cyborg: CybORG) -> Dict[str, object]:
    sessions = cyborg.environment_controller.state.sessions["Red"]
    active_hosts = sorted({session.hostname for session in sessions.values() if session.active})
    return {
        "active_red_hosts": active_hosts,
        "critical_host_compromised": any(host.startswith(CRITICAL_HOST_PREFIXES) for host in active_hosts),
        "red_session_count": len([session for session in sessions.values() if session.active]),
    }


def run_episode(scenario_path: Path, policy: PolicyConfig, seed: int, max_steps: int) -> Dict[str, object]:
    sg = FileReaderScenarioGenerator(str(scenario_path))
    cyborg = CybORG(
        sg,
        "sim",
        agents={"Red": RedMeanderAgent()},
        seed=np.random.RandomState(seed),
    )
    cyborg.reset(agent="Blue")

    blue_reward_total = 0.0
    red_reward_total = 0.0
    blue_success_count = 0
    red_action_counts: Counter[str] = Counter()
    blue_action_counts: Counter[str] = Counter()

    for step in range(1, max_steps + 1):
        action = Sleep() if policy.name == "baseline" else choose_blue_action(policy, step)
        result = cyborg.step("Blue", action)
        blue_action_counts[type(action).__name__] += 1

        blue_reward_total += cyborg.get_rewards()["Blue"]["HybridAvailabilityConfidentiality"]
        red_reward_total += cyborg.get_rewards()["Red"]["HybridImpactPwn"]

        if str(result.observation.get("success")).upper() == "TRUE":
            blue_success_count += 1

        last_red = cyborg.get_last_action("Red")
        if last_red is not None:
            red_action_counts[type(last_red).__name__] += 1

        if result.done:
            break

    session_summary = summarize_red_sessions(cyborg)
    return {
        "seed": seed,
        "policy": policy.name,
        "blue_reward_total": blue_reward_total,
        "red_reward_total": red_reward_total,
        "blue_success_count": blue_success_count,
        "blue_action_counts": dict(blue_action_counts),
        "red_action_counts": dict(red_action_counts),
        **session_summary,
    }


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "episode_count": len(results),
        "blue_reward_mean": mean(row["blue_reward_total"] for row in results),
        "red_reward_mean": mean(row["red_reward_total"] for row in results),
        "blue_success_mean": mean(row["blue_success_count"] for row in results),
        "critical_host_compromise_rate": mean(
            1.0 if row["critical_host_compromised"] else 0.0 for row in results
        ),
        "red_session_count_mean": mean(row["red_session_count"] for row in results),
    }


def run_validation(scenario_path: Path, episodes: int, max_steps: int) -> Dict[str, object]:
    policies = [
        PolicyConfig(name="baseline", camouflage_steps=0, host_cycle=[]),
        PolicyConfig(name="camouflage", camouflage_steps=8, host_cycle=DEFAULT_HOST_CYCLE),
    ]

    output: Dict[str, object] = {
        "scenario_path": str(scenario_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "policies": {},
    }

    for policy in policies:
        episode_rows = [
            run_episode(scenario_path=scenario_path, policy=policy, seed=20260315 + seed, max_steps=max_steps)
            for seed in range(episodes)
        ]
        output["policies"][policy.name] = {
            "config": {
                "camouflage_steps": policy.camouflage_steps,
                "host_cycle": policy.host_cycle,
            },
            "aggregate": aggregate_results(episode_rows),
            "episodes": episode_rows,
        }

    baseline_mean = output["policies"]["baseline"]["aggregate"]["blue_reward_mean"]
    camouflage_mean = output["policies"]["camouflage"]["aggregate"]["blue_reward_mean"]
    output["comparison"] = {
        "blue_reward_lift": camouflage_mean - baseline_mean,
        "interpretation": (
            "Positive blue_reward_lift means the current camouflage policy outperformed the no-camouflage baseline "
            "under the same CybORG Scenario2 and red-agent setting."
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a first-step CybORG validation for Chapter 3 camouflage ideas.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--scenario-path", type=Path, default=default_scenario_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "first_step_results.json",
    )
    args = parser.parse_args()

    results = run_validation(
        scenario_path=args.scenario_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    baseline = results["policies"]["baseline"]["aggregate"]
    camouflage = results["policies"]["camouflage"]["aggregate"]
    print("Baseline blue reward mean:", round(baseline["blue_reward_mean"], 4))
    print("Camouflage blue reward mean:", round(camouflage["blue_reward_mean"], 4))
    print("Blue reward lift:", round(results["comparison"]["blue_reward_lift"], 4))
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
