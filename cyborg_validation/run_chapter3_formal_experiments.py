from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from run_thesis_scenario_validation import (
    ScenarioSpec,
    default_mapping_path,
    default_scenario_path,
    run_validation,
)

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sg_deception_simulation"))

from chapter3_sim.config import GameConfig


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "chapter3_formal_experiments.json"


def extract_comparison(results: Dict[str, object], scenario_name: str) -> Dict[str, object]:
    scenario = results["scenarios"][scenario_name]
    return {
        "scenario": scenario_name,
        "prior_theta1": scenario["prior_theta1"],
        "comparison": scenario["comparison"],
        "truthful_aggregate": scenario["policies"]["truthful_baseline"]["aggregate"],
        "alternative_aggregate": scenario["policies"][scenario["comparison"]["compared_policy"]]["aggregate"],
    }


def run_primary_comparisons(
    episodes: int,
    max_steps: int,
    scenario_path: Path,
    mapping_path: Path,
    base_config: GameConfig,
    mapping_style: str,
) -> Dict[str, object]:
    results = run_validation(
        episodes=episodes,
        max_steps=max_steps,
        scenario_path=scenario_path,
        mapping_path=mapping_path,
        base_config=base_config,
        mapping_style=mapping_style,
    )
    return {
        "raw": results,
        "scenario_a_high_prior_theta1": extract_comparison(results, "scenario_a_high_prior_theta1"),
        "scenario_b_low_prior_theta1": extract_comparison(results, "scenario_b_low_prior_theta1"),
    }


def run_single_pair(
    name: str,
    prior_theta1: float,
    compared_policy: str,
    episodes: int,
    max_steps: int,
    scenario_path: Path,
    mapping_path: Path,
    base_config: GameConfig,
    mapping_style: str,
) -> Dict[str, object]:
    results = run_validation(
        episodes=episodes,
        max_steps=max_steps,
        scenario_path=scenario_path,
        mapping_path=mapping_path,
        base_config=base_config,
        mapping_style=mapping_style,
        scenario_specs=[ScenarioSpec(name, prior_theta1, ("truthful_baseline", compared_policy))],
    )
    return extract_comparison(results, name)


def run_sensitivity_sweeps(
    episodes: int,
    max_steps: int,
    scenario_path: Path,
    mapping_path: Path,
    base_config: GameConfig,
    mapping_style: str,
) -> Dict[str, List[Dict[str, object]]]:
    sweeps = {
        "prior_theta1": [0.3, 0.4, 0.5, 0.6, 0.7],
        "beta": [0.2, 0.5, 0.8, 1.1, 1.4],
        "c_theta1": [0.5, 1.0, 1.5, 2.0, 2.5],
        "c_theta2": [0.5, 1.0, 1.5, 2.0, 2.5],
    }
    output: Dict[str, List[Dict[str, object]]] = {}

    output["prior_theta1"] = []
    for prior in sweeps["prior_theta1"]:
        item = run_single_pair(
            name=f"prior_theta1_{prior:.2f}",
            prior_theta1=prior,
            compared_policy="pbne_production_camouflage",
            episodes=episodes,
            max_steps=max_steps,
            scenario_path=scenario_path,
            mapping_path=mapping_path,
            base_config=GameConfig(**{**base_config.__dict__}),
            mapping_style=mapping_style,
        )
        item["sweep_value"] = prior
        output["prior_theta1"].append(item)

    output["beta"] = []
    for beta in sweeps["beta"]:
        item = run_single_pair(
            name=f"beta_{beta:.2f}",
            prior_theta1=0.35,
            compared_policy="pbne_honeypot_camouflage",
            episodes=episodes,
            max_steps=max_steps,
            scenario_path=scenario_path,
            mapping_path=mapping_path,
            base_config=GameConfig(**{**base_config.__dict__, "beta": beta}),
            mapping_style=mapping_style,
        )
        item["sweep_value"] = beta
        output["beta"].append(item)

    output["c_theta1"] = []
    for cost in sweeps["c_theta1"]:
        item = run_single_pair(
            name=f"c_theta1_{cost:.2f}",
            prior_theta1=0.65,
            compared_policy="pbne_production_camouflage",
            episodes=episodes,
            max_steps=max_steps,
            scenario_path=scenario_path,
            mapping_path=mapping_path,
            base_config=GameConfig(**{**base_config.__dict__, "c_theta1": cost}),
            mapping_style=mapping_style,
        )
        item["sweep_value"] = cost
        output["c_theta1"].append(item)

    output["c_theta2"] = []
    for cost in sweeps["c_theta2"]:
        item = run_single_pair(
            name=f"c_theta2_{cost:.2f}",
            prior_theta1=0.35,
            compared_policy="pbne_honeypot_camouflage",
            episodes=episodes,
            max_steps=max_steps,
            scenario_path=scenario_path,
            mapping_path=mapping_path,
            base_config=GameConfig(**{**base_config.__dict__, "c_theta2": cost}),
            mapping_style=mapping_style,
        )
        item["sweep_value"] = cost
        output["c_theta2"].append(item)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Chapter 3 experiments on the thesis-specific CybORG scenario.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--scenario-path", type=Path, default=default_scenario_path())
    parser.add_argument("--mapping-path", type=Path, default=default_mapping_path())
    parser.add_argument(
        "--mapping-style",
        choices=("simple", "decoy_honeypot"),
        default="simple",
        help="Environment action mapping used to realize the signaling-game policy in CybORG.",
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()

    base_config = GameConfig(
        horizon=args.max_steps,
        prior_theta1=0.5,
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

    payload = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "scenario_path": str(args.scenario_path),
        "mapping_path": str(args.mapping_path),
        "mapping_style": args.mapping_style,
        "base_config": base_config.__dict__,
        "primary_comparisons": run_primary_comparisons(
            episodes=args.episodes,
            max_steps=args.max_steps,
            scenario_path=args.scenario_path,
            mapping_path=args.mapping_path,
            base_config=base_config,
            mapping_style=args.mapping_style,
        ),
        "sensitivity_sweeps": run_sensitivity_sweeps(
            episodes=args.episodes,
            max_steps=args.max_steps,
            scenario_path=args.scenario_path,
            mapping_path=args.mapping_path,
            base_config=base_config,
            mapping_style=args.mapping_style,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Primary scenario A blue reward lift:", round(payload["primary_comparisons"]["scenario_a_high_prior_theta1"]["comparison"]["blue_reward_lift"], 4))
    print("Primary scenario B blue reward lift:", round(payload["primary_comparisons"]["scenario_b_low_prior_theta1"]["comparison"]["blue_reward_lift"], 4))
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
