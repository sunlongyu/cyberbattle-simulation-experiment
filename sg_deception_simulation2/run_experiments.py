from __future__ import annotations

import csv
import json
from pathlib import Path

from cyber_simulation_core.config import DEFAULT_CONFIG
from cyber_simulation_core.experiments import (
    build_experiment_two_summary,
    build_gamma_horizon_rows,
    build_gamma_sensitivity_rows,
    build_sensitivity_rows,
    build_sensitivity_trajectory_rows,
    build_stage_rows,
    build_summary_rows,
    build_terminal_belief_rows,
    build_terminal_state_rows,
    build_terminal_uncertainty_rows,
    run_experiment_one,
    run_gamma_sensitivity_experiment,
    run_experiment_three,
    run_experiment_two,
)

BASE_DIR = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_one = run_experiment_one(DEFAULT_CONFIG)
    summary_rows = build_summary_rows(experiment_one)
    stage_rows = build_stage_rows(experiment_one)
    experiment_two = run_experiment_two(DEFAULT_CONFIG)
    experiment_two_summary = build_experiment_two_summary(experiment_two)
    experiment_three = run_experiment_three(DEFAULT_CONFIG)
    sensitivity_rows = build_sensitivity_rows(experiment_three)
    sensitivity_trajectory_rows = build_sensitivity_trajectory_rows(experiment_three)
    gamma_sensitivity = run_gamma_sensitivity_experiment(DEFAULT_CONFIG)
    gamma_sensitivity_rows = build_gamma_sensitivity_rows(gamma_sensitivity)
    gamma_horizon_rows = build_gamma_horizon_rows(gamma_sensitivity)
    terminal_belief_rows = build_terminal_belief_rows(experiment_two)
    terminal_uncertainty_rows = build_terminal_uncertainty_rows(experiment_two)
    terminal_state_rows = build_terminal_state_rows(experiment_two)

    json_path = output_dir / "experiment1_payoff_comparison.json"
    summary_csv_path = output_dir / "experiment1_payoff_summary.csv"
    stage_csv_path = output_dir / "experiment1_stage_paths.csv"
    experiment_two_json_path = output_dir / "experiment2_terminal_belief_comparison.json"
    experiment_three_json_path = output_dir / "experiment3_sensitivity_analysis.json"
    gamma_sensitivity_json_path = output_dir / "experiment3_discount_factor_sensitivity.json"
    terminal_belief_csv_path = output_dir / "experiment2_terminal_beliefs.csv"
    terminal_uncertainty_csv_path = output_dir / "experiment2_terminal_uncertainty_summary.csv"
    terminal_state_csv_path = output_dir / "experiment2_terminal_state_composition.csv"
    sensitivity_csv_path = output_dir / "experiment3_sensitivity_analysis.csv"
    sensitivity_trajectory_csv_path = output_dir / "experiment3_sensitivity_trajectories.csv"
    gamma_sensitivity_csv_path = output_dir / "experiment3_discount_factor_mechanism.csv"
    gamma_horizon_csv_path = output_dir / "experiment3_discount_factor_horizon_paths.csv"

    json_path.write_text(json.dumps(experiment_one, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(stage_csv_path, stage_rows)
    experiment_two_json_path.write_text(json.dumps(experiment_two_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment_three_json_path.write_text(json.dumps(experiment_three, indent=2, ensure_ascii=False), encoding="utf-8")
    gamma_sensitivity_json_path.write_text(json.dumps(gamma_sensitivity, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(terminal_belief_csv_path, terminal_belief_rows)
    _write_csv(terminal_uncertainty_csv_path, terminal_uncertainty_rows)
    _write_csv(terminal_state_csv_path, terminal_state_rows)
    _write_csv(sensitivity_csv_path, sensitivity_rows)
    _write_csv(sensitivity_trajectory_csv_path, sensitivity_trajectory_rows)
    _write_csv(gamma_sensitivity_csv_path, gamma_sensitivity_rows)
    _write_csv(gamma_horizon_csv_path, gamma_horizon_rows)

    print("Experiment 1 JSON written to", json_path)
    print("Experiment 1 summary CSV written to", summary_csv_path)
    print("Experiment 1 stage CSV written to", stage_csv_path)
    print("Experiment 2 JSON written to", experiment_two_json_path)
    print("Experiment 3 JSON written to", experiment_three_json_path)
    print("Experiment 3 discount-factor JSON written to", gamma_sensitivity_json_path)
    print("Experiment 2 terminal belief CSV written to", terminal_belief_csv_path)
    print("Experiment 2 terminal uncertainty CSV written to", terminal_uncertainty_csv_path)
    print("Experiment 2 terminal state CSV written to", terminal_state_csv_path)
    print("Experiment 3 sensitivity CSV written to", sensitivity_csv_path)
    print("Experiment 3 sensitivity trajectory CSV written to", sensitivity_trajectory_csv_path)
    print("Experiment 3 discount-factor mechanism CSV written to", gamma_sensitivity_csv_path)
    print("Experiment 3 discount-factor horizon CSV written to", gamma_horizon_csv_path)


if __name__ == "__main__":
    main()
