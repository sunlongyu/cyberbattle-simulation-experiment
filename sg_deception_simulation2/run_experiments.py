from __future__ import annotations

import csv
import json
from pathlib import Path

from cyber_simulation_core.config import DEFAULT_CONFIG
from cyber_simulation_core.experiments import (
    build_belief_trajectory_rows,
    build_experiment_two_summary,
    build_sensitivity_rows,
    build_sensitivity_trajectory_rows,
    build_state_probability_rows,
    build_stage_rows,
    build_summary_rows,
    build_terminal_belief_rows,
    build_terminal_state_rows,
    build_terminal_uncertainty_rows,
    run_experiment_one,
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
    belief_trajectory_rows = build_belief_trajectory_rows(experiment_two)
    terminal_belief_rows = build_terminal_belief_rows(experiment_two)
    state_probability_rows = build_state_probability_rows(experiment_two)
    terminal_uncertainty_rows = build_terminal_uncertainty_rows(experiment_two)
    terminal_state_rows = build_terminal_state_rows(experiment_two)

    json_path = output_dir / "experiment1_payoff_comparison.json"
    summary_csv_path = output_dir / "experiment1_payoff_summary.csv"
    stage_csv_path = output_dir / "experiment1_stage_paths.csv"
    experiment_two_json_path = output_dir / "experiment2_belief_dynamics.json"
    experiment_three_json_path = output_dir / "experiment3_sensitivity_analysis.json"
    belief_trajectory_csv_path = output_dir / "experiment2_belief_trajectories.csv"
    terminal_belief_csv_path = output_dir / "experiment2_terminal_beliefs.csv"
    state_probability_csv_path = output_dir / "experiment2_state_probability_evolution.csv"
    terminal_uncertainty_csv_path = output_dir / "experiment2_terminal_uncertainty_summary.csv"
    terminal_state_csv_path = output_dir / "experiment2_terminal_state_composition.csv"
    sensitivity_csv_path = output_dir / "experiment3_sensitivity_analysis.csv"
    sensitivity_trajectory_csv_path = output_dir / "experiment3_sensitivity_trajectories.csv"

    json_path.write_text(json.dumps(experiment_one, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(stage_csv_path, stage_rows)
    experiment_two_json_path.write_text(json.dumps(experiment_two_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment_three_json_path.write_text(json.dumps(experiment_three, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(belief_trajectory_csv_path, belief_trajectory_rows)
    _write_csv(terminal_belief_csv_path, terminal_belief_rows)
    _write_csv(state_probability_csv_path, state_probability_rows)
    _write_csv(terminal_uncertainty_csv_path, terminal_uncertainty_rows)
    _write_csv(terminal_state_csv_path, terminal_state_rows)
    _write_csv(sensitivity_csv_path, sensitivity_rows)
    _write_csv(sensitivity_trajectory_csv_path, sensitivity_trajectory_rows)

    print("Experiment 1 JSON written to", json_path)
    print("Experiment 1 summary CSV written to", summary_csv_path)
    print("Experiment 1 stage CSV written to", stage_csv_path)
    print("Experiment 2 JSON written to", experiment_two_json_path)
    print("Experiment 3 JSON written to", experiment_three_json_path)
    print("Experiment 2 trajectory CSV written to", belief_trajectory_csv_path)
    print("Experiment 2 terminal belief CSV written to", terminal_belief_csv_path)
    print("Experiment 2 state probability CSV written to", state_probability_csv_path)
    print("Experiment 2 terminal uncertainty CSV written to", terminal_uncertainty_csv_path)
    print("Experiment 2 terminal state CSV written to", terminal_state_csv_path)
    print("Experiment 3 sensitivity CSV written to", sensitivity_csv_path)
    print("Experiment 3 sensitivity trajectory CSV written to", sensitivity_trajectory_csv_path)


if __name__ == "__main__":
    main()
