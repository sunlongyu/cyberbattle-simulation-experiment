from __future__ import annotations

import json
from pathlib import Path

from cyber_simulation_core.config import DEFAULT_CONFIG
from cyber_simulation_core.experiments import (
    run_feasible_comparison_scenarios,
    run_sensitivity_analysis,
    run_strategy_comparison,
)


def main() -> None:
    output_dir = Path("sg_deception_simulation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = run_strategy_comparison(DEFAULT_CONFIG)
    feasible_scenarios = run_feasible_comparison_scenarios(DEFAULT_CONFIG)
    sensitivity = run_sensitivity_analysis(DEFAULT_CONFIG)

    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (output_dir / "feasible_scenarios.json").write_text(
        json.dumps(feasible_scenarios, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sensitivity_analysis.json").write_text(
        json.dumps(sensitivity, indent=2),
        encoding="utf-8",
    )

    print("Strategy comparison written to", output_dir / "strategy_comparison.json")
    print("Feasible comparison scenarios written to", output_dir / "feasible_scenarios.json")
    print("Sensitivity analysis written to", output_dir / "sensitivity_analysis.json")


if __name__ == "__main__":
    main()
