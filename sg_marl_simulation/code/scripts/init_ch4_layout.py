#!/usr/bin/env python3
"""Initialize the standardized Chapter 4 result layout."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from marl_core import ExperimentConfig, ExperimentArtifacts


def main() -> None:
    configs = [
        ExperimentConfig(
            module_id="module1_convergence",
            run_name="module0_bootstrap",
            description="Bootstrap layout for Chapter 4 module 1 outputs.",
            tags=["chapter4", "module0", "convergence"],
        ),
        ExperimentConfig(
            module_id="module2_policy",
            run_name="module0_bootstrap",
            description="Bootstrap layout for Chapter 4 module 2 outputs.",
            tags=["chapter4", "module0", "policy"],
        ),
        ExperimentConfig(
            module_id="module3_effectiveness",
            run_name="module0_bootstrap",
            description="Bootstrap layout for Chapter 4 module 3 outputs.",
            tags=["chapter4", "module0", "effectiveness"],
        ),
    ]
    for config in configs:
        artifacts = ExperimentArtifacts(config)
        artifacts.initialize()
        print(f"initialized {artifacts.root}")


if __name__ == "__main__":
    main()
