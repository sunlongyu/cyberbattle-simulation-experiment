from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from marl_core import CHAPTER4_MODULE_IDS, ExperimentArtifacts, ExperimentConfig
from marl_core.io import write_csv
from experiments.module3.config import (
    MODULE3_BASELINE_ENV,
    MODULE3_BASELINE_EVAL_EPISODES,
    MODULE3_BASELINE_TRAINING,
)
from experiments.module3.pipeline import run_variant


def main() -> None:
    run_name = "module3_plain_mappo_truthful_v2"
    family = "baseline_truthful"
    variant_name = "Plain-MAPPO"

    training = dict(MODULE3_BASELINE_TRAINING)
    training.update(
        {
            "episodes": 400,
            "learning_rate": 1e-4,
            "num_sgd_iter": 8,
        }
    )

    environment = dict(MODULE3_BASELINE_ENV)
    environment.update({"signal_mode": "truthful"})

    config = ExperimentConfig(
        module_id=CHAPTER4_MODULE_IDS["module3"],
        run_name=run_name,
        description="Formal Plain-MAPPO baseline with truthful non-deceptive signals.",
        seeds=[1],
        environment=environment,
        training=training,
        analysis={
            "evaluation_episodes": MODULE3_BASELINE_EVAL_EPISODES,
            "baseline": variant_name,
            "signal_mode": "truthful",
        },
        tags=["chapter4", "module3", "baseline", "plain-mappo", "truthful"],
    )
    artifacts = ExperimentArtifacts(config)
    artifacts.initialize()

    result = run_variant(
        family=family,
        variant_name=variant_name,
        env_overrides=environment,
        training_overrides=training,
        artifacts=artifacts,
        evaluation_episodes=MODULE3_BASELINE_EVAL_EPISODES,
    )

    summary_path = artifacts.csv_path("plain_mappo_truthful_summary.csv")
    write_csv(summary_path, [result.summary_row])

    pointer_path = artifacts.log_path("plain_mappo_truthful_latest.json")
    pointer_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "family": family,
                "variant_name": variant_name,
                "summary_csv": str(summary_path),
                "episode_log_csv": str(artifacts.csv_path(f"episode_log_{family}_{variant_name}.csv")),
                "evaluation_csv": str(artifacts.csv_path(f"evaluation_{family}_{variant_name}.csv")),
                "checkpoint_path": result.summary_row["checkpoint_path"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(result.summary_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
