from __future__ import annotations

from experiments.module3.pipeline import run_module3


if __name__ == "__main__":
    outputs = run_module3()
    print("module_root:", outputs["module_root"])
    print("results_summary:", outputs["results_summary"])
    print("baseline_table:", outputs["baseline_table"])
    print("ablation_table:", outputs["ablation_table"])
