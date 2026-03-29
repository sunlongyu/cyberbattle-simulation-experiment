from __future__ import annotations

from experiments.module3.pipeline import run_module3_ablation_refined


if __name__ == "__main__":
    outputs = run_module3_ablation_refined()
    print("module_root:", outputs["module_root"])
    print("results_summary:", outputs["results_summary"])
    print("ablation_table:", outputs["ablation_table"])
