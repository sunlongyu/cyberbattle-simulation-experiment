# Experiment Index

## Chapter 3: Signaling Game

Formal thesis-facing directory: `sg_deception_simulation_main/`

- `cyber_simulation_core/`: model, strategy, configuration, and experiment logic.
- `run_experiments.py`: generates Chapter 3 JSON and CSV outputs.
- `plot_results.py`: regenerates Chapter 3 figures from saved outputs.
- `results/`: thesis-traceable generated outputs.
- `results/figures/`: final Chapter 3 figure files.

Formal experiments:

1. Four-strategy defender cumulative utility comparison.
2. Four-strategy terminal belief comparison.
3. Parameter sensitivity analysis, including discount factor sensitivity.

## Chapter 4: MARL

Active directory: `sg_marl_simulation/`

- `code/`: code snapshot and runnable scripts for the Chapter 4 workflow.
- `module1_convergence/`: convergence and seed-stability results.
- `module2_policy/`: strategy distribution probability results.
- `module3_effectiveness/`: baseline comparison and ablation results.

Formal experiments:

1. Convergence and stability.
2. Strategy distribution probability.
3. Baseline key-metric point estimates and ablation.

## Local-Only Or Ignored Content

- `.venv/`, `.mplconfig/`, `__pycache__/`, `.DS_Store`: machine-local artifacts.

## When Adding New Results

1. Put code changes in the active chapter directory.
2. Save final JSON/CSV summaries under that chapter's `results/`, `csv/`, or `tables/` directory.
3. Save final figures under `figures/`.
4. Keep smoke-test or exploratory outputs out of Git unless they are needed to explain a thesis result.
