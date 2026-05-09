# Chapter 3 Signaling-Game Simulation

This is the active Chapter 3 implementation for the multi-stage signaling-game experiments.

## Contents

- `cyber_simulation_core/`: model, strategy, configuration, and experiment helpers.
- `run_experiments.py`: writes JSON and CSV outputs into `results/`.
- `plot_results.py`: regenerates figures under `results/figures/`.
- `EXPERIMENT_PLAN.md`: current theory-to-code mapping and output plan.
- `results/`: generated thesis-traceable data and figures.

## Run

From the repository root:

```bash
python sg_deception_simulation/run_experiments.py
python sg_deception_simulation/plot_results.py
```

## Maintenance Notes

- Keep this directory as the only active Chapter 3 signaling-game path.
- Do not add new `sg_deception_simulation*` sibling directories.
- If a new implementation branch is experimental, put it under `archive/` with a README explaining why.
