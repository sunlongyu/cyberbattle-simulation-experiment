# Chapter 3 Signaling-Game Simulation

This directory keeps an independent earlier/alternate Chapter 3 signaling-game implementation. The formal thesis-facing Chapter 3 result directory is `sg_deception_simulation2/`.

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

- Keep this directory separate from `sg_deception_simulation2/`.
- Do not copy result files between the two directories unless the target directory is explicitly being refreshed.
