# Chapter 4 Results Layout

All Chapter 4 experiment outputs are stored under `results/ch4/`.

## Modules

- `module1_convergence/`
- `module2_policy/`
- `module3_effectiveness/`

Each module uses the same subdirectories:

- `figures/`: PNG and PDF figures used in the thesis
- `tables/`: thesis-ready summary tables in CSV
- `csv/`: raw per-seed and per-episode metrics
- `configs/`: experiment config snapshots in JSON and optional YAML
- `logs/`: runtime logs and diagnostics

## Naming Rules

- Figures: `fig_4_x_suffix.png` and `fig_4_x_suffix.pdf`
- Tables: `tab_4_x_suffix.csv`
- Raw logs: `episode_log_seed{seed}.csv`
- Summary file: `results_summary.csv`

## Shared Metrics

The shared metrics helpers define a common convergence rule:

- final target = last `W=30` episode mean times `0.95`
- convergence step = first episode index after which rewards remain above target for `K=20` consecutive episodes

Reward volatility is reported as the sample standard deviation of the selected reward series.
