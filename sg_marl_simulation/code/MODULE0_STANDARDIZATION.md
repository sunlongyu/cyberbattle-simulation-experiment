# Module 0 Standardization

This document defines the shared Chapter 4 experiment protocol before module-specific rewrites.

## Scope

Module 0 does not change the thesis claims by itself. It standardizes:

- output directories
- config snapshots
- per-episode CSV logging
- summary CSV aggregation
- figure and table naming
- convergence and volatility metrics

## Shared Output Root

All Chapter 4 outputs live under:

`/Users/SL/Documents/expproject/MARL/results/ch4`

Module folders:

- `module1_convergence`
- `module2_policy`
- `module3_effectiveness`

Each module contains:

- `figures/`
- `tables/`
- `csv/`
- `configs/`
- `logs/`
- `README.md`

## Shared Naming Convention

- figures: `fig_4_x_suffix.png` and `fig_4_x_suffix.pdf`
- tables: `tab_4_x_suffix.csv`
- raw episode logs: `episode_log_seed{seed}.csv`
- aggregated summary: `results_summary.csv`

## Shared Metrics

The common convergence rule is:

- target reward = `0.95 * mean(last 30 episode rewards)`
- convergence step = first episode after which reward stays above target for 20 consecutive episodes

The common volatility rule is:

- sample standard deviation of the selected reward series

## Shared Python Utilities

New shared infrastructure lives in:

- `/Users/SL/Documents/expproject/MARL/marl_core/config.py`
- `/Users/SL/Documents/expproject/MARL/marl_core/defaults.py`
- `/Users/SL/Documents/expproject/MARL/marl_core/io.py`
- `/Users/SL/Documents/expproject/MARL/marl_core/metrics.py`
- `/Users/SL/Documents/expproject/MARL/marl_core/naming.py`
- `/Users/SL/Documents/expproject/MARL/marl_core/paths.py`

The thesis-aligned baseline environment and training defaults are centralized in
`marl_core/defaults.py` so later modules no longer maintain separate copies of
`N`, `T`, `beta`, payoff terms, batch size, or LSTM width.

## Bootstrap Command

Run once to create the standardized result layout:

```bash
python3 scripts/init_ch4_layout.py
```

## Integration Rule

When module-specific scripts are refactored, they should:

1. create an `ExperimentConfig`
2. initialize `ExperimentArtifacts`
3. log per-episode rows through `EpisodeLogger`
4. export aggregate rows through `SummaryWriter`
5. save the exact config snapshot used for the run
