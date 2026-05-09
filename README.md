# cyberbattle-simulation-experiment

This repository organizes the thesis experiments for cyber deception defense and adversarial decision making.

## Active Experiment Directories

- `sg_deception_simulation2/`: formal Chapter 3 multi-stage signaling-game experiment results and code. Use this directory for thesis-facing Chapter 3 outputs.
- `sg_deception_simulation/`: independent earlier/alternate Chapter 3 signaling-game experiment path. Keep it separate from `sg_deception_simulation2/`.
- `cyborg_validation/`: Chapter 3 validation and scenario-mapping experiments against the CyberBattle/CybORG-style environment. Keep scripts, mappings, scenarios, and compact result JSON files here; do not keep thesis rewrite drafts here.
- `sg_marl_simulation/`: Chapter 4 signaling-game-guided MARL experiments. This directory contains the chapter-organized code snapshot and final module outputs.

The formal Chapter 3 result set in `sg_deception_simulation2/` is limited to three experiments: defender cumulative utility comparison, terminal belief comparison, and parameter sensitivity analysis including discount factor sensitivity.

The formal Chapter 4 result set in `sg_marl_simulation/` is limited to three experiments: convergence/stability, strategy distribution probability, and baseline/ablation effectiveness comparison.

## Version-Control Rules

- Commit source code, experiment configs, thesis-ready JSON/CSV summaries, and final figures/tables.
- Do not commit virtual environments, Python caches, macOS metadata, Matplotlib caches, or large runtime logs.
- Keep formal Chapter 3 signaling-game work in `sg_deception_simulation2/`.
- Keep `sg_deception_simulation/`, `sg_deception_simulation2/`, and `sg_marl_simulation/` as separate experiment families.
- Keep new Chapter 4 MARL work in `sg_marl_simulation/`.

See `EXPERIMENT_INDEX.md` for a directory-level map.
