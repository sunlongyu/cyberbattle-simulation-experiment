# cyberbattle-simulation-experiment

This repository organizes the thesis experiments for cyber deception defense and adversarial decision making.

## Active Experiment Directories

- `sg_deception_simulation/`: Chapter 3 multi-stage signaling-game simulation. This is the active directory for formal Chapter 3 code, generated JSON/CSV outputs, figures, and analysis notes.
- `cyborg_validation/`: Chapter 3 validation and scenario-mapping experiments against the CyberBattle/CybORG-style environment. Keep scripts, mappings, scenarios, and compact result JSON files here.
- `sg_marl_simulation/`: Chapter 4 signaling-game-guided MARL experiments. This directory contains the chapter-organized code snapshot and final module outputs.

## Archive

- `archive/`: Historical snapshots or superseded experiment variants. Do not use archived code as the default source for new runs unless a README inside the archive explicitly says so.

## Version-Control Rules

- Commit source code, experiment configs, thesis-ready JSON/CSV summaries, and final figures/tables.
- Do not commit virtual environments, Python caches, macOS metadata, Matplotlib caches, or large runtime logs.
- Keep new Chapter 3 signaling-game work in `sg_deception_simulation/`.
- Keep new Chapter 4 MARL work in `sg_marl_simulation/`.
- Put exploratory or superseded variants under `archive/` only after recording why they are no longer active.

See `EXPERIMENT_INDEX.md` for a directory-level map.
