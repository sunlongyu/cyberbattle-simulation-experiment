# CyberBattle/CybORG Validation

This directory holds Chapter 3 validation and scenario-mapping experiments that connect the signaling-game assumptions to CyberBattle/CybORG-style environments.

## Contents

- `*_validation.py`: validation scripts.
- `run_*_experiments.py`: batch experiment entry points.
- `*_mapping.json`: scenario or host-type mappings.
- `scenarios/`: reusable scenario definitions.
- `results/`: compact JSON outputs for traceability.

## Maintenance Notes

- Keep validation results compact enough for Git review.
- Local Chinese writing drafts are ignored by `.gitignore`; promote a draft deliberately before committing it.
- Use `sg_deception_simulation/` for the formal Chapter 3 signaling-game implementation, not this directory.
