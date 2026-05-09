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
- Keep only experiment scripts, mappings, scenarios, and generated validation results here.
- Do not keep thesis rewrite drafts, polished prose, or chapter-writing notes in this directory.
- Use `sg_deception_simulation2/` for formal Chapter 3 signaling-game outputs, not this directory.
