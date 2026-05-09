# Experiment Index

## Chapter 3: Signaling Game

Formal thesis-facing directory: `sg_deception_simulation2/`

- `cyber_simulation_core/`: model, strategy, configuration, and experiment logic.
- `run_experiments.py`: generates Chapter 3 JSON and CSV outputs.
- `plot_results.py`: regenerates Chapter 3 figures from saved outputs.
- `results/`: thesis-traceable generated outputs.
- `results/figures/`: final Chapter 3 figure files.
- `results/figures/drafts/`: matched draft figure variants moved out of the temporary workspace.

Independent earlier/alternate directory: `sg_deception_simulation/`

- Keep this separate from `sg_deception_simulation2/`.
- Use it only when comparing or preserving the earlier Chapter 3 experiment path.

Related validation directory: `cyborg_validation/`

- Keep CyberBattle/CybORG mapping scripts, scenario files, and compact validation results here.
- Do not keep thesis rewrite drafts or prose-polishing notes in this directory.

## Chapter 4: MARL

Active directory: `sg_marl_simulation/`

- `code/`: code snapshot and runnable scripts for the Chapter 4 workflow.
- `module1_convergence/`: convergence and seed-stability results.
- `module2_policy/`: policy and signaling-mechanism results.
- `module3_effectiveness/`: baseline comparison and ablation results.
- `preview_figures/`: preview exports used for checking figure formatting.
- `module*/figures/drafts/`: matched draft figure variants moved out of the temporary workspace.
- `chapter4_text_replacements.md`: local text-replacement notes for Chapter 4 writing.

## Local-Only Or Ignored Content

- `sg_marl_simulation/**/logs/`: runtime logs and diagnostics.
- `.venv/`, `.mplconfig/`, `__pycache__/`, `.DS_Store`: machine-local artifacts.
- `tmp_ch4_drafts/`: temporary figure staging area; move matched files into the relevant result directory before committing.
- `cyborg_validation/*.md`: local thesis-writing drafts unless explicitly promoted. `README.md` is the only Markdown file expected in `cyborg_validation/`.

## When Adding New Results

1. Put code changes in the active chapter directory.
2. Save final JSON/CSV summaries under that chapter's `results/`, `csv/`, or `tables/` directory.
3. Save final figures under `figures/`.
4. Keep smoke-test or exploratory outputs out of Git unless they are needed to explain a thesis result.
