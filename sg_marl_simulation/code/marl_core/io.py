from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import csv
import json

from .config import ExperimentConfig
from .paths import ensure_module_layout


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            names.append(str(key))
    return names


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return path
    headers = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
    return path


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


@dataclass
class ExperimentArtifacts:
    config: ExperimentConfig

    def __post_init__(self) -> None:
        self.layout = ensure_module_layout(self.config.module_id, self.config.output.root_dir)

    @property
    def root(self) -> Path:
        return self.layout["root"]

    def config_path(self) -> Path:
        return self.layout["configs"] / f"{self.config.run_name}.json"

    def yaml_path(self) -> Path:
        return self.layout["configs"] / f"{self.config.run_name}.yaml"

    def readme_path(self) -> Path:
        return self.root / "README.md"

    def csv_path(self, name: str) -> Path:
        return self.layout["csv"] / name

    def table_path(self, name: str) -> Path:
        return self.layout["tables"] / name

    def figure_path(self, name: str) -> Path:
        return self.layout["figures"] / name

    def log_path(self, name: str) -> Path:
        return self.layout["logs"] / name

    def initialize(self) -> None:
        self.config.write(self.config_path())
        self.config.maybe_write_yaml(self.yaml_path())
        if self.config.output.write_readme:
            self.write_readme()

    def write_readme(self) -> Path:
        lines = [
            f"# {self.config.module_id}: {self.config.run_name}",
            "",
            "## Description",
            self.config.description or "No description provided.",
            "",
            "## Seeds",
            ", ".join(str(seed) for seed in self.config.seeds) if self.config.seeds else "Not specified",
            "",
            "## Output Layout",
            "- `figures/`: PNG and PDF figures",
            "- `tables/`: paper-ready tables in CSV",
            "- `csv/`: raw run logs and aggregated metrics",
            "- `configs/`: json/yaml experiment configs",
            "- `logs/`: text logs or auxiliary diagnostics",
            "",
            "## Reproduction",
            "- Record dependency versions before running module-specific experiments.",
            "- Keep random seeds fixed for all reported runs.",
            "- Use the shared naming helpers for all new figures and tables.",
        ]
        path = self.readme_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class EpisodeLogger:
    """Append-only CSV logger for episode-level metrics."""

    def __init__(self, path: Path, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = [str(name) for name in fieldnames]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Mapping[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})


class SummaryWriter:
    """Helper for accumulating and exporting summary rows."""

    def __init__(self) -> None:
        self.rows: List[Mapping[str, object]] = []

    def add(self, row: Mapping[str, object]) -> None:
        self.rows.append(dict(row))

    def write(self, path: Path) -> Path:
        return write_csv(path, self.rows)
