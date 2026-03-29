from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class OutputConfig:
    """Common output settings for all Chapter 4 experiment modules."""

    root_dir: str = "results/ch4"
    save_yaml: bool = False
    write_readme: bool = True
    image_dpi: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Serializable experiment configuration shared by all modules."""

    module_id: str
    run_name: str
    description: str = ""
    seeds: List[int] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output"] = self.output.to_dict()
        return data

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    def maybe_write_yaml(self, path: Path) -> Optional[Path]:
        if not self.output.save_yaml:
            return None
        try:
            import yaml  # type: ignore
        except ImportError:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, allow_unicode=True, sort_keys=False)
        return path
