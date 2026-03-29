from __future__ import annotations

from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def module_root(module_id: str, root_dir: str = "results/ch4") -> Path:
    return PROJECT_ROOT / root_dir / module_id


def ensure_module_layout(module_id: str, root_dir: str = "results/ch4") -> Dict[str, Path]:
    root = module_root(module_id, root_dir=root_dir)
    layout = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "csv": root / "csv",
        "configs": root / "configs",
        "logs": root / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
