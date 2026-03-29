#!/usr/bin/env python3
"""Run Chapter 4 module 1 experiments."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.module1.pipeline import run_module1


def main() -> None:
    result = run_module1()
    print(result["module_root"])
    print(result["summary_csv"])
    print(result["table_csv"])


if __name__ == "__main__":
    main()
