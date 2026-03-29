from __future__ import annotations


def _normalize_suffix(suffix: str) -> str:
    suffix = suffix.strip().replace(" ", "_")
    return suffix if suffix else "artifact"


def figure_name(section: str, suffix: str, ext: str = "png") -> str:
    return f"fig_{section}_{_normalize_suffix(suffix)}.{ext}"


def table_name(section: str, suffix: str, ext: str = "csv") -> str:
    return f"tab_{section}_{_normalize_suffix(suffix)}.{ext}"
