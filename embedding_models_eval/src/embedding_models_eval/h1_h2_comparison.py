"""
Utilities to consolidate H1 and H2 macro outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

METRIC_WITH_N_PATTERN = re.compile(
    r"^(?P<metric>[A-Za-z0-9_]+)@(?P<n>[0-9]+)_(?P<stat>mean|std)$"
)


def _list_macro_csv_files(path: Path) -> list[Path]:
    """Return macro CSV files from a file path or recursively from a directory."""
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("comparacao_modelos_macro.csv"))
        if not files:
            raise ValueError(f"No comparacao_modelos_macro.csv found under: {path}")
        return files
    raise ValueError(f"Path does not exist: {path}")


def _validate_macro_df(df: pd.DataFrame, source: Path) -> None:
    """Validate minimum schema for a macro table."""
    if "Modelo" not in df.columns:
        raise ValueError(f"Missing required column 'Modelo' in: {source}")
    if len(df.columns) < 2:
        raise ValueError(f"Macro table has no metric columns: {source}")


def load_macro_tables(path: str) -> pd.DataFrame:
    """Load and concatenate macro CSV tables."""
    base_path = Path(path).expanduser().resolve()
    csv_files = _list_macro_csv_files(base_path)
    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        _validate_macro_df(df, csv_file)
        tmp = df.copy()
        tmp["source_file"] = str(csv_file)
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    out["Modelo"] = out["Modelo"].astype(str)
    return out


def to_long_format(df: pd.DataFrame, hypothesis: str) -> pd.DataFrame:
    """Convert a macro table into a normalized long format."""
    metric_cols = [col for col in df.columns if col not in {"Modelo", "source_file"}]
    melted = df.melt(
        id_vars=["Modelo", "source_file"],
        value_vars=metric_cols,
        var_name="metric_key",
        value_name="value",
    )
    melted = melted.rename(columns={"Modelo": "model"})
    melted["hypothesis"] = hypothesis

    metrics: list[str] = []
    n_values: list[object] = []
    stats: list[object] = []
    for key in melted["metric_key"].astype(str).tolist():
        match = METRIC_WITH_N_PATTERN.match(key)
        if match is None:
            metrics.append(key)
            n_values.append(pd.NA)
            stats.append(pd.NA)
            continue
        metrics.append(match.group("metric"))
        n_values.append(int(match.group("n")))
        stats.append(match.group("stat"))

    melted["metric"] = metrics
    melted["n"] = n_values
    melted["stat"] = stats
    return melted[
        [
            "hypothesis",
            "model",
            "metric_key",
            "metric",
            "n",
            "stat",
            "value",
            "source_file",
        ]
    ]


def _to_records(df: pd.DataFrame) -> list[dict[str, object]]:
    """Serialize DataFrame rows to JSON-safe records."""
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        safe_row: dict[str, object] = {}
        for key, value in row.items():
            if pd.isna(value):
                safe_row[key] = None
            else:
                safe_row[str(key)] = value
        records.append(safe_row)
    return records


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write JSON with ASCII-safe serialization."""
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=True, indent=2)


def consolidate_h1_h2(h1_path: str, h2_path: str, out_dir: str) -> dict[str, Path]:
    """
    Build consolidated H1+H2 outputs.

    Returns paths to generated files.
    """
    h1_df = load_macro_tables(h1_path)
    h2_df = load_macro_tables(h2_path)

    long_h1 = to_long_format(h1_df, "H1")
    long_h2 = to_long_format(h2_df, "H2")
    combined_long = pd.concat([long_h1, long_h2], ignore_index=True)

    grouped = (
        combined_long.groupby(["hypothesis", "model", "metric_key"], as_index=False)["value"]
        .mean()
        .sort_values(["hypothesis", "model", "metric_key"])
    )
    combined_wide = grouped.pivot(
        index=["hypothesis", "model"],
        columns="metric_key",
        values="value",
    ).reset_index()
    combined_wide.columns.name = None

    out_base = Path(out_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    long_csv = out_base / "comparacao_h1_h2_long.csv"
    long_json = out_base / "comparacao_h1_h2_long.json"
    wide_csv = out_base / "comparacao_h1_h2.csv"
    wide_json = out_base / "comparacao_h1_h2.json"

    combined_long.to_csv(long_csv, index=False)
    _write_json(long_json, _to_records(combined_long))
    combined_wide.to_csv(wide_csv, index=False)
    _write_json(wide_json, _to_records(combined_wide))

    return {
        "long_csv": long_csv,
        "long_json": long_json,
        "wide_csv": wide_csv,
        "wide_json": wide_json,
    }
