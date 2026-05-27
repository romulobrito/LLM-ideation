"""
Configuration and execution helpers for H2 random baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from embedding_models_eval.metrics.feasible_range_random_baseline import summarize_model


@dataclass(frozen=True)
class H2BaselineInput:
    """Input pair: model name and detailed json path."""

    model_name: str
    detailed_json: str


def load_h2_random_baseline_config(path: str) -> Dict[str, Any]:
    """Load YAML config for H2 random baseline."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"H2 random baseline config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("H2 random baseline config root must be a mapping")
    return cfg


def validate_h2_random_baseline_config(cfg: Dict[str, Any]) -> None:
    """Validate required fields for H2 random baseline config."""
    protocol = cfg.get("protocol")
    dataset = cfg.get("dataset")
    models = cfg.get("models")

    if not isinstance(protocol, dict):
        raise ValueError("protocol section is required")
    if not isinstance(dataset, dict):
        raise ValueError("dataset section is required")
    if not isinstance(models, list) or not models:
        raise ValueError("models section must be a non-empty list")

    k_values = protocol.get("k_values")
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("protocol.k_values must be a non-empty list")
    if any(int(k) <= 0 for k in k_values):
        raise ValueError("protocol.k_values must contain positive integers")

    num_permutations = int(protocol.get("num_permutations", 0))
    if num_permutations <= 0:
        raise ValueError("protocol.num_permutations must be > 0")

    percentiles = protocol.get("percentiles")
    if not isinstance(percentiles, list) or not percentiles:
        raise ValueError("protocol.percentiles must be a non-empty list")
    if any(int(p) < 0 or int(p) > 100 for p in percentiles):
        raise ValueError("protocol.percentiles values must be in [0, 100]")

    seed = int(protocol.get("seed", 0))
    if seed < 0:
        raise ValueError("protocol.seed must be >= 0")

    output_csv = str(dataset.get("output_csv") or "").strip()
    output_json = str(dataset.get("output_json") or "").strip()
    if not output_csv:
        raise ValueError("dataset.output_csv is required")
    if not output_json:
        raise ValueError("dataset.output_json is required")

    seen: set[str] = set()
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{idx}] must be a mapping")
        model_name = str(model.get("name") or "").strip()
        detailed_json = str(model.get("detailed_json") or "").strip()
        if not model_name:
            raise ValueError(f"models[{idx}].name is required")
        if not detailed_json:
            raise ValueError(f"models[{idx}].detailed_json is required")
        if model_name in seen:
            raise ValueError(f"Duplicate model name in config: {model_name}")
        seen.add(model_name)


def build_h2_baseline_inputs(cfg: Dict[str, Any]) -> List[H2BaselineInput]:
    """Build model input list from config."""
    out: List[H2BaselineInput] = []
    for model in cfg["models"]:
        out.append(
            H2BaselineInput(
                model_name=str(model["name"]).strip(),
                detailed_json=str(model["detailed_json"]).strip(),
            )
        )
    return out


def run_h2_random_baseline_from_config(cfg: Dict[str, Any]) -> List[Dict[str, object]]:
    """Run baseline for all models from config and return long rows."""
    protocol = cfg["protocol"]
    k_values: Sequence[int] = [int(v) for v in protocol["k_values"]]
    num_permutations = int(protocol["num_permutations"])
    seed = int(protocol["seed"])
    percentiles: Sequence[int] = [int(v) for v in protocol["percentiles"]]

    rows: List[Dict[str, object]] = []
    inputs = build_h2_baseline_inputs(cfg)
    for idx, item in enumerate(inputs):
        model_rows = summarize_model(
            model_name=item.model_name,
            detailed_json_path=item.detailed_json,
            k_values=k_values,
            num_permutations=num_permutations,
            seed=seed + idx,
            percentiles=percentiles,
        )
        rows.extend(model_rows)
    return rows
