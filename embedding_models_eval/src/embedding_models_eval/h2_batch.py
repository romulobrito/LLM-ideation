"""
Batch orchestration for H2 runs from a YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class H2Job:
    """One H2 execution job."""

    model_name: str
    config_path: str
    output_dir: str
    model_override: str


def load_h2_batch_config(path: str) -> Dict[str, Any]:
    """Load YAML config for batch H2 execution."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"H2 batch config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("H2 batch config root must be a mapping")
    return cfg


def _as_set(values: List[str]) -> set[str]:
    return {str(v).strip() for v in values if str(v).strip()}


def validate_h2_batch_config(cfg: Dict[str, Any]) -> None:
    """Validate required fields and strict protocol constraints."""
    dataset = cfg.get("dataset")
    protocol = cfg.get("protocol")
    models = cfg.get("models")
    if not isinstance(dataset, dict):
        raise ValueError("dataset section is required")
    if not isinstance(protocol, dict):
        raise ValueError("protocol section is required")
    if not isinstance(models, list) or not models:
        raise ValueError("models section must be a non-empty list")

    input_json = str(dataset.get("input_json") or "").strip()
    if not input_json:
        raise ValueError("dataset.input_json is required")

    k_values = protocol.get("k_values")
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("protocol.k_values must be a non-empty list")
    if any(int(k) <= 0 for k in k_values):
        raise ValueError("protocol.k_values must be positive integers")

    exclude_winner = bool(protocol.get("exclude_winner", False))
    if not exclude_winner:
        raise ValueError(
            "protocol.exclude_winner must be true to match H1 protocol exactly"
        )

    required_models = protocol.get("required_models")
    if not isinstance(required_models, list) or not required_models:
        raise ValueError("protocol.required_models must be a non-empty list")

    model_names: List[str] = []
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{idx}] must be a mapping")
        name = str(model.get("name") or "").strip()
        run_config = str(model.get("config") or "").strip()
        output_subdir = str(model.get("output_subdir") or "").strip()
        if not name:
            raise ValueError(f"models[{idx}].name is required")
        if not run_config:
            raise ValueError(f"models[{idx}].config is required")
        if not output_subdir:
            raise ValueError(f"models[{idx}].output_subdir is required")
        model_names.append(name)

    model_set = _as_set(model_names)
    required_set = _as_set(required_models)
    if model_set != required_set:
        missing = sorted(required_set - model_set)
        extra = sorted(model_set - required_set)
        raise ValueError(
            "Configured H2 models do not match required_models exactly. "
            f"missing={missing}, extra={extra}"
        )


def build_h2_jobs(cfg: Dict[str, Any]) -> List[H2Job]:
    """Build H2 jobs from config."""
    output_root = str(cfg["dataset"]["output_root"]).strip()
    jobs: List[H2Job] = []
    for model in cfg["models"]:
        model_name = str(model["name"]).strip()
        config_path = str(model["config"]).strip()
        output_subdir = str(model["output_subdir"]).strip()
        model_override = str(model.get("model_override") or "").strip()
        jobs.append(
            H2Job(
                model_name=model_name,
                config_path=config_path,
                output_dir=f"{output_root}/{output_subdir}",
                model_override=model_override,
            )
        )
    return jobs
