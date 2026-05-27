"""
Run H2 random baseline from a YAML config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from embedding_models_eval.h2_random_baseline import (
    load_h2_random_baseline_config,
    run_h2_random_baseline_from_config,
    validate_h2_random_baseline_config,
)
from embedding_models_eval.metrics.feasible_range_random_baseline import write_csv


def main() -> None:
    """CLI entrypoint for H2 random baseline using YAML config."""
    parser = argparse.ArgumentParser(description="Run H2 random baseline from YAML config")
    parser.add_argument("--config", required=True, help="H2 random baseline YAML config")
    args = parser.parse_args()

    cfg = load_h2_random_baseline_config(args.config)
    validate_h2_random_baseline_config(cfg)
    rows: List[Dict[str, object]] = run_h2_random_baseline_from_config(cfg)

    output_csv = str(cfg["dataset"]["output_csv"])
    output_json = str(cfg["dataset"]["output_json"])

    write_csv(rows, output_csv)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    print(f"CSV gerado: {output_csv}")
    print(f"JSON gerado: {output_json}")
    print(f"Linhas: {len(rows)}")


if __name__ == "__main__":
    main()
