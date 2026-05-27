"""
Run H2 for all models defined in a YAML config.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from embedding_models_eval.h2_batch import (
    build_h2_jobs,
    load_h2_batch_config,
    validate_h2_batch_config,
)


def _k_values_as_csv(values: list[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2 batch from YAML config")
    parser.add_argument("--config", required=True, help="H2 batch YAML config path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing",
    )
    args = parser.parse_args()

    cfg = load_h2_batch_config(args.config)
    validate_h2_batch_config(cfg)

    input_json = str(cfg["dataset"]["input_json"])
    k_values = _k_values_as_csv([int(v) for v in cfg["protocol"]["k_values"]])
    exclude_winner = bool(cfg["protocol"]["exclude_winner"])
    jobs = build_h2_jobs(cfg)

    root_dir = Path(__file__).resolve().parents[1]
    run_h2_script = root_dir / "scripts" / "run_h2_eval.py"
    py_exec = "python"

    for idx, job in enumerate(jobs, start=1):
        cmd = [
            py_exec,
            str(run_h2_script),
            "--config",
            job.config_path,
            "--input-json",
            input_json,
            "--output-dir",
            job.output_dir,
            "--model-name",
            job.model_name,
            "--k-values",
            k_values,
        ]
        if exclude_winner:
            cmd.append("--exclude-winner")
        if job.model_override:
            cmd.extend(["--model-override", job.model_override])

        printable = " ".join(shlex.quote(x) for x in cmd)
        print(f"[{idx}/{len(jobs)}] {job.model_name}")
        print(f"CMD: {printable}")
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True, cwd=str(root_dir))

    comparison = cfg.get("comparison", {})
    if comparison and bool(comparison.get("enabled", False)):
        consolidate_script = root_dir / "scripts" / "build_h1_h2_comparison.py"
        cmd = [
            py_exec,
            str(consolidate_script),
            "--h1-path",
            str(comparison["h1_path"]),
            "--h2-path",
            str(comparison["h2_path"]),
            "--out-dir",
            str(comparison["out_dir"]),
        ]
        printable = " ".join(shlex.quote(x) for x in cmd)
        print("[final] Consolidating H1+H2")
        print(f"CMD: {printable}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(root_dir))


if __name__ == "__main__":
    main()
