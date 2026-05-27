"""
Run H2 evaluation and export macro outputs compatible with H1 consolidation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from embedding_models_eval.data.json_loader import iter_rows
from embedding_models_eval.h2_metrics import (
    aggregate_prompt_metrics,
    build_h2_detailed_rows,
    build_macro_row,
    build_story_votes_map,
    compute_prompt_h2_metrics,
    group_rows_by_prompt,
)
from embedding_models_eval.task_dissimilarity.config import (
    load_task_embedding_yaml,
    provider_from_task_config,
)
from embedding_models_eval.task_dissimilarity.ranking import (
    rank_stories_by_task_dissimilarity,
)


def _parse_k_values(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("k-values must be a non-empty list of positive integers")
    return vals


def _prompt_rows_to_stories(prompt_rows: List[dict], text_column: str) -> List[dict]:
    stories: List[dict] = []
    for idx, row in enumerate(prompt_rows):
        story_id = str(row.get("story_url") or "").strip()
        if not story_id:
            contest_number = str(row.get("contest_number") or "")
            prompt_url = str(row.get("context_prompt_url") or "")
            story_id = f"{contest_number}::{prompt_url}::idx{idx}"

        story_text = str(row.get(text_column) or "").strip()
        if not story_text:
            story_text = str(row.get("story_content") or "").strip()
        if not story_text:
            continue
        stories.append({"story_id": story_id, "story_text": story_text})
    return stories


def _write_macro_csv(path: Path, row: Dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2 metrics and export macro tables")
    parser.add_argument("--config", required=True, help="Task dissimilarity YAML config")
    parser.add_argument("--input-json", required=True, help="Legacy dataset JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--model-name",
        default="h2_model",
        help="Model display name for comparacao_modelos_macro",
    )
    parser.add_argument(
        "--model-override",
        default="",
        help="Override embedding.model from YAML config",
    )
    parser.add_argument(
        "--k-values",
        default="1,2,3,4,5,6,7",
        help="Comma-separated k values",
    )
    parser.add_argument(
        "--text-column",
        default="extracted_idea_250",
        help="Primary text column in legacy rows",
    )
    parser.add_argument(
        "--exclude-winner",
        action="store_true",
        default=False,
        help="Exclude winner proxy (highest likes) for comparability with H1",
    )
    args = parser.parse_args()

    k_values = _parse_k_values(args.k_values)

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = list(iter_rows(payload))
    grouped = group_rows_by_prompt(rows)

    cfg = load_task_embedding_yaml(args.config)
    if args.model_override:
        cfg["embedding"]["model"] = str(args.model_override).strip()
    provider, backend_label, model_label = provider_from_task_config(cfg)

    per_prompt_rows: List[Dict[str, float | str]] = []
    detailed_rows: List[Dict[str, float | str | None]] = []
    prompt_metrics: List[Dict[str, float]] = []

    for prompt_id, prompt_rows in grouped.items():
        if not prompt_rows:
            continue

        task_description = str(prompt_rows[0].get("context_prompt_title") or "").strip()
        if not task_description:
            task_description = str(prompt_rows[0].get("context_prompt_url") or "").strip()
        stories = _prompt_rows_to_stories(prompt_rows, args.text_column)
        if len(stories) < 2:
            continue

        ranking_output = rank_stories_by_task_dissimilarity(
            task_description=task_description,
            stories=stories,
            provider=provider,
            backend_label=backend_label,
            model_label=model_label,
        )
        predicted_story_ids = [str(r["story_id"]) for r in ranking_output["ranking"]]
        story_votes = build_story_votes_map(prompt_rows)
        metrics = compute_prompt_h2_metrics(
            predicted_story_ids=predicted_story_ids,
            story_votes=story_votes,
            k_values=k_values,
            exclude_winner=bool(args.exclude_winner),
        )
        prompt_metrics.append(metrics)
        row: Dict[str, float | str] = {"prompt_id": prompt_id}
        row.update(metrics)
        per_prompt_rows.append(row)
        detailed_rows.extend(
            build_h2_detailed_rows(
                prompt_id=str(prompt_id),
                prompt_rows=prompt_rows,
                predicted_story_ids=predicted_story_ids,
            )
        )

    if not prompt_metrics:
        raise ValueError("No prompt metrics were computed for H2")

    macro = aggregate_prompt_metrics(prompt_metrics)
    macro_row = build_macro_row(args.model_name, macro)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_csv = out_dir / "comparacao_modelos_macro.csv"
    macro_json = out_dir / "comparacao_modelos_macro.json"
    per_prompt_json = out_dir / "per_prompt_h2_metrics.json"
    detailed_json = out_dir / "resultados_h2_detalhado.json"

    _write_macro_csv(macro_csv, macro_row)
    with macro_json.open("w", encoding="utf-8") as f:
        json.dump([macro_row], f, ensure_ascii=True, indent=2)
    with per_prompt_json.open("w", encoding="utf-8") as f:
        json.dump(per_prompt_rows, f, ensure_ascii=True, indent=2)
    with detailed_json.open("w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, ensure_ascii=True, indent=2)

    print(f"Macro CSV: {macro_csv}")
    print(f"Macro JSON: {macro_json}")
    print(f"Per-prompt JSON: {per_prompt_json}")
    print(f"Detailed JSON: {detailed_json}")
    print(f"Prompts evaluated: {len(prompt_metrics)}")


if __name__ == "__main__":
    main()
