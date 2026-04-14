"""
Extrai JSONs task_description + stories a partir do JSON legado (concurso/prompts).

Console script: legacy-to-task-input (apos pip install -e .)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embedding_models_eval.task_dissimilarity.legacy_input import (
    extract_task_inputs_from_legacy_payload,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Legacy JSON -> task_rank JSON(s)")
    p.add_argument("--input", required=True, help="saida_final.json ou equivalente")
    p.add_argument("--output-dir", required=True, help="Pasta para task_000.json ...")
    p.add_argument("--max-prompts", type=int, default=1, help="Maximo de prompts exportados")
    p.add_argument(
        "--max-stories",
        type=int,
        default=20,
        help="Maximo de historias por prompt",
    )
    p.add_argument(
        "--text-column",
        default="extracted_idea_250",
        help="Coluna de texto (iter_rows)",
    )
    args = p.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    items = extract_task_inputs_from_legacy_payload(
        payload,
        text_column=args.text_column,
        max_prompts=args.max_prompts,
        max_stories_per_prompt=args.max_stories,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, obj in enumerate(items):
        path = out_dir / f"task_{i:03d}.json"
        path.write_text(
            json.dumps(obj, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    print(f"Wrote {len(items)} file(s) under {out_dir}")


if __name__ == "__main__":
    main()
