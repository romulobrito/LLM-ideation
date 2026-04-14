"""
CLI: task-dissimilarity-rank --config yaml --input json
"""

from __future__ import annotations

import argparse
import json
import sys

from embedding_models_eval.task_dissimilarity.config import (
    load_task_embedding_yaml,
    provider_from_task_config,
)
from embedding_models_eval.task_dissimilarity.io import (
    parse_task_rank_input,
    serialize_task_rank_output,
)
from embedding_models_eval.task_dissimilarity.ranking import (
    rank_stories_by_task_dissimilarity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ranking por dissimilaridade embedding (tarefa vs historias)"
    )
    parser.add_argument("--config", required=True, help="YAML com secao embedding")
    parser.add_argument("--input", required=True, help="JSON task_description + stories")
    parser.add_argument(
        "--output",
        default="",
        help="Arquivo JSON de saida (padrao: stdout)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_in = json.load(f)
    task, stories = parse_task_rank_input(raw_in)

    cfg = load_task_embedding_yaml(args.config)
    provider, backend_label, model_label = provider_from_task_config(cfg)

    out = rank_stories_by_task_dissimilarity(
        task,
        stories,
        provider,
        backend_label=backend_label,
        model_label=model_label,
    )
    text = serialize_task_rank_output(out)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
