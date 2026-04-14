"""
Conversao do JSON legado (concurso / prompts / historias) para o schema task_rank.

Politica padrao (documentada):
- task_description: titulo do prompt apenas (context_prompt_title).
- story_id: story_url; se vazio, id sintetico contest::prompt_url::idx.
- story_text: coluna configuravel (padrao extracted_idea_250); fallback story_content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from embedding_models_eval.data.json_loader import iter_rows


def _story_text_from_row(row: Dict[str, Any], text_column: str) -> str:
    if text_column in row and str(row.get(text_column) or "").strip():
        return str(row[text_column]).strip()
    if str(row.get("story_content") or "").strip():
        return str(row["story_content"]).strip()
    return ""


def extract_task_inputs_from_legacy_payload(
    payload: Any,
    *,
    text_column: str = "extracted_idea_250",
    max_prompts: Optional[int] = None,
    max_stories_per_prompt: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Agrupa linhas de iter_rows por (contest_number, context_prompt_url).

    Returns:
        Lista de objetos {\"task_description\", \"stories\"} compativeis com parse_task_rank_input.
    """
    rows = list(iter_rows(payload))
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("contest_number", ""), row.get("context_prompt_url", ""))
        groups.setdefault(key, []).append(row)

    results: List[Dict[str, Any]] = []
    for key, group_rows in groups.items():
        if not group_rows:
            continue
        title = str(group_rows[0].get("context_prompt_title") or "").strip()
        if not title:
            title = f"prompt:{key[1]}"
        stories: List[Dict[str, str]] = []
        for idx, row in enumerate(group_rows):
            sid = str(row.get("story_url") or "").strip()
            if not sid:
                sid = f"{key[0]}::{key[1]}::idx{idx}"
            stext = _story_text_from_row(row, text_column)
            if not stext:
                continue
            stories.append({"story_id": sid, "story_text": stext})
        if len(stories) < 1:
            continue
        if max_stories_per_prompt is not None:
            stories = stories[: max_stories_per_prompt]
        results.append({"task_description": title, "stories": stories})

    results.sort(key=lambda x: (x["task_description"], len(x["stories"])))
    if max_prompts is not None:
        results = results[: max_prompts]
    return results
